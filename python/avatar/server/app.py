"""FastAPI 服务：驱动流广播 + 音频流 + 渲染端管理 + 会话编排。

端点：
- GET  /health                健康检查
- GET  /renderers             渲染后端列表/状态
- POST /renderers/switch      手动切换后端 {name, reason}
- POST /renderers/telemetry   渲染端性能遥测（自动降级输入）
- GET  /sse/drive             JSON 驱动流 + 音频流（SSE，调试用）
- WS   /ws/drive              驱动流 + 音频流（binary=protobuf 封套 / text=JSON）
- POST /chat                  发起一轮对话（返回 trace_id）
- POST /barge_in              打断当前生成
- POST /tts/preview           文本 → 一段 TTS 音频（wav，快速验证）
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import time
import wave
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from avatar.env import load_dotenv
from avatar.pipeline.core import BargeIn
from avatar.pipeline.driving import DrivingStage
from avatar.pipeline.emotion import EmotionSmoother
from avatar.pipeline.llm import LLMAdapter, MockLLM
from avatar.pipeline.tts import AudioChunk as TTSAudioChunk, MockTTS, TTSAdapter, TTSTimeline
from avatar.protocol.jsoncodec import encode_audio_binary, encode_binary, encode_json
from avatar.renderer.backends import build_mock_registry
from avatar.renderer.base import RendererManager
from avatar.trace import Tracer, new_trace_id, setup_logging

load_dotenv()


def make_llm(kind: str | None = None) -> LLMAdapter:
    """LLM 工厂：mock（默认）| zhipu（真实 API）。环境变量 AVATAR_LLM 可设默认。"""
    kind = kind or os.environ.get("AVATAR_LLM", "mock")
    if kind == "zhipu":
        from avatar.pipeline.llm_zhipu import ZhipuLLM

        return ZhipuLLM()
    if kind == "mock":
        return MockLLM()
    raise ValueError(f"未知 LLM 后端: {kind}")


def make_tts(kind: str | None = None) -> TTSAdapter:
    """TTS 工厂：mock（默认，静音时间线）| sherpa（Matcha+Vocos 真实音频）。"""
    kind = kind or os.environ.get("AVATAR_TTS", "mock")
    if kind == "sherpa":
        from avatar.pipeline.tts_sherpa import SherpaTTS

        return SherpaTTS()
    if kind == "mock":
        return MockTTS()
    raise ValueError(f"未知 TTS 后端: {kind}")


DEFAULT_RENDERER = "flashhead"


class ChatRequest(BaseModel):
    text: str = "你好"
    renderer: str | None = None
    llm: str | None = None
    tts: str | None = None            # 本轮 TTS 后端（mock / sherpa）
    user_emotion: str | None = None  # 外部传入的用户情绪标签（如 SenseVoice 识别结果）


class SwitchRequest(BaseModel):
    name: str
    reason: str = "manual"


class PreviewRequest(BaseModel):
    text: str = "你好呀，很高兴见到你！"
    tts: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    app.state.service = ConversationService()
    await app.state.service.start()
    yield
    await app.state.service.stop()


class ConversationService:
    """全链路编排：LLM → 情绪平滑 → TTS 时间线 → 驱动帧 → 渲染端广播。

    流水线并行：各阶段以 asyncio.Queue 解耦（边生成、边分析、边合成、边渲染）。
    """

    def __init__(self) -> None:
        self.llm: LLMAdapter = make_llm()
        self.llm_kind = os.environ.get("AVATAR_LLM", "mock")
        self.tts: TTSAdapter = make_tts()
        self.tts_kind = os.environ.get("AVATAR_TTS", "mock")
        self.smoother = EmotionSmoother(window_size=7)
        self.barge_in = BargeIn()
        self.renderer = RendererManager(build_mock_registry())
        self.renderer.select(DEFAULT_RENDERER)
        self._subscribers: set[asyncio.Queue] = set()
        self._task: asyncio.Task | None = None
        self._turn_running = False
        self.last_error: str | None = None
        self.last_trace: dict | None = None  # 最近一轮的 trace 摘要
        self._history: list[dict] = []
        self._user_tracker = None  # 惰性初始化（避免无 torch 环境起服务失败）
        self._recognizer = None

    @property
    def recognizer(self):
        """SenseVoice 情感识别器（惰性加载，失败时返回 None 并记错误）。"""
        if self._recognizer is None:
            try:
                from avatar.pipeline.emotion_recognizer import SenseVoiceRecognizer

                self._recognizer = SenseVoiceRecognizer()
            except Exception as e:
                self.last_error = f"emotion recognizer init: {e}"
                return None
        return self._recognizer

    @property
    def user_tracker(self):
        from avatar.pipeline.emotion_recognizer import InputEmotionTracker

        if self._user_tracker is None:
            self._user_tracker = InputEmotionTracker()
        return self._user_tracker

    async def start(self) -> None:
        # 预热：后台加载 TTS 模型（首轮对话免 ~1.3s 冷启动）
        if self.tts_kind == "sherpa" and hasattr(self.tts, "ensure_loaded"):
            asyncio.create_task(self._warmup_tts())
        # 预热：SenseVoice 识别器（opt-in；torch+模型冷加载约 25s，
        # 首次语音交互前预热可消除。无 torch 环境勿开启）
        if os.environ.get("AVATAR_WARMUP_RECOGNIZER") == "1":
            asyncio.create_task(self._warmup_recognizer())

    async def _warmup_recognizer(self) -> None:
        import logging

        try:
            if self.recognizer is not None:
                # 做一次真实 dummy 推理：torch 首次推理有 kernel 编译/内存
                # 分配等一次性成本（低内存机可达 10s+），启动时付掉
                import math
                import struct

                sr = 16000
                dummy = struct.pack(
                    "<%dh" % sr,
                    *(int(3000 * math.sin(2 * math.pi * 220 * i / sr)) for i in range(sr)),
                )
                await asyncio.to_thread(self.recognizer.recognize, dummy, sr)
                logging.getLogger("avatar.trace").info("event=recognizer_preloaded(with_infer)")
        except Exception as e:
            self.last_error = f"recognizer warmup: {type(e).__name__}: {e}"

    async def _warmup_tts(self) -> None:
        import logging

        try:
            await self.tts.ensure_loaded()
            logging.getLogger("avatar.trace").info(
                "event=tts_preloaded engine=%s sr=%s", self.tts.name, self.tts.sample_rate)
        except Exception as e:
            self.last_error = f"tts warmup: {type(e).__name__}: {e}"

    async def stop(self) -> None:
        self.barge_in.trigger()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=2048)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    async def _broadcast(self, payload: str, binary: bytes | None = None) -> None:
        for q in list(self._subscribers):
            try:
                q.put_nowait((payload, binary))
            except asyncio.QueueFull:
                pass  # 慢消费者丢帧（非关键帧可丢弃）

    async def _emit(self, frame: dict) -> None:
        self.renderer.push(frame)
        if os.environ.get("AVATAR_DEBUG"):
            print(f"[emit] seq={frame['frame_seq']} ts={frame['timestamp_ms']} subs={len(self._subscribers)}", flush=True)
        await self._broadcast(encode_json(frame), encode_binary(frame))

    def _tts_sample_rate(self) -> int:
        try:
            return int(self.tts.sample_rate) or 16000
        except Exception:
            return 16000

    async def _emit_audio(self, a: TTSAudioChunk, tracer: Tracer) -> None:
        """音频块下发：JSON 通道 base64，二进制通道 ServerMessage{audio}。"""
        d = {
            "type": "audio",
            "timestamp_ms": a.timestamp_ms,
            "duration_ms": a.duration_ms,
            "pcm_b64": base64.b64encode(a.pcm_s16le).decode("ascii"),
            "sample_rate": self._tts_sample_rate(),
            "emotion": a.emotion.to_dict(),
            "sentence_done": a.sentence_done,
            "trace_id": tracer.trace_id,
        }
        await self._broadcast(encode_json(d), encode_audio_binary(d))

    async def start_turn(self, user_text: str, renderer: str | None = None, llm: str | None = None,
                         user_emotion: str | None = None, tts: str | None = None,
                         trace_id: str | None = None, tracer=None) -> None:
        """发起一轮对话：并行驱动 LLM 句流 → 平滑 → TTS 音频时间线 → 驱动帧。

        user_emotion: 外部识别到的用户情绪标签（SenseVoice / 客户端传入），
        用于共情上下文（LLM 感知用户情绪调整回应风格）。
        trace_id: 外部指定（如 /chat 预生成返回给调用方），缺省自增。
        tracer: 复用外部 Tracer（语音流场景：ASR 打点与对话轮同一条 trace，
        summary 可同时呈现 asr_final → llm_first_chunk 全链路耗时）。
        """
        if self._turn_running:
            self.barge_in.trigger()  # 新轮次打断旧轮次
            await asyncio.sleep(0.05)
        self.barge_in.clear()
        self._turn_running = True
        if llm:
            self.llm = make_llm(llm)
            self.llm_kind = llm
        if tts:
            self.tts = make_tts(tts)
            self.tts_kind = tts
        if renderer:
            self.renderer.switch(renderer, reason="per-turn")
        self._pending_user_emotion = user_emotion
        tracer = tracer or Tracer(trace_id or new_trace_id(), label=user_text[:30])
        try:
            await self._run_turn(user_text, tracer)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # 单轮失败不拖垮服务
            self.last_error = f"{type(e).__name__}: {e}"
            await self._broadcast(json.dumps(
                {"type": "error", "detail": self.last_error, "trace_id": tracer.trace_id}))
        finally:
            self._turn_running = False

    async def _run_turn(self, user_text: str, tracer: Tracer) -> None:
        timeline = TTSTimeline()
        driving = DrivingStage(timeline=timeline)
        ctx = {"history": self._history, "tracer": tracer, "trace_id": tracer.trace_id}
        # 共情上下文：SenseVoice 识别或外部传入的用户情绪
        user_emotion = getattr(self, "_pending_user_emotion", None)
        if user_emotion:
            from avatar.pipeline.emotion import emotion_anchor

            anchor = emotion_anchor(user_emotion)
            ctx["user_emotion"] = (
                f"（感知：用户当前情绪是 {user_emotion}，强度 {anchor.intensity:.1f}，"
                "请据此调整回应语气与共情表达）"
            )
        tracer.mark("turn_start", text=user_text[:24], llm=self.llm_kind, tts=self.tts_kind)
        await self._broadcast(json.dumps(
            {"type": "turn_start", "text": user_text, "trace_id": tracer.trace_id}))
        chunks: list = []
        total_audio_ms = 0
        marked_tts = marked_frame = False
        async for chunk in self.llm.stream_turn(user_text, ctx):
            if self.barge_in._event.is_set():
                break
            if tracer.elapsed_of("llm_first_chunk") is None:
                tracer.mark("llm_first_chunk", chars=len(chunk.text))
            chunks.append(chunk)
            emo = self.smoother.push(chunk.emotion)
            chunk.emotion = emo
            start_ms = timeline.clock_ms
            # TTS 先行：真实音频时长决定时间线（音画同步的时间源）
            audios: list[TTSAudioChunk] = []
            try:
                async for a in self.tts.synthesize(chunk, ctx, start_ms):
                    if self.barge_in._event.is_set():
                        break
                    audios.append(a)
                    if not marked_tts:
                        tracer.mark("tts_first_audio", dur_ms=a.duration_ms,
                                    sr=self._tts_sample_rate())
                        marked_tts = True
            except Exception as e:
                self.last_error = f"tts: {type(e).__name__}: {e}"
                await self._broadcast(json.dumps(
                    {"type": "error", "detail": self.last_error, "trace_id": tracer.trace_id}))
                audios = []  # TTS 失败：退化为纯驱动帧，语音不中断整轮
            if audios:
                total_audio_ms += sum(a.duration_ms for a in audios)
                end_ms = audios[-1].timestamp_ms + audios[-1].duration_ms
            else:
                _, end_ms = timeline.reserve(len(chunk.text))  # 无音频：按字数估计兜底
            timeline.clock_ms = max(timeline.clock_ms, end_ms)
            frames = driving.frames_for(chunk, start_ms, end_ms)
            for f in frames:
                f["trace_id"] = tracer.trace_id
            # 音频块与驱动帧按时间戳交错下发（时间相同则音频先行）
            def _ts(x):
                return x.timestamp_ms if hasattr(x, "timestamp_ms") else x["timestamp_ms"]

            merged = sorted(audios + frames, key=_ts)
            for item in merged:
                if self.barge_in._event.is_set():
                    break
                if isinstance(item, TTSAudioChunk):
                    await self._emit_audio(item, tracer)
                else:
                    if not marked_frame:
                        tracer.mark("drive_first_frame", ts=item["timestamp_ms"])
                        marked_frame = True
                    await self._emit(item)
                await asyncio.sleep(0)  # 让出事件循环（流式节奏）
        self._history.append({
            "user": user_text,
            "assistant": " ".join(c.text for c in chunks),
            "trace_id": tracer.trace_id,
            "audio_ms": total_audio_ms,
            "summary": tracer.summary(),
        })
        tracer.mark("turn_end", sentences=len(chunks), audio_ms=total_audio_ms)
        self.last_trace = tracer.summary()
        await self._broadcast(json.dumps(
            {"type": "turn_end", "trace_id": tracer.trace_id, "summary": tracer.summary()}))


def create_app() -> FastAPI:
    app = FastAPI(title="Avatar V2 Drive Service", lifespan=lifespan)

    # 开发期 CORS：Web 调试台（:8080 静态服务）跨域访问 API（:8765）。
    # 生产部署建议同源伺服或收紧 allow_origins。
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        svc: ConversationService = app.state.service
        return {
            "status": "ok",
            "service": "avatar-v2",
            "llm": svc.llm_kind,
            "tts": svc.tts_kind,
            "renderer": svc.renderer.status["current"],
            "turn_running": svc._turn_running,
            "subscribers": len(svc._subscribers),
            "last_trace": svc.last_trace,
            "last_error": svc.last_error,
        }

    @app.get("/renderers")
    async def renderers():
        svc: ConversationService = app.state.service
        return svc.renderer.status

    @app.post("/renderers/switch")
    async def switch(r: SwitchRequest):
        svc: ConversationService = app.state.service
        svc.renderer.switch(r.name, reason=r.reason)
        return svc.renderer.status

    @app.post("/renderers/telemetry")
    async def telemetry(tel: dict):
        svc: ConversationService = app.state.service
        svc.renderer.update_telemetry(tel)
        return {"ok": True, "current": svc.renderer.status["current"]}

    @app.post("/chat")
    async def chat(req: ChatRequest):
        svc: ConversationService = app.state.service
        trace_id = new_trace_id()
        asyncio.create_task(svc.start_turn(req.text, req.renderer, req.llm, req.user_emotion,
                                           req.tts, trace_id))
        return {"ok": True, "turn_started": True, "trace_id": trace_id}

    @app.post("/emotion/audio")
    async def emotion_audio(file: UploadFile = File(...)):
        """上传用户语音（wav）→ SenseVoice 识别文本 + 情绪 → 更新用户情绪状态。"""
        svc: ConversationService = app.state.service
        rec = svc.recognizer
        if rec is None:
            return JSONResponse({"ok": False, "error": svc.last_error}, status_code=503)
        data = await file.read()
        import io
        import wave

        with wave.open(io.BytesIO(data), "rb") as w:
            rate = w.getframerate()
            if rate != 16000:
                return JSONResponse({"ok": False, "error": f"采样率须为 16000，收到 {rate}"}, status_code=400)
            pcm = w.readframes(w.getnframes())
        try:
            r = await asyncio.to_thread(rec.recognize, pcm, 16000)
        except Exception as e:
            return JSONResponse({"ok": False, "error": f"{type(e).__name__}: {e}"}, status_code=500)
        svc.user_tracker.push(r)
        return {
            "ok": True,
            "text": r.text,
            "emotion": r.emotion.to_dict(),
            "raw_label": r.raw_label,
            "audio_events": r.audio_events,
            "language": r.language,
            "user_state": svc.user_tracker.current.to_dict() if svc.user_tracker.current else None,
        }

    @app.get("/emotion/state")
    async def emotion_state():
        svc: ConversationService = app.state.service
        t = svc.user_tracker
        return {"ok": True, "user_emotion": t.current.to_dict() if t.current else None,
                "last_text": t.last_text}

    @app.post("/voice/turn")
    async def voice_turn(file: UploadFile = File(...), llm: str | None = None,
                         tts: str | None = None, renderer: str | None = None):
        """语音输入闭环：用户语音 wav(16k) → SenseVoice 识别（文本+情绪）
        → 更新用户情绪状态 → 自动发起一轮带共情上下文的对话。

        llm/tts/renderer 为可选 query 参数，指定本轮后端。
        """
        svc: ConversationService = app.state.service
        rec = svc.recognizer
        if rec is None:
            return JSONResponse({"ok": False, "error": svc.last_error}, status_code=503)
        data = await file.read()
        import io
        import wave

        try:
            with wave.open(io.BytesIO(data), "rb") as w:
                rate = w.getframerate()
                if rate != 16000:
                    return JSONResponse(
                        {"ok": False, "error": f"采样率须为 16000，收到 {rate}"}, status_code=400)
                pcm = w.readframes(w.getnframes())
        except Exception as e:
            return JSONResponse({"ok": False, "error": f"wav 解析失败: {e}"}, status_code=400)
        try:
            r = await asyncio.to_thread(rec.recognize, pcm, 16000)
        except Exception as e:
            return JSONResponse({"ok": False, "error": f"{type(e).__name__}: {e}"}, status_code=500)
        svc.user_tracker.push(r)
        if not (r.text or "").strip():
            return JSONResponse({"ok": False, "error": "未识别到语音（静音或过短）"}, status_code=400)
        trace_id = new_trace_id()
        asyncio.create_task(svc.start_turn(r.text, renderer, llm, r.emotion.label, tts, trace_id))
        return {
            "ok": True,
            "text": r.text,
            "emotion": r.emotion.to_dict(),
            "raw_label": r.raw_label,
            "audio_events": r.audio_events,
            "language": r.language,
            "trace_id": trace_id,
        }

    @app.get("/history")
    async def history():
        svc: ConversationService = app.state.service
        return {"history": svc._history}

    @app.post("/barge_in")
    async def barge_in():
        svc: ConversationService = app.state.service
        svc.barge_in.trigger()
        return {"ok": True, "queues_flushed": True}

    @app.post("/tts/preview")
    async def tts_preview(req: PreviewRequest):
        """文本 → TTS 音频（wav），不经对话管线，快速验证 TTS 后端可用性。"""
        svc: ConversationService = app.state.service
        tts = make_tts(req.tts) if req.tts else svc.tts
        from avatar.pipeline.emotion import emotion_anchor
        from avatar.pipeline.llm import SentenceChunk

        chunk = SentenceChunk(text=req.text, emotion=emotion_anchor("calm", 0.4),
                              semantic_meta={"intent": "preview", "style": "自然引导", "cause": "手动预览"})
        pcm = bytearray()
        sr = 16000
        t0 = time.perf_counter()
        try:
            async for a in tts.synthesize(chunk, {}, 0):
                pcm += a.pcm_s16le
        except Exception as e:
            return JSONResponse({"ok": False, "error": f"{type(e).__name__}: {e}"}, status_code=500)
        try:
            sr = int(tts.sample_rate) or 16000
        except Exception:
            pass
        if not pcm:
            return JSONResponse({"ok": False, "error": "empty audio"}, status_code=503)
        dur_ms = len(pcm) // 2 * 1000 // sr
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(bytes(pcm))
        took_ms = (time.perf_counter() - t0) * 1000
        return Response(
            content=buf.getvalue(),
            media_type="audio/wav",
            headers={"X-Duration-Ms": str(dur_ms), "X-Synth-Ms": str(round(took_ms, 1)),
                     "X-Sample-Rate": str(sr), "X-RTF": str(round(took_ms / max(1, dur_ms), 3)),
                     "X-TTS": getattr(tts, "name", "mock")},
        )

    @app.get("/sse/drive")
    async def sse_drive():
        svc: ConversationService = app.state.service
        q = svc.subscribe()

        async def gen():
            try:
                while True:
                    payload, _ = await q.get()
                    yield f"data: {payload}\n\n"
            except asyncio.CancelledError:
                pass
            finally:
                svc.unsubscribe(q)

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.websocket("/ws/drive")
    async def ws_drive(ws: WebSocket):
        await ws.accept()
        svc: ConversationService = app.state.service
        q = svc.subscribe()
        mode = ["binary"]  # 默认二进制 protobuf；客户端首帧发 {"mode":"json"} 切 JSON
        try:
            reader = asyncio.create_task(_ws_reader(ws, svc, mode))
            while True:
                payload, binary = await q.get()
                if binary is None or mode[0] == "json":
                    # 事件消息（turn_start/turn_end/error）或 JSON 模式 → 文本帧
                    await ws.send_text(payload)
                else:
                    await ws.send_bytes(binary)
        except WebSocketDisconnect:
            pass
        finally:
            reader.cancel()
            svc.unsubscribe(q)

    async def _ws_reader(ws: WebSocket, svc: ConversationService, mode: list[str]) -> None:
        while True:
            try:
                raw = await ws.receive()
            except Exception:
                break
            if raw.get("type") == "websocket.disconnect":
                break
            msg = raw.get("text") or raw.get("bytes")
            if isinstance(msg, (bytes, bytearray)):
                continue
            try:
                obj = json.loads(msg)
            except json.JSONDecodeError:
                continue
            kind = obj.get("type")
            if kind == "mode":
                mode[0] = obj.get("mode", "binary")
            elif kind == "barge_in":
                svc.barge_in.trigger()
            elif kind == "chat":
                text = obj.get("text", "")
                asyncio.create_task(svc.start_turn(text, obj.get("renderer"), obj.get("llm"),
                                                   obj.get("user_emotion"), obj.get("tts"),
                                                   obj.get("trace_id")))

    @app.websocket("/ws/voice")
    async def ws_voice(ws: WebSocket):
        """流式语音识别：浏览器推 PCM，服务端增量分块识别实时回推。

        协议（客户端 → 服务端）：
        - text {"type":"start","llm":..,"tts":..,"auto_end":bool}  开始会话
        - binary 16k mono s16le PCM 块（录音期间持续推）
        - text {"type":"stop"}    手动收尾（最终识别 + 自动起对话轮）
        - text {"type":"cancel"}  丢弃本次录音

        服务端 → 客户端（JSON）：
        - {"type":"ready","trace_id":..,"sr":16000}
        - {"type":"partial","text":..}                 实时字幕
        - {"type":"final","text":..,"emotion":..}      最终识别
        - {"type":"turn_started","trace_id":..}        对话轮已派发
        - {"type":"error"|"cancelled",...}
        """
        await ws.accept()
        svc: ConversationService = app.state.service
        rec = svc.recognizer
        if rec is None:
            await ws.send_text(json.dumps(
                {"type": "error", "detail": svc.last_error or "recognizer unavailable"}))
            await ws.close()
            return

        session: object | None = None  # 当前 VoiceSession
        ticker: asyncio.Task | None = None
        opts: dict = {}

        async def on_event(ev: dict) -> None:
            await ws.send_text(json.dumps(ev, ensure_ascii=False))

        async def on_final(r) -> None:
            # 最终结果 → 共情起轮（与 /voice/turn 同语义），复用会话 Tracer
            if r is None or not (getattr(r, "text", "") or "").strip():
                return
            svc.user_tracker.push(r)
            assert session is not None
            tracer = session.tracer
            tracer.mark("turn_dispatch", text=r.text[:20])
            asyncio.create_task(svc.start_turn(
                r.text, opts.get("renderer"), opts.get("llm"),
                r.emotion.label, opts.get("tts"), tracer=tracer))
            await ws.send_text(json.dumps(
                {"type": "turn_started", "trace_id": tracer.trace_id}))

        async def run_ticker(s) -> None:
            try:
                while True:
                    await asyncio.sleep(0.1)
                    await s.tick()
            except asyncio.CancelledError:
                pass

        def stop_ticker() -> None:
            nonlocal ticker
            if ticker is not None:
                ticker.cancel()
                ticker = None

        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                raw = msg.get("bytes")
                if raw:
                    if session is not None and not session._done:
                        session.feed(raw)
                    continue
                text = msg.get("text")
                if not text:
                    continue
                try:
                    obj = json.loads(text)
                except json.JSONDecodeError:
                    continue
                kind = obj.get("type")
                if kind == "start":
                    if session is not None:
                        stop_ticker()
                        await session.cancel()
                    opts = obj
                    from avatar.pipeline.voice_stream import VoiceSession

                    session = VoiceSession(
                        recognizer=rec, on_event=on_event, on_final=on_final,
                        tracer=Tracer(new_trace_id(), label="voice"),
                        auto_end=bool(obj.get("auto_end", False)),
                        silence_ms=int(obj.get("silence_ms", 900)),
                    )
                    session.tracer.mark("voice_start", auto_end=session.auto_end)
                    ticker = asyncio.create_task(run_ticker(session))
                    await ws.send_text(json.dumps(
                        {"type": "ready", "trace_id": session.tracer.trace_id,
                         "sr": 16000, "fmt": "s16le_mono"}))
                elif kind == "stop" and session is not None:
                    stop_ticker()
                    await session.finalize("user_stop")
                elif kind == "cancel" and session is not None:
                    stop_ticker()
                    await session.cancel()
        except WebSocketDisconnect:
            pass
        finally:
            stop_ticker()

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return HTMLResponse("""<h3>Avatar V2 Drive Service</h3>
<p><a href="/docs">OpenAPI docs</a></p>
<p>SSE: <a href="/sse/drive">/sse/drive</a> &nbsp; WS: /ws/drive &nbsp; 控制: /chat /barge_in /renderers</p>
<p>Web 调试页: <code>python -m http.server -d web 8080</code> → renderer.html</p>""")

    return app


app = create_app()