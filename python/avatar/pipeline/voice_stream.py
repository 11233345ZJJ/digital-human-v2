"""流式语音会话：边说边识别（增量分块），实时推送部分/最终结果。

SenseVoice 原生非流式（需求 7 风险项）→ 采用「增量分块识别」策略：
- 音频持续累积；每隔 partial_interval_ms 对「尾窗」（最近 partial_window_ms）
  做一次识别，推 partial（实时字幕）
- finalize（用户停止 / 自动断句）时对全量音频（cap max_audio_ms）做最终识别
- 简单能量 VAD（100ms 块 RMS 阈值）支持自动断句：说话≥speech_min_ms 后
  静音≥silence_ms 自动 finalize，用户无需手动点停止
- 识别串行化：识别在线程池执行，单会话内 partial/final 互斥（识别器
  非线程安全；跳过式限频：忙则本周期不识别）
"""
from __future__ import annotations

import asyncio
import math
from array import array
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

BYTES_PER_MS_16K = 32  # 16000 Hz × 2 字节 / 1000


def rms_s16(pcm: bytes) -> float:
    """16-bit PCM 块的 RMS 能量。"""
    if not pcm:
        return 0.0
    a = array("h")
    a.frombytes(pcm[: len(pcm) // 2 * 2])
    return math.sqrt(sum(x * x for x in a) / len(a))


@dataclass
class VoiceSession:
    """一次流式语音会话（start → feed* → finalize/cancel）。

    事件回调 on_event(dict)：partial / final / cancelled / error。
    结果回调 on_final(EmotionRecognition | None)：final 后触发（可在此起对话轮）。
    """

    recognizer: Any
    on_event: Callable[[dict], Awaitable[None]]
    on_final: Callable[[Any], Awaitable[None]] | None = None
    tracer: Any | None = None
    sample_rate: int = 16000
    partial_interval_ms: int = 800     # 新增音频达到该量才尝试一次 partial
    partial_window_ms: int = 12000     # partial 只识别最近窗口（限制计算量）
    partial_min_ms: int = 300          # 尾窗不足该时长不识别（无意义）
    max_audio_ms: int = 30000          # finalize 最大识别时长（防爆内存/时长）
    auto_end: bool = False             # 能量 VAD 自动断句
    silence_ms: int = 900              # 自动断句静音阈值
    speech_min_ms: int = 600           # 自动断句要求的最小说话时长
    rms_threshold: float = 350.0       # 100ms 块判定为说话的 RMS 阈值

    # ---- 运行态 ----
    buf: bytearray = field(default_factory=bytearray, repr=False)
    _vad_pending: bytearray = field(default_factory=bytearray, repr=False)
    speech_ms: int = 0
    silence_ms_now: int = 0
    _has_speech: bool = False
    _last_partial_bytes: int = 0
    _partial_task: asyncio.Task | None = None
    _done: bool = False
    _final_result: Any | None = None
    _partial_count: int = 0
    _sem: asyncio.Semaphore = field(default_factory=asyncio.Semaphore, repr=False)

    @property
    def trace_id(self) -> str:
        return getattr(self.tracer, "trace_id", "") or ""

    @property
    def audio_ms(self) -> int:
        return len(self.buf) // BYTES_PER_MS_16K

    # ---- 输入 ----
    def feed(self, pcm: bytes) -> None:
        """追加原始 PCM（16k mono s16le），并更新能量 VAD 计数。"""
        if self._done or not pcm:
            return
        self.buf.extend(pcm)
        self._vad_pending.extend(pcm)
        step = self.sample_rate // 10 * 2  # 100ms 块
        while len(self._vad_pending) >= step:
            blk = bytes(self._vad_pending[:step])
            del self._vad_pending[:step]
            if rms_s16(blk) >= self.rms_threshold:
                self.speech_ms += 100
                self.silence_ms_now = 0
                self._has_speech = True
                self._speech_end_ms = self.audio_ms
            else:
                self.silence_ms_now += 100

    # ---- 周期驱动（端点以 ~100ms 调用） ----
    async def tick(self) -> None:
        if self._done:
            return
        # 自动断句：说过话 + 已静音超阈值 → 直接收尾
        if (self.auto_end and self._has_speech
                and self.speech_ms >= self.speech_min_ms
                and self.silence_ms_now >= self.silence_ms):
            await self.finalize("auto_end")
            return
        grew = len(self.buf) - self._last_partial_bytes
        if grew >= self.partial_interval_ms * BYTES_PER_MS_16K and (
                self._partial_task is None or self._partial_task.done()):
            self._partial_task = asyncio.create_task(self._run_partial())

    async def _run_partial(self) -> None:
        if self._done:
            return
        self._last_partial_bytes = len(self.buf)
        window = bytes(self.buf[-self.partial_window_ms * BYTES_PER_MS_16K:])
        if len(window) < self.partial_min_ms * BYTES_PER_MS_16K:
            return
        cover_ms = len(self.buf) // BYTES_PER_MS_16K
        async with self._sem:
            try:
                r = await asyncio.to_thread(self.recognizer.recognize, window, self.sample_rate)
            except Exception as e:
                await self._emit({"type": "error", "stage": "partial",
                                  "detail": f"{type(e).__name__}: {e}"})
                return
        text = (getattr(r, "text", "") or "").strip()
        if not text:
            return  # 静音/无有效内容不打扰前端
        self._partial_result = r
        self._partial_cover_ms = cover_ms
        self._partial_count += 1
        if self._partial_count == 1 and self.tracer is not None:
            self.tracer.mark("asr_first_partial", audio_ms=self.audio_ms)
        await self._emit({
            "type": "partial", "text": text, "audio_ms": self.audio_ms,
            "speech_ms": self.speech_ms, "trace_id": self.trace_id,
        })

    # ---- 收尾 ----
    def _fastpath_result(self) -> Any | None:
        """复用最近 partial 作为 final 的条件：
        该 partial 的识别窗口已覆盖全部语音（含最后一个有声块）。
        满足时省掉全量重识别（CPU 上 6s 音频可省 ~2.5s）。
        """
        r = self._partial_result
        if r is None or not self._has_speech:
            return None
        if self._partial_cover_ms >= self._speech_end_ms:
            return r
        return None

    async def finalize(self, reason: str = "user_stop") -> Any | None:
        """全量最终识别 → 推 final → on_final 回调。返回识别结果。

        fastpath：语音已结束且最近 partial 覆盖全部语音 → 直接复用 partial
        结果（auto_end 场景说完 ~1s 即可出 final，无需再等全量识别）。
        """
        if self._done:
            return self._final_result
        self._done = True
        if self._partial_task is not None and not self._partial_task.done():
            try:
                await self._partial_task
            except Exception:
                pass
        r = self._fastpath_result() if reason == "auto_end" else None
        fastpath = r is not None
        if r is None:
            audio = bytes(self.buf[: self.max_audio_ms * BYTES_PER_MS_16K])
            if len(audio) < 100 * BYTES_PER_MS_16K:
                # 几乎没录到东西：空 final，不触发对话轮
                await self._emit({"type": "final", "text": "", "empty": True,
                                  "reason": reason, "audio_ms": self.audio_ms,
                                  "trace_id": self.trace_id})
                if self.on_final is not None:
                    await self.on_final(None)
                return None
            async with self._sem:
                try:
                    r = await asyncio.to_thread(self.recognizer.recognize, audio, self.sample_rate)
                except Exception as e:
                    await self._emit({"type": "error", "stage": "final",
                                      "detail": f"{type(e).__name__}: {e}"})
                    if self.on_final is not None:
                        await self.on_final(None)
                    return None
            self._final_result = r
            audio_ms = len(audio) // BYTES_PER_MS_16K
        else:
            self._final_result = r
            audio_ms = self.audio_ms
        if self.tracer is not None:
            self.tracer.mark("asr_final", reason=reason, fastpath=fastpath, audio_ms=audio_ms)
        await self._emit({
            "type": "final", "text": getattr(r, "text", ""),
            "emotion": getattr(r, "emotion", None).to_dict() if getattr(r, "emotion", None) else None,
            "raw_label": getattr(r, "raw_label", ""),
            "audio_events": list(getattr(r, "audio_events", []) or []),
            "language": getattr(r, "language", ""),
            "audio_ms": audio_ms,
            "reason": reason, "fastpath": fastpath, "trace_id": self.trace_id,
        })
        if self.on_final is not None:
            await self.on_final(r)
        return r

    async def cancel(self) -> None:
        self._done = True
        if self._partial_task is not None and not self._partial_task.done():
            self._partial_task.cancel()
        await self._emit({"type": "cancelled", "trace_id": self.trace_id})

    async def _emit(self, ev: dict) -> None:
        try:
            await self.on_event(ev)
        except Exception:
            pass  # 连接断开等：事件推送失败不阻塞会话
