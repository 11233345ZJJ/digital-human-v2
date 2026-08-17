"""模拟语音流稳定性测试：构造「说话-停顿-说话-长静音」节奏，走 WS 实时推流，
验证增量识别（partial）与能量 VAD 自动断句（auto_end）的稳定性。

语料：SherpaTTS 本地合成 3 句真实中文语音（22.05k → 16k），
静音段用零字节模拟自然停顿。

场景（默认 3 轮）：
  会话A：句1 [0.5s 短停顿] 句2 [1.5s 长静音]  → 应 auto_end，final 含两句
  会话B（同连接重新 start）：句3 [1.2s 静音]   → 应 auto_end，final=句3
压力：--speed 2.0 以 2 倍速推流（50ms 块间隔），考察限频/丢块鲁棒性。

用法：
  python/.venv/bin/python scripts/voice_stream_sim.py [--rounds 3] [--speed 1.0]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import websockets  # noqa: E402

URI = "ws://127.0.0.1:8765/ws/voice"
SR = 16000
BYTES_PER_MS = 32  # 16k s16le

SENTENCES = [
    "今天天气真好，我们出去散步吧。",
    "顺便去超市买点水果和蔬菜。",
    "回来的时候记得取一下快递。",
]
KEYWORDS = [["散步"], ["蔬菜", "水果"], ["快递"]]


def silence(ms: int) -> bytes:
    return b"\x00\x00" * (SR * ms // 1000)


def resample_to_16k(pcm_22k: bytes, sr_from: int) -> bytes:
    import array

    a = array.array("h")
    a.frombytes(pcm_22k[: len(pcm_22k) // 2 * 2])
    n_out = int(len(a) * SR / sr_from)
    out = array.array("h", bytes(2 * n_out))
    ratio = len(a) / n_out
    for i in range(n_out):
        p = i * ratio
        i0 = int(p)
        f = p - i0
        x = a[i0] + ((a[i0 + 1] if i0 + 1 < len(a) else a[i0]) - a[i0]) * f
        out[i] = int(max(-32768, min(32767, x)))
    return out.tobytes()


async def synthesize_corpus() -> list[bytes]:
    """SherpaTTS 本地合成 3 句 16k PCM。"""
    from avatar.pipeline.emotion import emotion_anchor
    from avatar.pipeline.llm import SentenceChunk
    from avatar.pipeline.tts_sherpa import SherpaTTS

    out: list[bytes] = []
    tts = SherpaTTS()
    await tts.ensure_loaded()
    for text in SENTENCES:
        chunk = SentenceChunk(text=text, emotion=emotion_anchor("calm", 0.4),
                              semantic_meta={"intent": "sim", "style": "s", "cause": "c"})
        pcm = bytearray()
        async for a in tts.synthesize(chunk, {}, 0):
            pcm += a.pcm_s16le
        # 去掉句尾静音垫（模拟器自己控制停顿）
        body = bytes(pcm)[: int((len(pcm) / 2 / SR - 0.12) * SR) * 2]
        out.append(resample_to_16k(body, tts.sample_rate))
    return out


class Recorder:
    def __init__(self, t0: float):
        self.t0 = t0
        self.partials: list[tuple[float, str]] = []
        self.final: dict | None = None
        self.final_at: float = 0.0
        self.turn_started: bool = False
        self.errors: list[str] = []

    def on_msg(self, raw: str) -> None:
        o = json.loads(raw)
        t = time.perf_counter() - self.t0
        if o["type"] == "partial":
            self.partials.append((t, o["text"]))
        elif o["type"] == "final":
            self.final = o
            self.final_at = t
        elif o["type"] == "turn_started":
            self.turn_started = True
        elif o["type"] == "error":
            self.errors.append(o.get("detail", "?"))


async def run_session(track: list[tuple[bytes, str]], rec: Recorder,
                      speed: float) -> dict:
    """一个会话（独立 WS 连接，与浏览器行为一致）：推 [语音/静音] 序列，
    等 auto_end final + turn_started。后台常驻 reader 记录真实到达时间。
    """
    async with websockets.connect(URI, max_size=2**22) as ws:
        rt = asyncio.create_task(_reader(ws, rec))

        async def send_start():
            await ws.send(json.dumps({"type": "start", "llm": "mock", "tts": "sherpa",
                                      "auto_end": True, "silence_ms": 900}))

        await send_start()
        await asyncio.sleep(0.2)  # 等 ready（reader 在收）
        chunk_ms = int(100 / speed)
        total_ms = 0
        last_speech_end = None
        for pcm, tag in track:
            for i in range(0, len(pcm), BYTES_PER_MS * 100):
                await ws.send(pcm[i:i + BYTES_PER_MS * 100])
                await asyncio.sleep(chunk_ms / 1000)
            total_ms += len(pcm) // BYTES_PER_MS
            if tag == "speech":
                last_speech_end = time.perf_counter() - rec.t0
        # 等 auto_end final + turn_started（最长 15s）
        deadline = time.perf_counter() + 15
        while (rec.final is None or not rec.turn_started) and time.perf_counter() < deadline:
            await asyncio.sleep(0.05)
        rt.cancel()
    return {"last_speech_end": last_speech_end, "total_ms": total_ms}


async def _reader(ws, rec: Recorder) -> None:
    async for m in ws:
        rec.on_msg(m)


def verdict(rec: Recorder, want_keys: list[str], speech_end: float | None,
            total_ms: int) -> list[str]:
    problems = []
    if rec.errors:
        problems.append(f"error 事件: {rec.errors}")
    f = rec.final
    if f is None:
        problems.append("超时未收到 final（自动断句失效）")
        return problems
    if rec.final is not None and not rec.turn_started:
        problems.append("final 后未收到 turn_started（对话轮未派发）")
    if f.get("reason") != "auto_end":
        problems.append(f"final.reason={f.get('reason')}（期望 auto_end）")
    text = f.get("text", "")
    for k in want_keys:
        if k not in text:
            problems.append(f"final 文本缺少关键词『{k}』: {text}")
    if speech_end is not None:
        d = rec.final_at - speech_end
        # 预算：静音检测 ~1.0s + final 识别（fastpath≈0 / 全量 RTF≈0.5×时长）
        budget = 2.0 + (0 if f.get("fastpath") else total_ms / 1000 * 0.5)
        if d > budget:
            problems.append(f"断句→final 延迟 {d:.2f}s 超出预算 {budget:.1f}s")
    if not rec.partials:
        problems.append("流式期间无 partial（增量识别未工作）")
    return problems


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()

    print("合成测试语料（SherpaTTS）…")
    corpus = await synthesize_corpus()
    for s, pcm in zip(SENTENCES, corpus):
        print(f"  {len(pcm) // BYTES_PER_MS:5d}ms  {s}")
    # 场景：A = 句1 + 0.5s + 句2 + 1.5s长静音；B = 句3 + 1.2s 静音
    track_a = [(corpus[0], "speech"), (silence(500), "sil"),
               (corpus[1], "speech"), (silence(1500), "sil")]
    track_b = [(corpus[2], "speech"), (silence(1200), "sil")]

    all_ok = True
    t_start = time.perf_counter()
    for rnd in range(1, args.rounds + 1):
        for name, track, keys in (("A", track_a, KEYWORDS[0] + KEYWORDS[1]),
                                  ("B", track_b, KEYWORDS[2])):
            t0 = time.perf_counter()
            rec = Recorder(t0)
            info = await run_session(track, rec, args.speed)
            problems = verdict(rec, keys, info["last_speech_end"], info["total_ms"])
            status = "PASS" if not problems else "FAIL"
            all_ok &= not problems
            print(f"\n=== 第{rnd}轮 会话{name} [{status}] (speed={args.speed}x) ===")
            for ts, txt in rec.partials:
                print(f"  +{ts:6.2f}s 🎧 {txt}")
            if rec.final:
                f = rec.final
                d = rec.final_at - info["last_speech_end"] if info["last_speech_end"] else -1
                fp = "⚡fastpath" if f.get("fastpath") else "全量识别"
                print(f"  final({f['reason']}/{fp}) 延迟 {d:.2f}s: {f['text']}"
                      f" | 情绪 {f['emotion']['label']} | turn_started={rec.turn_started}")
            for p in problems:
                print(f"  ⚠ {p}")
    print(f"\n总计 {args.rounds * 2} 个会话，总耗时 {time.perf_counter() - t_start:.0f}s")
    print("总体:", "✅ 全部 PASS" if all_ok else "❌ 存在 FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
