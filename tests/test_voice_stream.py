"""VoiceSession 单测：能量 VAD 计数 / partial 增量 / 自动断句 / 窗口上限 / 取消。"""
from __future__ import annotations

import asyncio
import math
import sys
from array import array
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from avatar.pipeline.emotion import emotion_anchor
from avatar.pipeline.emotion_recognizer import EmotionRecognition
from avatar.pipeline.voice_stream import BYTES_PER_MS_16K, VoiceSession, rms_s16
from avatar.trace import Tracer


class FakeRec:
    """伪识别器：文本带音频毫秒数，记录每次调用时长（验证窗口上限）。"""

    def __init__(self, prefix="t"):
        self.calls: list[int] = []
        self.prefix = prefix

    def recognize(self, audio: bytes, sample_rate: int = 16000):
        ms = len(audio) // BYTES_PER_MS_16K
        self.calls.append(ms)
        return EmotionRecognition(text=f"{self.prefix}{ms}", emotion=emotion_anchor("happy", 0.5),
                                  raw_label="HAPPY", language="zh")


def pcm_sine(ms: int, amp: float = 0.3, sr: int = 16000) -> bytes:
    n = sr * ms // 1000
    a = array("h", (int(amp * 32767 * math.sin(2 * math.pi * 220 * i / sr)) for i in range(n)))
    return a.tobytes()


def pcm_silence(ms: int, sr: int = 16000) -> bytes:
    return b"\x00\x00" * (sr * ms // 1000)


async def _drain(session: VoiceSession, rounds: int = 5) -> None:
    for _ in range(rounds):
        await session.tick()
        await asyncio.sleep(0.01)  # 让 partial task 跑完（FakeRec 极快）


def make(rec: FakeRec | None = None, **kw) -> tuple[VoiceSession, list[dict], list]:
    rec = rec or FakeRec()
    events: list[dict] = []
    finals: list = []

    async def on_event(ev):
        events.append(ev)

    async def on_final(r):
        finals.append(r)

    s = VoiceSession(recognizer=rec, on_event=on_event, on_final=on_final,
                     tracer=Tracer("test0001", "t"), **kw)
    return s, events, finals


def test_rms_s16():
    assert rms_s16(b"") == 0.0
    assert rms_s16(pcm_silence(100)) == 0.0
    loud = rms_s16(pcm_sine(100, 0.3))
    assert loud > 3000  # 0.3 幅度正弦 RMS ≈ 0.3/√2 × 32767 ≈ 6958


def test_feed_vad_counters():
    s, _, _ = make()
    s.feed(pcm_sine(1000))
    assert s.speech_ms == 1000
    assert s.silence_ms_now == 0
    s.feed(pcm_silence(1000))
    assert s.silence_ms_now == 1000
    assert s.speech_ms == 1000  # 说话时长不回退
    s.feed(pcm_sine(200))
    assert s.silence_ms_now == 0
    assert s.speech_ms == 1200


def test_partial_emission_and_trace():
    s, events, _ = make()
    s.feed(pcm_sine(1500))
    asyncio.run(_drain(s))
    partials = [e for e in events if e["type"] == "partial"]
    assert partials, "应至少产生一个 partial"
    assert partials[0]["text"] == "t1500"  # 尾窗=全量
    assert partials[0]["trace_id"] == "test0001"
    assert s.tracer.elapsed_of("asr_first_partial") is not None


def test_partial_skip_when_busy_or_short():
    # 不足 partial_min_ms 不识别
    s, events, _ = make()
    s.feed(pcm_sine(200))
    asyncio.run(_drain(s))
    assert not [e for e in events if e["type"] == "partial"]


def test_partial_window_cap():
    rec = FakeRec()
    s, _, _ = make(rec, partial_window_ms=2000)
    s.feed(pcm_sine(6000))
    asyncio.run(_drain(s))
    assert rec.calls, "应发生识别调用"
    assert max(rec.calls) <= 2000, "partial 识别时长不得超过尾窗"


def test_finalize_user_stop():
    rec = FakeRec()
    s, events, finals = make(rec)
    s.feed(pcm_sine(1000))
    r = asyncio.run(s.finalize("user_stop"))
    assert r is not None and finals == [r]
    fin = [e for e in events if e["type"] == "final"][0]
    assert fin["text"] == "t1000"
    assert fin["reason"] == "user_stop"
    assert fin["emotion"]["label"] == "happy"
    assert s.tracer.elapsed_of("asr_final") is not None
    # 已收尾：后续 feed/tick 无效
    s.feed(pcm_sine(500))
    assert s.audio_ms == 1000


def test_finalize_empty_audio():
    s, events, finals = make()
    r = asyncio.run(s.finalize("user_stop"))
    assert r is None and finals == [None]
    fin = [e for e in events if e["type"] == "final"][0]
    assert fin["empty"] is True and fin["text"] == ""


def test_auto_end_triggers_finalize():
    s, events, finals = make(auto_end=True, silence_ms=900, speech_min_ms=600)
    s.feed(pcm_sine(1000))
    asyncio.run(_drain(s, rounds=2))
    assert not finals, "仍在说话，不应收尾"
    s.feed(pcm_silence(1000))
    asyncio.run(_drain(s, rounds=2))
    fins = [e for e in events if e["type"] == "final"]
    assert fins and fins[0]["reason"] == "auto_end"
    assert finals and finals[0] is not None


def test_auto_end_not_before_min_speech():
    s, events, _ = make(auto_end=True, silence_ms=900, speech_min_ms=600)
    s.feed(pcm_sine(300))     # 说话不足下限
    s.feed(pcm_silence(1200))
    asyncio.run(_drain(s, rounds=3))
    assert not [e for e in events if e["type"] == "final"]


def test_auto_end_fastpath_reuses_partial():
    """auto_end 且最近 partial 已覆盖全部语音 → 复用 partial 结果，不再全量识别。"""
    rec = FakeRec(prefix="p")
    s, events, finals = make(rec, auto_end=True, silence_ms=900)
    s.feed(pcm_sine(2000))
    asyncio.run(_drain(s, rounds=4))          # partial 已跑完（覆盖 2000ms）
    n_calls_before = len(rec.calls)
    s.feed(pcm_silence(1000))
    asyncio.run(_drain(s, rounds=3))          # 触发 auto_end
    fin = [e for e in events if e["type"] == "final"][0]
    assert fin["reason"] == "auto_end"
    assert fin["fastpath"] is True
    assert fin["text"].startswith("p")        # 复用的是 partial 的结果（前缀 p）
    assert len(rec.calls) == n_calls_before   # 未发生新的全量识别调用


def test_user_stop_always_full_recognition():
    """手动停止不走 fastpath：即使 partial 覆盖，仍全量识别（准确性优先）。"""
    rec = FakeRec(prefix="p")
    s, events, _ = make(rec)
    s.feed(pcm_sine(2000))
    asyncio.run(_drain(s, rounds=4))
    n_before = len(rec.calls)
    asyncio.run(s.finalize("user_stop"))
    assert len(rec.calls) == n_before + 1      # 多了一次全量识别
    fin = [e for e in events if e["type"] == "final"][0]
    assert fin.get("fastpath") is False


def test_cancel():
    s, events, finals = make()
    s.feed(pcm_sine(800))
    asyncio.run(s.cancel())
    assert [e for e in events if e["type"] == "cancelled"]
    assert finals == []  # cancel 不触发 final/on_final
    assert asyncio.run(s.finalize("user_stop")) is None  # 已终止，不再识别
