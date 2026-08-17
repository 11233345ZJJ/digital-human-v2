"""Trace 全链路 + sherpa-onnx TTS 适配器 + 音频封套编解码 单测。"""
from __future__ import annotations

import asyncio
import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from avatar.pipeline.emotion import emotion_anchor
from avatar.pipeline.llm import SentenceChunk
from avatar.pipeline.tts_sherpa import SherpaTTS, float_pcm_to_s16_bytes
from avatar.protocol.jsoncodec import (
    decode_binary,
    encode_audio_binary,
    encode_binary,
    validate_json,
)
from avatar.trace import Tracer, new_trace_id


# ---------------- Trace ----------------

def test_trace_id_format():
    tid = new_trace_id()
    assert len(tid) == 12
    int(tid, 16)  # 合法 hex


def test_tracer_marks_and_summary():
    t = Tracer.start("测试轮次")
    t.mark("turn_start")
    t.mark("llm_first_token")
    t.mark("llm_first_chunk")
    t.mark("tts_first_audio")
    t.mark("drive_first_frame")
    t.mark("turn_end")
    s = t.summary()
    assert s["trace_id"] == t.trace_id
    for ev in ("llm_first_token_ms", "llm_first_chunk_ms", "tts_first_audio_ms",
               "drive_first_frame_ms", "turn_end_ms"):
        assert ev in s and s[ev] >= 0.0
    # 分段耗时 = 两里程碑差
    seg = s["tts_first_audio__from_llm_first_chunk_ms"]
    assert abs(seg - (s["tts_first_audio_ms"] - s["llm_first_chunk_ms"])) < 0.1


def test_tracer_elapsed_monotonic():
    t = Tracer.start()
    a = t.mark("a")
    b = t.mark("b")
    assert b >= a
    assert t.elapsed_of("a") == a
    assert t.elapsed_of("missing") is None


# ---------------- SherpaTTS（伪引擎，不依赖模型文件） ----------------

class _FakeGenerated:
    def __init__(self, samples, sample_rate):
        self.samples = samples
        self.sample_rate = sample_rate


class _FakeEngine:
    sample_rate = 22050

    def __init__(self):
        self.calls = []

    def generate(self, text, sid=0, speed=1.0):
        self.calls.append((text, speed))
        # 0.5s 正弦 + 0.1s 静音
        import math
        n = self.sample_rate // 2
        samples = [0.5 * math.sin(2 * math.pi * 220 * i / self.sample_rate) for i in range(n)]
        return _FakeGenerated(samples + [0.0] * (self.sample_rate // 10), self.sample_rate)


def _chunk(text="你好呀，很高兴见到你！"):
    return SentenceChunk(
        text=text,
        emotion=emotion_anchor("happy", 0.8),
        semantic_meta={"intent": "打招呼", "style": "亲切问候", "cause": "测试"},
    )


def _make_fake_tts():
    tts = SherpaTTS()
    tts._engine = _FakeEngine()
    return tts


def test_float_pcm_conversion():
    b = float_pcm_to_s16_bytes([0.0, 1.0, -1.0, 0.5])
    assert len(b) == 8
    assert b == b"\x00\x00" + (32767).to_bytes(2, "little", signed=True) \
        + (-32767).to_bytes(2, "little", signed=True) \
        + int(0.5 * 32767).to_bytes(2, "little", signed=True)


def test_sherpa_tts_chunking_timestamps():
    tts = _make_fake_tts()
    assert tts.sample_rate == 22050
    chunks = asyncio.run(_collect(tts.synthesize(_chunk(), {}, start_ms=1000)))
    # 0.5s 语音 + 0.1s 静音 + 0.12s 句尾垫 = 0.72s → 15 块（50ms）
    assert len(chunks) == 15
    assert chunks[0].timestamp_ms == 1000
    assert chunks[-1].timestamp_ms == 1000 + 14 * 50
    assert all(c.emotion.label == "happy" for c in chunks)
    assert chunks[-1].sentence_done is True
    assert all(not c.sentence_done for c in chunks[:-1])
    total = sum(len(c.pcm_s16le) for c in chunks)
    assert total == (22050 * (500 + 100 + 120) // 1000) * 2  # s16 双字节
    # 首块调用 speed：arousal→语速
    text, speed = tts.engine.calls[0]
    assert text == "你好呀，很高兴见到你！"
    assert 0.9 <= speed <= 1.1


def test_sherpa_tts_empty_text():
    tts = _make_fake_tts()
    chunks = asyncio.run(_collect(tts.synthesize(_chunk("  "), {}, 0)))
    assert chunks == []


def test_speed_for_bounds():
    tts = _make_fake_tts()
    hi = emotion_anchor("happy", 1.0)   # arousal 高
    lo = emotion_anchor("sad", 0.3)
    s_hi, s_lo = tts._speed_for(hi), tts._speed_for(lo)
    assert 0.9 <= s_lo < s_hi <= 1.1


# ---------------- 音频封套编解码 ----------------

def test_audio_envelope_roundtrip():
    d = {
        "timestamp_ms": 250,
        "duration_ms": 50,
        "pcm": b"\x01\x02\x03\x04",
        "sample_rate": 22050,
        "emotion": emotion_anchor("calm", 0.4).to_dict(),
        "trace_id": "abc123def456",
    }
    raw = encode_audio_binary(d)
    out = decode_binary(raw)
    assert out["type"] == "audio"
    assert out["timestamp_ms"] == 250
    assert base64.b64decode(out["pcm_b64"]) == b"\x01\x02\x03\x04"
    assert out["sample_rate"] == 22050
    assert out["trace_id"] == "abc123def456"
    assert out["emotion"]["label"] == "calm"
    assert len(out["emotion"]["vector"]) == 8


def test_drive_frame_trace_id_roundtrip():
    frame = {
        "timestamp_ms": 300,
        "frame_seq": 7,
        "phoneme": "AA",
        "expression": {"name": "smile", "weight": 0.8},
        "head_pose": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
        "body_gesture": {"name": "wave", "params": {"speed": 1.0}},
        "emotion": emotion_anchor("happy", 0.7).to_dict(),
        "semantic_meta": {"intent": "i", "style": "s", "cause": "c"},
        "trace_id": "trace0001",
    }
    out = decode_binary(encode_binary(frame))
    assert out["trace_id"] == "trace0001"
    errs = validate_json(out)
    assert errs == []


async def _collect(agen):
    out = []
    async for x in agen:
        out.append(x)
    return out
