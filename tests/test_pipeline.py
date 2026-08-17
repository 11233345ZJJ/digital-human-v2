"""管线单测：情绪中间表示 / 平滑器 / 切句 / 时间线 / 驱动帧 / proto 编解码。"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import pytest

from avatar.pipeline.chunker import StreamingChunker
from avatar.pipeline.core import TimelineQueue
from avatar.pipeline.driving import DrivingStage
from avatar.pipeline.emotion import (
    Emotion,
    EmotionSmoother,
    emotion_anchor,
    slerp_vec,
    vector_to_vad,
)
from avatar.pipeline.llm import MockLLM
from avatar.pipeline.tts import TTSTimeline
from avatar.protocol import drive_pb2
from avatar.protocol.jsoncodec import (
    decode_binary,
    dict_to_proto,
    encode_binary,
    proto_to_dict,
    validate_json,
)
from avatar.renderer.base import RendererManager, ease_in_out_cubic
from avatar.renderer.backends import build_mock_registry


def test_emotion_anchor_invariant():
    e = emotion_anchor("happy", 0.8)
    assert len(e.vector) == 8
    assert abs(sum(e.vector) - 1.0) < 1e-6
    assert e.label == "happy"
    vad = vector_to_vad(e.vector)
    assert set(vad) == {"valence", "arousal", "dominance"}
    assert all(-1.0 <= v <= 1.0 for v in vad.values())


def test_slerp_endpoints_and_monotonic():
    a = emotion_anchor("calm").vector
    b = emotion_anchor("happy").vector
    assert slerp_vec(a, b, 0.0) == a
    assert slerp_vec(a, b, 1.0) == b
    m = slerp_vec(a, b, 0.5)
    assert all(min(x, y) <= m[i] <= max(x, y) for i, (x, y) in enumerate(zip(a, b)))


def test_smoother_no_jump():
    sm = EmotionSmoother(window_size=7, ewma_alpha=0.5)
    out = [sm.push(emotion_anchor("sad", 0.9)) for _ in range(3)]
    assert out[-1].label == "sad"
    out2 = [sm.push(emotion_anchor("happy", 0.9)) for _ in range(3)]
    vec = out2[-1].vector
    assert vec[0] < 0.9  # 平滑后不会瞬间跳到满强度 happy
    assert vec[0] > 0.1


def test_chunker_first_sentence_immediate():
    c = StreamingChunker()
    s = c.feed("你好呀。很高兴见到")
    assert s == ["你好呀。"]
    assert c.feed("你！") == ["很高兴见到你！"]


def test_chunker_en_abbreviation():
    c = StreamingChunker()
    assert c.feed("Hello world.") == ["Hello world."]
    assert c.feed(" This is a test!") == ["This is a test!"]


def test_chunker_does_not_split_abbreviation():
    c = StreamingChunker()
    out = c.feed("U.S.A. is great.")
    assert out[0] == "U.S.A."  # 不在缩写内部切分
    assert all(len(s) >= 4 for s in out)


def test_timeline_monotonic():
    tl = TTSTimeline()
    a = tl.reserve(10)
    b = tl.reserve(20)
    assert a[0] == 0 and a[1] == int(10 * 210)
    assert b[0] == a[1] and b[1] > b[0]


@pytest.mark.asyncio
async def test_mock_llm_streams_sentences():
    llm = MockLLM(delay_per_token_ms=0)
    out = [c async for c in llm.stream_turn("hi", {})]
    assert len(out) == 6
    assert all(c.emotion.label in {
        "happy", "calm", "surprised", "sad"} for c in out)


def test_driving_frames_ordered():
    from avatar.pipeline.emotion import emotion_anchor
    from avatar.pipeline.llm import SentenceChunk

    stage = DrivingStage()
    timeline = TTSTimeline()
    stage.timeline = timeline
    c = SentenceChunk(text="这是一个测试句子呀！", emotion=emotion_anchor("happy", 0.8),
                      semantic_meta={"intent": "x", "style": "兴奋推荐", "cause": "y"})
    start, end = timeline.reserve(len(c.text))
    frames = stage.frames_for(c, start, end)
    ts = [f["timestamp_ms"] for f in frames]
    assert ts == sorted(ts)
    assert all("expression" in f and "emotion" in f and "semantic_meta" in f for f in frames)
    assert frames[0]["body_gesture"]["name"] == "cheer"


def test_proto_roundtrip():
    from avatar.pipeline.llm import SentenceChunk
    from avatar.pipeline.emotion import emotion_anchor
    frame = {
        "timestamp_ms": 12345,
        "phoneme": "AA",
        "expression": {"name": "smile", "weight": 0.8},
        "head_pose": {"yaw": 0.1, "pitch": -0.05, "roll": 0.0},
        "body_gesture": {"name": "wave", "params": {"speed": 1.0}},
        "emotion": emotion_anchor("happy", 0.8).to_dict(),
        "semantic_meta": {"intent": "i", "style": "s", "cause": "c"},
        "frame_seq": 7,
    }
    msg = dict_to_proto(frame)
    assert msg.timestamp_ms == 12345
    assert list(msg.emotion.vector) == pytest.approx(list(frame["emotion"]["vector"]))
    raw = encode_binary(frame)
    back = decode_binary(raw)
    assert back["timestamp_ms"] == 12345
    assert back["emotion"]["label"] == "happy"
    assert back["expression"]["weight"] == pytest.approx(0.8)
    assert validate_json(back) == []


def test_proto_json_parity():
    from avatar.pipeline.emotion import emotion_anchor
    frame = {
        "timestamp_ms": 1, "frame_seq": 0,
        "expression": {"name": "neutral", "weight": 0.1},
        "head_pose": {"yaw": 0, "pitch": 0, "roll": 0},
        "body_gesture": {"name": "idle", "params": {}},
        "emotion": emotion_anchor("calm").to_dict(),
        "semantic_meta": {"intent": "", "style": "", "cause": ""},
    }
    raw = encode_binary(frame)
    back = decode_binary(raw)
    assert back["frame_seq"] == 0 and back["emotion"]["label"] == "calm"


def test_ease_in_out_cubic():
    assert ease_in_out_cubic(0.0) == 0.0
    assert ease_in_out_cubic(1.0) == 1.0
    assert ease_in_out_cubic(0.5) == pytest.approx(0.5)
    assert ease_in_out_cubic(0.25) < 0.25  # 缓入


def test_renderer_degrade_chain():
    mgr = RendererManager(build_mock_registry())
    mgr.select("flashhead")
    assert mgr.current.name == "flashhead"
    mgr.switch("live2d", reason="manual")
    assert mgr.current.name == "live2d"
    lower = mgr.registry.next_lower("live2d")
    assert lower == "audio"
    assert mgr.registry.next_lower("audio") is None


def test_renderer_auto_degrade_on_low_fps():
    import time

    mgr = RendererManager(build_mock_registry())
    mgr.degrade_policy["fps_duration_s"] = 0.001
    mgr.select("flashhead")
    mgr.update_telemetry({"renderer": "flashhead", "fps": 10.0, "gpu_temp_c": 50})
    time.sleep(0.005)
    mgr.update_telemetry({"renderer": "flashhead", "fps": 10.0, "gpu_temp_c": 50})
    assert mgr.current.name == "babylon3d"  # 降一级
    assert mgr.history[-1]["reason"].startswith("auto-degrade")


def test_renderer_auto_degrade_on_high_temp():
    mgr = RendererManager(build_mock_registry())
    mgr.select("babylon3d")
    mgr.update_telemetry({"renderer": "babylon3d", "fps": 60.0, "gpu_temp_c": 90})
    assert mgr.current.name == "live2d"


def test_timeline_queue_ordering_and_barge_in():
    q = TimelineQueue()
    q.push({"timestamp_ms": 300, "id": "a"})
    q.push({"timestamp_ms": 100, "id": "b"})
    q.push({"timestamp_ms": 200, "id": "c"})
    got = q.consume(200)
    assert [g["id"] for g in got] == ["b", "c"]
    q.push({"timestamp_ms": 500, "id": "d"})
    assert q.clear() == 2  # 剩余 a(300) + d(500)
    assert len(q) == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))