"""智谱 LLM 适配器单测：JSONL 流式解析 / 兜底路径 / 情绪标签归一化。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import pytest

from avatar.pipeline.emotion import estimate_emotion, normalize_label
from avatar.pipeline.llm_zhipu import _JsolEmitter, _obj_to_chunk, _parse_line, _sentence_to_chunk


def test_parse_valid_line():
    obj = _parse_line('{"t":"你好呀！","e":"happy","i":0.8,"intent":"打招呼","style":"亲切问候","cause":"x"}')
    assert obj is not None and obj["t"] == "你好呀！"
    assert _parse_line("not json") is None
    assert _parse_line('{"e":"happy"}') is None  # 缺 t


def test_emitter_streams_sentences_immediately():
    em = _JsolEmitter()
    chunks = em.feed('{"t":"你好呀！","e":"happy","i":0.8,"intent":"打招呼","style":"亲切问候","cause":"x"}\n{"t":"')
    assert len(chunks) == 1  # 首句即推
    assert chunks[0].text == "你好呀！"
    assert chunks[0].emotion.label == "happy"
    assert chunks[0].semantic_meta["style"] == "亲切问候"
    rest = em.feed('今天天气不错。","e":"calm","i":0.5,"intent":"闲聊","style":"自然引导","cause":"y"}\n')
    assert len(rest) == 1
    assert rest[0].text == "今天天气不错。"


def test_emitter_partial_line_waits():
    em = _JsolEmitter()
    assert em.feed('{"t":"还在路上') == []
    chunks = em.feed('…","e":"calm","i":0.3}\n')
    assert len(chunks) == 1 and chunks[0].text == "还在路上…"


def test_emitter_flush_incomplete_json_falls_back_to_rule():
    em = _JsolEmitter()
    assert em.feed('{"t":"未完成的行') == []
    out = em.flush()
    assert len(out) == 1
    assert out[0].emotion.label == "calm"  # 无法解析 → 规则兜底


def test_emitter_non_json_line_text_fallback():
    em = _JsolEmitter()
    chunks = em.feed("你好呀。很高兴见到你！\n")
    assert len(chunks) == 2
    assert chunks[0].text == "你好呀。"
    assert chunks[1].text == "很高兴见到你！"
    assert all(c.emotion.label == "calm" for c in chunks[:1])
    assert chunks[1].emotion.label == "happy"  # "很高兴"命中规则词典


def test_normalize_label_maps_cn():
    assert normalize_label("happy") == "happy"
    assert normalize_label("高兴") == "happy"
    assert normalize_label("愤怒") == "angry"
    assert normalize_label("震惊") == "calm"  # 未知词 → calm
    assert normalize_label("") == "calm"
    assert normalize_label(None) == "calm"


def test_obj_to_chunk_with_8dim_vector():
    c = _obj_to_chunk({"t": "ok", "e": "calm", "i": 0.3, "v": [0, 0, 0, 0, 0, 0, 0, 1]})
    assert c.emotion.vector == (0, 0, 0, 0, 0, 0, 0, 1)
    bad = _obj_to_chunk({"t": "ok", "e": "calm", "i": 0.3, "v": [1, 2]})
    assert len(bad.emotion.vector) == 8  # 非法向量 → 锚点


def test_estimate_emotion_rules():
    assert estimate_emotion("哈哈太好了").label == "happy"
    assert estimate_emotion("对不起，我很抱歉").label == "sad"
    assert estimate_emotion("简直气死我了").label == "angry"
    assert estimate_emotion("随便聊聊").label == "calm"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))