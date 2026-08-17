"""情感识别单测：SenseVoice token 解析 / 标签映射 / 输入侧情绪跟踪。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import pytest

from avatar.pipeline.emotion import emotion_anchor
from avatar.pipeline.emotion_recognizer import (
    EMOTION_DEFAULT_ALPHA,
    SENSEVOICE_EMOTION_MAP,
    InputEmotionTracker,
    SenseVoiceRecognizer,
    _parse_sensevoice_tokens,
    emotion_state_to_prompt,
)


def test_parse_full_tokens():
    text, raw, lang, events = _parse_sensevoice_tokens(
        "<|zh|><|HAPPY|><|Speech|><|woitn|>哈哈今天真是太开心了"
    )
    assert text == "哈哈今天真是太开心了"
    assert raw == "HAPPY"
    assert lang == "zh"
    assert events == ["Speech"]


def test_parse_no_tokens():
    text, raw, lang, events = _parse_sensevoice_tokens("普通文本。")
    assert text == "普通文本。"
    assert raw == "" and lang == "" and events == []


def test_parse_en_and_laughter():
    text, raw, lang, events = _parse_sensevoice_tokens("<|en|><|NEUTRAL|><|Laughter|>Hello!")
    assert text == "Hello!"
    assert lang == "en"
    assert events == ["Laughter"]


def test_emotion_map_covers_sensevoice_labels():
    # SenseVoice 官方 7 类情绪都能映射到统一 8 维
    for sv_label, unified in SENSEVOICE_EMOTION_MAP.items():
        anchor = emotion_anchor(unified)
        assert len(anchor.vector) == 8
        assert unified in EMOTION_DEFAULT_ALPHA


def test_tracker_smooths_and_prompt():
    t = InputEmotionTracker()
    rec = type("R", (), {})()
    from avatar.pipeline.emotion_recognizer import EmotionRecognition

    r1 = EmotionRecognition(text="哈哈太开心了", emotion=emotion_anchor("happy", 0.9), raw_label="HAPPY")
    cur = t.push(r1)
    assert cur.label == "happy"
    assert t.last_text == "哈哈太开心了"
    p = emotion_state_to_prompt(cur, t.last_text)
    assert "高兴" in p and "哈哈太开心了" in p
    assert emotion_state_to_prompt(None, "") == ""


def test_recognizer_instantiable_without_torch():
    """无 torch 环境（如 CI）也能 import 模块；懒加载不炸。"""
    r = SenseVoiceRecognizer(model_dir="/nonexistent")
    assert r._model is None  # 未加载
    assert r.device == "cpu"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))