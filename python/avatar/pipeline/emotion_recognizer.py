"""情感识别服务：SenseVoice（ASR + 情绪 + 音频事件一体）→ 统一 8 维情绪中间表示。

V2.0 需求 2.1/4.1：
- 主模型 SenseVoice（FunASR 系），输出文本 + 情绪标签 + 强度
- 识别结果映射到统一 8 维情绪向量（IndexTTS-2.5 标准），直接可喂 TTS/渲染层
- 滑动窗口 EWMA 平滑（复用 EmotionSmoother）
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from avatar.pipeline.emotion import EMOTION_DIMS, Emotion, EmotionSmoother, emotion_anchor, vector_to_vad

# SenseVoice 官方情绪标签 → 统一标签
SENSEVOICE_EMOTION_MAP: dict[str, str] = {
    "HAPPY": "happy",
    "SAD": "sad",
    "ANGRY": "angry",
    "NEUTRAL": "calm",
    "FEARFUL": "afraid",
    "DISGUSTED": "disgusted",
    "SURPRISED": "surprised",
}
# 强度标定：无明确置信度输出时按标签取默认强度
EMOTION_DEFAULT_ALPHA: dict[str, float] = {
    "happy": 0.7, "sad": 0.6, "angry": 0.8, "calm": 0.2,
    "afraid": 0.7, "disgusted": 0.6, "surprised": 0.8, "melancholic": 0.5,
}


@dataclass
class EmotionRecognition:
    """一次语音情绪识别的结果（文本 + 情绪 + 中间表示）。"""

    text: str
    emotion: Emotion
    raw_label: str = ""          # SenseVoice 原始标签（如 HAPPY）
    audio_events: list[str] = field(default_factory=list)  # 音频事件（如 music/speech/bgm）
    language: str = ""           # LID 结果
    duration_ms: int = 0


class EmotionRecognizer(Protocol):
    """情感识别器接口：输入音频 → 识别结果。真实引擎替换点。"""

    def recognize(self, audio: bytes, sample_rate: int = 16000) -> EmotionRecognition: ...


@dataclass
class SenseVoiceRecognizer:
    """SenseVoice（FunASR）后端：ASR + LID + 情绪 + 音频事件一体。

    model_dir: 模型目录（含 model.pt / config.yaml / tokens.json），缺省自动从
    modelscope 下载 iic/SenseVoiceSmall。可指定 device（cpu 默认）。
    """

    model_dir: str | None = None
    device: str = "cpu"
    _model: object | None = field(default=None, repr=False)

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        from funasr import AutoModel  # 延迟导入：无 torch 环境不阻塞其他功能

        # 低内存多核机器：限制 torch 线程数（默认 = 核数 24，与 TTS 并发时
        # 造成线程超订 + 内存抖动，识别延迟可从 ~2s 恶化到 ~50s）
        try:
            import torch

            torch.set_num_threads(min(4, os.cpu_count() or 4))
        except Exception:
            pass

        model_dir = self.model_dir
        if not model_dir:
            root = Path(__file__).resolve().parents[3] / "models" / "sensevoice"
            if (root / "model.pt").exists():
                model_dir = str(root)
        kwargs = {}
        if model_dir:
            kwargs["model"] = model_dir
        else:
            kwargs["model"] = "iic/SenseVoiceSmall"
            kwargs["trust_remote_code"] = True
        self._model = AutoModel(model=kwargs.get("model"),
                                trust_remote_code=kwargs.get("trust_remote_code", False),
                                device=self.device,
                                disable_update=True,
                                log_level="ERROR")
        return self._model

    def recognize(self, audio: bytes, sample_rate: int = 16000) -> EmotionRecognition:
        import numpy as np

        model = self._ensure_model()
        wav = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        res = model.generate(
            input=wav,
            cache={},
            language="auto",
            use_itn=False,
            batch_size_s=60,
            merge_vad=True,
            merge_length_s=15,
        )
        out = res[0] if isinstance(res, (list, tuple)) else res
        full = str(out.get("text", "") or "")
        text, raw_label, lang, events = _parse_sensevoice_tokens(full)
        label = SENSEVOICE_EMOTION_MAP.get(raw_label.upper(), "calm")
        alpha = EMOTION_DEFAULT_ALPHA.get(label, 0.3)
        anchor = emotion_anchor(label, alpha)
        emotion = Emotion(label=label, vector=anchor.vector, intensity=anchor.intensity,
                          **vector_to_vad(anchor.vector))
        return EmotionRecognition(
            text=text,
            emotion=emotion,
            raw_label=raw_label,
            audio_events=events,
            language=lang,
            duration_ms=int(len(wav) * 1000 // sample_rate),
        )


def _parse_sensevoice_tokens(full: str) -> tuple[str, str, str, list[str]]:
    """解析 SenseVoice 输出中的 <|...|> 特殊 token（funasr 1.4.x 拼在文本里）。

    返回 (纯净文本, 情绪标签, 语言, 音频事件列表)。
    """
    import re

    lang = ""
    raw_emotion = ""
    events: list[str] = []
    text_parts: list[str] = []
    for tok in re.split(r"(<\|[^|]+\|>)", full):
        if tok.startswith("<|") and tok.endswith("|>"):
            name = tok[2:-2].strip()
            if name in SENSEVOICE_EMOTION_MAP:
                raw_emotion = name
            elif re.fullmatch(r"zh|en|ja|ko|yue|pt|fr|es|ru|ar|de|it|nl|pl|uk|tr|id|vi|th|ms|jap|yue", name, re.I):
                lang = name.lower()
            elif name.lower() in ("speech", "music", "bgm", "applause", "laughter", "cry", "breath", "cough"):
                events.append(name)
        else:
            text_parts.append(tok)
    return "".join(text_parts).strip(), raw_emotion, lang, events


@dataclass
class InputEmotionTracker:
    """输入侧情绪跟踪：用户语音情绪 → 平滑 → 当前用户情绪状态。

    数字人可据此做共情（镜像用户情绪调整回应风格）。
    """

    smoother: EmotionSmoother = field(default_factory=lambda: EmotionSmoother(window_size=5))
    current: Emotion | None = None
    last_text: str = ""

    def push(self, rec: EmotionRecognition) -> Emotion:
        self.current = self.smoother.push(rec.emotion)
        self.last_text = rec.text
        return self.current

    def reset(self) -> None:
        self.smoother.reset()
        self.current = None
        self.last_text = ""


def emotion_state_to_prompt(emotion: Emotion | None, text: str) -> str:
    """把用户情绪状态转成 LLM 上下文提示（共情路径）。"""
    if emotion is None:
        return ""
    cn = {"happy": "高兴", "angry": "愤怒", "sad": "悲伤", "afraid": "恐惧",
          "disgusted": "厌恶", "melancholic": "忧郁", "surprised": "惊讶", "calm": "平静"}
    return (f"（感知：用户情绪 {cn.get(emotion.label, emotion.label)}，"
            f"强度 {emotion.intensity:.1f}；语音转写：{text or '无'}）")