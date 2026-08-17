"""统一情感中间表示：8 维情绪向量 + emo_alpha 强度 + VAD 辅助空间。

IndexTTS-2.5 标准 8 维顺序：happy/angry/sad/afraid/disgusted/melancholic/surprised/calm。
本模块同时供 TTS 与渲染层使用——两端共用同一中间表示，是降低级联误差的核心。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

EMOTION_DIMS: tuple[str, ...] = (
    "happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm",
)
EMOTION_INDEX = {name: i for i, name in enumerate(EMOTION_DIMS)}


@dataclass(frozen=True)
class Emotion:
    """8 维情绪向量 + 强度标量 + VAD 辅助（跨模型对齐用）。"""

    label: str = "calm"
    vector: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    intensity: float = 0.2
    valence: float = 0.1
    arousal: float = -0.2
    dominance: float = 0.0

    def __post_init__(self) -> None:
        assert len(self.vector) == 8, "8 维情绪向量"
        if self.label not in EMOTION_INDEX:
            raise ValueError(f"未知情绪标签: {self.label}")
        for v in self.vector:
            assert 0.0 <= v <= 1.0, "向量分量须在 [0,1]"

    @property
    def dominant_index(self) -> int:
        return max(range(8), key=lambda i: self.vector[i])

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "vector": list(self.vector),
            "intensity": round(self.intensity, 4),
            "valence": round(self.valence, 4),
            "arousal": round(self.arousal, 4),
            "dominance": round(self.dominance, 4),
        }


# 标准情绪锚点（向量 + 默认强度），供 Mock 与映射表使用
EMOTION_ANCHORS: dict[str, tuple[tuple[float, ...], float]] = {
    "happy": ((0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2), 0.8),
    "angry": ((0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1), 0.9),
    "sad": ((0.0, 0.0, 0.7, 0.0, 0.0, 0.2, 0.0, 0.1), 0.7),
    "afraid": ((0.0, 0.0, 0.0, 0.7, 0.0, 0.0, 0.3, 0.1), 0.7),
    "disgusted": ((0.0, 0.2, 0.0, 0.0, 0.6, 0.0, 0.0, 0.2), 0.5),
    "melancholic": ((0.0, 0.0, 0.3, 0.0, 0.0, 0.7, 0.0, 0.1), 0.6),
    "surprised": ((0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.1), 0.8),
    "calm": ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), 0.2),
}

# 渲染层表情名映射（两端映射表，Live2D/3D 各自填自己的参数名）
EMOTION_TO_EXPRESSION: dict[str, str] = {
    "happy": "smile",
    "angry": "anger",
    "sad": "sad",
    "afraid": "fear",
    "disgusted": "disgust",
    "melancholic": "melancholy",
    "surprised": "surprise",
    "calm": "neutral",
}


def emotion_anchor(label: str, intensity: float | None = None) -> Emotion:
    vec, default = EMOTION_ANCHORS[label]
    alpha = intensity if intensity is not None else default
    return Emotion(label=label, vector=vec, intensity=alpha, **vector_to_vad(vec))


def vector_to_vad(vector: tuple[float, ...]) -> dict[str, float]:
    """8 维向量 → VAD 三维空间（启发式线性映射，供跨模型对齐）。"""
    h, a, s, f, d, m, su, c = vector
    valence = (h + 0.5 * su + 0.2 * c) - (s + a + f + d + m)
    arousal = (h + a + f + su) - (s + m + 0.5 * c)
    dominance = (a + h + c) - (f + s + su + d)
    norm = max(abs(valence), abs(arousal), abs(dominance), 1e-6)
    scale = min(1.0, norm)  # 软钳制
    return {
        "valence": _clamp(valence / max(norm, 1e-6) * scale),
        "arousal": _clamp(arousal / max(norm, 1e-6) * scale),
        "dominance": _clamp(dominance / max(norm, 1e-6) * scale),
    }


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def lerp_vec(a: tuple[float, ...], b: tuple[float, ...], t: float) -> tuple[float, ...]:
    return tuple(ai + (bi - ai) * t for ai, bi in zip(a, b))


def slerp_vec(a: tuple[float, ...], b: tuple[float, ...], t: float) -> tuple[float, ...]:
    """情绪向量球面插值（slerp）——句间情绪变化时平滑过渡。"""
    dot = sum(x * y for x, y in zip(a, b))
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        return lerp_vec(a, b, t)
    theta = math.acos(dot)
    sin_t = math.sin(theta)
    k0 = math.sin((1 - t) * theta) / sin_t
    k1 = math.sin(t * theta) / sin_t
    return tuple(k0 * x + k1 * y for x, y in zip(a, b))


@dataclass
class EmotionSmoother:
    """滑动窗口 + 指数加权平滑 + 趋势预测，避免单句误判与情绪跳变。

    - 窗口大小 5–10 句（V2.0 规范）
    - EWMA 平滑（alpha 参数控制跟随速度）
    - 趋势预测：后 1/3 窗口均值 vs 前 2/3 窗口均值，向趋势方向轻微外推
    """

    window_size: int = 7
    ewma_alpha: float = 0.55
    trend_boost: float = 0.15
    _history: list[Emotion] = field(default_factory=list)
    _smoothed: Emotion | None = None

    def push(self, e: Emotion) -> Emotion:
        self._history.append(e)
        if len(self._history) > self.window_size:
            self._history.pop(0)
        raw = self._window_mean()
        if self._smoothed is None:
            self._smoothed = raw
        else:
            self._smoothed = self._ewma(self._smoothed, raw)
        return self._trend_correct(self._smoothed)

    def reset(self) -> None:
        self._history.clear()
        self._smoothed = None

    def _window_mean(self) -> Emotion:
        n = len(self._history)
        vec = tuple(sum(e.vector[i] for e in self._history) / n for i in range(8))
        intensity = sum(e.intensity for e in self._history) / n
        labels = [e.label for e in self._history]
        label = max(EMOTION_DIMS, key=lambda d: sum(1 for l in labels if l == d)) if labels else "calm"
        return Emotion(label=label, vector=vec, intensity=intensity, **vector_to_vad(vec))

    def _ewma(self, prev: Emotion, raw: Emotion) -> Emotion:
        vec = lerp_vec(prev.vector, raw.vector, self.ewma_alpha)
        intensity = prev.intensity + (raw.intensity - prev.intensity) * self.ewma_alpha
        label = EMOTION_DIMS[max(range(8), key=lambda i: vec[i])]
        return Emotion(label=label, vector=vec, intensity=intensity, **vector_to_vad(vec))

    def _trend_correct(self, cur: Emotion) -> Emotion:
        """趋势预测：前 2/3 vs 后 1/3 窗口均值差异 → 向趋势方向外推。"""
        n = len(self._history)
        if n < 4:
            return cur
        split = max(1, n * 2 // 3)
        early = _vec_mean([e.vector for e in self._history[:split]])
        late = _vec_mean([e.vector for e in self._history[split:]])
        delta = tuple((late[i] - early[i]) * self.trend_boost for i in range(8))
        vec = tuple(max(0.0, min(1.0, cur.vector[i] + delta[i])) for i in range(8))
        label = EMOTION_DIMS[max(range(8), key=lambda i: vec[i])]
        return Emotion(label=label, vector=vec, intensity=cur.intensity, **vector_to_vad(vec))


def _vec_mean(vecs: list[tuple[float, ...]]) -> tuple[float, ...]:
    n = len(vecs)
    return tuple(sum(v[i] for v in vecs) / n for i in range(8))


# 中文情绪词 → 标签映射（模型标注兜底用）
CN_EMOTION_WORDS: dict[str, tuple[str, ...]] = {
    "happy": ("高兴", "开心", "快乐", "太棒", "太好了", "喜欢", "爱", "哈哈", "不错"),
    "angry": ("生气", "愤怒", "气死", "气人", "讨厌", "烦", "恼火"),
    "sad": ("难过", "伤心", "悲伤", "哭", "遗憾", "抱歉", "对不起", "可惜"),
    "afraid": ("害怕", "担心", "怕", "危险", "吓", "紧张"),
    "disgusted": ("恶心", "厌恶", "差劲", "糟糕"),
    "melancholic": ("忧郁", "伤感", "思念", "怀念", "回忆", "落寞"),
    "surprised": ("哇", "惊讶", "吃惊", "居然", "竟然", "没想到", "真的吗"),
}

# 关键词 → 情绪标签（纯规则 fallback，LLM/标注缺失时使用）
RULE_EMOTION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "happy": ("哈哈", "开心", "高兴", "太好了", "太棒", "棒极了", "喜欢", "爱你", "真棒"),
    "sad": ("难过", "伤心", "悲伤", "哭", "遗憾", "对不起", "抱歉", "可惜", "呜呜"),
    "angry": ("生气", "愤怒", "气死", "讨厌", "烦死", "恼火", "混蛋"),
    "afraid": ("害怕", "担心", "怕", "危险", "吓人", "紧张", "慌"),
    "disgusted": ("恶心", "厌恶", "差劲", "糟糕", "烂"),
    "melancholic": ("思念", "怀念", "回忆", "忧郁", "伤感", "落寞", "想家"),
    "surprised": ("哇", "惊讶", "居然", "竟然", "没想到", "真的吗", "天哪"),
}


def estimate_emotion(text: str, fallback: str = "calm") -> Emotion:
    """纯规则情绪估计（无模型标注时的兜底路径）。"""
    label = fallback
    for cand, words in RULE_EMOTION_KEYWORDS.items():
        if any(w in text for w in words):
            label = cand
            break
    e = emotion_anchor(label)
    if label != "calm":
        e = Emotion(label=label, vector=e.vector, intensity=max(e.intensity, 0.5), **vector_to_vad(e.vector))
    return e


def normalize_label(raw: str) -> str:
    """模型标注 → 合法标签（含中文情绪词映射）。"""
    if not raw:
        return "calm"
    s = raw.strip().lower()
    if s in EMOTION_INDEX:
        return s
    for label, words in CN_EMOTION_WORDS.items():
        if any(w in raw for w in words):
            return label
    return "calm"