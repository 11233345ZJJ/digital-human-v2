"""驱动帧生成：语义句 + 时间线 → 时间戳有序 DriveCommand 流。

V2.0 需求 2.3/2.5：
- 情绪向量直接喂渲染器（统一中间表示）
- 表情/头部姿态/肢体动作生成（规则 + 检索混合，Mock 为规则版）
- 音画同步：所有帧共享 TTS 输出时钟
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from avatar.pipeline.emotion import (
    EMOTION_TO_EXPRESSION,
    Emotion,
    vector_to_vad,
)
from avatar.pipeline.llm import SentenceChunk
from avatar.pipeline.tts import EmotionInterpolator, TTSTimeline

FRAME_MS = 50  # 20 fps 驱动帧

# 表演风格 → 肢体动作（动作库检索键，两端映射）
STYLE_TO_GESTURE: dict[str, str] = {
    "亲切问候": "wave",
    "自然引导": "nod",
    "兴奋推荐": "cheer",
    "温柔安抚": "soothe",
    "鼓舞人心": "encourage",
    "从容收尾": "bow",
    "打招呼": "wave",
    "询问用户需求": "lean_in",
    "对用户想法表示赞叹": "applaud",
    "表达谨慎态度": "think",
    "鼓励用户": "fist_pump",
    "确认结论": "nod",
}
DEFAULT_GESTURE = "idle"

# 中文元音/辅音 → 视素近似（真实引擎应接入音素级时间线）
VISEME_BY_CHAR = {
    "a": "AA", "e": "EH", "i": "IH", "o": "OW", "u": "UW",
    "阿": "AA", "啊": "AA", "诶": "EH", "哦": "OW", "唔": "UW",
}
VISEME_SET = ["AA", "EH", "IH", "OW", "UW", "B", "D", "F", "K", "L", "M", "N", "P", "S", "SH", "T", "W", "Y"]


@dataclass
class DrivingStage:
    """句流 → DriveCommand 流（含情绪平滑过渡与语义元数据透传）。"""

    timeline: TTSTimeline = field(default_factory=TTSTimeline)
    interp: EmotionInterpolator = field(default_factory=EmotionInterpolator)
    frame_seq: int = 0
    head_base_yaw: float = 0.0

    def _viseme_for(self, ch: str, i: int) -> str:
        v = VISEME_BY_CHAR.get(ch.lower())
        if v:
            return v
        if ch.isalpha() or "\u4e00" <= ch <= "\u9fff":
            return VISEME_SET[i % len(VISEME_SET)]
        return "SIL"

    def _gesture_for(self, style: str) -> dict:
        name = STYLE_TO_GESTURE.get(style, DEFAULT_GESTURE)
        return {"name": name, "params": {"speed": 1.0, "amplitude": 0.8}}

    def _head_pose(self, t_ms: int, emotion: Emotion) -> dict:
        slow = math.sin(t_ms / 1800.0 + self.head_base_yaw)
        return {
            "yaw": round(0.08 * slow + emotion.arousal * 0.05, 4),
            "pitch": round(0.04 * math.cos(t_ms / 2400.0) - emotion.valence * 0.06, 4),
            "roll": round(0.02 * math.sin(t_ms / 3200.0), 4),
        }

    def frames_for(self, chunk: SentenceChunk, start_ms: int, end_ms: int) -> list[dict]:
        """一句 → 逐帧 DriveCommand（dict 形式，proto/json 双通道通用）。"""
        frames: list[dict] = []
        n = max(1, (end_ms - start_ms) // FRAME_MS)
        for i in range(n):
            t = i / n
            emo = self.interp.at(chunk.emotion, t)
            ts = start_ms + i * FRAME_MS
            expr_name = EMOTION_TO_EXPRESSION[emo.label]
            frames.append(
                {
                    "timestamp_ms": ts,
                    "phoneme": self._viseme_for(chunk.text[i % len(chunk.text)] if chunk.text else " ", i),
                    "expression": {"name": expr_name, "weight": round(emo.intensity, 4)},
                    "head_pose": self._head_pose(ts, emo),
                    "body_gesture": self._gesture_for(chunk.semantic_meta.get("style", "")),
                    "emotion": emo.to_dict(),
                    "semantic_meta": chunk.semantic_meta,
                    "frame_seq": self.frame_seq,
                }
            )
            self.frame_seq += 1
        self.interp.commit(chunk.emotion)
        return frames

    def reset(self) -> None:
        self.timeline.clock_ms = 0
        self.interp.prev = None
        self.frame_seq = 0