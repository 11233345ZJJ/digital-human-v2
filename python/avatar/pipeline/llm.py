"""LLM 适配器：流式语义句 + 行为指令双通道（全 Mock，真实引擎替换点）。

MockLLM 内置确定性情绪轨迹脚本（calm → happy → surprised → calm），
产出与 V2.0 统一情感中间表示对齐的 SentenceChunk。
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import AsyncIterator

from avatar.pipeline.emotion import Emotion, emotion_anchor
from avatar.pipeline.chunker import StreamingChunker

# Mock 对话脚本：每句 = (文本, 情绪标签, 强度, 意图, 风格, 诱因)
MOCK_SCRIPT: list[tuple[str, str, float, str, str, str]] = [
    ("你好呀，很高兴见到你！", "happy", 0.8, "打招呼", "亲切问候", "用户主动开启对话"),
    ("今天有什么想聊的吗？", "calm", 0.5, "询问用户需求", "自然引导", "对话开场白"),
    ("哇，这个想法真的太棒了！", "surprised", 0.9, "对用户想法表示赞叹", "兴奋推荐", "用户提出新颖方案"),
    ("嗯，让我想想，这里可能会有点问题。", "sad", 0.5, "表达谨慎态度", "温柔安抚", "发现方案存在风险"),
    ("不过没关系，我们可以一起找到解决办法的！", "happy", 0.7, "鼓励用户", "鼓舞人心", "希望消除用户顾虑"),
    ("好的，那我们就这样愉快地决定了。", "calm", 0.6, "确认结论", "从容收尾", "对话自然结束"),
]


@dataclass
class SentenceChunk:
    text: str
    emotion: Emotion
    semantic_meta: dict
    seq: int = 0


@dataclass
class LLMAdapter:
    """真实 LLM 替换点：实现 stream_turn 按句输出 SentenceChunk 即可。"""

    name: str = "mock"

    async def stream_turn(self, user_text: str, ctx: dict) -> AsyncIterator[SentenceChunk]:
        raise NotImplementedError


@dataclass
class MockLLM(LLMAdapter):
    delay_per_token_ms: float = 18.0
    tokens_per_sentence: float = 9.0

    async def stream_turn(self, user_text: str, ctx: dict) -> AsyncIterator[SentenceChunk]:
        chunker = StreamingChunker()
        seq = 0
        for text, label, alpha, intent, style, cause in MOCK_SCRIPT:
            await asyncio.sleep(self.delay_per_token_ms / 1000 * self.tokens_per_sentence)
            yield SentenceChunk(
                text=text,
                emotion=emotion_anchor(label, alpha),
                semantic_meta={"intent": intent, "style": style, "cause": cause},
                seq=seq,
            )
            seq += 1