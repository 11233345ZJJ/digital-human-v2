"""TTS 适配器：流式情感语音合成 + 时间线先推。

V2.0 需求 2.2：句间情绪变化平滑过渡（情绪向量 slerp）；
音频分块携带时间戳与情感标签；首块即带整句时长（时间线先推，
音频流式跟推）——保证渲染端口型帧不迟到。

Mock 实现：静音 PCM 块 + 时间戳，真实引擎（IndexTTS-2.5 / Qwen3-TTS /
VoXtream2 / sherpa-onnx）实现 TTSAdapter 接口即可替换。
"""
from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import AsyncIterator

from avatar.pipeline.emotion import Emotion, slerp_vec
from avatar.pipeline.llm import SentenceChunk

SAMPLE_RATE = 16000
CHUNK_MS = 50
BYTES_PER_CHUNK = SAMPLE_RATE * CHUNK_MS // 1000 * 2  # 16-bit mono

# 语速→每字时长估计（毫秒/字符，中文约 4-5 字/秒）
MS_PER_CHAR = 210.0


@dataclass
class AudioChunk:
    timestamp_ms: int
    duration_ms: int
    pcm_s16le: bytes
    emotion: Emotion
    phoneme: str = ""
    sentence_done: bool = False


@dataclass
class TTSAdapter:
    name: str = "mock"

    async def synthesize(self, chunk: SentenceChunk, ctx: dict, start_ms: int) -> AsyncIterator[AudioChunk]:
        raise NotImplementedError


@dataclass
class MockTTS(TTSAdapter):
    async def synthesize(self, chunk: SentenceChunk, ctx: dict, start_ms: int) -> AsyncIterator[AudioChunk]:
        n_chars = max(1, len(chunk.text))
        total_ms = int(n_chars * MS_PER_CHAR)
        n_chunks = max(1, math.ceil(total_ms / CHUNK_MS))
        sil = b"\x00\x00" * (BYTES_PER_CHUNK // 2)
        for i in range(n_chunks):
            await asyncio.sleep(0.001)
            ts = start_ms + i * CHUNK_MS
            yield AudioChunk(
                timestamp_ms=ts,
                duration_ms=CHUNK_MS,
                pcm_s16le=sil,
                emotion=chunk.emotion,
                sentence_done=(i == n_chunks - 1),
            )


@dataclass
class TTSTimeline:
    """句级音频时间线：决定驱动帧的时间戳锚点。"""

    clock_ms: int = 0  # 当前音频时钟（TTS 输出时钟 = 唯一时间戳源）

    def reserve(self, n_chars: int) -> tuple[int, int]:
        """为一句预留时间槽，返回 (start_ms, end_ms)。"""
        total_ms = int(max(1, n_chars) * MS_PER_CHAR)
        start = self.clock_ms
        self.clock_ms += total_ms
        return start, self.clock_ms


@dataclass
class EmotionInterpolator:
    """句间情绪过渡：句内按帧进度做情绪向量 slerp + 强度插值。"""

    prev: Emotion | None = None

    def at(self, cur: Emotion, progress: float) -> Emotion:
        progress = max(0.0, min(1.0, progress))
        if self.prev is None or progress >= 1.0:
            self.prev = cur
            return cur
        vec = slerp_vec(self.prev.vector, cur.vector, progress)
        intensity = self.prev.intensity + (cur.intensity - self.prev.intensity) * progress
        label = cur.label if progress > 0.5 else self.prev.label
        return Emotion(label=label, vector=vec, intensity=intensity)

    def commit(self, cur: Emotion) -> None:
        self.prev = cur