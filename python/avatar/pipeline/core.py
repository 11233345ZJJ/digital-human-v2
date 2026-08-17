"""管线核心：时间戳有序队列 + 打断复位 + 阶段组合。

V2.0 需求 2.6：驱动指令按时间戳有序队列传输，渲染器按时间戳逐帧消费；
动态缓冲防饥饿；打断时清空未消费队列、保留已渲染帧、平滑过渡。
"""
from __future__ import annotations

import asyncio
import heapq
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Generic, TypeVar

T = TypeVar("T")


@dataclass
class Timed:
    timestamp_ms: int


@dataclass
class TimelineQueue(Generic[T]):
    """按时间戳有序的消费队列。

    - push: 可乱序插入（heap 保证按时间戳消费）
    - consume(now): 取出 timestamp <= now 的项
    - barge_in: 清空未消费项（保留已消费的已渲染帧）
    """

    _heap: list[tuple[int, int, T]] = field(default_factory=list)
    _seq: int = 0

    def push(self, item: T) -> None:
        ts = getattr(item, "timestamp_ms", None)
        if ts is None and isinstance(item, dict):
            ts = item.get("timestamp_ms")
        heapq.heappush(self._heap, (int(ts), self._seq, item))
        self._seq += 1

    def consume(self, now_ms: int) -> list[T]:
        out: list[T] = []
        while self._heap and self._heap[0][0] <= now_ms:
            out.append(heapq.heappop(self._heap)[2])
        return out

    def peek(self) -> int | None:
        return self._heap[0][0] if self._heap else None

    def clear(self) -> int:
        n = len(self._heap)
        self._heap.clear()
        return n

    def __len__(self) -> int:
        return len(self._heap)

    @property
    def head_ms(self) -> int | None:
        return self.peek()

    @property
    def tail_ms(self) -> int | None:
        return self._heap[-1][0] if self._heap else None

    @property
    def span_ms(self) -> int:
        """缓冲水位（尾-头），供动态缓冲决策。"""
        if len(self._heap) < 2:
            return 0
        return self.tail_ms - self.head_ms  # type: ignore[operator]


class BargeIn:
    """打断事件：触发后所有阶段立即停止生成并复位。"""

    def __init__(self) -> None:
        self._event = asyncio.Event()

    def trigger(self) -> None:
        self._event.set()

    def clear(self) -> None:
        self._event.clear()

    async def wait(self) -> None:
        await self._event.wait()


@dataclass
class StageResult(Generic[T]):
    """单句处理结果（语义句 → 下游各阶段）。"""

    text: str
    emotion: object | None = None
    semantic_meta: dict | None = None
    payload: T | None = None


Stage = Callable[..., AsyncIterator]


class Pipeline:
    """阶段组合器：把 [llm, emotion, tts, driving] 串成一条异步生成链。

    并行语义：下游阶段各自以 asyncio.Queue 解耦，LLM 首句产出后
    立即触发 TTS 与渲染管线（边生成、边分析、边合成、边渲染）。
    """

    def __init__(self) -> None:
        self._stages: list[tuple[Stage, str]] = []

    def add(self, name: str, stage: Stage) -> "Pipeline":
        self._stages.append((name, stage))
        return self

    async def run(self, ctx: dict) -> AsyncIterator[object]:
        """串行组合执行；每级产出即推送给下一级。"""
        source: AsyncIterator = _once(ctx.get("input_text", ""))

        for name, stage in self._stages:
            source = stage(source, ctx)

        async for item in source:
            yield item


async def _once(v: str) -> AsyncIterator[str]:
    yield v