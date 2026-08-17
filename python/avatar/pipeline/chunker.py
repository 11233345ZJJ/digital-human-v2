"""流式切句器：LLM token 流 → 语义句流（首句即推）。

大 LLM 决策 + 小模型编排模式中的"小模型"角色：轻量、无状态、低延迟。
中文以句末标点为主切点，英文以 .!? 为主；超长 token 累积强制切分，
保证首句延迟不因长句而无限放大。
"""
from __future__ import annotations

from typing import AsyncIterator

_CN_END = "。！？!?…\n；;"
_EN_END = ".!?"
# 保留在句内的标点（避免把 "3.14" / "U.S.A" 切断）
_KEEP = ".%"


class StreamingChunker:
    def __init__(self, max_chars: int = 96, min_chars: int = 4) -> None:
        self.max_chars = max_chars
        self.min_chars = min_chars
        self._buf: list[str] = []

    def feed(self, text: str) -> list[str]:
        """喂入 token 片段，返回本次可发射的完整句。"""
        self._buf.append(text)
        joined = "".join(self._buf)
        sentences: list[str] = []
        start = 0
        i = 0
        n = len(joined)
        while i < n:
            ch = joined[i]
            is_end = ch in _CN_END or (ch in _EN_END and _is_en_boundary(joined, i))
            if is_end or (i - start >= self.max_chars):
                piece = joined[start : i + 1].strip()
                if len(piece) >= self.min_chars:
                    sentences.append(piece)
                start = i + 1
            i += 1
        tail = joined[start:]
        self._buf = [tail] if tail else []
        return sentences

    def flush(self) -> list[str]:
        tail = "".join(self._buf).strip()
        self._buf = []
        if len(tail) >= self.min_chars:
            return [tail]
        return []

    async def stream(self, tokens: AsyncIterator[str]) -> AsyncIterator[str]:
        """对 token 流逐片切句（首句即推）。"""
        async for tok in tokens:
            for s in self.feed(tok):
                yield s
        for s in self.flush():
            yield s


def _is_en_boundary(joined: str, i: int) -> bool:
    """英文句号需后跟空白/结尾/句末标点才视为切点（避免缩写误切）。"""
    if joined[i] != ".":
        return True
    nxt = joined[i + 1 : i + 2]
    return nxt == "" or nxt in " \t\n\"'”’)]}"


def is_ending(text: str) -> bool:
    return text[-1:] in "。！？!?…" or text.endswith((".", "!", "?"))