"""智谱 LLM 适配器：流式文本 + 行为指令双通道（JSONL 流式输出）。

设计（V2.0"大 LLM 决策"模式）：
- system prompt 要求模型按 JSON Lines 输出，每行一句：
  {"t": "文本", "e": "情绪标签", "i": 强度, "intent": "意图", "style": "风格", "cause": "诱因"}
- 流式 delta 累积 → 按行切分 → 完整行解析成功即发射（首句即推）
- 兜底：模型输出非法时退化为纯文本流式切句 + 规则情绪估计
"""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import AsyncIterator

import httpx

from avatar.env import load_dotenv
from avatar.pipeline.chunker import StreamingChunker
from avatar.pipeline.emotion import Emotion, emotion_anchor, estimate_emotion, normalize_label, vector_to_vad
from avatar.pipeline.llm import LLMAdapter, SentenceChunk

load_dotenv()

ZHIPU_BASE = os.environ.get("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/chat/completions")
ZHIPU_MODEL = os.environ.get("ZHIPU_MODEL", "glm-4-flash")

SYSTEM_PROMPT = """你是数字人对话引擎，负责生成带行为标注的回复。你必须严格按 JSON Lines 输出，
每一行是一个完整的 JSON 对象（不要输出任何其他内容、不要用 Markdown），字段：
- "t": 一句回复文本（短句，5-30 字，口语化）
- "e": 情绪标签，只能取 happy/angry/sad/afraid/disgusted/melancholic/surprised/calm 之一
- "i": 情绪强度，0-1 小数
- "intent": 表演意图（这句为什么这么说，如"回应用户称赞"）
- "style": 表演风格（如"亲切问候"/"温柔安抚"/"兴奋推荐"/"鼓舞人心"）
- "cause": 情绪诱因（简短）
示例：
{"t":"你好呀，很高兴见到你！","e":"happy","i":0.8,"intent":"打招呼","style":"亲切问候","cause":"用户主动开启对话"}
{"t":"好的，我们就这样决定。","e":"calm","i":0.5,"intent":"确认结论","style":"从容收尾","cause":"对话自然结束"}
请输出 2-4 行（即 2-4 句），按顺序自然展开。"""


@dataclass
class ZhipuLLM(LLMAdapter):
    """智谱 OpenAI 兼容流式接入。"""

    name: str = "zhipu"
    api_key: str = field(default_factory=lambda: os.environ.get("ZHIPU_API_KEY", ""))
    model: str = ZHIPU_MODEL
    base_url: str = ZHIPU_BASE
    timeout_s: float = 60.0
    max_history: int = 6

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("缺少 ZHIPU_API_KEY（.env 或环境变量）")

    def _history_messages(self, user_text: str, ctx: dict) -> list[dict]:
        system = SYSTEM_PROMPT
        if ctx.get("user_emotion"):
            system += "\n\n" + ctx["user_emotion"]
        msgs: list[dict] = [{"role": "system", "content": system}]
        for turn in ctx.get("history", [])[-self.max_history:]:
            msgs.append({"role": "user", "content": turn.get("user", "")})
            if turn.get("assistant"):
                msgs.append({"role": "assistant", "content": turn.get("assistant")})
        msgs.append({"role": "user", "content": user_text})
        return msgs

    async def stream_turn(self, user_text: str, ctx: dict) -> AsyncIterator[SentenceChunk]:
        payload = {
            "model": self.model,
            "messages": self._history_messages(user_text, ctx),
            "stream": True,
            "max_tokens": 1024,
            "temperature": 0.9,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        seq = 0
        emit = _JsolEmitter()
        tracer = ctx.get("tracer")  # 可选：全链路追踪打点
        first_token = True
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                async with client.stream("POST", self.base_url, json=payload, headers=headers) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            delta = json.loads(data)["choices"][0]["delta"].get("content", "")
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue
                        if delta and first_token:
                            first_token = False
                            if tracer:
                                tracer.mark("llm_first_token", model=self.model)
                        for chunk in emit.feed(delta):
                            yield chunk
                            seq += 1
            for chunk in emit.flush():
                yield chunk
                seq += 1
        except (httpx.HTTPError, httpx.TimeoutException) as e:
            # 网络失败：退化为规则情绪单句回复，保证语音不中断
            yield SentenceChunk(
                text="抱歉，我这边网络有点不稳定，请稍后再试。",
                emotion=estimate_emotion("抱歉"),
                semantic_meta={"intent": "网络异常提示", "style": "温柔安抚", "cause": "LLM 服务不可达"},
                seq=seq,
            )

    async def close(self) -> None:
        pass


@dataclass
class _JsolEmitter:
    """JSONL 流式解析器：delta 累积 → 按行切 → 完整行发射 SentenceChunk。

    行不完整/非法时缓存继续等待；flush 时剩余文本走纯文本切句 + 规则情绪。
    """

    buffer: str = ""
    chunker: StreamingChunker = field(default_factory=StreamingChunker)

    def feed(self, delta: str) -> list[SentenceChunk]:
        self.buffer += delta
        lines = self.buffer.split("\n")
        self.buffer = lines.pop()  # 最后一段可能不完整
        out: list[SentenceChunk] = []
        for line in lines:
            line = line.strip().rstrip(",")
            if not line:
                continue
            obj = _parse_line(line)
            if obj is None:
                # 非法行：按纯文本切句兜底
                out.extend(self._text_fallback(line))
            else:
                out.append(_obj_to_chunk(obj))
        return out

    def flush(self) -> list[SentenceChunk]:
        rest = self.buffer.strip()
        self.buffer = ""
        if not rest:
            return []
        obj = _parse_line(rest)
        if obj is not None:
            return [_obj_to_chunk(obj)]
        return self._text_fallback(rest)

    def _text_fallback(self, text: str) -> list[SentenceChunk]:
        out: list[SentenceChunk] = []
        for s in self.chunker.feed(text):
            out.append(_sentence_to_chunk(s))
        for s in self.chunker.flush():
            out.append(_sentence_to_chunk(s))
        return out


def _parse_line(line: str) -> dict | None:
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or not obj.get("t"):
        return None
    return obj


def _obj_to_chunk(obj: dict) -> SentenceChunk:
    text = str(obj["t"]).strip()
    label = normalize_label(str(obj.get("e", "calm")))
    try:
        alpha = max(0.0, min(1.0, float(obj.get("i", 0.5))))
    except (TypeError, ValueError):
        alpha = 0.5
    vec = obj.get("v")
    if isinstance(vec, (list, tuple)) and len(vec) == 8 and all(isinstance(x, (int, float)) for x in vec):
        vector = tuple(max(0.0, min(1.0, float(x))) for x in vec)
        emotion = Emotion(label=label, vector=vector, intensity=alpha, **vector_to_vad(vector))
    else:
        emotion = emotion_anchor(label, alpha)
    return SentenceChunk(
        text=text or "嗯嗯。",
        emotion=emotion,
        semantic_meta={
            "intent": str(obj.get("intent", "") or ""),
            "style": str(obj.get("style", "") or ""),
            "cause": str(obj.get("cause", "") or ""),
        },
    )


def _sentence_to_chunk(text: str) -> SentenceChunk:
    return SentenceChunk(
        text=text,
        emotion=estimate_emotion(text),
        semantic_meta={"intent": "", "style": "自然引导", "cause": "规则兜底"},
    )


async def _noop() -> None:
    """占位：确保本模块可独立 import 不触发网络。"""
    await asyncio.sleep(0)