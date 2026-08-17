"""全链路 Trace ID：从 /chat 到渲染端贯穿每个阶段。

V2.0 需求（可观测性）：每轮对话生成唯一 trace_id，
LLM/切句/情感/TTS/驱动各阶段打点耗时，日志按 trace_id 聚合，
便于定位延迟瓶颈（首响延迟拆解）与级联误差来源。

用法：
    tracer = Tracer.start("user_text")      # start_turn 处
    ctx["tracer"] = tracer                  # 传入管线各阶段
    tracer.mark("llm_first_chunk", seq=0)   # 各阶段打点
"""
from __future__ import annotations

import logging
import time
import uuid

logger = logging.getLogger("avatar.trace")

# 关键阶段里程碑（用于首响延迟拆解报告）
KEY_EVENTS = (
    "turn_start",
    "asr_first_partial",
    "asr_final",
    "llm_first_token",
    "llm_first_chunk",
    "tts_first_audio",
    "drive_first_frame",
    "turn_end",
)


def new_trace_id() -> str:
    """短随机 ID：12 位 hex，单进程内足够区分并发轮次。"""
    return uuid.uuid4().hex[:12]


def setup_logging(level: int = logging.INFO) -> None:
    """服务启动时调用：统一日志格式（含 trace 字段位）。"""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


class Tracer:
    """一轮对话的追踪器：事件打点 + 耗时统计。

    - mark(event): 记录自轮次开始（elapsed_ms）与上次打点（delta_ms）的耗时
    - summary(): 关键里程碑耗时报告（首响延迟拆解）
    """

    def __init__(self, trace_id: str, label: str = "") -> None:
        self.trace_id = trace_id
        self.label = label
        self.t0 = time.perf_counter()
        self.t_last = self.t0
        self.marks: list[tuple[str, float]] = []  # (event, elapsed_ms)

    @classmethod
    def start(cls, label: str = "") -> "Tracer":
        return cls(new_trace_id(), label)

    def mark(self, event: str, **kv) -> float:
        """打点：返回自轮次开始的耗时（毫秒）。"""
        now = time.perf_counter()
        elapsed = (now - self.t0) * 1000.0
        delta = (now - self.t_last) * 1000.0
        self.t_last = now
        self.marks.append((event, elapsed))
        extra = " ".join(f"{k}={v}" for k, v in kv.items())
        logger.info(
            "trace=%s event=%s elapsed_ms=%.1f delta_ms=%.1f %s",
            self.trace_id, event, elapsed, delta, extra,
        )
        return elapsed

    def elapsed_of(self, event: str) -> float | None:
        """某里程碑的累计耗时（毫秒）；未打点返回 None。"""
        for ev, ms in self.marks:
            if ev == event:
                return ms
        return None

    def summary(self) -> dict:
        """关键里程碑耗时报告（首响延迟拆解）。"""
        out: dict = {"trace_id": self.trace_id, "label": self.label}
        for ev in KEY_EVENTS:
            ms = self.elapsed_of(ev)
            if ms is not None:
                out[ev + "_ms"] = round(ms, 1)
        # 分段耗时（相邻关键事件之间的耗时）
        for a, b in (
            ("llm_first_token", "llm_first_chunk"),
            ("llm_first_chunk", "tts_first_audio"),
            ("tts_first_audio", "drive_first_frame"),
        ):
            ta, tb = self.elapsed_of(a), self.elapsed_of(b)
            if ta is not None and tb is not None:
                out[f"{b}__from_{a}_ms"] = round(tb - ta, 1)
        return out
