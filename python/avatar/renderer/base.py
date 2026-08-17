"""统一渲染接口 IRenderer + 注册表 + 动态切换（V2.0 需求 2.4）。

- 后端注册：flashhead（视频神经渲染）→ babylon3d → live2d → audio（纯音频兜底）
- 自动降级链：视频 → 3D → Live2D → 纯音频，每一级降低 GPU/CPU 需求
- 切换插值：参数空间 easeInOutCubic 过渡，避免"跳脸"
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable


@runtime_checkable
class IRenderer(Protocol):
    name: str
    kind: str
    cost_level: int  # 越小越省资源（audio=0 < live2d=1 < 3d=2 < flashhead=3）

    def activate(self) -> None: ...
    def deactivate(self) -> None: ...
    def push(self, frame: dict) -> None: ...


@dataclass
class RendererRegistry:
    """后端注册表：所有后端实现并注册，按名查询/切换。"""

    _backends: dict[str, IRenderer] = field(default_factory=dict)
    _degrade_chain: list[str] = field(default_factory=list)

    def register(self, r: IRenderer) -> None:
        self._backends[r.name] = r

    def set_degrade_chain(self, chain: list[str]) -> None:
        self._degrade_chain = chain

    def get(self, name: str) -> IRenderer:
        return self._backends[name]

    def names(self) -> list[str]:
        return list(self._backends)

    def next_lower(self, name: str) -> str | None:
        """降级链中的下一级（若当前不在链上则返回 None）。"""
        if not self._degrade_chain or name not in self._degrade_chain:
            return None
        i = self._degrade_chain.index(name)
        return self._degrade_chain[i + 1] if i + 1 < len(self._degrade_chain) else None


def ease_in_out_cubic(t: float) -> float:
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        return 4 * t * t * t
    return 1 - (-2 * t + 2) ** 3 / 2


@dataclass
class SwitchInterpolator:
    """切换过渡代理：在过渡窗口内对 expression/head_pose 做 easeInOutCubic 插值。

    渲染端只需消费过渡帧，不需要感知后端切换；帧级参数过渡保证不"跳脸"。
    """

    transition_ms: int = 400
    _start_ms: int | None = None
    _target: IRenderer | None = None

    def begin(self, target: IRenderer) -> None:
        self._target = target
        self._start_ms = time.monotonic() * 1000

    def active(self) -> bool:
        return self._start_ms is not None and time.monotonic() * 1000 - self._start_ms < self.transition_ms

    def blend(self, frame: dict) -> dict:
        """目标后端为 None（纯音频降级）时产出表情/姿态趋零的过渡帧。"""
        if not self.active() or self._target is None:
            return frame
        t = ease_in_out_cubic((time.monotonic() * 1000 - self._start_ms) / self.transition_ms)
        out = dict(frame)
        expr = out.get("expression") or {"name": "neutral", "weight": 0.0}
        out["expression"] = {"name": expr.get("name", "neutral"), "weight": round(expr.get("weight", 0.0) * t, 4)}
        hp = out.get("head_pose") or {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
        out["head_pose"] = {k: round(v * t, 4) for k, v in hp.items()}
        return out

    def end(self) -> None:
        self._start_ms = None
        self._target = None


@dataclass
class RendererManager:
    """渲染端管理：当前后端 + 手动/自动切换 + 降级策略。"""

    registry: RendererRegistry
    telemetry: dict = field(default_factory=dict)
    interpolator: SwitchInterpolator = field(default_factory=SwitchInterpolator)
    degrade_policy: dict = field(default_factory=lambda: {
        "min_fps": 20.0,
        "fps_duration_s": 5.0,
        "max_gpu_temp_c": 85.0,
        "webgpu_required": True,
        "webgpu_available": True,
    })
    current: IRenderer | None = None
    _switch_history: list[dict] = field(default_factory=list)
    _low_fps_since: float | None = None

    def select(self, name: str) -> IRenderer:
        self.current = self.registry.get(name)
        return self.current

    def switch(self, target_name: str, reason: str = "manual") -> IRenderer | None:
        """手动/自动切换：停旧启新 + 插值过渡。"""
        if self.current and self.current.name == target_name:
            return self.current
        target = self.registry.get(target_name)
        if self.current:
            self.interpolator.begin(target)
            self.current.deactivate()
        target.activate()
        old = self.current.name if self.current else None
        self.current = target
        self._switch_history.append({
            "from": old, "to": target.name, "reason": reason,
            "at_ms": int(time.monotonic() * 1000),
        })
        return self.current

    def push(self, frame: dict) -> None:
        if self.current is None:
            return
        self.current.push(self.interpolator.blend(frame) if self.interpolator.active() else frame)

    def update_telemetry(self, tel: dict) -> None:
        """渲染端性能遥测 → 自动降级判定（V2.0 需求 4.4）。"""
        self.telemetry[tel.get("renderer", "?")] = tel
        if self.current is None:
            return
        fps = tel.get("fps", 60.0)
        temp = tel.get("gpu_temp_c", 50.0)
        now = time.monotonic()
        policy = self.degrade_policy
        if fps < policy["min_fps"]:
            if self._low_fps_since is None:
                self._low_fps_since = now
            elif now - self._low_fps_since > policy["fps_duration_s"]:
                self._degrade(f"帧率 {fps:.1f}fps 持续低于 {policy['min_fps']}fps")
                self._low_fps_since = None
        else:
            self._low_fps_since = None
        if temp > policy["max_gpu_temp_c"]:
            self._degrade(f"GPU 温度 {temp:.0f}°C 超限 {policy['max_gpu_temp_c']}°C")

    def _degrade(self, reason: str) -> None:
        if self.current is None:
            return
        lower = self.registry.next_lower(self.current.name)
        if lower:
            self.switch(lower, reason=f"auto-degrade: {reason}")

    @property
    def history(self) -> list[dict]:
        return list(self._switch_history)

    @property
    def status(self) -> dict:
        return {
            "current": self.current.name if self.current else None,
            "available": self.registry.names(),
            "degrade_chain": self.registry._degrade_chain,
            "telemetry": self.telemetry,
            "switches": self._switch_history[-10:],
        }