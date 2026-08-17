"""Mock 渲染后端：视频神经渲染 / 3D / Live2D / 纯音频 四档。

真实后端（FlashHead / BabylonJS / Cubism）为前端 TS 实现，通过
WebSocket/SSE 消费驱动流；本模块 Mock 后端用于服务端自测与降级链验证。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MockRenderer:
    name: str
    kind: str
    cost_level: int
    gpu_load: float
    note: str = ""
    pushed: int = 0
    last_frame: dict | None = None

    def activate(self) -> None:
        pass

    def deactivate(self) -> None:
        pass

    def push(self, frame: dict) -> None:
        self.pushed += 1
        self.last_frame = frame

    def describe(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "cost_level": self.cost_level,
            "gpu_load": self.gpu_load,
            "note": self.note,
            "pushed": self.pushed,
            "has_frame": self.last_frame is not None,
        }


def build_mock_registry() -> Any:
    """构建四档 Mock 后端注册表 + 标准降级链。"""
    from avatar.renderer.base import RendererRegistry

    reg = RendererRegistry()
    reg.register(MockRenderer("flashhead", "video_neural", 3, gpu_load=0.9,
                              note="SoulX-FlashHead 1.3B, 96FPS@4090"))
    reg.register(MockRenderer("babylon3d", "3d", 2, gpu_load=0.6,
                              note="BabylonJS/Three.js + VRM"))
    reg.register(MockRenderer("vrm", "vrm_3d", 2, gpu_load=0.5,
                              note="Three.js + @pixiv/three-vrm（web/vrm.html 已实现）"))
    reg.register(MockRenderer("live2d", "live2d", 1, gpu_load=0.25,
                              note="Cubism 5 Web SDK"))
    reg.register(MockRenderer("audio", "audio_only", 0, gpu_load=0.0,
                              note="纯音频兜底（语音不中断）"))
    reg.set_degrade_chain(["flashhead", "babylon3d", "live2d", "audio"])
    return reg