"""sherpa-onnx 真实 TTS：Matcha-TTS（zh-baker）+ Vocos 声码器，CPU 实时。

V2.0 需求 4.2 端侧路径：sherpa-onnx 生态推理，RK3588 上 Matcha+Vocos
首音频约 145ms；本机 CPU 同样实时（RTF << 1）。

- 惰性加载：首次合成时才加载 OfflineTts（约 1–2s），服务启动不受影响
- 情绪→韵律：arousal 唤醒度映射语速（兴奋稍快、低落稍慢），句间自然过渡
- 输出：50ms PCM 块，时间戳取自真实音频时长（而非字数估计），
  驱动帧时间线以真实时长为准（音画同步误差 < 100ms 的前提）
"""
from __future__ import annotations

import asyncio
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

from avatar.pipeline.emotion import Emotion
from avatar.pipeline.llm import SentenceChunk
from avatar.pipeline.tts import AudioChunk, TTSAdapter

REPO_ROOT = Path(__file__).resolve().parents[3]
CHUNK_MS = 50  # 与驱动帧同粒度（20fps），音画按时间戳对齐
TAIL_SILENCE_MS = 120  # 句尾静音垫（自然句间停顿）


def float_pcm_to_s16_bytes(samples) -> bytes:
    """[-1,1] float 采样 → 16-bit little-endian PCM bytes。"""
    import array

    arr = array.array("h")
    for s in samples:
        v = int(max(-1.0, min(1.0, float(s))) * 32767.0)
        arr.append(v)
    return arr.tobytes()


@dataclass
class SherpaTTS(TTSAdapter):
    """Matcha + Vocos（中文女声 baker）。实现 TTSAdapter，可被 mock 替换。"""

    model_root: str = ""          # 含 matcha-icefall-zh-baker/ 与 vocos-*/ 的目录
    num_threads: int = 0          # 0 = 自动（min(4, cpu)）
    name: str = "sherpa-matcha-zh"
    _engine: object | None = None
    _load_lock: asyncio.Lock | None = None
    _load_error: str | None = None

    def __post_init__(self) -> None:
        root = self.model_root or os.environ.get(
            "AVATAR_TTS_MODEL_DIR", str(REPO_ROOT / "models" / "tts")
        )
        self.model_root = str(root)
        if not self.num_threads:
            self.num_threads = min(4, os.cpu_count() or 2)
        self._load_lock = asyncio.Lock()

    # ---- 模型路径解析 ----
    def _paths(self) -> dict[str, str]:
        root = Path(self.model_root)
        matcha = next(root.glob("matcha-icefall-zh-baker*"), None)
        if not matcha:
            raise RuntimeError(
                f"TTS 模型未找到（{root}）：需要 matcha-icefall-zh-baker/ 与 vocos 声码器。"
                "下载见 README『TTS 模型』一节"
            )
        acoustic = sorted(matcha.glob("model*.onnx"))  # model.onnx 或 model-steps-3.onnx
        if not acoustic:
            raise RuntimeError(f"matcha onnx 缺失：{matcha}")
        vocoder = next(iter(list(root.glob("vocos*/model*.onnx")) + list(root.glob("vocos*.onnx"))), None)
        if not vocoder:
            raise RuntimeError(
                f"vocos 声码器未找到（{root}）：下载 vocos-22khz-univ.onnx，"
                "见 README『TTS 模型』一节"
            )
        return {
            "acoustic": str(acoustic[0]),
            "vocoder": str(vocoder),
            "lexicon": str(matcha / "lexicon.txt"),
            "tokens": str(matcha / "tokens.txt"),
            "dict_dir": str(matcha / "dict"),
            "matcha_dir": str(matcha),
        }

    @property
    def engine(self):
        """惰性加载 OfflineTts（线程安全：并发首轮只加载一次）。"""
        if self._load_error:
            raise RuntimeError(self._load_error)
        if self._engine is None:
            raise RuntimeError("engine not loaded")  # pragma: no cover
        return self._engine

    async def ensure_loaded(self) -> None:
        if self._engine is not None or self._load_error:
            return
        async with self._load_lock:
            if self._engine is not None or self._load_error:
                return
            try:
                import sherpa_onnx

                p = self._paths()
                matcha_dir = Path(p["matcha_dir"])
                # 文本规则（数字/日期/电话号码 → 文字），存在才挂
                rule_fsts = ",".join(
                    str(matcha_dir / f)
                    for f in ("date.fst", "number.fst", "phone.fst")
                    if (matcha_dir / f).exists()
                )
                cfg = sherpa_onnx.OfflineTtsConfig(
                    model=sherpa_onnx.OfflineTtsModelConfig(
                        matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
                            acoustic_model=p["acoustic"],
                            vocoder=p["vocoder"],
                            lexicon=p["lexicon"],
                            tokens=p["tokens"],
                            dict_dir=p["dict_dir"],
                        ),
                        num_threads=self.num_threads,
                        provider="cpu",
                    ),
                    rule_fsts=rule_fsts,
                    max_num_sentences=1,
                )
                self._engine = await asyncio.to_thread(sherpa_onnx.OfflineTts, cfg)
            except Exception as e:  # 加载失败：记录并在后续调用中快速失败
                self._load_error = f"sherpa-onnx TTS init: {e}"
                raise

    @property
    def sample_rate(self) -> int:
        return int(self.engine.sample_rate)

    def _speed_for(self, emotion: Emotion) -> float:
        """唤醒度 → 语速：兴奋(+)稍快、低落(-)稍慢，幅度 ±8%。"""
        return max(0.9, min(1.1, 1.0 + 0.08 * emotion.arousal))

    def _generate(self, text: str, speed: float):
        return self.engine.generate(text, sid=0, speed=speed)

    async def synthesize(self, chunk: SentenceChunk, ctx: dict, start_ms: int) -> AsyncIterator[AudioChunk]:
        await self.ensure_loaded()
        text = chunk.text.strip()
        if not text:
            return
        speed = self._speed_for(chunk.emotion)
        # 阻塞推理放线程池，避免卡事件循环（句级合成，CPU 实时）
        generated = await asyncio.to_thread(self._generate, text, speed)
        pcm = float_pcm_to_s16_bytes(generated.samples)
        sr = int(generated.sample_rate) or self.sample_rate

        # 句尾静音垫（句间自然停顿）
        tail = b"\x00\x00" * int(sr * TAIL_SILENCE_MS / 1000)
        pcm += tail

        bytes_per_chunk = sr * CHUNK_MS // 1000 * 2
        n_chunks = max(1, math.ceil(len(pcm) / bytes_per_chunk))
        for i in range(n_chunks):
            piece = pcm[i * bytes_per_chunk:(i + 1) * bytes_per_chunk]
            if not piece:
                break
            dur = len(piece) // 2 * 1000 // sr
            yield AudioChunk(
                timestamp_ms=start_ms + i * CHUNK_MS,
                duration_ms=dur,
                pcm_s16le=piece,
                emotion=chunk.emotion,
                sentence_done=(i == n_chunks - 1),
            )
