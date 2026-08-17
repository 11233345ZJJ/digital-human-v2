"""驱动指令编解码：dict(JSON) ↔ Protobuf 双向转换。

- 二进制通道：ServerMessage 封套（驱动帧 / 音频块按时间戳交错）
- JSON 通道：Web 端调试（SSE / JSON WebSocket），音频块以 base64 下发
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

from google.protobuf.json_format import MessageToDict, ParseDict

from avatar.protocol import drive_pb2

EMOTION_PROTO_TO_NAME = {
    drive_pb2.EMOTION_UNSPECIFIED: "calm",
    drive_pb2.EMOTION_HAPPY: "happy",
    drive_pb2.EMOTION_ANGRY: "angry",
    drive_pb2.EMOTION_SAD: "sad",
    drive_pb2.EMOTION_AFRAID: "afraid",
    drive_pb2.EMOTION_DISGUSTED: "disgusted",
    drive_pb2.EMOTION_MELANCHOLIC: "melancholic",
    drive_pb2.EMOTION_SURPRISED: "surprised",
    drive_pb2.EMOTION_CALM: "calm",
}
EMOTION_NAME_TO_PROTO = {v: k for k, v in EMOTION_PROTO_TO_NAME.items()}


def _emotion_to_proto(emo: dict, target) -> None:
    target.label = EMOTION_NAME_TO_PROTO.get(emo.get("label", "calm"), drive_pb2.EMOTION_CALM)
    vec = emo.get("vector")
    if vec:
        target.vector.extend([float(v) for v in vec])
    target.intensity = float(emo.get("intensity", 0.0))
    target.valence = float(emo.get("valence", 0.0))
    target.arousal = float(emo.get("arousal", 0.0))
    target.dominance = float(emo.get("dominance", 0.0))


def dict_to_proto(frame: dict) -> drive_pb2.DriveCommand:
    msg = drive_pb2.DriveCommand()
    msg.timestamp_ms = int(frame["timestamp_ms"])
    msg.frame_seq = int(frame.get("frame_seq", 0))
    if frame.get("phoneme"):
        msg.phoneme = frame["phoneme"]
    if frame.get("trace_id"):
        msg.trace_id = frame["trace_id"]
    if frame.get("expression"):
        msg.expression.name = frame["expression"]["name"]
        msg.expression.weight = float(frame["expression"]["weight"])
    if frame.get("head_pose"):
        msg.head_pose.yaw = float(frame["head_pose"]["yaw"])
        msg.head_pose.pitch = float(frame["head_pose"]["pitch"])
        msg.head_pose.roll = float(frame["head_pose"]["roll"])
    if frame.get("body_gesture"):
        msg.body_gesture.name = frame["body_gesture"]["name"]
        for k, v in frame["body_gesture"].get("params", {}).items():
            msg.body_gesture.params[k] = float(v)
    _emotion_to_proto(frame.get("emotion") or {}, msg.emotion)
    meta = frame.get("semantic_meta") or {}
    msg.semantic_meta.intent = meta.get("intent", "")
    msg.semantic_meta.style = meta.get("style", "")
    msg.semantic_meta.cause = meta.get("cause", "")
    return msg


def proto_to_dict(msg: drive_pb2.DriveCommand) -> dict:
    d = MessageToDict(msg, preserving_proto_field_name=True)
    emo = d.get("emotion", {})
    emo["label"] = EMOTION_PROTO_TO_NAME.get(msg.emotion.label, "calm")
    emo.setdefault("vector", list(msg.emotion.vector))
    d["emotion"] = emo
    d["frame_seq"] = msg.frame_seq
    d["timestamp_ms"] = msg.timestamp_ms
    d.setdefault("phoneme", "")
    expr = d.setdefault("expression", {})
    expr.setdefault("name", "neutral")
    expr.setdefault("weight", 0.0)
    hp = d.setdefault("head_pose", {})
    hp.setdefault("yaw", 0.0)
    hp.setdefault("pitch", 0.0)
    hp.setdefault("roll", 0.0)
    bg = d.setdefault("body_gesture", {})
    bg.setdefault("name", "idle")
    bg.setdefault("params", {})
    meta = d.setdefault("semantic_meta", {})
    meta.setdefault("intent", "")
    meta.setdefault("style", "")
    meta.setdefault("cause", "")
    d.setdefault("trace_id", getattr(msg, "trace_id", ""))
    return d


def audio_to_proto(audio: dict) -> drive_pb2.AudioChunk:
    """音频块 dict → proto。dict 字段：
    timestamp_ms / duration_ms / pcm(bytes) / emotion / sample_rate / trace_id
    """
    msg = drive_pb2.AudioChunk()
    msg.timestamp_ms = int(audio["timestamp_ms"])
    msg.duration_ms = int(audio.get("duration_ms", 0))
    pcm = audio.get("pcm", audio.get("pcm_s16le", b""))
    if isinstance(pcm, str):  # base64（JSON 通道）
        msg.pcm_s16le = base64.b64decode(pcm)
    else:
        msg.pcm_s16le = pcm
    emo = audio.get("emotion") or {}
    if emo:
        _emotion_to_proto(emo, msg.emotion)
    if audio.get("phoneme"):
        msg.phoneme = audio["phoneme"]
    msg.sample_rate = int(audio.get("sample_rate", 0))
    if audio.get("trace_id"):
        msg.trace_id = audio["trace_id"]
    return msg


def proto_audio_to_dict(msg: drive_pb2.AudioChunk) -> dict:
    d = MessageToDict(msg, preserving_proto_field_name=True)
    d["pcm_b64"] = base64.b64encode(msg.pcm_s16le).decode("ascii")
    d["type"] = "audio"
    # uint64 经 MessageToDict 变字符串，这里转回 int（JSON 通道客户端方便）
    d["timestamp_ms"] = msg.timestamp_ms
    d["duration_ms"] = msg.duration_ms
    emo = d.get("emotion", {})
    emo["label"] = EMOTION_PROTO_TO_NAME.get(msg.emotion.label, "calm")
    emo.setdefault("vector", list(msg.emotion.vector))
    d["emotion"] = emo
    d["sample_rate"] = msg.sample_rate or 16000
    d["trace_id"] = msg.trace_id
    return d


def encode_binary(frame: dict) -> bytes:
    """驱动帧 → ServerMessage{command} 二进制（封套，与音频块交错）。"""
    env = drive_pb2.ServerMessage()
    env.command.CopyFrom(dict_to_proto(frame))
    return env.SerializeToString()


def encode_audio_binary(audio: dict) -> bytes:
    """音频块 → ServerMessage{audio} 二进制。"""
    env = drive_pb2.ServerMessage()
    env.audio.CopyFrom(audio_to_proto(audio))
    return env.SerializeToString()


def decode_binary(data: bytes) -> dict:
    """ServerMessage 二进制 → dict（kind=command 返回驱动帧，kind=audio 返回音频块）。"""
    env = drive_pb2.ServerMessage()
    env.ParseFromString(data)
    kind = env.WhichOneof("kind")
    if kind == "audio":
        return proto_audio_to_dict(env.audio)
    if kind == "batch":
        return {"type": "batch", "commands": [proto_to_dict(c) for c in env.batch.commands]}
    return proto_to_dict(env.command)


def encode_json(frame: dict) -> str:
    return json.dumps(frame, ensure_ascii=False, separators=(",", ":"))


SCHEMA_PATH = Path(__file__).resolve().parents[3] / "schema" / "drive.schema.json"


def validate_json(frame: dict) -> list[str]:
    """用 JSON Schema 校验一帧，返回错误列表（空 = 合法）。"""
    import jsonschema

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    try:
        jsonschema.validate(frame, schema)
        return []
    except jsonschema.ValidationError as e:
        return [f"{list(e.path)}: {e.message}"]