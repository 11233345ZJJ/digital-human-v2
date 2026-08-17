from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class EmotionLabel(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    EMOTION_UNSPECIFIED: _ClassVar[EmotionLabel]
    EMOTION_HAPPY: _ClassVar[EmotionLabel]
    EMOTION_ANGRY: _ClassVar[EmotionLabel]
    EMOTION_SAD: _ClassVar[EmotionLabel]
    EMOTION_AFRAID: _ClassVar[EmotionLabel]
    EMOTION_DISGUSTED: _ClassVar[EmotionLabel]
    EMOTION_MELANCHOLIC: _ClassVar[EmotionLabel]
    EMOTION_SURPRISED: _ClassVar[EmotionLabel]
    EMOTION_CALM: _ClassVar[EmotionLabel]
EMOTION_UNSPECIFIED: EmotionLabel
EMOTION_HAPPY: EmotionLabel
EMOTION_ANGRY: EmotionLabel
EMOTION_SAD: EmotionLabel
EMOTION_AFRAID: EmotionLabel
EMOTION_DISGUSTED: EmotionLabel
EMOTION_MELANCHOLIC: EmotionLabel
EMOTION_SURPRISED: EmotionLabel
EMOTION_CALM: EmotionLabel

class Emotion(_message.Message):
    __slots__ = ("label", "vector", "intensity", "valence", "arousal", "dominance")
    LABEL_FIELD_NUMBER: _ClassVar[int]
    VECTOR_FIELD_NUMBER: _ClassVar[int]
    INTENSITY_FIELD_NUMBER: _ClassVar[int]
    VALENCE_FIELD_NUMBER: _ClassVar[int]
    AROUSAL_FIELD_NUMBER: _ClassVar[int]
    DOMINANCE_FIELD_NUMBER: _ClassVar[int]
    label: EmotionLabel
    vector: _containers.RepeatedScalarFieldContainer[float]
    intensity: float
    valence: float
    arousal: float
    dominance: float
    def __init__(self, label: _Optional[_Union[EmotionLabel, str]] = ..., vector: _Optional[_Iterable[float]] = ..., intensity: _Optional[float] = ..., valence: _Optional[float] = ..., arousal: _Optional[float] = ..., dominance: _Optional[float] = ...) -> None: ...

class SemanticMeta(_message.Message):
    __slots__ = ("intent", "style", "cause")
    INTENT_FIELD_NUMBER: _ClassVar[int]
    STYLE_FIELD_NUMBER: _ClassVar[int]
    CAUSE_FIELD_NUMBER: _ClassVar[int]
    intent: str
    style: str
    cause: str
    def __init__(self, intent: _Optional[str] = ..., style: _Optional[str] = ..., cause: _Optional[str] = ...) -> None: ...

class Expression(_message.Message):
    __slots__ = ("name", "weight")
    NAME_FIELD_NUMBER: _ClassVar[int]
    WEIGHT_FIELD_NUMBER: _ClassVar[int]
    name: str
    weight: float
    def __init__(self, name: _Optional[str] = ..., weight: _Optional[float] = ...) -> None: ...

class HeadPose(_message.Message):
    __slots__ = ("yaw", "pitch", "roll")
    YAW_FIELD_NUMBER: _ClassVar[int]
    PITCH_FIELD_NUMBER: _ClassVar[int]
    ROLL_FIELD_NUMBER: _ClassVar[int]
    yaw: float
    pitch: float
    roll: float
    def __init__(self, yaw: _Optional[float] = ..., pitch: _Optional[float] = ..., roll: _Optional[float] = ...) -> None: ...

class BodyGesture(_message.Message):
    __slots__ = ("name", "params")
    class ParamsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    name: str
    params: _containers.ScalarMap[str, float]
    def __init__(self, name: _Optional[str] = ..., params: _Optional[_Mapping[str, float]] = ...) -> None: ...

class DriveCommand(_message.Message):
    __slots__ = ("timestamp_ms", "phoneme", "expression", "head_pose", "body_gesture", "emotion", "semantic_meta", "frame_seq", "trace_id")
    TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    PHONEME_FIELD_NUMBER: _ClassVar[int]
    EXPRESSION_FIELD_NUMBER: _ClassVar[int]
    HEAD_POSE_FIELD_NUMBER: _ClassVar[int]
    BODY_GESTURE_FIELD_NUMBER: _ClassVar[int]
    EMOTION_FIELD_NUMBER: _ClassVar[int]
    SEMANTIC_META_FIELD_NUMBER: _ClassVar[int]
    FRAME_SEQ_FIELD_NUMBER: _ClassVar[int]
    TRACE_ID_FIELD_NUMBER: _ClassVar[int]
    timestamp_ms: int
    phoneme: str
    expression: Expression
    head_pose: HeadPose
    body_gesture: BodyGesture
    emotion: Emotion
    semantic_meta: SemanticMeta
    frame_seq: int
    trace_id: str
    def __init__(self, timestamp_ms: _Optional[int] = ..., phoneme: _Optional[str] = ..., expression: _Optional[_Union[Expression, _Mapping]] = ..., head_pose: _Optional[_Union[HeadPose, _Mapping]] = ..., body_gesture: _Optional[_Union[BodyGesture, _Mapping]] = ..., emotion: _Optional[_Union[Emotion, _Mapping]] = ..., semantic_meta: _Optional[_Union[SemanticMeta, _Mapping]] = ..., frame_seq: _Optional[int] = ..., trace_id: _Optional[str] = ...) -> None: ...

class AudioChunk(_message.Message):
    __slots__ = ("timestamp_ms", "duration_ms", "pcm_s16le", "emotion", "phoneme", "sample_rate", "trace_id")
    TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    DURATION_MS_FIELD_NUMBER: _ClassVar[int]
    PCM_S16LE_FIELD_NUMBER: _ClassVar[int]
    EMOTION_FIELD_NUMBER: _ClassVar[int]
    PHONEME_FIELD_NUMBER: _ClassVar[int]
    SAMPLE_RATE_FIELD_NUMBER: _ClassVar[int]
    TRACE_ID_FIELD_NUMBER: _ClassVar[int]
    timestamp_ms: int
    duration_ms: int
    pcm_s16le: bytes
    emotion: Emotion
    phoneme: str
    sample_rate: int
    trace_id: str
    def __init__(self, timestamp_ms: _Optional[int] = ..., duration_ms: _Optional[int] = ..., pcm_s16le: _Optional[bytes] = ..., emotion: _Optional[_Union[Emotion, _Mapping]] = ..., phoneme: _Optional[str] = ..., sample_rate: _Optional[int] = ..., trace_id: _Optional[str] = ...) -> None: ...

class DriveEvent(_message.Message):
    __slots__ = ("type", "timestamp_ms", "detail", "renderer")
    class Type(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        EVENT_UNSPECIFIED: _ClassVar[DriveEvent.Type]
        EVENT_BARGE_IN: _ClassVar[DriveEvent.Type]
        EVENT_RENDERER_SWITCH: _ClassVar[DriveEvent.Type]
        EVENT_RENDERER_DEGRADE: _ClassVar[DriveEvent.Type]
        EVENT_TURN_START: _ClassVar[DriveEvent.Type]
        EVENT_TURN_END: _ClassVar[DriveEvent.Type]
        EVENT_PING: _ClassVar[DriveEvent.Type]
    EVENT_UNSPECIFIED: DriveEvent.Type
    EVENT_BARGE_IN: DriveEvent.Type
    EVENT_RENDERER_SWITCH: DriveEvent.Type
    EVENT_RENDERER_DEGRADE: DriveEvent.Type
    EVENT_TURN_START: DriveEvent.Type
    EVENT_TURN_END: DriveEvent.Type
    EVENT_PING: DriveEvent.Type
    TYPE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    RENDERER_FIELD_NUMBER: _ClassVar[int]
    type: DriveEvent.Type
    timestamp_ms: int
    detail: str
    renderer: str
    def __init__(self, type: _Optional[_Union[DriveEvent.Type, str]] = ..., timestamp_ms: _Optional[int] = ..., detail: _Optional[str] = ..., renderer: _Optional[str] = ...) -> None: ...

class RendererRegister(_message.Message):
    __slots__ = ("name", "kind", "capability")
    class CapabilityEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    CAPABILITY_FIELD_NUMBER: _ClassVar[int]
    name: str
    kind: str
    capability: _containers.ScalarMap[str, str]
    def __init__(self, name: _Optional[str] = ..., kind: _Optional[str] = ..., capability: _Optional[_Mapping[str, str]] = ...) -> None: ...

class RendererTelemetry(_message.Message):
    __slots__ = ("timestamp_ms", "fps", "gpu_temp_c", "gpu_util", "memory_mb", "buffer_ms", "renderer")
    TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    FPS_FIELD_NUMBER: _ClassVar[int]
    GPU_TEMP_C_FIELD_NUMBER: _ClassVar[int]
    GPU_UTIL_FIELD_NUMBER: _ClassVar[int]
    MEMORY_MB_FIELD_NUMBER: _ClassVar[int]
    BUFFER_MS_FIELD_NUMBER: _ClassVar[int]
    RENDERER_FIELD_NUMBER: _ClassVar[int]
    timestamp_ms: int
    fps: float
    gpu_temp_c: float
    gpu_util: float
    memory_mb: float
    buffer_ms: float
    renderer: str
    def __init__(self, timestamp_ms: _Optional[int] = ..., fps: _Optional[float] = ..., gpu_temp_c: _Optional[float] = ..., gpu_util: _Optional[float] = ..., memory_mb: _Optional[float] = ..., buffer_ms: _Optional[float] = ..., renderer: _Optional[str] = ...) -> None: ...

class DriveFrameBatch(_message.Message):
    __slots__ = ("commands",)
    COMMANDS_FIELD_NUMBER: _ClassVar[int]
    commands: _containers.RepeatedCompositeFieldContainer[DriveCommand]
    def __init__(self, commands: _Optional[_Iterable[_Union[DriveCommand, _Mapping]]] = ...) -> None: ...

class ServerMessage(_message.Message):
    __slots__ = ("command", "audio", "batch")
    COMMAND_FIELD_NUMBER: _ClassVar[int]
    AUDIO_FIELD_NUMBER: _ClassVar[int]
    BATCH_FIELD_NUMBER: _ClassVar[int]
    command: DriveCommand
    audio: AudioChunk
    batch: DriveFrameBatch
    def __init__(self, command: _Optional[_Union[DriveCommand, _Mapping]] = ..., audio: _Optional[_Union[AudioChunk, _Mapping]] = ..., batch: _Optional[_Union[DriveFrameBatch, _Mapping]] = ...) -> None: ...
