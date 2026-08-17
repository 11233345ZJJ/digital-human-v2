# Avatar V2 —— 大语言模型驱动数字人多模态交互系统

里程碑 1：**统一驱动 IR + 服务骨架**（全 Mock 跑通全链路）。
里程碑 2：**真实智谱 LLM 接入**（流式切句 + 行为指令双通道 + 多轮历史）。
里程碑 3：**SenseVoice 情感识别**（ASR + 情绪 + 音频事件一体 → 统一 8 维中间表示）。
里程碑 4：**sherpa-onnx 真实 TTS + Trace ID 全链路日志**（真实音频流 + 音画时间戳对齐 + 首响延迟拆解）。
里程碑 5：**语音输入闭环**（浏览器麦克风 → SenseVoice → 共情起轮，语音可随时打断播报）。
里程碑 6：**流式语音识别**（WS 实时推流 + 增量分块识别 + 实时字幕 + 能量 VAD 自动断句）。

以 LLM 为核心，从流式文本生成 → 情感分析 → 情感语音合成 → 数字人表情动作
驱动的全链路多模态交互系统。整体采用流式并行架构：首句生成后立即触发下游
管线，实现"边生成、边分析、边合成、边渲染"。

## 目录结构

```
proto/             统一驱动数据接口（Protobuf，线上契约唯一来源）
  drive.proto       DriveCommand / Emotion(8维+VAD) / AudioChunk / ServerMessage 封套
schema/            JSON Schema（Web 端调试用，与 proto 对齐）
python/avatar/
  env.py           .env 加载（ZHIPU_API_KEY 等，勿提交仓库）
  trace.py         Trace ID 全链路追踪（Tracer 打点 + 首响延迟拆解 summary）
  protocol/        生成绑定 + dict↔proto 编解码（二进制/JSON 双通道 + 音频封套）
  pipeline/        LLM 流式切句 → 情绪平滑 → TTS 时间线 → 驱动帧生成
    llm.py          LLMAdapter 接口 + MockLLM（确定性情绪轨迹脚本）
    llm_zhipu.py    ZhipuLLM：智谱 API 流式接入（JSONL 双通道 + 规则兜底）
    emotion.py      8 维情绪 + VAD + 滑动窗口平滑 + 规则情绪估计
    emotion_recognizer.py  SenseVoice 识别器 + 输入侧情绪跟踪（共情）
    chunker.py      流式切句器（首句即推，中英文标点保护）
    tts.py          TTSAdapter + MockTTS + 句级时间线 + 情绪 slerp 插值
    tts_sherpa.py   SherpaTTS：Matcha-TTS zh-baker + Vocos（CPU 实时，RTF≈0.04）
    driving.py      DriveCommand 帧生成（20fps，表情/姿态/动作映射）
  renderer/        IRenderer 注册表 + 降级链 + easeInOutCubic 切换插值
  server/          FastAPI：/ws/drive(protobuf) /sse/drive(JSON+音频) /chat /tts/preview
web/               SSE 调试台（SVG 头像 + WebAudio 音频播放 + 音画时钟同步）
tests/             单元测试（stdlib + pytest）
models/sensevoice/ SenseVoiceSmall 模型（model.pt + config + tokenizer，自动/手动下载）
models/tts/        Matcha-TTS zh-baker + vocos 声码器（手动下载，见下）
scripts/gen_proto.sh  重新生成 protobuf 绑定
```

## 快速开始

```bash
python -m venv python/.venv
python/.venv/bin/pip install -r requirements.txt
cp .env.example .env          # 填入 ZHIPU_API_KEY
./scripts/gen_proto.sh        # 生成 protobuf 绑定
python/.venv/bin/python -m pytest tests/ -q   # 30 项单测

# SenseVoice 模型（可选，情感识别需要）
mkdir -p models/sensevoice && cd models/sensevoice
aria2c -x 8 -s 8 --header="Accept-Encoding: identity" -d . -o model.pt \
  "https://modelscope.cn/models/iic/SenseVoiceSmall/resolve/master/model.pt"
for f in config.yaml tokens.json am.mvn chn_jpn_yue_eng_ko_spectok.bpe.model; do
  aria2c -x 8 -s 8 --header="Accept-Encoding: identity" -d . -o "$f" \
    "https://modelscope.cn/models/iic/SenseVoiceSmall/resolve/master/$f"
done
ln -sf chn_jpn_yue_eng_ko_spectok.bpe.model bpe.model
cd ../..

# TTS 模型（可选，真实语音需要）：Matcha zh-baker + vocos 声码器
mkdir -p models/tts && cd models/tts
aria2c -x 8 -s 8 "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-zh-baker.tar.bz2"
tar xjf matcha-icefall-zh-baker.tar.bz2 && rm matcha-icefall-zh-baker.tar.bz2
aria2c -x 8 -s 8 "https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-22khz-univ.onnx"
cd ../..

# 启动服务（默认 Mock LLM；用智谱 LLM 时设置环境变量）
cd python
AVATAR_LLM=zhipu AVATAR_TTS=sherpa .venv/bin/uvicorn avatar.server.app:app --host 127.0.0.1 --port 8765

# 调试台（浏览器）
python -m http.server -d web 8080   # 打开 http://127.0.0.1:8080/renderer.html
```

> 注意：SenseVoice 推理需要 torch（CPU wheel 即可）。
> `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu`
> 如遇 tmpfs 满（/tmp 过小），`export TMPDIR=/path/on/disk` 后再装。
> 识别器为惰性加载：无 torch 环境服务照常启动，仅情感 API 返回 503。
> TTS 同为惰性加载 + 启动后台预热：无模型环境服务照常启动，`tts=mock` 时间线兜底。

## LLM / TTS 后端选择

| 后端 | 选择方式 | 说明 |
| --- | --- | --- |
| mock LLM | 默认 | 确定性情绪轨迹脚本，离线自测 |
| zhipu LLM | `AVATAR_LLM=zhipu` 或 `/chat` 请求体 `{"llm":"zhipu"}` | 智谱 glm-4-flash 流式 |
| mock TTS | 默认 | 静音 PCM 时间线（无真实语音） |
| sherpa TTS | `AVATAR_TTS=sherpa` 或 `{"tts":"sherpa"}` | Matcha-TTS + Vocos 真实中文语音 |

`ZhipuLLM`（`pipeline/llm_zhipu.py`）实现 LLMAdapter：
- **JSONL 双通道流式**：system prompt 要求模型每行输出一句
  `{"t":文本, "e":情绪, "i":强度, "intent":意图, "style":风格, "cause":诱因}`，
  流式解析、首句即推——文本与行为指令同一次生成中并行产出
- **兜底路径**：模型输出非法时退化为纯文本切句 + 规则情绪词典
- **多轮历史**：最近 6 轮对话上下文自动携带
- 实测：首句 0.9–2.6s（智谱 API 端到端 TTFB，模型侧延迟为主），
  驱动帧流式到达，情绪标注随句平滑过渡

## API 一览

| 端点 | 说明 |
| --- | --- |
| `GET /health` | 健康检查（含 LLM/TTS 后端、最近一轮 trace 摘要） |
| `GET /renderers` | 渲染后端注册表 + 降级链 + 切换历史 |
| `POST /renderers/switch` | 手动切换 `{name, reason}`（切换期间 400ms easeInOutCubic 过渡） |
| `POST /renderers/telemetry` | 渲染端性能遥测 → 自动降级（fps<20 持续 5s / GPU>85°C） |
| `GET /sse/drive` | JSON 驱动帧流 + 音频块流（EventSource 调试，音频 base64） |
| `WS /ws/drive` | 驱动帧 + 音频块；默认 protobuf ServerMessage 封套二进制；首帧 `{"mode":"json"}` 可切 JSON；支持 `{"type":"chat"}` 与 `{"type":"barge_in"}` |
| `POST /chat` | 发起一轮对话，返回 `trace_id`；`{text, llm, tts, renderer, user_emotion}` |
| `WS /ws/voice` | 流式语音识别：推 16k PCM（binary）→ 实时回推 partial 字幕 / final / turn_started；`start`/`stop`/`cancel` 控制 |
| `POST /voice/turn` | 语音闭环（非流式兜底）：wav(16k) 上传（`?llm=&tts=` 可选）→ SenseVoice 识别文本+情绪 → 自动起带共情上下文的轮次 |
| `POST /tts/preview` | 文本 → wav（快速验证 TTS；响应头带 X-RTF/X-Synth-Ms/X-Duration-Ms） |
| `POST /emotion/audio` | 上传用户语音 wav(16k) → SenseVoice 识别（ASR+情绪+事件+LID）→ 更新用户情绪状态 |
| `GET /emotion/state` | 当前用户情绪状态（滑动窗口平滑后） |
| `GET /history` | 对话历史（含 trace_id / audio_ms / trace 摘要） |
| `POST /barge_in` | 打断：清空时间戳队列未消费帧 |

## 情感识别（SenseVoice）

`pipeline/emotion_recognizer.py`：
- **SenseVoice（FunASR）**：ASR + LID + 情绪 + 音频事件一体，CPU 推理
  RTF≈0.41（10s 音频约 1s）
- 情绪标签映射：SenseVoice 官方（HAPPY/SAD/ANGRY/NEUTRAL/FEARFUL/DISGUSTED/SURPRISED）
  → 统一 8 维情绪向量（IndexTTS-2.5 标准）+ 强度标定
- `_parse_sensevoice_tokens`：解析 funasr 1.4.x 输出中拼入文本的 `<|...|>` 特殊 token
- **输入侧情绪跟踪**（`InputEmotionTracker`）：滑动窗口(5 句) EWMA 平滑，
  用户情绪状态随语音持续更新
- **共情链路**：`/chat` 携带 `user_emotion` → LLM system prompt 注入用户情绪感知
  → 数字人调整回应语气（实测：angry 用户得到安抚式回应）
- 部署注意：funasr 1.4.x 需本地目录含 `config.yaml`/`tokens.json`/`am.mvn`/
  `bpe.model`（符号链接）才能加载；`configuration.json` 分支有 bug 需删除

## 情感语音合成（sherpa-onnx）

`pipeline/tts_sherpa.py`（`SherpaTTS` 实现 `TTSAdapter`）：
- **Matcha-TTS zh-baker + Vocos**（中文女声，22.05kHz，CPU 实时）：
  本机实测合成 RTF≈0.04（6.2s 音频 243ms 出），模型加载约 1.3s（启动后台预热）
- **rule_fsts**：挂接 date/number/phone 规则，"2026年8月17日" 等数字日期正常念读
- **情绪→韵律**：arousal 唤醒度映射语速（±8%，兴奋稍快/低落稍慢）
- **50ms PCM 块 + 时间戳**：真实音频时长驱动时间线（音画同步的时间源），
  每块携带情绪标签与 trace_id；句尾 120ms 静音垫
- **降级语义**：TTS 失败时广播 error 事件并退化为纯驱动帧（mock 时间线兜底），
  整轮对话不中断
- 二进制通道为 `ServerMessage{audio}` protobuf 封套，JSON/SSE 通道为
  `{"type":"audio", "pcm_b64":...}`；与驱动帧按时间戳交错下发

## 全链路 Trace ID

`trace.py`（`Tracer`）：每轮对话一个 12 位 hex trace_id，从 `/chat` 一路带到渲染端。
- **打点事件**：`turn_start → llm_first_token → llm_first_chunk → tts_first_audio
  → drive_first_frame → turn_end`，日志格式
  `trace=xxx event=yyy elapsed_ms=%.1f delta_ms=%.1f`
- **首响延迟拆解**：`summary()` 输出各里程碑累计耗时与分段耗时
  （如 `tts_first_audio__from_llm_first_chunk_ms` = 切句+合成耗时）
- **透传**：trace_id 写入每帧 DriveCommand、每个 AudioChunk、turn 事件、
  `/chat` 响应、`/history` 记录与 `/health.last_trace`
- **Web 调试台**实时显示最近一轮的延迟拆解面板

```protobuf
DriveCommand { ... string trace_id = 9; }
AudioChunk   { ... uint32 sample_rate = 6; string trace_id = 7; }
ServerMessage { oneof kind { DriveCommand command = 1; AudioChunk audio = 2; } }
```

## 语音输入闭环（流式识别）

Web 调试台「🎤 说话」按钮（点击开始/再点击停止发送）+「自动断句」开关：
- **实时推流**：getUserMedia（回声消除+降噪）→ 逐块降采样 16k s16le →
  WebSocket `/ws/voice` binary 推送（无整段上传等待）
- **增量分块识别**（`pipeline/voice_stream.py`）：SenseVoice 非流式 →
  每 800ms 对最近 12s 尾窗识别一次，回推 partial 实时字幕（识别串行化 +
  忙则跳过限频，partial 计算量有窗口上限）
- **自动断句**：100ms 块能量 VAD（RMS 阈值）——说话≥600ms 后静音≥900ms
  自动 finalize 并起轮，无需手动点停止；也可手动停止发送
- **final 衔接**：最终识别 → 用户情绪跟踪 → 共情起轮，复用同一条 trace
  （summary 呈现 asr_first_partial / asr_final → llm_first_chunk → tts_first_audio 全链路）
- **兜底**：WS 不可用时自动回退整段 WAV 上传 `/voice/turn`
- 实测：3.1s 语音流式期间出 2 条 partial（说完前 1s 即见字幕）；
  final 后 325ms 出 LLM 首句；稳态单次识别 RTF≈0.4（CPU）
- 调试台地址推断：静态服务(:8080) 打开时自动指向 http://127.0.0.1:8765，
  亦可用 `?api=http://host:port` 覆盖
- CLI 验证（非流式）：`curl -F file=@user16k.wav "http://127.0.0.1:8765/voice/turn?llm=zhipu&tts=sherpa"`

## 统一驱动数据接口（V2.0 核心）

```protobuf
DriveCommand {
  timestamp_ms: uint64      // 音频时钟对齐
  phoneme: string           // 口型视素
  expression: {name, weight}
  head_pose: {yaw, pitch, roll}
  body_gesture: {name, params}
  emotion: {label, vector[8], intensity, valence, arousal, dominance}
  semantic_meta: {intent, style, cause}
  frame_seq: uint32
}
```

- 8 维情绪向量顺序固定：happy/angry/sad/afraid/disgusted/melancholic/surprised/calm
- 同一 Emotion 同时喂 TTS 与渲染器（降低级联误差的核心机制）
- 二进制通道 protobuf 序列化；JSON 通道可用 `schema/drive.schema.json` 校验
- 帧按时间戳有序消费；TTS 输出时钟为唯一时间源

## 已实现

- [x] Protobuf 统一驱动 IR + Python 绑定 + JSON Schema 双契约
- [x] 8 维情绪向量 + emo_alpha 强度 + VAD 辅助空间 + 锚点表
- [x] 滑动窗口(7 句) EWMA 平滑 + 趋势预测（防单句误判/情绪跳变）
- [x] 句间情绪 slerp 球面插值（TTS 情感过渡）
- [x] 流式切句器（首句即推，中文/英文缩写保护）
- [x] Mock 全链路：LLM 双通道句流 → 平滑 → TTS 时间线 → 20fps 驱动帧
- [x] IRenderer 注册表 + 降级链 flashhead→babylon3d→live2d→audio
- [x] 手动/自动切换 + easeInOutCubic 插值过渡（不"跳脸"）
- [x] FastAPI：WS(protobuf 封套) + SSE(含音频) + 控制面 + 遥测自动降级
- [x] 打断（barge-in）复位 + 时间戳有序队列
- [x] Web SSE 调试台（SVG 头像 + WebAudio 实时播放 + 音画时钟同步 + 手势解锁）
- [x] 真实智谱 LLM：JSONL 双通道流式（文本+行为指令随句）、
      规则情绪兜底、多轮历史、每轮可切换 LLM 后端
- [x] SenseVoice 情感识别：ASR+情绪+事件+LID → 统一 8 维中间表示、
      输入侧情绪跟踪（EWMA 平滑）、共情上下文注入、音频上传 API
- [x] sherpa-onnx 真实 TTS：Matcha zh-baker + Vocos（CPU RTF≈0.04）、
      rule_fsts 数字日期念读、arousal→语速、启动预热、TTS 失败降级
- [x] Trace ID 全链路：proto 字段透传（帧/音频/事件/历史/健康检查）、
      六里程碑打点 + 首响延迟拆解 summary + Web 端 trace 面板
- [x] 音画同步下发：音频块与驱动帧按时间戳交错（同刻音频先行），
      客户端按音频时钟逐帧消费
- [x] 语音输入闭环：麦克风推按式录音 → 16k WAV → SenseVoice 识别 →
      共情自动起轮；录音即打断（语音接管）；识别器 opt-in 预热
- [x] 流式语音识别：WS 实时推 PCM → 增量分块 partial 实时字幕 →
      能量 VAD 自动断句 → final 无缝衔接对话轮（同一条 trace 全链路）
- [x] VoiceSession 单测 10 项（VAD 计数/partial 节流/窗口上限/自动收尾/取消）

## 下一步（里程碑 7 候选）

- 真实渲染后端（Cubism Live2D / Three.js VRM / WebGPU）
- WebRTC 音频推流 + 动态缓冲水位反馈
- LLM 直驱表情/动作指令通道（SoulLink 模式）+ 动作库检索
- 音素级口型时间线（替代当前字符近似视素）
- VAD 升级 silero（sherpa-onnx 自带）替代能量阈值，提升嘈杂环境鲁棒性

## 已知问题

- 低内存机器（≤4G）：SenseVoice(torch) 与 sherpa TTS 同进程常驻约 800MB+，
  系统换页时首次语音识别可从 ~2.5s 恶化到 20s+。已内置缓解：torch 线程限 4、
  TTS 线程限 4、启动预热。仍紧张时建议只开其一（tts=mock 或不开识别器）
- glm-4-flash 在跨轮上下文依赖（如"记住我的名字"）时可能答错，建议换
  glm-4-air / glm-4-plus 等更强模型（`ZHIPU_MODEL` 环境变量）
- 首句延迟 0.9–2.6s 受智谱 API TTFB 制约；目标 <1s 需端侧模型或投机解码