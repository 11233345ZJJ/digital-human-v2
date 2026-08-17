/* 驱动流调试台：SSE 消费 JSON 驱动帧 + 音频块（base64 PCM），
   WebAudio 实时播放 + 驱动帧按音频时钟逐帧消费（音画同步）。 */
"use strict";

const API = new URLSearchParams(location.search).get("api")
  || (location.origin.includes(":8080") ? "http://127.0.0.1:8765" : location.origin);
const WS_BASE = API.replace(/^http/, "ws");
const EMOTION_CN = {
  happy: "高兴", angry: "愤怒", sad: "悲伤", afraid: "恐惧",
  disgusted: "厌恶", melancholic: "忧郁", surprised: "惊讶", calm: "平静",
};

const $ = (id) => document.getElementById(id);

function log(msg) {
  const el = $("log");
  el.insertAdjacentHTML("beforeend", `<div>${new Date().toLocaleTimeString()} ${msg}</div>`);
  el.scrollTop = el.scrollHeight;
}

/* ---- 用户情绪（SenseVoice 识别结果展示） ---- */
fetch(`${API}/emotion/state`).then(r => r.json()).then(s => {
  if (s.ok && s.user_emotion) {
    log(`用户情绪: ${EMOTION_CN[s.user_emotion.label] || s.user_emotion.label} (${s.user_emotion.intensity})`);
  }
});

/* ---- 渲染端切换按钮 ---- */
fetch(`${API}/renderers`).then(r => r.json()).then(s => {
  const box = $("switchBtns");
  for (const name of s.available) {
    const b = document.createElement("button");
    b.className = "btn";
    b.textContent = "切到 " + name + (name === s.current ? " ✓" : "");
    b.onclick = () => {
      fetch(`${API}/renderers/switch`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, reason: "web-debug" }),
      }).then(r => r.json()).then(st => log(`切换 → ${st.current}`));
    };
    box.appendChild(b);
  }
});

/* ================= 音频播放（WebAudio，需用户手势解锁） ================= */
let actx = null;            // AudioContext（点击"启用声音"后创建）
let audioQueue = [];        // 待调度 {pcm(Int16), sampleRate, timestampMs, durMs}
let scheduledSources = [];  // 已调度的 AudioBufferSourceNode（打断时可停）
let lastScheduleEnd = 0;    // 上一块调度到的 ctx 时间（秒）
let playbackOrigin = null;  // 本轮首块调度到的 ctx 时间 → 音频时钟零点
let audioChunksRecv = 0, audioMsRecv = 0;

function enableSound() {
  if (!actx) {
    actx = new (window.AudioContext || window.webkitAudioContext)();
    log(`声音已启用（AudioContext ${actx.sampleRate}Hz）`);
    $("soundBtn").textContent = "声音已启用 ✓";
    $("soundBtn").disabled = true;
  }
  actx.resume();
}

function b64ToI16(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Int16Array(bytes.buffer);
}

function scheduleAudio() {
  if (!actx || audioQueue.length === 0) return;
  const PREBUFFER_MS = 250; // 起播缓冲：防网络抖动断流
  const now = actx.currentTime;
  if (playbackOrigin === null) {
    if (audioMsRecv < PREBUFFER_MS) return; // 攒够再起播
    playbackOrigin = now + 0.05;
    lastScheduleEnd = playbackOrigin;
  }
  while (audioQueue.length) {
    const item = audioQueue[0];
    const start = Math.max(lastScheduleEnd, now + 0.01);
    const buf = actx.createBuffer(1, item.pcm.length, item.sampleRate);
    const ch = buf.getChannelData(0);
    for (let i = 0; i < item.pcm.length; i++) ch[i] = item.pcm[i] / 32768;
    const src = actx.createBufferSource();
    src.buffer = buf;
    src.connect(actx.destination);
    src.start(start);
    scheduledSources.push(src);
    lastScheduleEnd = start + buf.duration;
    audioQueue.shift();
  }
  $("audioBuf").textContent = `${audioChunksRecv} 块 / ${audioMsRecv}ms`;
}

function stopAudio() {
  for (const s of scheduledSources) { try { s.stop(); } catch (e) {} }
  scheduledSources = [];
  audioQueue = [];
  playbackOrigin = null;
  lastScheduleEnd = 0;
  $("audioBuf").textContent = "-（已清空）";
}

/* ---- 音频时钟（毫秒）：有音频走 AudioContext，无音频走墙钟 ---- */
let wallOrigin = null;
function clockMs() {
  if (actx && playbackOrigin !== null) {
    return Math.max(0, (actx.currentTime - playbackOrigin) * 1000);
  }
  if (wallOrigin !== null) return performance.now() - wallOrigin;
  return -1;
}

/* ================= 驱动帧调度（按时间戳消费） ================= */
let frameQueue = [];   // {frame, timestamp_ms} 按时间戳排序
let frameCount = 0;

function pushFrame(f) {
  frameQueue.push(f);
  frameQueue.sort((a, b) => a.timestamp_ms - b.timestamp_ms);
  if (wallOrigin === null && playbackOrigin === null) wallOrigin = performance.now();
}

function consumeFrames() {
  const now = clockMs();
  if (now < 0) return;
  while (frameQueue.length && frameQueue[0].timestamp_ms <= now + 50) {
    renderFrame(frameQueue.shift());
  }
}

function renderFrame(f) {
  frameCount++;
  $("status").textContent =
    `渲染后端: ${f.expression?.name ?? "-"} / 帧: ${frameCount} / 时间戳: ${f.timestamp_ms}ms / `;

  const emo = f.emotion || {};
  const label = emo.label || "calm";
  $("emotionLabel").textContent = EMOTION_CN[label] || label;
  $("expression").textContent = `${f.expression?.name ?? "-"} × ${((f.expression?.weight ?? 0) * 100).toFixed(0)}%`;
  $("phoneme").textContent = f.phoneme || "-";
  $("gesture").textContent = f.body_gesture?.name ?? "-";
  const meta = f.semantic_meta || {};
  $("meta").textContent = `${meta.intent || "-"} / ${meta.style || "-"}`;

  const bars = $("emobars");
  if (!bars.dataset.built) {
    bars.dataset.built = "1";
    bars.innerHTML = (emo.vector || []).map((_, i) =>
      `<div style="font-size:11px">${Object.keys(EMOTION_CN)[i]}<div class="bar"><i id="bar${i}"></i></div></div>`).join("");
  }
  (emo.vector || []).forEach((v, i) => { $(`bar${i}`).style.width = `${(v * 100).toFixed(0)}%`; });

  const mood = moodFromEmotion(label);
  $("mouth").setAttribute("d", mouthPath(mood, f.phoneme));
  document.getElementById("eyeL").style.transform = `translate(${f.head_pose?.yaw * 40 || 0}px, 0)`;
  document.getElementById("eyeR").style.transform = `translate(${f.head_pose?.yaw * 40 || 0}px, 0)`;
}

function moodFromEmotion(label) {
  return { happy: "smile", sad: "frown", angry: "frown", surprised: "open" }[label] || "neutral";
}
function mouthPath(mood, phoneme) {
  const open = (phoneme === "AA" || phoneme === "OW" || phoneme === "UW" || mood === "open") ? 22 : 12;
  const curl = mood === "smile" ? "M65 145 Q100 152 135 145" : mood === "frown" ? "M65 152 Q100 142 135 152" : "M65 147 Q100 143 135 147";
  return `${curl} Q100 ${155 + open} 65 145`;
}

/* ---- 渲染主循环 ---- */
setInterval(() => {
  scheduleAudio();
  consumeFrames();
  const t = clockMs();
  $("audioClock").textContent = `音频时钟: ${t < 0 ? "-" : t.toFixed(0) + "ms"} / 排队帧: ${frameQueue.length}`;
}, 30);

/* ================= SSE 消息分发 ================= */
function showTrace(s) {
  if (!s) return;
  const lines = [];
  if (s.llm_first_token_ms != null) lines.push(`llm_first_token   ${s.llm_first_token_ms}ms`);
  if (s.llm_first_chunk_ms != null) lines.push(`llm_first_chunk   ${s.llm_first_chunk_ms}ms`);
  if (s.tts_first_audio_ms != null) lines.push(`tts_first_audio   ${s.tts_first_audio_ms}ms`);
  if (s.drive_first_frame_ms != null) lines.push(`drive_first_frame ${s.drive_first_frame_ms}ms`);
  if (s.turn_end_ms != null) lines.push(`turn_end          ${s.turn_end_ms}ms`);
  if (s.tts_first_audio__from_llm_first_chunk_ms != null)
    lines.push(`→ 切句+合成耗时   ${s.tts_first_audio__from_llm_first_chunk_ms}ms`);
  $("trace").textContent = lines.join("\n") || JSON.stringify(s);
}

const es = new EventSource(`${API}/sse/drive`);
$("conn").textContent = "(SSE 已连接)";
$("conn").style.color = "#9d9";
es.onmessage = (ev) => {
  let obj;
  try { obj = JSON.parse(ev.data); } catch { return; }
  switch (obj.type) {
    case "turn_start":
      // 新轮次：复位音画队列与时钟（打断语义同）
      stopAudio();
      frameQueue = [];
      wallOrigin = null;
      log(`turn_start: ${obj.text} [trace ${obj.trace_id}]`);
      break;
    case "turn_end":
      showTrace(obj.summary);
      log(`turn_end [trace ${obj.trace_id}]` +
          (obj.summary?.turn_end_ms != null ? ` 全链路 ${obj.summary.turn_end_ms}ms` : ""));
      break;
    case "error":
      log(`ERROR: ${obj.detail} [trace ${obj.trace_id ?? "-"}]`);
      break;
    case "audio":
      audioChunksRecv++;
      audioMsRecv += obj.duration_ms || 0;
      audioQueue.push({
        pcm: b64ToI16(obj.pcm_b64),
        sampleRate: obj.sample_rate || 16000,
        timestampMs: obj.timestamp_ms,
        durMs: obj.duration_ms,
      });
      if (wallOrigin === null && playbackOrigin === null) wallOrigin = performance.now();
      break;
    default:
      if (obj.timestamp_ms != null) pushFrame(obj); // 驱动帧
  }
};
es.onerror = () => { $("conn").textContent = "(SSE 断开)"; $("conn").style.color = "#f77"; };

/* ================= 麦克风语音输入（WS 流式识别闭环） =================
   点击开始 → 抓麦克风 PCM 实时推 /ws/voice（16k s16le）→
   服务端增量分块识别回推 partial（实时字幕）→ 停止/自动断句 →
   final → 服务端自动起对话轮。WS 不可用时回退整段上传 /voice/turn。 */
let micRecording = false;
let micCtx = null, micStream = null, micProc = null, micSrc = null;
let micSamples = [];     // 原始 Float32 分块（兜底整段上传用）
let micSampleRate = 0;
let micTimer = null, micT0 = 0;
let voiceWs = null, voiceOpen = false, voiceFallback = false, voicePending = [];

async function toggleMic() {
  if (micRecording) { stopMic(); return; }
  await startMic();
}

async function startMic() {
  try {
    // 数字人正在说话 → 先打断（语音可随时接管）
    await fetch(`${API}/barge_in`, { method: "POST" });
    stopAudio();
    frameQueue = [];
    wallOrigin = null;

    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
    micCtx = new (window.AudioContext || window.webkitAudioContext)();
    micSampleRate = micCtx.sampleRate;
    micSrc = micCtx.createMediaStreamSource(micStream);
    micProc = micCtx.createScriptProcessor(4096, 1, 1);
    micSamples = [];
    openVoiceWs();
    micProc.onaudioprocess = (e) => {
      const f32 = new Float32Array(e.inputBuffer.getChannelData(0));
      micSamples.push(f32);
      streamVoiceChunk(f32);
    };
    micSrc.connect(micProc);
    micProc.connect(micCtx.destination); // 某些浏览器必须连接才回调
    micRecording = true;
    micT0 = performance.now();
    $("micBtn").textContent = "⏹ 停止并发送";
    $("micBtn").style.color = "#f77";
    $("asrLive").textContent = "";
    micTimer = setInterval(() => {
      $("micStatus").textContent =
        `录音中 ${(performance.now() - micT0) / 1000 | 0}s${voiceOpen ? " · 流式识别中" : ""}`;
    }, 200);
  } catch (e) {
    log(`麦克风不可用: ${e.message || e}`);
    cleanupMic();
  }
}

/* ---- 流式通道 ---- */
function openVoiceWs() {
  voiceOpen = false; voiceFallback = false; voicePending = [];
  try {
    voiceWs = new WebSocket(`${WS_BASE}/ws/voice`);
    voiceWs.binaryType = "arraybuffer";
  } catch { voiceFallback = true; return; }
  voiceWs.onopen = () => {
    voiceWs.send(JSON.stringify({
      type: "start", llm: $("llmSel").value, tts: $("ttsSel").value,
      auto_end: $("autoEnd").checked,
    }));
  };
  voiceWs.onmessage = (ev) => {
    let o; try { o = JSON.parse(ev.data); } catch { return; }
    switch (o.type) {
      case "ready":
        voiceOpen = true;
        for (const c of voicePending) voiceWs.send(c);
        voicePending = [];
        break;
      case "partial":
        $("asrLive").textContent = `🎧 ${o.text}`;
        break;
      case "final":
        $("asrLive").textContent = o.text ? `✅ ${o.text}` : "";
        if (o.text) {
          $("userEmotion").textContent =
            `当前用户情绪: ${EMOTION_CN[o.emotion.label]} × ${o.emotion.intensity.toFixed(1)}`;
          log(`🎤✅ "${o.text}" | 情绪 ${o.raw_label} → ${EMOTION_CN[o.emotion.label]}` +
              (o.reason === "auto_end" ? " | 自动断句" : "") + ` [trace ${o.trace_id}]`);
        } else {
          log("未识别到语音内容");
        }
        if (micRecording) finishRecording();  // 自动断句：本地也停录
        break;
      case "turn_started":
        log(`对话已自动开始 [trace ${o.trace_id}]`);
        break;
      case "error":
        log(`语音流错误: ${o.detail}`);
        break;
      case "cancelled":
        log("录音已取消");
        break;
    }
  };
  voiceWs.onerror = () => { if (!voiceOpen) voiceFallback = true; };
  voiceWs.onclose = () => { voiceOpen = false; };
}

function streamVoiceChunk(f32) {
  if (voiceFallback) return;
  const i16 = f32ToI16(resampleLinear(f32, micSampleRate, 16000));
  if (voiceWs && voiceOpen && voiceWs.readyState === 1) {
    voiceWs.send(i16.buffer);
  } else {
    voicePending.push(i16.buffer);  // 连接建立前缓存
  }
}

function f32ToI16(f32) {
  const out = new Int16Array(f32.length);
  for (let i = 0; i < f32.length; i++) {
    const s = Math.max(-1, Math.min(1, f32[i]));
    out[i] = s < 0 ? s * 32768 : s * 32767;
  }
  return out;
}

function stopMic() {
  if (!micRecording) return;
  const wasStreaming = voiceWs && voiceOpen && voiceWs.readyState === 1;
  const secs = (performance.now() - micT0) / 1000;
  const sr = micSampleRate;
  const chunks = micSamples;
  finishRecording();
  if (wasStreaming) {
    $("micStatus").textContent = "最终识别中…";
    voiceWs.send(JSON.stringify({ type: "stop" }));
    setTimeout(() => { try { voiceWs && voiceWs.close(); } catch (e) {} }, 10000);
    return;
  }
  // 兜底：整段上传（WS 不可用）
  if (secs < 0.4 || !chunks.length) { log("录音太短，已忽略"); return; }
  const merged = mergeFloat32(chunks);
  const wav = encodeWav16kMono(resampleLinear(merged, sr, 16000));
  $("micStatus").textContent = `识别中（${secs.toFixed(1)}s 音频）…`;
  const fd = new FormData();
  fd.append("file", new Blob([wav], { type: "audio/wav" }), "user.wav");
  const q = `llm=${$("llmSel").value}&tts=${$("ttsSel").value}`;
  fetch(`${API}/voice/turn?${q}`, { method: "POST", body: fd })
    .then(r => r.json())
    .then(s => {
      $("micStatus").textContent = "";
      if (!s.ok) { log(`语音识别失败: ${s.error}`); return; }
      $("userEmotion").textContent = `当前用户情绪: ${EMOTION_CN[s.emotion.label]} × ${s.emotion.intensity.toFixed(1)}`;
      log(`🎤 "${s.text}" | 情绪 ${s.raw_label} → ${EMOTION_CN[s.emotion.label]} | 起轮 [trace ${s.trace_id}]`);
    })
    .catch(e => { $("micStatus").textContent = ""; log(`上传失败: ${e}`); });
}

/* 停止本地录音（不动 WS 连接），恢复按钮 */
function finishRecording() {
  if (micTimer) { clearInterval(micTimer); micTimer = null; }
  if (micProc) { try { micProc.disconnect(); } catch (e) {} micProc = null; }
  if (micSrc) { try { micSrc.disconnect(); } catch (e) {} micSrc = null; }
  if (micStream) { for (const t of micStream.getTracks()) t.stop(); micStream = null; }
  if (micCtx) { micCtx.close().catch(() => {}); micCtx = null; }
  micRecording = false;
  $("micBtn").textContent = "🎤 说话（点击开始/停止）";
  $("micBtn").style.color = "";
  $("micStatus").textContent = "";
}

function cleanupMic() {
  finishRecording();
  if (voiceWs) { try { voiceWs.close(); } catch (e) {} voiceWs = null; }
  voiceOpen = false;
  $("asrLive").textContent = "";
}

function mergeFloat32(chunks) {
  let n = 0;
  for (const c of chunks) n += c.length;
  const out = new Float32Array(n);
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

function resampleLinear(input, from, to) {
  if (from === to) return input;
  const ratio = from / to;
  const n = Math.floor(input.length / ratio);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const pos = i * ratio;
    const i0 = pos | 0;
    const frac = pos - i0;
    const a = input[i0] ?? 0, b = input[i0 + 1] ?? a;
    out[i] = a + (b - a) * frac;
  }
  return out;
}

function encodeWav16kMono(samples) {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const wstr = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  wstr(0, "RIFF"); v.setUint32(4, 36 + samples.length * 2, true); wstr(8, "WAVE");
  wstr(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, 16000, true); v.setUint32(28, 32000, true);
  v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  wstr(36, "data"); v.setUint32(40, samples.length * 2, true);
  let o = 44;
  for (let i = 0; i < samples.length; i++, o += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(o, s < 0 ? s * 32768 : s * 32767, true);
  }
  return buf;
}

/* ================= 控制 ================= */
function chat() {
  const text = $("chatText").value.trim() || "你好";
  fetch(`${API}/chat`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, llm: $("llmSel").value, tts: $("ttsSel").value }),
  })
    .then(r => r.json())
    .then(s => log(`已发起对话 [trace ${s.trace_id}] (llm=${$("llmSel").value}, tts=${$("ttsSel").value})`));
}
function bargeIn() {
  fetch(`${API}/barge_in`, { method: "POST" }).then(() => {
    stopAudio();
    frameQueue = [];
    wallOrigin = null;
    log("已发送打断（音画队列已清空）");
  });
}
function uploadAudio() {
  const file = $("audioFile").files[0];
  if (!file) { log("请先选择 wav 文件"); return; }
  const fd = new FormData();
  fd.append("file", file);
  fetch(`${API}/emotion/audio`, { method: "POST", body: fd })
    .then(r => r.json())
    .then(s => {
      if (!s.ok) { log(`识别失败: ${s.error}`); return; }
      log(`识别: "${s.text}" | 情绪 ${s.raw_label} → ${EMOTION_CN[s.emotion.label]}(×${(s.emotion.intensity).toFixed(1)})`);
      $("userEmotion").textContent = `当前用户情绪: ${EMOTION_CN[s.emotion.label]} × ${s.emotion.intensity.toFixed(1)}`;
    });
}
