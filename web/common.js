/* 渲染后端共享运行时：音频播放（TTS 时钟）+ DriveCommand WS 订阅 + 时间戳帧队列。
   VRM（vrm.js）与 Live2D（live2d.js）共用，保证各后端音画同步语义一致。 */
"use strict";

export const API = new URLSearchParams(location.search).get("api")
  || (location.origin.includes(":8080") ? "http://127.0.0.1:8765" : location.origin);
export const WS_BASE = API.replace(/^http/, "ws");

export function makeLog() {
  const el = document.getElementById("log");
  return (m) => {
    el.insertAdjacentHTML("beforeend", `<div>${new Date().toLocaleTimeString()} ${m}</div>`);
    el.scrollTop = el.scrollHeight;
  };
}

/* ================= 音频（TTS 输出时钟 = 唯一时间源） ================= */
export const audio = {
  ctx: null,
  queue: [], sources: [], lastEnd: 0, origin: null,
  msRecv: 0,

  enable() {
    if (!this.ctx) {
      this.ctx = new (window.AudioContext || window.webkitAudioContext)();
      const b = document.getElementById("soundBtn");
      b.textContent = "声音已启用 ✓"; b.disabled = true;
    }
    this.ctx.resume();
    return this.ctx.sampleRate;
  },

  push(pcmB64, sampleRate, durMs) {
    const bin = atob(pcmB64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    const pcm = new Int16Array(bytes.buffer);
    this.msRecv += durMs || 0;
    this.queue.push({ pcm, sr: sampleRate || 16000 });
  },

  schedule() {
    if (!this.ctx || !this.queue.length) return;
    const PREBUFFER_MS = 250;
    const now = this.ctx.currentTime;
    if (this.origin === null) {
      if (this.msRecv < PREBUFFER_MS) return;   // 攒够再起播
      this.origin = now + 0.05;
      this.lastEnd = this.origin;
    }
    while (this.queue.length) {
      const item = this.queue[0];
      const start = Math.max(this.lastEnd, now + 0.01);
      const buf = this.ctx.createBuffer(1, item.pcm.length, item.sr);
      const ch = buf.getChannelData(0);
      for (let i = 0; i < item.pcm.length; i++) ch[i] = item.pcm[i] / 32768;
      const src = this.ctx.createBufferSource();
      src.buffer = buf;
      src.connect(this.ctx.destination);
      src.start(start);
      this.sources.push(src);
      this.lastEnd = start + buf.duration;
      this.queue.shift();
    }
  },

  stop() {
    for (const s of this.sources) { try { s.stop(); } catch (e) {} }
    this.sources = []; this.queue = []; this.origin = null; this.lastEnd = 0;
  },

  /* 音频时钟（毫秒）；无音频时由帧队列退化为墙钟 */
  clockMs() {
    if (this.ctx && this.origin !== null) return Math.max(0, (this.ctx.currentTime - this.origin) * 1000);
    return -1;
  },
};

/* ================= 时间戳帧队列（落后时跳到最新帧） ================= */
export function makeFrameQueue() {
  let queue = [];
  let wallOrigin = null;
  return {
    push(f) {
      queue.push(f);
      queue.sort((a, b) => a.timestamp_ms - b.timestamp_ms);
      if (wallOrigin === null && audio.origin === null) wallOrigin = performance.now();
    },
    reset() { queue = []; wallOrigin = null; },
    takeLatest() {
      const now = audio.clockMs() >= 0 ? audio.clockMs()
        : (wallOrigin !== null ? performance.now() - wallOrigin : -1);
      if (now < 0) return null;
      let latest = null;
      while (queue.length && queue[0].timestamp_ms <= now + 50) latest = queue.shift();
      return latest;
    },
    pending: () => queue.length,
    clockMs: () => audio.clockMs() >= 0 ? audio.clockMs()
      : (wallOrigin !== null ? performance.now() - wallOrigin : -1),
  };
}

/* ================= WS 订阅（JSON 模式） + 控制 ================= */
export function connectDrive({ onQueue, onEvent, onStatus }) {
  function connect() {
    const ws = new WebSocket(`${WS_BASE}/ws/drive`);
    ws.onopen = () => {
      ws.send(JSON.stringify({ type: "mode", mode: "json" }));
      onStatus?.("connected");
    };
    ws.onmessage = (ev) => {
      let o; try { o = JSON.parse(ev.data); } catch { return; }
      switch (o.type) {
        case "turn_start":
          audio.stop(); onQueue?.reset?.();
          onEvent?.("turn_start", o);
          break;
        case "turn_end": onEvent?.("turn_end", o); break;
        case "error": onEvent?.("error", o); break;
        case "audio": audio.push(o.pcm_b64, o.sample_rate, o.duration_ms); break;
        default:
          if (o.timestamp_ms != null) onQueue?.push?.(o);
      }
    };
    ws.onclose = () => { onStatus?.("disconnected"); setTimeout(connect, 2000); };
  }
  connect();
}

export function postChat({ text, llm, tts, renderer }, log) {
  return fetch(`${API}/chat`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, llm, tts, renderer }),
  }).then(r => r.json()).then(s => log?.(`已发起对话 [trace ${s.trace_id}]`));
}

export function postBargeIn(log) {
  return fetch(`${API}/barge_in`, { method: "POST" })
    .then(r => r.json()).then(() => log?.("已打断（音画队列清空）"));
}
