/* VRM 渲染后端：Three.js + @pixiv/three-vrm 消费 DriveCommand 流。
   - WS JSON 模式订阅 /ws/drive（DriveCommand JSON 契约 = schema/drive.schema.json）
   - 音频块 WebAudio 实时播放（TTS 输出时钟 = 唯一时间源）
   - 驱动帧按音频时钟逐帧消费 → VRM 表情/viseme/头姿/程序化手势
   - 表情 preset 探测式映射（VRM0/VRM1 双格式兼容，缺则跳过） */
"use strict";

import * as THREE from "three";
import { GLTFLoader } from "./vendor/jsm/loaders/GLTFLoader.js";
import { VRMLoaderPlugin, VRMUtils } from "./vendor/three-vrm.module.min.js";

const API = new URLSearchParams(location.search).get("api")
  || (location.origin.includes(":8080") ? "http://127.0.0.1:8765" : location.origin);
const WS_BASE = API.replace(/^http/, "ws");
const MODEL_URL = new URLSearchParams(location.search).get("model")
  || "./vendor/VRM1_Constraint_Twist_Sample.vrm";

const $ = (id) => document.getElementById(id);
const log = (m) => {
  const el = $("log");
  el.insertAdjacentHTML("beforeend", `<div>${new Date().toLocaleTimeString()} ${m}</div>`);
  el.scrollTop = el.scrollHeight;
};

/* ================= Three.js 场景 ================= */
const canvas = $("canvas");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x17171f);

const camera = new THREE.PerspectiveCamera(30, innerWidth / innerHeight, 0.1, 50);
camera.position.set(0, 1.35, 2.6);
camera.lookAt(0, 1.0, 0);

scene.add(new THREE.AmbientLight(0xffffff, 0.9));
const dir = new THREE.DirectionalLight(0xffffff, 2.2);
dir.position.set(1.2, 2.4, 1.8);
scene.add(dir);
const rim = new THREE.DirectionalLight(0x88aaff, 0.8);
rim.position.set(-1.5, 1.8, -1.6);
scene.add(rim);

/* ================= VRM 加载 ================= */
let vrm = null;
const loader = new GLTFLoader();
loader.register((parser) => new VRMLoaderPlugin(parser));

/* 表情候选映射：DriveCommand.expression.name → VRM preset 候选（探测式） */
const EXPR_CANDIDATES = {
  smile: ["happy", "joy", "fun"],
  anger: ["angry", "angry01"],
  sad: ["sad", "sorrow"],
  fear: ["surprised"],          // VRM 无标准 fear：退化为惊讶低值
  disgust: ["disgusted", "angry"],
  melancholy: ["relaxed", "sorrow", "sad"],
  surprise: ["surprised"],
  neutral: [],
};
/* 视素映射：DriveCommand.phoneme → VRM viseme preset 候选（VRM1 aa…，VRM0 a…） */
const VISEME_MAP = {
  AA: ["aa", "a"], EH: ["ee", "e"], IH: ["ih", "i"], OW: ["oh", "o"], UW: ["ou", "u"],
  B: ["ih", "i"], D: ["ih", "i"], F: ["ih", "i"], K: ["ih", "i"], L: ["ih", "i"],
  M: ["ih", "i"], N: ["ih", "i"], P: ["ih", "i"], S: ["ih", "i"], SH: ["ih", "i"],
  T: ["ih", "i"], W: ["ou", "u"], Y: ["ih", "i"],
};
const VISEME_PRESETS = ["aa", "ih", "ou", "ee", "oh", "a", "i", "u", "e", "o"];

loader.load(
  MODEL_URL,
  (gltf) => {
    vrm = gltf.userData.vrm;
    VRMUtils.rotateVRM0(vrm);              // VRM0 朝向修正
    vrm.scene.traverse((o) => { if (o.isMesh) o.frustumCulled = false; });
    scene.add(vrm.scene);
    vrm.lookAt.target = camera;            // 目光跟随相机
    const em = vrm.expressionManager;
    const names = em ? Object.keys(em.expressionMap) : [];
    log(`VRM 加载完成，可用表情 preset: ${names.join(",") || "(无)"}`);
    $("modelState").textContent = `模型: ${MODEL_URL.split("/").pop()} | 表情 ${names.length} 种`;
    // 打印到控制台便于验收核对
    console.log("[VRM] expressions:", names);
  },
  (ev) => { $("modelState").textContent = `模型加载中… ${(ev.loaded / 1048576).toFixed(1)}MB`; },
  (err) => { log(`模型加载失败: ${err}`); $("modelState").textContent = "模型加载失败"; },
);

/* ================= 驱动状态（消费后的当前目标值） ================= */
const drive = {
  exprName: "neutral", exprWeight: 0,
  viseme: "",
  headYaw: 0, headPitch: 0, headRoll: 0,
  gesture: "idle",
};

/* ================= 音频播放（TTS 输出时钟） ================= */
let actx = null;
let audioQueue = [], scheduledSources = [], lastScheduleEnd = 0, playbackOrigin = null;
let audioMsRecv = 0;

function enableSound() {
  if (!actx) {
    actx = new (window.AudioContext || window.webkitAudioContext)();
    $("soundBtn").textContent = "声音已启用 ✓";
    $("soundBtn").disabled = true;
    log(`声音已启用（${actx.sampleRate}Hz）`);
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
  if (!actx || !audioQueue.length) return;
  const PREBUFFER_MS = 250;
  const now = actx.currentTime;
  if (playbackOrigin === null) {
    if (audioMsRecv < PREBUFFER_MS) return;
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
}

function stopAudio() {
  for (const s of scheduledSources) { try { s.stop(); } catch (e) {} }
  scheduledSources = []; audioQueue = []; playbackOrigin = null; lastScheduleEnd = 0;
}

let wallOrigin = null;
function clockMs() {
  if (actx && playbackOrigin !== null) return Math.max(0, (actx.currentTime - playbackOrigin) * 1000);
  if (wallOrigin !== null) return performance.now() - wallOrigin;
  return -1;
}

/* ================= 帧队列（时间戳有序消费） ================= */
let frameQueue = [], frameCount = 0;

function pushFrame(f) {
  frameQueue.push(f);
  frameQueue.sort((a, b) => a.timestamp_ms - b.timestamp_ms);
  if (wallOrigin === null && playbackOrigin === null) wallOrigin = performance.now();
}

function consumeFrames() {
  const now = clockMs();
  if (now < 0) return;
  let latest = null;
  while (frameQueue.length && frameQueue[0].timestamp_ms <= now + 50) {
    latest = frameQueue.shift();          // 跳帧策略：落后时直接消费到最新
  }
  if (latest) applyDrive(latest);
}

function applyDrive(f) {
  frameCount++;
  drive.exprName = f.expression?.name || "neutral";
  drive.exprWeight = Math.min(1, Math.max(0, f.expression?.weight ?? 0));
  drive.viseme = f.phoneme || "";
  drive.headYaw = f.head_pose?.yaw || 0;
  drive.headPitch = f.head_pose?.pitch || 0;
  drive.headRoll = f.head_pose?.roll || 0;
  const g = f.body_gesture?.name;
  if (g && g !== drive.gesture) {
    drive.gesture = g;
    gesture.start(g, performance.now());
  }
  $("hud").textContent =
    `时钟 ${now0()} | 帧 ${frameCount} | ts ${f.timestamp_ms}ms | ` +
    `${drive.exprName}×${(drive.exprWeight * 100) | 0}% | 口型 ${drive.viseme || "-"} | ` +
    `动作 ${drive.gesture}${f.trace_id ? " | trace " + f.trace_id : ""}`;
}
function now0() { const t = clockMs(); return t < 0 ? "-" : (t / 1000).toFixed(1) + "s"; }

/* ================= 程序化手势（normalized bones） ================= */
const GESTURE_DUR = { wave: 2.2, cheer: 2.0, nod: 1.2, bow: 2.4, soothe: 2.4,
  think: 2.6, applaud: 2.0, "fist_pump": 1.6, "lean_in": 2.4, encourage: 1.8, idle: Infinity };
const gesture = {
  name: "idle", t0: performance.now(),
  start(name, now) {
    gesture.name = name in GESTURE_DUR ? name : "idle";
    gesture.t0 = now;
  },
};
const ease = (t) => t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

/* 每帧计算：{bone: {x,y,z}} 欧拉角增量 */
function gesturePose(nowMs) {
  const g = gesture.name;
  const t = (nowMs - gesture.t0) / 1000;
  const dur = GESTURE_DUR[g] ?? 2.0;
  if (t > dur && g !== "idle") gesture.start("idle", nowMs);
  const p = g === "idle" ? 0 : ease(Math.min(1, t / 0.35)) * ease(Math.min(1, Math.max(0, (dur - t) / 0.5)));
  const out = {};
  const put = (bone, x = 0, y = 0, z = 0) => {
    const o = out[bone] || (out[bone] = { x: 0, y: 0, z: 0 });
    o.x += x * p; o.y += y * p; o.z += z * p;
  };
  const s = Math.sin(t * Math.PI * 2);
  switch (g) {
    case "idle":
      out.spine = { x: 0.02 * Math.sin(t * 1.4), z: 0.015 * Math.sin(t * 0.9) };
      out.leftUpperArm = { x: 0, y: 0, z: -0.05 - 0.02 * Math.sin(t * 1.4) };
      out.rightUpperArm = { x: 0, y: 0, z: 0.05 + 0.02 * Math.sin(t * 1.4 + 1) };
      break;
    case "wave":  // 右臂高举挥动
      put("rightUpperArm", 0, 0, 1.15);
      put("rightLowerArm", 0, 0, 0.45 + 0.35 * Math.sin(t * 6));
      put("leftUpperArm", 0, 0, 0.12);
      break;
    case "cheer": // 双臂上举挥动
      put("rightUpperArm", 0, 0, 1.25 + 0.12 * Math.sin(t * 5));
      put("leftUpperArm", 0, 0, -1.25 - 0.12 * Math.sin(t * 5 + 0.5));
      put("rightLowerArm", 0, 0, 0.35);
      put("leftLowerArm", 0, 0, -0.35);
      break;
    case "nod":   // 点头（叠加在头姿上）
      put("head", 0.16 * Math.max(0, Math.sin(t * Math.PI * 3)), 0, 0);
      break;
    case "bow":   // 鞠躬
      put("spine", 0.5 * Math.sin(Math.min(1, t / dur) * Math.PI), 0, 0);
      put("head", 0.2 * Math.sin(Math.min(1, t / dur) * Math.PI), 0, 0);
      break;
    case "soothe": // 安抚：双手前伸下压
      put("rightUpperArm", -0.35, 0, 0.55);
      put("leftUpperArm", -0.35, 0, -0.55);
      put("rightLowerArm", -0.5 + 0.08 * Math.sin(t * 3), 0, 0);
      put("leftLowerArm", -0.5 + 0.08 * Math.sin(t * 3 + 0.4), 0, 0);
      break;
    case "think": // 右手托腮
      put("rightUpperArm", -0.55, 0, 0.85);
      put("rightLowerArm", -1.25, 0.2, 0);
      put("head", 0.06, 0.1, -0.05);
      break;
    case "applaud": // 鼓掌
      put("rightUpperArm", -0.3, 0, 0.6);
      put("leftUpperArm", -0.3, 0, -0.6);
      put("rightLowerArm", 0, -0.5 + 0.25 * Math.abs(Math.sin(t * 9)), 0);
      put("leftLowerArm", 0, 0.5 - 0.25 * Math.abs(Math.sin(t * 9)), 0);
      break;
    case "fist_pump": // 握拳下压
      put("rightUpperArm", 0, 0, 1.1 - 0.3 * Math.sin(t * 5));
      put("rightLowerArm", 0, 0, 0.9 - 0.5 * Math.sin(t * 5));
      break;
    case "lean_in": // 前倾
      put("spine", 0.16, 0, 0);
      put("neck", 0.05, 0, 0);
      break;
    case "encourage": // 鼓励：双臂前摊
      put("rightUpperArm", -0.25, 0, 0.5);
      put("leftUpperArm", -0.25, 0, -0.5);
      put("rightLowerArm", -0.6, 0, 0.1 * Math.sin(t * 3));
      put("leftLowerArm", -0.6, 0, -0.1 * Math.sin(t * 3 + 0.5));
      break;
  }
  return out;
}

/* ================= 渲染主循环 ================= */
let lastRaf = performance.now();
let blinkNext = performance.now() + 2000;

function tick() {
  requestAnimationFrame(tick);
  const now = performance.now();
  const delta = Math.min(0.1, (now - lastRaf) / 1000);
  lastRaf = now;

  scheduleAudio();
  consumeFrames();

  if (vrm) {
    const em = vrm.expressionManager;
    if (em) {
      /* 表情：探测式 preset 映射，缺则跳过 */
      const cands = EXPR_CANDIDATES[drive.exprName] || [];
      const has = (n) => n && Object.prototype.hasOwnProperty.call(em.expressionMap, n);
      for (const key of Object.keys(EXPR_CANDIDATES)) {
        for (const n of EXPR_CANDIDATES[key]) {
          if (has(n)) em.setValue(n, 0);         // 先清全部候选
        }
      }
      for (const n of cands) {
        if (has(n)) {
          // fear/disgust 退化映射时降权，避免过火
          const w = drive.exprWeight * ((drive.exprName === "fear" || drive.exprName === "disgust") ? 0.6 : 1);
          em.setValue(n, w);
          break;
        }
      }
      /* 视素 */
      for (const v of VISEME_PRESETS) if (has(v)) em.setValue(v, 0);
      for (const vis of VISEME_MAP[drive.viseme] || []) {
        if (has(vis)) { em.setValue(vis, 0.65); break; }
      }
      /* 自动眨眼（自然感） */
      if (now > blinkNext) { blinkNext = now + 2200 + Math.random() * 2600; }
      const bt = now - (blinkNext - 2600);
      const blink = bt > 0 && bt < 130 ? Math.sin((bt / 130) * Math.PI) : 0;
      if (has("blink")) em.setValue("blink", blink);
    }

    /* 骨骼：头姿 + 手势叠加 */
    const h = vrm.humanoid;
    const gp = gesturePose(now);
    const headNode = h.getNormalizedBoneNode("head");
    if (headNode) {
      const go = gp.head || { x: 0, y: 0, z: 0 };
      headNode.rotation.set(
        THREE.MathUtils.degToRad(0) + drive.headPitch + go.x,
        drive.headYaw + go.y,
        drive.headRoll + go.z,
      );
    }
    const BONES = ["spine", "neck", "leftUpperArm", "rightUpperArm",
      "leftLowerArm", "rightLowerArm"];
    for (const b of BONES) {
      if (b === "head") continue;
      const node = h.getNormalizedBoneNode(b);
      if (!node) continue;
      const o = gp[b];
      node.rotation.set(o ? o.x : 0, o ? o.y : 0, o ? o.z : 0);
    }

    vrm.update(delta);
  }
  renderer.render(scene, camera);
}
tick();

addEventListener("resize", () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

/* 鼠标拖拽轨道视角（轻量实现） */
let dragging = false, px = 0, py = 0, camTheta = 0, camPhi = 0.15;
canvas.addEventListener("pointerdown", (e) => { dragging = true; px = e.clientX; py = e.clientY; });
addEventListener("pointerup", () => { dragging = false; });
addEventListener("pointermove", (e) => {
  if (!dragging) return;
  camTheta -= (e.clientX - px) * 0.005;
  camPhi = Math.max(-0.5, Math.min(0.8, camPhi + (e.clientY - py) * 0.004));
  px = e.clientX; py = e.clientY;
  const r = 2.6;
  camera.position.set(r * Math.sin(camTheta), 1.35 + r * Math.sin(camPhi), r * Math.cos(camTheta));
  camera.lookAt(0, 1.0, 0);
});

/* ================= WS 订阅 DriveCommand 流 ================= */
function connect() {
  const ws = new WebSocket(`${WS_BASE}/ws/drive`);
  ws.onopen = () => {
    ws.send(JSON.stringify({ type: "mode", mode: "json" }));
    $("conn").textContent = "(WS 已连接 · JSON 模式)";
    $("conn").style.color = "#9d9";
  };
  ws.onmessage = (ev) => {
    let o; try { o = JSON.parse(ev.data); } catch { return; }
    switch (o.type) {
      case "turn_start":
        stopAudio(); frameQueue = []; wallOrigin = null;
        log(`turn_start: ${o.text} [trace ${o.trace_id}]`);
        break;
      case "turn_end":
        log(`turn_end${o.summary ? "（全链路 " + (o.summary.turn_end_ms ?? "?") + "ms）" : ""}`);
        break;
      case "error": log(`ERROR: ${o.detail}`); break;
      case "audio":
        audioMsRecv += o.duration_ms || 0;
        audioQueue.push({ pcm: b64ToI16(o.pcm_b64), sampleRate: o.sample_rate || 16000 });
        if (wallOrigin === null && playbackOrigin === null) wallOrigin = performance.now();
        break;
      default:
        if (o.timestamp_ms != null) pushFrame(o);
    }
  };
  ws.onclose = () => { $("conn").textContent = "(WS 断开，2s 重连)"; $("conn").style.color = "#f77"; setTimeout(connect, 2000); };
}
connect();

/* ================= 控制 ================= */
function chat() {
  const text = $("chatText").value.trim() || "介绍一下你自己";
  fetch(`${API}/chat`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, llm: $("llmSel").value, tts: $("ttsSel").value, renderer: "vrm" }),
  }).then(r => r.json()).then(s => log(`已发起对话 [trace ${s.trace_id}]`));
}
function bargeIn() {
  fetch(`${API}/barge_in`, { method: "POST" }).then(() => {
    stopAudio(); frameQueue = []; wallOrigin = null;
    log("已打断（音画队列清空）");
  });
}

/* ES module 作用域 → 暴露给内联 onclick */
window.enableSound = enableSound;
window.chat = chat;
window.bargeIn = bargeIn;
