/* VRM 渲染后端：Three.js + @pixiv/three-vrm 消费 DriveCommand 流。
   音频/WS/帧队列等共享运行时见 common.js（与 Live2D 后端一致）。 */
"use strict";

import * as THREE from "three";
import { GLTFLoader } from "./vendor/jsm/loaders/GLTFLoader.js";
import { VRMLoaderPlugin, VRMUtils } from "./vendor/three-vrm.module.min.js";
import { API, makeLog, audio, makeFrameQueue, connectDrive, postChat, postBargeIn } from "./common.js";

const log = makeLog();
const MODEL_URL = new URLSearchParams(location.search).get("model")
  || "./vendor/VRM1_Constraint_Twist_Sample.vrm";

/* ================= Three.js 场景 ================= */
const canvas = document.getElementById("canvas");
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

const EXPR_CANDIDATES = {
  smile: ["happy", "joy", "fun"],
  anger: ["angry", "angry01"],
  sad: ["sad", "sorrow"],
  fear: ["surprised"],
  disgust: ["disgusted", "angry"],
  melancholy: ["relaxed", "sorrow", "sad"],
  surprise: ["surprised"],
  neutral: [],
};
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
    VRMUtils.rotateVRM0(vrm);
    vrm.scene.traverse((o) => { if (o.isMesh) o.frustumCulled = false; });
    scene.add(vrm.scene);
    vrm.lookAt.target = camera;
    const em = vrm.expressionManager;
    const names = em ? Object.keys(em.expressionMap) : [];
    log(`VRM 加载完成，可用表情 preset: ${names.join(",") || "(无)"}`);
    document.getElementById("modelState").textContent =
      `模型: ${MODEL_URL.split("/").pop()} | 表情 ${names.length} 种`;
    console.log("[VRM] expressions:", names);
  },
  (ev) => { document.getElementById("modelState").textContent = `模型加载中… ${(ev.loaded / 1048576).toFixed(1)}MB`; },
  (err) => { log(`模型加载失败: ${err}`); document.getElementById("modelState").textContent = "模型加载失败"; },
);

/* ================= 驱动状态 ================= */
const drive = { exprName: "neutral", exprWeight: 0, viseme: "",
  headYaw: 0, headPitch: 0, headRoll: 0, gesture: "idle" };

const frames = makeFrameQueue();
let frameCount = 0;

function applyDrive(f) {
  frameCount++;
  drive.exprName = f.expression?.name || "neutral";
  drive.exprWeight = Math.min(1, Math.max(0, f.expression?.weight ?? 0));
  drive.viseme = f.phoneme || "";
  drive.headYaw = f.head_pose?.yaw || 0;
  drive.headPitch = f.head_pose?.pitch || 0;
  drive.headRoll = f.head_pose?.roll || 0;
  const g = f.body_gesture?.name;
  if (g && g !== drive.gesture) { drive.gesture = g; gesture.start(g, performance.now()); }
  const t = frames.clockMs();
  document.getElementById("hud").textContent =
    `时钟 ${t < 0 ? "-" : (t / 1000).toFixed(1) + "s"} | 帧 ${frameCount} | ts ${f.timestamp_ms}ms | ` +
    `${drive.exprName}×${(drive.exprWeight * 100) | 0}% | 口型 ${drive.viseme || "-"} | ` +
    `动作 ${drive.gesture}${f.trace_id ? " | trace " + f.trace_id : ""}`;
}

/* ================= 程序化手势 ================= */
const GESTURE_DUR = { wave: 2.2, cheer: 2.0, nod: 1.2, bow: 2.4, soothe: 2.4,
  think: 2.6, applaud: 2.0, "fist_pump": 1.6, "lean_in": 2.4, encourage: 1.8, idle: Infinity };
const gesture = { name: "idle", t0: performance.now(),
  start(name, now) { gesture.name = name in GESTURE_DUR ? name : "idle"; gesture.t0 = now; } };
const ease = (t) => t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

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
  switch (g) {
    case "idle":
      out.spine = { x: 0.02 * Math.sin(t * 1.4), z: 0.015 * Math.sin(t * 0.9) };
      out.leftUpperArm = { x: 0, y: 0, z: -0.05 - 0.02 * Math.sin(t * 1.4) };
      out.rightUpperArm = { x: 0, y: 0, z: 0.05 + 0.02 * Math.sin(t * 1.4 + 1) };
      break;
    case "wave":
      put("rightUpperArm", 0, 0, 1.15);
      put("rightLowerArm", 0, 0, 0.45 + 0.35 * Math.sin(t * 6));
      put("leftUpperArm", 0, 0, 0.12);
      break;
    case "cheer":
      put("rightUpperArm", 0, 0, 1.25 + 0.12 * Math.sin(t * 5));
      put("leftUpperArm", 0, 0, -1.25 - 0.12 * Math.sin(t * 5 + 0.5));
      put("rightLowerArm", 0, 0, 0.35);
      put("leftLowerArm", 0, 0, -0.35);
      break;
    case "nod": put("head", 0.16 * Math.max(0, Math.sin(t * Math.PI * 3)), 0, 0); break;
    case "bow":
      put("spine", 0.5 * Math.sin(Math.min(1, t / dur) * Math.PI), 0, 0);
      put("head", 0.2 * Math.sin(Math.min(1, t / dur) * Math.PI), 0, 0);
      break;
    case "soothe":
      put("rightUpperArm", -0.35, 0, 0.55);
      put("leftUpperArm", -0.35, 0, -0.55);
      put("rightLowerArm", -0.5 + 0.08 * Math.sin(t * 3), 0, 0);
      put("leftLowerArm", -0.5 + 0.08 * Math.sin(t * 3 + 0.4), 0, 0);
      break;
    case "think":
      put("rightUpperArm", -0.55, 0, 0.85);
      put("rightLowerArm", -1.25, 0.2, 0);
      put("head", 0.06, 0.1, -0.05);
      break;
    case "applaud":
      put("rightUpperArm", -0.3, 0, 0.6);
      put("leftUpperArm", -0.3, 0, -0.6);
      put("rightLowerArm", 0, -0.5 + 0.25 * Math.abs(Math.sin(t * 9)), 0);
      put("leftLowerArm", 0, 0.5 - 0.25 * Math.abs(Math.sin(t * 9)), 0);
      break;
    case "fist_pump":
      put("rightUpperArm", 0, 0, 1.1 - 0.3 * Math.sin(t * 5));
      put("rightLowerArm", 0, 0, 0.9 - 0.5 * Math.sin(t * 5));
      break;
    case "lean_in": put("spine", 0.16, 0, 0); put("neck", 0.05, 0, 0); break;
    case "encourage":
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

  audio.schedule();
  const f = frames.takeLatest();
  if (f) applyDrive(f);

  if (vrm) {
    const em = vrm.expressionManager;
    if (em) {
      const has = (n) => n && Object.prototype.hasOwnProperty.call(em.expressionMap, n);
      for (const key of Object.keys(EXPR_CANDIDATES)) {
        for (const n of EXPR_CANDIDATES[key]) if (has(n)) em.setValue(n, 0);
      }
      for (const n of EXPR_CANDIDATES[drive.exprName] || []) {
        if (has(n)) {
          const w = drive.exprWeight * ((drive.exprName === "fear" || drive.exprName === "disgust") ? 0.6 : 1);
          em.setValue(n, w);
          break;
        }
      }
      for (const v of VISEME_PRESETS) if (has(v)) em.setValue(v, 0);
      for (const vis of VISEME_MAP[drive.viseme] || []) {
        if (has(vis)) { em.setValue(vis, 0.65); break; }
      }
      if (now > blinkNext) blinkNext = now + 2200 + Math.random() * 2600;
      const bt = now - (blinkNext - 2600);
      const blink = bt > 0 && bt < 130 ? Math.sin((bt / 130) * Math.PI) : 0;
      if (has("blink")) em.setValue("blink", blink);
    }

    const h = vrm.humanoid;
    const gp = gesturePose(now);
    const headNode = h.getNormalizedBoneNode("head");
    if (headNode) {
      const go = gp.head || { x: 0, y: 0, z: 0 };
      headNode.rotation.set(drive.headPitch + go.x, drive.headYaw + go.y, drive.headRoll + go.z);
    }
    for (const b of ["spine", "neck", "leftUpperArm", "rightUpperArm", "leftLowerArm", "rightLowerArm"]) {
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

/* 鼠标拖拽轨道视角 */
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

/* ================= WS + 控制 ================= */
const connEl = document.getElementById("conn");
connectDrive({
  onQueue: frames,
  onEvent: (type, o) => {
    if (type === "turn_start") log(`turn_start: ${o.text} [trace ${o.trace_id}]`);
    else if (type === "turn_end") log(`turn_end${o.summary ? "（全链路 " + (o.summary.turn_end_ms ?? "?") + "ms）" : ""}`);
    else if (type === "error") log(`ERROR: ${o.detail}`);
  },
  onStatus: (s) => {
    connEl.textContent = s === "connected" ? "(WS 已连接 · JSON 模式)" : "(WS 断开，2s 重连)";
    connEl.style.color = s === "connected" ? "#9d9" : "#f77";
  },
});

function chat() {
  const text = document.getElementById("chatText").value.trim() || "介绍一下你自己";
  postChat({ text, llm: llmSel.value, tts: ttsSel.value, renderer: "vrm" }, log);
}
function bargeIn() { postBargeIn(log).then(() => { audio.stop(); frames.reset(); }); }
function enableSound() { log(`声音已启用（${audio.enable()}Hz）`); }

window.enableSound = enableSound;
window.chat = chat;
window.bargeIn = bargeIn;
