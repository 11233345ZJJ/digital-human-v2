/* Live2D 渲染后端（VRM 的降级档）：pixi.js + pixi-live2d-display 消费 DriveCommand。
   - DriveCommand → Cubism 标准参数直驱（头姿角度/口型开合/表情 blend）
   - body_gesture → 模型 Tap motion 组触发（检索式，FORCE 优先级）
   - 禁用模型 Idle 组：头部/口型全部以 DriveCommand 为唯一驱动源
     （保留 physics 摆动 + EyeBlink 自动眨眼 + LipSync 组绑定）
   - 共享运行时（音频/WS/帧队列）见 common.js */
"use strict";

import { makeLog, audio, makeFrameQueue, connectDrive, postChat, postBargeIn } from "./common.js";

const log = makeLog();
const MODEL_URL = new URLSearchParams(location.search).get("model")
  || "./vendor/live2d/haru/haru_greeter_t03.model3.json";

/* ============ PIXI 应用（全局 PIXI 由 <script> 标签引入） ============ */
const app = new PIXI.Application({
  width: innerWidth, height: innerHeight,
  backgroundAlpha: 0, antialias: true,
  resolution: Math.min(devicePixelRatio, 2), autoDensity: true,
});
document.getElementById("stage").appendChild(app.view);
document.body.style.background =
  "radial-gradient(ellipse at 50% 30%, #23233a 0%, #14141c 70%)";

/* ============ 模型加载 ============ */
let model = null;
let core = null;        // cubism4 CoreModel（setParameterValueById）
const TAP_MOTIONS = 5;  // haru Tap 组动作数

const { Live2DModel } = PIXI.live2d;
Live2DModel.from(MODEL_URL, { autoInteract: false })
  .then((m) => {
    model = m;
    core = m.internalModel.coreModel;

    /* 禁用 Idle 自动组：头/身体由 DriveCommand 驱动，physics/眨眼保留 */
    try { delete m.internalModel.settings.motions.Idle; } catch (e) {}
    m.internalModel.motionManager.stopAllMotions();

    /* 舞台适配：等比缩放居中，头部在屏高 3/4 处 */
    const scale = Math.min(app.screen.width / m.internalModel.originalWidth,
                           app.screen.height / m.internalModel.originalHeight) * 0.92;
    m.scale.set(scale);
    m.anchor.set(0.5, 0.5);
    m.position.set(app.screen.width / 2, app.screen.height * 0.62);
    app.stage.addChild(m);

    const st = m.internalModel.settings;
    const motions = Object.values(st?.motions || {}).reduce((n, g) => n + g.length, 0);
    const exprs = (st?.expressions || []).length;
    log(`Live2D 加载完成（haru · Cubism4）：动作 ${motions} 个 / 表情 ${exprs} 个`);
    document.getElementById("modelState").textContent =
      `模型: ${MODEL_URL.split("/").pop()} | 动作 ${motions} | 表情 ${exprs}`;
  })
  .catch((e) => {
    log(`Live2D 加载失败: ${e?.message || e}`);
    document.getElementById("modelState").textContent = "模型加载失败";
  });

/* ============ 驱动状态 ============ */
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
  if (g && g !== drive.gesture) {
    drive.gesture = g;
    triggerGesture(g);
  }
  const t = frames.clockMs();
  document.getElementById("hud").textContent =
    `时钟 ${t < 0 ? "-" : (t / 1000).toFixed(1) + "s"} | 帧 ${frameCount} | ts ${f.timestamp_ms}ms | ` +
    `${drive.exprName}×${(drive.exprWeight * 100) | 0}% | 口型 ${drive.viseme || "-"} | ` +
    `动作 ${drive.gesture}${f.trace_id ? " | trace " + f.trace_id : ""}`;
}

/* 手势 → Tap motion 检索（模型动作库；nod 类微动作走参数直驱） */
const GESTURE_TO_MOTION = { wave: 0, cheer: 1, applaud: 2, bow: 3, encourage: 4,
  soothe: 0, think: 1, "fist_pump": 2, "lean_in": 4, applauds: 2 };
let nodUntil = 0;

function triggerGesture(name) {
  if (!model) return;
  if (name === "nod") { nodUntil = performance.now() + 1200; return; }
  const idx = GESTURE_TO_MOTION[name];
  if (idx === undefined) return;
  model.motion("Tap", idx % TAP_MOTIONS, 3 /* FORCE */)
    .catch(() => {});
}

/* ============ DriveCommand → Cubism 参数 ============ */
const D2R = 180 / Math.PI;
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

/* 视素 → 口型开合/嘴形 */
const MOUTH_OPEN = { AA: 1.0, OW: 0.85, UW: 0.6, EH: 0.5, IH: 0.35 };
const MOUTH_FORM = { AA: 0.6, EH: 0.4, IH: -0.1, OW: -0.5, UW: -0.7 };
/* 表情 → 参数增量 blend（weight 加权） */
const EXPR_PARAMS = {
  smile:     { ParamMouthForm: [0, 0.9], ParamEyeLSmile: [0, 1], ParamEyeRSmile: [0, 1], ParamCheek: [0, 0.6] },
  anger:     { ParamBrowForm: [0, -1], ParamBrowLY: [0, -0.4], ParamBrowRY: [0, -0.4], ParamEyeLOpen: [1, 0.6] },
  sad:       { ParamBrowForm: [0, -0.6], ParamEyeLOpen: [1, 0.55], ParamEyeROpen: [1, 0.55], ParamMouthForm: [0, -0.4] },
  fear:      { ParamBrowLY: [0, 0.8], ParamBrowRY: [0, 0.8], ParamEyeLOpen: [1, 1.3], ParamEyeROpen: [1, 1.3] },
  disgust:   { ParamBrowForm: [0, -0.8], ParamMouthForm: [0, -0.5], ParamEyeLOpen: [1, 0.5] },
  melancholy:{ ParamEyeLOpen: [1, 0.45], ParamEyeROpen: [1, 0.45], ParamBrowForm: [0, -0.3], ParamMouthForm: [0, -0.3] },
  surprise:  { ParamBrowLY: [0, 1], ParamBrowRY: [0, 1], ParamEyeLOpen: [1, 1.4], ParamEyeROpen: [1, 1.4], ParamMouthOpenY: [0, 0.35] },
};

function applyParams() {
  if (!core) return;
  const set = (id, v) => { try { core.setParameterValueById(id, v); } catch (e) {} };
  const w = drive.exprWeight;

  /* 头姿（弧度 → 度，±30 clamp）+ 身体联动 + 眼球 */
  set("ParamAngleX", clamp(drive.headYaw * D2R, -30, 30));
  set("ParamAngleY", clamp(drive.headPitch * D2R, -30, 30));
  set("ParamAngleZ", clamp(drive.headRoll * D2R, -30, 30));
  set("ParamBodyAngleX", clamp(drive.headYaw * D2R * 0.4, -10, 10));
  set("ParamBodyAngleY", clamp(drive.headPitch * D2R * 0.3, -10, 10));
  set("ParamEyeBallX", clamp(drive.headYaw * 2, -1, 1));
  set("ParamEyeBallY", clamp(-drive.headPitch * 2, -1, 1));

  /* 口型：视素开合 + 嘴形；无音素时闭嘴（呼吸微动） */
  const now = performance.now();
  const open = MOUTH_OPEN[drive.viseme] ?? (drive.viseme ? 0.15 : 0);
  const form = MOUTH_FORM[drive.viseme] ?? 0;
  set("ParamMouthOpenY", open);
  set("ParamMouthForm", form);

  /* 表情 blend：base + (target-base)*weight */
  const expr = EXPR_PARAMS[drive.exprName];
  if (expr) {
    for (const [id, [base, target]] of Object.entries(expr)) {
      set(id, base + (target - base) * w);
    }
  }

  /* nod 微动作（参数直驱） */
  if (now < nodUntil) {
    const t = (nodUntil - now) / 1200;
    set("ParamAngleY", clamp(drive.headPitch * D2R - 14 * Math.sin(t * Math.PI * 3), -30, 30));
  }
}

/* 参数覆盖点：renderer prerender（每帧、所有 motion/physics 更新之后、绘制之前） */
app.renderer.on("prerender", applyParams);

/* ============ 主循环：音频调度 + 帧消费 ============ */
app.ticker.add(() => {
  audio.schedule();
  const f = frames.takeLatest();
  if (f) applyDrive(f);
});

addEventListener("resize", () => {
  app.renderer.resize(innerWidth, innerHeight);
  if (model) {
    const scale = Math.min(app.screen.width / model.internalModel.originalWidth,
                           app.screen.height / model.internalModel.originalHeight) * 0.92;
    model.scale.set(scale);
    model.position.set(app.screen.width / 2, app.screen.height * 0.62);
  }
});

/* ============ WS + 控制 ============ */
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
  postChat({ text, llm: llmSel.value, tts: ttsSel.value, renderer: "live2d" }, log);
}
function bargeIn() { postBargeIn(log).then(() => { audio.stop(); frames.reset(); }); }
function enableSound() { log(`声音已启用（${audio.enable()}Hz）`); }

window.enableSound = enableSound;
window.chat = chat;
window.bargeIn = bargeIn;
