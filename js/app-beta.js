import { Input, BlobSource, ALL_FORMATS, CanvasSink, VideoSampleSink, AudioBufferSink, Output, Mp4OutputFormat, BufferTarget, CanvasSource, AudioBufferSource, canEncodeAudio, canEncodeVideo, QUALITY_HIGH } from '../vendor/mediabunny.min.mjs';
import { solveTrack, cameraPath } from './track.js';
import { onsetEnvelope, align, renderAligned, toMono } from './audio-beta.js';

const $ = id => document.getElementById(id);
const Q = new URLSearchParams(location.search);
const MODEL = 'lite';
const RUNMODE = Q.get('rm') === 'video' ? 'VIDEO' : 'IMAGE';
const S = {
  file: null, input: null, vt: null, at: null, W: 0, H: 0, t0: 0, dur: 0,
  samples: [], thumbs: [], thumbW: 0, thumbH: 0, anchors: new Map(), anchorHist: [], track: null,
  focus: 'loose', view: 'full', cur: 0, path: null, pathStep: 1 / 60,
  origAudio: null, // {sr, ch:[Float32Array]} 对齐到视频时间 0..dur
  ref: null, refEnv: null, vidEnv: null, alignRes: null, nudge: 0, amode: 'ref', aligned: null,
  res: 720, outBlob: null,
};
window.__S = S; // 调试用

// ---------- 工具 ----------
function status(el, html, p) {
  el.innerHTML = html + (p != null ? `<div class="bar"><i style="width:${Math.round(p * 100)}%"></i></div>` : '');
}
const sleep = ms => new Promise(r => setTimeout(r, ms));
const fmt = t => t.toFixed(1) + 's';

// ---------- 能力检测 ----------
const caps = {
  webcodecs: typeof VideoEncoder !== 'undefined' && typeof VideoDecoder !== 'undefined',
  audioEnc: typeof AudioEncoder !== 'undefined',
  share: !!navigator.canShare,
};
$('caps').textContent = `模型:${MODEL} WebCodecs:${caps.webcodecs ? '✓' : '✗'} 音频编码:${caps.audioEnc ? '✓' : '✗'}`;

// ---------- 姿态模型（本地托管，不走 Google CDN） ----------
let landmarkerP = null, tsCounter = 0;
function getLandmarker() {
  if (!landmarkerP) landmarkerP = (async () => {
    const { FilesetResolver, PoseLandmarker, ObjectDetector } = await import('../vendor/vision_bundle.mjs');
    const base = new URL('../vendor/wasm', import.meta.url).href;
    const fileset = await FilesetResolver.forVisionTasks(base);
    const pose = await PoseLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: new URL(`../models/pose_landmarker_${MODEL}.task`, import.meta.url).href, delegate: 'CPU' },
      runningMode: RUNMODE, numPoses: 6,
      minPoseDetectionConfidence: 0.35, minPosePresenceConfidence: 0.35, minTrackingConfidence: 0.35,
    });
    const det = await ObjectDetector.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: new URL('../models/efficientdet_lite0.tflite', import.meta.url).href, delegate: 'CPU' },
      runningMode: 'IMAGE', scoreThreshold: 0.25, categoryAllowlist: ['person'], maxResults: 10,
    });
    return { pose, det };
  })();
  return landmarkerP;
}
// 页面打开就开始预加载模型
getLandmarker().catch(e => console.warn('预加载模型失败', e));

// ---------- 颜色直方图（上衣 + 下装），用来区分穿着不同的人 ----------
function regionHist(data, w, h, x0, y0, x1, y1, out, off) {
  x0 = Math.max(0, Math.floor(x0)); y0 = Math.max(0, Math.floor(y0));
  x1 = Math.min(w, Math.ceil(x1)); y1 = Math.min(h, Math.ceil(y1));
  let n = 0;
  const step = Math.max(1, Math.floor(Math.sqrt(((x1 - x0) * (y1 - y0)) / 400)));
  for (let y = y0; y < y1; y += step) for (let x = x0; x < x1; x += step) {
    const i = (y * w + x) * 4;
    const b = (data[i] >> 6) * 16 + (data[i + 1] >> 6) * 4 + (data[i + 2] >> 6);
    out[off + b]++; n++;
  }
  for (let k = 0; k < 64; k++) out[off + k] = n ? out[off + k] / n : 1 / 64;
}

function poseBox(lm) { // 归一化 -> 全分辨率像素框 + 关键点
  const P = i => [lm[i].x * S.W, lm[i].y * S.H];
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i < lm.length; i++) {
    const v = lm[i].visibility ?? 1; if (v < 0.15 && i > 10) continue;
    const [x, y] = P(i); x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y);
  }
  const nose = P(0), sh = [(P(11)[0] + P(12)[0]) / 2, (P(11)[1] + P(12)[1]) / 2], hip = [(P(23)[0] + P(24)[0]) / 2, (P(23)[1] + P(24)[1]) / 2];
  y0 = Math.min(y0, nose[1] - 0.55 * Math.max(5, sh[1] - nose[1]));
  const h = y1 - y0; y1 += 0.02 * h; x0 -= 0.05 * h; x1 += 0.05 * h;
  return { box: [Math.max(0, x0), Math.max(0, y0), Math.min(S.W, x1), Math.min(S.H, y1)], sh, hip };
}
function boxHist(img, cw, ch, box) { // 上衣区 + 下装区的颜色分布（按框比例取，保证同一个人前后一致）
  const sx = cw / S.W, sy = ch / S.H, hist = new Float32Array(128);
  const [x0, y0, x1, y1] = box, w = x1 - x0, h = y1 - y0;
  regionHist(img, cw, ch, (x0 + 0.3 * w) * sx, (y0 + 0.2 * h) * sy, (x1 - 0.3 * w) * sx, (y0 + 0.45 * h) * sy, hist, 0);
  regionHist(img, cw, ch, (x0 + 0.25 * w) * sx, (y0 + 0.55 * h) * sy, (x1 - 0.25 * w) * sx, (y0 + 0.9 * h) * sy, hist, 64);
  return hist;
}
let tileCanvas = null;
function detectFrame(models, c, cw, ch, img) {
  const k = S.W / cw;
  // 横屏视频分块检测（后排的人在整幅缩小后太小，容易漏检）
  const ar = Math.max(cw, ch) / Math.min(cw, ch);
  const nT = cw > ch ? (ar > 1.5 ? 3 : ar > 1.15 ? 2 : 1) : 1;
  const raw = [];
  if (nT === 1) {
    for (const d of models.det.detect(c).detections || []) { const b = d.boundingBox; raw.push({ box: [b.originX, b.originY, b.originX + b.width, b.originY + b.height], score: d.categories[0].score }); }
  } else {
    if (!tileCanvas) { tileCanvas = document.createElement('canvas'); tileCanvas.width = tileCanvas.height = 384; }
    const tctx = tileCanvas.getContext('2d');
    for (let t = 0; t < nT; t++) {
      const x0 = Math.round((cw - ch) * t / (nT - 1)), sc = ch / 384;
      tctx.drawImage(c, x0, 0, ch, ch, 0, 0, 384, 384);
      for (const d of models.det.detect(tileCanvas).detections || []) {
        const b = d.boundingBox, bx = [x0 + b.originX * sc, b.originY * sc, x0 + (b.originX + b.width) * sc, (b.originY + b.height) * sc];
        // 被块边缘切开的半个人丢掉（相邻块里有完整的）
        if ((t > 0 && bx[0] < x0 + 0.02 * ch) || (t < nT - 1 && bx[2] > x0 + 0.98 * ch)) continue;
        raw.push({ box: bx, score: d.categories[0].score });
      }
    }
  }
  const dets = raw.map(r => ({ box: r.box.map(v => v * k), score: r.score }));
  const pr = RUNMODE === 'VIDEO' ? models.pose.detectForVideo(c, (tsCounter += 100)) : models.pose.detect(c);
  const poses = (pr.landmarks || []).map(poseBox).filter(p => p.box[2] - p.box[0] > 4 && p.box[3] - p.box[1] > 10);
  for (const p of poses) {
    let bi = -1, bv = 0.6;
    dets.forEach((d, j) => { const v = contain(p.box, d.box); if (v > bv) { bv = v; bi = j; } });
    if (bi >= 0) { if (!dets[bi].sh) { dets[bi].sh = p.sh; dets[bi].hip = p.hip; } }
    else dets.push({ box: p.box, sh: p.sh, hip: p.hip, score: 0.3 });
  }
  return nms(dets).map(d => ({ box: d.box, sh: d.sh || null, hip: d.hip || null, hist: boxHist(img, cw, ch, d.box) }));
}
function contain(a, b) { // a 有多少比例落在 b 里面
  const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])), iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  return ix * iy / Math.max(1, (a[2] - a[0]) * (a[3] - a[1]));
}
function nms(dets) {
  dets.sort((a, b) => b.score - a.score);
  const keep = [];
  for (const d of dets) {
    const dup = keep.findIndex(k => {
      if (iouB(k.box, d.box) > 0.5) return true;
      // 同一个人的半身框套在全身框里
      const [sm, bg] = (d.box[3] - d.box[1]) < (k.box[3] - k.box[1]) ? [d.box, k.box] : [k.box, d.box];
      return contain(sm, bg) > 0.85 && Math.abs((sm[0] + sm[2]) - (bg[0] + bg[2])) / 2 < 0.25 * (bg[2] - bg[0]);
    });
    if (dup >= 0) {
      const k = keep[dup];
      if (!k.sh && d.sh) { k.sh = d.sh; k.hip = d.hip; }
      if ((d.box[3] - d.box[1]) > (k.box[3] - k.box[1]) && contain(k.box, d.box) > 0.85) k.box = d.box;
      continue;
    }
    keep.push(d);
  }
  return keep;
}
function iouB(a, b) { const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])), iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1])); const I = ix * iy; return I / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - I); }

// ---------- 1. 导入 + 分析 ----------
$('file').addEventListener('change', async e => {
  const f = e.target.files[0]; if (!f) return;
  e.target.value = '';
  try { await importVideo(f); } catch (err) { console.error(err); status($('import-status'), `<span class="err">出错了：${err.message || err}</span>`); }
});

async function importVideo(file) {
  resetAll();
  S.file = file;
  const st = $('import-status');
  status(st, '读取视频…');
  const input = new Input({ source: new BlobSource(file), formats: ALL_FORMATS });
  const vt = await input.getPrimaryVideoTrack();
  if (!vt) throw new Error('没找到视频轨道');
  if (!(await vt.canDecode())) throw new Error(`这个浏览器解码不了该视频（编码：${vt.codec}）。请更新 iOS，或在相机设置里改成“兼容性最佳”再录。`);
  S.input = input; S.vt = vt;
  S.at = await input.getPrimaryAudioTrack();
  S.W = vt.displayWidth; S.H = vt.displayHeight;
  S.t0 = await vt.getFirstTimestamp();
  S.dur = (await vt.computeDuration()) - S.t0;
  if (S.dur > 75) status(st, `<span class="warn">视频 ${fmt(S.dur)}，比较长，处理会慢一些。</span>`);
  const v = $('srcvideo'); v.src = URL.createObjectURL(file); v.load();

  status(st, '加载识别模型（首次约 10–20 秒，之后会缓存）…', 0);
  const lmk = await getLandmarker();

  const step = Math.max(0.125, S.dur / 400);
  const times = []; for (let t = 0; t < S.dur - 0.02; t += step) times.push(S.t0 + t);
  const scale = 512 / Math.max(S.W, S.H);
  const cw = Math.round(S.W * scale), ch = Math.round(S.H * scale);
  S.thumbW = cw; S.thumbH = ch;
  const sink = new CanvasSink(vt, { width: cw, height: ch, fit: 'fill', poolSize: 2 });
  const tStart = performance.now();
  let i = 0;
  for await (const wc of sink.canvasesAtTimestamps(times)) {
    if (!wc) { i++; continue; }
    const c = wc.canvas, ctx = c.getContext('2d', { willReadFrequently: true });
    const img = ctx.getImageData(0, 0, cw, ch).data;
    const dets = detectFrame(lmk, c, cw, ch, img);
    S.samples.push({ t: wc.timestamp, dets });
    S.thumbs.push(await new Promise(r => c.toBlob(r, 'image/jpeg', 0.7))); // 存成 JPEG，省内存
    i++;
    const el = (performance.now() - tStart) / 1000;
    status(st, `识别画面里的人… ${i}/${times.length}（已用 ${el.toFixed(0)} 秒${i > 5 ? `，预计还要 ${Math.max(1, Math.round(el / i * (times.length - i)))} 秒` : ''}）\n处理时请保持本页面在前台、不要锁屏`, i / times.length);
  }
  const people = S.samples.reduce((m, s) => Math.max(m, s.dets.length), 0);
  status(st, `<span class="ok">分析完成</span>：${fmt(S.dur)}，${S.W}×${S.H}，最多识别到 ${people} 人，用时 ${((performance.now() - tStart) / 1000).toFixed(0)} 秒。`);
  // 原声
  loadOrigAudio().catch(e => console.warn('原声读取失败', e));
  ['s-select', 's-focus', 's-audio', 's-export'].forEach(id => $(id).hidden = false);
  // 单人自动选中：找第一个只有一个人的帧
  const solo = S.samples.findIndex(s => s.dets.length === 1);
  if (people === 1 && solo >= 0) addAnchor(solo, 0);
  seekTo(solo >= 0 && people === 1 ? solo : S.samples.findIndex(s => s.dets.length > 0));
  updateAnchorsUI();
  $('s-select').scrollIntoView({ behavior: 'smooth' });
}

function resetAll() {
  S.samples = []; S.thumbs = []; thumbCache.clear(); S.anchors = new Map(); S.anchorHist = [];
  S.track = null; S.path = null; S.origAudio = null; S.vidEnv = null; S.alignRes = null; S.aligned = null; S.outBlob = null;
  $('result').hidden = true; $('export-status').innerHTML = ''; $('align-status').innerHTML = '';
}

async function loadOrigAudio() {
  if (!S.at || !(await S.at.canDecode())) return;
  const sr = S.at.sampleRate, nch = Math.min(2, S.at.numberOfChannels);
  const len = Math.ceil(S.dur * sr);
  const ch = Array.from({ length: nch }, () => new Float32Array(len));
  const sink = new AudioBufferSink(S.at);
  for await (const { buffer, timestamp } of sink.buffers()) {
    const off = Math.round((timestamp - S.t0) * sr);
    for (let c = 0; c < nch; c++) {
      const src = buffer.getChannelData(Math.min(c, buffer.numberOfChannels - 1));
      const dst = ch[c];
      for (let i = 0; i < src.length; i++) { const k = off + i; if (k >= 0 && k < len) dst[k] = src[i]; }
    }
  }
  S.origAudio = { sr, ch };
  updateAlignBtn();
}

// ---------- 2. 选人 / 预览 ----------
const view = $('view'), vctx = view.getContext('2d');
// 明确设定画布显示尺寸（等比），保证点击坐标和画面像素一一对应
function fitView() {
  const maxW = view.parentElement.clientWidth || 360, maxH = window.innerHeight * 0.62;
  const k = Math.min(maxW / view.width, maxH / view.height);
  const w = Math.round(view.width * k) + 'px', h = Math.round(view.height * k) + 'px';
  if (view.style.width !== w) view.style.width = w;
  if (view.style.height !== h) view.style.height = h;
}
window.addEventListener('resize', () => S.samples.length && fitView());
function sampleAt(t) { // 轨道时间 -> 最近 sample 下标
  const s = S.samples; if (!s.length) return 0;
  let lo = 0, hi = s.length - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (s[m].t <= t) lo = m; else hi = m; }
  return Math.abs(s[hi].t - t) < Math.abs(s[lo].t - t) ? hi : lo;
}
function seekTo(i) {
  i = Math.max(0, Math.min(S.samples.length - 1, i)); S.cur = i;
  $('scrub').value = Math.round(1000 * (S.samples[i].t - S.t0) / S.dur);
  $('tlabel').textContent = fmt(S.samples[i].t - S.t0);
  draw();
}
function ensurePath() {
  if (!S.track) { S.path = null; return; }
  if (S.path && S.path.focus === S.focus) return;
  const times = []; for (let t = 0; t <= S.dur + 1e-6; t += S.pathStep) times.push(S.t0 + t);
  S.path = cameraPath(S.samples, S.track, S.focus, S.W, S.H, times);
  S.path.focus = S.focus;
}
function cropAt(t) { ensurePath(); if (!S.path) return null; const k = Math.max(0, Math.min(S.path.length - 1, Math.round((t - S.t0) / S.pathStep))); return S.path[k]; }

const thumbCache = new Map();
async function getThumb(i) {
  if (thumbCache.has(i)) return thumbCache.get(i);
  const bmp = await createImageBitmap(S.thumbs[i]);
  thumbCache.set(i, bmp);
  if (thumbCache.size > 12) { const [k, v] = thumbCache.entries().next().value; v.close && v.close(); thumbCache.delete(k); }
  return bmp;
}
let drawSeq = 0;
function draw(videoEl) {
  if (!S.samples.length) return;
  if (!videoEl) { const seq = ++drawSeq, i = S.cur; getThumb(i).then(b => { if (seq === drawSeq && !playing) drawFrame(null, b); }); return; }
  drawFrame(videoEl, null);
}
function drawFrame(videoEl, thumb) {
  const t = videoEl ? videoEl.currentTime + S.t0 : S.samples[S.cur].t;
  const i = videoEl ? sampleAt(t) : S.cur;
  const src = videoEl || thumb;
  const srcW = videoEl ? videoEl.videoWidth : src.width, srcH = videoEl ? videoEl.videoHeight : src.height;
  const k = srcW / S.W;
  if (S.view === 'crop' && S.track) {
    const c = cropAt(t);
    const VW = 360, VH = 640;
    if (view.width !== VW || view.height !== VH) { view.width = VW; view.height = VH; }
    fitView();
    vctx.drawImage(src, c.x * k, c.y * k, c.w * k, c.h * k, 0, 0, VW, VH);
    $('viewtip').textContent = '直拍预览';
    return;
  }
  const VW = Math.min(720, S.W), VH = Math.round(VW * S.H / S.W);
  if (view.width !== VW || view.height !== VH) { view.width = VW; view.height = VH; }
  fitView();
  vctx.drawImage(src, 0, 0, srcW, srcH, 0, 0, VW, VH);
  const s = VW / S.W;
  const sel = S.track ? S.track[i] : null;
  vctx.lineWidth = Math.max(2, VW / 240);
  S.samples[i].dets.forEach((d, j) => {
    const isSel = sel && sel.det === j;
    vctx.strokeStyle = isSel ? '#ff4f9a' : 'rgba(255,255,255,.55)';
    vctx.setLineDash(isSel ? [] : [6, 6]);
    vctx.strokeRect(d.box[0] * s, d.box[1] * s, (d.box[2] - d.box[0]) * s, (d.box[3] - d.box[1]) * s);
  });
  vctx.setLineDash([]);
  if (sel && sel.det < 0) { // 丢失时显示预测位置
    vctx.strokeStyle = 'rgba(255,79,154,.6)'; vctx.setLineDash([3, 5]);
    vctx.strokeRect(sel.box[0] * s, sel.box[1] * s, (sel.box[2] - sel.box[0]) * s, (sel.box[3] - sel.box[1]) * s); vctx.setLineDash([]);
  }
  if (S.track) { const c = cropAt(t); vctx.strokeStyle = '#ffd60a'; vctx.lineWidth = 1.5; vctx.strokeRect(c.x * s, c.y * s, c.w * s, c.h * s); }
  if (S.anchors.has(i)) { vctx.fillStyle = '#ff4f9a'; vctx.font = `${Math.round(VW / 30)}px sans-serif`; vctx.fillText('📍 点选帧', 10, VH - 12); }
  $('viewtip').textContent = !S.track ? '点你自己' : (sel && sel.det < 0 ? '这一刻没识别到你（被挡住？），用预测位置' : '');
}

$('scrub').addEventListener('input', e => { stopPlay(); const t = S.t0 + (e.target.value / 1000) * S.dur; seekTo(sampleAt(t)); });
view.addEventListener('click', e => {
  if (S.view !== 'full' || !S.samples.length) return;
  stopPlay();
  const r = view.getBoundingClientRect();
  const x = (e.clientX - r.left) / r.width * S.W, y = (e.clientY - r.top) / r.height * S.H;
  const dets = S.samples[S.cur].dets;
  let best = -1, bd = Infinity;
  dets.forEach((d, j) => {
    const inside = x >= d.box[0] && x <= d.box[2] && y >= d.box[1] && y <= d.box[3];
    const dist = Math.hypot(x - (d.box[0] + d.box[2]) / 2, y - (d.box[1] + d.box[3]) / 2) / (d.box[3] - d.box[1]);
    const score = inside ? dist : dist + 1;
    if (score < bd) { bd = score; best = j; }
  });
  if (best < 0 || bd > 1.2) { $('viewtip').textContent = '这一帧没识别到人，换一帧试试'; return; }
  addAnchor(S.cur, best);
});
function addAnchor(i, j) {
  S.anchors.set(i, j); S.anchorHist.push(i);
  S.track = solveTrack(S.samples, S.anchors); S.path = null;
  updateAnchorsUI(); draw();
}
$('undo').addEventListener('click', () => {
  const i = S.anchorHist.pop(); if (i == null) return;
  S.anchors.delete(i);
  S.track = S.anchors.size ? solveTrack(S.samples, S.anchors) : null; S.path = null;
  updateAnchorsUI(); draw();
});
function updateAnchorsUI() {
  const n = S.anchors.size;
  $('anchors').textContent = n ? `已点选 ${n} 次` + (S.track ? `，跟拍覆盖 ${Math.round(100 * S.track.filter(x => x.det >= 0).length / S.track.length)}% 的画面` : '') : '还没选人';
  $('undo').hidden = !n;
  const W = $('warns'); W.innerHTML = '';
  if (S.track && S.track.warnings && S.track.warnings.length) {
    const lab = document.createElement('div'); lab.className = 'hint'; lab.style.width = '100%'; lab.style.margin = '0';
    lab.textContent = '这些时刻可能有问题，点一下跳过去看看，跟错了就再点一次你自己：'; W.appendChild(lab);
    for (const w of S.track.warnings.slice(0, 12)) {
      const b = document.createElement('button'); b.className = 'w ' + w.kind;
      b.textContent = (w.kind === 'lost' ? '没找到你 ' : '⚠️ 旁边有很像的人 ') + (w.t0 - S.t0).toFixed(1) + 's';
      b.onclick = () => { stopPlay(); if (S.view !== 'full') document.querySelector('#viewmode [data-v=full]').click(); seekTo(w.i); };
      W.appendChild(b);
    }
  }
  $('export').disabled = !S.track;
}
document.querySelectorAll('#viewmode button').forEach(b => b.addEventListener('click', () => {
  if (b.dataset.v === 'crop' && !S.track) { $('viewtip').textContent = '先点一下你自己'; return; }
  document.querySelectorAll('#viewmode button').forEach(x => x.classList.toggle('on', x === b));
  S.view = b.dataset.v; draw(playing ? $('srcvideo') : null);
}));
document.querySelectorAll('#focus button').forEach(b => b.addEventListener('click', () => {
  document.querySelectorAll('#focus button').forEach(x => x.classList.toggle('on', x === b));
  S.focus = b.dataset.f; S.path = null; draw();
}));

// 播放（带对齐音频预览）
let playing = false, rafId = 0, audioCtx = null, audioNode = null;
async function startPlay() {
  const v = $('srcvideo');
  v.currentTime = Math.max(0, S.samples[S.cur].t - S.t0);
  const useRef = S.aligned && S.amode !== 'orig';
  v.muted = !!(useRef && S.amode === 'ref');
  v.volume = S.amode === 'mix' ? 0.4 : 1;
  playing = true; $('play').textContent = '❚❚ 暂停';
  if (useRef) {
    audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    await audioCtx.resume();
  }
  await v.play().catch(err => { console.warn(err); v.muted = true; return v.play(); });
  if (useRef) {
    const buf = audioCtx.createBuffer(S.aligned.ch.length, S.aligned.ch[0].length, S.aligned.sr);
    S.aligned.ch.forEach((c, k) => buf.copyToChannel(c, k));
    audioNode = audioCtx.createBufferSource(); audioNode.buffer = buf; audioNode.connect(audioCtx.destination);
    audioNode.start(0, v.currentTime);
  }
  const loop = () => { if (!playing) return; draw(v); const i = sampleAt(v.currentTime + S.t0); S.cur = i; $('scrub').value = Math.round(1000 * v.currentTime / S.dur); $('tlabel').textContent = fmt(v.currentTime); if (v.ended) { stopPlay(); return; } rafId = requestAnimationFrame(loop); };
  loop();
}
function stopPlay() {
  if (!playing) return;
  playing = false; cancelAnimationFrame(rafId); $('play').textContent = '▶︎ 播放';
  $('srcvideo').pause();
  if (audioNode) { try { audioNode.stop(); } catch { } audioNode = null; }
  draw();
}
$('play').addEventListener('click', () => playing ? stopPlay() : startPlay());

// ---------- 4. 音频对齐 ----------
$('afile').addEventListener('change', async e => {
  const f = e.target.files[0]; if (!f) return; e.target.value = '';
  $('afile-name').textContent = '读取中… ' + f.name;
  try {
    S.ref = await decodeAnyAudio(f); S.refEnv = null; S.alignRes = null; S.aligned = null;
    $('afile-name').textContent = `✓ ${f.name}（${fmt(S.ref.ch[0].length / S.ref.sr)}）`;
  } catch (err) { console.error(err); $('afile-name').innerHTML = `<span class="err">读不了这个文件：${err.message || err}</span>`; S.ref = null; }
  updateAlignBtn();
});
async function decodeAnyAudio(f) {
  const buf = await f.arrayBuffer();
  try {
    const ctx = new (window.OfflineAudioContext || window.webkitOfflineAudioContext)(2, 44100, 44100);
    const ab = await new Promise((res, rej) => { const p = ctx.decodeAudioData(buf.slice(0), res, rej); if (p && p.then) p.then(res, rej); });
    return { sr: ab.sampleRate, ch: Array.from({ length: Math.min(2, ab.numberOfChannels) }, (_, i) => ab.getChannelData(i)) };
  } catch (e) {
    // 再试：当作视频文件取音轨
    const input = new Input({ source: new BlobSource(f), formats: ALL_FORMATS });
    const at = await input.getPrimaryAudioTrack(); if (!at) throw new Error('文件里没有音频');
    const parts = []; let sr = at.sampleRate, nch = Math.min(2, at.numberOfChannels), total = 0;
    for await (const { buffer } of new AudioBufferSink(at).buffers()) { parts.push(buffer); total += buffer.length; }
    const ch = Array.from({ length: nch }, () => new Float32Array(total));
    let o = 0; for (const b of parts) { for (let c = 0; c < nch; c++) ch[c].set(b.getChannelData(Math.min(c, b.numberOfChannels - 1)), o); o += b.length; }
    return { sr, ch };
  }
}
function updateAlignBtn() { $('align').disabled = !(S.ref && S.origAudio); if (S.ref && !S.origAudio && S.samples.length) $('align-status').innerHTML = '<span class="warn">这个视频没有可用的现场声音，无法自动对齐。</span>'; }
$('speed').addEventListener('change', () => { $('speed-custom').hidden = $('speed').value !== 'custom'; });
$('align').addEventListener('click', async () => {
  const st = $('align-status');
  try {
    status(st, '分析节拍…', 0.05); await sleep(30);
    if (!S.vidEnv) S.vidEnv = onsetEnvelope(toMono(S.origAudio.ch), S.origAudio.sr);
    if (!S.refEnv) S.refEnv = onsetEnvelope(toMono(S.ref.ch), S.ref.sr);
    const sv = $('speed').value;
    const speed = sv === 'auto' ? null : sv === 'custom' ? parseFloat($('speed-custom').value) : parseFloat(sv);
    status(st, speed ? '对齐中…' : '自动识别倍速 + 对齐中…', 0.3); await sleep(30);
    const r = align(S.vidEnv, S.refEnv, speed);
    S.alignRes = r; S.nudge = 0;
    await renderAlignedAudio();
    const conf = r.score > 0.35 ? '<span class="ok">高</span>' : r.score > 0.18 ? '<span class="warn">中</span>' : '<span class="err">低（可能不是同一首歌/版本）</span>';
    S.alignConf = conf;
    showAlignStatus();
    renderCands();
    $('man-off').value = r.offset.toFixed(2);
    $('align-tune').hidden = false;
  } catch (err) { console.error(err); status(st, `<span class="err">对齐失败：${err.message || err}</span>`); }
});
const mmss = t => { const s = Math.abs(t), m = Math.floor(s / 60); return `${t < 0 ? '-' : ''}${m}:${(s - m * 60).toFixed(1).padStart(4, '0')}`; };
function showAlignStatus() {
  const r = S.alignRes;
  status($('align-status'), `倍速 ${r.speed.toFixed(3).replace(/0+$/, '').replace(/\.$/, '')}x，视频开头对应原曲 ${mmss(r.offset)}（${r.offset.toFixed(2)}s），可信度 ${S.alignConf}${S.nudge ? `（微调 ${S.nudge >= 0 ? '+' : ''}${S.nudge.toFixed(2)}s）` : ''}`);
}
function renderCands() {
  const r = S.alignRes, box = $('cands'); box.innerHTML = '';
  const cs = r.cands || [];
  $('cand-wrap').hidden = cs.length < 2;
  cs.forEach(c => {
    const b = document.createElement('button');
    b.textContent = `${mmss(c.offset)}${c.rel >= 0.999 ? ' ★' : ''}`;
    b.title = `匹配度 ${(c.rel * 100).toFixed(0)}%`;
    b.classList.toggle('on', Math.abs(c.offset - r.offset) < 0.05);
    b.addEventListener('click', async () => {
      stopPlay(); r.offset = c.offset; r.score = c.score; S.nudge = 0;
      box.querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b));
      $('man-off').value = c.offset.toFixed(2);
      await renderAlignedAudio(); showAlignStatus();
    });
    box.appendChild(b);
  });
}
$('man-apply').addEventListener('click', async () => {
  const v = parseFloat($('man-off').value); if (!S.alignRes || !isFinite(v)) return;
  stopPlay(); S.alignRes.offset = v; S.nudge = 0;
  $('cands').querySelectorAll('button').forEach(x => x.classList.remove('on'));
  await renderAlignedAudio(); showAlignStatus();
});
async function renderAlignedAudio() {
  const r = S.alignRes; const outSr = S.origAudio ? S.origAudio.sr : 44100;
  const ch = renderAligned(S.ref.ch, S.ref.sr, r.offset + S.nudge * r.speed, r.speed, S.dur, outSr);
  S.aligned = { sr: outSr, ch };
}
document.querySelectorAll('#align-tune [data-n]').forEach(b => b.addEventListener('click', async () => {
  S.nudge = +(S.nudge + parseFloat(b.dataset.n)).toFixed(2); stopPlay(); await renderAlignedAudio(); showAlignStatus();
}));
document.querySelectorAll('#amode button').forEach(b => b.addEventListener('click', () => {
  document.querySelectorAll('#amode button').forEach(x => x.classList.toggle('on', x === b)); S.amode = b.dataset.m; stopPlay();
}));

function finalAudio() {
  const o = S.origAudio, a = S.aligned;
  if (!a || S.amode === 'orig') return o;
  if (S.amode === 'ref' || !o) return a;
  const n = a.ch[0].length, ch = [0, 1].slice(0, Math.max(a.ch.length, o.ch.length)).map(c => {
    const A = a.ch[Math.min(c, a.ch.length - 1)], B = o.ch[Math.min(c, o.ch.length - 1)], out = new Float32Array(n);
    for (let i = 0; i < n; i++) { const v = 0.85 * A[i] + 0.4 * (B[i] || 0); out[i] = v > 1 ? 1 : v < -1 ? -1 : v; }
    return out;
  });
  return { sr: a.sr, ch };
}

// ---------- 5. 导出 ----------
document.querySelectorAll('#res button').forEach(b => b.addEventListener('click', () => {
  document.querySelectorAll('#res button').forEach(x => x.classList.toggle('on', x === b)); S.res = +b.dataset.r;
}));
$('export').addEventListener('click', async () => {
  stopPlay();
  const st = $('export-status'); $('export').disabled = true; $('result').hidden = true;
  try {
    const OW = S.res === 1080 ? 1080 : 720, OH = OW * 16 / 9;
    const aud = finalAudio();
    const vOk = caps.webcodecs && await canEncodeVideo('avc', { width: OW, height: OH });
    let aCodec = null;
    if (aud && caps.audioEnc) aCodec = (await canEncodeAudio('aac', { numberOfChannels: aud.ch.length, sampleRate: aud.sr })) ? 'aac' : (await canEncodeAudio('opus', { numberOfChannels: aud.ch.length, sampleRate: 48000 })) ? 'opus' : null;
    let blob = null;
    let note = '';
    if (vOk && (!aud || aCodec)) {
      // 依次尝试：高清失败（iOS 上偶发 Decoder failure）时自动降一档，不让用户白等
      const plan = OW === 1080 ? [[1080, 2560], [1080, 1920], [720, 1920]] : [[720, 1920], [720, 1280]];
      let lastErr = null;
      for (let k = 0; k < plan.length && !blob; k++) {
        const [w, cap] = plan[k];
        try {
          if (k > 0) status(st, `上一种方式失败（${lastErr && lastErr.message}），换个更省内存的方式重试…`);
          blob = await withVideoReleased(() => exportWebCodecs(w, w * 16 / 9, aud, aCodec, st, cap));
          if (w !== OW) note = `（1080p 在这台手机上失败了，已自动改为 ${w}p）`;
        } catch (e) { console.error('export attempt failed', w, cap, e); lastErr = e; }
      }
      if (!blob) throw lastErr;
    } else blob = await exportRecorder(OW, OH, aud, st);
    S.outBlob = blob;
    const url = URL.createObjectURL(blob);
    $('outvideo').src = url; $('download').href = url;
    $('result').hidden = false;
    status(st, `<span class="ok">完成</span>：${(blob.size / 1048576).toFixed(1)} MB${note}${aCodec === 'opus' ? '（音频为 Opus 编码，个别播放器可能不兼容）' : ''}`);
  } catch (err) { console.error(err); status(st, `<span class="err">导出失败：${err.message || err}</span>`); }
  $('export').disabled = false;
});

// 导出时先释放预览用的 <video>，它在 iOS 上也占着一个硬件解码器
async function withVideoReleased(fn) {
  const v = $('srcvideo'), src = v.src;
  v.pause(); v.removeAttribute('src'); v.load();
  try { return await fn(); } finally { v.src = src; v.load(); }
}

async function exportWebCodecs(OW, OH, aud, aCodec, st, cap = 1920) {
  ensurePath();
  if (Q.get('failExport') && cap >= +Q.get('failExport')) throw new Error('测试用：模拟失败');
  // 先把整帧（已按旋转信息摆正）画到中间画布，再从画布裁 9:16。
  // 不直接对 VideoFrame 用带源矩形的 drawImage：iOS Safari 会忽略源矩形，把整帧压扁塞进去（比例变形的原因）。
  const s = Math.min(1, cap / Math.max(S.W, S.H));
  const iw = Math.round(S.W * s), ih = Math.round(S.H * s);
  const frameSink = new CanvasSink(S.vt, { width: iw, height: ih, fit: 'fill', poolSize: 1 });
  const canvas = document.createElement('canvas'); canvas.width = OW; canvas.height = OH;
  const ctx = canvas.getContext('2d', { alpha: false });
  const out = new Output({ format: new Mp4OutputFormat({ fastStart: 'in-memory' }), target: new BufferTarget() });
  const vsrc = new CanvasSource(canvas, { codec: 'avc', quality: QUALITY_HIGH, keyFrameInterval: 1 });
  out.addVideoTrack(vsrc, { frameRate: 30 });
  let asrc = null;
  if (aud && aCodec) { asrc = new AudioBufferSource({ codec: aCodec, bitrate: 160000 }); out.addAudioTrack(asrc); }
  await out.start();
  const t1 = performance.now();
  ctx.imageSmoothingQuality = 'high';
  for await (const wc of frameSink.canvases(S.t0, S.t0 + S.dur)) {
    const t = wc.timestamp - S.t0;
    const c = cropAt(wc.timestamp);
    // 严格保证 9:16：以高度为准算宽度，并限制在画面内
    let ch = Math.min(c.h * s, ih, iw * 16 / 9), cw = ch * 9 / 16;
    const cx = Math.min(iw - cw, Math.max(0, c.x * s + (c.w * s - cw) / 2));
    const cy = Math.min(ih - ch, Math.max(0, c.y * s + (c.h * s - ch) / 2));
    ctx.drawImage(wc.canvas, cx, cy, cw, ch, 0, 0, OW, OH);
    const d = wc.duration || 1 / 30;
    if (t < 0) continue;
    await vsrc.add(t, d);
    status(st, `渲染中… ${Math.min(100, Math.round(100 * t / S.dur))}%`, t / S.dur);
  }
  if (asrc) {
    status(st, '编码音频…', 1);
    const sr = aCodec === 'opus' ? 48000 : aud.sr;
    let ch = aud.ch;
    if (sr !== aud.sr) ch = ch.map(c => { const n = Math.round(c.length * sr / aud.sr), r = new Float32Array(n), k = aud.sr / sr; for (let i = 0; i < n; i++) { const p = i * k, j = Math.floor(p), u = p - j; r[i] = (c[j] || 0) * (1 - u) + (c[j + 1] || 0) * u; } return r; });
    const ab = new AudioBuffer({ length: ch[0].length, numberOfChannels: ch.length, sampleRate: sr });
    ch.forEach((c, k) => ab.copyToChannel(c, k));
    await asrc.add(ab);
  }
  status(st, '封装 MP4…', 1);
  await out.finalize();
  console.log('export ms', performance.now() - t1);
  return new Blob([out.target.buffer], { type: 'video/mp4' });
}

// 兜底：不支持 WebCodecs 编码的旧系统，用实时录制（需要和视频一样长的时间）
async function exportRecorder(OW, OH, aud, st) {
  if (typeof MediaRecorder === 'undefined' || !HTMLCanvasElement.prototype.captureStream) throw new Error('这台设备的浏览器不支持导出，请更新 iOS');
  const mime = ['video/mp4;codecs=avc1,mp4a.40.2', 'video/mp4', 'video/webm'].find(m => MediaRecorder.isTypeSupported(m));
  const canvas = document.createElement('canvas'); canvas.width = OW; canvas.height = OH;
  const ctx = canvas.getContext('2d', { alpha: false });
  const stream = canvas.captureStream(30);
  const ac = new (window.AudioContext || window.webkitAudioContext)(); await ac.resume();
  let node = null;
  if (aud) {
    const dest = ac.createMediaStreamDestination();
    const buf = ac.createBuffer(aud.ch.length, aud.ch[0].length, aud.sr); aud.ch.forEach((c, k) => buf.copyToChannel(c, k));
    node = ac.createBufferSource(); node.buffer = buf; node.connect(dest);
    stream.addTrack(dest.stream.getAudioTracks()[0]);
  }
  const rec = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 8e6 });
  const chunks = []; rec.ondataavailable = e => e.data.size && chunks.push(e.data);
  const v = $('srcvideo'); v.muted = true; v.currentTime = 0;
  await new Promise(r => v.addEventListener('seeked', r, { once: true }));
  rec.start(500); await v.play(); node && node.start();
  await new Promise(res => {
    const loop = () => {
      const c = cropAt(v.currentTime + S.t0), k = v.videoWidth / S.W;
      ctx.drawImage(v, c.x * k, c.y * k, c.w * k, c.h * k, 0, 0, OW, OH);
      status(st, `实时录制中（旧系统兜底模式）… ${Math.round(100 * v.currentTime / S.dur)}%`, v.currentTime / S.dur);
      if (v.ended) return res(); requestAnimationFrame(loop);
    }; loop();
  });
  rec.stop(); node && node.stop();
  await new Promise(r => rec.onstop = r);
  return new Blob(chunks, { type: mime.split(';')[0] });
}

$('share').addEventListener('click', async () => {
  const b = S.outBlob; if (!b) return;
  const ext = b.type.includes('webm') ? 'webm' : 'mp4';
  const file = new File([b], `直拍_${new Date().toISOString().slice(5, 16).replace(/[-T:]/g, '')}.${ext}`, { type: b.type });
  if (navigator.canShare && navigator.canShare({ files: [file] })) {
    try { await navigator.share({ files: [file] }); } catch (e) { if (e.name !== 'AbortError') alert('分享失败：' + e.message); }
  } else { $('download').click(); }
});

window.__appReady = true;
// ---------- PWA ----------
if ('serviceWorker' in navigator && location.protocol === 'https:') navigator.serviceWorker.register('sw.js').catch(() => { });
