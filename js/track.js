// 纯算法：跨帧关联 + 虚拟镜头路径。无 DOM 依赖，可在 Node 中测试。
// samples: [{t, dets:[{box:[x0,y0,x1,y1] (像素), hip:[x,y], sh:[x,y], hist:Float32Array}]}]

export function iou(a, b) {
  const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = ix * iy;
  const ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return ua > 0 ? inter / ua : 0;
}
const cx = b => (b[0] + b[2]) / 2, cy = b => (b[1] + b[3]) / 2, bh = b => b[3] - b[1];

export function histSim(a, b) { // Bhattacharyya 系数，0..1
  if (!a || !b) return 0.5;
  let s = 0;
  for (let i = 0; i < a.length; i++) s += Math.sqrt(a[i] * b[i]);
  // 两段直方图（上衣+下装）各自归一化，总和为2
  return s / 2;
}

// 从 anchor 帧开始单向跟踪（dir=+1/-1），到 stop（不含）为止。
function trackDir(samples, a, detIdx, dir, stop) {
  const out = new Map();
  let prev = samples[a].dets[detIdx];
  const refHist = prev.hist;
  let curHist = prev.hist ? Float32Array.from(prev.hist) : null;
  let box = prev.box.slice();
  let vx = 0, vy = 0, lost = 0;
  out.set(a, { det: detIdx, box: box.slice(), conf: 1 });
  for (let i = a + dir; i !== stop; i += dir) {
    const dets = samples[i].dets;
    const dt = Math.abs(samples[i].t - samples[i - dir].t) || 0.1;
    const h = bh(box);
    const px = cx(box) + vx * dt, py = cy(box) + vy * dt;
    const pred = [box[0] + vx * dt, box[1] + vy * dt, box[2] + vx * dt, box[3] + vy * dt];
    const maxDist = 0.6 + 0.35 * lost; // 以身高为单位
    let best = -1, bestCost = Infinity, second = Infinity;
    for (let k = 0; k < dets.length; k++) {
      const d = dets[k];
      const dist = Math.hypot(cx(d.box) - px, cy(d.box) - py) / h;
      if (dist > maxDist) continue;
      const size = Math.abs(Math.log(bh(d.box) / h));
      const app = 1 - Math.max(histSim(refHist, d.hist), histSim(curHist, d.hist));
      const ov = iou(pred, d.box);
      const wApp = lost > 0 ? 6 : 4;
      const cost = 2.2 * dist + 1.2 * size + wApp * app - 0.8 * ov;
      if (cost < bestCost) { second = bestCost; bestCost = cost; best = k; }
      else if (cost < second) second = cost;
    }
    const accept = best >= 0 && bestCost < (lost > 0 ? 1.6 : 2.2);
    if (accept) {
      const d = dets[best];
      const nvx = (cx(d.box) - cx(box)) / dt, nvy = (cy(d.box) - cy(box)) / dt;
      vx = 0.6 * vx + 0.4 * nvx; vy = 0.6 * vy + 0.4 * nvy;
      // 限制速度，防止跳变放大
      const vmax = 3 * h; const vn = Math.hypot(vx, vy);
      if (vn > vmax) { vx *= vmax / vn; vy *= vmax / vn; }
      box = d.box.slice();
      lost = 0;
      const margin = second - bestCost;
      if (curHist && d.hist && margin > 0.4) for (let j = 0; j < curHist.length; j++) curHist[j] = 0.92 * curHist[j] + 0.08 * d.hist[j];
      out.set(i, { det: best, box: box.slice(), conf: Math.max(0.2, Math.min(1, margin)) });
    } else {
      lost++;
      vx *= 0.7; vy *= 0.7;
      box = [box[0] + vx * dt, box[1] + vy * dt, box[2] + vx * dt, box[3] + vy * dt];
      out.set(i, { det: -1, box: box.slice(), conf: 0 });
    }
  }
  return out;
}

// 全局最优关联（动态规划 / Viterbi）：在所有帧的检测框里找一条“最像同一个人”的路径。
// 节点 = 某帧的某个检测框；边 = 相隔 k 帧的两个框（k>1 表示中间几帧没检测到她，如被挡住）。
// 代价 = 位移 + 尺寸突变 + 跳帧惩罚 + 与点选时外观（衣服颜色）的差异。锚点帧强制经过用户点的那个框。
export function solveTrack(samples, anchors, opts = {}) {
  const n = samples.length;
  if (!anchors.size) return null;
  const keys = [...anchors.keys()].sort((a, b) => a - b);
  const fps = n > 1 ? (n - 1) / Math.max(1e-6, samples[n - 1].t - samples[0].t) : 8;
  const K = Math.max(2, Math.round((opts.maxGapSec ?? 3) * fps));
  const SKIP = opts.skip ?? 0.5, W_POS = opts.wPos ?? 4, W_SIZE = opts.wSize ?? 1.0, W_APP = opts.wApp ?? 4, VMAX = opts.vmax ?? 1.2, APP_DB = opts.appDb ?? 0.1, W_JUMP = opts.wJump ?? 30;
  // 参考外观：所有锚点外观的平均
  let ref = null;
  for (const a of keys) { const h = samples[a].dets[anchors.get(a)].hist; if (!h) continue; if (!ref) ref = new Float32Array(h.length); for (let i = 0; i < h.length; i++) ref[i] += h[i] / keys.length; }
  // 每帧允许的节点
  const allowed = samples.map((s, i) => anchors.has(i) ? [anchors.get(i)] : s.dets.map((_, j) => j));
  // 外观代价取相对值：每帧里最像的那个人代价为 0（避免为了少付外观代价而故意跳帧）
  const unary = samples.map(s => { const sims = s.dets.map(d => histSim(ref, d.hist)); const mx = Math.max(0, ...sims); return sims.map(v => W_APP * Math.max(0, mx - v - APP_DB)); });
  const best = samples.map(s => new Float64Array(s.dets.length).fill(Infinity));
  const back = samples.map(s => new Int32Array(s.dets.length * 2).fill(-1));
  // 下一个锚点（不可跨越）
  const prevAnchor = new Int32Array(n).fill(-1);
  for (let i = 0, last = -1; i < n; i++) { prevAnchor[i] = last; if (anchors.has(i)) last = i; }
  const firstA = keys[0], lastA = keys[keys.length - 1];
  for (let i = 0; i < n; i++) {
    const dets = samples[i].dets;
    for (const j of allowed[i]) {
      const d = dets[j], c = [cx(d.box), cy(d.box)], h = bh(d.box);
      let bv = Infinity, bi = -1, bj = -1;
      if (i <= firstA) bv = SKIP * i; // 从这里开始（之前的帧都算跳过）
      const lo = Math.max(0, i - K, prevAnchor[i]);
      for (let p = i - 1; p >= lo; p--) {
        const k = i - p, dp = samples[p].dets;
        for (const q of allowed[p]) {
          const base = best[p][q]; if (base === Infinity) continue;
          const e = dp[q], he = bh(e.box), hm = (h + he) / 2;
          const dist = Math.hypot(c[0] - cx(e.box), c[1] - cy(e.box)) / hm;
          const speed = dist * fps / k; // 每秒移动几个身高
          const cost = base + W_POS * dist * dist / k + (speed > VMAX ? W_JUMP * (dist - VMAX * k / fps) : 0) + W_SIZE * Math.abs(Math.log(h / he)) + SKIP * (k - 1);
          if (cost < bv) { bv = cost; bi = p; bj = q; }
        }
      }
      best[i][j] = bv + unary[i][j];
      back[i][2 * j] = bi; back[i][2 * j + 1] = bj;
    }
  }
  // 终点
  let ei = -1, ej = -1, ev = Infinity;
  for (let i = lastA; i < n; i++) for (const j of allowed[i]) { const v = best[i][j] + SKIP * (n - 1 - i); if (v < ev) { ev = v; ei = i; ej = j; } }
  const nodes = [];
  for (let i = ei, j = ej; i >= 0; ) { nodes.push([i, j]); const pi = back[i][2 * j], pj = back[i][2 * j + 1]; i = pi; j = pj; }
  nodes.reverse();
  const res = new Array(n);
  for (const [i, j] of nodes) res[i] = { det: j, box: samples[i].dets[j].box.slice(), conf: 1 };
  // 没检测到的帧：线性插值 / 首尾保持
  for (let a = 0; a < nodes.length; a++) {
    const [i] = nodes[a];
    const nx = a + 1 < nodes.length ? nodes[a + 1][0] : n;
    for (let m = i + 1; m < nx; m++) {
      if (nx < n) { const u = (m - i) / (nx - i), A = res[i].box, B = res[nx].box; res[m] = { det: -1, box: A.map((v, q) => v * (1 - u) + B[q] * u), conf: 0 }; }
      else res[m] = { det: -1, box: res[i].box.slice(), conf: 0 };
    }
  }
  for (let m = 0; m < nodes[0][0]; m++) res[m] = { det: -1, box: res[nodes[0][0]].box.slice(), conf: 0 };
  return res;
}

// 聚焦档位 -> 每个 sample 的目标裁切 (中心x, 中心y, 高度)，像素
export const FOCUS = {
  loose: { label: '全身宽松' },
  tight: { label: '全身紧' },
  half: { label: '半身' },
};

function targetFor(det, box, mode, W, H) {
  const h = bh(box), w = box[2] - box[0];
  let ch, ccy, ccx = cx(box);
  if (mode === 'half' && det && det.hip && det.sh) {
    const top = box[1] - 0.06 * h;
    const bottom = det.hip[1] + 0.22 * Math.max(10, det.hip[1] - det.sh[1]);
    ch = (bottom - top) * 1.18;
    ccy = (top + bottom) / 2;
    ccx = (det.sh[0] + det.hip[0]) / 2 * 0.6 + ccx * 0.4;
  } else if (mode === 'half') {
    ch = h * 0.7; ccy = box[1] + h * 0.32;
  } else if (mode === 'tight') {
    ch = Math.max(h * 1.16, w * 16 / 9 * 0.95);
    ccy = cy(box);
  } else {
    ch = Math.max(h * 1.5, w * 16 / 9 * 1.1);
    ccy = cy(box) + h * 0.02;
  }
  return [ccx, ccy, ch];
}

function gaussSmooth(arr, sigma) {
  if (sigma <= 0) return arr.slice();
  const r = Math.ceil(sigma * 3), ker = [];
  for (let i = -r; i <= r; i++) ker.push(Math.exp(-(i * i) / (2 * sigma * sigma)));
  const out = new Float64Array(arr.length);
  for (let i = 0; i < arr.length; i++) {
    let s = 0, ws = 0;
    for (let j = -r; j <= r; j++) {
      const k = Math.min(arr.length - 1, Math.max(0, i + j));
      s += arr[k] * ker[j + r]; ws += ker[j + r];
    }
    out[i] = s / ws;
  }
  return out;
}

// 生成逐帧镜头路径。times: 输出帧时间（秒，轨道时间）。返回 [{x,y,w,h}] 像素裁切框（9:16）
export function cameraPath(samples, track, mode, W, H, times, opts = {}) {
  const maxH = Math.min(H, W * 16 / 9);
  const ts = samples.map(s => s.t);
  const tg = track.map((tr, i) => {
    const det = tr.det >= 0 ? samples[i].dets[tr.det] : null;
    return targetFor(det, tr.box, mode, W, H);
  });
  // 插值到输出帧
  const nx = [], ny = [], nh = [];
  let j = 0;
  for (const t of times) {
    while (j < ts.length - 2 && ts[j + 1] < t) j++;
    let a = j, b = Math.min(j + 1, ts.length - 1);
    let u = ts[b] > ts[a] ? (t - ts[a]) / (ts[b] - ts[a]) : 0;
    u = Math.max(0, Math.min(1, u));
    nx.push(tg[a][0] * (1 - u) + tg[b][0] * u);
    ny.push(tg[a][1] * (1 - u) + tg[b][1] * u);
    nh.push(Math.min(maxH, tg[a][2] * (1 - u) + tg[b][2] * u));
  }
  const fps = times.length > 1 ? 1 / (times[1] - times[0]) : 30;
  const sx = gaussSmooth(nx, (opts.posSmooth ?? 0.22) * fps);
  const sy = gaussSmooth(ny, (opts.posSmooth ?? 0.22) * fps);
  const sh = gaussSmooth(nh, (opts.sizeSmooth ?? 0.6) * fps);
  return times.map((_, i) => {
    const h = Math.min(maxH, Math.max(maxH * 0.18, sh[i]));
    const w = h * 9 / 16;
    const x = Math.min(W - w, Math.max(0, sx[i] - w / 2));
    const y = Math.min(H - h, Math.max(0, sy[i] - h / 2));
    return { x, y, w, h };
  });
}
