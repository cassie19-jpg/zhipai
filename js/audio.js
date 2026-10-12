// 音频对齐 + 保持音调的变速（WSOLA）。纯算法，无 DOM 依赖。
export const ENV_RATE = 100; // 包络采样率 Hz

function fftInPlace(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k, b = a + len / 2;
        const xr = re[b] * cr - im[b] * ci, xi = re[b] * ci + im[b] * cr;
        re[b] = re[a] - xr; im[b] = im[a] - xi; re[a] += xr; im[a] += xi;
        const ncr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = ncr;
      }
    }
  }
}

export function toMono(channels) {
  if (channels.length === 1) return channels[0];
  const n = channels[0].length, out = new Float32Array(n);
  for (const c of channels) for (let i = 0; i < n; i++) out[i] += c[i] / channels.length;
  return out;
}

// 谱通量起音包络，ENV_RATE Hz
export function onsetEnvelope(mono, sr) {
  const dec = Math.max(1, Math.round(sr / 11025));
  const r = sr / dec;
  const n = Math.floor(mono.length / dec), x = new Float32Array(n);
  for (let i = 0; i < n; i++) { let s = 0; for (let k = 0; k < dec; k++) s += mono[i * dec + k]; x[i] = s / dec; }
  const N = 512, hop = r / ENV_RATE;
  const frames = Math.max(0, Math.floor((n - N) / hop));
  const win = new Float32Array(N); for (let i = 0; i < N; i++) win[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / N);
  const re = new Float64Array(N), im = new Float64Array(N);
  let prev = new Float32Array(N / 2), cur = new Float32Array(N / 2);
  const env = new Float32Array(frames);
  for (let f = 0; f < frames; f++) {
    const o = Math.round(f * hop);
    for (let i = 0; i < N; i++) { re[i] = x[o + i] * win[i]; im[i] = 0; }
    fftInPlace(re, im);
    let s = 0;
    for (let b = 1; b < N / 2; b++) {
      const m = Math.log(1 + 100 * Math.hypot(re[b], im[b]));
      cur[b] = m; const d = m - prev[b]; if (d > 0) s += d;
    }
    env[f] = s; const t = prev; prev = cur; cur = t;
  }
  // 去局部均值（~1s），半波整流，归一化
  const w = 50, out = new Float32Array(frames);
  let acc = 0; const cs = new Float64Array(frames + 1);
  for (let i = 0; i < frames; i++) { acc += env[i]; cs[i + 1] = acc; }
  let mx = 1e-9;
  for (let i = 0; i < frames; i++) {
    const a = Math.max(0, i - w), b = Math.min(frames, i + w + 1);
    const v = env[i] - (cs[b] - cs[a]) / (b - a);
    out[i] = v > 0 ? v : 0; if (out[i] > mx) mx = out[i];
  }
  for (let i = 0; i < frames; i++) out[i] /= mx;
  return out;
}

function stretchEnv(env, s) { // r_s[m] = env(m*s)
  const n = Math.floor((env.length - 1) / s), out = new Float32Array(n);
  for (let m = 0; m < n; m++) { const p = m * s, i = Math.floor(p), u = p - i; out[m] = env[i] * (1 - u) + env[i + 1] * u; }
  return out;
}

// 归一化互相关：返回 {lag, score, ratio}，lag 为 ref 包络中视频起点的位置（帧）
function nccBest(v, r) {
  const Nv = v.length, Nr = r.length;
  let size = 1; while (size < Nr + Nv) size <<= 1;
  const vm = v.reduce((a, b) => a + b, 0) / Nv;
  const ar = new Float64Array(size), ai = new Float64Array(size), br = new Float64Array(size), bi = new Float64Array(size);
  let vn = 0;
  for (let i = 0; i < Nv; i++) { const z = v[i] - vm; br[i] = z; vn += z * z; }
  for (let i = 0; i < Nr; i++) ar[i] = r[i];
  fftInPlace(ar, ai); fftInPlace(br, bi);
  // corr[k] = sum_i r[i+k] v[i]  -> A * conj(B)
  for (let i = 0; i < size; i++) { const xr = ar[i] * br[i] + ai[i] * bi[i], xi = ai[i] * br[i] - ar[i] * bi[i]; ar[i] = xr; ai[i] = -xi; }
  fftInPlace(ar, ai); // 共轭技巧做逆变换
  const cs = new Float64Array(Nr + 1), cs2 = new Float64Array(Nr + 1);
  for (let i = 0; i < Nr; i++) { cs[i + 1] = cs[i] + r[i]; cs2[i + 1] = cs2[i] + r[i] * r[i]; }
  // 允许视频比原曲先开始（负延迟）最多 Nv/2
  const scores = [];
  let best = -Infinity, bestK = 0;
  const kMin = -Math.floor(Nv / 2), kMax = Nr - Math.floor(Nv / 2);
  for (let k = kMin; k < kMax; k++) {
    const a = Math.max(0, k), b = Math.min(Nr, k + Nv);
    if (b - a < Nv / 2) { scores.push(0); continue; }
    const len = b - a, mean = (cs[b] - cs[a]) / len;
    const varr = (cs2[b] - cs2[a]) - len * mean * mean;
    const idx = ((k % size) + size) % size;
    const c = ar[idx] / size;
    const sc = varr > 1e-9 ? c / Math.sqrt(varr * vn) : 0;
    scores.push(sc);
    if (sc > best) { best = sc; bestK = k; }
  }
  // 次高峰（排除 ±0.5s）
  let second = 0;
  for (let i = 0; i < scores.length; i++) { const k = i + kMin; if (Math.abs(k - bestK) > ENV_RATE * 0.5 && scores[i] > second) second = scores[i]; }
  // 抛物线插值亚帧
  const bi0 = bestK - kMin; let frac = 0;
  if (bi0 > 0 && bi0 < scores.length - 1) { const y0 = scores[bi0 - 1], y1 = scores[bi0], y2 = scores[bi0 + 1]; const d = y0 - 2 * y1 + y2; if (d < 0) frac = 0.5 * (y0 - y2) / d; }
  return { lag: bestK + frac, score: best, ratio: second > 0 ? best / second : 9 };
}

// 对齐：videoEnv/refEnv 为 onsetEnvelope 结果。speed = null 表示自动检测。
// 返回 {speed, offset(秒，视频0时刻对应的原曲位置), score, ratio}
export function align(videoEnv, refEnv, speed = null, onProgress) {
  const test = s => { const r = nccBest(videoEnv, stretchEnv(refEnv, s)); return { speed: s, offset: r.lag * s / ENV_RATE, score: r.score, ratio: r.ratio }; };
  if (speed) return test(speed);
  let best = null; const grid = [];
  for (let s = 0.5; s <= 1.301; s += 0.01) grid.push(+s.toFixed(3));
  grid.forEach((s, i) => { const r = test(s); if (!best || r.score > best.score) best = r; onProgress && onProgress((i + 1) / (grid.length + 20)); });
  let c = best.speed;
  for (let s = c - 0.01; s <= c + 0.0101; s += 0.001) { const r = test(+s.toFixed(4)); if (r.score > best.score) best = r; }
  // 常见档位吸附：0.5/0.6/0.7/0.75/0.8/0.85/0.9/0.95/1.0/1.1/1.2/1.25
  for (const p of [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.1, 1.2, 1.25]) {
    if (Math.abs(p - best.speed) <= 0.006) { const r = test(p); if (r.score >= best.score * 0.97) best = r; }
  }
  onProgress && onProgress(1);
  return best;
}

// WSOLA 变速保持音调。rate>1 变快。输入/输出: Float32Array[] 每声道
export function wsola(channels, sr, rate) {
  if (Math.abs(rate - 1) < 1e-4) return channels.map(c => c.slice());
  const N = Math.round(sr * 0.046) & ~1, Hs = N / 2, tol = Math.round(sr * 0.012);
  const len = channels[0].length, outLen = Math.floor(len / rate);
  const mono = toMono(channels);
  const win = new Float32Array(N); for (let i = 0; i < N; i++) win[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / N);
  const out = channels.map(() => new Float32Array(outLen + N)), wsum = new Float32Array(outLen + N);
  let prevPos = 0; // 上一帧在输入中的实际位置
  for (let o = 0, f = 0; o < outLen; o += Hs, f++) {
    const nominal = Math.round(o * rate);
    let pos = nominal;
    if (f > 0) {
      // 自然延续位置 prevPos+Hs，在 nominal±tol 中找与其最相似的片段
      const natural = prevPos + Hs;
      let bestC = -Infinity, bestP = nominal;
      const L = Math.min(Hs, N);
      for (let step of [8, 1]) {
        const lo = step === 8 ? nominal - tol : bestP - 8, hi = step === 8 ? nominal + tol : bestP + 8;
        for (let p = lo; p <= hi; p += step) {
          if (p < 0 || p + N >= len || natural + L >= len) continue;
          let c = 0;
          for (let i = 0; i < L; i += 2) c += mono[p + i] * mono[natural + i];
          if (c > bestC) { bestC = c; bestP = p; }
        }
      }
      pos = bestP;
    }
    pos = Math.max(0, Math.min(len - N - 1, pos));
    if (pos < 0) break;
    for (let ch = 0; ch < channels.length; ch++) { const src = channels[ch], dst = out[ch]; for (let i = 0; i < N; i++) dst[o + i] += src[pos + i] * win[i]; }
    for (let i = 0; i < N; i++) wsum[o + i] += win[i];
    prevPos = pos;
  }
  return out.map(c => { const r = new Float32Array(outLen); for (let i = 0; i < outLen; i++) r[i] = wsum[i] > 1e-3 ? c[i] / wsum[i] : c[i]; return r; });
}

// 生成与视频对齐的原曲轨：长度 = durSec*sr，视频时间 t 对应原曲 offset + speed*t
export function renderAligned(refChannels, refSr, offset, speed, durSec, outSr) {
  const startS = Math.round(offset * refSr);
  const needIn = Math.ceil(speed * durSec * refSr) + refSr; // 多取 1s
  const segs = refChannels.map(c => {
    const seg = new Float32Array(needIn);
    for (let i = 0; i < needIn; i++) { const k = startS + i; seg[i] = k >= 0 && k < c.length ? c[k] : 0; }
    return seg;
  });
  let st = wsola(segs, refSr, speed);
  // 重采样到 outSr
  const outLen = Math.round(durSec * outSr), ratio = refSr / outSr;
  return st.map(c => { const r = new Float32Array(outLen); for (let i = 0; i < outLen; i++) { const p = i * ratio, k = Math.floor(p), u = p - k; r[i] = k + 1 < c.length ? c[k] * (1 - u) + c[k + 1] * u : 0; } return r; });
}
