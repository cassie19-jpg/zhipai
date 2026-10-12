// v1.2：只缓存体积大的模型 / wasm（缓存优先，跨版本保留）。页面和脚本不经过 Service Worker，避免 iOS 上脚本加载异常。
const ASSETS = 'zhipai-assets-v1';
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', e => e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== ASSETS).map(k => caches.delete(k)))).then(() => self.clients.claim())));
self.addEventListener('fetch', e => {
  const u = new URL(e.request.url);
  if (e.request.method !== 'GET' || u.origin !== location.origin) return;
  if (!/\.(wasm|task|tflite)$/.test(u.pathname)) return; // 只管二进制模型，其他请求（页面、脚本）交给浏览器
  e.respondWith(caches.open(ASSETS).then(async c => {
    const hit = await c.match(e.request);
    if (hit) return hit;
    const r = await fetch(e.request);
    if (r.ok && !r.redirected) c.put(e.request, r.clone()).catch(() => { });
    return r;
  }).catch(() => fetch(e.request)));
});
