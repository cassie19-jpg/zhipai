const V = 'zhipai-v1.1';
const BIG = /\/(models|vendor)\//;
self.addEventListener('install', e => self.skipWaiting());
self.addEventListener('activate', e => e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== V).map(k => caches.delete(k)))).then(() => self.clients.claim())));
self.addEventListener('fetch', e => {
  const u = new URL(e.request.url);
  if (e.request.method !== 'GET' || u.origin !== location.origin) return;
  if (BIG.test(u.pathname)) { // 模型/wasm：缓存优先
    e.respondWith(caches.open(V).then(async c => (await c.match(e.request)) || fetch(e.request).then(r => { if (r.ok) c.put(e.request, r.clone()); return r; })));
  } else { // 页面/脚本：网络优先，离线用缓存
    e.respondWith(fetch(e.request).then(r => { if (r.ok) { const rc = r.clone(); caches.open(V).then(c => c.put(e.request, rc)); } return r; }).catch(() => caches.match(e.request)));
  }
});
