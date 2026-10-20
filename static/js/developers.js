/* Fills the "Live service statistics" card from /api/stats and /api/model. */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const fmtMs = (v) => (typeof v === "number" ? `${v < 10 ? v.toFixed(1) : Math.round(v)} ms` : "–");

  Promise.all([
    fetch("/api/stats").then((r) => r.json()),
    fetch("/api/model").then((r) => r.json()),
  ]).then(([stats, model]) => {
    const predict = (stats.endpoints || {})["POST /api/predict"];
    $("st-requests").textContent = String(stats.requests_total ?? 0);
    $("st-p50").textContent = predict ? fmtMs(predict.latency_ms.p50) : "no data yet";
    $("st-p95").textContent = predict ? fmtMs(predict.latency_ms.p95) : "no data yet";
    $("st-backend").textContent = (stats.inference && stats.inference.plain_prediction_backend) || "–";
    const cache = stats.cache || {};
    const dec = stats.decisions || {};
    const parts = [
      `Decisions so far: ${Object.entries(dec).map(([k, v]) => `${k} ${v}`).join(", ") || "none"}.`,
      cache.enabled ? `Cache: ${cache.hits} hit(s), ${cache.misses} miss(es).` : "Cache is off.",
      `ONNX: ${(model.serving && model.serving.onnx) || "–"}.`,
    ];
    $("st-detail").textContent = parts.join(" ");
  }).catch(() => {
    $("st-detail").textContent = "Statistics are unavailable right now.";
  });
})();
