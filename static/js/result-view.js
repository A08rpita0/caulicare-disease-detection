/* Renders an analysis result. Shared by the Analyze page (live result) and the
   Report page (saved result from history), so both always show the same content.
   The result page is deliberately simple; methodology, metrics and limitations
   live on the About page and in the README. */
(function () {
  "use strict";
  const esc = (s) => CC.escapeHtml(s);
  const items = (arr, n) => (arr || []).slice(0, n);
  const list = (arr, n) => {
    const shown = items(arr, n);
    return shown.length ? `<ul class="detail-list">${shown.map((i) => `<li>${esc(i)}</li>`).join("")}</ul>` : "";
  };
  // Same stroke icons as templates/_icons.html, so the result page matches the rest of the site.
  const ICONS = {
    chart: '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
    eye: '<path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
    scan: '<path d="M4 8V6a2 2 0 0 1 2-2h2M16 4h2a2 2 0 0 1 2 2v2M20 16v2a2 2 0 0 1-2 2h-2M8 20H6a2 2 0 0 1-2-2v-2"/><path d="M4 12h16"/>',
    compare: '<path d="M12 3v18"/><rect x="3" y="6" width="6" height="12" rx="1.5"/><rect x="15" y="6" width="6" height="12" rx="1.5"/>',
    book: '<path d="M4 5a2 2 0 0 1 2-2h13v16H6a2 2 0 0 0-2 2z"/><path d="M4 21V5M8 7h7"/>',
    leaf: '<path d="M11 20A7 7 0 0 1 4 13c0-6 6-9 16-9 0 10-3 16-9 16z"/><path d="M4 20c4-4 7-7 11-10"/>',
    alert: '<path d="M12 9v4M12 17h.01"/><path d="M10.3 3.9L1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/>',
  };
  const icon = (name) => `<svg class="ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[name]}</svg>`;
  const section = (name, title) => `<div class="section-label"><h2><span class="sec-icon">${icon(name)}</span>${esc(title)}</h2></div>`;
  const figure = (src, alt, caption, cls = "") => `<figure><img ${cls ? `class="${cls}" ` : ""}src="${esc(src)}" alt="${esc(alt)}"><figcaption>${esc(caption)}</figcaption></figure>`;

  function header(entry, p, q) {
    const warnings = (q.warnings || []).map((w) => `<div class="notice warn">${esc(w.message)}</div>`).join("");
    return `
      <div class="card result-card">
        <span class="eyebrow" style="margin:0 0 12px">${icon("leaf")} Analysis result</span>
        <div class="result-head">
          <div class="thumb"><img src="${esc(entry.image)}" alt="Analysed leaf photo"></div>
          <div>
            <div class="small muted">Disease detected</div>
            <div class="result-label">${esc(p.disease)}</div>
            <div class="confidence-line">
              <span class="confidence-value">${CC.pct(p.confidence)}</span>
              <span class="small muted">confidence</span>
              <span class="badge ${esc(p.confidence_level)}">${esc(p.confidence_label)}</span>
            </div>
            <div class="meter" aria-hidden="true"><span class="${esc(p.confidence_level)}" style="width:${Math.max(p.confidence * 100, 2).toFixed(1)}%"></span></div>
            ${p.confidence_level !== "high" ? `<p class="small" style="margin:10px 0 0">${esc(p.confidence_message)}</p>` : ""}
          </div>
        </div>
        ${warnings ? `<div class="stack" style="margin-top:14px">${warnings}</div>` : ""}
      </div>`;
  }

  function breakdown(p) {
    const rows = Object.entries(p.all_predictions).sort((a, b) => b[1] - a[1]).map(([name, v], i) => `
      <div class="bar-row${i === 0 ? " top" : ""}"><span>${esc(name)}</span>
        <div class="bar-track" role="img" aria-label="${esc(name)} ${CC.pct(v)}"><div class="bar-fill" data-w="${Math.max(v * 100, 0.6).toFixed(2)}"></div></div>
        <span class="bar-val">${CC.pct(v)}</span></div>`).join("");
    return `${section("chart", "Prediction breakdown")}<div class="card"><div class="bars">${rows}</div></div>`;
  }

  function affected(entry, sev) {
    let body;
    if (!sev) {
      body = '<p class="muted small" style="margin:0">Not available for this result.</p>';
    } else if (!sev.available) {
      body = `<p class="muted small" style="margin:0">${esc(sev.reason)}</p>`;
    } else {
      const band = { Low: "high", Moderate: "moderate", High: "uncertain", "Very high": "uncertain" }[sev.band] || "";
      body = `
        <div class="xai-grid affected">
          ${figure(entry.image, "Original photo", "Original image")}
          ${figure(sev.mask, "Leaf with the estimated affected areas highlighted in red", "Affected region", "region-img")}
          ${figure(sev.overlay, "Original photo with the estimated affected areas tinted red", "Overlay")}
        </div>
        <div class="legend"><span class="swatch-key" style="background:#e82424"></span>Estimated affected area</div>
        <div class="stat-chips">
          <div class="stat-chip"><span>Estimated affected area</span><b>${sev.affected_percent.toFixed(0)}%</b></div>
          <div class="stat-chip"><span>Visual severity</span><b><span class="badge ${band}">${esc(sev.band)}</span></b></div>
        </div>
        <p class="tiny" style="margin:10px 0 0">An estimate based on leaf colour, not a measured diagnosis.</p>`;
    }
    return `${section("eye", "Visual analysis")}<div class="card"><h3 class="card-sub">Where is it affected?</h3>${body}</div>`;
  }

  function gradcam(entry, p) {
    const x = p.explanation;
    const body = x
      ? `<div class="xai-grid two">
          ${figure(entry.image, "Original photo", "Original image")}
          ${figure(x.overlay, "Grad-CAM heatmap over the photo", "Grad-CAM")}
        </div>
        <div class="legend"><span>Less</span><span class="ramp"></span><span>More influence</span></div>
        <p class="small muted" style="margin:10px 0 0">Highlights the areas that influenced the model's prediction.</p>`
      : '<p class="muted small" style="margin:0">Not available for this image.</p>';
    return `${section("scan", "Why the model looked here")}<div class="card">${body}</div>`;
  }

  function similarImages(sim) {
    if (!sim) return "";
    if (!sim.available || !sim.results || !sim.results.length) {
      return `${section("compare", "Visually similar images")}<div class="card">
        <p class="small muted inline-note" style="margin:0">${icon("alert")}${esc(sim.reason || "Visual similarity search is temporarily unavailable.")}</p></div>`;
    }
    const cards = sim.results.map((r) => `
      <figure class="sim-card">
        <img src="${esc(r.thumbnail)}" alt="Reference image: ${esc(r.label)}" loading="lazy">
        <figcaption><b>${esc(r.label)}</b><span>${Math.round(r.similarity * 100)}% visual similarity</span></figcaption>
      </figure>`).join("");
    return `${section("compare", "Visually similar images")}<div class="card">
      <p class="small muted" style="margin:0">Reference images that look visually similar to your image.</p>
      <div class="sim-grid">${cards}</div>
      <p class="tiny" style="margin:10px 0 0">Reference images: VegNet dataset, CC BY 4.0 (<a href="/about#attribution">credits</a>).</p></div>`;
  }

  function aboutDisease(d, expanded) {
    if (!d) return "";
    const healthy = d.id === "no-disease";
    const prevention = items(d.prevention, 3).concat(items(d.general_management, healthy ? 0 : 1));
    const sources = (d.sources || []).map((s) => `<a href="${esc(s.url)}" target="_blank" rel="noopener noreferrer">${esc(s.title)}</a>`).join(" · ");
    const more = [
      d.visual_signs && d.visual_signs.length ? `<h3>Visual signs</h3>${list(d.visual_signs, 10)}` : "",
      d.general_management && d.general_management.length ? `<h3>${healthy ? "Good practice" : "Management"}</h3>${list(d.general_management, 10)}` : "",
      d.when_to_seek_expert_help && d.when_to_seek_expert_help.length ? `<h3>When to ask an expert</h3>${list(d.when_to_seek_expert_help, 10)}` : "",
      sources ? `<p class="tiny" style="margin:10px 0 0">Sources: ${sources}</p>` : "",
    ].join("");
    return `${section("book", "About this disease")}
      <div class="card">
        <h3 style="margin-bottom:4px">${esc(d.name)}</h3>
        <p class="muted" style="margin:0 0 14px">${esc(d.short_description)}</p>
        <div class="info-cols">
          <div><h3>${healthy ? "What a healthy leaf looks like" : "Common symptoms"}</h3>${list(d.common_symptoms, 4)}</div>
          <div><h3>${healthy ? "Keeping plants healthy" : "Prevention and management"}</h3>${list(prevention, 4)}</div>
        </div>
        ${more ? `<details class="more"${expanded ? " open" : ""}><summary>More information</summary><div class="more-body">${more}
          <p class="no-print" style="margin:10px 0 0"><a href="/diseases/${esc(d.id)}">Open in the Disease Library</a></p></div></details>` : ""}
      </div>`;
  }

  function render(container, entry, opts = {}) {
    const data = entry.data, p = data.prediction;
    const expanded = !opts.actions; // the printable report shows everything
    container.innerHTML = [
      header(entry, p, data.quality || {}),
      breakdown(p),
      affected(entry, data.severity),
      gradcam(entry, p),
      similarImages(data.similar_images),
      aboutDisease(data.disease, expanded),
      opts.actions || "",
      '<p class="tiny result-foot">CauliCare is a screening aid, not a diagnosis. <a href="/about">How it works and its limitations</a>.</p>',
    ].join("");
    // Animate the bars after insertion; the final width is set even without animation support.
    requestAnimationFrame(() => container.querySelectorAll(".bar-fill").forEach((b) => (b.style.width = `${b.dataset.w}%`)));
  }

  window.CCResult = { render };
})();
