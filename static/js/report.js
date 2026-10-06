/* Report page: renders a saved result for printing / PDF, or downloads it as a standalone HTML file. */
(function () {
  "use strict";
  const id = new URLSearchParams(location.search).get("id");
  let entry = id ? CC.history.get(id) : null;
  if (!entry) {
    try {
      const current = JSON.parse(sessionStorage.getItem("caulicare:current") || "null");
      if (current && current.id === id) entry = current;
    } catch { /* storage unavailable */ }
  }
  const report = document.getElementById("report");
  if (!entry) {
    report.classList.add("hidden");
    document.querySelectorAll("#print-btn, #html-btn").forEach((b) => b.classList.add("hidden"));
    document.getElementById("report-missing").classList.remove("hidden");
    return;
  }

  document.getElementById("report-date").textContent = CC.formatDate(entry.ts);
  document.getElementById("report-file").textContent = entry.fileName || "";
  CCResult.render(document.getElementById("report-body"), entry);
  document.getElementById("print-btn").addEventListener("click", () => window.print());

  document.getElementById("html-btn").addEventListener("click", async () => {
    let css = "";
    try { css = await (await fetch(document.querySelector('link[rel="stylesheet"][href*="styles.css"]').href)).text(); } catch { /* unstyled fallback */ }
    report.querySelectorAll(".bar-fill").forEach((b) => (b.style.width = `${b.dataset.w}%`));
    // A static file has no tabs: show every analysis panel. All images are already inline data URIs.
    const copy = report.cloneNode(true);
    copy.querySelectorAll(".tabs").forEach((t) => t.remove());
    copy.querySelectorAll(".tab-panel").forEach((pnl) => pnl.removeAttribute("hidden"));
    copy.querySelectorAll(".print-only").forEach((h) => h.classList.remove("print-only"));
    copy.querySelectorAll("img[loading]").forEach((img) => img.removeAttribute("loading"));
    const html = `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CauliCare report - ${CC.escapeHtml(entry.data.prediction.disease)}</title><style>${css}</style></head>
<body><main class="container report" style="padding:24px 20px">${copy.outerHTML}</main></body></html>`;
    const url = URL.createObjectURL(new Blob([html], { type: "text/html" }));
    const a = Object.assign(document.createElement("a"), { href: url, download: `caulicare-report-${new Date(entry.ts).toISOString().slice(0, 10)}.html` });
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
})();
