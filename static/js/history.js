/* History page: results saved in this browser's localStorage. */
(function () {
  "use strict";
  const listEl = document.getElementById("history-list");
  const empty = document.getElementById("history-empty");
  const count = document.getElementById("history-count");
  const clearBtn = document.getElementById("clear-btn");
  const esc = (s) => CC.escapeHtml(s);

  function render() {
    const items = CC.history.list();
    empty.classList.toggle("hidden", items.length > 0);
    clearBtn.classList.toggle("hidden", items.length === 0);
    count.textContent = items.length ? `${items.length} saved ${items.length === 1 ? "analysis" : "analyses"}` : "";
    listEl.innerHTML = items.map((h) => {
      const p = h.data.prediction;
      return `<li class="card history-item">
        ${h.thumb ? `<img src="${esc(h.thumb)}" alt="">` : '<span class="ph"></span>'}
        <div style="min-width:0"><h3>${esc(p.disease)}</h3>
          <div class="row" style="gap:8px"><span class="badge ${esc(p.confidence_level)}">${esc(p.confidence_label)}</span><span class="small">${CC.pct(p.confidence)}</span></div>
          <div class="tiny" style="margin-top:4px">${esc(CC.formatDate(h.ts))}${h.fileName ? ` · ${esc(h.fileName)}` : ""}</div></div>
        <div class="h-actions">
          <a class="btn btn-secondary btn-sm" href="/report?id=${encodeURIComponent(h.id)}">View</a>
          <button class="btn btn-ghost btn-sm" type="button" data-delete="${esc(h.id)}" aria-label="Delete this analysis">Delete</button>
        </div></li>`;
    }).join("");
  }

  listEl.addEventListener("click", (e) => {
    const id = e.target.closest("[data-delete]")?.dataset.delete;
    if (id) { CC.history.remove(id); render(); }
  });
  clearBtn.addEventListener("click", () => {
    if (window.confirm("Delete all saved analyses from this browser?")) { CC.history.clear(); render(); }
  });
  render();
})();
