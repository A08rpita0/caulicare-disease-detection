/* Disease Library search: ranks knowledge-base passages for a symptom query (BM25 on the server). */
(function () {
  "use strict";
  const form = document.getElementById("search-form");
  const input = document.getElementById("search-input");
  const status = document.getElementById("search-status");
  const results = document.getElementById("search-results");
  if (!form) return;

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const q = input.value.trim();
    results.innerHTML = "";
    if (!q) { status.textContent = "Type a symptom or keyword to search."; return; }
    status.textContent = "Searching…";
    try {
      const res = await fetch(`/api/search?q=${encodeURIComponent(q)}`);
      const data = await res.json();
      if (!res.ok || !data.success) throw new Error(data.error || "Search failed.");
      if (!data.results.length) { status.textContent = `No passages matched "${q}". Try other words, e.g. "spots", "yellow" or "underside".`; return; }
      status.textContent = `${data.results.length} matching passages from the knowledge base:`;
      results.innerHTML = data.results.map((r) => `
        <li><a href="/diseases/${CC.escapeHtml(r.disease_id)}">${CC.escapeHtml(r.disease)}</a>
          <span class="tiny"> · ${CC.escapeHtml(r.section)}</span>
          <div class="small" style="margin-top:4px">${CC.escapeHtml(r.text)}</div></li>`).join("");
    } catch (err) {
      status.textContent = err instanceof TypeError ? "Couldn't reach the server. Please try again." : err.message;
    }
  });
})();
