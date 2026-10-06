/* Shared helpers: navigation, safe browser storage, analysis history, formatting. */
(function () {
  "use strict";
  const header = document.querySelector(".site-header");
  const toggle = document.querySelector(".nav-toggle");
  const links = document.getElementById("nav-links");
  if (toggle && links) {
    toggle.addEventListener("click", () => {
      const open = links.classList.toggle("open");
      toggle.setAttribute("aria-expanded", String(open));
      toggle.setAttribute("aria-label", open ? "Close menu" : "Open menu");
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && links.classList.contains("open")) { links.classList.remove("open"); toggle.setAttribute("aria-expanded", "false"); toggle.focus(); }
    });
  }
  const onScroll = () => header && header.classList.toggle("scrolled", window.scrollY > 4);
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  const store = {
    get(key, fallback) { try { const v = localStorage.getItem(key); return v ? JSON.parse(v) : fallback; } catch { return fallback; } },
    set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); return true; } catch { return false; } },
    remove(key) { try { localStorage.removeItem(key); } catch { /* storage unavailable */ } },
  };

  // History lives only in this browser (localStorage). Each entry keeps a small copy
  // of the photo and the API result so a past analysis can be reopened or printed.
  const HISTORY_KEY = "caulicare:history:v2";
  const MAX_ITEMS = 12;
  const history = {
    list: () => store.get(HISTORY_KEY, []),
    get: (id) => history.list().find((h) => h.id === id) || null,
    add(entry) {
      let items = [entry, ...history.list().filter((h) => h.id !== entry.id)].slice(0, MAX_ITEMS);
      // When storage is full, drop the oldest entries until it fits.
      while (items.length && !store.set(HISTORY_KEY, items)) items = items.slice(0, -1);
      return items.length > 0;
    },
    remove(id) { store.set(HISTORY_KEY, history.list().filter((h) => h.id !== id)); },
    clear: () => store.remove(HISTORY_KEY),
  };

  function escapeHtml(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  // Softmax scores are never shown as "100%" or "0.0%": both would read as certainty.
  const pct = (v) => (v >= 0.999 ? ">99.9%" : v < 0.001 ? "<0.1%" : `${(v * 100).toFixed(1)}%`);

  function formatDate(ts) {
    try { return new Date(ts).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }); } catch { return ""; }
  }

  /* Downscale an image (URL or data URL) to a JPEG data URL whose longest side is `max`. */
  function downscale(src, max, quality = 0.82, square = false) {
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        const c = document.createElement("canvas");
        const ctx = c.getContext("2d");
        if (square) {
          const s = Math.min(img.width, img.height);
          c.width = c.height = max;
          ctx.drawImage(img, (img.width - s) / 2, (img.height - s) / 2, s, s, 0, 0, max, max);
        } else {
          const k = Math.min(1, max / Math.max(img.width, img.height));
          c.width = Math.round(img.width * k); c.height = Math.round(img.height * k);
          ctx.drawImage(img, 0, 0, c.width, c.height);
        }
        try { resolve(c.toDataURL("image/jpeg", quality)); } catch { resolve(null); }
      };
      img.onerror = () => resolve(null);
      img.src = src;
    });
  }

  window.CC = { store, history, escapeHtml, pct, formatDate, downscale };
})();
