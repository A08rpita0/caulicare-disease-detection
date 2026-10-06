/* Analyze page: file selection, client-side checks, upload, and result states. */
(function () {
  "use strict";
  const MAX_MB = Number(document.body.dataset.maxUploadMb) || 8;
  const TYPES = ["image/jpeg", "image/png", "image/webp"];
  const EXT = /\.(jpe?g|png|webp)$/i;
  // The leaf check can wait for a sleeping similarity service to wake up.
  const CHECK_TIMEOUT_MS = 100000;
  const ANALYZE_TIMEOUT_MS = 90000;
  const REFUSALS = { low_quality: "Low image quality", not_cauliflower_leaf: "Not a cauliflower leaf" };
  const $ = (id) => document.getElementById(id);

  const el = {
    input: $("file-input"), drop: $("dropzone"), previewWrap: $("preview-wrap"), preview: $("preview-img"),
    meta: $("file-meta"), analyze: $("analyze-btn"), remove: $("remove-btn"), clientErr: $("client-error"),
    upload: $("upload-card"), loading: $("loading"), errCard: $("error-card"), errMsg: $("error-msg"),
    refusal: $("refusal-card"), result: $("result"),
  };
  let file = null, previewUrl = null, busy = false;

  fetch("/api/health").then((r) => r.json()).then((h) => {
    if (!h.model_loaded && !h.model_file_present) $("model-status").classList.remove("hidden");
  }).catch(() => { /* health is informational only */ });

  // ---------------------------------------------------------------- states
  function show(state) {
    el.upload.classList.toggle("hidden", state === "loading");
    el.loading.classList.toggle("hidden", state !== "loading");
    el.errCard.classList.toggle("hidden", state !== "error");
    el.refusal.classList.toggle("hidden", state !== "refusal");
    el.result.classList.toggle("hidden", state !== "result");
    const hasFile = Boolean(file);
    el.drop.classList.toggle("hidden", hasFile);
    el.previewWrap.classList.toggle("hidden", !hasFile);
  }

  function clientError(msg) {
    el.clientErr.querySelector("span").textContent = msg || "";
    el.clientErr.classList.toggle("hidden", !msg);
  }

  function validate(f) {
    if (!f) return "Please choose an image file.";
    if (f.size === 0) return "The selected file is empty. Please choose another photo.";
    const typeOk = TYPES.includes(f.type) || (!f.type && EXT.test(f.name));
    if (!typeOk || !EXT.test(f.name)) return "Please upload a JPG, PNG, or WEBP image.";
    if (f.size > MAX_MB * 1024 * 1024) return `The image is too large. Please upload an image under ${MAX_MB} MB.`;
    return null;
  }

  function setFile(f) {
    if (busy) return;
    const err = validate(f);
    if (err) { clientError(err); return; }
    clientError("");
    file = f;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(f);
    el.preview.src = previewUrl;
    el.meta.textContent = `${f.name} · ${(f.size / 1024 / 1024).toFixed(2)} MB`;
    show("preview");
    el.analyze.disabled = false;
  }

  function reset() {
    if (busy) return;
    file = null;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
    el.preview.removeAttribute("src");
    clientError("");
    show("idle");
  }

  // The dropzone is a <label for="file-input">, so clicking or tapping it opens the
  // native picker without JavaScript. Clearing the value lets the same file be re-chosen.
  el.input.addEventListener("change", () => {
    const f = el.input.files && el.input.files[0];
    el.input.value = "";
    if (f) setFile(f);
  });
  ["dragenter", "dragover"].forEach((t) => el.drop.addEventListener(t, (e) => { e.preventDefault(); el.drop.classList.add("drag"); }));
  ["dragleave", "dragend"].forEach((t) => el.drop.addEventListener(t, () => el.drop.classList.remove("drag")));
  el.drop.addEventListener("drop", (e) => {
    e.preventDefault();
    el.drop.classList.remove("drag");
    const files = e.dataTransfer && e.dataTransfer.files;
    if (!files || !files.length) { clientError("Please drop an image file."); return; }
    if (files.length > 1) { clientError("Please drop one image at a time."); return; }
    setFile(files[0]);
  });
  // Dropping a file anywhere else must not navigate away from the page.
  ["dragover", "drop"].forEach((t) => window.addEventListener(t, (e) => e.preventDefault()));
  el.remove.addEventListener("click", reset);
  el.analyze.addEventListener("click", analyze);
  $("retry-btn").addEventListener("click", analyze);

  // -------------------------------------------------------------- request
  function messageFor(status, data) {
    if (data && data.error) return data.error;
    if (status === 413) return `The image is too large. Please upload an image under ${MAX_MB} MB.`;
    if (status === 429) return "Too many requests. Please wait a moment and try again.";
    if (status === 503) return "The analysis service is temporarily unavailable. Please try again in a minute.";
    return "We couldn't process this image. Please try another photo.";
  }

  // Each step is a real request: /api/verify runs the image and cauliflower-leaf
  // checks; /api/predict (which repeats those checks on the server) runs the disease model.
  function step(name, state) {
    const li = document.querySelector(`#steps [data-step="${name}"]`);
    if (li) li.dataset.state = state;
  }

  function loadingText(title, sub) {
    $("loading-title").textContent = title;
    $("loading-sub").textContent = sub;
  }

  async function post(url, timeoutMs) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), timeoutMs);
    try {
      const fd = new FormData();
      fd.append("image", file, file.name);
      const res = await fetch(url, { method: "POST", body: fd, signal: ctrl.signal });
      let data = null;
      try { data = await res.json(); } catch { data = null; }
      return { res, data };
    } finally {
      clearTimeout(timer);
    }
  }

  // A refusal (low quality, not a cauliflower leaf) is a normal outcome, not an error.
  function isRefusal(data) {
    return Boolean(data && data.success === false && REFUSALS[data.status]);
  }

  async function analyze() {
    if (busy) return;
    if (!file) { clientError("Please choose a photo of a cauliflower leaf first."); return; }
    busy = true;
    el.analyze.disabled = true;
    ["check", "leaf", "analyze"].forEach((n) => step(n, "pending"));
    step("check", "active");
    loadingText("Checking image…", "Making sure the photo is usable and shows a cauliflower leaf.");
    show("loading");
    try {
      let { res, data } = await post("/api/verify", CHECK_TIMEOUT_MS);
      if (isRefusal(data)) { step("check", "failed"); showRefusal(data); return; }
      if (!res.ok || !data || !data.success) throw new Error(messageFor(res.status, data));
      step("check", "done");
      step("leaf", "done");
      step("analyze", "active");
      loadingText("Analyzing disease…", "Cauliflower leaf detected. Running the disease model and visual analyses.");

      ({ res, data } = await post("/api/predict", ANALYZE_TIMEOUT_MS));
      if (isRefusal(data)) { showRefusal(data); return; }
      if (!res.ok || !data || !data.success) throw new Error(messageFor(res.status, data));
      step("analyze", "done");
      await showResult(data);
    } catch (err) {
      el.errMsg.textContent = err.name === "AbortError"
        ? "The analysis took too long. Please try again."
        : err instanceof TypeError ? "Couldn't reach the server. Check your connection and try again." : err.message;
      show("error");
    } finally {
      busy = false;
      el.analyze.disabled = false;
    }
  }

  // --------------------------------------------------------------- results
  function showRefusal(data) {
    $("refusal-img").src = previewUrl;
    $("refusal-title").textContent = data.title;
    $("refusal-msg").textContent = data.message;
    $("refusal-badge").textContent = REFUSALS[data.status] || "Not analyzed";
    $("refusal-reasons").innerHTML = (data.reasons || []).map((r) => `<li>${CC.escapeHtml(r)}</li>`).join("");
    show("refusal");
    el.refusal.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  async function showResult(data) {
    const entry = {
      id: `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`,
      ts: Date.now(),
      fileName: file.name,
      image: (await CC.downscale(previewUrl, 640)) || previewUrl,
      thumb: await CC.downscale(previewUrl, 120, 0.75, true),
      data,
    };
    const saved = CC.history.add(entry);
    try { sessionStorage.setItem("caulicare:current", JSON.stringify(entry)); } catch { /* optional */ }
    const actions = `
      <div class="actions no-print" style="margin-top:24px">
        <button class="btn btn-primary" type="button" id="again-btn">${"Analyze another image"}</button>
        <a class="btn btn-secondary" href="/report?id=${encodeURIComponent(entry.id)}">Download report</a>
        <a class="btn btn-ghost" href="/diseases">View disease library</a>
      </div>
      <p class="tiny no-print" style="margin-top:10px">${saved ? "Saved to your history in this browser only." : "History is unavailable in this browser, so this result was not saved."}</p>`;
    CCResult.render(el.result, entry, { actions });
    show("result");
    $("again-btn").addEventListener("click", () => {
      reset();
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
    el.result.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  show("idle");
})();
