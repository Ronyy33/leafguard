// "Check a leaf" page: upload -> /predict -> show result -> /feedback.
(function () {
  const { api, el } = window.LG;
  const MAX_MB = 5;
  const ALLOWED = ["image/jpeg", "image/png", "image/webp"];
  const $ = id => document.getElementById(id);

  let file = null;
  let previewUrl = null;
  let lastPrediction = null;
  let busy = false;

  function showError(msg) {
    $("upload-error").textContent = msg;
    $("upload-error").hidden = !msg;
  }

  function setBusy(b) {
    busy = b;
    $("analyze-btn").disabled = b || !file;
    $("analyze-label").textContent = b ? "Checking…" : "Check this leaf";
    const sp = $("analyze-btn").querySelector(".spinner");
    if (b && !sp) $("analyze-btn").prepend(el("span", { class: "spinner", "aria-hidden": "true" }));
    if (!b && sp) sp.remove();
    $("dropzone").style.pointerEvents = b ? "none" : "";
  }

  function choose(f) {
    showError("");
    if (!f) return;
    if (!ALLOWED.includes(f.type)) {
      showError("That file isn't a photo we can read. Please choose a JPG, PNG or WEBP image.");
      return;
    }
    if (f.size > MAX_MB * 1024 * 1024) {
      showError(`That photo is too large (${(f.size / 1048576).toFixed(1)} MB). The limit is ${MAX_MB} MB.`);
      return;
    }
    file = f;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(f);
    $("preview-img").src = previewUrl;
    $("file-name").textContent = f.name;
    $("file-size").textContent = `${(f.size / 1024).toFixed(0)} KB`;
    $("preview").hidden = false;
    $("analyze-btn").disabled = false;
    $("reset-btn").hidden = false;
  }

  function reset() {
    file = null; lastPrediction = null;
    $("file-input").value = "";
    $("preview").hidden = true;
    $("analyze-btn").disabled = true;
    $("reset-btn").hidden = true;
    $("result").hidden = true;
    $("result-empty").hidden = false;
    showError("");
  }

  function render(r) {
    lastPrediction = r;
    $("result-empty").hidden = true;
    $("result").hidden = false;
    if (!r.looks_like_leaf) {
      $("verdict").className = "verdict unknown";
      $("verdict-icon").textContent = "❓";
      $("verdict-kicker").textContent = "This may not be a leaf – result unreliable";
    } else {
      $("verdict").className = "verdict " + (r.is_healthy ? "healthy" : "disease");
      $("verdict-icon").textContent = r.is_healthy ? "✅" : "⚠️";
      $("verdict-kicker").textContent = r.is_healthy ? "Looks healthy" : "Possible disease detected";
    }
    $("verdict-label").textContent = r.display_name;
    $("conf-text").textContent = (100 * r.confidence).toFixed(1) + "%";
    requestAnimationFrame(() => { $("conf-bar").style.width = (100 * r.confidence).toFixed(1) + "%"; });

    const w = $("warnings"); w.replaceChildren();
    r.warnings.forEach(msg => w.append(el("div", { class: "alert warn", text: msg })));

    const ul = $("top3"); ul.replaceChildren();
    r.top3.forEach(c => ul.append(el("li", {},
      el("div", { class: "row" }, el("span", { text: c.display_name }), el("span", { text: (100 * c.probability).toFixed(1) + "%" })),
      el("div", { class: "bar" }, el("span", { style: `width:${(100 * c.probability).toFixed(1)}%` })))));

    $("res-photo").src = previewUrl;
    if (r.heatmap_png_base64) {
      $("res-heat").src = "data:image/png;base64," + r.heatmap_png_base64;
      $("heat-fig").hidden = false;
    } else $("heat-fig").hidden = true;

    $("fb-yes").disabled = $("fb-no").disabled = false;
    $("fb-correct").hidden = true;
    $("fb-msg").hidden = true;
    $("meta").replaceChildren(
      el("span", { text: `Model v${r.model_version}` }),
      el("span", { text: `${r.latency_ms} ms` }),
      el("span", { text: `via ${r.inference_backend === "mlflow_serve" ? "MLflow model server" : "local fallback"}` }),
      el("span", { text: `ID ${r.prediction_id.slice(0, 8)}` }));
    if (window.matchMedia("(max-width: 860px)").matches) $("result-title").scrollIntoView({ behavior: "smooth" });
  }

  async function analyze() {
    if (!file || busy) return;
    setBusy(true); showError("");
    const form = new FormData();
    form.append("file", file, file.name);
    try {
      render(await api("/predict?explain=true", { method: "POST", form, timeoutMs: 30000 }));
    } catch (e) {
      showError(e.message);
    } finally {
      setBusy(false);
    }
  }

  async function sendFeedback(trueLabel) {
    if (!lastPrediction) return;
    $("fb-yes").disabled = $("fb-no").disabled = $("fb-send").disabled = true;
    const box = $("fb-msg");
    try {
      await api("/feedback", { method: "POST", body: { prediction_id: lastPrediction.prediction_id, true_label: trueLabel } });
      box.className = "alert ok"; box.textContent = "Thank you! Your feedback was recorded.";
      $("fb-correct").hidden = true;
    } catch (e) {
      box.className = "alert error"; box.textContent = e.message;
      $("fb-yes").disabled = $("fb-no").disabled = $("fb-send").disabled = false;
    }
    box.hidden = false;
  }

  async function loadClasses() {
    try {
      const classes = await api("/classes");
      $("fb-select").replaceChildren(...classes.map(c => el("option", { value: c.label, text: c.display_name })));
    } catch (_) { /* status pill already shows offline */ }
  }

  document.addEventListener("DOMContentLoaded", () => {
    const dz = $("dropzone");
    $("file-input").addEventListener("change", e => choose(e.target.files[0]));
    dz.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("file-input").click(); } });
    ["dragenter", "dragover"].forEach(t => dz.addEventListener(t, e => { e.preventDefault(); dz.classList.add("drag"); }));
    ["dragleave", "drop"].forEach(t => dz.addEventListener(t, e => { e.preventDefault(); dz.classList.remove("drag"); }));
    dz.addEventListener("drop", e => choose(e.dataTransfer.files[0]));
    $("analyze-btn").addEventListener("click", analyze);
    $("reset-btn").addEventListener("click", reset);
    $("fb-yes").addEventListener("click", () => sendFeedback(lastPrediction.label));
    $("fb-no").addEventListener("click", () => {
      $("fb-correct").hidden = false;
      $("fb-send").disabled = false;
      const other = lastPrediction.top3.find(c => c.label !== lastPrediction.label);
      if (other) $("fb-select").value = other.label;
    });
    $("fb-send").addEventListener("click", () => sendFeedback($("fb-select").value));
    loadClasses();
  });
})();
