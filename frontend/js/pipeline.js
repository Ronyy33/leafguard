// Pipeline console: polls /pipeline/status and /drift on the backend.
(function () {
  const { api, el, pct, fmtTime, CFG } = window.LG;
  const $ = id => document.getElementById(id);
  const STAGES = [
    ["ingest", "Ingest", "Ray Data"],
    ["validate", "Validate", "Ray Data"],
    ["preprocess", "Preprocess", "Ray Data"],
    ["train", "Train", "PyTorch · MLflow"],
    ["evaluate", "Evaluate & register", "MLflow Registry"],
  ];
  let confirmRollback = false;

  function msg(text, kind = "info") {
    const m = $("action-msg");
    m.className = "alert " + kind; m.textContent = text; m.hidden = !text;
  }

  function badge(text, kind) { return el("span", { class: "badge " + kind, text }); }

  function renderDag(trainer) {
    const running = trainer.current;
    const last = running || (trainer.history || [])[0];
    const stages = (last && last.stages) || {};
    const runningStage = running ? Object.keys(stages).pop() : null;
    const dag = $("dag"); dag.replaceChildren();
    STAGES.forEach(([key, name, tool], i) => {
      let cls = "stage", time = "–";
      if (key in stages) { cls += key === runningStage ? " running" : " done"; time = stages[key] + " s"; }
      if (running && key === runningStage) time = "running…";
      dag.append(el("div", { class: cls },
        el("div", { class: "name", text: name }), el("div", { class: "tool", text: tool }),
        el("div", { class: "time", text: time })));
      if (i < STAGES.length - 1) dag.append(el("div", { class: "arrow", "aria-hidden": "true", text: "→" }));
    });
    const r = trainer.reports || {};
    const parts = [];
    if (r.ingest_metrics?.ingest_files_per_sec) parts.push(`ingest ${r.ingest_metrics.ingest_files_per_sec} files/s`);
    if (r.validation?.validate_files_per_sec) parts.push(`validate ${r.validation.validate_files_per_sec} files/s`);
    if (r.preprocess_metrics?.preprocess_images_per_sec) parts.push(`preprocess ${r.preprocess_metrics.preprocess_images_per_sec} img/s`);
    if (r.train_metrics?.train_total_seconds) parts.push(`training ${r.train_metrics.train_total_seconds} s`);
    $("throughput").textContent = parts.length ? "Throughput: " + parts.join(" · ") : "";

    const v = r.validation;
    $("quality").replaceChildren(...(v && v.validated_files ? [
      el("div", { class: "drift-row" }, el("span", { text: "Files checked" }), el("strong", { text: v.validated_files })),
      el("div", { class: "drift-row" }, el("span", { text: "Valid" }), badge(v.valid_files, "ok")),
      el("div", { class: "drift-row" }, el("span", { text: "Rejected" }), badge(v.bad_files, v.bad_files ? "warn" : "ok")),
      el("div", { class: "drift-row" }, el("span", { text: "Issue types" }),
        el("span", { text: Object.entries(v.issues_by_type || {}).map(([k, n]) => `${k}: ${n}`).join(", ") || "none" })),
      el("div", { class: "drift-row" }, el("span", { text: "Classes" }), el("span", { text: Object.keys(v.class_counts || {}).length })),
    ] : [el("p", { class: "muted", text: "No validation report yet." })]));
  }

  function renderTrainer(t) {
    const state = t.state || "unknown";
    $("k-state").replaceChildren(badge(state, { running: "warn", succeeded: "ok", idle: "neutral", failed: "bad" }[state] || "bad"));
    $("retrain-btn").disabled = state === "running" || state === "unreachable";
    const hist = t.history || [];
    const ok = hist.filter(h => h.status === "succeeded").length, bad = hist.filter(h => h.status === "failed").length;
    $("k-runs").textContent = `${ok} / ${bad}`;
    const tb = $("history").querySelector("tbody"); tb.replaceChildren();
    const rows = (t.current ? [t.current] : []).concat(hist);
    if (!rows.length) tb.append(el("tr", {}, el("td", { colspan: 6, class: "muted", text: "No pipeline runs since the trainer started." })));
    rows.forEach(h => tb.append(el("tr", {},
      el("td", { text: fmtTime(h.started * 1000) }),
      el("td", { text: h.reason }),
      el("td", {}, badge(h.status, { running: "warn", succeeded: "ok", failed: "bad" }[h.status])),
      el("td", { text: h.duration_s ? h.duration_s + " s" : "…" }),
      el("td", { text: Object.keys(h.stages || {}).length }),
      el("td", { text: h.eval?.model_version ? `v${h.eval.model_version} ${h.eval.promoted ? "(promoted)" : "(not promoted)"}` : "–" }))));
    const log = $("log");
    if (t.log_tail?.length) {
      const atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 10;
      log.textContent = t.log_tail.join("\n");
      if (atBottom) log.scrollTop = log.scrollHeight;
    }
    renderDag(t);
  }

  function renderRegistry(data) {
    $("k-version").textContent = data.served_version ? "v" + data.served_version : "–";
    const prod = data.model_versions.find(v => v.is_production);
    $("k-f1").textContent = prod?.test_macro_f1 ? pct(+prod.test_macro_f1) : "–";
    const tb = $("versions").querySelector("tbody"); tb.replaceChildren();
    data.model_versions.forEach(v => tb.append(el("tr", {},
      el("td", { text: "v" + v.version }),
      el("td", {}, v.is_production ? badge("production", "ok") : v.rolled_back ? badge("rolled back", "bad") : badge("archived", "neutral")),
      el("td", { text: v.test_macro_f1 ? pct(+v.test_macro_f1) : "–" }),
      el("td", {}, el("a", { href: `${CFG.MLFLOW_URL}/#/experiments/${data.experiment_id}/runs/${v.run_id}`, target: "_blank", rel: "noopener", text: v.run_id.slice(0, 8) })),
      el("td", { text: fmtTime(v.created) }))));
    $("rollback-btn").disabled = data.model_versions.length < 2;

    const rb = $("runs").querySelector("tbody"); rb.replaceChildren();
    data.runs.forEach(r => rb.append(el("tr", {},
      el("td", { text: r.name || r.run_id.slice(0, 8) }),
      el("td", {}, badge(r.status.toLowerCase(), { finished: "ok", running: "warn", failed: "bad" }[r.status.toLowerCase()] || "neutral")),
      el("td", { text: pct(r.val_macro_f1) }),
      el("td", { text: pct(r.test_macro_f1) }),
      el("td", { text: r.duration_s + " s" }),
      el("td", {}, el("code", { text: r.git_commit || "–" })),
      el("td", { text: fmtTime(r.start_time) }))));
  }

  function renderDrift(d) {
    const rows = ["brightness", "blur"].map(f => {
      const z = d.zscores[f] ?? 0;
      const drifting = d.drifting_features.includes(f);
      return el("div", { class: "drift-row" },
        el("span", { text: f === "blur" ? "Sharpness (blur)" : "Brightness" }),
        el("span", {}, el("code", { text: "z = " + z.toFixed(2) }), " ",
          badge(drifting ? "DRIFT" : "normal", drifting ? "bad" : "ok")));
    });
    $("drift").replaceChildren(...rows,
      el("p", { class: "small muted", style: "margin-top:.5rem",
        text: `${d.samples_in_window} of ${d.window_size} recent uploads in the window. Alert when |z| > ${d.threshold} (with at least 20 uploads); a drift alert triggers retraining automatically.` }));
  }

  async function refresh() {
    try {
      const [p, d] = await Promise.all([api("/pipeline/status"), api("/drift")]);
      renderTrainer(p.trainer);
      renderRegistry(p);
      renderDrift(d);
    } catch (e) {
      msg("Could not load pipeline status: " + e.message, "error");
    }
  }

  async function retrain() {
    $("retrain-btn").disabled = true;
    try {
      const r = await api("/retrain/trigger?reason=manual%20(console)", { method: "POST" });
      msg(r.accepted ? "Pipeline started. Progress appears below." : "Not started: " + r.detail, r.accepted ? "ok" : "warn");
    } catch (e) { msg(e.message, "error"); }
    refresh();
  }

  async function rollback() {
    const btn = $("rollback-btn");
    if (!confirmRollback) {        // two-step confirmation instead of confirm()
      confirmRollback = true;
      btn.textContent = "Click again to confirm";
      setTimeout(() => { confirmRollback = false; btn.textContent = "Roll back model"; }, 4000);
      return;
    }
    confirmRollback = false; btn.textContent = "Roll back model"; btn.disabled = true;
    try {
      const r = await api("/model/rollback", { method: "POST" });
      msg(r.detail, "ok");
    } catch (e) { msg(e.message, "error"); }
    refresh(); window.LG.refreshStatus();
  }

  document.addEventListener("DOMContentLoaded", () => {
    $("retrain-btn").addEventListener("click", retrain);
    $("rollback-btn").addEventListener("click", rollback);
    $("drift-reset").addEventListener("click", async () => {
      try { renderDrift(await api("/drift/reset", { method: "POST" })); } catch (e) { msg(e.message, "error"); }
    });
    const links = [
      ["MLflow", CFG.MLFLOW_URL, "Experiments, runs & model registry"],
      ["Grafana", CFG.GRAFANA_URL + "/d/leafguard", "Live operations dashboard"],
      ["Prometheus", CFG.PROMETHEUS_URL + "/alerts", "Metrics & alert rules"],
      ["Alertmanager", CFG.ALERTMANAGER_URL, "Active alerts & routing"],
      ["API docs", CFG.API_BASE_URL + "/docs", "Interactive REST documentation"],
    ];
    $("tool-links").replaceChildren(...links.map(([n, u, d]) =>
      el("a", { href: u, target: "_blank", rel: "noopener" }, el("strong", { text: n }), el("span", { text: d }))));
    $("grafana").src = `${CFG.GRAFANA_URL}/d/leafguard?orgId=1&refresh=5s&kiosk`;
    refresh();
    setInterval(refresh, 5000);
  });
})();
