// Shared helpers: REST client for the LeafGuard backend + header status pill.
// The frontend never talks to the model directly – only to API_BASE_URL.
(function () {
  const CFG = window.LEAFGUARD_CONFIG || { API_BASE_URL: "http://localhost:8000" };
  const BASE = CFG.API_BASE_URL.replace(/\/$/, "");

  class ApiError extends Error {
    constructor(message, status) { super(message); this.status = status; }
  }

  /** fetch wrapper: JSON in/out, readable error messages, timeout. */
  async function api(path, { method = "GET", body, form, timeoutMs = 20000 } = {}) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), timeoutMs);
    const opts = { method, signal: ctrl.signal, headers: {} };
    if (form) opts.body = form;
    else if (body !== undefined) { opts.body = JSON.stringify(body); opts.headers["Content-Type"] = "application/json"; }
    let res;
    try {
      res = await fetch(BASE + path, opts);
    } catch (e) {
      throw new ApiError(e.name === "AbortError"
        ? "The server took too long to answer. Please try again."
        : "Cannot reach the LeafGuard service. Check that it is running.", 0);
    } finally { clearTimeout(timer); }
    let data = null;
    try { data = await res.json(); } catch (_) { /* non-JSON body */ }
    if (!res.ok) {
      let msg = (data && (data.detail || data.error)) || `Request failed (${res.status})`;
      if (Array.isArray(msg)) msg = msg.map(d => d.msg).join("; ");
      throw new ApiError(msg, res.status);
    }
    return data;
  }

  function el(tag, attrs = {}, ...children) {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") n.className = v;
      else if (k === "text") n.textContent = v;
      else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
      else n.setAttribute(k, v);
    }
    for (const c of children.flat()) if (c != null) n.append(c.nodeType ? c : document.createTextNode(c));
    return n;
  }

  const pct = (x, d = 1) => (x == null ? "–" : (100 * x).toFixed(d) + "%");
  const fmtTime = ms => (ms ? new Date(ms).toLocaleString() : "–");

  async function refreshStatus() {
    const dot = document.getElementById("api-dot");
    const txt = document.getElementById("api-text");
    if (!dot) return;
    try {
      const r = await api("/ready", { timeoutMs: 5000 });
      if (r.ready) {
        const info = await api("/model/info", { timeoutMs: 5000 });
        dot.className = "dot ok";
        txt.textContent = `Online · model v${info.version}`;
      } else {
        dot.className = "dot busy";
        txt.textContent = "Starting up…";
      }
    } catch (e) {
      dot.className = "dot bad";
      txt.textContent = "Offline";
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    refreshStatus();
    setInterval(refreshStatus, 15000);
  });

  window.LG = { api, ApiError, el, pct, fmtTime, CFG, refreshStatus };
})();
