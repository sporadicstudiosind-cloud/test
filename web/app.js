/* Iridium-1 explorer. Hand-rolled SVG: no library, no network, no build step. */
const NS = "http://www.w3.org/2000/svg";
const tip = document.getElementById("tooltip");

const fmt = {
  int: (n) => n.toLocaleString("en-US"),
  si(n, digits = 2) {
    const u = [[1e12, "T"], [1e9, "B"], [1e6, "M"], [1e3, "k"]];
    for (const [s, suf] of u) if (Math.abs(n) >= s) return (n / s).toFixed(digits) + " " + suf;
    return n.toFixed(0);
  },
  bytes(n) {
    const u = [[1e12, "TB"], [1e9, "GB"], [1e6, "MB"], [1e3, "kB"]];
    for (const [s, suf] of u) if (Math.abs(n) >= s) return (n / s).toFixed(2) + " " + suf;
    return n.toFixed(0) + " B";
  },
  sci(n) {
    if (n === 0) return "0";
    const e = Math.floor(Math.log10(Math.abs(n)));
    return (n / Math.pow(10, e)).toFixed(2) + "e" + e;
  },
  pct: (x) => (100 * x).toFixed(1) + " %",
};

const el = (tag, attrs = {}, text) => {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (text != null) n.textContent = text;
  return n;
};
const html = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};

function showTip(evt, lines) {
  tip.innerHTML = lines.join("<br>");
  tip.style.opacity = "1";
  const pad = 14;
  let x = evt.clientX + pad, y = evt.clientY + pad;
  const r = tip.getBoundingClientRect();
  if (x + r.width > innerWidth - 8) x = evt.clientX - r.width - pad;
  if (y + r.height > innerHeight - 8) y = evt.clientY - r.height - pad;
  tip.style.left = x + "px"; tip.style.top = y + "px";
}
const hideTip = () => { tip.style.opacity = "0"; };

function hoverable(node, lines) {
  node.classList.add("hit");
  node.addEventListener("pointermove", (e) => showTip(e, lines));
  node.addEventListener("pointerleave", hideTip);
  return node;
}

/* ---------- chart primitives ---------- */

function figureShell(host, title, sub, series) {
  host.innerHTML = "";
  if (title) host.appendChild(html("p", "fig-title", title));
  if (sub) host.appendChild(html("p", "fig-sub", sub));
  if (series && series.length > 1) {
    const leg = html("div", "legend");
    series.forEach((s) => {
      const span = html("span");
      const sw = html("span", "swatch");
      sw.style.background = s.color;
      span.append(sw, document.createTextNode(s.label));
      leg.appendChild(span);
    });
    host.appendChild(leg);
  }
  return host;
}

/** Horizontal grouped bars. Magnitude across a few named categories. */
function barsH(host, { title, sub, rows, series, valueFmt, caption, logScale }) {
  figureShell(host, title, sub, series);
  const W = 640, rowH = series.length > 1 ? 40 : 30, padL = 168, padR = 74, padT = 6;
  const H = padT + rows.length * rowH + 16;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
  const max = Math.max(...rows.flatMap((r) => series.map((s) => r[s.key] || 0)));
  const scale = (v) => {
    if (!logScale) return max > 0 ? ((W - padL - padR) * v) / max : 0;
    if (v <= 0) return 0;
    const lo = Math.log10(Math.max(1, max / 1e7)), hi = Math.log10(max);
    return Math.max(2, ((W - padL - padR) * (Math.log10(v) - lo)) / (hi - lo));
  };
  const barH = series.length > 1 ? 11 : 14;
  rows.forEach((r, i) => {
    const y0 = padT + i * rowH;
    svg.appendChild(el("text", {
      x: padL - 12, y: y0 + rowH / 2, "text-anchor": "end",
      "dominant-baseline": "middle", class: "vlabel",
    }, r.label));
    series.forEach((s, j) => {
      const v = r[s.key] || 0;
      const w = scale(v);
      // 2px gap between adjacent bars in a group
      const y = y0 + (rowH - series.length * barH - (series.length - 1) * 2) / 2
              + j * (barH + 2);
      const g = el("g");
      g.appendChild(el("rect", {
        x: padL, y, width: Math.max(w, 1.5), height: barH,
        rx: 4, fill: s.color, class: "mark",
      }));
      g.appendChild(el("text", {
        x: padL + Math.max(w, 1.5) + 8, y: y + barH / 2,
        "dominant-baseline": "middle", class: "tick",
      }, valueFmt(v, r)));
      hoverable(g, [`<b>${r.label}</b>`, `${s.label}: ${valueFmt(v, r)}`]);
      svg.appendChild(g);
    });
  });
  svg.appendChild(el("line", {
    x1: padL, x2: padL, y1: padT, y2: H - 14, class: "axis",
  }));
  host.appendChild(svg);
  if (caption) {
    const c = html("figcaption");
    c.innerHTML = caption;
    host.appendChild(c);
  }
}

/** Line chart, one or two series, with a crosshair. */
function lines(host, { title, sub, points, series, xLabel, yLabel, caption, yFmt }) {
  figureShell(host, title, sub, series);
  const W = 560, H = 240, padL = 58, padR = 16, padT = 10, padB = 34;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
  const xs = points.map((p) => p.x);
  const allY = points.flatMap((p) => series.map((s) => p[s.key]).filter((v) => v != null));
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  let y0 = Math.min(...allY), y1 = Math.max(...allY);
  if (y0 === y1) { y0 -= 1; y1 += 1; }
  y0 = Math.min(y0, y0 - (y1 - y0) * 0.06); y1 += (y1 - y0) * 0.08;
  const X = (v) => padL + ((W - padL - padR) * (v - x0)) / (x1 - x0 || 1);
  const Y = (v) => H - padB - ((H - padT - padB) * (v - y0)) / (y1 - y0 || 1);

  for (let i = 0; i <= 4; i++) {
    const v = y0 + ((y1 - y0) * i) / 4;
    svg.appendChild(el("line", { x1: padL, x2: W - padR, y1: Y(v), y2: Y(v), class: "grid" }));
    svg.appendChild(el("text", {
      x: padL - 8, y: Y(v), "text-anchor": "end", "dominant-baseline": "middle", class: "tick",
    }, (yFmt || ((n) => n.toFixed(1)))(v)));
  }
  svg.appendChild(el("line", { x1: padL, x2: W - padR, y1: Y(y0), y2: Y(y0), class: "axis" }));
  [x0, (x0 + x1) / 2, x1].forEach((v) =>
    svg.appendChild(el("text", {
      x: X(v), y: H - padB + 16, "text-anchor": "middle", class: "tick",
    }, Math.round(v).toString())));
  svg.appendChild(el("text", {
    x: (padL + W - padR) / 2, y: H - 4, "text-anchor": "middle", class: "tick",
  }, xLabel));
  if (yLabel) svg.appendChild(el("text", {
    x: 12, y: padT + 4, class: "tick",
  }, yLabel));

  series.forEach((s) => {
    const pts = points.filter((p) => p[s.key] != null);
    const d = pts.map((p, i) => `${i ? "L" : "M"}${X(p.x).toFixed(1)},${Y(p[s.key]).toFixed(1)}`).join(" ");
    svg.appendChild(el("path", { d, fill: "none", stroke: s.color, "stroke-width": 2,
      "stroke-linejoin": "round", "stroke-linecap": "round" }));
    if (pts.length <= 30) pts.forEach((p) => {
      const c = el("circle", { cx: X(p.x), cy: Y(p[s.key]), r: 4, fill: s.color,
        stroke: "var(--surface-1)", "stroke-width": 2, class: "mark" });
      hoverable(c, [`<b>${xLabel} ${p.x}</b>`,
        ...series.filter((q) => p[q.key] != null)
          .map((q) => `${q.label}: ${(yFmt || ((n) => n.toFixed(3)))(p[q.key])}`)]);
      svg.appendChild(c);
    });
  });

  const overlay = el("rect", { x: padL, y: padT, width: W - padL - padR,
    height: H - padT - padB, fill: "transparent" });
  const cross = el("line", { y1: padT, y2: H - padB, class: "grid", opacity: 0,
    stroke: "var(--border-strong)", "stroke-dasharray": "3 3" });
  svg.append(cross, overlay);
  overlay.addEventListener("pointermove", (e) => {
    const r = svg.getBoundingClientRect();
    const vx = x0 + ((e.clientX - r.left) / r.width * W - padL) / (W - padL - padR) * (x1 - x0);
    let best = points[0];
    for (const p of points) if (Math.abs(p.x - vx) < Math.abs(best.x - vx)) best = p;
    cross.setAttribute("x1", X(best.x)); cross.setAttribute("x2", X(best.x));
    cross.setAttribute("opacity", 1);
    showTip(e, [`<b>${xLabel} ${typeof best.x === "number" ? best.x.toFixed(2).replace(/\.00$/, "") : best.x}</b>`,
      ...series.filter((s) => best[s.key] != null)
        .map((s) => `${s.label}: ${(yFmt || ((n) => n.toFixed(4)))(best[s.key])}`)]);
  });
  overlay.addEventListener("pointerleave", () => { cross.setAttribute("opacity", 0); hideTip(); });

  host.appendChild(svg);
  if (caption) { const c = html("figcaption"); c.innerHTML = caption; host.appendChild(c); }
}

function tile(host, label, value, sub, kind) {
  const t = html("div", "tile" + (kind ? " " + kind : ""));
  t.append(html("div", "label", label), html("div", "value", value));
  if (sub) t.appendChild(html("div", "sub", sub));
  host.appendChild(t);
  return t;
}

function table(node, head, rows, rowClass) {
  node.innerHTML = "";
  const thead = document.createElement("thead");
  const tr = document.createElement("tr");
  head.forEach((h) => tr.appendChild(html("th", null, h)));
  thead.appendChild(tr); node.appendChild(thead);
  const tb = document.createElement("tbody");
  rows.forEach((r) => {
    const row = document.createElement("tr");
    if (rowClass) row.className = rowClass(r);
    r.cells.forEach((c) => {
      const td = html("td", c.cls || null);
      if (c.html) td.innerHTML = c.html; else td.textContent = c.text;
      row.appendChild(td);
    });
    tb.appendChild(row);
  });
  node.appendChild(tb);
}

/* ---------- topology diagram ---------- */
function topology(host, cfg) {
  const W = 900, H = 420;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": "Iridium-1 token path" });
  const defs = el("defs");
  const m = el("marker", { id: "arrow", viewBox: "0 0 10 10", refX: 9, refY: 5,
    markerWidth: 6, markerHeight: 6, orient: "auto-start-reverse" });
  m.appendChild(el("path", { d: "M0,0 L10,5 L0,10 z",
    fill: getComputedStyle(document.body).getPropertyValue("--border-strong").trim() }));
  defs.appendChild(m); svg.appendChild(defs);

  const cs = getComputedStyle(document.body);
  const tok = (n) => cs.getPropertyValue(n).trim();
  const PAINT = {
    core: [tok("--series-1"), `color-mix(in srgb, ${tok("--series-1")} 16%, ${tok("--surface-2")})`],
    stack: [tok("--series-3"), `color-mix(in srgb, ${tok("--series-3")} 14%, ${tok("--surface-2")})`],
    router: [tok("--series-2"), `color-mix(in srgb, ${tok("--series-2")} 16%, ${tok("--surface-2")})`],
    "": [tok("--border-strong"), tok("--surface-2")],
  };
  const box = (x, y, w, h, cls, title, sub) => {
    const g = el("g");
    const [stroke, fill] = PAINT[cls] || PAINT[""];
    g.appendChild(el("rect", { x, y, width: w, height: h, rx: 8,
      fill, stroke, "stroke-width": 1.25 }));
    g.appendChild(el("text", { x: x + w / 2, y: y + (sub ? h / 2 - 6 : h / 2),
      "text-anchor": "middle", "dominant-baseline": "middle",
      fill: tok("--text-primary"), "font-size": 12.5, "font-weight": 600 }, title));
    if (sub) g.appendChild(el("text", { x: x + w / 2, y: y + h / 2 + 12,
      "text-anchor": "middle", "dominant-baseline": "middle",
      fill: tok("--text-secondary"), "font-size": 11 }, sub));
    svg.appendChild(g); return g;
  };
  const edge = (d, cls = "") => svg.appendChild(el("path", {
    d, fill: "none", "stroke-width": 1.5, "marker-end": "url(#arrow)",
    stroke: cls === "loop" ? tok("--series-2") : tok("--border-strong"),
    ...(cls === "loop" ? { "stroke-dasharray": "4 3" } : {}),
  }));

  box(330, 14, 240, 40, "", "Omnimodal codecs", "text · image · video · audio · field · geo · action");
  edge("M450,56 L450,80");
  box(300, 84, 300, 48, "core", `Control core — stage I`,
      `layers 1–${cfg.core_layers / 2} · d=${cfg.core_d_model} · owns the causal KV cache`);
  edge("M450,134 L450,158");
  box(330, 160, 240, 44, "router", "Macro-router",
      `top-${cfg.top_k} of ${cfg.n_superstacks} · focus · halt`);

  const n = cfg.n_superstacks, sw = 128, gap = 14;
  const total = n * sw + (n - 1) * gap, startX = (W - total) / 2;
  const names = (cfg.specializations && cfg.specializations.length)
    ? cfg.specializations : Array.from({ length: n }, (_, i) => "stack " + i);
  names
    .forEach((name, i) => {
      const x = startX + i * (sw + gap);
      edge(`M450,206 C450,232 ${x + sw / 2},228 ${x + sw / 2},248`);
      box(x, 250, sw, 56, "stack", "S" + i,
          `${cfg.superstack_layers} layers`);
      const label = name.replace(/_/g, " ");
      svg.appendChild(el("text", { x: x + sw / 2, y: 320, "text-anchor": "middle",
        "font-size": "10", fill: tok("--text-secondary") },
        label.length > 22 ? label.slice(0, 21) + "…" : label));
      edge(`M${x + sw / 2},330 C${x + sw / 2},352 450,348 450,364`);
    });

  box(300, 366, 300, 44, "core", "Control core — stage II",
      `layers ${cfg.core_layers / 2 + 1}–${cfg.core_layers} · integrate · emit`);
  edge(`M600,388 C660,388 700,300 700,230 C700,150 620,110 602,108`, "loop");
  svg.appendChild(el("text", { x: 712, y: 232, "font-size": "10.5",
    fill: tok("--series-2") }, `ponder ×${cfg.max_loops}`));
  svg.appendChild(el("text", { x: 6, y: 264, "font-size": "10.5",
    fill: tok("--text-secondary") }, "each stack keeps"));
  svg.appendChild(el("text", { x: 6, y: 278, "font-size": "10.5",
    fill: tok("--text-secondary") }, "sparse local KV"));
  svg.appendChild(el("text", { x: 6, y: 292, "font-size": "10.5",
    fill: tok("--text-secondary") }, "+ a bridge onto"));
  svg.appendChild(el("text", { x: 6, y: 306, "font-size": "10.5",
    fill: tok("--text-secondary") }, "the core history"));
  host.innerHTML = ""; host.appendChild(svg);
}

/* ---------- page ---------- */
const COLORS = () => {
  const s = getComputedStyle(document.body);
  return {
    a: s.getPropertyValue("--series-1").trim(),
    b: s.getPropertyValue("--series-2").trim(),
    c: s.getPropertyValue("--series-3").trim(),
  };
};

let DATA = null;

async function boot() {
  DATA = await (await fetch("./data.json")).json();
  render();
  document.getElementById("theme-toggle").addEventListener("click", () => {
    const cur = document.documentElement.getAttribute("data-theme");
    const next = cur === "dark" ? "light" : cur === "light" ? "dark"
      : (matchMedia("(prefers-color-scheme: dark)").matches ? "light" : "dark");
    document.documentElement.setAttribute("data-theme", next);
    render();
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
}

function render() {
  const C = COLORS();
  const t = DATA.test1b || {};
  const p = t.parameters || {};
  const v = DATA.validation;

  /* hero chips */
  const chips = document.getElementById("hero-chips");
  chips.innerHTML = "";
  const chip = (text, cls) => {
    const c = html("span", "chip" + (cls ? " " + cls : ""));
    c.innerHTML = text; chips.appendChild(c);
  };
  chip(`<b>${fmt.int(p.total || 0)}</b> parameters built`, "ok");
  chip(`formula − modules = <b>${v.param_formula_delta}</b>`, "ok");
  chip(`<b>${DATA.tests.count}</b> tests passing`, "ok");
  chip(`cache parity <b>${v.cache_parity_fp64_one_loop === 0 ? "bit-exact" : "—"}</b>`, "ok");
  chip(`Taylor–Green <b>${fmt.sci(v.taylor_green_relative_l2)}</b>`, "ok");
  chip(`task accuracy <b>below baseline</b>`, "no");

  /* topology */
  topology(document.getElementById("topology"), t.geometry || {
    core_layers: 16, core_d_model: 768, n_superstacks: 5, superstack_layers: 26,
    top_k: 2, max_loops: 3, specializations: [],
  });

  /* 1B tiles */
  const rt = document.getElementById("rung-tiles"); rt.innerHTML = "";
  const f1 = (t.forward || {}).loops_1 || {};
  const lora = t.lora || {};
  tile(rt, "Total parameters", fmt.int(p.total || 0), "delta 0 vs the formula", "good");
  tile(rt, "Active / token", `${fmt.si(p.active_min || 0)} – ${fmt.si(p.active_max || 0)}`,
       `${(p.gflops_per_token_min || 0).toFixed(2)}–${(p.gflops_per_token_max || 0).toFixed(2)} GFLOP`);
  tile(rt, "Forward throughput", `${(f1.tokens_per_second || 0).toFixed(0)} tok/s`,
       `${t.threads || 4} CPU threads, 1 ponder loop`);
  tile(rt, "Cache parity (fp32)", fmt.sci((t.cache_parity || {}).max_abs_delta_fp32 || 0),
       "fp32 round-off; the fp64 gate is exact");
  tile(rt, "LoRA adapters", fmt.si(lora.adapter_parameters || 0),
       `${fmt.pct(lora.adapter_fraction || 0)} of the model`);
  tile(rt, "Adam state if unfrozen",
       `${((lora.memory_argument || {}).full_adam_state_gb || 0).toFixed(1)} GB`,
       `against ${((lora.memory_argument || {}).machine_ram_gb || 15)} GB of RAM`, "bad");

  /* routing census */
  const before = (lora.routing_before || {}).tokens_per_stack || [];
  const after = (lora.routing_after || {}).tokens_per_stack || [];
  const specs = (t.geometry || {}).specializations || (DATA.ladder.find((r) => r.rung === "test1b") || {}).specializations || [];
  if (before.length) {
    barsH(document.getElementById("fig-routing"), {
      title: "Where the router sends tokens",
      sub: "Five superstacks, before and after training the gate",
      series: [{ key: "before", label: "at initialisation", color: C.b },
               { key: "after", label: `after ${lora.steps} steps`, color: C.a }],
      rows: before.map((b, i) => ({
        label: (specs[i] || "stack " + i).replace(/_/g, " ").slice(0, 22),
        before: b, after: after[i] || 0,
      })),
      valueFmt: (x) => fmt.int(x),
      caption: `At initialisation a fixed random gate gives every token the same top-2, so
        <b>${(lora.routing_before || {}).stacks_used} of 5</b> stacks take all the traffic and the rest are
        dead weight. After ${lora.steps} steps of the balance objective,
        <b>${(lora.routing_after || {}).stacks_used} of 5</b> are in use
        (dispatch entropy ${((lora.routing_before || {}).entropy_nats || 0).toFixed(2)} →
        ${((lora.routing_after || {}).entropy_nats || 0).toFixed(2)} nats of a possible
        ${((lora.routing_after || {}).max_entropy_nats || 1.609).toFixed(2)}).`,
    });
  }

  /* lora loss */
  if ((lora.history || []).length) {
    lines(document.getElementById("fig-lora"), {
      title: "Adapter training at 1 B",
      sub: `${lora.adapted_modules} adapted modules, base weights frozen`,
      points: lora.history.map((h) => ({ x: h.step, total: h.total })),
      series: [{ key: "total", label: "total loss", color: C.a }],
      xLabel: "step", yFmt: (n) => n.toFixed(1),
      caption: `Mean loss fell from <b>${(lora.loss_first_quarter || 0).toFixed(2)}</b> over the first
        quarter to <b>${(lora.loss_last_quarter || 0).toFixed(2)}</b> over the last, in
        ${(lora.seconds || 0).toFixed(0)} s. A short smoke run, not a trained model —
        it establishes that gradients flow end to end at this size, nothing more.`,
    });
  }

  /* ladder */
  const sel = document.getElementById("rung-select");
  if (!sel.options.length) {
    DATA.ladder.forEach((r) => {
      const o = document.createElement("option");
      o.value = r.rung;
      o.textContent = `${r.rung} — ${fmt.si(r.params)} params${r.built ? "  (built)" : ""}`;
      sel.appendChild(o);
    });
    sel.value = "test1b";
    sel.addEventListener("change", render);
    document.getElementById("ctx").addEventListener("input", render);
  }
  renderLadder(C);

  /* verified tiles */
  const vt = document.getElementById("verified-tiles"); vt.innerHTML = "";
  tile(vt, "Cache parity, 1 loop", "bit-exact", "cached decode = uncached forward, fp64", "good");
  tile(vt, "Cache parity, 2 loops", fmt.sci(v.cache_parity_fp64_two_loops), "4 ULP of float64", "good");
  tile(vt, "Taylor–Green", fmt.sci(v.taylor_green_relative_l2), "relative L2 vs the exact solution", "good");
  tile(vt, "Leray projection", fmt.sci(v.leray_idempotent), "idempotency residual", "good");
  tile(vt, "Spectral equivariance", fmt.sci(v.spectral_shift_equivariance), "shift-equivariance error", "good");
  tile(vt, "Parameter formula", "delta 0", "formula vs instantiated modules", "good");

  /* findings */
  table(document.getElementById("findings-table"),
    ["ID", "The plan said", "What is actually true", "Status"],
    DATA.findings.map((f) => ({
      cells: [
        { text: f.id, cls: "num" },
        { text: f.plan, cls: "wrap-cell" },
        { text: f.finding, cls: "wrap-cell" },
        { html: `<span class="status ${f.status}">${f.status}</span>` },
      ],
    })));

  /* waterfall */
  renderWaterfall(C);

  /* results */
  const g = DATA.nano.graded;
  barsH(document.getElementById("fig-graded"), {
    title: "Graded accuracy vs a prompt-ignoring baseline",
    sub: "free-running generation, checked against independent computation",
    series: [{ key: "trained", label: "trained model", color: C.a },
             { key: "baseline", label: "baseline", color: C.b }],
    rows: g.map((r) => ({ label: r.family.replace(/_/g, " "), trained: r.trained, baseline: r.baseline })),
    valueFmt: (x) => x.toFixed(3),
    caption: `Four of five families sit <b>at or below</b> a baseline that ignores the prompt.
      The one head that learned — a 24-way opcode classification conditioned on text and a
      rendered image — went from chance to <b>1.000</b>, which exercises codecs, router,
      superstacks and action head end to end. The field head reached NRMSE
      <b>${DATA.nano.field.trained_nrmse}</b> against persistence at
      <b>${DATA.nano.field.persistence_nrmse}</b>.`,
  });
  const lc = DATA.nano.loss_curve;
  if (lc.length) lines(document.getElementById("fig-loss"), {
    title: "Phase-1 training loss",
    sub: "34 M parameters, 800 steps, CPU",
    points: lc.map((s) => ({ x: s.step, total: s.total, text: s.text })),
    series: [{ key: "total", label: "total", color: C.a },
             { key: "text", label: "text (bytes)", color: C.c }],
    xLabel: "step", yFmt: (n) => n.toFixed(1),
    caption: `Byte cross-entropy fell <b>6.01 → 0.29</b> nats. Low teacher-forced loss with zero
      graded accuracy is the expected shape when an answer needs six consecutive bytes exactly
      right — not a contradiction, and the reason the next run needs a digit-level objective.`,
  });
  document.getElementById("results-note").innerHTML =
    `<b>What this does not establish:</b> any claim about physical competence, numeric reasoning,
     anti-sycophancy or agentic capability. Those are what the tasks were built to measure, and the
     measurement says no. The architecture is verified; the capability is not.`;

  document.getElementById("rung-note").innerHTML =
    `<b>Why LoRA:</b> fp32 Adam over 1.00 B parameters needs about
     ${((lora.memory_argument || {}).full_adam_state_gb || 16).toFixed(0)} GB of optimizer state against
     ${((lora.memory_argument || {}).machine_ram_gb || 15)} GB of RAM here. Freezing the base weights and
     training ${fmt.si(lora.adapter_parameters || 0)} adapter parameters brings that to
     ${((lora.memory_argument || {}).lora_adam_state_gb || 0).toFixed(2)} GB. That is the same path the
     continual-learning design uses, so it is a constraint met by the architecture rather than around it.`;

  document.getElementById("footer-meta").textContent =
    `torch ${t.torch || "—"} · ${t.threads || 4} CPU threads · model built in ` +
    `${(t.build_seconds || 0).toFixed(1)} s · ${DATA.tests.count} tests passing`;
}

function renderLadder(C) {
  const rung = document.getElementById("rung-select").value;
  const r = DATA.ladder.find((x) => x.rung === rung);
  const ctxExp = +document.getElementById("ctx").value;
  const ctx = Math.pow(2, ctxExp);
  document.getElementById("ctx-readout").textContent =
    `${fmt.int(ctx)} tokens → ${fmt.bytes(r.kv_bytes_per_token * ctx)} of KV cache`;

  const lt = document.getElementById("ladder-tiles"); lt.innerHTML = "";
  tile(lt, "Total parameters", fmt.si(r.params), r.name);
  tile(lt, "Core / stacks / codecs",
       `${fmt.si(r.core_params, 1)} / ${fmt.si(r.stack_params, 1)} / ${fmt.si(r.codec_params, 1)}`,
       `${r.core_layers} core layers, ${r.n_stacks}×${r.stack_layers} stack layers`);
  tile(lt, "Active / token", `${fmt.si(r.active_min, 1)} – ${fmt.si(r.active_max, 1)}`,
       `top-${r.top_k}, depth ${r.min_depth}–${r.stack_layers}`);
  tile(lt, "FLOPs / token", `${r.gflops_max.toFixed(2)} GFLOP`, `min ${r.gflops_min.toFixed(2)} · up to ${r.max_loops} loops`);
  tile(lt, "Weights BF16 / MXFP4", `${fmt.bytes(r.bf16_bytes)} / ${fmt.bytes(r.mxfp4_bytes)}`,
       "MXFP4 is 4.25 bits/param, scale included");
  tile(lt, "KV cache", `${fmt.bytes(r.kv_bytes_per_token)}/token`, `${fmt.bytes(r.kv_bytes_per_token * ctx)} at ${fmt.si(ctx, 0)} tokens`);

  barsH(document.getElementById("fig-ladder"), {
    title: "Parameters across the ladder",
    sub: "log scale — filled dots mark the rungs that have been instantiated",
    series: [{ key: "params", label: "total parameters", color: C.a }],
    rows: DATA.ladder.map((x) => ({ label: x.rung + (x.built ? " ●" : ""), params: x.params })),
    valueFmt: (x) => fmt.si(x),
    logScale: true,
    caption: `Three rungs are built and tested: <b>tiny</b> (unit tests), <b>nano</b> (trained),
      <b>test1b</b> (1.00 B, measured on CPU). The rest are costed by the same formulae — and a
      cost model is a model, not a result.`,
  });

  table(document.getElementById("ladder-table"),
    ["Rung", "Params", "Core", "Superstacks", "Active/token", "GFLOP/token", "MXFP4", "Geometry"],
    DATA.ladder.map((x) => ({
      built: x.built,
      cells: [
        { text: x.rung },
        { text: fmt.si(x.params), cls: "num" },
        { text: fmt.si(x.core_params, 1), cls: "num" },
        { text: `${x.n_stacks} × ${x.stack_layers}L`, cls: "num" },
        { text: fmt.si(x.active_max, 1), cls: "num" },
        { text: x.gflops_max.toFixed(2), cls: "num" },
        { text: fmt.bytes(x.mxfp4_bytes), cls: "num" },
        { text: `d=${x.core_d_model}, ${x.core_layers}L core, top-${x.top_k}`, cls: "num" },
      ],
    })), (r) => (r.built ? "built" : ""));

  const geo = DATA.ladder.find((x) => x.rung === "test1b");
  table(document.getElementById("rung-geometry"),
    ["Component", "Value", "Component", "Value"],
    [
      { cells: [{ text: "core layers" }, { text: `${geo.core_layers} (8 + 8 stages)`, cls: "num" },
                { text: "superstacks" }, { text: `${geo.n_stacks} × ${geo.stack_layers} layers`, cls: "num" }] },
      { cells: [{ text: "d_model" }, { text: fmt.int(geo.core_d_model), cls: "num" },
                { text: "attention" }, { text: `${geo.core_heads} Q / ${geo.core_kv_heads} KV heads`, cls: "num" }] },
      { cells: [{ text: "d_ff (SwiGLU)" }, { text: fmt.int(geo.core_d_ff), cls: "num" },
                { text: "bridge stride" }, { text: `every ${geo.cross_stride} layers`, cls: "num" }] },
      { cells: [{ text: "routing" }, { text: `top-${geo.top_k} of ${geo.n_stacks}`, cls: "num" },
                { text: "ponder loop" }, { text: `up to ${geo.max_loops} passes`, cls: "num" }] },
      { cells: [{ text: "focus depth" }, { text: `${geo.min_depth}–${geo.stack_layers} layers`, cls: "num" },
                { text: "spectral blocks" }, { text: "science stack only", cls: "num" }] },
    ]);
}

function renderWaterfall(C) {
  const w = DATA.physics, iv = w.intervention;
  const q = +document.getElementById("q").value;
  const g = 9.80665;
  const hn = Math.pow((q * w.channel.manning) / Math.sqrt(w.channel.slope), 0.6);
  const hc = Math.pow((q * q) / g, 1 / 3);
  document.getElementById("q-readout").textContent =
    `q = ${q.toFixed(2)} m²/s → normal depth ${hn.toFixed(4)} m, critical depth ${hc.toFixed(4)} m`;

  const wt = document.getElementById("waterfall-tiles"); wt.innerHTML = "";
  tile(wt, "Measured depth ratio", iv.depth_ratio_measured.toFixed(6),
       `solver, q ${iv.q_before} → ${iv.q_after}`, "good");
  tile(wt, "Manning's law", iv.depth_ratio_normal_law.toFixed(6), "q^(3/5) — the answer");
  tile(wt, "Critical depth law", iv.depth_ratio_critical_law.toFixed(6), "q^(2/3) — a different question");
  tile(wt, "The naive answer", "2.000000", "wrong by 32 %", "bad");
  tile(wt, "Velocity ratio", iv.velocity_ratio_measured.toFixed(6), "q^(2/5), the other two fifths");
  tile(wt, "Froude number", `${iv.froude_before.toFixed(3)} → ${iv.froude_after.toFixed(3)}`,
       "stays subcritical, so no jump forms");

  lines(document.getElementById("fig-waterfall"), {
    title: "Depth against discharge — two laws, not one",
    sub: `slope ${w.channel.slope}, Manning n = ${w.channel.manning}`,
    points: w.curve.map((p) => ({ x: p.q, normal: p.normal_depth, critical: p.critical_depth })),
    series: [{ key: "normal", label: "normal depth  q^(3/5)", color: C.a },
             { key: "critical", label: "critical depth  q^(2/3)", color: C.c }],
    xLabel: "discharge q (m²/s)", yFmt: (n) => n.toFixed(2),
    caption: `Doubling the discharge multiplies the normal depth by
      <b>${iv.depth_ratio_normal_law.toFixed(4)}</b> and the critical depth by
      <b>${iv.depth_ratio_critical_law.toFixed(4)}</b>. The solver, started far from equilibrium,
      converges to <b>${iv.depth_ratio_measured.toFixed(6)}</b> — recovering Manning's exponent
      without being told it. Discharge closes along the channel to
      ${fmt.sci(iv.discharge_error)}.`,
  });
}

boot();
