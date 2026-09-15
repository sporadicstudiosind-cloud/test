/* Live probe UI. Talks to /api/chat and draws the routing telemetry. */
const NS = "http://www.w3.org/2000/svg";
const tip = document.getElementById("tooltip");
const $ = (id) => document.getElementById(id);

const el = (t, a = {}, txt) => {
  const n = document.createElementNS(NS, t);
  for (const [k, v] of Object.entries(a)) n.setAttribute(k, v);
  if (txt != null) n.textContent = txt;
  return n;
};
const h = (t, c, txt) => {
  const n = document.createElement(t);
  if (c) n.className = c;
  if (txt != null) n.textContent = txt;
  return n;
};
const si = (n) => (n >= 1e9 ? (n / 1e9).toFixed(2) + " B"
  : n >= 1e6 ? (n / 1e6).toFixed(1) + " M" : n.toLocaleString("en-US"));

const PRESETS = [
  "CHAN|S=0.0020|n=0.030|q=3.00|h=",
  "CHAN|S=0.0035|n=0.025|q=5.00|x2.00|r=",
  "CLAIM|doubling the discharge doubles the flow depth|verdict=",
  "SCENE|cube@1.0,2.0,0.0|do=",
  "FIELD|nu=0.050|T=0.30|next=",
];

let MODELS = [];

function showTip(evt, lines) {
  tip.innerHTML = lines.join("<br>");
  tip.style.opacity = "1";
  let x = evt.clientX + 14, y = evt.clientY + 14;
  const r = tip.getBoundingClientRect();
  if (x + r.width > innerWidth - 8) x = evt.clientX - r.width - 14;
  if (y + r.height > innerHeight - 8) y = evt.clientY - r.height - 14;
  tip.style.left = x + "px"; tip.style.top = y + "px";
}
const hideTip = () => { tip.style.opacity = "0"; };

function colors() {
  const s = getComputedStyle(document.body);
  return { a: s.getPropertyValue("--series-1").trim(),
           b: s.getPropertyValue("--series-2").trim(),
           muted: s.getPropertyValue("--text-muted").trim() };
}

/* horizontal bars: token share per superstack */
function drawDispatch(host, routing, maxDepth) {
  const C = colors();
  host.innerHTML = "";
  host.appendChild(h("p", "fig-title", "Where the router sent this prompt"));
  host.appendChild(h("p", "fig-sub",
    "token share per superstack; the number after each bar is the mean depth reached"));
  const leg = h("div", "legend");
  [["token share", C.a], ["depth reached / " + maxDepth, C.b]].forEach(([label, col]) => {
    const sp = h("span"); const sw = h("span", "swatch"); sw.style.background = col;
    sp.append(sw, document.createTextNode(label)); leg.appendChild(sp);
  });
  host.appendChild(leg);

  const W = 560, rowH = 42, padL = 172, padR = 62, padT = 4;
  const H = padT + routing.length * rowH + 8;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
  const track = W - padL - padR;
  routing.forEach((r, i) => {
    const y0 = padT + i * rowH;
    svg.appendChild(el("text", {
      x: padL - 12, y: y0 + rowH / 2, "text-anchor": "end",
      "dominant-baseline": "middle", class: "vlabel",
    }, r.name.length > 24 ? r.name.slice(0, 23) + "…" : r.name));
    // share bar
    const wShare = Math.max(r.share * track, r.share > 0 ? 2 : 0);
    const g = el("g"); g.classList.add("hit");
    g.appendChild(el("rect", { x: padL, y: y0 + 7, width: Math.max(wShare, 1),
      height: 12, rx: 4, fill: r.share > 0 ? C.a : C.muted, opacity: r.share > 0 ? 1 : .3 }));
    // depth bar (2px gap below)
    const wDepth = maxDepth ? Math.max((r.expected_depth / maxDepth) * track, 0) : 0;
    g.appendChild(el("rect", { x: padL, y: y0 + 21, width: Math.max(wDepth, r.expected_depth > 0 ? 2 : 0),
      height: 8, rx: 3, fill: C.b, opacity: r.expected_depth > 0 ? 1 : .25 }));
    g.appendChild(el("text", { x: padL + Math.max(wShare, wDepth, 2) + 8, y: y0 + rowH / 2,
      "dominant-baseline": "middle", class: "tick" },
      `${(100 * r.share).toFixed(0)}%  ·  ${r.expected_depth.toFixed(1)}L`));
    g.addEventListener("pointermove", (e) => showTip(e, [
      `<b>S${r.stack} — ${r.name}</b>`,
      `tokens routed here: ${r.tokens}`,
      `share: ${(100 * r.share).toFixed(1)}%`,
      `mean depth: ${r.expected_depth.toFixed(2)} of ${maxDepth} layers`,
    ]));
    g.addEventListener("pointerleave", hideTip);
    svg.appendChild(g);
  });
  svg.appendChild(el("line", { x1: padL, x2: padL, y1: padT, y2: H - 6, class: "axis" }));
  host.appendChild(svg);
  const cap = h("figcaption");
  const used = routing.filter((r) => r.tokens > 0).length;
  cap.innerHTML = `<b>${used} of ${routing.length}</b> superstacks received traffic for this
    prompt. Top-2 routing means every token visits two, so the shares sum to 200% of the
    token count — a stack at 0% saw none of it.`;
  host.appendChild(cap);
}

function tile(host, label, value, sub, kind) {
  const t = h("div", "tile" + (kind ? " " + kind : ""));
  t.append(h("div", "label", label), h("div", "value", value));
  if (sub) t.appendChild(h("div", "sub", sub));
  host.appendChild(t);
}

function renderTelemetry(res) {
  const t = res.telemetry;
  const host = $("telemetry-tiles"); host.innerHTML = "";
  tile(host, "Prompt tokens", String(t.prompt_tokens), "bytes, through the codecs");
  tile(host, "Expected ponder loops", t.expected_loops.toFixed(2),
       `requested up to ${t.loops_requested}`);
  tile(host, "Mean focus", t.mean_focus.toFixed(3), "drives depth into the stacks");
  tile(host, "Router entropy", t.router_entropy.toFixed(3),
       `over ${t.routing.length} stacks`);
  tile(host, "Generation speed", `${res.tokens_per_second.toFixed(1)} tok/s`,
       `${res.tokens_generated} tokens in ${res.seconds.toFixed(2)} s`);
  tile(host, "Parameters", si(res.model.parameters),
       res.model.trained ? "trained weights" : "random init", res.model.trained ? "" : "bad");
  drawDispatch($("fig-dispatch"), t.routing, t.max_depth);
}

function addMessage(who, body, meta) {
  const m = h("div", "msg " + who);
  m.append(h("div", "who", who === "user" ? "you" : "iridium-1"),
           h("div", "body", body));
  if (meta) m.appendChild(h("div", "meta", meta));
  $("transcript").appendChild(m);
  m.scrollIntoView({ behavior: "smooth", block: "nearest" });
  return m;
}

function controlsReadout() {
  $("ctl-readout").textContent =
    `${$("tokens").value} tokens · temp ${(+$("temp").value).toFixed(1)} · ` +
    `${$("loops").value} loop${$("loops").value === "1" ? "" : "s"}`;
  const m = MODELS.find((x) => x.name === $("model").value);
  $("model-caveat").textContent = m ? m.caveat : "";
}

async function send() {
  const prompt = $("prompt").value.trim();
  if (!prompt) return;
  $("send").disabled = true;
  addMessage("user", prompt);
  $("prompt").value = "";
  const pending = addMessage("assistant", "…",
    "running the forward pass — first request after a deploy can take ~20 s");
  try {
    const res = await fetch("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt, model: $("model").value,
        max_new_tokens: +$("tokens").value,
        temperature: +$("temp").value,
        loops: +$("loops").value,
      }),
    });
    // Never surface a blank reason. HTTP/2 leaves `statusText` empty, and a
    // proxy error (a 502 while the container restarts) is HTML or a JSON shape
    // this endpoint never produces - both used to render as "request failed:"
    // with nothing after it.
    const raw = await res.text();
    let data;
    try {
      data = JSON.parse(raw);
    } catch {
      throw new Error(
        res.status === 502 || res.status === 503
          ? `${res.status} — the model container is restarting. Give it ~30 s and try again.`
          : `${res.status} — unexpected response: ${raw.slice(0, 120) || "(empty body)"}`
      );
    }
    if (!res.ok) {
      throw new Error(data.error ? `${res.status} — ${data.error}`
                                 : `${res.status} — ${data.message || "request rejected"}`);
    }
    pending.querySelector(".body").textContent =
      data.text && data.text.trim() ? data.text : "(emitted no printable bytes)";
    pending.querySelector(".meta").textContent =
      `${data.model.label} · ${data.tokens_generated} tokens · ` +
      `${data.seconds.toFixed(2)} s · stopped: ${data.stopped} · ` +
      `${data.telemetry.stacks_used}/${data.telemetry.routing.length} stacks used`;
    renderTelemetry(data);
  } catch (err) {
    const why = (err && err.message) ? err.message
      : "the request did not complete (network error or the container went away)";
    pending.querySelector(".body").textContent = "request failed — " + why;
    pending.querySelector(".meta").textContent =
      "nothing was computed; the telemetry panel still shows the last successful run";
  } finally {
    $("send").disabled = false;
    $("prompt").focus();
  }
}

async function boot() {
  const data = await (await fetch("/api/models")).json();
  MODELS = data.models;
  const sel = $("model");
  MODELS.forEach((m) => {
    const o = document.createElement("option");
    o.value = m.name;
    o.textContent = `${m.label} — ${si(m.parameters)}`;
    sel.appendChild(o);
  });
  const presets = $("presets");
  presets.appendChild(h("span", "readout", "try:"));
  PRESETS.forEach((p) => {
    const b = h("button", null, p.length > 34 ? p.slice(0, 33) + "…" : p);
    b.title = p;
    b.addEventListener("click", () => { $("prompt").value = p; $("prompt").focus(); });
    presets.appendChild(b);
  });
  ["tokens", "temp", "loops", "model"].forEach((id) =>
    $(id).addEventListener("input", controlsReadout));
  controlsReadout();
  $("send").addEventListener("click", send);
  $("prompt").addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") send();
  });
  $("theme-toggle").addEventListener("click", () => {
    const cur = document.documentElement.getAttribute("data-theme");
    const next = cur === "dark" ? "light" : cur === "light" ? "dark"
      : (matchMedia("(prefers-color-scheme: dark)").matches ? "light" : "dark");
    document.documentElement.setAttribute("data-theme", next);
  });
  addMessage("assistant",
    "Ready. Pick a preset below or type a prompt — the telemetry panel fills in on the right.",
    "no forward pass run yet");
}

boot();
