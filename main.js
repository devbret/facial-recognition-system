const $ = (s) => document.querySelector(s);

const SLOTS = ["--s1", "--s2", "--s3", "--s4", "--s5", "--s6", "--s7", "--s8"];

const state = {
  runs: [],
  runDir: null,
  data: null,
  allRuns: null,
  colorMap: new Map(),
  ui: {},
};

let trendToken = 0;

function resetUi() {
  state.ui = {
    threshold: null,
    person: "all",
    matchedOnly: false,
    faceIdx: 0,
    metric: null,
    normalize: false,
    x: null,
    y: null,
    trendStat: "similarity",
    sort: null,
  };
}

function el(tag, attrs = {}, kids = []) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "text") n.textContent = v;
    else if (k === "class") n.className = v;
    else n.setAttribute(k, v);
  }
  for (const kid of [].concat(kids)) if (kid != null) n.append(kid);
  return n;
}

function svgEl(tag, attrs = {}, kids = []) {
  const n = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "text") n.textContent = v;
    else n.setAttribute(k, v);
  }
  for (const kid of [].concat(kids)) if (kid != null) n.append(kid);
  return n;
}

function fmt(v) {
  if (typeof v !== "number" || !isFinite(v)) return String(v);
  if (Number.isInteger(v)) return v.toLocaleString("en-US");
  const a = Math.abs(v);
  const d = a >= 100 ? 1 : a >= 10 ? 2 : a >= 1 ? 3 : 4;
  return v.toFixed(d);
}

function label(key) {
  return key
    .split("_")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

function cssVar(name) {
  return getComputedStyle(document.documentElement)
    .getPropertyValue(name)
    .trim();
}

function hexToRgb(h) {
  h = h.replace("#", "");
  if (h.length === 3) h = [...h].map((c) => c + c).join("");
  const n = parseInt(h, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function mix(a, b, t) {
  const A = hexToRgb(a),
    B = hexToRgb(b);
  return (
    "#" +
    A.map((v, i) =>
      Math.round(v + (B[i] - v) * t)
        .toString(16)
        .padStart(2, "0"),
    ).join("")
  );
}

function inkFor(hex) {
  const [r, g, b] = hexToRgb(hex);
  return (0.299 * r + 0.587 * g + 0.114 * b) / 255 > 0.6
    ? "#0b0b0b"
    : "#ffffff";
}

function divScale() {
  const mid = cssVar("--div-mid"),
    pos = cssVar("--div-pos"),
    neg = cssVar("--div-neg");
  return (t) =>
    t < 0 ? mix(mid, neg, Math.min(1, -t)) : mix(mid, pos, Math.min(1, t));
}

function thr() {
  return state.ui.threshold ?? state.data.threshold;
}

function bestName(f) {
  if (f.best_match !== undefined) return f.best_match;
  return f.name !== "Unknown" ? f.name : null;
}

function dispName(f) {
  const b = bestName(f);
  return b && f.similarity >= thr() ? b : "Unknown";
}

function isMatched(f) {
  return dispName(f) !== "Unknown";
}

function entityOf(f) {
  return bestName(f) || "Unknown";
}

function color(name) {
  return state.colorMap.has(name)
    ? `var(${state.colorMap.get(name)})`
    : "var(--muted)";
}

function extendColors(names) {
  for (const n of names) {
    if (n === "Unknown" || state.colorMap.has(n)) continue;
    if (state.colorMap.size < 8)
      state.colorMap.set(n, SLOTS[state.colorMap.size]);
  }
}

function faceTag(f, i) {
  return `${i + 1}. ${dispName(f)}`;
}

function statValue(f, key) {
  if (key === "similarity") return f.similarity;
  if (key === "detector_confidence") return f.detector_confidence;
  return f.metrics[key];
}

function visibleFaces() {
  return state.data.faces
    .map((f, i) => ({ f, i }))
    .filter(({ f }) => {
      if (state.ui.person !== "all" && entityOf(f) !== state.ui.person)
        return false;
      if (state.ui.matchedOnly && !isMatched(f)) return false;
      return true;
    });
}

function note(text) {
  return el("p", { class: "empty-note", text });
}

function setPanel(id, controls, nodes) {
  const p = document.getElementById("p-" + id);
  p.querySelector(".panel-controls").replaceChildren(
    ...(controls || []).filter(Boolean),
  );
  const body = p.querySelector(".panel-body");
  body.replaceChildren(...[].concat(nodes).filter(Boolean));
  return body;
}

const tip = $("#tooltip");

function tipShow(rows, x, y) {
  tip.replaceChildren(
    ...rows.map((r) =>
      el("div", { class: "tip-row" }, [
        el("span", { class: "tip-v", text: String(r.v) }),
        el("span", { class: "tip-l", text: r.l }),
      ]),
    ),
  );
  tip.hidden = false;
  const pad = 12,
    w = tip.offsetWidth,
    h = tip.offsetHeight;
  let L = x + pad,
    T = y + pad;
  if (L + w > innerWidth - 8) L = x - w - pad;
  if (T + h > innerHeight - 8) T = y - h - pad;
  tip.style.left = Math.max(4, L) + "px";
  tip.style.top = Math.max(4, T) + "px";
}

function tipHide() {
  tip.hidden = true;
}

function attachTip(node, rowsFn) {
  node.addEventListener("pointermove", (e) =>
    tipShow(rowsFn(), e.clientX, e.clientY),
  );
  node.addEventListener("pointerleave", tipHide);
  node.tabIndex = 0;
  node.addEventListener("focus", () => {
    const r = node.getBoundingClientRect();
    tipShow(rowsFn(), r.left + r.width / 2, r.top + r.height / 2);
  });
  node.addEventListener("blur", tipHide);
}

function niceTicks(lo, hi, n) {
  if (lo === hi) {
    lo -= 1;
    hi += 1;
  }
  const step0 = (hi - lo) / Math.max(1, n);
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const norm = step0 / mag;
  const step = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10) * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-6; v += step)
    out.push(+v.toFixed(10));
  return out;
}

function pca2(vectors) {
  const n = vectors.length,
    d = vectors[0].length;
  const mean = new Array(d).fill(0);
  vectors.forEach((v) => v.forEach((x, j) => (mean[j] += x / n)));
  const X = vectors.map((v) => v.map((x, j) => x - mean[j]));
  const mul = (w) => {
    const t = X.map((row) => row.reduce((s, x, j) => s + x * w[j], 0));
    const out = new Array(d).fill(0);
    X.forEach((row, i) => row.forEach((x, j) => (out[j] += x * t[i])));
    return out;
  };
  const norm = (v) => Math.sqrt(v.reduce((s, x) => s + x * x, 0));
  const comps = [],
    evs = [];
  for (let c = 0; c < 2; c++) {
    let w = new Array(d).fill(0).map(() => Math.random() - 0.5);
    for (let it = 0; it < 80; it++) {
      let v = mul(w);
      for (const p of comps) {
        const dp = v.reduce((s, x, j) => s + x * p[j], 0);
        v = v.map((x, j) => x - dp * p[j]);
      }
      const nv = norm(v) || 1;
      w = v.map((x) => x / nv);
    }
    const Cw = mul(w);
    evs.push(
      Math.max(
        0,
        w.reduce((s, x, j) => s + x * Cw[j], 0),
      ),
    );
    comps.push(w);
  }
  const total =
    X.reduce((s, row) => s + row.reduce((q, x) => q + x * x, 0), 0) || 1;
  return {
    scores: X.map((row) =>
      comps.map((p) => row.reduce((s, x, j) => s + x * p[j], 0)),
    ),
    varFrac: evs.map((e) => e / total),
  };
}

async function init() {
  try {
    const res = await fetch("output/runs.json", { cache: "no-store" });
    if (!res.ok) throw new Error(res.status);
    state.runs = await res.json();
    if (!state.runs.length) throw new Error("empty");
    const sel = $("#run-select");
    sel.replaceChildren(
      ...state.runs.map((r) => el("option", { value: r, text: r })),
    );
    sel.addEventListener("change", () => loadRun(sel.value));
    await loadRun(state.runs[0]);
  } catch (e) {
    showFallback();
  }
}

async function loadRun(dir) {
  $("#app").style.opacity = state.data ? ".5" : "1";
  try {
    const res = await fetch(
      `output/${encodeURIComponent(dir)}/biometrics.json`,
      { cache: "no-store" },
    );
    if (!res.ok) throw new Error(res.status);
    setData(await res.json(), dir);
  } catch (e) {
    showFallback();
  }
  $("#app").style.opacity = "1";
}

function setData(data, dir) {
  state.data = data;
  state.runDir = dir;
  state.colorMap = new Map();
  resetUi();
  extendColors(data.faces.map(entityOf));
  $("#app").hidden = false;
  document.querySelector(".filters").hidden = false;
  buildFilters();
  renderDynamic();
}

function showFallback() {
  if (!state.data) {
    $("#app").hidden = true;
    document.querySelector(".filters").hidden = true;
  }
}

function buildFilters() {
  const d = state.data;
  const wrap = $("#extra-filters");
  wrap.replaceChildren();
  const entities = [...new Set(d.faces.map(entityOf))];
  if (entities.length > 1) {
    const sel = el("select", {}, [
      el("option", { value: "all", text: "all people" }),
      ...entities.map((n) => el("option", { value: n, text: n })),
    ]);
    sel.value = state.ui.person;
    sel.addEventListener("change", () => {
      state.ui.person = sel.value;
      renderDynamic();
    });
    wrap.append(
      el("label", { class: "ctl" }, [el("span", { text: "Person" }), sel]),
    );
  }
  const chk = el("input", { type: "checkbox" });
  chk.checked = state.ui.matchedOnly;
  chk.addEventListener("change", () => {
    state.ui.matchedOnly = chk.checked;
    renderDynamic();
  });
  wrap.append(
    el("label", { class: "ctl" }, [chk, el("span", { text: "Matched Only" })]),
  );
  const slider = el("input", {
    type: "range",
    min: "0",
    max: "1",
    step: "0.005",
  });
  slider.value = String(thr());
  const val = el("span", { class: "thr-value", text: fmt(thr()) });
  slider.addEventListener("input", () => {
    state.ui.threshold = parseFloat(slider.value);
    val.textContent = fmt(thr());
    renderDynamic();
  });
  const reset = el("button", { text: "Reset" });
  reset.addEventListener("click", () => {
    state.ui.threshold = null;
    slider.value = String(thr());
    val.textContent = fmt(thr());
    renderDynamic();
  });
  wrap.append(
    el("label", { class: "ctl" }, [
      el("span", { text: "Threshold" }),
      slider,
      val,
    ]),
    reset,
  );
}

function renderDynamic() {
  renderKpis();
  renderFaces();
  renderAnomalies();
  renderSimilarity();
  renderExplorer();
  renderScatter();
  renderEmbeddingMap();
  renderMatrix();
  renderFingerprints();
  renderTrends();
  renderTable();
}

function statTile(lbl, value) {
  return el("div", { class: "tile" }, [
    el("span", { class: "tile-label", text: lbl }),
    el("span", { class: "tile-value", text: String(value) }),
  ]);
}

function renderKpis() {
  const d = state.data;
  const vis = visibleFaces();
  const matched = vis.filter(({ f }) => isMatched(f)).length;
  const avg = (fn) =>
    vis.length ? fmt(vis.reduce((s, { f }) => s + fn(f), 0) / vis.length) : "-";
  $("#kpis").replaceChildren(
    statTile("Images", d.images_analyzed),
    statTile("Faces", vis.length),
    statTile("Matched", matched),
    statTile("Unknown", vis.length - matched),
    statTile(
      "Avg Similarity",
      avg((f) => f.similarity),
    ),
    statTile(
      "Avg Confidence",
      avg((f) => f.detector_confidence),
    ),
  );
}

function faceMedia(f) {
  const media = el("div", { class: "face-media" });
  const img = el("img", { alt: `input ${f.image}`, loading: "lazy" });
  img.addEventListener("error", () => {
    if (!img.dataset.alt && state.runDir) {
      img.dataset.alt = "1";
      img.src = `output/${encodeURIComponent(state.runDir)}/annotated/${encodeURIComponent(f.image)}`;
    } else media.remove();
  });
  img.src = `input/${encodeURIComponent(f.image)}`;
  media.append(img);
  const { width: W, height: H } = f.image_size;
  const svg = svgEl("svg", {
    viewBox: `0 0 ${W} ${H}`,
    preserveAspectRatio: "xMidYMid meet",
  });
  const varName = isMatched(f)
    ? state.colorMap.get(entityOf(f)) || "--muted"
    : "--muted";
  const hex = cssVar(varName);
  svg.append(
    svgEl("rect", {
      x: f.box.x,
      y: f.box.y,
      width: f.box.width,
      height: f.box.height,
      fill: "none",
      stroke: hex,
      "stroke-width": 2,
      "vector-effect": "non-scaling-stroke",
    }),
  );
  const r = Math.max(W, H) * 0.008;
  for (const [px, py] of Object.values(f.landmarks))
    svg.append(
      svgEl("circle", {
        cx: px,
        cy: py,
        r,
        fill: hex,
        stroke: "var(--surface)",
        "stroke-width": 2,
        "vector-effect": "non-scaling-stroke",
      }),
    );
  const name = dispName(f);
  const fs = Math.max(W, H) * 0.032;
  const tw = name.length * fs * 0.6 + fs * 0.8;
  let bx = f.box.x + f.box.width / 2;
  bx = Math.max(tw / 2, Math.min(W - tw / 2, bx));
  let by = f.box.y + f.box.height + fs * 1.35;
  if (by > H - fs * 0.3) by = Math.max(fs * 1.1, f.box.y - fs * 0.55);
  svg.append(
    svgEl("rect", {
      x: bx - tw / 2,
      y: by - fs * 1.05,
      width: tw,
      height: fs * 1.45,
      rx: fs * 0.3,
      fill: hex,
    }),
  );
  svg.append(
    svgEl("text", {
      x: bx,
      y: by,
      "text-anchor": "middle",
      fill: inkFor(hex),
      "font-size": fs,
      "font-weight": "600",
      text: name,
    }),
  );
  media.append(svg);
  return media;
}

function meter(sim) {
  const t = thr();
  const track = el("div", { class: "meter-track" });
  const fill = el("div", { class: "meter-fill" });
  fill.style.width = Math.max(0, Math.min(1, sim)) * 100 + "%";
  const tick = el("div", { class: "meter-tick" });
  tick.style.left = Math.max(0, Math.min(1, t)) * 100 + "%";
  track.append(fill, tick);
  attachTip(track, () => [
    { v: fmt(sim), l: "Similarity" },
    { v: fmt(t), l: "Threshold" },
  ]);
  return el("div", { class: "meter-row" }, [
    el("span", { class: "meter-label", text: "Similarity" }),
    track,
    el("span", { class: "meter-value", text: fmt(sim) }),
  ]);
}

function renderFaces() {
  const vis = visibleFaces();
  if (!vis.length) {
    setPanel("faces", [], [note("No faces match the current filters.")]);
    return;
  }
  state.ui.faceIdx = Math.max(
    0,
    Math.min(vis.length - 1, state.ui.faceIdx || 0),
  );
  const idx = state.ui.faceIdx;
  const prev = el("button", {
    class: "nav-btn",
    text: "<",
    "aria-label": "previous face",
  });
  const next = el("button", {
    class: "nav-btn",
    text: ">",
    "aria-label": "next face",
  });
  prev.disabled = idx === 0;
  next.disabled = idx === vis.length - 1;
  prev.addEventListener("click", () => {
    state.ui.faceIdx--;
    renderFaces();
  });
  next.addEventListener("click", () => {
    state.ui.faceIdx++;
    renderFaces();
  });
  const counter = el("span", {
    class: "ctl",
    text: `${idx + 1} / ${vis.length}`,
  });
  const { f, i } = vis[idx];
  const card = el("div", { class: "face-card" });
  card.append(faceMedia(f));
  const body = el("div", { class: "face-body" });
  body.append(
    el("div", { class: "face-head" }, [
      el("span", { class: "face-name", text: faceTag(f, i) }),
      isMatched(f)
        ? el("span", { class: "badge good", text: "+ Matched" })
        : el("span", { class: "badge", text: "? Unknown" }),
    ]),
  );
  body.append(
    el("div", {
      class: "face-sub",
      text: `${f.image} | face ${f.face} | ${f.image_size.width}x${f.image_size.height}px`,
    }),
  );
  body.append(meter(f.similarity));
  const mini = el("div", { class: "mini" });
  const rows = [
    ["Roll", fmt(f.metrics.roll_degrees) + " deg"],
    ["Symmetry", fmt(f.metrics.bilateral_symmetry)],
    ["Sharpness", fmt(f.metrics.sharpness)],
    ["Face Area", (f.metrics.face_area_ratio * 100).toFixed(1) + "%"],
  ];
  for (const [k, v] of rows)
    mini.append(
      el("span", { class: "k", text: k }),
      el("span", { class: "v", text: v }),
    );
  body.append(mini);
  card.append(body);
  setPanel("faces", [prev, counter, next], [card]);
}

function anomaliesFor(vis) {
  const t = thr();
  const out = [];
  const add = (i, f, sev, text) => out.push({ i, f, sev, text });
  for (const { f, i } of vis) {
    if (!isMatched(f) && bestName(f))
      add(
        i,
        f,
        "crit",
        `below threshold, but closest to ${bestName(f)} (${fmt(f.similarity)})`,
      );
    if (Math.abs(f.similarity - t) <= 0.05)
      add(
        i,
        f,
        "warn",
        `borderline similarity: ${fmt(f.similarity)} vs threshold ${fmt(t)}`,
      );
    if (f.detector_confidence < 0.8)
      add(
        i,
        f,
        "warn",
        `low detector confidence (${fmt(f.detector_confidence)})`,
      );
    if (Math.abs(f.metrics.roll_degrees) > 15)
      add(
        i,
        f,
        "warn",
        `strong head tilt (${fmt(f.metrics.roll_degrees)} deg)`,
      );
    if (f.metrics.bilateral_symmetry < 0.9)
      add(
        i,
        f,
        "warn",
        `asymmetric landmark geometry (${fmt(f.metrics.bilateral_symmetry)}), possible off-angle pose`,
      );
    if (f.metrics.sharpness < 50)
      add(
        i,
        f,
        "warn",
        `blurry face crop (sharpness ${fmt(f.metrics.sharpness)})`,
      );
    if (f.metrics.brightness < 40 || f.metrics.brightness > 215)
      add(
        i,
        f,
        "warn",
        `poor exposure (brightness ${fmt(f.metrics.brightness)})`,
      );
  }
  if (vis.length >= 4) {
    const keys = Object.keys(vis[0].f.metrics);
    for (const k of keys) {
      const vals = vis.map(({ f }) => f.metrics[k]);
      const mean = vals.reduce((s, v) => s + v, 0) / vals.length;
      const sd = Math.sqrt(
        vals.reduce((s, v) => s + (v - mean) ** 2, 0) / vals.length,
      );
      if (!sd) continue;
      vis.forEach(({ f, i }, idx) => {
        const z = (vals[idx] - mean) / sd;
        if (Math.abs(z) >= 2.5)
          add(i, f, "warn", `${label(k)} is an outlier (z = ${fmt(z)})`);
      });
    }
  }
  return out;
}

function renderAnomalies() {
  const items = anomaliesFor(visibleFaces());
  document.getElementById("p-anomalies").hidden = !items.length;
  document.getElementById("board").classList.toggle("no-anoms", !items.length);
  if (!items.length) return;
  setPanel(
    "anomalies",
    [],
    items.map((it) =>
      el("div", { class: "anom-row" }, [
        el("span", {
          class: `anom-ic ${it.sev}`,
          text: it.sev === "crit" ? "!!" : "!",
        }),
        el("span", {
          class: "anom-face",
          text: `${faceTag(it.f, it.i)} | ${it.f.image}`,
        }),
        el("span", { class: "anom-text", text: it.text }),
      ]),
    ),
  );
}

function legend(names) {
  if (names.length < 2) return null;
  return el(
    "div",
    { class: "legend" },
    names.map((n) => {
      const sw = el("span", { class: "swatch" });
      sw.style.background = color(n);
      return el("span", { class: "legend-chip" }, [
        sw,
        el("span", { text: n }),
      ]);
    }),
  );
}

function tableTwin(headers, rows) {
  const t = el("table", {}, [
    el(
      "thead",
      {},
      el(
        "tr",
        {},
        headers.map((h) => el("th", { text: h })),
      ),
    ),
    el(
      "tbody",
      {},
      rows.map((r) =>
        el(
          "tr",
          {},
          r.map((c) =>
            el("td", {
              text: typeof c === "number" ? fmt(c) : String(c),
            }),
          ),
        ),
      ),
    ),
  ]);
  return el("details", { class: "table-view", open: "" }, [
    el("summary", { text: "Table" }),
    el("div", { class: "scroll-y" }, t),
  ]);
}

function rampBar(...stops) {
  const bar = el("div", { class: "ramp" });
  bar.style.background = `linear-gradient(to right, ${stops.join(", ")})`;
  return bar;
}

function barRows(items, domain) {
  const lo = Math.min(0, domain.lo),
    hi = Math.max(0, domain.hi);
  const span = hi - lo || 1;
  const zero = (-lo / span) * 100;
  const wrap = el("div", { class: "bars" });
  for (const it of items) {
    const row = el("div", { class: "bar-row" });
    row.append(
      el("span", {
        class: "bar-label",
        text: it.label,
        title: it.sub || "",
      }),
    );
    const track = el("div", { class: "bar-track" });
    const fill = el("div", { class: "bar-fill" });
    const frac = (Math.abs(it.value) / span) * 100;
    if (it.value >= 0) {
      fill.style.left = zero + "%";
      fill.style.width = frac + "%";
      fill.style.borderRadius = "0 4px 4px 0";
    } else {
      fill.style.left = zero - frac + "%";
      fill.style.width = frac + "%";
      fill.style.borderRadius = "4px 0 0 4px";
    }
    fill.style.background = it.color;
    track.append(fill);
    if (lo < 0) {
      const z = el("div", { class: "bar-zero" });
      z.style.left = zero + "%";
      track.append(z);
    }
    if (it.tick != null) {
      const tk = el("div", { class: "bar-thresh" });
      tk.style.left = ((it.tick - lo) / span) * 100 + "%";
      track.append(tk);
    }
    row.append(track, el("span", { class: "bar-value", text: fmt(it.value) }));
    attachTip(row, () => it.tip);
    wrap.append(row);
  }
  return wrap;
}

function dotPlot(points, xLabel, yLabel) {
  const xs = points.map((p) => p.x),
    ys = points.map((p) => p.y);
  const pad = (a, b) => {
    const p = (b - a) * 0.1 || Math.abs(b) * 0.1 || 1;
    return [a - p, b + p];
  };
  const [xlo, xhi] = pad(Math.min(...xs), Math.max(...xs));
  const [ylo, yhi] = pad(Math.min(...ys), Math.max(...ys));
  const W = 720,
    H = 400,
    m = { l: 64, r: 16, t: 14, b: 52 };
  const X = (v) => m.l + ((v - xlo) / (xhi - xlo)) * (W - m.l - m.r);
  const Y = (v) => H - m.b - ((v - ylo) / (yhi - ylo)) * (H - m.t - m.b);
  const svg = svgEl("svg", {
    viewBox: `0 0 ${W} ${H}`,
    class: "plot",
    role: "img",
    "aria-label": `${xLabel} vs ${yLabel}`,
  });
  for (const t of niceTicks(ylo, yhi, 5)) {
    svg.append(
      svgEl("line", {
        x1: m.l,
        x2: W - m.r,
        y1: Y(t),
        y2: Y(t),
        stroke: "var(--grid)",
        "stroke-width": 1,
      }),
    );
    svg.append(
      svgEl("text", {
        x: m.l - 8,
        y: Y(t) + 3.5,
        "text-anchor": "end",
        class: "axis-text",
        text: fmt(t),
      }),
    );
  }
  for (const t of niceTicks(xlo, xhi, 6)) {
    svg.append(
      svgEl("line", {
        x1: X(t),
        x2: X(t),
        y1: m.t,
        y2: H - m.b,
        stroke: "var(--grid)",
        "stroke-width": 1,
      }),
    );
    svg.append(
      svgEl("text", {
        x: X(t),
        y: H - m.b + 18,
        "text-anchor": "middle",
        class: "axis-text",
        text: fmt(t),
      }),
    );
  }
  svg.append(
    svgEl("line", {
      x1: m.l,
      x2: W - m.r,
      y1: H - m.b,
      y2: H - m.b,
      stroke: "var(--baseline)",
      "stroke-width": 1,
    }),
  );
  svg.append(
    svgEl("line", {
      x1: m.l,
      x2: m.l,
      y1: m.t,
      y2: H - m.b,
      stroke: "var(--baseline)",
      "stroke-width": 1,
    }),
  );
  svg.append(
    svgEl("text", {
      x: m.l + (W - m.l - m.r) / 2,
      y: H - 10,
      "text-anchor": "middle",
      class: "axis-title",
      text: xLabel,
    }),
  );
  const yc = m.t + (H - m.t - m.b) / 2;
  svg.append(
    svgEl("text", {
      x: 14,
      y: yc,
      "text-anchor": "middle",
      class: "axis-title",
      transform: `rotate(-90 14 ${yc})`,
      text: yLabel,
    }),
  );
  for (const p of points) {
    svg.append(
      svgEl("circle", {
        cx: X(p.x),
        cy: Y(p.y),
        r: 5,
        fill: p.color,
        stroke: "var(--surface)",
        "stroke-width": 2,
      }),
    );
    const hit = svgEl("circle", {
      cx: X(p.x),
      cy: Y(p.y),
      r: 16,
      fill: "transparent",
    });
    attachTip(hit, () => p.tip);
    svg.append(hit);
  }
  return svg;
}

function renderSimilarity() {
  const vis = visibleFaces();
  if (!vis.length) {
    setPanel("similarity", [], [note("No faces match the current filters.")]);
    return;
  }
  const names = [...new Set(vis.map(({ f }) => entityOf(f)))];
  setPanel(
    "similarity",
    [],
    [
      legend(names),
      barRows(
        vis.map(({ f, i }) => ({
          label: faceTag(f, i),
          sub: f.image,
          value: f.similarity,
          color: color(entityOf(f)),
          tick: thr(),
          tip: [
            { v: fmt(f.similarity), l: "Similarity" },
            { v: fmt(f.detector_confidence), l: "Detector Confidence" },
            { v: isMatched(f) ? "Matched" : "Unknown", l: f.image },
          ],
        })),
        { lo: 0, hi: 1 },
      ),
      tableTwin(
        ["#", "Name", "Image", "Similarity", "Detector Confidence", "Matched"],
        vis.map(({ f, i }) => [
          i + 1,
          dispName(f),
          f.image,
          f.similarity,
          f.detector_confidence,
          isMatched(f) ? "yes" : "no",
        ]),
      ),
    ],
  );
}

function renderExplorer() {
  const vis = visibleFaces();
  if (!vis.length) {
    setPanel("explorer", [], [note("No faces match the current filters.")]);
    return;
  }
  const metricKeys = Object.keys(vis[0].f.metrics);
  if (!state.ui.metric || !metricKeys.includes(state.ui.metric))
    state.ui.metric = metricKeys[0];
  const sel = el(
    "select",
    {},
    metricKeys.map((k) => el("option", { value: k, text: label(k) })),
  );
  sel.value = state.ui.metric;
  sel.addEventListener("change", () => {
    state.ui.metric = sel.value;
    renderExplorer();
  });
  const chk = el("input", { type: "checkbox" });
  chk.checked = state.ui.normalize;
  chk.addEventListener("change", () => {
    state.ui.normalize = chk.checked;
    renderExplorer();
  });
  const names = [...new Set(vis.map(({ f }) => entityOf(f)))];
  const k = state.ui.metric;
  const raw = vis.map(({ f }) => f.metrics[k]);
  const maxAbs = Math.max(...raw.map(Math.abs)) || 1;
  const view = (v) => (state.ui.normalize ? v / maxAbs : v);
  setPanel(
    "explorer",
    [
      el("label", { class: "ctl" }, [el("span", { text: "Metric" }), sel]),
      el("label", { class: "ctl" }, [chk, el("span", { text: "Relative" })]),
    ],
    [
      legend(names),
      barRows(
        vis.map(({ f, i }) => ({
          label: faceTag(f, i),
          sub: f.image,
          value: view(f.metrics[k]),
          color: color(entityOf(f)),
          tip: [
            { v: fmt(f.metrics[k]), l: label(k) },
            ...(state.ui.normalize
              ? [{ v: fmt(view(f.metrics[k])), l: "Relative To Max" }]
              : []),
            { v: f.image, l: "Image" },
          ],
        })),
        {
          lo: Math.min(...raw.map(view)),
          hi: Math.max(...raw.map(view)),
        },
      ),
      tableTwin(
        ["#", "Name", "Image", label(k)],
        vis.map(({ f, i }) => [i + 1, dispName(f), f.image, f.metrics[k]]),
      ),
    ],
  );
}

function renderScatter() {
  const vis = visibleFaces();
  if (!vis.length) {
    setPanel("scatter", [], [note("No faces match the current filters.")]);
    return;
  }
  const metricKeys = Object.keys(vis[0].f.metrics);
  const axisKeys = ["similarity", "detector_confidence", ...metricKeys];
  if (!state.ui.x || !axisKeys.includes(state.ui.x))
    state.ui.x = axisKeys.includes("roll_degrees")
      ? "roll_degrees"
      : axisKeys[0];
  if (!state.ui.y || !axisKeys.includes(state.ui.y)) state.ui.y = "similarity";
  const mkSel = (key, val) => {
    const s = el(
      "select",
      {},
      axisKeys.map((k) => el("option", { value: k, text: label(k) })),
    );
    s.value = val;
    s.addEventListener("change", () => {
      state.ui[key] = s.value;
      renderScatter();
    });
    return s;
  };
  const names = [...new Set(vis.map(({ f }) => entityOf(f)))];
  const kx = state.ui.x,
    ky = state.ui.y;
  setPanel(
    "scatter",
    [
      el("label", { class: "ctl" }, [
        el("span", { text: "X" }),
        mkSel("x", kx),
      ]),
      el("label", { class: "ctl" }, [
        el("span", { text: "Y" }),
        mkSel("y", ky),
      ]),
    ],
    [
      legend(names),
      dotPlot(
        vis.map(({ f, i }) => ({
          x: statValue(f, kx),
          y: statValue(f, ky),
          color: color(entityOf(f)),
          tip: [
            { v: fmt(statValue(f, kx)), l: label(kx) },
            { v: fmt(statValue(f, ky)), l: label(ky) },
            { v: faceTag(f, i), l: f.image },
          ],
        })),
        label(kx),
        label(ky),
      ),
      tableTwin(
        ["#", "Name", "Image", label(kx), label(ky)],
        vis.map(({ f, i }) => [
          i + 1,
          dispName(f),
          f.image,
          statValue(f, kx),
          statValue(f, ky),
        ]),
      ),
    ],
  );
}

function renderEmbeddingMap() {
  const vis = visibleFaces();
  if (vis.length < 2) {
    setPanel("pca", [], [note("Needs at least two visible faces.")]);
    return;
  }
  const names = [...new Set(vis.map(({ f }) => entityOf(f)))];
  const { scores, varFrac } = pca2(vis.map(({ f }) => f.embedding));
  setPanel(
    "pca",
    [],
    [
      legend(names),
      dotPlot(
        vis.map(({ f, i }, idx) => ({
          x: scores[idx][0],
          y: scores[idx][1],
          color: color(entityOf(f)),
          tip: [
            { v: faceTag(f, i), l: f.image },
            { v: fmt(f.similarity), l: "Similarity" },
            {
              v: `${fmt(scores[idx][0])}, ${fmt(scores[idx][1])}`,
              l: "PC1, PC2",
            },
          ],
        })),
        `PC1 (${(varFrac[0] * 100).toFixed(0)}% of variance)`,
        `PC2 (${(varFrac[1] * 100).toFixed(0)}% of variance)`,
      ),
      tableTwin(
        ["#", "Name", "Image", "PC1", "PC2"],
        vis.map(({ f, i }, idx) => [
          i + 1,
          dispName(f),
          f.image,
          scores[idx][0],
          scores[idx][1],
        ]),
      ),
    ],
  );
}

function renderMatrix() {
  const vis = visibleFaces();
  if (vis.length < 2) {
    setPanel("matrix", [], [note("Needs at least two visible faces.")]);
    return;
  }
  const n = vis.length,
    scale = divScale();
  const grid = el("div", { class: "matrix" });
  grid.style.gridTemplateColumns = `auto repeat(${n},minmax(24px,56px))`;
  grid.append(el("div"));
  const hdr = ({ f, i }) =>
    el("div", {
      class: "mx-h",
      text: `F${i + 1}`,
      title: `${faceTag(f, i)} | ${f.image}`,
    });
  for (const v of vis) grid.append(hdr(v));
  for (const a of vis) {
    grid.append(hdr(a));
    for (const b of vis) {
      const sim = a.f.embedding.reduce(
        (s, v, k) => s + v * b.f.embedding[k],
        0,
      );
      const bg = scale(Math.max(-1, Math.min(1, sim)));
      const cell = el("div", { class: "mx-cell" });
      cell.style.background = bg;
      if (n <= 10) {
        cell.textContent = sim.toFixed(2);
        cell.style.color = inkFor(bg);
      }
      attachTip(cell, () => [
        { v: fmt(sim), l: "Cosine Similarity" },
        { v: faceTag(a.f, a.i), l: a.f.image },
        { v: faceTag(b.f, b.i), l: b.f.image },
      ]);
      grid.append(cell);
    }
  }
  const rows = [];
  for (let i = 0; i < n; i++)
    for (let j = i; j < n; j++)
      rows.push([
        `F${vis[i].i + 1} ${dispName(vis[i].f)}`,
        `F${vis[j].i + 1} ${dispName(vis[j].f)}`,
        vis[i].f.embedding.reduce(
          (s, v, k) => s + v * vis[j].f.embedding[k],
          0,
        ),
      ]);
  setPanel(
    "matrix",
    [],
    [
      grid,
      el("div", { class: "ramp-row" }, [
        el("span", { text: "-1" }),
        rampBar(scale(-1), scale(0), scale(1)),
        el("span", { text: "+1" }),
      ]),
      tableTwin(["Face A", "Face B", "Cosine Similarity"], rows),
    ],
  );
}

function renderFingerprints() {
  const vis = visibleFaces();
  if (!vis.length) {
    setPanel("fingerprints", [], [note("No faces match the current filters.")]);
    return;
  }
  const scale = divScale();
  const maxAbs =
    Math.max(...vis.flatMap(({ f }) => f.embedding.map(Math.abs))) || 1;
  const nodes = [];
  for (const { f, i } of vis) {
    const row = el("div", { class: "fp-row" });
    row.append(
      el("span", {
        class: "fp-label",
        text: faceTag(f, i),
        title: f.image,
      }),
    );
    const strip = el("div", { class: "strip" });
    f.embedding.forEach((v) => {
      const c = el("span");
      c.style.background = scale(v / maxAbs);
      strip.append(c);
    });
    strip.tabIndex = 0;
    strip.addEventListener("pointermove", (e) => {
      const r = strip.getBoundingClientRect();
      const idx = Math.max(
        0,
        Math.min(
          f.embedding.length - 1,
          Math.floor(((e.clientX - r.left) / r.width) * f.embedding.length),
        ),
      );
      tipShow(
        [
          { v: fmt(f.embedding[idx]), l: `Dimension ${idx}` },
          { v: faceTag(f, i), l: f.image },
        ],
        e.clientX,
        e.clientY,
      );
    });
    strip.addEventListener("pointerleave", tipHide);
    row.append(strip);
    nodes.push(row);
  }
  nodes.push(
    el("div", { class: "ramp-row" }, [
      el("span", { text: `-${fmt(maxAbs)}` }),
      rampBar(scale(-1), scale(0), scale(1)),
      el("span", { text: `+${fmt(maxAbs)}` }),
    ]),
  );
  const dims = vis[0].f.embedding.length;
  const rows = [];
  for (let k = 0; k < dims; k++)
    rows.push([k, ...vis.map(({ f }) => f.embedding[k])]);
  nodes.push(tableTwin(["Dim", ...vis.map(({ f, i }) => faceTag(f, i))], rows));
  setPanel("fingerprints", [], nodes);
}

function runShort(dir) {
  return dir.length >= 19
    ? `${dir.slice(5, 10)} ${dir.slice(11, 16).replace("-", ":")}`
    : dir;
}

async function ensureAllRuns() {
  if (state.allRuns) return state.allRuns;
  const entries = await Promise.all(
    state.runs.map(async (r) => {
      try {
        const res = await fetch(
          `output/${encodeURIComponent(r)}/biometrics.json`,
          { cache: "no-store" },
        );
        return [r, res.ok ? await res.json() : null];
      } catch (e) {
        return [r, null];
      }
    }),
  );
  state.allRuns = entries.filter((e) => e[1]);
  return state.allRuns;
}

function renderTrends() {
  const tok = ++trendToken;
  const single = state.runs.length < 2;
  document.getElementById("p-trends").hidden = single;
  document.getElementById("board").classList.toggle("no-trends", single);
  if (single) return;
  setPanel("trends", [], [note("Loading runs...")]);
  ensureAllRuns().then(() => {
    if (tok === trendToken) fillTrends();
  });
}

function fillTrends() {
  const runs = [...state.allRuns].reverse();
  const d = state.data;
  const metricKeys = d.faces.length ? Object.keys(d.faces[0].metrics) : [];
  const statKeys = ["similarity", "detector_confidence", ...metricKeys];
  if (!statKeys.includes(state.ui.trendStat)) state.ui.trendStat = "similarity";
  const sel = el(
    "select",
    {},
    statKeys.map((k) => el("option", { value: k, text: label(k) })),
  );
  sel.value = state.ui.trendStat;
  sel.addEventListener("change", () => {
    state.ui.trendStat = sel.value;
    renderTrends();
  });
  const controls = [
    el("label", { class: "ctl" }, [el("span", { text: "Mean Of" }), sel]),
  ];
  const key = state.ui.trendStat;
  const ent = (f) =>
    f.best_match !== undefined ? f.best_match || "Unknown" : f.name;
  const persons = [...new Set(runs.flatMap(([, data]) => data.faces.map(ent)))];
  const shown =
    state.ui.person === "all"
      ? persons
      : persons.filter((p) => p === state.ui.person);
  extendColors(shown);
  const series = shown.map((p) => ({
    name: p,
    pts: runs
      .map(([dir, data], xi) => {
        const vals = data.faces
          .filter((f) => ent(f) === p)
          .map((f) => statValue(f, key));
        return vals.length
          ? { xi, y: vals.reduce((s, v) => s + v, 0) / vals.length }
          : null;
      })
      .filter(Boolean),
  }));
  const allY = series.flatMap((s) => s.pts.map((p) => p.y));
  if (!allY.length) {
    setPanel("trends", controls, [note("No data for the current filters.")]);
    return;
  }
  const padF = (a, b) => {
    const p = (b - a) * 0.12 || Math.abs(b) * 0.1 || 1;
    return [a - p, b + p];
  };
  const [ylo, yhi] = padF(Math.min(...allY), Math.max(...allY));
  const W = 1180,
    H = 340,
    m = { l: 64, r: 20, t: 14, b: 46 };
  const n = runs.length;
  const X = (xi) => m.l + (n === 1 ? 0.5 : xi / (n - 1)) * (W - m.l - m.r);
  const Y = (v) => H - m.b - ((v - ylo) / (yhi - ylo)) * (H - m.t - m.b);
  const svg = svgEl("svg", {
    viewBox: `0 0 ${W} ${H}`,
    class: "plot",
    role: "img",
    "aria-label": `${label(key)} across runs`,
  });
  for (const t of niceTicks(ylo, yhi, 5)) {
    svg.append(
      svgEl("line", {
        x1: m.l,
        x2: W - m.r,
        y1: Y(t),
        y2: Y(t),
        stroke: "var(--grid)",
        "stroke-width": 1,
      }),
    );
    svg.append(
      svgEl("text", {
        x: m.l - 8,
        y: Y(t) + 3.5,
        "text-anchor": "end",
        class: "axis-text",
        text: fmt(t),
      }),
    );
  }
  svg.append(
    svgEl("line", {
      x1: m.l,
      x2: W - m.r,
      y1: H - m.b,
      y2: H - m.b,
      stroke: "var(--baseline)",
      "stroke-width": 1,
    }),
  );
  const every = Math.ceil(n / 8);
  runs.forEach(([dir], xi) => {
    if (xi % every) return;
    svg.append(
      svgEl("text", {
        x: X(xi),
        y: H - m.b + 18,
        "text-anchor": "middle",
        class: "axis-text",
        text: runShort(dir),
      }),
    );
  });
  const hair = svgEl("line", {
    y1: m.t,
    y2: H - m.b,
    stroke: "var(--baseline)",
    "stroke-width": 1,
    visibility: "hidden",
  });
  svg.append(hair);
  for (const s of series) {
    const pathD = s.pts
      .map((p, idx) => `${idx ? "L" : "M"}${X(p.xi)},${Y(p.y)}`)
      .join("");
    if (s.pts.length > 1)
      svg.append(
        svgEl("path", {
          d: pathD,
          fill: "none",
          stroke: color(s.name),
          "stroke-width": 2,
          "stroke-linejoin": "round",
          "stroke-linecap": "round",
        }),
      );
    for (const p of s.pts)
      svg.append(
        svgEl("circle", {
          cx: X(p.xi),
          cy: Y(p.y),
          r: 4,
          fill: color(s.name),
          stroke: "var(--surface)",
          "stroke-width": 2,
        }),
      );
  }
  const overlay = svgEl("rect", {
    x: m.l,
    y: m.t,
    width: W - m.l - m.r,
    height: H - m.t - m.b,
    fill: "transparent",
  });
  overlay.addEventListener("pointermove", (e) => {
    const r = overlay.getBoundingClientRect();
    const frac = (e.clientX - r.left) / r.width;
    const xi = Math.max(0, Math.min(n - 1, Math.round(frac * (n - 1))));
    hair.setAttribute("x1", X(xi));
    hair.setAttribute("x2", X(xi));
    hair.setAttribute("visibility", "visible");
    const rows = [{ v: runShort(runs[xi][0]), l: "Run" }];
    for (const s of series) {
      const p = s.pts.find((q) => q.xi === xi);
      if (p) rows.push({ v: fmt(p.y), l: s.name });
    }
    tipShow(rows, e.clientX, e.clientY);
  });
  overlay.addEventListener("pointerleave", () => {
    hair.setAttribute("visibility", "hidden");
    tipHide();
  });
  svg.append(overlay);
  setPanel("trends", controls, [
    legend(series.map((s) => s.name)),
    svg,
    tableTwin(
      ["Run", ...series.map((s) => s.name)],
      runs.map(([dir], xi) => [
        runShort(dir),
        ...series.map((s) => s.pts.find((q) => q.xi === xi)?.y ?? "-"),
      ]),
    ),
  ]);
}

function renderTable() {
  const vis = visibleFaces();
  if (!vis.length) {
    setPanel("table", [], [note("No faces match the current filters.")]);
    return;
  }
  const metricKeys = Object.keys(vis[0].f.metrics);
  const cols = [
    { key: "#", get: ({ i }) => i + 1 },
    { key: "Image", get: ({ f }) => f.image },
    { key: "Face", get: ({ f }) => f.face },
    { key: "Name", get: ({ f }) => dispName(f) },
    { key: "Best Match", get: ({ f }) => bestName(f) || "-" },
    { key: "Matched", get: ({ f }) => (isMatched(f) ? "yes" : "no") },
    { key: "Similarity", get: ({ f }) => f.similarity },
    { key: "Detector Confidence", get: ({ f }) => f.detector_confidence },
    ...metricKeys.map((k) => ({
      key: label(k),
      get: ({ f }) => f.metrics[k],
    })),
  ];
  const sort = state.ui.sort;
  const rows = [...vis];
  if (sort) {
    const col = cols.find((c) => c.key === sort.key);
    if (col)
      rows.sort((a, b) => {
        const va = col.get(a),
          vb = col.get(b);
        const cmp =
          typeof va === "number" && typeof vb === "number"
            ? va - vb
            : String(va).localeCompare(String(vb));
        return cmp * sort.dir;
      });
  }
  const t = el("table", {}, [
    el(
      "thead",
      {},
      el(
        "tr",
        {},
        cols.map((c) => {
          const th = el("th", {
            class: "sortable",
            text:
              c.key +
              (sort && sort.key === c.key ? (sort.dir > 0 ? " ^" : " v") : ""),
          });
          th.addEventListener("click", () => {
            state.ui.sort =
              sort && sort.key === c.key
                ? { key: c.key, dir: -sort.dir }
                : { key: c.key, dir: 1 };
            renderTable();
          });
          return th;
        }),
      ),
    ),
    el(
      "tbody",
      {},
      rows.map((r) =>
        el(
          "tr",
          {},
          cols.map((c) =>
            el("td", {
              text:
                typeof c.get(r) === "number" ? fmt(c.get(r)) : String(c.get(r)),
            }),
          ),
        ),
      ),
    ),
  ]);
  setPanel("table", [], [t]);
}

matchMedia("(prefers-color-scheme: dark)").addEventListener(
  "change",
  () => state.data && renderDynamic(),
);

init();
