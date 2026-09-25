/* QuishGuard UI components (vanilla JS, no dependencies).
   Security rule for every component: text from a QR code is only ever inserted with
   textContent, never as HTML, and links are shown defanged and never made clickable. */
(function(){
"use strict";
const KEY = new URLSearchParams(location.search).get("key") || "";
const NS = "http://www.w3.org/2000/svg";

function el(tag, props, ...kids){
  const e = document.createElement(tag);
  if(props) for(const k in props){ if(k === "style" && typeof props[k] === "object") Object.assign(e.style, props[k]);
    else if(k.startsWith("data-") || k.startsWith("aria-") || k === "role") e.setAttribute(k, props[k]); else e[k] = props[k]; }
  for(const c of kids.flat()) if(c != null && c !== false) e.append(c);
  return e;
}
function svg(tag, attrs, ...kids){ const e = document.createElementNS(NS, tag); for(const k in attrs||{}) e.setAttribute(k, attrs[k]); for(const c of kids) e.append(c); return e; }
function withKey(u){ return KEY ? u + (u.includes("?") ? "&" : "?") + "key=" + encodeURIComponent(KEY) : u; }
function errText(d){ if(!d) return ""; if(typeof d === "string") return d; if(Array.isArray(d)) return d.map(x => x.msg || "").join("; "); return JSON.stringify(d); }
async function api(path, opts = {}){
  const r = await fetch(withKey(path), {...opts, headers: {"X-QG-Key": KEY, ...(opts.headers || {})}});
  const data = await r.json().catch(() => ({}));
  if(!r.ok) throw new Error(errText(data.detail) || ("Error " + r.status));
  return data;
}

/* ---------- icons (24px stroke) ---------- */
const P = {
  shield: "M12 2.5 4.5 5.5v6c0 4.9 3.2 8.9 7.5 10.5 4.3-1.6 7.5-5.6 7.5-10.5v-6z",
  check: "m5 12.5 4.5 4.5L19 7.5",
  link: "M10 14a4.5 4.5 0 0 0 6.4 0l3-3a4.5 4.5 0 0 0-6.4-6.4l-1 1M14 10a4.5 4.5 0 0 0-6.4 0l-3 3a4.5 4.5 0 0 0 6.4 6.4l1-1",
  qr: "M4 4h6v6H4zM14 4h6v6h-6zM4 14h6v6H4zM14 14h2v2h-2zM18 14h2v2h-2zM14 18h2v2h-2zM18 18h2v2h-2z",
  image: "M4 5h16v14H4zM4 16l5-5 4 4 3-3 4 4",
  alert: "M12 3 2 20h20zM12 10v4M12 17v.5",
  info: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zM12 11v6M12 7.5v.5",
  scan: "M4 8V5a1 1 0 0 1 1-1h3M16 4h3a1 1 0 0 1 1 1v3M20 16v3a1 1 0 0 1-1 1h-3M8 20H5a1 1 0 0 1-1-1v-3M4 12h16",
  camera: "M4 8h3l2-3h6l2 3h3v11H4zM12 16.5a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7z",
  photo: "M4 5h16v14H4zM8.5 10.5a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3zM20 15l-5-5-9 9",
  bell: "M6 16V11a6 6 0 1 1 12 0v5l2 2H4zM10 20a2 2 0 0 0 4 0",
  belloff: "M6 16V11a6 6 0 0 1 9.5-4.9M18 11v5l2 2H8M10 20a2 2 0 0 0 4 0M3 3l18 18",
  phone: "M8 2.5h8a1.5 1.5 0 0 1 1.5 1.5v16a1.5 1.5 0 0 1-1.5 1.5H8A1.5 1.5 0 0 1 6.5 20V4A1.5 1.5 0 0 1 8 2.5zM11 18.5h2",
  download: "M12 4v11M7 11l5 5 5-5M5 20h14",
  user: "M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8zM4 21a8 8 0 0 1 16 0",
  search: "M11 18a7 7 0 1 0 0-14 7 7 0 0 0 0 14zM20 20l-3.5-3.5",
  x: "M6 6l12 12M18 6 6 18",
  copy: "M9 9h11v11H9zM5 15V4h11",
  clock: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zM12 7v5l3 2",
  grid: "M4 4h7v7H4zM13 4h7v7h-7zM4 13h7v7H4zM13 13h7v7h-7z",
  list: "M8 6h12M8 12h12M8 18h12M4 6h.5M4 12h.5M4 18h.5",
  chart: "M4 20V10M10 20V4M16 20v-8M22 20H2",
  arrow: "M5 12h14M13 6l6 6-6 6",
  down: "M12 5v14M6 13l6 6 6-6",
  file: "M6 3h8l4 4v14H6zM14 3v4h4",
  globe: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zM3 12h18M12 3c2.5 2.5 3.5 5.5 3.5 9s-1 6.5-3.5 9c-2.5-2.5-3.5-5.5-3.5-9s1-6.5 3.5-9z",
  hash: "M9 3 7 21M17 3l-2 18M4 8.5h16M3.5 15.5h16",
  refresh: "M20 11a8 8 0 0 0-14.9-3M4 5v4h4M4 13a8 8 0 0 0 14.9 3M20 19v-4h-4",
  home: "M3.5 11 12 3.8l8.5 7.2M5.5 9.6V20h5v-5.5h3V20h5V9.6",
  menu: "M4 7h16M4 12h16M4 17h16",
  dots: "M5 12h.01M12 12h.01M19 12h.01",
  back: "M15 5l-7 7 7 7",
  flash: "M13 2.5 4.5 13.5H11l-1 8 8.5-11H12z",
  flashoff: "M13 2.5 10.6 5.6M8.7 8.1l-4.2 5.4H11l-1 8 4.6-6M16.3 12.5l2.2-2H14M3 3l18 18",
  timer: "M12 21a8 8 0 1 0 0-16 8 8 0 0 0 0 16zM12 9v4.2l2.6 1.6M9.5 2.5h5",
  cpu: "M7 7h10v10H7zM10 10h4v4h-4zM10 3.5V7M14 3.5V7M10 17v3.5M14 17v3.5M3.5 10H7M3.5 14H7M17 10h3.5M17 14h3.5",
  lock: "M6 11h12v10H6zM8.5 11V8a3.5 3.5 0 0 1 7 0v3M12 15v2",
  shieldcheck: "M12 2.5 4.5 5.5v6c0 4.9 3.2 8.9 7.5 10.5 4.3-1.6 7.5-5.6 7.5-10.5v-6zM8.8 12l2.2 2.2 4.2-4.4",
  trash: "M4 7h16M9 7V4h6v3M6.5 7l1 13h9l1-13",
};
function icon(name, size = 16, sw = 2){
  const s = svg("svg", {width: size, height: size, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor",
    "stroke-width": sw, "stroke-linecap": "round", "stroke-linejoin": "round", "aria-hidden": "true"});
  s.append(svg("path", {d: P[name] || P.info})); return s;
}
function logo(size = 28){
  const s = svg("svg", {width: size, height: size, viewBox: "0 0 24 24", "aria-hidden": "true"});
  s.append(svg("path", {d: P.shield, fill: "var(--brand)"}));
  s.append(svg("path", {d: "M8.5 8.5h2.5V11H8.5zM13 8.5h2.5V11H13zM8.5 13h2.5v2.5H8.5zM13 13h1.2v1.2H13zM14.8 14.8H16V16h-1.2z", fill: "var(--brand-ink)"}));
  return s;
}

/* ---------- tiers ---------- */
const TIERS = ["Critical", "Phishing", "Suspicious", "Safe"];
const TIER = {
  Safe:       {color: "var(--safe)",  title: "Safe",            head: "No action needed.",                       desc: "No warning signs were found in this code."},
  Suspicious: {color: "var(--susp)",  title: "Suspicious",      head: "Be careful",                              desc: "Something is unusual. Open it only if you trust who placed this code."},
  Phishing:   {color: "var(--phish)", title: "Phishing",        head: "Do not open this link.",                  desc: "The link looks like phishing. The security team has been alerted."},
  Critical:   {color: "var(--crit)",  title: "Critical threat", head: "Do not open or share this QR code.",      desc: "The link is malicious and the code looks tampered with. The security team has been alerted."},
};
const STATUS = {new: "New alert", acknowledged: "Acknowledged", resolved: "Resolved", false_positive: "False positive",
                false_negative: "Missed attack", none: "No alert"};
const scoreColor = v => v == null ? "var(--faint)" : v > 75 ? "var(--crit)" : v > 50 ? "var(--phish)" : v > 25 ? "var(--susp)" : "var(--safe)";

function fmtTime(iso, withDate){ const d = new Date(iso); const t = d.toLocaleTimeString([], {hour: "2-digit", minute: "2-digit", second: "2-digit"});
  return (withDate || d.toDateString() !== new Date().toDateString()) ? d.toLocaleDateString([], {day: "numeric", month: "short"}) + ", " + t : t; }
const fmt = (v, d = 1) => v == null ? "–" : Number(v).toFixed(d);

/* ---------- components ---------- */
function Badge(tier){ return el("span", {className: "badge b-" + tier}, el("i"), tier); }
function StatusBadge(status){ return el("span", {className: "status s-" + status}, el("i"), STATUS[status] || status); }

function ScoreRing(score, tier, size = 120, opts = {}){
  const stroke = Math.max(6, Math.round(size / 13)), r = (size - stroke) / 2, c = 2 * Math.PI * r;
  const v = Math.max(0, Math.min(100, Number(score) || 0));
  const col = opts.color || (TIER[tier] ? TIER[tier].color : scoreColor(v));
  const arc = svg("circle", {class: "arc", cx: size/2, cy: size/2, r, fill: "none", stroke: col, "stroke-width": stroke,
    "stroke-linecap": "round", "stroke-dasharray": c, "stroke-dashoffset": c});
  const s = svg("svg", {width: size, height: size},
    svg("circle", {cx: size/2, cy: size/2, r, fill: "none", stroke: opts.track || "var(--border)", "stroke-width": stroke}), arc);
  requestAnimationFrame(() => requestAnimationFrame(() => arc.setAttribute("stroke-dashoffset", c * (1 - v / 100))));
  const center = el("div", {className: "c"},
    el("b", {textContent: Math.round(v), style: {fontSize: Math.round(size * .3) + "px"}}),
    el("small", {textContent: "/100", style: {fontSize: Math.max(10, Math.round(size * .1)) + "px"}}));
  if(opts.label !== false && tier) center.append(el("em", {textContent: tier, style: {fontSize: Math.max(9, Math.round(size * .075)) + "px", color: col}}));
  return el("div", {className: "ring", role: "img", "aria-label": `${tier || "Risk"} score ${Math.round(v)} out of 100`}, s, center);
}

function ProgressBar(label, value, opts = {}){
  const fill = el("div", {className: "fill"}); fill.style.background = opts.color || scoreColor(value);
  const lab = el("span", {className: "l"}, label, opts.tag ? el("span", {className: "badge b-neutral", textContent: opts.tag, style: {height: "20px", fontSize: "11px"}}) : null);
  const row = el("div", {className: "pb" + (opts.muted ? " muted" : "")}, lab,
    el("span", {className: "v"}, value == null ? "–" : fmt(value, opts.decimals ?? 1), el("small", {textContent: " / 100"})),
    el("div", {className: "track", role: "progressbar", "aria-valuemin": 0, "aria-valuemax": 100, "aria-valuenow": value ?? 0, "aria-label": label}, fill));
  requestAnimationFrame(() => requestAnimationFrame(() => fill.style.width = (value == null ? 0 : Math.max(1.5, value)) + "%"));
  return row;
}

/* Reasons from the model -> small explanation cards (plain words first, the number second). */
const REASON_TITLES = [
  [/character patterns/, "Suspicious URL character patterns"], [/TLD often used/, "Risky domain ending"],
  [/download file type/, "Links straight to a file download"], [/raw IP address/, "Uses an IP address, not a name"],
  [/explicit port/, "Unusual port in the link"], [/hyphens in host/, "Unusual hostname (hyphens)"],
  [/digits in host|share of digits/, "Many digits in the link"], [/subdomain|host name parts|host length|number of dots/, "Unusual hostname structure"],
  [/domain name length/, "Domain name length"], [/phishing words/, "Phishing words in the link"], [/link shortener/, "Link shortener hides the destination"],
  [/free hosting/, "Free hosting or dynamic DNS"], [/punycode/, "Look-alike letters (punycode)"], [/brand name in subdomain/, "Brand name used as a disguise"],
  [/'@'/, "'@' trick in the link"], [/randomness/, "Random-looking characters"], [/path length|path depth|longest word|number of URL parts|URL length|query/, "Unusual link structure"],
  [/TLD length/, "Domain ending length"], [/script page/, "Script page (.php, .html)"],
];
function reasonCard(text){
  let kind = "info", ic = "info", title = text, detail = "";
  const m = text.match(/^(Link|Image)( \(advisory\))?: (.*?)(?: = ([-\d.]+) \((raises risk|lowers risk)\))?\.?$/);
  if(text.startsWith("Link: on a popular platform")){ title = "Popular platform"; detail = text.replace(/^Link: /, ""); ic = "globe"; }
  else if(m && m[1] === "Link" && m[5]){
    const feat = m[3]; title = (REASON_TITLES.find(([re]) => re.test(feat)) || [0, feat.charAt(0).toUpperCase() + feat.slice(1)])[1];
    kind = m[5] === "raises risk" ? "up" : "down"; ic = "link";
    detail = feat.charAt(0).toUpperCase() + feat.slice(1) + ": " + m[4] + " · " + m[5];
  } else if(m && m[1] === "Image"){ title = "Suspicious QR image characteristics" + (m[2] ? " (advisory)" : ""); detail = m[3]; ic = "image"; kind = m[2] ? "info" : "up"; }
  else if(/could not be read/.test(text)){ title = "Code could not be read normally"; detail = "It may be damaged or altered."; ic = "alert"; kind = "up"; }
  return {kind, node: el("div", {className: "reason " + kind}, el("div", {className: "ic"}, icon(ic, 16)), el("div", null, el("b", {textContent: title}), detail ? el("span", {textContent: detail}) : null))};
}
function ReasonCards(reasons, tier){
  const cards = (reasons || []).map(reasonCard);
  const main = cards.filter(c => c.kind !== "down"), calm = cards.filter(c => c.kind === "down");
  const wrap = el("div", {className: "reasons"});
  const primary = tier === "Safe" ? calm.concat(main) : main;
  const secondary = tier === "Safe" ? [] : calm;
  for(const c of primary) wrap.append(c.node);
  if(!primary.length && !secondary.length) wrap.append(el("p", {className: "muted", textContent: "No specific warning signs.", style: {margin: 0}}));
  if(secondary.length){ const d = el("details", {className: "more"}, el("summary", {textContent: `${secondary.length} reassuring signal${secondary.length > 1 ? "s" : ""}`}), el("div", {className: "reasons"}, secondary.map(c => c.node))); wrap.append(d); }
  return wrap;
}

function HeatmapPanel(photo, heatmap, advisory){
  const fig = (src, cap) => { const f = el("div", {className: "frame", role: "button", tabIndex: 0, "aria-label": "Enlarge " + cap}, el("img", {src: withKey(src), alt: cap}));
    f.onclick = f.onkeydown = e => { if(e.type === "click" || e.key === "Enter") openImage(withKey(src), cap); }; return el("figure", null, f, el("figcaption", null, el("span", {textContent: cap}))); };
  const grid = el("div", {className: "heat"});
  if(photo) grid.append(fig(photo, "Original scan"));
  if(photo && heatmap) grid.append(el("span", {className: "arrow"}, icon("arrow", 18)));
  if(heatmap) grid.append(fig(heatmap, "Anomaly heatmap"));
  const legend = el("div", {className: "legend"}, el("div", {style: {flex: 1, display: "flex", flexDirection: "column", gap: "4px"}},
    el("div", {className: "grad"}), el("div", {className: "lbls"}, el("span", {textContent: "Normal"}), el("span", {textContent: "Unusual"}), el("span", {textContent: "High anomaly"}))));
  const box = el("div", {style: {display: "flex", flexDirection: "column", gap: "12px"}}, grid, heatmap ? legend : null);
  if(advisory) box.append(el("p", {className: "muted", style: {margin: 0, fontSize: "12.5px"}, textContent: "Image analysis is advisory for camera photos: it is shown for review, the verdict comes from the link check."}));
  return box;
}

function Timeline(items){ return el("ol", {className: "tl"}, items.map(it => el("li", {className: it.kind || ""}, el("span", {className: "dot"}), el("div", null, el("b", {textContent: it.title}), el("span", {textContent: it.text}))))); }

function openModal(content, cls = ""){
  const ov = el("div", {className: "overlay"}); const m = el("div", {className: "modal " + cls, role: "dialog", "aria-modal": "true"}, content);
  ov.append(m); const close = () => { ov.remove(); document.removeEventListener("keydown", esc); };
  const esc = e => { if(e.key === "Escape") close(); };
  ov.addEventListener("click", e => { if(e.target === ov || cls === "img") close(); }); document.addEventListener("keydown", esc);
  document.body.append(ov); return close;
}
function openImage(src, alt){ return openModal(el("img", {src, alt}), "img"); }

function toast(title, text, tier, onClick){
  let box = document.getElementById("toasts"); if(!box){ box = el("div", {id: "toasts", "aria-live": "assertive"}); document.body.append(box); }
  const t = el("button", {className: "toast " + (tier || "")}, el("span", {className: "bar"}), el("div", null, el("b", {textContent: title}), el("span", {textContent: text})));
  t.onclick = () => { t.remove(); onClick && onClick(); }; box.prepend(t); setTimeout(() => t.remove(), 12000);
}

async function copyText(text, btn){
  try{ await navigator.clipboard.writeText(text); }
  catch(e){ const ta = el("textarea", {value: text}); document.body.append(ta); ta.select(); try{ document.execCommand("copy"); }catch(_){} ta.remove(); }
  if(btn){ const old = btn.lastChild.textContent; btn.lastChild.textContent = "Copied"; setTimeout(() => btn.lastChild.textContent = old, 1400); }
}

window.QG = {KEY, el, svg, api, withKey, icon, logo, TIERS, TIER, STATUS, scoreColor, fmt, fmtTime,
             Badge, StatusBadge, ScoreRing, ProgressBar, ReasonCards, HeatmapPanel, Timeline, openModal, openImage, toast, copyText};
})();
