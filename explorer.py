"""The interactive leaderboard: pick the reference model and the series types.

Exports, per stratum, the per-series pairwise verdicts the skaters summarizer
computes (canonical_pairs.csv: the same Diebold-Mariano test and log-score gap
as the published summary, for every pair of roster models), each model's
per-series interval hits (canonical_per_series.csv), and each series' tags
(series_tags.csv, from pre-test history only). The page re-aggregates any
subset in the browser: counts of per-series verdicts, medians of per-series
gaps, pooled coverage. It never scores anything itself.
"""
from __future__ import annotations

import collections
import csv
import json
import os
import sys

TAG_GROUPS = [
    ("Dynamics", ["martingale", "mean-reverting", "momentum"]),
    ("Values", ["continuous", "mixed", "sticky"]),
    ("Shape", ["seasonal", "heavy tails", "vol clustering"]),
    ("Asset class", ["rates", "fx", "credit", "equity", "commodity", "other"]),
]


def export(bench, out_dir, roster_names, aliases=None):
    """Write <out_dir>/data/<stratum>.json; returns the list of strata written.
    aliases: {shown arm: [fallback arms]} -- on a stratum where the shown arm has
    no rows, the first fallback that has rows is shown under its name (the page
    says which run it is through FP_RUN)."""
    aliases = aliases or {}
    tags_by = {}
    for r in csv.DictReader(open(os.path.join(bench, "series_tags.csv"))):
        t = [x for x in r["tags"].split(";") if x] + [r["asset_class"]]
        corpus = r["corpus"]
        tags_by[(corpus, r["series"])] = t
    pairs = collections.defaultdict(lambda: collections.defaultdict(list))
    seen = collections.defaultdict(set)
    for r in csv.DictReader(open(os.path.join(bench, "canonical_pairs.csv"))):
        st = r["study"]
        key = f'{r["a"]}|{r["b"]}'
        pairs[st][key].append((r["series"], r["verdict"], float(r["dLL"]),
                               float(r["crps_ratio"]) if r["crps_ratio"] else None))
        seen[st].add(r["series"])
    cov = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in csv.DictReader(open(os.path.join(bench, "canonical_per_series.csv"))):
        cov[r["study"]][r["model"]][r["series"]] = (int(r["in90"]), int(r["n_steps"]))
    os.makedirs(os.path.join(out_dir, "data"), exist_ok=True)
    written = []
    for st in sorted(pairs):
        corpus = st.split(":")[0]
        series = sorted(seen[st])
        idx = {s: i for i, s in enumerate(series)}
        all_tags = [t for _g, ts in TAG_GROUPS for t in ts]
        tix = {t: i for i, t in enumerate(all_tags)}
        stags = [[tix[t] for t in tags_by.get((corpus, s), []) if t in tix] for s in series]
        P = {k: [[idx[s], v, round(d, 4), None if c is None else round(c, 4)] for s, v, d, c in rows]
             for k, rows in pairs[st].items()}
        C = {m: [[idx[s], h, n] for s, (h, n) in d.items() if s in idx] for m, d in cov[st].items()}
        # aliases: {shown arm: [candidate arms in priority order]}; the first
        # candidate with rows on this stratum is shown under the shown arm's name
        shown_run = {}
        for shown, cands in aliases.items():
            for fb in cands:
                if any(fb in k.split("|") for k in P):
                    if fb != shown:
                        P = {"|".join(shown if x == fb else x for x in k.split("|")): v for k, v in P.items()
                             if shown not in k.split("|")}
                        if fb in C:
                            C[shown] = C.pop(fb)
                    shown_run[shown] = fb
                    break
        P = {k: v for k, v in P.items() if all(x in roster_names for x in k.split("|"))}
        models = [m for m in roster_names if any(m in k.split("|") for k in P)]
        try:
            sys.path.insert(0, bench)
            from learned_split import split_of
            held = [1 if split_of(sid) == "valid" else 0 for sid in series]
        except Exception:                                   # noqa: BLE001
            held = [1] * len(series)
        data = {"stratum": st, "runs": shown_run, "held": held, "series": series, "tags": all_tags, "groups": TAG_GROUPS,
                "series_tags": stags, "models": models, "pairs": P, "cov": C}
        with open(os.path.join(out_dir, "data", st.replace(":", "-") + ".json"), "w") as fh:
            json.dump(data, fh, separators=(",", ":"))
        written.append(st)
    return written


EXPLORER_JS = r"""
(function () {
  var LIC = window.FP_LIC || {}, NAMES = window.FP_NAMES || {}, KIND = window.FP_KIND || {}, STRATA = window.FP_STRATA || [];
  var state = { kind: null, freq: null, ref: 'laplace', tags: {}, data: null, sort: { col: 'net', dir: -1 }, heldOnly: false };
  var cache = {};
  function name(m) { return NAMES[m] || m; }
  function dot(m) { return '<span class="dot ' + (KIND[m] || 'other') + '"></span>'; }
  function median(a) { if (!a.length) return null; a = a.slice().sort(function (x, y) { return x - y; });
    var k = a.length >> 1; return a.length % 2 ? a[k] : 0.5 * (a[k - 1] + a[k]); }
  function licName(m) { return (LIC[m] || ['?'])[0]; }
  function lic(m) { var l = LIC[m]; if (!l) return '<td class="muted">&ndash;</td>';
    return '<td><a class="lic ' + l[1] + '" href="' + l[2] + '" title="' + l[3] + '">' + l[0] + '</a></td>'; }
  function pct(x) { return Math.round(100 * x) + '%'; }
  function stratum() { var s = STRATA.filter(function (x) { return x.kind === state.kind && x.freq === state.freq; })[0]; return s && s.id; }

  function groupsShown(d) {   // asset class only for prices
    return d.groups.filter(function (g) { return g[0] !== 'Asset class' || state.kind === 'Prices'; });
  }
  function selected(d) {
    var tix = {}; d.tags.forEach(function (t, i) { tix[t] = i; });
    var groups = groupsShown(d).map(function (g) { return g[1].filter(function (t) { return state.tags[t]; }); });
    var keep = new Uint8Array(d.series.length);
    for (var i = 0; i < d.series.length; i++) {
      var st = d.series_tags[i], ok = true;
      for (var g = 0; g < groups.length && ok; g++) {
        if (!groups[g].length) continue;
        ok = groups[g].some(function (t) { return st.indexOf(tix[t]) >= 0; });
      }
      if (ok && state.heldOnly && d.held && !d.held[i]) ok = false;
      keep[i] = ok ? 1 : 0;
    }
    return keep;
  }
  function covOf(d, m, keep) { var c = d.cov[m] || [], h = 0, s = 0;
    c.forEach(function (x) { if (keep[x[0]]) { h += x[1]; s += x[2]; } }); return s ? h / s : null; }
  function rows(d, keep) {
    var ref = state.ref, out = [];
    d.models.forEach(function (m) {
      if (m === ref) return;
      var rs = d.pairs[m + '|' + ref], flip = false;
      if (!rs) { rs = d.pairs[ref + '|' + m]; flip = true; }
      if (!rs) return;
      var w = 0, dr = 0, l = 0, dl = [], cr = [];
      rs.forEach(function (r) {
        if (!keep[r[0]]) return;
        var v = r[1]; if (flip) v = v === 'W' ? 'L' : v === 'L' ? 'W' : v;
        if (v === 'W') w++; else if (v === 'L') l++; else dr++;
        dl.push(flip ? -r[2] : r[2]);
        if (r[3] !== null && r[3] > 0) cr.push(flip ? 1 / r[3] : r[3]);
      });
      var n = w + dr + l; if (!n) return;
      out.push({ m: m, n: n, w: w, d: dr, l: l, win: w / n, loss: l / n, net: (w - l) / n, dll: median(dl), crps: median(cr), cov: covOf(d, m, keep) });
    });
    var nref = 0; out.forEach(function (r) { if (r.n > nref) nref = r.n; });
    out.push({ m: ref, ref: true, n: nref, w: 0, d: 0, l: 0, win: 0, loss: 0, net: 0, dll: 0, crps: 1, cov: covOf(d, ref, keep) });
    var c = state.sort.col, dir = state.sort.dir;
    out.sort(function (a, b) {
      var f = c === 'name' ? name : c === 'lic' ? licName : null, x = f ? f(a.m) : a[c], y = f ? f(b.m) : b[c];
      if (x === null) return 1; if (y === null) return -1;
      return (x < y ? -1 : x > y ? 1 : 0) * dir;
    });
    return out;
  }
  function render() {
    var d = state.data; if (!d) return;
    var keep = selected(d), nsel = 0; for (var i = 0; i < keep.length; i++) nsel += keep[i];
    var rs = rows(d, keep);
    var body = rs.map(function (r) {
      if (r.ref) return '<tr class="yard"><td>' + dot(r.m) + '<b>' + name(r.m) + '</b></td><td class="muted" colspan="2">reference</td>' +
        '<td class="num">+0.000</td><td class="num">1.000</td><td class="num">' + (r.cov === null ? '&ndash;' : r.cov.toFixed(2)) + '</td>' + lic(r.m) + '</tr>';
      return '<tr><td>' + dot(r.m) + '<b>' + name(r.m) + '</b></td><td class="num">' + r.n.toLocaleString() + '</td>' +
        '<td><span class="wdl"><i class="w" style="width:' + (100 * r.win).toFixed(1) + '%"></i><i class="d" style="width:' +
        (100 * r.d / r.n).toFixed(1) + '%"></i><i class="l" style="width:' + (100 * r.loss).toFixed(1) + '%"></i></span>' +
        '<span class="wdlnum">' + pct(r.win) + ' / ' + pct(r.loss) + '</span></td>' +
        '<td class="num">' + (r.dll >= 0 ? '+' : '') + r.dll.toFixed(3) + '</td>' +
        '<td class="num">' + (r.crps === null ? '&ndash;' : r.crps.toFixed(3)) + '</td>' +
        '<td class="num">' + (r.cov === null ? '&ndash;' : r.cov.toFixed(2)) + '</td>' + lic(r.m) + '</tr>';
    }).join('');
    document.getElementById('ex-body').innerHTML = body;
    var best = rs.filter(function (r) { return !r.ref; }).sort(function (a, b) { return b.dll - a.dll; }).filter(function (r) { return r.dll > 0 && r.w > r.l; })[0];
    var runs = d.runs || {}, notes = [];
    Object.keys(runs).forEach(function (k) { notes.push(name(k) + ' on this stratum: run ' + runs[k].replace(/^learned\(|\)$/g, '') + '.'); });
    document.getElementById('ex-runs').textContent = notes.join(' ');
    document.getElementById('ex-pick').innerHTML = best
      ? nsel.toLocaleString() + ' series. Best against ' + name(state.ref) + ': ' + dot(best.m) + '<b>' + name(best.m) + '</b>, ' +
        (best.dll >= 0 ? '+' : '') + best.dll.toFixed(3) + ' nats at the median, ahead on ' + pct(best.win) + ', behind on ' + pct(best.loss) + '.'
      : nsel.toLocaleString() + ' series. Nothing beats ' + dot(state.ref) + '<b>' + name(state.ref) + '</b> both at the median and on more series than it loses.';
    Array.prototype.forEach.call(document.querySelectorAll('#ex-table th[data-col]'), function (th) {
      th.classList.toggle('sorted', th.dataset.col === state.sort.col);
      th.setAttribute('data-dir', th.dataset.col === state.sort.col ? (state.sort.dir > 0 ? '▲' : '▼') : '');
    });
  }
  function seg(id, opts, cur, onPick) {
    var el = document.getElementById(id);
    el.innerHTML = opts.map(function (o) {
      return '<button class="seg' + (o[0] === cur ? ' on' : '') + '"' + (o[2] ? ' disabled' : '') + ' data-v="' + o[0] + '">' + o[1] + '</button>';
    }).join('');
    Array.prototype.forEach.call(el.querySelectorAll('.seg'), function (b) { b.addEventListener('click', function () { onPick(b.dataset.v); }); });
  }
  function controls() {
    var kinds = []; STRATA.forEach(function (s) { if (kinds.indexOf(s.kind) < 0) kinds.push(s.kind); });
    seg('ex-kind', kinds.map(function (k) { return [k, k]; }), state.kind, function (v) {
      state.kind = v; var f = STRATA.filter(function (s) { return s.kind === v; });
      if (!f.some(function (s) { return s.freq === state.freq; })) state.freq = f[0].freq;
      go();
    });
    var freqs = STRATA.filter(function (s) { return s.kind === state.kind; }).map(function (s) { return [s.freq, s.freq]; });
    seg('ex-freq', freqs, state.freq, function (v) { state.freq = v; go(); });
  }
  function chips(d) {
    document.getElementById('ex-ref').innerHTML = d.models.map(function (m) {
      return '<option value="' + m + '"' + (m === state.ref ? ' selected' : '') + '>' + name(m) + '</option>'; }).join('');
    document.getElementById('ex-tags').innerHTML = groupsShown(d).map(function (g) {
      return '<div class="chipgroup"><span class="glabel">' + g[0] + '</span>' + g[1].map(function (t) {
        return '<button class="chip' + (state.tags[t] ? ' on' : '') + '" data-tag="' + t + '">' + t + '</button>'; }).join('') + '</div>';
    }).join('');
    Array.prototype.forEach.call(document.querySelectorAll('#ex-tags .chip'), function (b) {
      b.addEventListener('click', function () { state.tags[b.dataset.tag] = !state.tags[b.dataset.tag];
        b.classList.toggle('on', !!state.tags[b.dataset.tag]); render(); });
    });
  }
  function hash() { try { history.replaceState(null, '', '#' + encodeURIComponent(state.kind) + '/' + encodeURIComponent(state.freq)); } catch (e) {} }
  function go() {
    controls(); hash();
    var st = stratum(); if (!st) return;
    var apply = function (d) { state.data = d; if (d.models.indexOf(state.ref) < 0) state.ref = d.models.indexOf('laplace') >= 0 ? 'laplace' : d.models[0]; chips(d); render(); };
    if (cache[st]) return apply(cache[st]);
    document.getElementById('ex-body').innerHTML = '<tr><td colspan="6" class="muted">loading&hellip;</td></tr>';
    fetch('./data/' + st.replace(':', '-') + '.json').then(function (r) { return r.json(); }).then(function (d) {
      cache[st] = d; if (stratum() === st) apply(d);
    });
  }
  document.getElementById('ex-ref').addEventListener('change', function (e) { state.ref = e.target.value; render(); });
  document.getElementById('ex-held').addEventListener('change', function (e) { state.heldOnly = e.target.checked; render(); });
  document.getElementById('ex-clear').addEventListener('click', function () { state.tags = {}; if (state.data) { chips(state.data); render(); } });
  Array.prototype.forEach.call(document.querySelectorAll('#ex-table th[data-col]'), function (th) {
    th.addEventListener('click', function () {
      var c = th.dataset.col;
      if (state.sort.col === c) state.sort.dir = -state.sort.dir;
      else { state.sort.col = c; state.sort.dir = (c === 'name' || c === 'lic' || c === 'loss' || c === 'crps') ? 1 : -1; }
      render();
    });
  });
  var h = decodeURIComponent((location.hash || '').slice(1)).split('/');
  var s0 = STRATA.filter(function (s) { return s.kind === h[0] && s.freq === h[1]; })[0] || STRATA[0];
  state.kind = s0.kind; state.freq = s0.freq; go();
})();
"""

EXPLORER_CSS = """
    .extabs { display:flex; flex-wrap:wrap; gap:4px; border-bottom:2px solid #ddd; margin:12px 0 10px; }
    .extab { background:none; border:none; border-bottom:3px solid transparent; margin-bottom:-2px; padding:8px 14px;
             font:inherit; font-size:.95rem; color:#555; cursor:pointer; }
    .extab.on { color:#4a3aff; border-bottom-color:#4a3aff; font-weight:600; }
    .controls { display:flex; flex-wrap:wrap; gap:10px 18px; align-items:center; margin:6px 0 4px; }
    .controls select { font:inherit; padding:3px 6px; }
    .chipgroup { display:flex; flex-wrap:wrap; gap:5px; align-items:center; margin:4px 0; }
    .glabel { font-size:.8rem; color:#777; width:86px; }
    .chip { font:inherit; font-size:.8rem; padding:2px 10px; border-radius:12px; border:1px solid #ccc;
            background:#fff; color:#444; cursor:pointer; }
    .chip.on { background:#4a3aff; border-color:#4a3aff; color:#fff; }
    .linkbtn { background:none; border:none; color:#4a3aff; cursor:pointer; font:inherit; font-size:.85rem; padding:0; }
    .picker { display:grid; grid-template-columns:auto 1fr; gap:8px 14px; align-items:center; margin:14px 0 8px; }
    .picker .lab { font-size:.82rem; color:#777; }
    .segs { display:flex; flex-wrap:wrap; gap:0; }
    .seg { font:inherit; font-size:.88rem; padding:4px 14px; border:1px solid #ccc; background:#fff; color:#444; cursor:pointer; margin-left:-1px; }
    .seg:first-child { border-radius:4px 0 0 4px; margin-left:0; } .seg:last-child { border-radius:0 4px 4px 0; }
    .seg.on { background:#4a3aff; border-color:#4a3aff; color:#fff; position:relative; }
    .lic { font-size:.72rem; padding:1px 6px; border-radius:8px; white-space:nowrap; text-decoration:none; }
    .lic.open { background:#e3f4e8; color:#1d6b35; } .lic.restricted { background:#fbf0d9; color:#8a5a00; }
    .lic.noncommercial { background:#fbe3e3; color:#9b2121; }
    @media (prefers-color-scheme: dark) { .lic.open { background:#163d24; color:#9fe0b3; }
      .lic.restricted { background:#3d2f12; color:#f0cf87; } .lic.noncommercial { background:#401a1a; color:#f2a8a8; } }
    #ex-table th[data-col] { cursor:pointer; user-select:none; }
    #ex-table th[data-col]:hover { color:#4a3aff; }
    #ex-table th[data-col]::after { content:attr(data-dir); font-size:.7rem; margin-left:3px; }
    .dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:7px; vertical-align:middle; }
    .dot.foundation { background:#8f84ff; } .dot.classical { background:#e09b2d; } .dot.ours { background:#1a8c4a; }
    .dot.other { background:#aaa; }
    .legend { font-size:.82rem; color:#666; display:flex; gap:16px; flex-wrap:wrap; margin:4px 0 0; }
    .foot { font-size:.82rem; color:#666; margin-top:4px; }
"""
