"""foundationalprediction.com: generated from the skaters canonical store.

Every number on the site is read from files the skaters repository commits:
benchmarks/canonical_summary_vs_laplace.csv and _coverage.csv (the paired
per-series study), benchmarks/perf_results.json (cost per observation) and
benchmarks/frontier3d.json (footprint). Nothing is typed. The roster is the
best models only, each with code and a pinned checkpoint that was run here.

    SKATERS_DIR=../skaters python build.py      # writes docs/
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SKATERS = os.path.abspath(os.environ.get("SKATERS_DIR", os.path.join(HERE, "..", "skaters")))
BENCH = os.path.join(SKATERS, "benchmarks")
OUT = os.path.join(HERE, "docs")

# the strata of the FRED panel, in the order the tables show them
STRATA = [
    ("weekly:econ",    "Weekly economic series"),
    ("monthly:econ",   "Monthly economic series"),
    ("daily:econ",     "Daily economic series"),
    ("m4-hourly:econ", "Hourly, strongly seasonal (M4)"),
    ("daily:price",    "Daily asset prices and returns"),
]

# the roster: display name, store arm, perf model, frontier3d name, kind, note
TAGS = {"laplace": ["online", "no training"], "learned laplace": ["trained", "held-out half"],
        "Timestral-209M": ["zero-shot", "21 q"], "t0-gamma": ["zero-shot", "21 q"],
        "TimesFM 3.0": ["zero-shot", "9 q"], "TimesFM 2.5": ["zero-shot", "9 q"],
        "TiRex": ["zero-shot", "9 q"], "Chronos-Bolt": ["zero-shot", "9 q"]}

ROSTER = [
    ("laplace",         "laplace",             "laplace (Rust) k=1", "laplace (Rust core)", "online, training-free",
     "the yardstick; every other row is paired against it"),
    ("learned laplace", None,                  None,                 None,                  "trained on half of FRED by family",
     "scored on the other half only; see the skaters page for the model"),
    ("Timestral-209M",  "Timestral-209M(q21)", "Timestral-209M",     "Timestral-209M",      "zero-shot, pretrained",
     "native 21 quantile levels"),
    ("t0-gamma",        "t0-gamma(q21)",       "t0-gamma",           "t0-gamma",            "zero-shot, pretrained",
     "native 21 quantile levels"),
    ("TimesFM 3.0",     "TimesFM3(wide)",      "TimesFM-3.0",        "TimesFM 3.0",         "zero-shot, pretrained",
     "nine deciles, generous reconstruction"),
    ("TimesFM 2.5",     "TimesFM(wide)",       "TimesFM-2.5-200M",   "TimesFM 2.5",         "zero-shot, pretrained",
     "nine deciles, generous reconstruction"),
    ("TiRex",           "TiRex(wide)",         "TiRex",              "TiRex",               "zero-shot, pretrained",
     "nine deciles, generous reconstruction"),
    ("Chronos-Bolt",    "Chronos(wide)",       "Chronos-Bolt-small", "Chronos-Bolt-small",  "zero-shot, pretrained",
     "nine deciles, generous reconstruction"),
]


def load():
    vs = {}
    for r in csv.DictReader(open(os.path.join(BENCH, "canonical_summary_vs_laplace.csv"))):
        vs[(r["study"], r["model"])] = r
    cov = {}
    for r in csv.DictReader(open(os.path.join(BENCH, "canonical_summary_coverage.csv"))):
        cov[(r["study"], r["method"])] = r
    perf = json.load(open(os.path.join(BENCH, "perf_results.json")))
    warm = {r["model"]: r["ms"] for r in perf["rows"] if r["regime"] == "warm"}
    f3 = json.load(open(os.path.join(BENCH, "frontier3d.json")))
    mb = {p["name"]: p["mb"] for p in f3["points"]}
    return vs, cov, warm, mb, perf.get("meta", {}), f3.get("meta", {})


def learned_arm(vs, stratum):
    """The learned model's mixture arm with the most validation series on this stratum."""
    best = None
    for (st, m), r in vs.items():
        if st != stratum or not m.startswith("learned(") and m != "learned":
            continue
        if "expert" in m:
            continue
        if best is None or int(r["n_series"]) > int(vs[(st, best)]["n_series"]):
            best = m
    return best


def provenance():
    def git(*args):
        return subprocess.run(["git", "-C", SKATERS, *args], capture_output=True, text=True).stdout.strip()
    return {"skaters_commit": git("rev-parse", "--short", "HEAD"),
            "skaters_date": git("log", "-1", "--format=%cs"),
            "built": time.strftime("%Y-%m-%d")}


def fmt_pct(x):
    return f"{x:.0%}"


def table(stratum, vs, cov, warm, mb):
    rows = []
    for name, arm, pm, fm, kind, note in ROSTER:
        arm_ = learned_arm(vs, stratum) if name == "learned laplace" else arm
        if arm_ is None:
            continue
        r = vs.get((stratum, arm_))
        if arm_ == "laplace":
            c = cov.get((stratum, "laplace"))
            rows.append(dict(name=name, kind=kind, note=note, n=None, w=None, d=None, l=None, dll=0.0,
                             crps=1.0, cov=float(c["cov_central90"]) if c else None,
                             ms=warm.get(pm), mb=mb.get(fm), yardstick=True))
            continue
        if not r:
            continue
        c = cov.get((stratum, arm_))
        rows.append(dict(name=name, kind=kind, note=note + (f" ({arm_})" if name == "learned laplace" else ""),
                         n=int(r["n_series"]), w=int(r["win"]), d=int(r["draw"]), l=int(r["loss"]),
                         dll=float(r["med_dLL"]), crps=float(r["med_crps_ratio"]),
                         cov=float(c["cov_central90"]) if c else None,
                         ms=warm.get(pm), mb=mb.get(fm), yardstick=False))
    rows.sort(key=lambda x: -x["dll"])
    trs = []
    for x in rows:
        if x["yardstick"]:
            wdl = '<td class="muted" colspan="2">yardstick</td>'
        else:
            tot = (x["w"] + x["d"] + x["l"]) or 1
            bar = (f'<span class="wdl"><i class="w" style="width:{x["w"]/tot*100:.1f}%"></i>'
                   f'<i class="d" style="width:{x["d"]/tot*100:.1f}%"></i>'
                   f'<i class="l" style="width:{x["l"]/tot*100:.1f}%"></i></span>')
            wdl = f'<td class="num">{x["n"]:,}</td><td>{bar}<span class="wdlnum">{fmt_pct(x["w"]/tot)} / {fmt_pct(x["l"]/tot)}</span></td>'
        ms = "&ndash;" if x["ms"] is None else (f"{x['ms']*1000:.0f} &micro;s" if x["ms"] < 1 else f"{x['ms']:.0f} ms")
        size = "&ndash;" if x["mb"] is None else (f"{x['mb']*1000:.0f} KB" if x["mb"] < 1 else f"{x['mb']:,.0f} MB")
        covs = "&ndash;" if x["cov"] is None else f"{x['cov']:.2f}"
        cls = ' class="yard"' if x["yardstick"] else ""
        tags = "".join(f'<span class="tag">{t}</span>' for t in TAGS.get(x["name"], []))
        trs.append(f'<tr{cls}><td title="{x["kind"]}; {x["note"]}"><b>{x["name"]}</b> {tags}</td>'
                   f'{wdl}<td class="num">{x["dll"]:+.3f}</td><td class="num">{x["crps"]:.3f}</td>'
                   f'<td class="num">{covs}</td><td class="num">{ms}</td><td class="num">{size}</td></tr>')
    # the pick: ahead of laplace at the median AND on more series than it is behind
    pick = next((x for x in rows if not x["yardstick"] and x["w"] > x["l"] and x["dll"] > 0), None)
    return "\n".join(trs), pick, rows


TAB_LABEL = {"weekly:econ": "Weekly", "monthly:econ": "Monthly", "daily:econ": "Daily",
             "m4-hourly:econ": "Hourly", "daily:price": "Prices"}

TAB_JS = """
(function () {
  var tabs = Array.prototype.slice.call(document.querySelectorAll('.tab'));
  var panels = Array.prototype.slice.call(document.querySelectorAll('.panel'));
  function show(id) {
    if (!document.getElementById(id)) id = tabs[0].dataset.tab;
    tabs.forEach(function (t) { var on = t.dataset.tab === id; t.classList.toggle('on', on); t.setAttribute('aria-selected', on); });
    panels.forEach(function (p) { p.hidden = p.id !== id; });
    if (id === 'frontier' && window.Plotly) { var el = document.getElementById('frontier3d'); if (el && el.data) Plotly.Plots.resize(el); }
    try { history.replaceState(null, '', '#' + id); } catch (e) {}
  }
  tabs.forEach(function (t) { t.addEventListener('click', function () { show(t.dataset.tab); }); });
  show((location.hash || '').slice(1));
})();
"""

NAV = """  <header class="site-header">
    <div class="nav-inner">
      <a class="brand" href="./index.html">foundational prediction</a>
      <nav>
        <span class="menu" tabindex="0"><span class="menu-label">Leaderboard &#9662;</span>
          <span class="drop">
            <a href="./index.html">Next data point (k = 1)</a>
            <a href="./gift.html">GIFT reconciliation</a>
          </span>
        </span>
        <span class="menu" tabindex="0"><span class="menu-label">Performance &#9662;</span>
          <span class="drop">
            <a href="./frontier.html">Overview</a>
            <a href="./speed.html">Speed</a>
            <a href="./size.html">Size</a>
          </span>
        </span>
        <span class="menu" tabindex="0"><span class="menu-label">Models &#9662;</span>
          <span class="drop">
            <a href="./laplace.html">laplace</a>
            <a href="./laplacepp.html">granger</a>
          </span>
        </span>
        <a href="./methodology.html">Methodology</a>
        <a href="./papers.html">Papers</a>
        <a href="https://skaters.microprediction.org/">skaters</a>
        <a href="https://github.com/microprediction/foundationalprediction">GitHub</a>
      </nav>
    </div>
  </header>"""

STYLE = """
    .wdl { display:inline-flex; height:13px; width:120px; border-radius:3px; overflow:hidden; border:1px solid #ddd; vertical-align:middle; }
    .wdl i { display:block; height:100%; }
    .wdl .w { background:#1a8c4a; } .wdl .d { background:#c8c8d4; } .wdl .l { background:#c0392b; }
    .wdlnum { font-size:.8rem; color:#666; margin-left:8px; }
    .muted { color:#777; } .small { font-size:.82rem; }
    table.lb { border-collapse:collapse; width:100%; margin:6px 0 10px; }
    table.lb th, table.lb td { border-bottom:1px solid #e5e5e5; padding:7px 8px; text-align:left; vertical-align:middle; font-size:.93rem; }
    table.lb th { border-bottom:2px solid #bbb; font-weight:600; }
    table.lb td.num, table.lb th.num { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
    table.lb td { white-space:nowrap; }
    .tag { display:inline-block; font-size:.72rem; color:#555; background:#f0f0f4; border-radius:3px;
           padding:1px 6px; margin-left:4px; vertical-align:middle; font-weight:400; }
    .wdlnum { display:inline-block; min-width:72px; }
    tr.yard td { background:#f4f9f5; }
    .pick { background:#f6f4ff; border-left:4px solid #4a3aff; padding:8px 12px; margin:10px 0 18px; }
    .stamp { color:#888; font-size:.85em; font-style:italic; }
    .foot { font-size:.82rem; color:#666; margin-top:4px; }
    .tabs { display:flex; flex-wrap:wrap; gap:4px; border-bottom:2px solid #ddd; margin:14px 0 0; }
    .tab { background:none; border:none; border-bottom:3px solid transparent; margin-bottom:-2px;
           padding:8px 14px; font:inherit; font-size:.95rem; color:#555; cursor:pointer; }
    .tab:hover { color:#111; }
    .tab.on { color:#4a3aff; border-bottom-color:#4a3aff; font-weight:600; }
    .panel { padding-top:6px; overflow-x:auto; }
    header.site-header nav .menu .drop .drop-head { display:block; padding:8px 14px 2px; font-size:.7rem;
           letter-spacing:.06em; text-transform:uppercase; color:rgba(255,255,255,.55); }
    header.site-header nav .menu .drop { max-height:80vh; overflow-y:auto; }
    ol.refs li, ul.refs li { font-size:.92rem; margin-bottom:6px; }
"""


def page(title, body, desc, extra_css=""):
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>{title}</title>
  <meta name="description" content="{desc}" />
  <link rel="stylesheet" href="./academic.css" />
  <style>{STYLE}{extra_css}</style>
</head>
<body>
{NAV}
  <main>
{body}
  </main>
  <footer>
    Generated from the <a href="https://github.com/microprediction/skaters">skaters</a> canonical store.
    Part of the <a href="https://repos.microprediction.org/">microprediction</a> family.
  </footer>
</body>
</html>
"""


def index(vs, cov, warm, mb, prov, perf_meta):
    sections, picks, tabs = [], [], []
    for st, long in STRATA:
        trs, pick, rows = table(st, vs, cov, warm, mb)
        if not rows:
            continue
        if pick:
            picks.append((long, pick))
            lead = (f'<div class="pick">For the next {long.lower().replace(" series", "")} observation the best-scoring '
                    f'model here is <b>{pick["name"]}</b>: {pick["dll"]:+.3f} nats against laplace at the median, '
                    f'ahead on {fmt_pct(pick["w"]/(pick["w"]+pick["d"]+pick["l"]))} of series and behind on '
                    f'{fmt_pct(pick["l"]/(pick["w"]+pick["d"]+pick["l"]))}.</div>')
        else:
            lead = ('<div class="pick">No model on the roster beats laplace more often than it loses on this '
                    'stratum; laplace is the pick.</div>')
        slug = st.replace(":", "-")
        tabs.append(f'<button class="tab" role="tab" data-tab="{slug}">{TAB_LABEL.get(st, long)}</button>')
        sections.append(f"""
    <section class="panel" role="tabpanel" id="{slug}">
    {lead}
    <table class="lb">
      <thead><tr><th>model</th><th class="num">series</th>
      <th title="Share of series won / lost against laplace by a Diebold-Mariano test; the rest are draws">win / loss</th>
      <th class="num" title="Median per-series log-score gap in nats; positive beats laplace">&Delta;LL</th>
      <th class="num" title="Median CRPS ratio to laplace; below 1 is better">CRPS</th>
      <th class="num" title="Central 90% interval coverage; target 0.90">cov<sub>90</sub></th>
      <th class="num" title="Cost of one new observation, streaming, one CPU">per obs</th>
      <th class="num" title="Bytes a deployment carries">size</th></tr></thead>
      <tbody>
{trs}
      </tbody>
    </table>
    </section>""")
    tabs.append('<button class="tab" role="tab" data-tab="frontier">All three axes</button>')
    sections.append("""
    <section class="panel" role="tabpanel" id="frontier">
      <div id="frontier3d" style="width:100%;min-height:520px;"></div>
      <p id="frontier3d-caption" class="muted small"></p>
    </section>""")
    body = f"""
    <h1>Which model for the next data point?</h1>
    <p class="lede">The next observation, scored alone. Every model paired against laplace on the same series.</p>
    <div class="tabs" role="tablist">{"".join(tabs)}</div>
{"".join(sections)}
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
    <script src="./frontier3d.js"></script>
    <script>{TAB_JS}</script>
    <p class="stamp">Built {prov['built']} from skaters commit {prov['skaters_commit']} ({prov['skaters_date']}).
      Cost measured {perf_meta.get('date', '')} on {perf_meta.get('machine', '')}.</p>
"""
    return page("Foundational prediction: the next data point", body,
                "Which model to use for the next data point: foundation models and a training-free baseline, paired on identical series, one horizon isolated.")


def methodology(prov):
    body = """
    <h1>Methodology</h1>
    <p class="lede">How the leaderboards are scored. How laplace and granger (the trained laplace, formerly shown here as laplace++) are built, and the
      transforms and ensembles underneath them, is in the
      <a href="https://skaters.microprediction.org/guide.html">skaters guide</a>.</p>
    <h2>The question</h2>
    <p>Which model should forecast the next observation of a stream? That question needs the one-step
      metric isolated. Suites that pool steps 1 to H into one score cannot answer it, and a model that is
      strong at step one and ordinary after is indistinguishable there from one that is flat.</p>
    <h2>The protocol</h2>
    <p>Every model forecasts the one-step change of a series from a fixed context, rolling over a test
      window; every forecast becomes a full predictive density scored by log likelihood and CRPS at the
      realised value. Pretrained models run zero-shot, each in its own environment, on CPU. Their
      quantile outputs are turned into densities by a stated reconstruction: native levels where the
      model emits them (21 for Timestral and t0-gamma), the generous bandwidth otherwise, because a
      tight reconstruction of nine deciles manufactures about half a nat of apparent deficit.</p>
    <h2>Pairing and the test</h2>
    <p>Every pair of models is compared per series on identical series and windows. The leaderboards
      pair each model against laplace, the training-free online reference; the explorer lets you pick
      any model in its menu as the reference, granger, a foundation model, the conformal baseline or a
      classical one, because the per-series test is precomputed for every pair. Per series the mean
      log-score differential is tested by Diebold&ndash;Mariano with a Newey&ndash;West standard
      error; a win or a loss is a difference beyond 1.96 standard errors, the rest are draws. A draw
      says the window cannot separate the two, not that they are equal, so the median gap is the
      better measure of size.</p>
    <h2>Baselines that keep learning</h2>
    <p>The classical baselines in published suites are fit once at the window origin and never updated.
      Here every baseline is online: it sees every observation before it forecasts the next. The gap
      between foundation models and classical methods shown here is therefore the gap a practitioner
      running a stream would see. AutoARIMA, AutoETS, AutoTheta and GARCH-t are refit at every step.</p>
    <h2>Trained models</h2>
    <p>granger (formerly labelled laplace++ on this site) is the one trained model here. It is trained on half of each panel, split by series
      family, and every number shown for it is on the other half. Training data added from outside the
      panel is never one of the held-out series. How it is built and what it guarantees are on the
      <a href="laplacepp.html">granger</a> page.</p>
    <h2>Roster</h2>
    <p>Only the best models in each class, each with public code and a pinned checkpoint that was run
      here. Submissions that cannot be rerun are not listed.</p>
    <h2>Licences and versions</h2>
    <p>The licence column gives each model's terms. Timestral and TiRex are free below a revenue
      threshold; TimesFM 3.0 and TimeH2O are licensed for testing, evaluation and research only, and
      Moirai 2.0 (CC BY-NC 4.0) for non-commercial use; the other foundation models, laplace, granger and the classical models are open. All
      evaluation here is non-commercial benchmarking. TimeH2O is h2oai/timeh2o-v1 at Hugging Face
      revision c556990, whose weights are unchanged since revision 60b68fa of 2026-10-22, run zero-shot
      on CPU on 2026-11-03 and 2026-11-04 by the protocol above.</p>
    <h2>Training-data overlap</h2>
    <p>A pretrained model may have seen the evaluation series, or earlier stretches of them, in
      training. The panels are FRED economic and price series and the M4 hourly set. What each model's
      own documentation says:</p>
    <ul>
      <li><b>Chronos-2</b>: pretraining includes all 414 M4 hourly series, the whole hourly panel here,
        and other M4 frequencies; no FRED dataset is named
        (<a href="https://arxiv.org/abs/2510.15821">report</a>, Table 6).</li>
      <li><b>Moirai 2.0</b>: pretraining includes GIFT-Eval's pretraining set, which contains
        <code>fred_md</code> (107 FRED-MD monthly series), and GIFT-Eval's training split, which covers
        every M4 frequency (<a href="https://arxiv.org/abs/2511.11698">paper</a>).</li>
      <li><b>TTM-R3</b>: trained on selected GIFT-Eval pretraining and training datasets; the list is
        not published (<a href="https://huggingface.co/ibm-granite/granite-timeseries-ttm-r3">model card</a>).</li>
      <li><b>TiRex-2</b>: pretrained on the Chronos datasets and LOTSA, which include M4 and the Monash
        archive's <code>fred_md</code> (<a href="https://huggingface.co/NX-AI/TiRex-2">model card</a>).</li>
      <li><b>TinyCast</b>: GIFT-Eval's pretraining set with the datasets that overlap the GIFT-Eval and
        Chronos zero-shot test sets removed; the exclusion list is not published
        (<a href="https://arxiv.org/abs/2608.15767">paper</a>).</li>
      <li><b>Toto 2.0</b>: Datadog observability metrics and synthetic series only, no public
        forecasting data (<a href="https://arxiv.org/abs/2605.20119">paper</a>).</li>
      <li><b>Timestral, t0-gamma, TimeH2O, TimesFM, Chronos-Bolt and TiRex</b>: not yet checked against
        these panels.</li>
      <li><b>granger</b>: runs on the weekly, monthly and M4 hourly panels. It is trained on one half
        of each, split by series family, and scored on the other; the weekly run also trained on
        GIFT-Eval's weekly pretraining series (NN5, road and web traffic), none of them in the panels.</li>
    </ul>
    <p>Where a model has seen a panel, its row on that panel measures recall as well as forecasting.</p>
    <h2>Source</h2>
    <p>The tables are generated from the skaters repository's committed summaries, the per-step store
      behind them, its cost measurements and its footprint measurements, by
      <code>build.py</code> in this repository. The same files drive
      <a href="https://skaters.microprediction.org/benchmarks.html">skaters.microprediction.org</a>,
      which carries the per-model pages, the sandwich results and the papers.</p>
"""
    return page("Foundational prediction: methodology", body,
                "How the tables are computed: one-step densities, paired Diebold-Mariano tests, online baselines, a family split for trained models.")


def horizons():
    body = """
    <h1>Per-horizon scores</h1>
    <p>The next data point is step one. This page will carry the same table at every step from 1 to H,
      each isolated, so a model's curve over the fan is visible instead of pooled. The k&gt;1 arms are
      being scored; the page fills in as they land.</p>
"""
    return page("Foundational prediction: horizons", body, "Per-horizon scores, each step isolated.")


def gift():
    body = """
    <h1>GIFT-Eval reconciliation</h1>
    <p>GIFT-Eval reports one number per model and configuration, pooled over its horizon (8 steps for
      weekly series, 12 for monthly, in the short-term track) and over rolling windows, as MASE and a
      quantile loss on nine levels. This page will carry those published numbers for the roster models
      beside the per-horizon and density columns this site adds, so every row can be reconciled with
      the leaderboard and then read further. Pretraining overlap with the evaluation data is declared
      per model where it is known.</p>
"""
    return page("Foundational prediction: GIFT-Eval reconciliation", body,
                "The published GIFT-Eval numbers for the roster beside the per-horizon and density columns.")



# store arm -> (display name, kind) for every model the explorer can show
FP_MODELS = {
    "laplace": ("laplace", "ours"), "learned(e2)": ("granger", "ours"),
    "Timestral-209M(q21)": ("Timestral-209M", "foundation"), "t0-gamma(q21)": ("t0-gamma", "foundation"),
    "TimesFM3(wide)": ("TimesFM 3.0", "foundation"), "TimesFM(wide)": ("TimesFM 2.5", "foundation"),
    "TiRex(wide)": ("TiRex", "foundation"), "Chronos(wide)": ("Chronos-Bolt", "foundation"),
    "TimeH2O-v1(wide)": ("TimeH2O", "foundation"),
    # #329 additions: the 21-level Chronos-2 natively, the nine-decile models through the generous nozzle
    "Chronos-2": ("Chronos-2", "foundation"), "TiRex-2(wide)": ("TiRex-2", "foundation"),
    "Toto2-2.5B(wide)": ("Toto 2.0 2.5B", "foundation"), "Toto2-313m(wide)": ("Toto 2.0 313M", "foundation"),
    "TTM-R3(wide)": ("TTM-R3", "foundation"), "Moirai-2.0-small(wide)": ("Moirai 2.0 small", "foundation"),
    "TinyCast(wide)": ("TinyCast", "foundation"),
    # #355 additions (November 2026): nine-decile models through the generous
    # nozzle, iAmTime on its 21 native levels, and the conformal baseline
    "Simons-297M(wide)": ("Simons 297M", "foundation"), "Simons-60M(wide)": ("Simons 60M", "foundation"),
    "ci-proto(wide)": ("ci-proto", "foundation"), "Hirondel-7M(wide)": ("Hirondel-7M", "foundation"),
    "iAmTime-base": ("iAmTime-base", "foundation"), "TimeSSM-2.5M(wide)": ("TimeSSM-2.5M", "foundation"),
    "CSP": ("CSP", "conformal"), "CSP&lap": ("CSP & laplace portfolio", "conformal"),
    "AutoARIMA": ("AutoARIMA", "classical"), "AutoETS": ("AutoETS", "classical"),
    "AutoTheta": ("AutoTheta", "classical"), "GARCH-t": ("GARCH-t", "classical"),
    # November 2026 additions (skaters #383): headline arms, native levels where the model has 99
    "PatchTST-FM-r2(q99)": ("PatchTST-FM r2", "foundation"), "Tafsut-base(wide)": ("Tafsut", "foundation"),
    "Tabby-Pretrain(q99)": ("Tabby-Pretrain", "foundation"), "Yug(wide)": ("Yug", "foundation"),
    "CITRAS-FM(wide)": ("CITRAS-FM", "foundation"), "TS-ICL(q99)": ("TS-ICL", "foundation"),
    "Kairos-50m(wide)": ("Kairos 50M", "foundation"), "Timer-S1(wide)": ("Timer-S1", "foundation"),
    "Kumo-large-reg(q99)": ("Kumo Tabular large", "foundation"),
}
# store arm -> (short label, class, link, tooltip). class: open, restricted (free below a
# revenue threshold), noncommercial
_APACHE = ("Apache-2.0", "open", "https://www.apache.org/licenses/LICENSE-2.0", "Apache License 2.0: commercial use allowed")
FP_LICENSES = {
    "laplace": ("MIT", "open", "https://github.com/microprediction/skaters/blob/main/LICENSE", "MIT: commercial use allowed"),
    "learned(e2)": ("MIT", "open", "https://github.com/microprediction/skaters/blob/main/LICENSE", "MIT: commercial use allowed"),
    "Timestral-209M(q21)": ("Modified MIT", "restricted", "https://huggingface.co/mistralai/Timestral-209M/blob/main/LICENSE",
                            "Modified MIT: no rights for companies above $20M monthly revenue"),
    "t0-gamma(q21)": _APACHE[:2] + ("https://huggingface.co/theforecastingcompany/t0-gamma",) + _APACHE[3:],
    "TimesFM3(wide)": ("Non-commercial", "noncommercial", "https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE",
                       "TimesFM Non-Commercial License v1.0: testing, evaluation and research only"),
    "TimesFM(wide)": _APACHE[:2] + ("https://huggingface.co/google/timesfm-2.5-200m-pytorch",) + _APACHE[3:],
    "TiRex(wide)": ("NXAI Community", "restricted", "https://huggingface.co/NX-AI/TiRex/blob/main/LICENSE",
                    "NXAI Community License: commercial terms apply above EUR 100M annual revenue"),
    "Chronos(wide)": _APACHE[:2] + ("https://huggingface.co/amazon/chronos-bolt-small",) + _APACHE[3:],
    "TimeH2O-v1(wide)": ("Non-commercial", "noncommercial", "https://huggingface.co/h2oai/timeh2o-v1",
                         "H2O Model Non-Commercial Licence v1.0: testing, evaluation, benchmarking and research only"),
    "Chronos-2": _APACHE[:2] + ("https://huggingface.co/amazon/chronos-2",) + _APACHE[3:],
    "TiRex-2(wide)": _APACHE[:2] + ("https://huggingface.co/NX-AI/TiRex-2",) + _APACHE[3:],
    "Toto2-2.5B(wide)": _APACHE[:2] + ("https://huggingface.co/Datadog/Toto-2.0-2.5B",) + _APACHE[3:],
    "Toto2-313m(wide)": _APACHE[:2] + ("https://huggingface.co/Datadog/Toto-2.0-313m",) + _APACHE[3:],
    "TTM-R3(wide)": _APACHE[:2] + ("https://huggingface.co/ibm-granite/granite-timeseries-ttm-r3",) + _APACHE[3:],
    "Moirai-2.0-small(wide)": ("CC BY-NC 4.0", "noncommercial", "https://huggingface.co/Salesforce/moirai-2.0-R-small",
                               "Creative Commons Attribution-NonCommercial 4.0: non-commercial use only"),
    "TinyCast(wide)": _APACHE[:2] + ("https://huggingface.co/raws-labs/tinycast",) + _APACHE[3:],
    "Simons-297M(wide)": _APACHE[:2] + ("https://huggingface.co/abel-lab/Simons-297M",) + _APACHE[3:],
    "Simons-60M(wide)": _APACHE[:2] + ("https://huggingface.co/abel-lab/Simons-60M",) + _APACHE[3:],
    "ci-proto(wide)": ("ci-proto licence", "open", "https://huggingface.co/ContinuousIntelligence/ci-proto",
                       "ci-proto Code and Model Licence v1.0: free for commercial use, with a public attribution notice and a one-time e-mail to Continuous Intelligence on commercial use or redistribution"),
    "Hirondel-7M(wide)": _APACHE[:2] + ("https://huggingface.co/Tncrd/Hirondel-7M",) + _APACHE[3:],
    "iAmTime-base": _APACHE[:2] + ("https://huggingface.co/anishsaha/iAmTime-base-forecast",) + _APACHE[3:],
    "TimeSSM-2.5M(wide)": _APACHE[:2] + ("https://huggingface.co/IUseAMouse/timessm-2.5m",) + _APACHE[3:],
    "CSP": ("MIT", "open", "https://github.com/valeman/csp-forecaster", "MIT (csp-forecaster, the author's reference package): commercial use allowed"),
    "CSP&lap": ("MIT", "open", "https://github.com/valeman/csp-forecaster", "MIT (csp-forecaster) for the CSP half; the portfolio weights and laplace are MIT (skaters)"),
    "AutoARIMA": _APACHE[:2] + ("https://github.com/Nixtla/statsforecast",) + _APACHE[3:],
    "AutoETS": _APACHE[:2] + ("https://github.com/Nixtla/statsforecast",) + _APACHE[3:],
    "AutoTheta": _APACHE[:2] + ("https://github.com/Nixtla/statsforecast",) + _APACHE[3:],
    "GARCH-t": ("NCSA", "open", "https://github.com/bashtage/arch/blob/main/LICENSE.md", "University of Illinois/NCSA (arch package): commercial use allowed"),
    "PatchTST-FM-r2(q99)": ("OpenMDW-1.0", "open", "https://huggingface.co/ibm-granite/granite-timeseries-patchtst-fm-r2",
                            "OpenMDW-1.0 (Open Model, Data and Weights): commercial use allowed"),
    "Tafsut-base(wide)": ("MIT", "open", "https://huggingface.co/Tafsut-FM/tafsut-univariate-base", "MIT: commercial use allowed"),
    "Tabby-Pretrain(q99)": ("CC BY-NC 4.0", "noncommercial", "https://huggingface.co/paris-noah/Tabby-Pretrain",
                            "Creative Commons Attribution-NonCommercial 4.0: non-commercial use only"),
    "Yug(wide)": ("CC BY-NC 4.0", "noncommercial", "https://huggingface.co/birlaailabs/yug",
                  "Weights CC BY-NC 4.0 (code Apache-2.0): non-commercial use only"),
    "CITRAS-FM(wide)": ("CC BY-NC-SA 4.0", "noncommercial", "https://huggingface.co/hitachi-nlp/citras-fm",
                        "Creative Commons Attribution-NonCommercial-ShareAlike 4.0: non-commercial use only"),
    "TS-ICL(q99)": ("Non-commercial", "noncommercial", "https://huggingface.co/taharnbl/TS-ICL",
                    "TS-ICL licence (other): non-commercial use, see the model card"),
    "Kairos-50m(wide)": _APACHE[:2] + ("https://huggingface.co/mldi-lab/Kairos_50m",) + _APACHE[3:],
    "Timer-S1(wide)": _APACHE[:2] + ("https://huggingface.co/bytedance-research/Timer-S1",) + _APACHE[3:],
    "Kumo-large-reg(q99)": ("OpenMDW-1.1", "open", "https://huggingface.co/nvidia/Kumo-Tabular",
                            "OpenMDW-1.1 (Open Model, Data and Weights): commercial use allowed"),
}
# laplace++ runs in priority order: the most-trained run scored on a stratum is shown
# monthly: the run with the most training data scored so far; weekly: e3 (a third
# pass, e4, overfit on held-out weekly series and is not listed)
_LAPLACEPP_DEFAULT = ["learned(wswa2)", "learned(swa6)", "learned(swa5)", "learned(swa4)", "learned(swa3)", "learned(e4s21000)", "learned(e4s18000)", "learned(e4s15000)", "learned(e4s12000)",
                  "learned(e4s9000)", "learned(e3)", "learned(e2)", "learned(e2s1000)", "learned(e2s500)"]
# weekly: gpswa (FRED weekly train + GIFT-Eval weekly pretraining, running average)
# beats wswa2 head to head 238-62 on 1,522 held-out series; hourly's gpswa is
# worse than e3, so the order is per stratum
LAPLACEPP_RUNS = {"weekly:econ": ["learned(gpswa)"] + _LAPLACEPP_DEFAULT, "*": _LAPLACEPP_DEFAULT}

# the explorer's menu: the strongest size of each family plus the classical and
# conformal arms. Every pair of these is shipped per series, so the list stays
# short; the leaderboards above list every arm in FP_MODELS from the summary.
EXPLORER_MODELS = [
    "laplace", "learned(e2)",
    "Timestral-209M(q21)", "t0-gamma(q21)", "TimesFM3(wide)", "TimesFM(wide)", "TiRex(wide)", "Chronos(wide)",
    "TimeH2O-v1(wide)", "Chronos-2", "Toto2-2.5B(wide)", "Simons-297M(wide)", "iAmTime-base",
    "ci-proto(wide)", "Hirondel-7M(wide)",
    "PatchTST-FM-r2(q99)", "Timer-S1(wide)", "Tafsut-base(wide)", "TS-ICL(q99)", "Kumo-large-reg(q99)",
    "CSP", "CSP&lap",
    "AutoARIMA", "AutoETS", "AutoTheta", "GARCH-t",
]

FP_STRATA = [
    {"id": "weekly:econ", "kind": "Economic", "freq": "Weekly"},
    {"id": "monthly:econ", "kind": "Economic", "freq": "Monthly"},
    {"id": "daily:econ", "kind": "Economic", "freq": "Daily"},
    {"id": "m4-hourly:econ", "kind": "Economic", "freq": "Hourly (M4)"},
    {"id": "daily:price", "kind": "Prices", "freq": "Daily"},
]


def leaderboard(prov, written, pages=None):
    import explorer
    strata = [x for x in FP_STRATA if x["id"] in written]
    names = {k: v[0] for k, v in FP_MODELS.items()}
    kinds = {k: v[1] for k, v in FP_MODELS.items()}
    body = f"""
    <h1>Which model for the next data point?</h1>
    <p class="lede">The next observation, scored alone. Every model paired against a reference on the same series.</p>
    <div class="picker">
      <span class="lab">Series</span>
      <div style="display:flex;flex-wrap:wrap;align-items:center;gap:8px 14px">
        <div class="segs" id="ex-kind"></div>
        <span style="display:inline-flex;flex-wrap:wrap;align-items:center;gap:8px">
          <span class="lab">Frequency</span><span class="segs" id="ex-freq"></span></span>
      </div>
      <span class="lab">Reference</span><div><select id="ex-ref"></select>
        <label class="small muted" style="margin-left:14px" title="granger is trained on one half of the series and scored only on the other. Tick to score every model on that held-out half, so all rows cover the same series.">
        <input type="checkbox" id="ex-held"> held-out series only</label></div>
    </div>
    <details><summary class="small muted">Series types <button class="linkbtn" id="ex-clear">clear</button></summary>
      <div id="ex-tags"></div>
    </details>
    <p id="ex-pick" class="pick"></p>
    <div style="overflow-x:auto">
    <table class="lb" id="ex-table">
      <thead><tr><th data-col="name">model</th><th class="num" data-col="n">series</th>
      <th data-col="net" title="Share of series won / lost against the reference, by a per-series Diebold-Mariano test on log predictive density; the rest are draws; sorts by win minus loss">win / loss*</th>
      <th class="num" data-col="dll" title="Median per-series log-score gap in nats; positive beats the reference">&Delta;LL</th>
      <th class="num" data-col="crps" title="Median per-series CRPS ratio to the reference; below 1 is better">CRPS</th>
      <th class="num" data-col="cov" title="Central 90% interval coverage, pooled over the selected series; target 0.90">cov<sub>90</sub></th>
      <th data-col="lic" title="Licence of the model weights or package; hover a cell for the terms">licence</th></tr></thead>
      <tbody id="ex-body"></tbody>
    </table>
    </div>
    <p class="foot" id="ex-runs"></p>
    <p class="foot">granger is trained on one half of each panel (split by series family) and scored only on the other
      half, so its row covers fewer series; tick <i>held-out series only</i> to put every model on that same half.</p>
    <p class="foot">* By log score: a per-series Diebold&ndash;Mariano test on the log predictive density, 64 one-step
      points per series. Click a column to sort.</p>
    <div class="legend"><span><span class="dot foundation"></span>foundation model</span>
      <span><span class="dot classical"></span>classical, refit every step</span>
      <span><span class="dot ours"></span>laplace and granger</span>
      <span><span class="lic open">open</span> commercial use allowed</span>
      <span><span class="lic restricted">restricted</span> free below a revenue threshold</span>
      <span><span class="lic noncommercial">non-commercial</span> research and evaluation only</span></div>
    <p class="foot">TimeH2O: h2oai/timeh2o-v1 at Hugging Face revision c556990 (weights unchanged since 60b68fa, 2026-10-22), evaluated 2026-11-03 to 2026-11-04
      under its non-commercial licence. Method: <a href="methodology.html">methodology</a>.</p>
    <p class="stamp">Built {prov['built']} from skaters commit {prov['skaters_commit']} ({prov['skaters_date']}).</p>
    <script>window.FP_NAMES = {json.dumps(names)}; window.FP_KIND = {json.dumps(kinds)}; window.FP_STRATA = {json.dumps(strata)}; window.FP_LIC = {json.dumps(FP_LICENSES)}; window.FP_PAGES = {json.dumps(pages or {})};</script>
    <script>{explorer.EXPLORER_JS}</script>
"""
    return page("Foundational prediction: the next data point", body,
                "Which model to use for the next data point: foundation, classical and online models paired on identical series, filtered by series type, with a choice of reference.",
                extra_css=explorer.EXPLORER_CSS)


def frontier_page(prov):
    body = f"""
    <h1>Performance</h1>
    <p class="lede">One point per model. Drag to rotate. <a href="https://lnkd.in/p/ebYmNbPA">Discuss</a>.</p>
    <div id="frontier3d" style="width:100%;min-height:560px;"></div>
    <p id="frontier3d-caption" class="muted small"></p>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
    <script src="./frontier3d.js"></script>
    <p class="stamp">Built {prov['built']} from skaters commit {prov['skaters_commit']}.</p>
"""
    return page("Foundational prediction: performance", body,
                "Accuracy, cost per observation and footprint of each model in one chart.")



def _fmt_ms(ms):
    if ms is None or ms != ms:
        return "&ndash;"
    return f"{ms * 1000:.0f} &micro;s" if ms < 1 else f"{ms:,.1f} ms"


def _kind_of(name):
    n = name.lower()
    if n.startswith("laplace"):
        return "ours"
    if any(x in n for x in ("arima", "ets", "theta", "garch", "prophet")):
        return "classical"
    return "foundation"


_KCOL = {"ours": "#1a8c4a", "foundation": "#8f84ff", "classical": "#e09b2d", "conformal": "#d9534f"}


def _times(r):
    """A multiple to two significant figures: 1.7x, 14x, 370x, 37,000x."""
    import math
    p = 10 ** (math.floor(math.log10(r)) - 1)
    r = round(r / p) * p                    # two significant figures first, then the format
    return f"{r:.1f}×" if r < 10 else f"{r:,.0f}×"


def hbars(items, fmt, unit_note, power=0.25, ref=None):
    """Horizontal bars, length proportional to value**power (0 < power < 1 sits
    between a log scale and a linear one: 0.25 keeps a 10,000x range readable
    while the gaps still look like gaps). items = [(label, value)], ascending.
    With ref = (label, value), each bar above it also states its multiple of
    that value, since the compressed scale understates the ratios."""
    import math
    items = [(l, v) for l, v in items if v and v > 0]
    if not items:
        return ""
    vmax = max(v for _, v in items)
    W, LW, RW, BH, GAP = 760, 190, (190 if ref else 90), 20, 8
    PW = W - LW - RW
    H = len(items) * (BH + GAP) + 34
    def x(v):
        return LW + PW * (v / vmax) ** power
    out = [f'<svg viewBox="0 0 {W} {H}" style="width:100%;max-width:{W}px;height:auto;display:block;margin:10px 0 4px" '
           f'role="img" aria-label="Bar chart, fourth-root scale, {unit_note}">']
    vmin = min(v for _, v in items)
    k = 10 ** math.ceil(math.log10(vmin))
    ticks = []
    while k <= vmax * 1.0001:
        ticks.append(k); k *= 10
    for t in ticks:
        xx = x(t)
        out.append(f'<line x1="{xx:.1f}" y1="0" x2="{xx:.1f}" y2="{H - 26}" stroke="#e5e5e5"/>')
        out.append(f'<text x="{xx:.1f}" y="{H - 10}" font-size="11" fill="#888" text-anchor="middle">{fmt(t)}</text>')
    for i, (label, v) in enumerate(items):
        y = i * (BH + GAP)
        col = _KCOL[_kind_of(label)]
        out.append(f'<text x="{LW - 10}" y="{y + BH * 0.72:.1f}" font-size="13" fill="#333" text-anchor="end">{label}</text>')
        out.append(f'<rect x="{LW}" y="{y}" width="{max(x(v) - LW, 2):.1f}" height="{BH}" rx="3" fill="{col}"/>')
        note = fmt(v)
        if ref and v >= 1.5 * ref[1]:
            note += f'  <tspan fill="#999">{_times(v / ref[1])} {ref[0]}</tspan>'
        out.append(f'<text x="{x(v) + 6:.1f}" y="{y + BH * 0.72:.1f}" font-size="12" fill="#555">{note}</text>')
    out.append("</svg>")
    return "".join(out)


def _fmt_mb(mb):
    if mb < 1:
        return f"{mb * 1000:,.0f} KB"
    if mb < 10:
        return f"{mb:.1f} MB"
    return f"{mb:,.0f} MB" if mb < 1000 else f"{mb / 1000:,.1f} GB"


def _fmt_ms_plain(ms):
    return f"{ms * 1000:,.0f} µs" if ms < 1 else (f"{ms:,.0f} ms" if ms < 1000 else f"{ms / 1000:,.1f} s")



def speed_page(prov, perf, perf_meta):
    rows = perf["rows"]
    models = []
    for r in rows:
        if r["model"] not in models:
            models.append(r["model"])
    by = {(r["model"], r["regime"]): r["ms"] for r in rows}
    warm = sorted(models, key=lambda m: by.get((m, "warm"), float("inf")))
    trs = []
    for m in warm:
        cold = [by.get((m, c)) for c in ("cold b=1", "cold b=16", "cold b=64")]
        trs.append(f'<tr><td><b>{m}</b></td><td class="num">{_fmt_ms(by.get((m, "warm")))}</td>'
                   + "".join(f'<td class="num">{_fmt_ms(c)}</td>' for c in cold) + "</tr>")
    body = f"""
    <h1>Speed</h1>
    <p class="lede">Milliseconds per forecast on one CPU, one thread.</p>
    {hbars([(m, by.get((m, "warm"))) for m in warm], _fmt_ms_plain, "streaming cost per new observation", ref=("laplace (Python)", by.get(("laplace (Python) k=1", "warm"))) if by.get(("laplace (Python) k=1", "warm")) else None)}
    <p class="foot">Streaming: the cost of one new observation. Bar length is the fourth root of the time,
      between a log and a linear scale.</p>
    <table class="lb">
      <thead><tr><th>model</th>
      <th class="num" title="One new observation arrives: an online model updates its state, a stateless model re-runs on its context">streaming</th>
      <th class="num" title="A series never seen before, one at a time">batch, 1</th>
      <th class="num" title="Per series, 16 series per call">batch, 16</th>
      <th class="num" title="Per series, 64 series per call">batch, 64</th></tr></thead>
      <tbody>
{chr(10).join(trs)}
      </tbody>
    </table>
    <p class="foot">Streaming is the cost of the next data point. Batch is the cost of a fresh series, where a
      pretrained model amortises one forward pass over many series. Hover a column for its definition.</p>
    <p class="stamp">Measured {perf_meta.get('date', '')} on {perf_meta.get('machine', '')}; built {prov['built']}.</p>
"""
    return page("Foundational prediction: speed", body, "Cost per forecast, streaming and batch, on one CPU.")


def size_page(prov, f3):
    pts = sorted(f3["points"], key=lambda p: p["mb"])
    trs = []
    for q in pts:
        mb = q["mb"]
        size = f"{mb * 1000:.0f} KB" if mb < 1 else f"{mb:,.0f} MB"
        trs.append(f'<tr><td><b>{q["name"]}</b></td><td class="num">{size}</td>'
                   f'<td class="muted small">{q["footprint_how"]}</td></tr>')
    body = f"""
    <h1>Size</h1>
    <p class="lede">What a deployment carries. <a href="https://lnkd.in/p/et-Thq-a">Discuss</a>.</p>
    {hbars([(q["name"], q["mb"]) for q in pts], _fmt_mb, "bytes each model carries", ref=next((("laplace", q["mb"]) for q in pts if q["name"] == "laplace (pure Python)"), None))}
    <p class="foot">Bar length is the fourth root of the size, between a log and a linear scale.</p>
    <table class="lb">
      <thead><tr><th>model</th><th class="num">size</th><th>measured as</th></tr></thead>
      <tbody>
{chr(10).join(trs)}
      </tbody>
    </table>
    <p class="stamp">Built {prov['built']}.</p>
"""
    return page("Foundational prediction: size", body, "Bytes each model carries: weights for pretrained models, the package for laplace.")



# what each laplace++ run was trained on (the run shown per stratum comes from LAPLACEPP_RUNS)
LAPLACEPP_TRAINING = {
    "learned(gpswa)": "the FRED weekly training half plus GIFT-Eval's weekly pretraining series (NN5, road traffic, web traffic), running weight average",
    "learned(wswa2)": "the FRED weekly training half, three passes, averaged with a run that added leak-checked daily-to-weekly series",
    "learned(swa5)": "the FRED monthly training half, four passes, average of five checkpoints along the fourth",
    "learned(e3)": "the M4 hourly training half (split by series id), three passes",
}


def _summary_rows():
    vs, cov = {}, {}
    for r in csv.DictReader(open(os.path.join(BENCH, "canonical_summary_vs_laplace.csv"))):
        vs[(r["study"], r["model"])] = r
    for r in csv.DictReader(open(os.path.join(BENCH, "canonical_summary_coverage.csv"))):
        cov[(r["study"], r["method"])] = float(r["cov_central90"])
    return vs, cov


def laplace_page(prov, perf):
    by = {(r["model"], r["regime"]): r["ms"] for r in perf["rows"]}
    rows = "".join(
        f'<tr><td><b>{m}</b></td><td class="num">{_fmt_ms(by.get((m, "warm")))}</td>'
        f'<td class="num">{_fmt_ms(by.get((m, "cold")))}</td></tr>'
        for m in ("laplace (Rust) k=1", "laplace (Python) k=1") if (m, "warm") in by)
    body = f"""
    <h1>laplace</h1>
    <p class="lede">The reference model on every leaderboard here: an online forecaster that needs no training.</p>
    <p>laplace updates its state with each new observation and issues a full predictive density for the
      next one. It is the default skater in the <a href="https://skaters.microprediction.org/">skaters</a>
      package, pure Python with a compiled Rust core, and the same model has twins in JavaScript, R and
      Julia that agree with the Python to 1e-6. How it is built, from its transform chains to the tail
      splice, is in the <a href="https://skaters.microprediction.org/guide.html">skaters guide</a>.</p>
    <h2>Cost</h2>
    <table class="lb"><thead><tr><th>build</th><th class="num">one update</th>
      <th class="num">cold start, 256 points</th></tr></thead><tbody>{rows}</tbody></table>
    <p class="foot">Measured on one CPU thread ({perf.get("meta", {}).get("machine", "")}), skaters
      {perf.get("meta", {}).get("version", "")}; foundation-model costs are on the <a href="speed.html">Speed</a> page.</p>
    <h2>Why it is the reference</h2>
    <p>It sees every observation before forecasting the next, as a deployed forecaster would, and it
      costs microseconds. A model that cannot beat it on the next data point is not worth its weight for
      that job. laplace is also one of the two components of <a href="laplacepp.html">granger</a>.</p>
    <p class="stamp">Built {prov['built']} from skaters commit {prov['skaters_commit']}.</p>
"""
    return page("laplace", body, "laplace: the training-free online forecaster used as the reference on every leaderboard.")


def laplacepp_page(prov):
    vs, cov = _summary_rows()
    trs, shown = [], []
    for x in FP_STRATA:
        st = x["id"]
        label = f'{x["kind"]}, {x["freq"]}'
        f = os.path.join(OUT, "data", st.replace(":", "-") + ".json")
        run = json.load(open(f)).get("runs", {}).get("learned(e2)") if os.path.exists(f) else None
        if run is not None and (st, run) not in vs:
            run = None
        if run is None:
            why = "not trained: too few price series on the training side" if "price" in st else "no trained run beats laplace yet"
            trs.append(f'<tr><td>{label}</td><td colspan="6" class="muted">{why}</td></tr>')
            continue
        shown.append(run)
        r = vs[(st, run)]
        e = vs.get((st, run.replace("learned(", "learned-expert(")))
        n, w, l = int(r["n_series"]), int(r["win"]), int(r["loss"])
        expert = "" if e is None else f'{float(e["med_dLL"]):+.3f}'
        trs.append(f'<tr><td>{label}</td><td class="num">{n:,}</td><td class="num">{w:,}</td>'
                   f'<td class="num">{l:,}</td><td class="num">{float(r["med_dLL"]):+.3f}</td>'
                   f'<td class="num">{expert}</td>'
                   f'<td class="num">{cov.get((st, run), float("nan")):.2f}</td></tr>')
    notes = "".join(f"<li><b>{name.replace('learned(', '').rstrip(')')}</b>: {LAPLACEPP_TRAINING.get(name, 'see the skaters repository')}.</li>"
                    for name in dict.fromkeys(shown))
    body = f"""
    <h1>granger</h1>
    <p class="lede">The trained laplace. Shown on this site as laplace++ until November 2026; the package calls it <code>granger</code>.</p>
    <p class="lede">laplace with its parameters learned from data, mixed with the original so that it can
      never do much worse.</p>
    <p>It ships in the skaters package as <code>granger</code>, in all five ports; the
      <a href="https://skaters.microprediction.org/guide.html">skaters guide</a> describes how it and
      laplace are built.</p>
    <h2>The mixture</h2>
    <p>Two forecasters each issue a predictive density for the next value: the original laplace,
      <i>p</i><sub>lap</sub>, and a trained copy of laplace's own machinery, <i>p</i><sub>exp</sub>.
      granger issues</p>
    <p class="pick"><i>q<sub>t</sub></i>(<i>y</i>) = <i>w<sub>t</sub></i> <i>p</i><sub>lap,t</sub>(<i>y</i>)
      + (1 &minus; <i>w<sub>t</sub></i>) <i>p</i><sub>exp,t</sub>(<i>y</i>),</p>
    <p>where <i>w<sub>t</sub></i> is the posterior weight on laplace given the series so far: a prior
      &pi; = 0.25 times laplace's likelihood on the observations to date, against 0.75 times the trained
      model's. This is Vovk's Aggregating Algorithm for log loss. Multiplying the mixture densities along
      a series telescopes the weights, so for every series, whatever the data,</p>
    <p class="pick">total log score of granger &ge; total log score of laplace &minus; log 4,</p>
    <p>and it is never more than log(4/3) behind the trained model. The bound needs the exact log
      densities in the weights (no floor). It concerns log score, not coverage; the coverage below is
      measured, not guaranteed.</p>
    <h2>Training and scoring</h2>
    <p>Each panel is split by series family. The trained model learns on one half by prequential
      truncated backpropagation (each forecast is issued before its outcome is used), and every number
      below is on the other half. Training data from outside the panel is never one of the held-out
      series, and series derived from FRED are screened out if they nearly duplicate one.</p>
    <h2>Against laplace, held-out series</h2>
    <table class="lb"><thead><tr><th>panel</th><th class="num">series</th><th class="num">granger wins</th>
      <th class="num">laplace wins</th><th class="num">median gain</th><th class="num">trained model alone</th>
      <th class="num">cov<sub>90</sub></th></tr></thead><tbody>{"".join(trs)}</tbody></table>
    <p class="foot">Wins are per-series Diebold&ndash;Mariano tests on log score; the rest are draws. Median
      gain is the per-series mean log-score difference in nats per step, at the median series.</p>
    <h2>Runs shown</h2>
    <ul class="small">{notes}</ul>
    <p class="stamp">Built {prov['built']} from skaters commit {prov['skaters_commit']}.</p>
"""
    return page("granger", body, "granger: a learned laplace mixed with the original by the Aggregating Algorithm, with a log-score guarantee against laplace.")


PAPERS = [
    dict(slug="horizon-note",
         title="On the Importance of Evaluating Foundational Time-Series Models One Step Ahead in an Investment Context",
         authors="Peter Cotton", status="technical note, November 2026"),
]


def _abstract(slug):
    """The paper's abstract, read from its tex so the card never goes stale."""
    import re as _re
    tex = open(os.path.join(HERE, "papers", slug, slug + ".tex")).read()
    ab = _re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, _re.S).group(1).strip()
    ab = _re.sub(r"\\(?:code|texttt|emph)\{([^}]*)\}", r"\1", ab)
    ab = ab.replace("D_{\\mathrm{KL}}(P\\|Q)", "<i>D</i><sub>KL</sub>(<i>P</i>&#8214;<i>Q</i>)")
    ab = ab.replace("G\\^arleanu", "G&acirc;rleanu").replace("$", "").replace("~", " ")
    ab = _re.sub(r"\\[A-Za-z]+", "", ab)          # any macro left over is dropped rather than shown raw
    return ab


def papers_page(prov):
    cards = []
    for p in PAPERS:
        cards.append(f"""
    <div class="paper">
      <h2><a href="./{p['slug']}.pdf">{p['title']}</a></h2>
      <p class="meta muted small">{p['authors']} &middot; {p['status']} &middot;
        <a href="./{p['slug']}.pdf">PDF</a> &middot;
        <a href="https://github.com/microprediction/foundationalprediction/blob/main/papers/{p['slug']}/{p['slug']}.tex">tex</a> &middot;
        <a href="https://github.com/microprediction/foundationalprediction/blob/main/papers/{p['slug']}/{p['slug']}.bib">bib</a></p>
      <p class="abstract">{_abstract(p['slug'])}</p>
    </div>""")
    body = f"""
    <h1>Papers</h1>
    <p class="lede">Notes and papers behind the leaderboards. Each lives in its own folder under
      <code>papers/</code> in the repository, with the PDF served from this site.</p>
{''.join(cards)}
    <h2>Related</h2>
    <p>The benchmark paper that defines the protocol used here,
      <a href="https://skaters.microprediction.org/papers.html">A stronger univariate benchmark for
      foundational time-series model evaluation</a>, lives with the <code>skaters</code> package.</p>
    <p class="stamp">Built {prov['built']}.</p>
"""
    return page("Foundational prediction: papers", body,
                "Notes and papers behind the leaderboards: the one-step case for evaluating time-series models in an investment context.")


# ---------------------------------------------------------------- model pages
# One page per model used on the site. Foundation-model metadata (vendor, size,
# licence, how it was run, links) is read from the skaters repository's
# benchmarks/foundation_pages.py (its MODELS dict), so the two sites say the
# same thing; the classical and conformal arms are described here.
FM_META_PATH = os.environ.get("SKATERS_PAGES", os.path.join(SKATERS, "benchmarks", "foundation_pages.py"))


def _fm_meta():
    try:
        src = open(FM_META_PATH).read()
    except OSError:
        return {}
    i = src.index("\nMODELS = {"); j = src.index("\n}\n", i) + 3
    ns = {"dict": dict, "_fixed_tight": lambda arm: ""}
    return eval(src[i + len("\nMODELS = "):j], ns)


_LOCAL_MODELS = {
    "csp": dict(name="Conformal Seasonal Pools", kind="conformal", perf_names=("CSP",), vendor="Valeriy Manokhin (csp-forecaster)",
                version="0.2.0", license="MIT",
                blurb="CSP forecasts a series by drawing from its own past at the same phase of the cycle, around a "
                      "seasonal-naive forecast. Nothing is trained; the pool is the predictive.",
                note="Run from the author's package as its README directs, default configuration, refit at every "
                     "step on the trailing 750 observations with 256 samples; season 24 on hourly, 5 on daily, 52 "
                     "on weekly, 12 on monthly. The combination arms join it with laplace: &amp;lap is the "
                     "likelihood-weighted portfolio, +lap stacks CSP on laplace's residuals, ~lap the reverse, "
                     "@lap recalibrates CSP's PIT against laplace.",
                links=[("Paper", "https://arxiv.org/abs/2605.03789"), ("Code", "https://github.com/valeman/csp-forecaster"),
                       ("Review", "https://conformalprediction.net/csp.html")],
                arch="A same-phase pool and a residual pool around the seasonal naive, mixed and sampled. Its density "
                     "is conditional on the phase for free, which is where it beats laplace on strongly seasonal "
                     "series; on series that do not repeat, the pool describes a regime that has moved on.",
                arms=[("CSP", "default configuration"), ("CSP(paper)", "paper-exact pool"), ("CSP&lap", "portfolio with laplace"),
                      ("CSP@lap", "PIT-recalibrated against laplace"), ("CSP+lap", "on laplace's residuals"), ("CSP~lap", "laplace on its residuals")]),
    "autoarima": dict(name="AutoARIMA", kind="classical", vendor="Nixtla (statsforecast)", version="statsforecast", license="Apache-2.0",
                      blurb="Automatic ARIMA order selection by AICc, the classical workhorse.",
                      note="Refit at every step on the full history to date, so it is online in the sense every arm here is: it "
                           "sees every observation before forecasting the next. The +id, +aci and +pid arms add a conformal "
                           "interval, adaptive conformal inference and a PID-controlled conformal interval on top.",
                      links=[("Code", "https://github.com/Nixtla/statsforecast")],
                      arch="Gaussian predictive from the fitted ARIMA; the conformal wrappers adjust its quantiles.",
                      arms=[("AutoARIMA", "refit every step"), ("AutoARIMA+id", "with a conformal interval"), ("AutoARIMA+aci", "with ACI"), ("AutoARIMA+pid", "with a PID conformal interval")]),
    "autoets": dict(name="AutoETS", kind="classical", vendor="Nixtla (statsforecast)", version="statsforecast", license="Apache-2.0",
                    blurb="Automatic exponential smoothing with error, trend and seasonal components chosen by AICc.",
                    note="Refit at every step on the full history to date.", links=[("Code", "https://github.com/Nixtla/statsforecast")],
                    arch="Gaussian predictive from the fitted state-space form.", arms=[("AutoETS", "refit every step")]),
    "autotheta": dict(name="AutoTheta", kind="classical", vendor="Nixtla (statsforecast)", version="statsforecast", license="Apache-2.0",
                      blurb="The Theta method, the M3 competition winner, with its parameters chosen automatically.",
                      note="Refit at every step on the full history to date.", links=[("Code", "https://github.com/Nixtla/statsforecast")],
                      arch="Gaussian predictive around the theta-line forecast.", arms=[("AutoTheta", "refit every step")]),
    "garcht": dict(name="GARCH-t", kind="classical", vendor="Kevin Sheppard (arch)", version="arch", license="NCSA",
                   blurb="GARCH(1,1) with Student-t innovations: the standard model for returns with volatility clustering.",
                   note="Refit at every step. On price and return series it is the recommendation, and laplace ties it there.",
                   links=[("Code", "https://github.com/bashtage/arch")],
                   arch="Conditional-variance recursion with heavy-tailed innovations; a fitted scale, not a mean model.",
                   arms=[("GARCH-t", "refit every step")]),
}


def model_specs():
    """Ordered list of page specs: foundation models from the skaters metadata that have an arm on the
    site, then the conformal baseline, then the classical arms."""
    vs, _ = _summary_rows()
    arms_in_store = {m for (_, m) in vs}
    out = []
    for slug, m in _fm_meta().items():
        arms = []
        for a, lab in [(m.get("key"), "headline"), (m.get("tight_key"), "tight reconstruction")] + list(m.get("variants", [])):
            if a and a in arms_in_store and a not in [x for x, _ in arms]:
                arms.append((a, lab))
        if not arms or not any(a in FP_MODELS for a, _ in arms):
            continue
        out.append(dict(slug=slug, name=m["name"], kind="foundation", vendor=m.get("vendor", ""), version=m.get("version", ""),
                        license=m.get("license", ""), blurb=m.get("blurb", ""), note=m.get("note", ""), links=m.get("links", []),
                        arch=m.get("arch", ""), arms=arms))
    out.sort(key=lambda d: d["name"].lower())
    for slug in ("csp", "autoarima", "autoets", "autotheta", "garcht"):
        m = dict(_LOCAL_MODELS[slug]); m["slug"] = slug
        m["arms"] = [(a, lab) for a, lab in m["arms"] if a in arms_in_store]
        out.append(m)
    return out


def _bar(w, d, l):
    tot = (w + d + l) or 1
    return (f'<span class="wdl"><i class="w" style="width:{w/tot*100:.1f}%"></i><i class="d" style="width:{d/tot*100:.1f}%"></i>'
            f'<i class="l" style="width:{l/tot*100:.1f}%"></i></span><span class="wdlnum">{fmt_pct(w/tot)} / {fmt_pct(l/tot)}</span>')


def _norm(s):
    return "".join(ch for ch in s.lower() if ch.isalnum())


# Papers behind each model, beyond the paper links in the shared metadata. Only
# references with a verified venue or arXiv number are listed; models with no
# published paper say so.
MODEL_REFS = {
    "ttm3": ['Ekambaram, V., Jati, A., Dayama, P., et al. (2024). Tiny Time Mixers (TTMs): fast pre-trained models for enhanced zero/few-shot forecasting of multivariate time series. <i>NeurIPS 2024</i>. <a href="https://arxiv.org/abs/2401.03955">arXiv:2401.03955</a>'],
    "timesfm3": ['Das, A., Kong, W., Sen, R., and Zhou, Y. (2024). A decoder-only foundation model for time-series forecasting. <i>ICML 2024</i>. <a href="https://arxiv.org/abs/2310.10688">arXiv:2310.10688</a> (the TimesFM architecture; version 3.0 is documented on its model card).'],
    "autoarima": ['Hyndman, R. J. and Khandakar, Y. (2008). Automatic time series forecasting: the forecast package for R. <i>Journal of Statistical Software</i> 27(3). <a href="https://doi.org/10.18637/jss.v027.i03">doi:10.18637/jss.v027.i03</a>'],
    "autoets": ['Hyndman, R. J., Koehler, A. B., Snyder, R. D. and Grose, S. (2002). A state space framework for automatic forecasting using exponential smoothing methods. <i>International Journal of Forecasting</i> 18(3), 439&ndash;454. <a href="https://doi.org/10.1016/S0169-2070(01)00110-8">doi:10.1016/S0169-2070(01)00110-8</a>'],
    "autotheta": ['Assimakopoulos, V. and Nikolopoulos, K. (2000). The theta model: a decomposition approach to forecasting. <i>International Journal of Forecasting</i> 16(4), 521&ndash;530. <a href="https://doi.org/10.1016/S0169-2070(00)00066-2">doi:10.1016/S0169-2070(00)00066-2</a>',
                  'Hyndman, R. J. and Billah, B. (2003). Unmasking the Theta method. <i>International Journal of Forecasting</i> 19(2), 287&ndash;290. <a href="https://doi.org/10.1016/S0169-2070(01)00143-1">doi:10.1016/S0169-2070(01)00143-1</a>'],
    "garcht": ['Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. <i>Journal of Econometrics</i> 31(3), 307&ndash;327. <a href="https://doi.org/10.1016/0304-4076(86)90063-1">doi:10.1016/0304-4076(86)90063-1</a>',
               'Bollerslev, T. (1987). A conditionally heteroskedastic time series model for speculative prices and rates of return. <i>Review of Economics and Statistics</i> 69(3), 542&ndash;547. <a href="https://doi.org/10.2307/1925546">doi:10.2307/1925546</a>'],
}
NO_PAPER = {"timestral", "timeh2o", "ciproto", "hirondel", "simons", "timessm"}
SITE_PAPERS = ['Cotton, P. (2026). A stronger univariate benchmark for foundational time-series model evaluation. <a href="https://skaters.microprediction.org/papers.html">skaters papers</a> (the protocol and the reconstruction question used on every table here).',
               'Cotton, P. (2026). On the importance of evaluating foundational time-series models one step ahead in an investment context. <a href="./horizon-note.pdf">PDF</a> (why the leaderboards are one step ahead).']


def references_block(spec):
    items = [f'{t}: <a href="{u}">{u}</a>' for t, u in spec.get("links", []) if t.lower().startswith(("paper", "report"))]
    items += MODEL_REFS.get(spec["slug"], [])
    if not items and spec["slug"] in NO_PAPER:
        items = ["No paper has been published for this model; the model card and code linked above are its documentation."]
    out = "<h2>References</h2><ol class='refs'>" + "".join(f"<li>{x}</li>" for x in items) + "</ol>" if items else ""
    out += "<h3>Evaluated on this site in</h3><ul class='refs'>" + "".join(f"<li>{x}</li>" for x in SITE_PAPERS) + "</ul>"
    return out


def model_page(spec, prov, perf, mb):
    vs, cov = _summary_rows()
    trs = []
    for arm, lab in spec["arms"]:
        first = True
        for x in FP_STRATA:
            r = vs.get((x["id"], arm))
            if r is None:
                continue
            c = cov.get((x["id"], arm))
            w, d, l = int(r["win"]), int(r["draw"]), int(r["loss"])
            med = float(r["med_dLL"]); crps = r.get("med_crps_ratio", "")
            cov90 = f'{c:.3f}' if c is not None else "&ndash;"
            armcell = f'<td><code>{arm}</code><br><span class="muted small">{lab}</span></td>' if first else '<td></td>'
            first = False
            trs.append(f'<tr>{armcell}<td>{x["kind"]}, {x["freq"]}</td><td class="num">{int(r["n_series"]):,}</td>'
                       f'<td>{_bar(w, d, l)}</td><td class="num">{med:+.3f}</td><td class="num">{float(crps):.3f}</td><td class="num">{cov90}</td></tr>')
    table = ('<table class="lb"><thead><tr><th>arm</th><th>panel</th><th class="num">series</th><th>wins / losses vs laplace</th>'
             '<th class="num">median &Delta;LL</th><th class="num">CRPS ratio</th><th class="num">coverage 90</th></tr></thead>'
             f'<tbody>{"".join(trs)}</tbody></table>') if trs else '<p class="muted">No scored arm on the current panel.</p>'
    # cost and size, best-effort name match against the perf and footprint records
    key = _norm(spec["name"])
    names = spec.get("perf_names")
    match = (lambda m: m in names) if names else (lambda m: _norm(m).startswith(key[:6]) or key.startswith(_norm(m)[:6]))
    warm = [(m, ms) for (m, reg), ms in perf.items() if reg == "warm" and match(m)]
    size = [(m, v) for m, v in mb.items() if _norm(m).startswith(key[:6]) or key.startswith(_norm(m)[:6])]
    cost = ""
    if warm or size:
        cost = "<h2>Cost and size</h2><ul>"
        for m, ms in warm[:3]:
            cost += f"<li>{m}: {_fmt_ms(ms)} per forecast on one CPU thread (<a href=\"speed.html\">speed</a>)</li>"
        for m, v in size[:3]:
            cost += f"<li>{m}: {v:,.0f} MB of weights and runtime (<a href=\"size.html\">size</a>)</li>"
        cost += "</ul>"
    links = " &middot; ".join(f'<a href="{u}">{t}</a>' for t, u in spec.get("links", []))
    from urllib.parse import quote as _q
    head_arm = next((a for a, _ in spec["arms"] if a in FP_MODELS), spec["arms"][0][0] if spec["arms"] else None)
    res = "".join(f'<li><a href="{u}">{t}</a> &mdash; <span class="muted small">{u}</span></li>' for t, u in spec.get("links", []))
    resources = (f'<h2>Resources</h2><ul class="refs">{res}</ul>') if res else ""
    tool = (f'<p>On the <a href="https://foundationalprediction.com/#All/All/{_q(head_arm)}">interactive leaderboard</a> '
            f'you can make {spec["name"]} the reference and see every other model paired against it, on any panel or subset '
            f'of series; the tables below keep laplace as the reference.</p>') if head_arm else ""
    body = f"""
    <h1>{spec['name']}</h1>
    <p class="muted small">{spec['kind']} &middot; {spec['vendor']} &middot; {spec['version']} &middot; licence: {spec['license']}{(' &middot; ' + links) if links else ''}</p>
    <p class="lede">{spec['blurb']}</p>
    <h2>How it was run</h2>
    <p>{spec['note']}</p>
    {('<p>' + spec['arch'] + '</p>') if spec.get('arch') else ''}
    {resources}
    <h2>Against laplace, one step ahead</h2>
    {tool}
    <p class="small muted">Per series, Diebold&ndash;Mariano on the log score; a win is the model ahead beyond 1.96 standard
      errors, a loss laplace ahead, the rest draws. Median &Delta;LL is the model's log score minus laplace's, in nats per step;
      CRPS ratio is model over laplace (below 1 favours the model); coverage is the share of outcomes inside the central 90%.
      The <a href="methodology.html">methodology</a> page has the protocol; the <a href="index.html">leaderboards</a> place
      the model among the rest.</p>
    {table}
    {cost}
    {references_block(spec)}
    <p class="stamp">Built {prov['built']} from skaters commit {prov['skaters_commit']}.</p>
"""
    return page(f"Foundational prediction: {spec['name']}", body, f"{spec['name']}: how it was run and how it compares with laplace one step ahead on every panel.")


def models_nav_block(specs):
    head = lambda t: f'<span class="drop-head">{t}</span>'
    items = [head("Reference"), '<a href="./laplace.html">laplace</a>', '<a href="./laplacepp.html">granger</a>']
    for kind, title in (("foundation", "Foundational models"), ("conformal", "Conformal"), ("classical", "Classical")):
        group = [s for s in specs if s["kind"] == kind]
        if group:
            items.append(head(title))
            items += [f'<a href="./model-{s["slug"]}.html">{s["name"]}</a>' for s in group]
    return "\n".join("            " + it for it in items)


# A model is listed only once it is scored on at least this share of the series the
# best-covered model has in every stratum. New arms join the study breadth-first and
# would otherwise appear with a few dozen series beside rows with thousands.
MIN_SHARE = 0.5
# Only arms still filling in are gated. Arms already published stay up while the
# panel grows under them (the daily:price stratum grew to 8.7k series in November).
GATED = {"PatchTST-FM-r2(q99)", "Tafsut-base(wide)", "Tabby-Pretrain(q99)", "Yug(wide)",
         "CITRAS-FM(wide)", "TS-ICL(q99)", "Kairos-50m(wide)", "Timer-S1(wide)",
         "Kumo-large-reg(q99)"}


def gate_by_coverage(vs):
    """Drop from FP_MODELS / EXPLORER_MODELS every non-reference arm below MIN_SHARE
    coverage in any stratum; return the held-back arms with their coverage."""
    global FP_MODELS, EXPLORER_MODELS
    full = {}
    for (study, _model), r in vs.items():
        full[study] = max(full.get(study, 0), int(r["n_series"]))
    held = {}
    for arm, (_name, kind) in list(FP_MODELS.items()):
        if kind == "ours" or arm not in GATED:
            continue
        short = [f"{s} {int(vs[(s, arm)]['n_series']) if (s, arm) in vs else 0}/{n}"
                 for s, n in full.items()
                 if (s, arm) not in vs or int(vs[(s, arm)]["n_series"]) < MIN_SHARE * n]
        if short:
            held[arm] = short
    FP_MODELS = {k: v for k, v in FP_MODELS.items() if k not in held}
    EXPLORER_MODELS = [m for m in EXPLORER_MODELS if m not in held]
    return held


def main():
    vs, cov, warm, mb, perf_meta, f3_meta = load()
    held = gate_by_coverage(vs)
    for arm, short in held.items():
        print(f"[build] held back (coverage below {MIN_SHARE:.0%}): {arm}: {', '.join(short)}")
    prov = provenance()
    specs = model_specs()
    global NAV
    i0 = NAV.index('<a href="./laplace.html">laplace</a>'); i1 = NAV.index('<a href="./laplacepp.html">granger</a>') + len('<a href="./laplacepp.html">granger</a>')
    NAV = NAV[:NAV.rfind("\n", 0, i0) + 1] + models_nav_block(specs) + NAV[i1:]
    os.makedirs(OUT, exist_ok=True)
    shutil.copy(os.path.join(SKATERS, "docs", "academic.css"), os.path.join(OUT, "academic.css"))
    shutil.copy(os.path.join(SKATERS, "docs", "assets", "frontier3d.js"), os.path.join(OUT, "frontier3d.js"))
    import explorer
    written = explorer.export(BENCH, OUT, EXPLORER_MODELS, aliases={"learned(e2)": LAPLACEPP_RUNS})
    pages = {"laplace": "laplace.html", "learned(e2)": "laplacepp.html"}
    for spec in specs:
        for a, _lab in spec["arms"]:
            pages.setdefault(a, f"model-{spec['slug']}.html")
    open(os.path.join(OUT, "index.html"), "w").write(leaderboard(prov, written, pages))
    open(os.path.join(OUT, "frontier.html"), "w").write(frontier_page(prov))
    perf = json.load(open(os.path.join(BENCH, "perf_results.json")))
    f3 = json.load(open(os.path.join(BENCH, "frontier3d.json")))
    open(os.path.join(OUT, "speed.html"), "w").write(speed_page(prov, perf, perf.get("meta", {})))
    open(os.path.join(OUT, "size.html"), "w").write(size_page(prov, f3))
    open(os.path.join(OUT, "methodology.html"), "w").write(methodology(prov))
    open(os.path.join(OUT, "laplace.html"), "w").write(laplace_page(prov, perf))
    open(os.path.join(OUT, "laplacepp.html"), "w").write(laplacepp_page(prov))
    perf_by = {(r["model"], r["regime"]): r["ms"] for r in perf["rows"]}
    for spec in specs:
        open(os.path.join(OUT, f"model-{spec['slug']}.html"), "w").write(model_page(spec, prov, perf_by, mb))
    open(os.path.join(OUT, "gift.html"), "w").write(gift())
    open(os.path.join(OUT, "papers.html"), "w").write(papers_page(prov))
    open(os.path.join(OUT, ".nojekyll"), "w").write("")
    json.dump(prov, open(os.path.join(OUT, "provenance.json"), "w"), indent=1)
    print(f"[build] docs/ written from skaters {prov['skaters_commit']}")


if __name__ == "__main__":
    main()
