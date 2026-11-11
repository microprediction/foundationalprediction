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
            <a href="./laplacepp.html">laplace++</a>
          </span>
        </span>
        <a href="./methodology.html">Methodology</a>
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
    <p>Every model is paired against laplace, the training-free online yardstick, on identical series
      and windows. Per series the mean log-score differential is tested by Diebold&ndash;Mariano with a
      Newey&ndash;West standard error; a win or a loss is a difference beyond 1.96 standard errors, the
      rest are draws. A draw says the window cannot separate the two, not that they are equal, so the
      median gap is the better measure of size.</p>
    <h2>Baselines that keep learning</h2>
    <p>The classical baselines in published suites are fit once at the window origin and never updated.
      Here every baseline is online: it sees every observation before it forecasts the next. The gap
      between foundation models and classical methods shown here is therefore the gap a practitioner
      running a stream would see. AutoARIMA, AutoETS, AutoTheta and GARCH-t are refit at every step.</p>
    <h2>Trained models</h2>
    <p>laplace++ is the one trained model here. It is trained on half of each panel, split by series
      family, and every number shown for it is on the other half. Training data added from outside the
      panel is never one of the held-out series. How it is built and what it guarantees are on the
      <a href="laplacepp.html">laplace++</a> page.</p>
    <h2>Roster</h2>
    <p>Only the best models in each class, each with public code and a pinned checkpoint that was run
      here. Submissions that cannot be rerun are not listed.</p>
    <h2>Licences and versions</h2>
    <p>The licence column gives each model's terms. Of the seven foundation models, three are open,
      two are free below a revenue threshold, and TimesFM 3.0 and TimeH2O are licensed for testing,
      evaluation and research only. laplace, laplace++ and the classical models are open. All
      evaluation here is non-commercial benchmarking. TimeH2O is h2oai/timeh2o-v1 at Hugging Face
      revision c556990, whose weights are unchanged since revision 60b68fa of 2026-10-22, run zero-shot
      on CPU on 2026-11-03 and 2026-11-04 by the protocol above.</p>
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
    "laplace": ("laplace", "ours"), "learned(e2)": ("laplace++", "ours"),
    "Timestral-209M(q21)": ("Timestral-209M", "foundation"), "t0-gamma(q21)": ("t0-gamma", "foundation"),
    "TimesFM3(wide)": ("TimesFM 3.0", "foundation"), "TimesFM(wide)": ("TimesFM 2.5", "foundation"),
    "TiRex(wide)": ("TiRex", "foundation"), "Chronos(wide)": ("Chronos-Bolt", "foundation"),
    "TimeH2O-v1(wide)": ("TimeH2O", "foundation"),
    "AutoARIMA": ("AutoARIMA", "classical"), "AutoETS": ("AutoETS", "classical"),
    "AutoTheta": ("AutoTheta", "classical"), "GARCH-t": ("GARCH-t", "classical"),
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
    "AutoARIMA": _APACHE[:2] + ("https://github.com/Nixtla/statsforecast",) + _APACHE[3:],
    "AutoETS": _APACHE[:2] + ("https://github.com/Nixtla/statsforecast",) + _APACHE[3:],
    "AutoTheta": _APACHE[:2] + ("https://github.com/Nixtla/statsforecast",) + _APACHE[3:],
    "GARCH-t": ("NCSA", "open", "https://github.com/bashtage/arch/blob/main/LICENSE.md", "University of Illinois/NCSA (arch package): commercial use allowed"),
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

FP_STRATA = [
    {"id": "weekly:econ", "kind": "Economic", "freq": "Weekly"},
    {"id": "monthly:econ", "kind": "Economic", "freq": "Monthly"},
    {"id": "daily:econ", "kind": "Economic", "freq": "Daily"},
    {"id": "m4-hourly:econ", "kind": "Economic", "freq": "Hourly (M4)"},
    {"id": "daily:price", "kind": "Prices", "freq": "Daily"},
]


def leaderboard(prov, written):
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
        <span class="lab">Frequency</span><div class="segs" id="ex-freq"></div>
      </div>
      <span class="lab">Reference</span><div><select id="ex-ref"></select>
        <label class="small muted" style="margin-left:14px" title="laplace++ is trained on one half of the series and scored only on the other. Tick to score every model on that held-out half, so all rows cover the same series.">
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
    <p class="foot">laplace++ is trained on one half of each panel (split by series family) and scored only on the other
      half, so its row covers fewer series; tick <i>held-out series only</i> to put every model on that same half.</p>
    <p class="foot">* By log score: a per-series Diebold&ndash;Mariano test on the log predictive density, 64 one-step
      points per series. Click a column to sort.</p>
    <div class="legend"><span><span class="dot foundation"></span>foundation model</span>
      <span><span class="dot classical"></span>classical, refit every step</span>
      <span><span class="dot ours"></span>laplace and laplace++</span>
      <span><span class="lic open">open</span> commercial use allowed</span>
      <span><span class="lic restricted">restricted</span> free below a revenue threshold</span>
      <span><span class="lic noncommercial">non-commercial</span> research and evaluation only</span></div>
    <p class="foot">TimeH2O: h2oai/timeh2o-v1 at Hugging Face revision c556990 (weights unchanged since 60b68fa, 2026-10-22), evaluated 2026-11-03 to 2026-11-04
      under its non-commercial licence. Method: <a href="methodology.html">methodology</a>.</p>
    <p class="stamp">Built {prov['built']} from skaters commit {prov['skaters_commit']} ({prov['skaters_date']}).</p>
    <script>window.FP_NAMES = {json.dumps(names)}; window.FP_KIND = {json.dumps(kinds)}; window.FP_STRATA = {json.dumps(strata)}; window.FP_LIC = {json.dumps(FP_LICENSES)};</script>
    <script>{explorer.EXPLORER_JS}</script>
"""
    return page("Foundational prediction: the next data point", body,
                "Which model to use for the next data point: foundation, classical and online models paired on identical series, filtered by series type, with a choice of reference.",
                extra_css=explorer.EXPLORER_CSS)


def frontier_page(prov):
    body = f"""
    <h1>Performance</h1>
    <p class="lede">One point per model. Drag to rotate.</p>
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


_KCOL = {"ours": "#1a8c4a", "foundation": "#8f84ff", "classical": "#e09b2d"}


def _times(r):
    """A multiple to two significant figures: 1.7x, 14x, 370x, 37,000x."""
    import math
    if r < 10:
        return f"{r:.1f}×"
    p = 10 ** (math.floor(math.log10(r)) - 1)
    return f"{round(r / p) * p:,.0f}×"


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
    return f"{mb * 1000:,.0f} KB" if mb < 1 else (f"{mb:,.0f} MB" if mb < 1000 else f"{mb / 1000:,.1f} GB")


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
    <p class="lede">What a deployment carries.</p>
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
      package, pure Python with a compiled Rust core, and the same model has twins in JavaScript and R
      that agree with the Python to 1e-6.</p>
    <h2>Cost</h2>
    <table class="lb"><thead><tr><th>build</th><th class="num">one update</th>
      <th class="num">cold start, 256 points</th></tr></thead><tbody>{rows}</tbody></table>
    <p class="foot">Measured on one CPU thread ({perf.get("meta", {}).get("machine", "")}), skaters
      {perf.get("meta", {}).get("version", "")}; foundation-model costs are on the <a href="speed.html">Speed</a> page.</p>
    <h2>Why it is the reference</h2>
    <p>It sees every observation before forecasting the next, as a deployed forecaster would, and it
      costs microseconds. A model that cannot beat it on the next data point is not worth its weight for
      that job. laplace is also one of the two components of <a href="laplacepp.html">laplace++</a>.</p>
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
    <h1>laplace++</h1>
    <p class="lede">laplace with its parameters learned from data, mixed with the original so that it can
      never do much worse.</p>
    <h2>The mixture</h2>
    <p>Two forecasters each issue a predictive density for the next value: the original laplace,
      <i>p</i><sub>lap</sub>, and a trained copy of laplace's own machinery, <i>p</i><sub>exp</sub>.
      laplace++ issues</p>
    <p class="pick"><i>q<sub>t</sub></i>(<i>y</i>) = <i>w<sub>t</sub></i> <i>p</i><sub>lap,t</sub>(<i>y</i>)
      + (1 &minus; <i>w<sub>t</sub></i>) <i>p</i><sub>exp,t</sub>(<i>y</i>),</p>
    <p>where <i>w<sub>t</sub></i> is the posterior weight on laplace given the series so far: a prior
      &pi; = 0.25 times laplace's likelihood on the observations to date, against 0.75 times the trained
      model's. This is Vovk's Aggregating Algorithm for log loss. Multiplying the mixture densities along
      a series telescopes the weights, so for every series, whatever the data,</p>
    <p class="pick">total log score of laplace++ &ge; total log score of laplace &minus; log 4,</p>
    <p>and it is never more than log(4/3) behind the trained model. The bound needs the exact log
      densities in the weights (no floor). It concerns log score, not coverage; the coverage below is
      measured, not guaranteed.</p>
    <h2>Training and scoring</h2>
    <p>Each panel is split by series family. The trained model learns on one half by prequential
      truncated backpropagation (each forecast is issued before its outcome is used), and every number
      below is on the other half. Training data from outside the panel is never one of the held-out
      series, and series derived from FRED are screened out if they nearly duplicate one.</p>
    <h2>Against laplace, held-out series</h2>
    <table class="lb"><thead><tr><th>panel</th><th class="num">series</th><th class="num">laplace++ wins</th>
      <th class="num">laplace wins</th><th class="num">median gain</th><th class="num">trained model alone</th>
      <th class="num">cov<sub>90</sub></th></tr></thead><tbody>{"".join(trs)}</tbody></table>
    <p class="foot">Wins are per-series Diebold&ndash;Mariano tests on log score; the rest are draws. Median
      gain is the per-series mean log-score difference in nats per step, at the median series.</p>
    <h2>Runs shown</h2>
    <ul class="small">{notes}</ul>
    <p class="stamp">Built {prov['built']} from skaters commit {prov['skaters_commit']}.</p>
"""
    return page("laplace++", body, "laplace++: a learned laplace mixed with the original by the Aggregating Algorithm, with a log-score guarantee against laplace.")


def main():
    vs, cov, warm, mb, perf_meta, f3_meta = load()
    prov = provenance()
    os.makedirs(OUT, exist_ok=True)
    shutil.copy(os.path.join(SKATERS, "docs", "academic.css"), os.path.join(OUT, "academic.css"))
    shutil.copy(os.path.join(SKATERS, "docs", "assets", "frontier3d.js"), os.path.join(OUT, "frontier3d.js"))
    import explorer
    written = explorer.export(BENCH, OUT, list(FP_MODELS), aliases={"learned(e2)": LAPLACEPP_RUNS})
    open(os.path.join(OUT, "index.html"), "w").write(leaderboard(prov, written))
    open(os.path.join(OUT, "frontier.html"), "w").write(frontier_page(prov))
    perf = json.load(open(os.path.join(BENCH, "perf_results.json")))
    f3 = json.load(open(os.path.join(BENCH, "frontier3d.json")))
    open(os.path.join(OUT, "speed.html"), "w").write(speed_page(prov, perf, perf.get("meta", {})))
    open(os.path.join(OUT, "size.html"), "w").write(size_page(prov, f3))
    open(os.path.join(OUT, "methodology.html"), "w").write(methodology(prov))
    open(os.path.join(OUT, "laplace.html"), "w").write(laplace_page(prov, perf))
    open(os.path.join(OUT, "laplacepp.html"), "w").write(laplacepp_page(prov))
    open(os.path.join(OUT, "gift.html"), "w").write(gift())
    open(os.path.join(OUT, ".nojekyll"), "w").write("")
    json.dump(prov, open(os.path.join(OUT, "provenance.json"), "w"), indent=1)
    print(f"[build] docs/ written from skaters {prov['skaters_commit']}")


if __name__ == "__main__":
    main()
