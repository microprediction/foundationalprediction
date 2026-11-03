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
        trs.append(f'<tr{cls}><td><b>{x["name"]}</b><br><span class="muted small">{x["kind"]}; {x["note"]}</span></td>'
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
      <a class="brand" href="/">foundational prediction</a>
      <nav>
        <a href="/">Next data point</a>
        <a href="/methodology.html">Methodology</a>
        <a href="/horizons.html">Horizons</a>
        <a href="/gift.html">GIFT reconciliation</a>
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
    table.lb td.num, table.lb th.num { text-align:right; font-variant-numeric:tabular-nums; }
    tr.yard td { background:#f4f9f5; }
    .pick { background:#f6f4ff; border-left:4px solid #4a3aff; padding:8px 12px; margin:10px 0 18px; }
    .stamp { color:#888; font-size:.85em; font-style:italic; }
    .tabs { display:flex; flex-wrap:wrap; gap:4px; border-bottom:2px solid #ddd; margin:14px 0 0; }
    .tab { background:none; border:none; border-bottom:3px solid transparent; margin-bottom:-2px;
           padding:8px 14px; font:inherit; font-size:.95rem; color:#555; cursor:pointer; }
    .tab:hover { color:#111; }
    .tab.on { color:#4a3aff; border-bottom-color:#4a3aff; font-weight:600; }
    .panel { padding-top:6px; overflow-x:auto; }
"""


def page(title, body, desc):
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>{title}</title>
  <meta name="description" content="{desc}" />
  <link rel="stylesheet" href="./academic.css" />
  <style>{STYLE}</style>
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
      running a stream would see. The fully online ARIMA and ETS rows are being added.</p>
    <h2>Trained models</h2>
    <p>A model trained on this panel is scored only on series whose family it never saw: the panel is
      split by series family, and the learned rows report the held-out half. The clean out-of-domain
      test is GIFT-Eval's data under this protocol.</p>
    <h2>Roster</h2>
    <p>Only the best models in each class, each with public code and a pinned checkpoint that was run
      here. Submissions that cannot be rerun are not listed.</p>
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


def main():
    vs, cov, warm, mb, perf_meta, f3_meta = load()
    prov = provenance()
    os.makedirs(OUT, exist_ok=True)
    shutil.copy(os.path.join(SKATERS, "docs", "academic.css"), os.path.join(OUT, "academic.css"))
    shutil.copy(os.path.join(SKATERS, "docs", "assets", "frontier3d.js"), os.path.join(OUT, "frontier3d.js"))
    open(os.path.join(OUT, "index.html"), "w").write(index(vs, cov, warm, mb, prov, perf_meta))
    open(os.path.join(OUT, "methodology.html"), "w").write(methodology(prov))
    open(os.path.join(OUT, "horizons.html"), "w").write(horizons())
    open(os.path.join(OUT, "gift.html"), "w").write(gift())
    open(os.path.join(OUT, ".nojekyll"), "w").write("")
    json.dump(prov, open(os.path.join(OUT, "provenance.json"), "w"), indent=1)
    print(f"[build] docs/ written from skaters {prov['skaters_commit']}")


if __name__ == "__main__":
    main()
