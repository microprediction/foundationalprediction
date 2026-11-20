"""Approximate importance weights over forecast leads for a trader with costs.

Garleanu and Pedersen (2013): the aim portfolio weights the expected one-interval
price change at lead j by z(1-z)^(j-1), z = gamma/(gamma + a). With no discounting,
lambda/gamma = (a/gamma)(1 + a/gamma).

Calibration (ours, approximate): match a linear one-way cost c to the model's
quadratic cost at the trade the policy makes, a fraction a/lambda of the Markowitz
position n*. With n* = alpha*P/(gamma*sigma^2) and alpha the expected return per
interval, that gives a/gamma = 2c/alpha, so

    1 - z = (2c/alpha) / (1 + 2c/alpha),   mean lead = 1 + 2c/alpha.

The leads that matter run out to roughly the time accumulated alpha takes to pay
for a round trip.
"""
import json

# (label, one-way cost in bps, per-interval volatility in percent, information coefficient, interval)
# Illustrative assumptions, not measurements. alpha = IC * sigma per interval.
PRESETS = [
    ("Frictionless, any interval", 0.0, 1.0, 0.02, "interval"),
    ("Daily index futures, 1 bp", 1.0, 1.0, 0.02, "day"),
    ("Weekly FX, 2 bp", 2.0, 1.4, 0.03, "week"),
    ("Monthly asset allocation, 10 bp", 10.0, 4.0, 0.05, "month"),
    ("Daily US large-cap equity, 5 bp", 5.0, 1.5, 0.02, "day"),
    ("Daily equity, 30 bp", 30.0, 2.0, 0.02, "day"),
    ("Hourly crypto (taker), 10 bp", 10.0, 0.6, 0.02, "hour"),
]


def ratio(c_bps, sigma_pct, ic):
    """2c/alpha with alpha = IC * sigma (both per interval, in bps)."""
    alpha_bps = ic * sigma_pct * 100.0
    return 2.0 * c_bps / alpha_bps


def weights(r, n):
    """w_j = z(1-z)^(j-1), j = 1..n, with 1 - z = r/(1+r)."""
    q = r / (1.0 + r)
    return [(1.0 - q) * q ** (j - 1) for j in range(1, n + 1)]


def _row(label, c, s, ic, unit):
    r = ratio(c, s, ic)
    w = weights(r, 400)
    w2 = [x * x for x in w]
    t2 = sum(w2)
    return dict(label=label, c=c, sigma=s, ic=ic, unit=unit, alpha=ic * s * 100.0, mean=1.0 + r,
                w1=w[0], m5=sum(w[:5]), m12=sum(w[:12]), sq1=w2[0] / t2)


def page_body():
    rows = [_row(*p) for p in PRESETS]
    trs = "".join(
        f'<tr><td>{x["label"]}</td><td class="num">{x["c"]:g}</td><td class="num">{x["sigma"]:g}%</td>'
        f'<td class="num">{x["ic"]:g}</td><td class="num">{x["alpha"]:.1f}</td>'
        f'<td class="num">{x["mean"]:.1f} {x["unit"]}s</td><td class="num">{x["w1"]:.0%}</td>'
        f'<td class="num">{x["m5"]:.0%}</td><td class="num">{x["m12"]:.0%}</td></tr>'
        for x in rows)
    presets_js = json.dumps([dict(label=p[0], c=p[1], sigma=p[2], ic=p[3], unit=p[4]) for p in PRESETS])
    return f"""
    <h1>Horizon weights</h1>
    <p class="lede">How much each forecast lead matters to a trader who pays to trade, approximately.</p>
    <p>A frictionless log-optimal trader uses only the next interval's predictive distribution, which is why the
      <a href="index.html">leaderboard</a> scores one step ahead (<a href="horizon-note.pdf">note</a>). With trading costs,
      later leads enter today's trade. In the quadratic-cost model of Gârleanu and Pedersen (2013) the expected
      one-interval price change at lead <i>j</i> gets weight <i>z</i>(1&minus;<i>z</i>)<sup><i>j</i>&minus;1</sup>.
      This page puts a number on <i>z</i> for common trading set-ups.</p>

    <div class="picker hw-picker">
      <label class="lab" for="hw-preset">Set-up</label><select id="hw-preset"></select>
      <label class="lab" for="hw-c">one-way cost (bp)</label><input id="hw-c" type="number" min="0" step="0.5">
      <label class="lab" for="hw-s">volatility per interval (%)</label><input id="hw-s" type="number" min="0.01" step="0.1">
      <label class="lab" for="hw-ic">signal IC</label><input id="hw-ic" type="number" min="0.001" step="0.005">
    </div>
    <p id="hw-summary" class="pick"></p>
    <div id="hw-chart" class="hw-chart" role="img" aria-label="Weight on each forecast lead"></div>
    <p class="small muted">Bars: the weight on each lead. Dashed line: equal weight on each of 48 leads, which is how a score
      pooled over a 48-step window (GIFT-Eval's hourly setting) weights them. Hover a bar for its value.</p>

    <h2>Presets</h2>
    <table class="lb"><thead><tr><th>set-up</th><th class="num">cost (bp)</th><th class="num">vol</th><th class="num">IC</th>
      <th class="num" title="expected return per interval, IC times volatility">alpha (bp)</th><th class="num">mean lead</th>
      <th class="num">lead 1</th><th class="num">leads 1&ndash;5</th><th class="num">leads 1&ndash;12</th></tr></thead>
      <tbody>{trs}</tbody></table>
    <p class="small muted">The cost, volatility and IC columns are illustrative assumptions, not measurements. Edit them above.</p>

    <h2>Where the number comes from</h2>
    <p>Gârleanu and Pedersen's trader pays &frac12;&lambda;&sigma;<sup>2</sup>(&Delta;<i>n</i>)<sup>2</sup> to trade and
      has risk aversion &gamma;. The weights depend on the cost only through &lambda;/&gamma;, and with no discounting
      <i>z</i> = &gamma;/(&gamma; + <i>a</i>) with &lambda;/&gamma; = (<i>a</i>/&gamma;)(1 + <i>a</i>/&gamma;).</p>
    <p>Real costs are closer to linear: a one-way cost <i>c</i> per unit traded. We match the two at the trade the
      policy actually makes, a fraction <i>a</i>/&lambda; of the Markowitz position. That gives
      <i>a</i>/&gamma; = 2<i>c</i>/&alpha;, where &alpha; is the expected return per interval, taken here as IC times
      volatility. So the mean lead is 1 + 2<i>c</i>/&alpha;. In words: the leads that matter run out to about the
      time the signal's accumulated alpha takes to pay for a round trip.</p>

    <h2>What the weights are not</h2>
    <p>They are an approximation in three places.</p>
    <ul>
      <li>The calibration from linear to quadratic cost is ours. The <a href="horizon-note.pdf">note</a> deliberately
        stops at the model; this page adds a rough map from a trading set-up to it.</li>
      <li>The weights are a trading policy for expected returns. The loss from forecast errors <i>e</i><sub><i>j</i></sub>
        is proportional to (&Sigma;<i>w</i><sub><i>j</i></sub><i>e</i><sub><i>j</i></sub>)<sup>2</sup>, which has
        cross-lead terms. If the errors at different leads were independent, the importance of lead <i>j</i> would be
        <i>w</i><sub><i>j</i></sub><sup>2</sup>, which is even more concentrated on lead 1. Either way, a weighted sum of
        per-lead scores is an approximation to the decision loss, not the loss itself.</li>
      <li>Lead <i>j</i> means the return over the single interval <i>j</i> steps ahead. A pooled benchmark scores the
        level at step <i>j</i>, which bundles every interval up to <i>j</i>.</li>
    </ul>
    <script>window.HW_PRESETS = {presets_js};</script>
    <script>{HW_JS}</script>
"""


HW_CSS = """
    .hw-picker { display:grid; grid-template-columns:max-content 1fr; gap:6px 12px; align-items:center; max-width:520px; }
    .hw-picker input, .hw-picker select { max-width:260px; }
    .hw-chart { position:relative; margin:8px 0 4px; }
    .hw-chart svg { display:block; width:100%; height:auto; }
    .hw-tip { position:absolute; pointer-events:none; background:var(--panel); border:1px solid var(--border);
              border-radius:4px; padding:3px 7px; font-size:.8rem; color:var(--text); display:none; white-space:nowrap; }
"""

HW_JS = r"""
(function () {
  var P = window.HW_PRESETS, sel = document.getElementById('hw-preset');
  var ci = document.getElementById('hw-c'), si = document.getElementById('hw-s'), ii = document.getElementById('hw-ic');
  var unit = 'interval';
  P.forEach(function (p, i) { var o = document.createElement('option'); o.value = i; o.textContent = p.label; sel.appendChild(o); });
  function load(i) { var p = P[i]; ci.value = p.c; si.value = p.sigma; ii.value = p.ic; unit = p.unit; draw(); }
  function pct(x) { return (100 * x).toFixed(x < 0.1 ? 1 : 0) + '%'; }
  function draw() {
    var c = +ci.value, s = +si.value, ic = +ii.value;
    var alpha = ic * s * 100, r = alpha > 0 ? 2 * c / alpha : 0, q = r / (1 + r);
    var w = []; for (var j = 1; j <= 400; j++) w.push((1 - q) * Math.pow(q, j - 1));
    var cum = 0, n = 1; for (var k = 0; k < w.length; k++) { cum += w[k]; if (cum >= 0.95) { n = k + 1; break; } }
    n = Math.max(12, Math.min(n, 120));
    var m5 = w.slice(0, 5).reduce(function (a, b) { return a + b; }, 0);
    document.getElementById('hw-summary').innerHTML = 'Alpha ' + alpha.toFixed(1) + ' bp per ' + unit +
      '. Mean lead <b>' + (1 + r).toFixed(1) + ' ' + unit + 's</b>; lead 1 carries <b>' + pct(w[0]) +
      '</b> of the weight and leads 1&ndash;5 carry ' + pct(m5) + '.';
    var W = 640, H = 236, L = 40, B = 44, T = 8, R = 8, pw = (W - L - R) / n, ymax = Math.max(w[0], 1 / 48) * 1.1;
    var y = function (v) { return T + (H - T - B) * (1 - v / ymax); };
    var svg = '<svg viewBox="0 0 ' + W + ' ' + H + '" xmlns="http://www.w3.org/2000/svg">';
    [0, 0.25, 0.5, 0.75, 1].forEach(function (f) { var v = f * ymax / 1.1, yy = y(v);
      svg += '<line x1="' + L + '" x2="' + (W - R) + '" y1="' + yy + '" y2="' + yy + '" stroke="#e8e8e8"/>' +
             '<text x="' + (L - 6) + '" y="' + (yy + 4) + '" text-anchor="end" font-size="11" fill="#5a5a5a">' + pct(v) + '</text>'; });
    for (var i = 0; i < n; i++) {
      var x = L + i * pw + 1, bw = Math.max(1, pw - 2), y0 = y(w[i]), h = Math.max(0, (H - B) - y0);
      svg += '<rect class="hwb" data-i="' + i + '" x="' + x + '" y="' + y0 + '" width="' + bw + '" height="' + h + '" rx="' + Math.min(4, bw / 2) + '" fill="#4a3aff"/>';
      svg += '<rect class="hwh" data-i="' + i + '" x="' + (L + i * pw) + '" y="' + T + '" width="' + pw + '" height="' + (H - T - B) + '" fill="transparent"/>';
      if (i === 0 || (i + 1) % Math.ceil(n / 12) === 0) svg += '<text x="' + (x + bw / 2) + '" y="' + (H - B + 16) + '" text-anchor="middle" font-size="11" fill="#5a5a5a">' + (i + 1) + '</text>';
    }
    var ug = y(1 / 48), ue = L + Math.min(48, n) * pw;
    svg += '<line x1="' + L + '" x2="' + ue + '" y1="' + ug + '" y2="' + ug + '" stroke="#5a5a5a" stroke-width="2" stroke-dasharray="5 4"/>';
    svg += '<text x="' + (W - R) + '" y="' + (H - 2) + '" text-anchor="end" font-size="11" fill="#5a5a5a">lead (' + unit + 's ahead)</text></svg>';
    var box = document.getElementById('hw-chart'); box.innerHTML = svg + '<div class="hw-tip"></div>';
    var tip = box.querySelector('.hw-tip');
    box.querySelectorAll('.hwh').forEach(function (el) {
      el.addEventListener('mousemove', function (e) { var i = +el.getAttribute('data-i'), rb = box.getBoundingClientRect();
        tip.style.display = 'block'; tip.textContent = 'lead ' + (i + 1) + ': ' + pct(w[i]);
        tip.style.left = (e.clientX - rb.left + 12) + 'px'; tip.style.top = (e.clientY - rb.top - 28) + 'px'; });
      el.addEventListener('mouseleave', function () { tip.style.display = 'none'; });
    });
  }
  sel.addEventListener('change', function () { load(+sel.value); });
  [ci, si, ii].forEach(function (el) { el.addEventListener('input', draw); });
  sel.value = 5; load(5);
})();
"""
