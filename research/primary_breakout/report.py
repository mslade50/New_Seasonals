"""Standalone HTML report for the primary-breakout research run."""

from __future__ import annotations

from collections.abc import Mapping
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from .engine import BacktestResult


def _pct(value: object, digits: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(number):
        return "—"
    return f"{number * 100:.{digits}f}%"


def _num(value: object, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(number):
        return "—"
    return f"{number:.{digits}f}"


def _money(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(number):
        return "—"
    return f"${number:,.0f}"


def _line_svg(
    frame: pd.DataFrame,
    columns: Mapping[str, str],
    width: int = 960,
    height: int = 310,
) -> str:
    data = frame[list(columns)].dropna(how="all")
    if data.empty:
        return "<p class='muted'>No chart data.</p>"
    normalized = data / data.iloc[0] * 100.0
    values = normalized.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    y_min = max(float(np.nanmin(finite)) * 0.92, 1e-6)
    y_max = float(np.nanmax(finite)) * 1.04
    log_min, log_max = np.log(y_min), np.log(max(y_max, y_min * 1.01))
    left, right, top, bottom = 56, 16, 18, 38
    plot_w, plot_h = width - left - right, height - top - bottom

    def xy(index: int, value: float) -> tuple[float, float]:
        x = left + plot_w * index / max(len(normalized) - 1, 1)
        y = top + plot_h * (log_max - np.log(max(value, 1e-9))) / (log_max - log_min)
        return x, y

    parts = [
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='Growth of one dollar'>",
        f"<rect x='0' y='0' width='{width}' height='{height}' rx='12' fill='#fbfcfe'/>",
    ]
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = top + plot_h * frac
        level = np.exp(log_max - (log_max - log_min) * frac)
        parts.append(f"<line x1='{left}' y1='{y:.1f}' x2='{width-right}' y2='{y:.1f}' stroke='#e5e9f0'/>")
        parts.append(f"<text x='{left-8}' y='{y+4:.1f}' text-anchor='end' font-size='11' fill='#697586'>{level:.0f}</text>")
    colors = ["#1455d9", "#9b5de5", "#d97706"]
    for color, column in zip(colors, columns):
        series = normalized[column]
        points = []
        for idx, value in enumerate(series):
            if np.isfinite(value) and value > 0:
                x, y = xy(idx, float(value))
                points.append(f"{x:.1f},{y:.1f}")
        if points:
            parts.append(
                f"<polyline points='{' '.join(points)}' fill='none' stroke='{color}' stroke-width='2.2' stroke-linejoin='round'/>"
            )
    legend_x = left
    for color, label in zip(colors, columns.values()):
        parts.append(f"<rect x='{legend_x}' y='{height-23}' width='14' height='3' fill='{color}'/>")
        parts.append(f"<text x='{legend_x+20}' y='{height-18}' font-size='12' fill='#344054'>{escape(label)}</text>")
        legend_x += 165
    parts.append(
        f"<text x='{left}' y='{height-4}' font-size='10' fill='#697586'>{normalized.index[0].date()}</text>"
    )
    parts.append(
        f"<text x='{width-right}' y='{height-4}' text-anchor='end' font-size='10' fill='#697586'>{normalized.index[-1].date()}</text>"
    )
    parts.append("</svg>")
    return "".join(parts)


def _drawdown_svg(frame: pd.DataFrame, width: int = 960, height: int = 230) -> str:
    data = frame[["Drawdown", "BenchmarkDrawdown", "EqualWeightDrawdown"]]
    y_min = min(float(data.min().min()), -0.01)
    left, right, top, bottom = 56, 16, 16, 34
    plot_w, plot_h = width - left - right, height - top - bottom

    def xy(index: int, value: float) -> tuple[float, float]:
        x = left + plot_w * index / max(len(data) - 1, 1)
        y = top + plot_h * (0.0 - value) / (0.0 - y_min)
        return x, y

    parts = [f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='Drawdown comparison'>"]
    parts.append(f"<rect width='{width}' height='{height}' rx='12' fill='#fbfcfe'/>")
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        value = y_min * frac
        y = top + plot_h * frac
        parts.append(f"<line x1='{left}' y1='{y:.1f}' x2='{width-right}' y2='{y:.1f}' stroke='#e5e9f0'/>")
        parts.append(f"<text x='{left-8}' y='{y+4:.1f}' text-anchor='end' font-size='11' fill='#697586'>{value*100:.0f}%</text>")
    colors = ["#1455d9", "#9b5de5", "#d97706"]
    for color, column in zip(colors, data.columns):
        points = []
        for idx, value in enumerate(data[column]):
            if np.isfinite(value):
                x, y = xy(idx, float(value))
                points.append(f"{x:.1f},{y:.1f}")
        parts.append(f"<polyline points='{' '.join(points)}' fill='none' stroke='{color}' stroke-width='1.8'/>")
    parts.append("</svg>")
    return "".join(parts)


def _table(headers: list[str], rows: list[list[str]], classes: str = "") -> str:
    head = "".join(f"<th>{escape(str(header))}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f"<div class='table-wrap'><table class='{classes}'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def render_report(
    result: BacktestResult,
    robustness: pd.DataFrame,
    periods: pd.DataFrame,
    bootstrap: Mapping[str, float],
    gates: Mapping[str, bool],
    excluded_instruments: list[str],
    universe_hash: str,
    data_hash: str,
    output_path: str | Path,
) -> None:
    """Write the human-first standalone HTML report."""

    metrics = result.metrics
    passed = bool(gates) and all(gates.values())
    verdict = (
        "ADVANCE TO RAW / POINT-IN-TIME VALIDATION"
        if passed
        else "DO NOT ADVANCE YET — EXPLORATORY GATES FAILED"
    )
    verdict_class = "pass" if passed else "fail"

    comparison_rows = []
    for label, prefix in (("Breakout strategy", ""), ("SPY", "spy_"), ("Equal-weight primary", "ew_")):
        comparison_rows.append(
            [
                escape(label),
                _pct(metrics.get(f"{prefix}cagr")),
                _num(metrics.get(f"{prefix}sharpe")),
                _pct(metrics.get(f"{prefix}ann_vol")),
                _pct(metrics.get(f"{prefix}max_drawdown")),
                _num(metrics.get(f"{prefix}calmar")),
            ]
        )

    period_rows = []
    for row in periods.to_dict("records"):
        period_rows.append(
            [
                escape(str(row["Period"])),
                _pct(row.get("cagr")),
                _num(row.get("sharpe")),
                _pct(row.get("max_drawdown")),
                _pct(row.get("spy_cagr")),
                _pct(row.get("ew_cagr")),
            ]
        )

    robustness_rows = []
    for row in robustness.to_dict("records"):
        primary = bool(row.get("Primary", False))
        robustness_rows.append(
            [
                f"<strong>{escape(str(row['Variant']))}</strong>" if primary else escape(str(row["Variant"])),
                _pct(row.get("CAGR")),
                _num(row.get("Sharpe")),
                _pct(row.get("MaxDD")),
                f"{int(row.get('Trades', 0)):,}",
                _pct(row.get("AverageGross")),
            ]
        )

    yearly_rows = [
        [
            str(int(row.Year)),
            _pct(row.Strategy),
            _pct(row.SPY),
            _pct(row.EqualWeightPrimary),
        ]
        for row in result.yearly_returns.itertuples(index=False)
    ]

    trades = result.trades
    if trades.empty:
        contributor_rows: list[list[str]] = []
        exit_rows: list[list[str]] = []
    else:
        contributors = (
            trades.groupby("Ticker")
            .agg(PnL=("PnL", "sum"), Trades=("Ticker", "size"), AvgR=("R", "mean"))
            .sort_values("PnL", ascending=False)
        )
        display_contrib = pd.concat([contributors.head(8), contributors.tail(5)]).drop_duplicates()
        contributor_rows = [
            [escape(str(idx)), _money(row.PnL), f"{int(row.Trades):,}", _num(row.AvgR)]
            for idx, row in display_contrib.iterrows()
        ]
        exits = trades.groupby("ExitReason").agg(Trades=("Ticker", "size"), PnL=("PnL", "sum"), AvgR=("R", "mean"))
        exit_rows = [
            [escape(str(idx)), f"{int(row.Trades):,}", _money(row.PnL), _num(row.AvgR)]
            for idx, row in exits.sort_values("Trades", ascending=False).iterrows()
        ]

    candidate_summary = (
        result.candidates.groupby(["Status", "Reason"]).size().sort_values(ascending=False)
        if not result.candidates.empty
        else pd.Series(dtype=int)
    )
    candidate_rows = [
        [escape(str(status)), escape(str(reason)), f"{int(count):,}"]
        for (status, reason), count in candidate_summary.items()
    ]
    gate_rows = [
        [escape(name.replace("_", " ").title()), "<span class='gate pass'>PASS</span>" if value else "<span class='gate fail'>FAIL</span>"]
        for name, value in gates.items()
    ]
    readout_items = [
        (
            f"The base case compounded at <strong>{_pct(metrics.get('cagr'))}</strong> "
            f"with a <strong>{_num(metrics.get('sharpe'))}</strong> Sharpe, versus "
            f"{_pct(metrics.get('spy_cagr'))} for SPY and "
            f"{_pct(metrics.get('ew_cagr'))} for the same-list equal-weight comparator."
        )
    ]
    weak_periods = periods[(periods["cagr"] <= 0) | (periods["sharpe"] <= 0)]
    if not weak_periods.empty:
        labels = ", ".join(escape(str(value)) for value in weak_periods["Period"])
        readout_items.append(
            f"The non-positive regime gate was triggered by: <strong>{labels}</strong>."
        )
    weak_neighbors = robustness[
        ~robustness["Primary"].astype(bool)
        & robustness["Variant"].str.startswith(("Breakout:", "CK multiplier:"))
        & ((robustness["CAGR"] <= 0) | (robustness["Sharpe"] <= 0))
    ]
    if not weak_neighbors.empty:
        labels = ", ".join(escape(str(value)) for value in weak_neighbors["Variant"])
        readout_items.append(
            "Nearby parameter behavior was not uniformly positive; the failing neighbor was "
            f"<strong>{labels}</strong>."
        )
    cost_20 = (
        robustness.loc[robustness["CostBps"] == 20.0]
        if "CostBps" in robustness
        else pd.DataFrame()
    )
    if not cost_20.empty:
        row = cost_20.iloc[0]
        readout_items.append(
            f"At 20 bps per side, the result was still positive but thin: "
            f"{_pct(row['CAGR'])} CAGR and {_num(row['Sharpe'])} Sharpe."
        )
    readout_html = "<ul>" + "".join(f"<li>{item}</li>" for item in readout_items) + "</ul>"

    css = """
    :root{--ink:#162033;--muted:#667085;--line:#e5e9f0;--blue:#1455d9;--bg:#f3f6fa;--panel:#fff;--green:#067647;--red:#b42318;--amber:#b54708}
    *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 Inter,Segoe UI,Arial,sans-serif}
    main{max-width:1180px;margin:0 auto;padding:34px 24px 64px}.eyebrow{text-transform:uppercase;letter-spacing:.12em;color:var(--blue);font-weight:800;font-size:12px}
    h1{font-size:38px;line-height:1.12;margin:8px 0 10px;letter-spacing:-.03em}h2{font-size:23px;margin:0 0 14px}h3{font-size:16px;margin:0 0 8px}.lede{font-size:17px;color:#475467;max-width:900px}
    .verdict{margin:24px 0;padding:18px 20px;border-radius:12px;font-weight:800;letter-spacing:.02em}.verdict.pass{background:#ecfdf3;color:var(--green);border:1px solid #abefc6}.verdict.fail{background:#fef3f2;color:var(--red);border:1px solid #fecdca}
    .grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:20px 0}.card,.panel{background:var(--panel);border:1px solid var(--line);border-radius:14px;box-shadow:0 1px 2px rgba(16,24,40,.04)}.card{padding:17px}.card .label{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.06em}.card .value{font-size:26px;font-weight:800;margin-top:4px}.card .sub{font-size:12px;color:var(--muted)}
    .panel{padding:22px;margin:18px 0}.two{display:grid;grid-template-columns:1fr 1fr;gap:18px}.muted{color:var(--muted)}.warning{background:#fffaeb;border-left:4px solid #f79009;padding:13px 15px;border-radius:8px;color:#7a2e0e}
    .table-wrap{overflow:auto;border:1px solid var(--line);border-radius:10px}table{width:100%;border-collapse:collapse;background:#fff}th,td{text-align:right;padding:10px 12px;border-bottom:1px solid var(--line);white-space:nowrap}th{background:#f8fafc;color:#475467;font-size:12px;text-transform:uppercase;letter-spacing:.04em}th:first-child,td:first-child{text-align:left}tr:last-child td{border-bottom:0}.gate{font-weight:800}.gate.pass{color:var(--green)}.gate.fail{color:var(--red)}
    code{background:#f2f4f7;padding:2px 5px;border-radius:4px}ul{padding-left:20px}.source{font-size:12px;color:var(--muted);word-break:break-all}.pill{display:inline-block;padding:4px 8px;border-radius:999px;background:#eef4ff;color:#1849a9;font-size:12px;font-weight:700;margin:2px}
    @media(max-width:800px){.grid{grid-template-columns:1fr 1fr}.two{grid-template-columns:1fr}h1{font-size:30px}}@media(max-width:480px){.grid{grid-template-columns:1fr}main{padding:24px 14px}}
    """

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Primary-Universe 100-Day Breakout Research</title><style>{css}</style></head><body><main>
<div class="eyebrow">Research-only systematic equity study · as of {escape(str(metrics['end']))}</div>
<h1>100-day breakout + Chande–Kroll stop</h1>
<p class="lede">A causal, next-open portfolio simulation across the 162 stock-like names in the current primary universe. Candidates are fresh close breakouts, admitted in descending raw 100-session ROC order and sized to a maximum 0.5% of prior-close equity.</p>
<div class="verdict {verdict_class}">{verdict}</div>
<div class="grid">
  <div class="card"><div class="label">Net CAGR</div><div class="value">{_pct(metrics['cagr'])}</div><div class="sub">SPY {_pct(metrics['spy_cagr'])}</div></div>
  <div class="card"><div class="label">Sharpe</div><div class="value">{_num(metrics['sharpe'])}</div><div class="sub">Bootstrap 90%: {_num(bootstrap.get('sharpe_p05'))} to {_num(bootstrap.get('sharpe_p95'))}</div></div>
  <div class="card"><div class="label">Max drawdown</div><div class="value">{_pct(metrics['max_drawdown'])}</div><div class="sub">Equal-weight primary {_pct(metrics['ew_max_drawdown'])}</div></div>
  <div class="card"><div class="label">Trades</div><div class="value">{int(metrics['trades']):,}</div><div class="sub">Avg {metrics['average_positions']:.1f} positions</div></div>
</div>
<div class="warning"><strong>Evidence posture:</strong> exploratory only. The universe is today’s curated primary list applied backward, and execution uses the rolling-vintage adjusted cache. That creates survivorship/selection bias and is not exact evidence for frozen overnight dollar stops.</div>

<section class="panel"><h2>First read</h2>
{_table(['Portfolio','CAGR','Sharpe','Volatility','Max DD','Calmar'], comparison_rows)}
{readout_html}
<div style="margin-top:18px">{_line_svg(result.equity, {'Equity':'Breakout strategy','BenchmarkEquity':'SPY','EqualWeightEquity':'Equal-weight primary'})}</div>
</section>
<section class="panel"><h2>Decision gates</h2><p class="muted">A pass advances this idea only to raw-bar and point-in-time-universe validation. It never authorizes production.</p>{_table(['Gate','Result'], gate_rows)}</section>

<section class="panel"><h2>Frozen specification</h2><div class="two"><div>
<ul><li>Universe: 162 stock-like tickers from <code>LIQUID_PLUS_COMMODITIES</code>; 35 ETFs, spot indices, and commodity vehicles excluded.</li>
<li>Signal: <code>Close[t] &gt; max(High[t-100:t-1])</code>; fresh means today true and yesterday false.</li>
<li>Rank: unscaled adjusted-price ROC, <code>Close[t]/Close[t-100]-1</code>, descending; ticker breaks ties.</li>
<li>Entry: next session open + {result.config.cost_bps:g} bps; one position per name.</li></ul></div><div>
<ul><li>Chande–Kroll: Wilder ATR({result.config.ck_atr_period}), {result.config.ck_atr_multiple:g}×, confirmation {result.config.ck_stop_period}; stop ratchets and becomes active next session.</li>
<li>Risk: at most {_pct(result.config.risk_fraction)} of prior-close equity; whole shares.</li>
<li>Admission caps: {_pct(result.config.max_name_fraction)} per name, {_pct(result.config.max_gross_fraction)} gross, {_pct(result.config.max_open_risk_fraction)} aggregate current stop-risk.</li>
<li>No time exit. Final-day liquidation is a reporting convention only.</li></ul></div></div></section>

<section class="panel"><h2>Regime behavior</h2>{_table(['Period','Strategy CAGR','Sharpe','Max DD','SPY CAGR','EW primary CAGR'], period_rows)}</section>
<section class="panel"><h2>Robustness, not optimization</h2><p class="muted">The primary row was frozen before the run. Neighbor rows ask whether the conclusion survives plausible costs and nearby mechanics.</p>{_table(['Variant','CAGR','Sharpe','Max DD','Trades','Avg gross'], robustness_rows)}</section>
<section class="panel"><h2>Drawdown</h2>{_drawdown_svg(result.equity)}</section>

<section class="panel"><h2>Calendar returns</h2>{_table(['Year','Strategy','SPY','Equal-weight primary'], yearly_rows)}</section>
<section class="two"><div class="panel"><h2>Contribution concentration</h2>{_table(['Ticker','PnL','Trades','Average R'], contributor_rows)}</div><div class="panel"><h2>Exit pathways</h2>{_table(['Exit reason','Trades','PnL','Average R'], exit_rows)}</div></section>
<section class="panel"><h2>Capacity and realized risk</h2>
<div class="grid"><div class="card"><div class="label">Average gross</div><div class="value">{_pct(metrics['average_gross'])}</div></div><div class="card"><div class="label">Time invested</div><div class="value">{_pct(metrics['time_in_market'])}</div></div><div class="card"><div class="label">Achieved / target risk</div><div class="value">{_pct(metrics['average_achieved_risk_pct'])}</div></div><div class="card"><div class="label">Annual turnover</div><div class="value">{_pct(metrics['annual_turnover'])}</div></div></div>
{_table(['Candidate status','Reason / binding constraint','Count'], candidate_rows)}</section>

<section class="panel"><h2>Limitations and next evidence</h2><ol>
<li><strong>Static-universe bias:</strong> the current 162-name primary stock list is applied to history. Failed, acquired, and removed names are absent; index membership is not point-in-time.</li>
<li><strong>Execution basis:</strong> the source cache is adjusted OHLCV and only refreshes a trailing window. A frozen overnight stop should ultimately be tested on immutable raw/as-traded bars with explicit corporate-action and dividend handling.</li>
<li><strong>Daily-bar ambiguity:</strong> entries occur at the open, so entry-day lows are eligible for the active stop; stop updates computed after the close apply only from the next session.</li>
<li><strong>Risk is capped, not guaranteed:</strong> name/cash/gross constraints often size below 0.5%, while overnight gaps can lose more than the modeled stop budget.</li>
</ol></section>

<section class="panel"><h2>Source ledger</h2>
<p class="source"><strong>Prices:</strong> {escape(str(result.data_quality.get('prices_path','')))}<br><strong>Price SHA-256:</strong> {escape(data_hash)}<br><strong>Universe SHA-256:</strong> {escape(universe_hash)}<br><strong>Universe coverage:</strong> {len(result.universe)} usable / {result.data_quality.get('requested_universe_count')} requested; calendar {escape(str(result.data_quality.get('calendar_start')))} to {escape(str(result.data_quality.get('calendar_end')))}.</p>
<details><summary>Excluded primary instruments ({len(excluded_instruments)})</summary><p>{' '.join(f'<span class="pill">{escape(t)}</span>' for t in excluded_instruments)}</p></details>
</section>
</main></body></html>"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(html, encoding="utf-8")
    temporary.replace(path)
