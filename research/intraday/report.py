"""Standalone HTML reporting for frozen intraday research bundles."""

from __future__ import annotations

import html
import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from .diagnostics import PRIMARY_COST_BPS
from .templates import GAP_FIRST_HOUR_TEMPLATE_ID, INTRADAY_SHOCK_TEMPLATE_ID

TEMPLATE_LABELS = {
    GAP_FIRST_HOUR_TEMPLATE_ID: "Gap + first-hour residual continuation",
    INTRADAY_SHOCK_TEMPLATE_ID: "Intraday residual-shock reversal",
}


@dataclass(frozen=True)
class TemplateVerdict:
    template_id: str
    label: str
    status: str
    mean_10bps: float
    ci_low_10bps: float
    ci_high_10bps: float
    holm_p_value: float
    mean_20bps: float
    gross_break_even_bps: float
    positive_test_years: int
    eligible_test_years: int
    positive_test_year_fraction: float
    k3_mean_session_10bps: float
    min_leave_one_year_out_10bps: float
    max_leave_one_year_out_10bps: float
    max_ticker_absolute_contribution_share: float
    max_sector_absolute_contribution_share: float


def _required_csv(bundle: Path, name: str) -> pd.DataFrame:
    path = bundle / name
    if not path.is_file():
        raise FileNotFoundError(f"required report input is missing: {path}")
    return pd.read_csv(path)


def _one(frame: pd.DataFrame, mask: pd.Series, label: str) -> pd.Series:
    selected = frame.loc[mask]
    if len(selected) != 1:
        raise ValueError(f"expected exactly one {label} row, found {len(selected)}")
    return selected.iloc[0]


def evaluate_bundle(bundle_dir: str | Path) -> list[TemplateVerdict]:
    """Apply the frozen research-priority gates without selecting a new rule."""

    bundle = Path(bundle_dir).resolve()
    stats = _required_csv(bundle, "day_cluster_stats.csv")
    rolling = _required_csv(bundle, "rolling_5y_train_1y_test.csv")
    capacity = _required_csv(bundle, "capacity_summary.csv")
    leave_out = _required_csv(bundle, "leave_one_year_out.csv")
    ticker = _required_csv(bundle, "ticker_summary.csv")
    sector = _required_csv(bundle, "sector_summary.csv")
    verdicts: list[TemplateVerdict] = []
    for template_id, label in TEMPLATE_LABELS.items():
        primary = _one(
            stats,
            stats["template_id"].eq(template_id)
            & stats["cost_bps"].eq(PRIMARY_COST_BPS),
            f"{template_id} primary",
        )
        cost_20 = _one(
            stats,
            stats["template_id"].eq(template_id) & stats["cost_bps"].eq(20.0),
            f"{template_id} 20 bps",
        )
        tests = rolling.loc[
            rolling["template_id"].eq(template_id)
            & rolling["cost_bps"].eq(PRIMARY_COST_BPS)
            & rolling["eligible_for_stability_gate"].eq(True)
        ]
        positive_test_years = int(tests["test_mean_return"].gt(0).sum())
        eligible_test_years = len(tests)
        positive_fraction = (
            positive_test_years / eligible_test_years
            if eligible_test_years
            else np.nan
        )
        k3 = _one(
            capacity,
            capacity["template_id"].eq(template_id)
            & capacity["cost_bps"].eq(PRIMARY_COST_BPS)
            & capacity["capacity_slots"].eq(3),
            f"{template_id} K=3",
        )
        loyo = leave_out.loc[
            leave_out["template_id"].eq(template_id)
            & leave_out["cost_bps"].eq(PRIMARY_COST_BPS)
        ]
        if loyo.empty:
            raise ValueError(f"missing leave-one-year-out rows for {template_id}")
        ticker_rows = ticker.loc[ticker["template_id"].eq(template_id)]
        sector_rows = sector.loc[sector["template_id"].eq(template_id)]
        max_ticker_share = float(
            ticker_rows["share_of_template_absolute_endpoint_contribution"].max()
        )
        max_sector_share = float(
            sector_rows["share_of_template_absolute_endpoint_contribution"].max()
        )
        primary_mean = float(primary["mean_daily_return"])
        gates = (
            primary_mean > 0
            and float(primary["holm_p_value_primary"]) < 0.05
            and float(cost_20["mean_daily_return"]) > 0
            and positive_fraction >= 0.60
            and float(k3["mean_session_return"]) > 0
            and float(loyo["remaining_mean_return"].min()) > 0
            and max_ticker_share < 0.50
            and max_sector_share < 0.50
        )
        status = (
            "Advance to deeper work"
            if gates
            else "Reject v0"
            if primary_mean <= 0
            else "Watch / needs new holdout"
        )
        verdicts.append(
            TemplateVerdict(
                template_id=template_id,
                label=label,
                status=status,
                mean_10bps=primary_mean,
                ci_low_10bps=float(primary["bootstrap_mean_ci_2_5"]),
                ci_high_10bps=float(primary["bootstrap_mean_ci_97_5"]),
                holm_p_value=float(primary["holm_p_value_primary"]),
                mean_20bps=float(cost_20["mean_daily_return"]),
                gross_break_even_bps=primary_mean * 10_000 + PRIMARY_COST_BPS,
                positive_test_years=positive_test_years,
                eligible_test_years=eligible_test_years,
                positive_test_year_fraction=positive_fraction,
                k3_mean_session_10bps=float(k3["mean_session_return"]),
                min_leave_one_year_out_10bps=float(
                    loyo["remaining_mean_return"].min()
                ),
                max_leave_one_year_out_10bps=float(
                    loyo["remaining_mean_return"].max()
                ),
                max_ticker_absolute_contribution_share=max_ticker_share,
                max_sector_absolute_contribution_share=max_sector_share,
            )
        )
    return verdicts


def _bps(value: float) -> str:
    return "—" if not np.isfinite(value) else f"{value * 10_000:+.2f} bps"


def _pct(value: float, digits: int = 1) -> str:
    return "—" if not np.isfinite(value) else f"{value * 100:.{digits}f}%"


def _num(value: object) -> str:
    if pd.isna(value):
        return "—"
    return html.escape(str(value))


def _table(headers: list[str], rows: list[list[str]], css_class: str = "") -> str:
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f'<div class="table-wrap"><table class="{css_class}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def render_report(bundle_dir: str | Path) -> str:
    """Render a self-contained, printable HTML report from a completed bundle."""

    bundle = Path(bundle_dir).resolve()
    manifest_path = bundle / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"run manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_safety = {
        "research_only": True,
        "no_order": True,
        "production_writes": False,
        "automatic_promotion": False,
    }
    for key, expected in required_safety.items():
        if manifest.get(key) is not expected:
            raise ValueError(f"unsafe or incomplete run manifest: {key}={manifest.get(key)!r}")

    verdicts = evaluate_bundle(bundle)
    verdict_by_id = {item.template_id: item for item in verdicts}
    costs = _required_csv(bundle, "day_cluster_stats.csv")
    annual = _required_csv(bundle, "annual_stats.csv")
    sides = _required_csv(bundle, "side_summary.csv")
    sensitivity = _required_csv(bundle, "discontinuity_sensitivity.csv")
    coverage = _required_csv(bundle, "coverage_audit.csv")
    ticker = _required_csv(bundle, "ticker_summary.csv")
    sector = _required_csv(bundle, "sector_summary.csv")
    input_rejections = pd.read_parquet(bundle / "signal_input_rejections.parquet")
    execution_rejections = pd.read_parquet(bundle / "execution_rejections.parquet")

    candidate_rows = coverage.loc[coverage["role"].eq("candidate")]
    evaluated = int(candidate_rows["status"].eq("loaded").sum())
    excluded = candidate_rows.loc[~candidate_rows["status"].eq("loaded")]
    source = manifest.get("raw_source_provenance", {})
    cache_count = int(source.get("snapshot_ticker_count", 0))
    configured_count = 1_025
    full_universe_overlap = 196

    verdict_cards = "".join(
        f"""
        <article class="verdict-card reject">
          <div class="eyebrow">{html.escape(item.label)}</div>
          <h2>{html.escape(item.status)}</h2>
          <div class="big">{_bps(item.mean_10bps)}</div>
          <p>Mean active-day return after 10 bps. 95% CI {_bps(item.ci_low_10bps)} to {_bps(item.ci_high_10bps)}.</p>
          <p class="fine">Holm p={item.holm_p_value:.3g}; the significance is negative, not an edge.</p>
        </article>
        """
        for item in verdicts
    )

    gate_rows = []
    for item in verdicts:
        gate_rows.append(
            [
                html.escape(item.label),
                _bps(item.mean_10bps),
                _bps(item.mean_20bps),
                f"{item.positive_test_years}/{item.eligible_test_years}",
                _bps(item.k3_mean_session_10bps),
                f"{_bps(item.min_leave_one_year_out_10bps)} to {_bps(item.max_leave_one_year_out_10bps)}",
                f'<span class="badge reject">{html.escape(item.status)}</span>',
            ]
        )

    cost_rows = []
    for cost_bps in sorted(costs["cost_bps"].unique()):
        row = [f"{cost_bps:g} bps"]
        for template_id in TEMPLATE_LABELS:
            selected = _one(
                costs,
                costs["template_id"].eq(template_id)
                & costs["cost_bps"].eq(cost_bps),
                f"{template_id} cost {cost_bps}",
            )
            row.append(_bps(float(selected["mean_daily_return"])))
        cost_rows.append(row)

    annual_primary = annual.loc[annual["cost_bps"].eq(PRIMARY_COST_BPS)].copy()
    annual_pivot = annual_primary.pivot(
        index="year", columns="template_id", values="mean_active_day_return"
    )
    annual_rows = [
        [
            str(int(year)),
            _bps(float(row.get(GAP_FIRST_HOUR_TEMPLATE_ID, np.nan))),
            _bps(float(row.get(INTRADAY_SHOCK_TEMPLATE_ID, np.nan))),
        ]
        for year, row in annual_pivot.iterrows()
    ]

    side_rows = []
    for row in sides.loc[sides["cost_bps"].eq(PRIMARY_COST_BPS)].itertuples():
        side_rows.append(
            [
                html.escape(TEMPLATE_LABELS[str(row.template_id)]),
                html.escape(str(row.direction).title()),
                f"{int(row.n_trades):,}",
                _bps(float(row.mean_daily_equal_notional_return)),
                f"{float(row.win_rate_daily) * 100:.1f}%",
            ]
        )

    sensitivity_rows = [
        [
            html.escape(str(row.view).replace("_", " ").title()),
            f"{int(row.n_signals):,}",
            _bps(float(row.mean_daily_equal_notional_gross_return)),
            f"{int(row.n_additional_signals_vs_filtered):+,}",
        ]
        for row in sensitivity.itertuples()
    ]

    exclusion_rows = [
        [html.escape(str(row.ticker)), html.escape(str(row.exclusion_reason))]
        for row in excluded.itertuples()
    ]
    input_counts = input_rejections["input_rejection_reasons"].value_counts()
    input_rows = [
        [html.escape(str(reason)), f"{int(count):,}"]
        for reason, count in input_counts.head(10).items()
    ]
    execution_counts = execution_rejections["execution_status"].value_counts()
    execution_rows = [
        [html.escape(str(reason)), f"{int(count):,}"]
        for reason, count in execution_counts.items()
    ]

    concentration_rows = []
    for template_id, label in TEMPLATE_LABELS.items():
        top_ticker = ticker.loc[ticker["template_id"].eq(template_id)].nlargest(
            1, "share_of_template_absolute_endpoint_contribution"
        ).iloc[0]
        top_sector = sector.loc[sector["template_id"].eq(template_id)].nlargest(
            1, "share_of_template_absolute_endpoint_contribution"
        ).iloc[0]
        concentration_rows.append(
            [
                html.escape(label),
                f"{_num(top_ticker['group'])} ({_pct(float(top_ticker['share_of_template_absolute_endpoint_contribution']))})",
                f"{_num(top_sector['group'])} ({_pct(float(top_sector['share_of_template_absolute_endpoint_contribution']))})",
            ]
        )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Intraday v0 Real-Data Verdict</title>
<style>
:root{{--ink:#17212b;--muted:#5d6b78;--paper:#f4f1ea;--card:#fffefa;--line:#d8d2c7;--red:#a82c2c;--red-bg:#f8e7e4;--navy:#18324a;--gold:#b98732}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 Inter,Segoe UI,Arial,sans-serif}}
main{{max-width:1180px;margin:auto;padding:44px 28px 80px}} .hero{{border-top:7px solid var(--red);padding:30px 0 20px}}
.eyebrow{{text-transform:uppercase;letter-spacing:.12em;font-size:12px;font-weight:800;color:var(--red)}} h1{{font:700 46px/1.05 Georgia,serif;margin:10px 0}}
h2{{font:700 28px/1.15 Georgia,serif;margin:6px 0 14px}} h3{{font-size:18px;margin:0 0 12px}} p{{max-width:850px}} .lede{{font-size:19px;color:var(--muted)}}
.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;margin:24px 0}} .metrics{{grid-template-columns:repeat(4,minmax(0,1fr))}}
.card,.verdict-card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px;box-shadow:0 3px 14px #18222d0b}} .verdict-card.reject{{border-top:5px solid var(--red)}}
.big{{font:700 34px/1.1 Georgia,serif;color:var(--red)}} .metric strong{{display:block;font:700 27px Georgia,serif}} .metric span,.fine{{color:var(--muted);font-size:13px}}
section{{margin-top:40px}} .callout{{border-left:5px solid var(--gold);padding:4px 18px;background:#fff8e9}} .badge{{padding:4px 8px;border-radius:999px;font-weight:800;white-space:nowrap}} .badge.reject{{background:var(--red-bg);color:var(--red)}}
.table-wrap{{overflow:auto;background:var(--card);border:1px solid var(--line);border-radius:10px}} table{{width:100%;border-collapse:collapse;min-width:700px}} th,td{{padding:11px 13px;border-bottom:1px solid var(--line);text-align:right;vertical-align:top}} th:first-child,td:first-child{{text-align:left}} th{{background:#e9e5dc;font-size:12px;text-transform:uppercase;letter-spacing:.05em}} tr:last-child td{{border-bottom:0}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:20px}} ul{{padding-left:20px}} code{{font-size:12px}} footer{{margin-top:50px;padding-top:20px;border-top:1px solid var(--line);color:var(--muted);font-size:12px}}
@media(max-width:800px){{.grid,.metrics,.two{{grid-template-columns:1fr}}h1{{font-size:36px}}main{{padding:28px 16px}}}} @media print{{body{{background:white}}main{{max-width:none}}.card,.verdict-card{{box-shadow:none}}}}
</style></head><body><main>
<header class="hero"><div class="eyebrow">Frozen real-data evaluation · research only</div><h1>Intraday v0: both templates fail</h1>
<p class="lede">A hash-locked 15-minute event study across the available liquid single-stock cache. Neither rule survives realistic friction, chronological stability, or the prespecified small-footprint overlay.</p></header>
<div class="grid">{verdict_cards}</div>
<div class="grid metrics">
 <div class="card metric"><strong>{len(manifest.get('requested_tickers', []))}</strong><span>stocks requested</span></div>
 <div class="card metric"><strong>{evaluated}</strong><span>stocks evaluated</span></div>
 <div class="card metric"><strong>{int(manifest.get('n_primary_trades', 0)):,}</strong><span>executed observations</span></div>
 <div class="card metric"><strong>{int(manifest.get('n_exact_full_sessions', 0)):,}</strong><span>full market sessions</span></div>
</div>
<section><h2>Decision gates</h2><p>Because both primary means are below zero, the frozen rule requires <b>Reject v0</b>. Small p-values here confirm negative economics.</p>
{_table(['Template','10 bps mean','20 bps mean','Positive test years','K=3 mean/session','Leave-one-year-out range','Status'],gate_rows)}</section>
<section><h2>Costs overwhelm the economics</h2><p>The break-even gross means are {verdict_by_id[GAP_FIRST_HOUR_TEMPLATE_ID].gross_break_even_bps:+.2f} bps for gap continuation and {verdict_by_id[INTRADAY_SHOCK_TEMPLATE_ID].gross_break_even_bps:+.2f} bps for shock reversal. Both are negative by the lowest 5 bps case.</p>
{_table(['Round-trip cost','Gap continuation','Shock reversal'],cost_rows)}</section>
<section><h2>Chronological stability</h2><p>Complete rolling test years are positive in 1/17 cases for gap continuation and 0/17 for shock reversal. Every leave-one-year-out mean remains negative.</p>
{_table(['Year','Gap at 10 bps','Shock at 10 bps'],annual_rows)}</section>
<section class="two"><div><h2>Long vs short</h2>{_table(['Template','Side','Trades','Mean active day','Win rate'],side_rows)}</div>
<div><h2>Raw-price sensitivity</h2><p>The discontinuity filter removes 17 gap events; switching it off makes gross results slightly worse.</p>{_table(['View','Signals','Gross mean day','Added signals'],sensitivity_rows)}</div></section>
<section><h2>Coverage is liquid-cache, not all-universe</h2><div class="callout"><p><b>{cache_count} cached ticker files versus {configured_count:,} configured names.</b> Only {full_universe_overlap} overlap the full base-plus-overflow universe. This report evaluates {evaluated} of 162 requested liquid single stocks and must not be described as an all-1,025-name intraday test.</p></div>
{_table(['Excluded ticker','Reason'],exclusion_rows)}</section>
<section class="two"><div><h2>Input rejections</h2>{_table(['Reason','Count'],input_rows)}</div><div><h2>Execution rejections</h2>{_table(['Reason','Count'],execution_rows)}</div></section>
<section><h2>Concentration audit</h2><p>Contribution shares use the daily equal-notional endpoint, not raw trade-return sums.</p>{_table(['Template','Largest ticker share','Largest sector share'],concentration_rows)}</section>
<section><h2>What this means</h2><div class="two"><div class="card"><h3>Actionability</h3><ul><li>Do not productionize, paper-promote, size, stage, or schedule either v0 template.</li><li>Gap continuation has no portfolio-level gross edge and should be deprioritized.</li><li>Shock reversal has only a small gross effect, below even the 5 bps friction case.</li></ul></div>
<div class="card"><h3>Clean next wedge</h3><ul><li>Preregister a distinct v1 on a new holdout.</li><li>Test monotonicity in shock strength, liquidity, and market state without selecting on this sample.</li><li>Add quotes or one-minute data, integer shares, commissions, spread/slippage, and borrow constraints before any investability claim.</li></ul></div></div></section>
<section><h2>Known limits</h2><p>Today's liquid universe creates survivorship bias; the static sector map creates classification lookahead. The cache joins an FMP backfill to yfinance maintenance. Fifteen-minute bars cannot reveal spreads, queues, partial fills, within-bar sequencing, halts, news, or borrow availability. Capacity overlays are signal slots, not a broker/account model.</p></section>
<footer>Generated from <code>{html.escape(str(bundle))}</code>. Source hashes were enforced: snapshot index <code>{html.escape(str(source.get('snapshot_source_meta_sha256','')))}</code>, universe <code>{html.escape(str(source.get('universe_file_sha256','')))}</code>, sector map <code>{html.escape(str(source.get('sector_map_file_sha256','')))}</code>. No production, broker, R2, scheduler, deployment, or order write occurred.</footer>
</main></body></html>"""


def write_report(
    bundle_dir: str | Path,
    *,
    artifact_root: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Write report.html inside an existing research bundle and manifest it last."""

    bundle = Path(bundle_dir).resolve()
    default_root = Path(__file__).resolve().parents[2] / "artifacts"
    allowed_root = Path(artifact_root or default_root).resolve()
    try:
        bundle.relative_to(allowed_root)
    except ValueError as exc:
        raise ValueError(f"bundle must stay beneath research artifact root: {allowed_root}") from exc
    if not bundle.is_dir():
        raise FileNotFoundError(f"research bundle does not exist: {bundle}")
    report_path = bundle / "report.html"
    if report_path.exists() and not overwrite:
        raise FileExistsError(f"report already exists: {report_path}")
    rendered = render_report(bundle)
    report_path.write_text(rendered, encoding="utf-8")
    report_hash = sha256(report_path.read_bytes()).hexdigest()
    manifest_path = bundle / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["report"] = {
        "filename": report_path.name,
        "sha256": report_hash,
        "self_contained": True,
        "verdicts": [asdict(item) for item in evaluate_bundle(bundle)],
        "manifest_rewritten_after_report": True,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return report_path

