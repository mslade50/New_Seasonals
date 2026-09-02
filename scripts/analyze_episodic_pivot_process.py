"""Build a reproducible EP cadence and winner-trait research report.

This is descriptive research over the frozen historical artifacts.  It does
not optimize policy, create a recommendation, or touch any broker surface.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

OUTCOME_20 = "excess_next_open_to_close_20d_pct"
OUTCOME_60 = "excess_next_open_to_close_60d_pct"
TRAIT_FIELDS = (
    "gap_pct",
    "event_rvol_20",
    "prior_atr_pct_14",
    "prior_63d_return_pct",
    "prior_addv_63",
    "event_date_cluster_size",
    "unique_source_clusters",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _require_columns(frame: pd.DataFrame, required: set[str], *, label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"{label} input is missing required columns: {', '.join(missing)}")


def _period_stats(events: pd.DataFrame, *, freq: str) -> dict[str, float | int]:
    if events.empty:
        raise ValueError("cannot summarize an empty cadence cohort")
    periods = events["date"].dt.to_period(freq)
    all_periods = pd.period_range(periods.min(), periods.max(), freq=freq)
    counts = periods.value_counts().sort_index().reindex(all_periods, fill_value=0)
    active = counts[counts > 0]
    return {
        "total_periods": len(counts),
        "active_periods": len(active),
        "mean_all": float(counts.mean()),
        "mean_active": float(active.mean()),
        "active_median": float(active.median()),
        "active_p25": float(active.quantile(0.25)),
        "active_p75": float(active.quantile(0.75)),
        "p90_all": float(counts.quantile(0.90)),
        "zero_rate_pct": float(100.0 * counts.eq(0).mean()),
    }


def cadence_summary(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    windows = (
        ("Full history", events),
        ("2020+", events[events["date"] >= pd.Timestamp("2020-01-01")]),
        (
            "Consumed 2024-2026",
            events[events["date"] >= pd.Timestamp("2024-01-01")],
        ),
    )
    rows: list[dict[str, Any]] = []
    for label, frame in windows:
        for period, freq in (("week", "W-SUN"), ("month", "M"), ("year", "Y")):
            rows.append(
                {
                    "window": label,
                    "period": period,
                    "events": len(frame),
                    "start": frame["date"].min().date().isoformat(),
                    "end": frame["date"].max().date().isoformat(),
                    **_period_stats(frame, freq=freq),
                }
            )
    by_year = (
        events.assign(year=events["date"].dt.year)
        .groupby("year", as_index=False)
        .size()
        .rename(columns={"size": "events"})
    )
    return pd.DataFrame(rows), by_year


def _summary_row(frame: pd.DataFrame, cohort: str, field: str) -> dict[str, Any]:
    values = pd.to_numeric(frame[field], errors="coerce").dropna()
    return {
        "cohort": cohort,
        "field": field,
        "n": len(values),
        "median": float(values.median()),
        "p25": float(values.quantile(0.25)),
        "p75": float(values.quantile(0.75)),
        "mean": float(values.mean()),
    }


def winner_analysis(
    events: pd.DataFrame,
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    eligible_20 = events[events[OUTCOME_20].notna()].copy()
    eligible_60 = events[events[OUTCOME_60].notna()].copy()
    balanced = events[events[[OUTCOME_20, OUTCOME_60]].notna().all(axis=1)].copy()
    threshold_20 = float(eligible_20[OUTCOME_20].quantile(0.90))
    threshold_60 = float(eligible_60[OUTCOME_60].quantile(0.90))
    top_20 = eligible_20[eligible_20[OUTCOME_20] >= threshold_20].copy()
    top_60 = eligible_60[eligible_60[OUTCOME_60] >= threshold_60].copy()
    durable = balanced[
        (balanced[OUTCOME_20] > 0) & (balanced[OUTCOME_60] >= threshold_60)
    ].copy()
    overlap = set(zip(top_20["ticker"], top_20["date"], strict=True)) & set(
        zip(top_60["ticker"], top_60["date"], strict=True)
    )

    cohorts = {
        "Balanced baseline": balanced,
        "Top-decile 20d": top_20,
        "Top-decile 60d": top_60,
        "Durable": durable,
    }
    trait_rows = [
        _summary_row(frame, label, field)
        for label, frame in cohorts.items()
        for field in TRAIT_FIELDS
    ]

    composition_rows: list[dict[str, Any]] = []
    for label, frame in cohorts.items():
        n = len(frame)
        composition_rows.extend(
            [
                {
                    "cohort": label,
                    "trait": "earnings_date_match",
                    "n": n,
                    "rate_pct": float(100.0 * frame["earnings_date_match"].mean()),
                },
                {
                    "cohort": label,
                    "trait": "preopen_sec_earnings_guidance",
                    "n": n,
                    "rate_pct": float(
                        100.0
                        * frame["preopen_sec_event_type"]
                        .eq("EARNINGS_GUIDANCE")
                        .mean()
                    ),
                },
                {
                    "cohort": label,
                    "trait": "no_preopen_sec_evidence",
                    "n": n,
                    "rate_pct": float(
                        100.0
                        * frame["preopen_sec_event_type"]
                        .eq("NO_PREOPEN_SEC_EVIDENCE")
                        .mean()
                    ),
                },
                {
                    "cohort": label,
                    "trait": "secondary_earnings_guidance",
                    "n": n,
                    "rate_pct": float(
                        100.0
                        * frame["secondary_context_event_type"]
                        .eq("EARNINGS_GUIDANCE")
                        .mean()
                    ),
                },
            ]
        )

    atr_rows: list[dict[str, Any]] = []
    for label, start in (
        ("Full history", None),
        ("2020+", pd.Timestamp("2020-01-01")),
        ("Consumed 2024-2026", pd.Timestamp("2024-01-01")),
    ):
        frame = balanced if start is None else balanced[balanced["date"] >= start]
        bins = pd.qcut(
            frame["prior_atr_pct_14"],
            4,
            labels=("Q1", "Q2", "Q3", "Q4"),
            duplicates="drop",
        )
        is_durable = (frame[OUTCOME_20] > 0) & (frame[OUTCOME_60] >= threshold_60)
        for quartile in bins.cat.categories:
            selected = bins.eq(quartile)
            atr_rows.append(
                {
                    "window": label,
                    "atr_quartile": str(quartile),
                    "n": int(selected.sum()),
                    "durable_winners": int((selected & is_durable).sum()),
                    "durable_rate_pct": float(100.0 * is_durable[selected].mean()),
                    "atr_min_pct": float(frame.loc[selected, "prior_atr_pct_14"].min()),
                    "atr_max_pct": float(frame.loc[selected, "prior_atr_pct_14"].max()),
                }
            )

    tail_20 = eligible_20[OUTCOME_20].sort_values()
    tail_60 = eligible_60[OUTCOME_60].sort_values()
    top_events = pd.concat(
        [
            top_20.nlargest(10, OUTCOME_20).assign(top_list="TOP_20D"),
            durable.nlargest(10, OUTCOME_60).assign(top_list="DURABLE_60D"),
        ],
        ignore_index=True,
    )[
        [
            "top_list",
            "ticker",
            "date",
            OUTCOME_20,
            OUTCOME_60,
            "gap_pct",
            "event_rvol_20",
            "prior_atr_pct_14",
            "prior_63d_return_pct",
            "prior_addv_63",
            "event_date_cluster_size",
            "evidence_posture",
            "preopen_sec_event_type",
            "secondary_context_event_type",
        ]
    ]
    facts = {
        "eligible_20_n": len(eligible_20),
        "eligible_60_n": len(eligible_60),
        "balanced_n": len(balanced),
        "top_20_threshold_pct": threshold_20,
        "top_60_threshold_pct": threshold_60,
        "top_20_n": len(top_20),
        "top_60_n": len(top_60),
        "durable_n": len(durable),
        "top_decile_overlap_n": len(overlap),
        "top_decile_overlap_pct_of_top_20": float(100.0 * len(overlap) / len(top_20)),
        "durable_20d_median_pct": float(durable[OUTCOME_20].median()),
        "durable_60d_median_pct": float(durable[OUTCOME_60].median()),
        "mean_20_pct": float(tail_20.mean()),
        "median_20_pct": float(tail_20.median()),
        "mean_20_excluding_top_1pct_pct": float(
            tail_20[tail_20 <= tail_20.quantile(0.99)].mean()
        ),
        "mean_60_pct": float(tail_60.mean()),
        "median_60_pct": float(tail_60.median()),
        "mean_60_excluding_top_1pct_pct": float(
            tail_60[tail_60 <= tail_60.quantile(0.99)].mean()
        ),
    }
    return (
        facts,
        pd.DataFrame(trait_rows),
        pd.DataFrame(composition_rows),
        pd.DataFrame(atr_rows),
        top_events,
    )


def _fmt_number(value: float, digits: int = 1) -> str:
    if not math.isfinite(value):
        return "—"
    return f"{value:,.{digits}f}"


def _html_report(
    *,
    cadence: pd.DataFrame,
    by_year: pd.DataFrame,
    facts: dict[str, Any],
    traits: pd.DataFrame,
    composition: pd.DataFrame,
    atr_quartiles: pd.DataFrame,
    top_events: pd.DataFrame,
    input_hashes: dict[str, str],
) -> str:
    recent = cadence[cadence["window"].eq("Consumed 2024-2026")].set_index(
        "period"
    )
    recent_week = recent.loc["week"]
    recent_month = recent.loc["month"]
    full_years = by_year[by_year["year"].isin([2024, 2025])]
    yearly_label = "–".join(str(value) for value in full_years["events"].tolist())

    cadence_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row.window))}</td>"
        f"<td>{html.escape(str(row.period))}</td>"
        f"<td>{int(row.events):,}</td>"
        f"<td>{row.mean_all:.2f}</td>"
        f"<td>{row.active_median:.0f} [{row.active_p25:.0f}, {row.active_p75:.0f}]</td>"
        f"<td>{row.p90_all:.1f}</td>"
        f"<td>{row.zero_rate_pct:.1f}%</td>"
        "</tr>"
        for row in cadence.itertuples(index=False)
    )
    year_rows = "".join(
        f"<tr><td>{int(row.year)}</td><td>{int(row.events)}</td>"
        f"<td>{'YTD through Aug. 14' if int(row.year) == 2026 else 'Full year'}</td></tr>"
        for row in by_year[by_year["year"] >= 2020].itertuples(index=False)
    )

    field_labels = {
        "gap_pct": "Opening gap",
        "event_rvol_20": "Event-day RVOL (ex post)",
        "prior_atr_pct_14": "Prior ATR(14) / close",
        "prior_63d_return_pct": "Prior 63d return",
        "prior_addv_63": "Prior 63d ADDV",
        "event_date_cluster_size": "Same-date cluster size",
        "unique_source_clusters": "Unique source clusters",
    }
    trait_pivot = traits.pivot(index="field", columns="cohort", values="median")
    trait_rows = ""
    for field in TRAIT_FIELDS:
        suffix = "%" if field in {"gap_pct", "prior_atr_pct_14", "prior_63d_return_pct"} else ""
        scale = 1e-6 if field == "prior_addv_63" else 1.0
        value_suffix = "m" if field == "prior_addv_63" else suffix
        trait_rows += (
            "<tr>"
            f"<td>{html.escape(field_labels[field])}</td>"
            + "".join(
                f"<td>{_fmt_number(float(trait_pivot.loc[field, cohort]) * scale, 2)}{value_suffix}</td>"
                for cohort in (
                    "Balanced baseline",
                    "Top-decile 20d",
                    "Top-decile 60d",
                    "Durable",
                )
            )
            + "</tr>"
        )

    atr_rows = "".join(
        f"<tr><td>{html.escape(str(row.window))}</td><td>{html.escape(str(row.atr_quartile))}</td>"
        f"<td>{int(row.n)}</td><td>{row.atr_min_pct:.2f}%–{row.atr_max_pct:.2f}%</td>"
        f"<td>{row.durable_rate_pct:.2f}%</td></tr>"
        for row in atr_quartiles.itertuples(index=False)
    )

    comp_pivot = composition.pivot(index="trait", columns="cohort", values="rate_pct")
    comp_labels = {
        "earnings_date_match": "Calendar earnings-date match",
        "preopen_sec_earnings_guidance": "Current-CIK pre-open SEC earnings context",
        "no_preopen_sec_evidence": "No pre-open SEC evidence",
        "secondary_earnings_guidance": "Secondary earnings context",
    }
    composition_rows = "".join(
        "<tr>"
        f"<td>{html.escape(comp_labels[field])}</td>"
        + "".join(
            f"<td>{float(comp_pivot.loc[field, cohort]):.1f}%</td>"
            for cohort in (
                "Balanced baseline",
                "Top-decile 20d",
                "Top-decile 60d",
                "Durable",
            )
        )
        + "</tr>"
        for field in comp_labels
    )

    top_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row.top_list))}</td>"
        f"<td><strong>{html.escape(str(row.ticker))}</strong><br><small>{pd.Timestamp(row.date).date()}</small></td>"
        f"<td>{float(getattr(row, OUTCOME_20)):+.1f}%</td>"
        f"<td>{float(getattr(row, OUTCOME_60)):+.1f}%</td>"
        f"<td>{float(row.gap_pct):.1f}% / {float(row.event_rvol_20):.1f}x / {float(row.prior_atr_pct_14):.1f}%</td>"
        f"<td>{html.escape(str(row.evidence_posture))}</td>"
        "</tr>"
        for row in top_events.itertuples(index=False)
    )

    hashes = "".join(
        f"<li><code>{html.escape(name)}</code>: <code>{html.escape(digest)}</code></li>"
        for name, digest in input_hashes.items()
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Episodic Pivot process review</title>
<style>
:root{{--ink:#172033;--muted:#667085;--paper:#f4f6f8;--card:#fff;--navy:#12304a;--teal:#147d78;--amber:#a15c00;--red:#a73535;--line:#d8e0e8}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:1180px;margin:auto;padding:38px 24px 72px}} h1,h2{{font-family:Georgia,serif;color:var(--navy)}} h1{{font-size:42px;line-height:1.04;margin:7px 0 10px}} h2{{font-size:27px;margin:0 0 8px}} h3{{font-size:13px;text-transform:uppercase;letter-spacing:.09em;color:var(--teal);margin:0 0 5px}}
.eyebrow{{font-size:11px;text-transform:uppercase;letter-spacing:.16em;font-weight:800;color:var(--teal)}} .dek{{max-width:820px;color:var(--muted);font-size:17px}} .callout{{background:#fff7e8;border-left:5px solid var(--amber);padding:14px 17px;border-radius:9px;margin:22px 0}} .callout strong{{color:#7a4300}}
.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0 34px}} .metric{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}} .metric span{{display:block;color:var(--muted);font-size:12px}} .metric strong{{display:block;color:var(--navy);font-size:27px;margin-top:3px}}
section{{background:var(--card);border:1px solid var(--line);border-radius:13px;padding:22px;margin:0 0 18px;box-shadow:0 4px 16px rgba(20,40,60,.04)}} .lead-grid,.process-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}} .lead-grid article,.process-grid article{{border-top:3px solid #b9d9d6;padding-top:10px}} .lead-grid p,.process-grid p{{margin:0;color:#344054}}
table{{width:100%;border-collapse:collapse;margin-top:14px;font-variant-numeric:tabular-nums}} th{{text-align:left;background:#eef3f7;color:#344054;font-size:12px}} th,td{{padding:9px 10px;border-bottom:1px solid var(--line);vertical-align:top}} tr:last-child td{{border-bottom:0}} small,.note{{color:var(--muted)}} .scroll{{overflow:auto}} code{{font-size:12px;word-break:break-all}} ul{{padding-left:20px}} li{{margin:7px 0}} .good{{color:#166534;font-weight:700}} .caution{{color:#8a4b00;font-weight:700}} footer{{color:var(--muted);font-size:12px;margin-top:26px}}
@media(max-width:850px){{.metrics{{grid-template-columns:1fr 1fr}} .lead-grid,.process-grid{{grid-template-columns:1fr}}}} @media(max-width:520px){{.metrics{{grid-template-columns:1fr}} h1{{font-size:34px}}}}
</style></head><body><main>
<div class="eyebrow">Episodic Pivot · Historical process review</div>
<h1>A clustered research queue, not a daily signal factory</h1>
<p class="dek">The recent mechanical proxy produces a manageable number of strict, prior-ATR-qualified events. Its 20-session typical result is approximately flat, while the 60-session distribution has a positive but right-tailed center. The useful conclusion is operational: keep the gate broad, make catalyst proof prospective, and preserve separate fast-versus-durable outcome labels.</p>
<div class="callout"><strong>Research-only evidence.</strong> Every outcome is a frictionless SPY-excess observation from the next open. The event volume confirmation is ex post, the company panel is survivor-biased, and no historical catalyst identity is point-in-time validated. Nothing here supports automatic entry, sizing, staging, or production activation.</div>
<div class="metrics">
  <div class="metric"><span>Recent weekly mean</span><strong>{recent_week.mean_all:.1f}</strong><span>strict events; {recent_week.zero_rate_pct:.0f}% zero weeks</span></div>
  <div class="metric"><span>Recent monthly mean</span><strong>{recent_month.mean_all:.1f}</strong><span>strict events</span></div>
  <div class="metric"><span>2024 / 2025 full years</span><strong>{html.escape(yearly_label)}</strong><span>events; 2026 reached 140 through Aug. 14</span></div>
  <div class="metric"><span>Recent weekly P90</span><strong>{recent_week.p90_all:.0f}</strong><span>burst-capacity planning number</span></div>
</div>

<section><h2>First-read conclusions</h2><div class="lead-grid">
  <article><h3>Cadence</h3><p>Plan on roughly <strong>2–3 ATR-qualified names per week</strong> and <strong>10–12 per month</strong> in the recent sample. Expect clusters, not a smooth stream.</p></article>
  <article><h3>Best common trait</h3><p><strong>Higher prior ATR</strong> is the only measured trait whose durable-winner enrichment repeats across full history, 2020+, and the consumed 2024–26 slice. It is exploratory evidence, not a new threshold.</p></article>
  <article><h3>Weak discriminators</h3><p>Gap size, event RVOL, liquidity, prior trend, and historical news categories do not create a stable validated ranking. News timing and issuer identity are too weak historically for causal use.</p></article>
</div></section>

<section><h2>What should run each day</h2><div class="process-grid">
  <article><h3>19:20 ET · Queue</h3><p>Capture and validate the after-hours TradingView screen for the next NYSE session. Store the full export and its displayed count. Do no broker work.</p></article>
  <article><h3>08:20 ET · Verify</h3><p>Combine the night queue with fresh premarket discovery, then pull adjusted daily bars directly from yfinance for every broad move nominee. Strictly require a clean, completed-session ATR(14) above 4% before spending the news budget.</p></article>
  <article><h3>After ATR · Explain</h3><p>Research causal news for at most 25 eligible names, optionally add read-only IBKR execution data, and write a review-only HTML report. Missing IBKR data suppresses sizing but does not turn candidate research into a failed run.</p></article>
</div></section>

<section><h2>Expected candidate load</h2><p class="note">Counts are distinct ticker-date events in the strict historical proxy: positive ≥10% open gap, prior close ≥$3, prior ADDV ≥$5m, ex-post event RVOL ≥2x, prior 63d return ≤20%, first confirmed event in 126 sessions, and prior ATR(14)/close strictly above 4%.</p><div class="scroll"><table><thead><tr><th>Window</th><th>Period</th><th>Events</th><th>Mean / all periods</th><th>Active median [IQR]</th><th>P90 / all</th><th>Zero rate</th></tr></thead><tbody>{cadence_rows}</tbody></table></div>
<h3 style="margin-top:20px">Recent annual counts</h3><div class="scroll"><table><thead><tr><th>Year</th><th>Events</th><th>Status</th></tr></thead><tbody>{year_rows}</tbody></table></div></section>

<section><h2>Best-performing cohorts</h2><p>Top-decile cutoffs were <strong>{facts['top_20_threshold_pct']:.2f}%</strong> SPY excess at 20 sessions (N={facts['top_20_n']}) and <strong>{facts['top_60_threshold_pct']:.2f}%</strong> at 60 sessions (N={facts['top_60_n']}). “Durable” requires positive 20d excess plus top-decile 60d excess (N={facts['durable_n']}). Only <strong>{facts['top_decile_overlap_n']}</strong> events—{facts['top_decile_overlap_pct_of_top_20']:.1f}% of the fast cohort—landed in both tails.</p>
<div class="scroll"><table><thead><tr><th>Median trait</th><th>Balanced baseline</th><th>Top 20d</th><th>Top 60d</th><th>Durable</th></tr></thead><tbody>{trait_rows}</tbody></table></div>
<p class="note">Durable median returns were {facts['durable_20d_median_pct']:.1f}% at 20d and {facts['durable_60d_median_pct']:.1f}% at 60d. These are outcome-defined cohorts and therefore descriptive, not tradable rules.</p></section>

<section><h2>The repeatable trait: volatility within an already volatile set</h2><p>Durable-winner rates generally rise through prior-ATR quartiles. This pattern repeats by era, but quartiles and winner labels were observed after outcomes and many traits were inspected. Keep 4% as the current minimum; store the continuous ATR value and treat higher bands as a <em>research-priority feature</em> until prospective evidence accumulates.</p><div class="scroll"><table><thead><tr><th>Window</th><th>ATR quartile</th><th>N</th><th>ATR range</th><th>Durable rate</th></tr></thead><tbody>{atr_rows}</tbody></table></div></section>

<section><h2>Why historical news cannot rank the queue</h2><p>No event has point-in-time-validated primary catalyst identity, and every secondary trajectory label is unresolved. Winners actually have <em>less</em> current-CIK pre-open SEC earnings context and more missing pre-open evidence than the baseline; that is an evidence-quality and era confound, not proof that non-earnings events are superior.</p><div class="scroll"><table><thead><tr><th>Historical context rate</th><th>Baseline</th><th>Top 20d</th><th>Top 60d</th><th>Durable</th></tr></thead><tbody>{composition_rows}</tbody></table></div></section>

<section><h2>How this should change the forward process</h2><ol>
  <li><strong>Do not turn the screen into an automatic entry rule.</strong> The 20d mean/median is {facts['mean_20_pct']:+.2f}%/{facts['median_20_pct']:+.2f}%; excluding the top 1% leaves a {facts['mean_20_excluding_top_1pct_pct']:+.2f}% mean.</li>
  <li><strong>Track two outcomes.</strong> Fast 20d and durable 60d winners overlap only {facts['top_decile_overlap_pct_of_top_20']:.1f}%, so a single “worked/did not work” label hides different behaviors.</li>
  <li><strong>Use continuous prior ATR for triage, not sizing.</strong> Higher ATR deserves earlier research attention, while the strict >4% floor remains unchanged until prospective, clustered validation says otherwise.</li>
  <li><strong>Freeze causal evidence before the open.</strong> Store issuer identity, exact publication/acceptance time, primary-document text, quantified surprise, and whether the news was genuinely new. These are the variables history could not recover.</li>
  <li><strong>Manage clusters as one risk event.</strong> Winners and losers arrive in correlated date clusters; a seven-name morning is not seven independent bets. The shadow report should display cluster size and shared macro/sector context.</li>
  <li><strong>Collect execution reality.</strong> Log 08:20 and final-refresh spread, depth, halt state, slippage estimate, skipped capacity, and hypothetical entry versus actual prints. Keep every object structurally non-executable.</li>
</ol></section>

<section><h2>Largest historical observations</h2><p class="note">Outliers are shown to make the right tail auditable, not to imply repeatability. They remain exposed to survivorship, ticker reuse, current-identity mapping, corporate-action adjustments, regime clustering, and ex-post volume confirmation.</p><div class="scroll"><table><thead><tr><th>List</th><th>Event</th><th>20d excess</th><th>60d excess</th><th>Gap / RVOL / prior ATR</th><th>Evidence posture</th></tr></thead><tbody>{top_rows}</tbody></table></div></section>

<section><h2>Method and frozen inputs</h2><ul><li>Basis-review-cleared events: 1,300; date range 2000-07-06 through 2026-08-14.</li><li>20d eligible N={facts['eligible_20_n']}; 60d eligible N={facts['eligible_60_n']}; balanced N={facts['balanced_n']}.</li><li>60d mean/median is {facts['mean_60_pct']:+.2f}%/{facts['median_60_pct']:+.2f}%; excluding the top 1% leaves {facts['mean_60_excluding_top_1pct_pct']:+.2f}%.</li>{hashes}</ul></section>
<footer>Generated from frozen local research artifacts. This report prioritizes diligence only; it is not an investment recommendation or an instruction to trade.</footer>
</main></body></html>"""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an EP cadence and winner-trait research artifact"
    )
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    candidates_path = args.candidates.resolve()
    evidence_path = args.evidence.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing analysis: {output_dir}")

    candidates = pd.read_parquet(candidates_path)
    evidence = pd.read_parquet(evidence_path)
    candidate_fields = {
        "ticker",
        "date",
        "basis_review_cleared",
        "earnings_date_match",
        OUTCOME_20,
        OUTCOME_60,
        *(field for field in TRAIT_FIELDS if field != "unique_source_clusters"),
    }
    evidence_fields = [
        "ticker",
        "date",
        "event_id",
        "evidence_posture",
        "preopen_sec_event_type",
        "secondary_context_event_type",
        "trajectory_posture",
        "unique_source_clusters",
    ]
    _require_columns(candidates, candidate_fields, label="candidate")
    _require_columns(evidence, set(evidence_fields), label="evidence")
    candidates["date"] = pd.to_datetime(candidates["date"]).dt.normalize()
    evidence["date"] = pd.to_datetime(evidence["date"]).dt.normalize()
    if candidates.duplicated(["ticker", "date"]).any():
        raise SystemExit("candidate input has duplicate ticker-date rows")
    if evidence.duplicated(["ticker", "date"]).any():
        raise SystemExit("evidence input has duplicate ticker-date rows")
    events = candidates[candidates["basis_review_cleared"].fillna(False)].copy()
    events = events.merge(
        evidence[evidence_fields],
        on=["ticker", "date"],
        how="left",
        validate="one_to_one",
    )
    if events["event_id"].isna().any():
        raise SystemExit("evidence input does not cover every basis-cleared event")
    if not events["prior_atr_pct_14"].gt(4.0).all():
        raise SystemExit("analysis population violates the strict prior ATR >4% gate")

    output_dir.mkdir(parents=True, exist_ok=False)

    cadence, by_year = cadence_summary(events)
    facts, traits, composition, atr_quartiles, top_events = winner_analysis(events)
    input_hashes = {
        candidates_path.name: _sha256(candidates_path),
        evidence_path.name: _sha256(evidence_path),
    }
    facts.update(
        {
            "basis_review_cleared_n": len(events),
            "unique_tickers": int(events["ticker"].nunique()),
            "unique_event_dates": int(events["date"].nunique()),
            "min_date": events["date"].min().date().isoformat(),
            "max_date": events["date"].max().date().isoformat(),
            "input_hashes": input_hashes,
            "safety": {
                "research_only": True,
                "investment_recommendation": False,
                "order_staging_performed": False,
                "broker_contacted": False,
                "production_deployed": False,
            },
        }
    )

    cadence.to_csv(output_dir / "cadence_summary.csv", index=False)
    by_year.to_csv(output_dir / "events_by_year.csv", index=False)
    traits.to_csv(output_dir / "winner_trait_comparison.csv", index=False)
    composition.to_csv(output_dir / "news_context_comparison.csv", index=False)
    atr_quartiles.to_csv(output_dir / "atr_quartile_durable_rates.csv", index=False)
    top_events.to_csv(
        output_dir / "top_historical_observations.csv",
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )
    _json_dump(output_dir / "summary.json", facts)
    (output_dir / "report.html").write_text(
        _html_report(
            cadence=cadence,
            by_year=by_year,
            facts=facts,
            traits=traits,
            composition=composition,
            atr_quartiles=atr_quartiles,
            top_events=top_events,
            input_hashes=input_hashes,
        ),
        encoding="utf-8",
    )
    artifact_names = sorted(path.name for path in output_dir.iterdir())
    _json_dump(
        output_dir / "manifest.json",
        {
            "schema_version": 1,
            "research_only": True,
            "inputs": input_hashes,
            "artifacts": {
                name: {
                    "sha256": _sha256(output_dir / name),
                    "size_bytes": (output_dir / name).stat().st_size,
                }
                for name in artifact_names
            },
        },
    )
    print(f"Wrote EP process review: {output_dir}")
    print("Safety: descriptive research only; no broker or production action.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
