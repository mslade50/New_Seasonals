"""Backtest the frozen settlement-adjusted month-end cash-dash strategy.

This script is intentionally self-contained and read-only with respect to the
project's existing data. It reads data/master_prices.parquet and writes all
research artifacts beside this file.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
PRICE_PATH = REPO_ROOT / "data" / "master_prices.parquet"

PAPER_SAMPLE_END = pd.Timestamp("2013-12-31")
T2_START = pd.Timestamp("2017-09-05")
T1_START = pd.Timestamp("2024-05-28")
PRIMARY_HOLD_SESSIONS = 3
PRIMARY_COST_BPS = 4.0


@dataclass(frozen=True)
class SummaryStats:
    n: int
    mean_pct: float
    median_pct: float
    t_stat: float
    p_value_two_sided: float
    win_rate_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    payoff_ratio: float
    profit_factor: float
    best_pct: float
    worst_pct: float
    compounded_pct: float
    max_drawdown_pct: float
    annualized_return_pct: float
    annualized_sharpe: float


def load_spy() -> pd.DataFrame:
    df = pd.read_parquet(PRICE_PATH, filters=[("ticker", "==", "SPY")])
    df = df.sort_values("date").drop_duplicates("date").set_index("date")
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"SPY price data are missing columns: {sorted(missing)}")
    if not df.index.is_monotonic_increasing:
        raise ValueError("SPY dates must be increasing")
    return df


def settlement_days_for_month_end(month_end: pd.Timestamp) -> int:
    """Return the standard U.S. equity settlement lag for that month-end."""
    if month_end >= T1_START:
        return 1
    if month_end >= T2_START:
        return 2
    return 3


def build_events(
    prices: pd.DataFrame,
    hold_sessions: int = PRIMARY_HOLD_SESSIONS,
    cost_bps: float = PRIMARY_COST_BPS,
    static_entry_offset: int | None = None,
) -> pd.DataFrame:
    """Build monthly events.

    `static_entry_offset` is a negative offset from month-end and is used only
    for control portfolios. With None, entry is -(settlement_days + 1).
    """
    dates = prices.index
    close = prices["Close"].astype(float).to_numpy()
    position = pd.Series(np.arange(len(dates)), index=dates)
    month_ends = prices.groupby(prices.index.to_period("M")).tail(1).index
    rows: list[dict[str, object]] = []

    for month_end in month_ends:
        t_pos = int(position.loc[month_end])
        settlement_days = settlement_days_for_month_end(month_end)
        entry_offset = (
            static_entry_offset
            if static_entry_offset is not None
            else -(settlement_days + 1)
        )
        entry_pos = t_pos + entry_offset
        exit_pos = entry_pos + hold_sessions
        pressure_start_pos = entry_pos - 5
        if pressure_start_pos < 0 or entry_pos < 0 or exit_pos >= len(dates):
            continue

        gross_return = close[exit_pos] / close[entry_pos] - 1.0
        net_return = gross_return - cost_bps / 10_000.0
        pressure_5d = close[entry_pos] / close[pressure_start_pos] - 1.0
        rows.append(
            {
                "month": str(month_end.to_period("M")),
                "month_end": month_end,
                "month_end_weekday": month_end.day_name(),
                "month_end_is_friday": bool(month_end.weekday() == 4),
                "settlement_days": settlement_days,
                "settlement_regime": f"T+{settlement_days}",
                "entry_offset": entry_offset,
                "entry_date": dates[entry_pos],
                "exit_date": dates[exit_pos],
                "hold_sessions": hold_sessions,
                "entry_close": close[entry_pos],
                "exit_close": close[exit_pos],
                "pressure_5d": pressure_5d,
                "gross_return": gross_return,
                "cost_bps": cost_bps,
                "net_return": net_return,
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["year"] = pd.to_datetime(out["month_end"]).dt.year
        out["sample"] = np.where(
            pd.to_datetime(out["month_end"]) <= PAPER_SAMPLE_END,
            "paper_era_through_2013",
            "holdout_2014_plus",
        )
    return out


def summary_stats(returns: pd.Series, dates: pd.Series) -> SummaryStats:
    r = pd.Series(returns, dtype=float).dropna()
    if r.empty:
        raise ValueError("Cannot summarize an empty return series")
    n = len(r)
    std = r.std(ddof=1)
    t_stat = float(r.mean() / (std / math.sqrt(n))) if n > 1 and std > 0 else math.nan
    p_value = float(2 * stats.t.sf(abs(t_stat), n - 1)) if n > 1 else math.nan
    wins = r[r > 0]
    losses = r[r < 0]
    avg_win = wins.mean() if len(wins) else math.nan
    avg_loss = losses.mean() if len(losses) else math.nan
    payoff = avg_win / abs(avg_loss) if len(wins) and len(losses) else math.nan
    profit_factor = wins.sum() / abs(losses.sum()) if len(losses) else math.inf
    equity = (1.0 + r).cumprod()
    max_drawdown = (equity / equity.cummax() - 1.0).min()

    d = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    elapsed_years = max((d.max() - d.min()).days / 365.2425, 1 / 12)
    annualized_return = equity.iloc[-1] ** (1.0 / elapsed_years) - 1.0
    annualized_sharpe = r.mean() / std * math.sqrt(12) if std > 0 else math.nan

    return SummaryStats(
        n=n,
        mean_pct=float(r.mean() * 100),
        median_pct=float(r.median() * 100),
        t_stat=t_stat,
        p_value_two_sided=p_value,
        win_rate_pct=float((r > 0).mean() * 100),
        avg_win_pct=float(avg_win * 100),
        avg_loss_pct=float(avg_loss * 100),
        payoff_ratio=float(payoff),
        profit_factor=float(profit_factor),
        best_pct=float(r.max() * 100),
        worst_pct=float(r.min() * 100),
        compounded_pct=float((equity.iloc[-1] - 1.0) * 100),
        max_drawdown_pct=float(max_drawdown * 100),
        annualized_return_pct=float(annualized_return * 100),
        annualized_sharpe=float(annualized_sharpe),
    )


def bootstrap_mean_ci(
    returns: pd.Series, *, draws: int = 50_000, seed: int = 73
) -> dict[str, float]:
    r = pd.Series(returns, dtype=float).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    means = rng.choice(r, size=(draws, len(r)), replace=True).mean(axis=1)
    return {
        "draws": draws,
        "seed": seed,
        "mean_pct": float(r.mean() * 100),
        "ci_2_5_pct": float(np.quantile(means, 0.025) * 100),
        "ci_97_5_pct": float(np.quantile(means, 0.975) * 100),
        "probability_mean_le_zero": float((means <= 0).mean()),
    }


def leave_one_year_out(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year in sorted(events["year"].unique()):
        remaining = events.loc[events["year"] != year, "net_return"]
        rows.append(
            {
                "omitted_year": int(year),
                "n": len(remaining),
                "mean_net_pct": remaining.mean() * 100,
            }
        )
    return pd.DataFrame(rows)


def non_event_baseline(prices: pd.DataFrame, events: pd.DataFrame) -> pd.Series:
    """Greedily form non-overlapping three-session blocks outside event risk."""
    dates = prices.index
    close = prices["Close"].astype(float).to_numpy()
    position = pd.Series(np.arange(len(dates)), index=dates)
    occupied: set[int] = set()
    for row in events.itertuples(index=False):
        a = int(position.loc[pd.Timestamp(row.entry_date)])
        b = int(position.loc[pd.Timestamp(row.exit_date)])
        occupied.update(range(a + 1, b + 1))

    returns = []
    i = 0
    while i + PRIMARY_HOLD_SESSIONS < len(dates):
        risk_sessions = set(range(i + 1, i + PRIMARY_HOLD_SESSIONS + 1))
        if risk_sessions.isdisjoint(occupied):
            returns.append(close[i + PRIMARY_HOLD_SESSIONS] / close[i] - 1.0)
            i += PRIMARY_HOLD_SESSIONS
        else:
            i += 1
    return pd.Series(returns, dtype=float, name="baseline_gross_return")


def welch_difference(a: pd.Series, b: pd.Series) -> dict[str, float]:
    a = pd.Series(a, dtype=float).dropna()
    b = pd.Series(b, dtype=float).dropna()
    test = stats.ttest_ind(a, b, equal_var=False)
    return {
        "mean_a_pct": float(a.mean() * 100),
        "mean_b_pct": float(b.mean() * 100),
        "difference_pct": float((a.mean() - b.mean()) * 100),
        "welch_t": float(test.statistic),
        "p_value_two_sided": float(test.pvalue),
    }


def fmt(value: float, decimals: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{decimals}f}"


def make_results_markdown(
    events: pd.DataFrame,
    summaries: dict[str, dict[str, object]],
    mechanism: dict[str, object],
    bootstrap: dict[str, float],
    loyo: pd.DataFrame,
    robustness: pd.DataFrame,
    controls: pd.DataFrame,
    yearly: pd.DataFrame,
    decision: dict[str, object],
) -> str:
    full = summaries["full"]
    holdout = summaries["holdout_2014_plus"]
    paper = summaries["paper_era_through_2013"]
    friday = mechanism["friday_vs_other"]
    pressure = mechanism["pressure_relation"]
    baseline = mechanism["event_vs_non_event_baseline"]

    summary_rows = []
    for label, s in [
        ("Full sample", full),
        ("Paper era through 2013", paper),
        ("Untouched holdout, 2014+", holdout),
    ]:
        summary_rows.append(
            f"| {label} | {s['n']} | {fmt(s['mean_pct'], 3)}% | "
            f"{fmt(s['t_stat'])} | {fmt(s['win_rate_pct'], 1)}% | "
            f"{fmt(s['profit_factor'])} | {fmt(s['worst_pct'])}% |"
        )

    regime_table = controls.loc[controls["table"] == "regime_offset_matrix"].copy()
    regime_lines = [
        f"| {r.settlement_regime} | {int(r.entry_offset)} | {int(r.n)} | "
        f"{fmt(r.mean_net_pct, 3)}% | {fmt(r.t_stat)} |"
        for r in regime_table.itertuples(index=False)
    ]

    robust_lines = [
        f"| {int(r.hold_sessions)} | {fmt(r.cost_bps, 0)} | {int(r.n)} | "
        f"{fmt(r.mean_net_pct, 3)}% | {fmt(r.t_stat)} | {fmt(r.win_rate_pct, 1)}% |"
        for r in robustness.itertuples(index=False)
    ]

    yearly_lines = [
        f"| {int(r.year)} | {int(r.n)} | {fmt(r.mean_net_pct, 3)}% | "
        f"{fmt(r.compounded_net_pct, 2)}% |"
        for r in yearly.itertuples(index=False)
    ]

    verdict = "GRADUATES" if decision["graduates"] else "REJECTED"
    reasons = "\n".join(f"- {x}" for x in decision["reasons"])
    lo_min = loyo.loc[loyo["mean_net_pct"].idxmin()]
    lo_max = loyo.loc[loyo["mean_net_pct"].idxmax()]

    return f"""# Settlement-Adjusted Month-End Cash-Dash Reversal: Results

## Verdict

**{verdict} under the frozen decision rule.**

{reasons}

The primary test contains {len(events)} monthly events from
{events['entry_date'].min().date()} through {events['exit_date'].max().date()}.
All reported strategy returns below include the frozen 4 bp round-trip cost.

## Primary evidence

| Sample | N | Mean/event | t-stat | Win rate | Profit factor | Worst event |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(summary_rows)}

The full-sample bootstrap mean is {fmt(bootstrap['mean_pct'], 3)}%, with a 95%
month-resampled interval of [{fmt(bootstrap['ci_2_5_pct'], 3)}%,
{fmt(bootstrap['ci_97_5_pct'], 3)}%] and P(mean <= 0) =
{fmt(bootstrap['probability_mean_le_zero'], 4)}. The event-only compounded
return is {fmt(full['compounded_pct'], 1)}%, the annualized return is
{fmt(full['annualized_return_pct'], 2)}%, and maximum event-curve drawdown is
{fmt(full['max_drawdown_pct'], 2)}% while invested roughly three sessions per
month.

Leaving out any single calendar year produces mean returns from
{fmt(lo_min['mean_net_pct'], 3)}% (omit {int(lo_min['omitted_year'])}) to
{fmt(lo_max['mean_net_pct'], 3)}% (omit {int(lo_max['omitted_year'])}).

## Mechanism checks

1. **Friday payment overlap.** Friday month-ends average
   {fmt(friday['mean_a_pct'], 3)}% versus {fmt(friday['mean_b_pct'], 3)}% for
   other weekdays (difference {fmt(friday['difference_pct'], 3)}%, Welch
   t={fmt(friday['welch_t'])}).
2. **Prior selling pressure.** The Pearson correlation between the five-session
   return ending at entry and the subsequent gross reversal is
   {fmt(pressure['pearson_r'], 3)} (p={fmt(pressure['pearson_p'], 4)}); the
   Spearman correlation is {fmt(pressure['spearman_r'], 3)}. Events following
   negative pressure average {fmt(pressure['negative_pressure_mean_gross_pct'], 3)}%
   gross versus {fmt(pressure['nonnegative_pressure_mean_gross_pct'], 3)}% after
   nonnegative pressure.
3. **Settlement timing.** The matrix below shows each settlement regime against
   each candidate entry offset. The frozen rule uses the bold economic mapping
   T+3/T-4, T+2/T-3, and T+1/T-2; controls are diagnostic, not optimized
   replacements.

| Regime | Entry offset | N | Mean net | t-stat |
|---|---:|---:|---:|---:|
{chr(10).join(regime_lines)}

As a broad baseline, non-event, non-overlapping three-session SPY blocks average
{fmt(baseline['mean_b_pct'], 3)}% gross versus {fmt(baseline['mean_a_pct'], 3)}%
for the scheduled event windows (difference {fmt(baseline['difference_pct'], 3)}%,
Welch t={fmt(baseline['welch_t'])}).

## Robustness map

Only the 3-session, 4-bp row is the frozen primary specification.

| Hold sessions | Cost (bp) | N | Mean net | t-stat | Win rate |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(robust_lines)}

## Calendar-year returns

| Year | N | Mean/event | Compounded |
|---:|---:|---:|---:|
{chr(10).join(yearly_lines)}

## Interpretation boundary

This is evidence for a recurring liquidity premium, not proof that every event
is caused by month-end payments. It is a low-frequency equity-beta trade with
gap risk and no stop. Its main live risk is structural decay: settlement has
already shortened to T+1, and the T+1 regime has a much smaller sample. The
strategy should therefore remain a research sleeve until the modern regime has
enough observations for a prospective recheck.
"""


def main() -> None:
    prices = load_spy()
    events = build_events(prices)

    # Boundary and construction guards.
    regime_by_month = events.set_index("month")["settlement_regime"]
    assert regime_by_month.loc["2017-08"] == "T+3"
    assert regime_by_month.loc["2017-09"] == "T+2"
    assert regime_by_month.loc["2024-04"] == "T+2"
    assert regime_by_month.loc["2024-05"] == "T+1"
    assert (events["exit_date"] > events["entry_date"]).all()
    assert events["month"].is_unique

    summaries: dict[str, dict[str, object]] = {}
    summaries["full"] = asdict(summary_stats(events["net_return"], events["exit_date"]))
    for sample, group in events.groupby("sample", sort=False):
        summaries[str(sample)] = asdict(summary_stats(group["net_return"], group["exit_date"]))
    for regime, group in events.groupby("settlement_regime", sort=False):
        summaries[f"regime_{regime}"] = asdict(
            summary_stats(group["net_return"], group["exit_date"])
        )

    friday = events.loc[events["month_end_is_friday"], "gross_return"]
    other = events.loc[~events["month_end_is_friday"], "gross_return"]
    friday_test = welch_difference(friday, other)
    friday_test.update({"n_friday": len(friday), "n_other": len(other)})

    pressure_valid = events[["pressure_5d", "gross_return"]].dropna()
    pearson = stats.pearsonr(pressure_valid["pressure_5d"], pressure_valid["gross_return"])
    spearman = stats.spearmanr(pressure_valid["pressure_5d"], pressure_valid["gross_return"])
    neg = pressure_valid.loc[pressure_valid["pressure_5d"] < 0, "gross_return"]
    nonneg = pressure_valid.loc[pressure_valid["pressure_5d"] >= 0, "gross_return"]
    pressure_relation = {
        "n": len(pressure_valid),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_r": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "negative_pressure_n": len(neg),
        "negative_pressure_mean_gross_pct": float(neg.mean() * 100),
        "nonnegative_pressure_n": len(nonneg),
        "nonnegative_pressure_mean_gross_pct": float(nonneg.mean() * 100),
        "negative_minus_nonnegative": welch_difference(neg, nonneg),
    }

    baseline = non_event_baseline(prices, events)
    baseline_test = welch_difference(events["gross_return"], baseline)
    baseline_test.update({"n_event": len(events), "n_baseline": len(baseline)})

    mechanism: dict[str, object] = {
        "friday_vs_other": friday_test,
        "pressure_relation": pressure_relation,
        "event_vs_non_event_baseline": baseline_test,
    }

    # Frozen robustness map: hold horizon and friction are not selected on.
    robust_rows = []
    for hold in range(1, 6):
        for cost in (0.0, 4.0, 10.0, 20.0):
            test_events = build_events(prices, hold_sessions=hold, cost_bps=cost)
            s = summary_stats(test_events["net_return"], test_events["exit_date"])
            robust_rows.append(
                {
                    "hold_sessions": hold,
                    "cost_bps": cost,
                    "n": s.n,
                    "mean_net_pct": s.mean_pct,
                    "t_stat": s.t_stat,
                    "win_rate_pct": s.win_rate_pct,
                    "profit_factor": s.profit_factor,
                }
            )
    robustness = pd.DataFrame(robust_rows)

    # Within-regime timing matrix: preferred offset should move toward T.
    control_frames = []
    for offset in (-4, -3, -2):
        c = build_events(
            prices,
            hold_sessions=PRIMARY_HOLD_SESSIONS,
            cost_bps=PRIMARY_COST_BPS,
            static_entry_offset=offset,
        )
        for regime, group in c.groupby("settlement_regime", sort=False):
            s = summary_stats(group["net_return"], group["exit_date"])
            control_frames.append(
                {
                    "table": "regime_offset_matrix",
                    "settlement_regime": regime,
                    "entry_offset": offset,
                    "n": s.n,
                    "mean_net_pct": s.mean_pct,
                    "t_stat": s.t_stat,
                    "win_rate_pct": s.win_rate_pct,
                }
            )
    controls = pd.DataFrame(control_frames)

    yearly = (
        events.groupby("year", as_index=False)
        .agg(
            n=("net_return", "size"),
            mean_net_pct=("net_return", lambda x: x.mean() * 100),
            compounded_net_pct=("net_return", lambda x: ((1 + x).prod() - 1) * 100),
        )
    )
    loyo = leave_one_year_out(events)
    bootstrap = bootstrap_mean_ci(events["net_return"])

    holdout_positive = summaries["holdout_2014_plus"]["mean_pct"] > 0
    yearly_positive_count = int((yearly["compounded_net_pct"] > 0).sum())
    regime_means = {
        k.removeprefix("regime_"): v["mean_pct"]
        for k, v in summaries.items()
        if k.startswith("regime_")
    }
    all_regimes_positive = all(x > 0 for x in regime_means.values())

    friday_pass = friday_test["difference_pct"] > 0
    pressure_pass = pressure_relation["pearson_r"] < 0
    # Timing passes if the mapped offset is the best of the tested offsets in
    # at least two regimes. This is deliberately directional, not a p-value hunt.
    timing_best = (
        controls.sort_values(
            ["settlement_regime", "mean_net_pct"], ascending=[True, False]
        )
        .groupby("settlement_regime")
        .first()["entry_offset"]
        .to_dict()
    )
    predicted = {"T+3": -4, "T+2": -3, "T+1": -2}
    timing_matches = sum(timing_best.get(k) == v for k, v in predicted.items())
    timing_pass = timing_matches >= 2
    mechanism_passes = int(friday_pass) + int(pressure_pass) + int(timing_pass)

    graduates = bool(
        holdout_positive
        and all_regimes_positive
        and mechanism_passes >= 2
        and yearly_positive_count >= math.ceil(len(yearly) / 2)
    )
    reasons = [
        f"2014+ holdout mean after costs is {summaries['holdout_2014_plus']['mean_pct']:.3f}% "
        f"({'pass' if holdout_positive else 'fail'}).",
        f"Positive calendar years: {yearly_positive_count}/{len(yearly)}; each settlement-regime mean "
        f"positive = {all_regimes_positive}.",
        f"Mechanism checks passed: {mechanism_passes}/3 "
        f"(Friday={friday_pass}, pressure={pressure_pass}, timing={timing_pass}; "
        f"predicted offset was best in {timing_matches}/3 regimes).",
    ]
    decision = {
        "graduates": graduates,
        "reasons": reasons,
        "holdout_positive": bool(holdout_positive),
        "all_regimes_positive": bool(all_regimes_positive),
        "positive_years": yearly_positive_count,
        "total_years": len(yearly),
        "mechanism_passes": mechanism_passes,
        "friday_pass": bool(friday_pass),
        "pressure_pass": bool(pressure_pass),
        "timing_pass": bool(timing_pass),
        "timing_matches": timing_matches,
        "timing_best_offsets": timing_best,
    }

    events.to_csv(HERE / "events.csv", index=False, float_format="%.10f")
    yearly.to_csv(HERE / "yearly.csv", index=False, float_format="%.10f")
    loyo.to_csv(HERE / "leave_one_year_out.csv", index=False, float_format="%.10f")
    robustness.to_csv(HERE / "robustness.csv", index=False, float_format="%.10f")
    controls.to_csv(HERE / "controls.csv", index=False, float_format="%.10f")

    payload = {
        "data": {
            "source": str(PRICE_PATH.relative_to(REPO_ROOT)),
            "first_date": str(prices.index.min().date()),
            "last_date": str(prices.index.max().date()),
            "rows": len(prices),
        },
        "frozen_spec": {
            "instrument": "SPY",
            "entry": "MOC at T-(settlement_days+1)",
            "exit": f"MOC {PRIMARY_HOLD_SESSIONS} sessions later",
            "cost_bps_round_trip": PRIMARY_COST_BPS,
            "paper_sample_end": str(PAPER_SAMPLE_END.date()),
        },
        "summaries": summaries,
        "bootstrap": bootstrap,
        "mechanism": mechanism,
        "decision": decision,
    }
    (HERE / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    (HERE / "RESULTS.md").write_text(
        make_results_markdown(
            events,
            summaries,
            mechanism,
            bootstrap,
            loyo,
            robustness,
            controls,
            yearly,
            decision,
        ),
        encoding="utf-8",
    )

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

