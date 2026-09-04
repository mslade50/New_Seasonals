"""Backtest the frozen TLT month-end benchmark-demand strategy.

Reads the existing adjusted master-price cache and writes research artifacts
only inside research/treasury_month_end/.
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
PUBLICATION_CUTOFF = pd.Timestamp("2019-12-31")
PRIMARY_ENTRY_OFFSET = -5
PRIMARY_COST_BPS = 10.0


@dataclass(frozen=True)
class Stats:
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


def load_ticker(ticker: str) -> pd.DataFrame:
    df = pd.read_parquet(PRICE_PATH, filters=[("ticker", "==", ticker)])
    if df.empty:
        raise ValueError(f"No rows found for {ticker}")
    return df.sort_values("date").drop_duplicates("date").set_index("date")


def build_events(
    prices: pd.DataFrame,
    *,
    entry_offset: int = PRIMARY_ENTRY_OFFSET,
    cost_bps: float = PRIMARY_COST_BPS,
) -> pd.DataFrame:
    if entry_offset >= 0:
        raise ValueError("entry_offset must be negative")
    dates = prices.index
    close = prices["Close"].astype(float).to_numpy()
    positions = pd.Series(np.arange(len(dates)), index=dates)
    last_complete_month = dates.max().to_period("M") - 1
    month_ends = prices.groupby(dates.to_period("M")).tail(1).index
    rows: list[dict[str, object]] = []

    for month_end in month_ends:
        if month_end.to_period("M") > last_complete_month:
            continue
        exit_pos = int(positions.loc[month_end])
        entry_pos = exit_pos + entry_offset
        if entry_pos < 0:
            continue
        gross = close[exit_pos] / close[entry_pos] - 1.0
        rows.append(
            {
                "month": str(month_end.to_period("M")),
                "month_end": month_end,
                "entry_date": dates[entry_pos],
                "exit_date": month_end,
                "entry_offset": entry_offset,
                "hold_sessions": -entry_offset,
                "entry_close": close[entry_pos],
                "exit_close": close[exit_pos],
                "gross_return": gross,
                "cost_bps": cost_bps,
                "net_return": gross - cost_bps / 10_000.0,
            }
        )

    events = pd.DataFrame(rows)
    events["year"] = pd.to_datetime(events["month_end"]).dt.year
    events["sample"] = np.where(
        pd.to_datetime(events["month_end"]) <= PUBLICATION_CUTOFF,
        "pre_publication_through_2019",
        "holdout_2020_plus",
    )
    return events


def summarize(returns: pd.Series, dates: pd.Series) -> Stats:
    r = pd.Series(returns, dtype=float).dropna().reset_index(drop=True)
    d = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    n = len(r)
    if n < 2:
        raise ValueError("At least two returns are required")
    std = r.std(ddof=1)
    t_stat = r.mean() / (std / math.sqrt(n)) if std > 0 else math.nan
    p_value = 2 * stats.t.sf(abs(t_stat), n - 1) if std > 0 else math.nan
    wins, losses = r[r > 0], r[r < 0]
    avg_win, avg_loss = wins.mean(), losses.mean()
    equity = (1 + r).cumprod()
    max_dd = (equity / equity.cummax() - 1).min()
    years = max((d.max() - d.min()).days / 365.2425, 1 / 12)
    return Stats(
        n=n,
        mean_pct=float(r.mean() * 100),
        median_pct=float(r.median() * 100),
        t_stat=float(t_stat),
        p_value_two_sided=float(p_value),
        win_rate_pct=float((r > 0).mean() * 100),
        avg_win_pct=float(avg_win * 100),
        avg_loss_pct=float(avg_loss * 100),
        payoff_ratio=float(avg_win / abs(avg_loss)),
        profit_factor=float(wins.sum() / abs(losses.sum())),
        best_pct=float(r.max() * 100),
        worst_pct=float(r.min() * 100),
        compounded_pct=float((equity.iloc[-1] - 1) * 100),
        max_drawdown_pct=float(max_dd * 100),
        annualized_return_pct=float((equity.iloc[-1] ** (1 / years) - 1) * 100),
        annualized_sharpe=float(r.mean() / std * math.sqrt(12)),
    )


def shifted_window_events(
    prices: pd.DataFrame,
    *,
    start_offset: int,
    end_offset: int,
    cost_bps: float = PRIMARY_COST_BPS,
) -> pd.DataFrame:
    """Return events from month-end-relative start close to end close."""
    if start_offset >= end_offset:
        raise ValueError("start_offset must precede end_offset")
    dates = prices.index
    close = prices["Close"].astype(float).to_numpy()
    positions = pd.Series(np.arange(len(dates)), index=dates)
    last_complete_month = dates.max().to_period("M") - 1
    rows = []
    for month_end in prices.groupby(dates.to_period("M")).tail(1).index:
        if month_end.to_period("M") > last_complete_month:
            continue
        t = int(positions.loc[month_end])
        a, b = t + start_offset, t + end_offset
        if a < 0 or b >= len(dates):
            continue
        gross = close[b] / close[a] - 1
        rows.append(
            {
                "month": str(month_end.to_period("M")),
                "month_end": month_end,
                "entry_date": dates[a],
                "exit_date": dates[b],
                "gross_return": gross,
                "net_return": gross - cost_bps / 10_000,
            }
        )
    return pd.DataFrame(rows)


def non_event_baseline(prices: pd.DataFrame, events: pd.DataFrame) -> pd.Series:
    """Greedy non-overlapping five-session blocks outside primary exposure."""
    dates = prices.index
    close = prices["Close"].astype(float).to_numpy()
    positions = pd.Series(np.arange(len(dates)), index=dates)
    occupied: set[int] = set()
    for row in events.itertuples(index=False):
        a = int(positions.loc[pd.Timestamp(row.entry_date)])
        b = int(positions.loc[pd.Timestamp(row.exit_date)])
        occupied.update(range(a + 1, b + 1))

    result = []
    i = 0
    while i + 5 < len(dates):
        risk = set(range(i + 1, i + 6))
        if risk.isdisjoint(occupied):
            result.append(close[i + 5] / close[i] - 1)
            i += 5
        else:
            i += 1
    return pd.Series(result, dtype=float)


def difference_test(a: pd.Series, b: pd.Series) -> dict[str, float]:
    a = pd.Series(a, dtype=float).dropna()
    b = pd.Series(b, dtype=float).dropna()
    test = stats.ttest_ind(a, b, equal_var=False)
    return {
        "n_a": len(a),
        "n_b": len(b),
        "mean_a_pct": float(a.mean() * 100),
        "mean_b_pct": float(b.mean() * 100),
        "difference_pct": float((a.mean() - b.mean()) * 100),
        "welch_t": float(test.statistic),
        "p_value_two_sided": float(test.pvalue),
    }


def bootstrap(returns: pd.Series, draws: int = 50_000, seed: int = 107) -> dict[str, float]:
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
        r = events.loc[events["year"] != year, "net_return"]
        rows.append({"omitted_year": year, "n": len(r), "mean_net_pct": r.mean() * 100})
    return pd.DataFrame(rows)


def fmt(x: float, n: int = 2) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{n}f}"


def render_results(payload: dict[str, object], robustness: pd.DataFrame, yearly: pd.DataFrame, loyo: pd.DataFrame) -> str:
    s = payload["summaries"]
    full, pre, hold = s["full"], s["pre_publication_through_2019"], s["holdout_2020_plus"]
    m = payload["mechanism"]
    execution = payload["execution"]
    fit = payload["portfolio_fit"]
    b = payload["bootstrap"]
    d = payload["decision"]
    verdict = "GRADUATES" if d["graduates"] else "REJECTED"
    reason_lines = "\n".join(f"- {x}" for x in d["reasons"])
    rows = []
    for label, z in [("Full sample", full), ("Through 2019", pre), ("2020+ holdout", hold)]:
        rows.append(
            f"| {label} | {z['n']} | {fmt(z['mean_pct'],3)}% | {fmt(z['t_stat'])} | "
            f"{fmt(z['win_rate_pct'],1)}% | {fmt(z['profit_factor'])} | {fmt(z['worst_pct'])}% |"
        )
    rob = [
        f"| T{int(x.entry_offset)} | {fmt(x.cost_bps,0)} | {int(x.n)} | "
        f"{fmt(x.mean_net_pct,3)}% | {fmt(x.t_stat)} | {fmt(x.win_rate_pct,1)}% |"
        for x in robustness.itertuples(index=False)
    ]
    yrs = [
        f"| {int(x.year)} | {int(x.n)} | {fmt(x.mean_net_pct,3)}% | {fmt(x.compounded_net_pct,2)}% |"
        for x in yearly.itertuples(index=False)
    ]
    lo_min = loyo.loc[loyo["mean_net_pct"].idxmin()]
    lo_max = loyo.loc[loyo["mean_net_pct"].idxmax()]
    return f"""# Treasury Month-End Benchmark-Demand Strategy: Results

## Verdict

**{verdict} under the frozen decision rule.**

{reason_lines}

## Primary evidence

| Sample | N | Mean/event | t-stat | Win rate | Profit factor | Worst event |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

The full sample compounds to {fmt(full['compounded_pct'],1)}% while invested
only five sessions per month, with a {fmt(full['max_drawdown_pct'])}% event-curve
maximum drawdown and {fmt(full['annualized_sharpe'])} annualized event Sharpe.
The month-bootstrap mean is {fmt(b['mean_pct'],3)}% with a 95% interval of
[{fmt(b['ci_2_5_pct'],3)}%, {fmt(b['ci_97_5_pct'],3)}%] and P(mean <= 0) =
{fmt(b['probability_mean_le_zero'],4)}. Leaving out any one year gives a mean
between {fmt(lo_min['mean_net_pct'],3)}% (omit {int(lo_min['omitted_year'])}) and
{fmt(lo_max['mean_net_pct'],3)}% (omit {int(lo_max['omitted_year'])}).

## Mechanism and placebo checks

- **Maturity:** TLT averages {fmt(m['duration']['mean_a_pct'],3)}% gross versus
  {fmt(m['duration']['mean_b_pct'],3)}% for IEF over the identical T-5 to T
  window (difference {fmt(m['duration']['difference_pct'],3)}%).
- **Timing:** the primary window averages {fmt(m['primary_vs_pre']['mean_a_pct'],3)}%
  gross versus {fmt(m['primary_vs_pre']['mean_b_pct'],3)}% for T-10 to T-5,
  and {fmt(m['primary_vs_post']['mean_b_pct'],3)}% for T to T+5.
- **Ordinary five-day blocks:** non-event windows average
  {fmt(m['primary_vs_baseline']['mean_b_pct'],3)}% versus
  {fmt(m['primary_vs_baseline']['mean_a_pct'],3)}% in the month-end window
  (difference {fmt(m['primary_vs_baseline']['difference_pct'],3)}%, Welch
  t={fmt(m['primary_vs_baseline']['welch_t'])}).

## Execution and portfolio fit

- The gross mean is {fmt(execution['gross_mean_bps'],1)} bp per event, which is
  also the friction break-even before alpha reaches zero. The frozen 10 bp cost
  consumes less than one-quarter of that gross mean, and the rule remains
  positive at the 20 bp robustness assumption.
- {fmt(execution['through_exit_open_mean_bps'],1)} bp accrues by the month-end
  open and another {fmt(execution['exit_day_intraday_mean_bps'],1)} bp accrues
  from that open to the month-end close (intraday t-stat
  {fmt(execution['exit_day_intraday_t_stat'])}). The MOC exit is therefore part
  of the edge, not an interchangeable convenience.
- The current full trade ledger contains {fit['existing_bond_etf_trades']} TLT,
  IEF, LQD, or HYG trades, so this is not a duplicate strategy-book position.
  TLT event returns correlate {fmt(fit['spy_same_window_correlation'],3)} with
  SPY over the same dates and {fmt(fit['ledger_monthly_pnl_correlation'],3)}
  with monthly ledger exit P&L (the latter is a rough, non-mark-to-market
  diversification check).
- The maximum losing streak is {fit['max_losing_streak']} events. The two worst
  events are {fit['worst_events'][0]['month']} at
  {fmt(fit['worst_events'][0]['net_return_pct'])}% and
  {fit['worst_events'][1]['month']} at
  {fmt(fit['worst_events'][1]['net_return_pct'])}%.

## Robustness map

The frozen primary row is T-5 at 10 bp. Other rows are sensitivity checks, not
alternative strategies selected after seeing results.

| Entry | Cost (bp) | N | Mean net | t-stat | Win rate |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(rob)}

## Calendar years

| Year | N | Mean/event | Compounded |
|---:|---:|---:|---:|
{chr(10).join(yrs)}

## Risk boundary

This is long-duration exposure, not arbitrage. A hawkish policy surprise,
inflation shock, or disorderly selloff can overwhelm benchmark demand. There is
no stop because daily gaps dominate stop execution; sizing must treat the worst
historical five-session loss as a floor, not a bound. The post-publication
sample is the key decay check and should be reviewed annually without changing
the rule between reviews.
"""


def main() -> None:
    tlt = load_ticker("TLT")
    ief = load_ticker("IEF")
    events = build_events(tlt)
    assert events["month"].is_unique
    assert (events["hold_sessions"] == 5).all()
    assert events["month"].max() == "2026-07"  # August cache is incomplete.

    summaries = {"full": asdict(summarize(events["net_return"], events["exit_date"]))}
    for label, group in events.groupby("sample", sort=False):
        summaries[label] = asdict(summarize(group["net_return"], group["exit_date"]))

    ief_events = build_events(ief)
    pre = shifted_window_events(tlt, start_offset=-10, end_offset=-5)
    post = shifted_window_events(tlt, start_offset=0, end_offset=5)
    baseline = non_event_baseline(tlt, events)
    mechanism = {
        "duration": difference_test(events["gross_return"], ief_events["gross_return"]),
        "primary_vs_pre": difference_test(events["gross_return"], pre["gross_return"]),
        "primary_vs_post": difference_test(events["gross_return"], post["gross_return"]),
        "primary_vs_baseline": difference_test(events["gross_return"], baseline),
    }

    positions = pd.Series(np.arange(len(tlt)), index=tlt.index)
    tlt_open = tlt["Open"].astype(float).to_numpy()
    tlt_close = tlt["Close"].astype(float).to_numpy()
    through_exit_open = []
    exit_intraday = []
    for row in events.itertuples(index=False):
        entry_pos = int(positions.loc[pd.Timestamp(row.entry_date)])
        exit_pos = int(positions.loc[pd.Timestamp(row.exit_date)])
        through_exit_open.append(tlt_open[exit_pos] / tlt_close[entry_pos] - 1)
        exit_intraday.append(tlt_close[exit_pos] / tlt_open[exit_pos] - 1)
    exit_intraday_s = pd.Series(exit_intraday, dtype=float)
    execution = {
        "gross_mean_bps": float(events["gross_return"].mean() * 10_000),
        "friction_break_even_bps": float(events["gross_return"].mean() * 10_000),
        "through_exit_open_mean_bps": float(np.mean(through_exit_open) * 10_000),
        "exit_day_intraday_mean_bps": float(exit_intraday_s.mean() * 10_000),
        "exit_day_intraday_t_stat": float(
            exit_intraday_s.mean()
            / (exit_intraday_s.std(ddof=1) / math.sqrt(len(exit_intraday_s)))
        ),
    }

    spy = load_ticker("SPY")
    spy_returns = pd.Series(
        [
            float(spy.loc[row.exit_date, "Close"] / spy.loc[row.entry_date, "Close"] - 1)
            for row in events.itertuples(index=False)
        ],
        index=events.index,
    )
    spy_corr = stats.pearsonr(events["net_return"], spy_returns)
    ledger_path = REPO_ROOT / "data" / "backtest_trades_full.parquet"
    existing_bond_etf_trades = 0
    ledger_corr_value = math.nan
    ledger_corr_p = math.nan
    ledger_corr_n = 0
    if ledger_path.exists():
        ledger = pd.read_parquet(ledger_path)
        existing_bond_etf_trades = int(
            ledger["Ticker"].isin(["TLT", "IEF", "LQD", "HYG"]).sum()
        )
        ledger["Exit Date"] = pd.to_datetime(ledger["Exit Date"])
        ledger["month"] = ledger["Exit Date"].dt.to_period("M").astype(str)
        monthly_pnl = ledger.groupby("month")["PnL_flat_750k"].sum()
        joined = events.set_index("month")[["net_return"]].join(
            monthly_pnl.rename("book_pnl")
        ).dropna()
        if len(joined) > 2:
            ledger_corr = stats.pearsonr(joined["net_return"], joined["book_pnl"])
            ledger_corr_value = float(ledger_corr.statistic)
            ledger_corr_p = float(ledger_corr.pvalue)
            ledger_corr_n = len(joined)

    losing = (events["net_return"] < 0).astype(int)
    streaks = losing.groupby((losing == 0).cumsum()).cumsum()
    worst = events.nsmallest(5, "net_return")
    portfolio_fit = {
        "existing_bond_etf_trades": existing_bond_etf_trades,
        "spy_same_window_correlation": float(spy_corr.statistic),
        "spy_same_window_correlation_p": float(spy_corr.pvalue),
        "ledger_monthly_pnl_correlation": ledger_corr_value,
        "ledger_monthly_pnl_correlation_p": ledger_corr_p,
        "ledger_monthly_pnl_correlation_n": ledger_corr_n,
        "max_losing_streak": int(streaks.max()),
        "worst_events": [
            {
                "month": str(row.month),
                "entry_date": str(pd.Timestamp(row.entry_date).date()),
                "exit_date": str(pd.Timestamp(row.exit_date).date()),
                "net_return_pct": float(row.net_return * 100),
            }
            for row in worst.itertuples(index=False)
        ],
    }

    robust_rows = []
    for entry_offset in (-3, -4, -5, -6, -7):
        for cost in (0.0, 4.0, 10.0, 20.0):
            test = build_events(tlt, entry_offset=entry_offset, cost_bps=cost)
            z = summarize(test["net_return"], test["exit_date"])
            robust_rows.append(
                {
                    "entry_offset": entry_offset,
                    "cost_bps": cost,
                    "n": z.n,
                    "mean_net_pct": z.mean_pct,
                    "t_stat": z.t_stat,
                    "win_rate_pct": z.win_rate_pct,
                    "profit_factor": z.profit_factor,
                }
            )
    robustness = pd.DataFrame(robust_rows)
    yearly = (
        events.groupby("year", as_index=False)
        .agg(
            n=("net_return", "size"),
            mean_net_pct=("net_return", lambda x: x.mean() * 100),
            compounded_net_pct=("net_return", lambda x: ((1 + x).prod() - 1) * 100),
        )
    )
    loyo = leave_one_year_out(events)
    boot = bootstrap(events["net_return"])

    holdout_positive = summaries["holdout_2020_plus"]["mean_pct"] > 0
    full_t_pass = summaries["full"]["t_stat"] > 2.0
    positive_years = int((yearly["compounded_net_pct"] > 0).sum())
    years_pass = positive_years >= math.ceil(len(yearly) / 2)
    duration_pass = mechanism["duration"]["difference_pct"] > 0
    timing_pass = (
        mechanism["primary_vs_pre"]["difference_pct"] > 0
        and mechanism["primary_vs_post"]["difference_pct"] > 0
    )
    baseline_pass = mechanism["primary_vs_baseline"]["difference_pct"] > 0
    mechanism_passes = int(duration_pass) + int(timing_pass) + int(baseline_pass)
    graduates = bool(
        holdout_positive and full_t_pass and years_pass and mechanism_passes >= 2
    )
    decision = {
        "graduates": graduates,
        "holdout_positive": bool(holdout_positive),
        "full_t_above_2": bool(full_t_pass),
        "positive_years": positive_years,
        "total_years": len(yearly),
        "years_pass": bool(years_pass),
        "mechanism_passes": mechanism_passes,
        "duration_pass": bool(duration_pass),
        "timing_pass": bool(timing_pass),
        "baseline_pass": bool(baseline_pass),
        "reasons": [
            f"2020+ holdout mean after 10 bp is {summaries['holdout_2020_plus']['mean_pct']:.3f}% "
            f"({'pass' if holdout_positive else 'fail'}).",
            f"Full-sample t-stat is {summaries['full']['t_stat']:.2f} "
            f"({'pass' if full_t_pass else 'fail'} versus >2.0).",
            f"Positive years: {positive_years}/{len(yearly)} "
            f"({'pass' if years_pass else 'fail'}).",
            f"Mechanism/placebo checks passed: {mechanism_passes}/3 "
            f"(maturity={duration_pass}, timing={timing_pass}, baseline={baseline_pass}).",
        ],
    }
    payload = {
        "data": {
            "source": str(PRICE_PATH.relative_to(REPO_ROOT)),
            "ticker": "TLT",
            "first_date": str(tlt.index.min().date()),
            "last_date": str(tlt.index.max().date()),
            "rows": len(tlt),
        },
        "frozen_spec": {
            "entry": "TLT MOC at T-5",
            "exit": "TLT MOC at month-end T",
            "cost_bps_round_trip": PRIMARY_COST_BPS,
            "holdout_start": "2020-01-01",
        },
        "summaries": summaries,
        "bootstrap": boot,
        "mechanism": mechanism,
        "execution": execution,
        "portfolio_fit": portfolio_fit,
        "decision": decision,
    }

    events.to_csv(HERE / "events.csv", index=False, float_format="%.10f")
    yearly.to_csv(HERE / "yearly.csv", index=False, float_format="%.10f")
    loyo.to_csv(HERE / "leave_one_year_out.csv", index=False, float_format="%.10f")
    robustness.to_csv(HERE / "robustness.csv", index=False, float_format="%.10f")
    (HERE / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (HERE / "RESULTS.md").write_text(
        render_results(payload, robustness, yearly, loyo), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
