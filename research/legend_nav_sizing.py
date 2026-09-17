"""Read-only, event-driven test of native Legend long NAV allocations.

Run with --source-root pointing to the archive-owning checkout. --fetch permits
read-only IBKR historical requests and Yahoo raw daily/action downloads only.
No broker orders, runtime environment edits, or live sizing imports are made.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from legend_etf.core import normalize_minutes, prior_wilder_atr14, simulate_etf_trade
from legend_etf.etf_source import evaluate_etf_setup
from legend_etf.calendar import session_labels

TZ = "America/New_York"
WEIGHTS = {"SPY": .40, "QQQ": .30}
# The research interpreter's older calendar package omits this announced closure.
# https://ir.theice.com/press/news-details/2024/The-New-York-Stock-Exchange-Will-Close-Markets-on-January-9-to-Honor-the-Passing-of-Former-President-Jimmy-Carter-on-National-Day-of-Mourning/default.aspx
KNOWN_CLOSURES = pd.DatetimeIndex(["2025-01-09"])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_candidates(source):
    path = source / "artifacts/legend-spy-qqq-20260916/release_candidate_parity.candidates.csv"
    candidates = pd.read_csv(path)
    selected = []
    inputs = [{"path": str(path), "sha256": sha(path)}]
    for symbol in WEIGHTS:
        bars_path = source / f"data/intraday/{symbol}_15min.parquet"
        inputs.append({"path": str(bars_path), "sha256": sha(bars_path)})
        bars = normalize_minutes(pd.read_parquet(bars_path), timestamp_col="ts", naive_tz=TZ)
        bars.index = bars.index.tz_convert(TZ)
        for item in candidates.loc[candidates.etf.eq(symbol)].to_dict("records"):
            date = item["entry_date"]
            start = (pd.Timestamp(date) - pd.Timedelta(days=20)).tz_localize(TZ)
            sample = bars.loc[start:pd.Timestamp(date, tz=TZ)]
            actual = evaluate_etf_setup(sample, entry_date=date)
            if (not actual["qualifies"] or actual["history_sha256"] != item["history_sha256"]
                    or not np.isclose(actual["initial_ema"], item["initial_ema"], atol=1e-9, rtol=0)):
                raise ValueError(f"Candidate verification failed: {symbol} {date}")
            stamp = pd.Timestamp(f"{date} 09:30", tz=TZ)
            opened = float(bars.loc[stamp, "open"]) if stamp in bars.index else np.nan
            selected.append({**item, "archive_open": opened})
    return pd.DataFrame(selected).sort_values(["entry_date", "etf"]), inputs


def event_path(source, output, symbol, day):
    cached = source / "artifacts/legend-etf-execution-backtest-20260901/ibkr_1m_events" / {"SPY": "ES", "QQQ": "NQ"}[symbol] / f"{day}.parquet"
    return cached if cached.exists() else output / "ibkr_1m_events" / symbol / f"{day}.parquet"


async def fetch_events(candidates, source, output, port, client_id):
    from ib_insync import IB, Stock, util
    ib = IB()
    log = {"readonly": True, "results": []}
    await ib.connectAsync("127.0.0.1", port, clientId=client_id, readonly=True, timeout=20)
    try:
        contracts = {}
        for symbol in WEIGHTS:
            qualified = await ib.qualifyContractsAsync(Stock(symbol, "SMART", "USD"))
            if len(qualified) != 1:
                raise ValueError(f"Cannot qualify {symbol}")
            contracts[symbol] = qualified[0]
        todo = [r for r in candidates.to_dict("records") if not event_path(source, output, r["etf"], r["entry_date"]).exists()]
        for number, row in enumerate(todo, 1):
            symbol, day = row["etf"], row["entry_date"]
            path = event_path(source, output, symbol, day)
            result = {"etf": symbol, "entry_date": day}
            try:
                bars = await ib.reqHistoricalDataAsync(contracts[symbol],
                    endDateTime=f"{day.replace('-', '')} 16:00:00 America/New_York",
                    durationStr="1 D", barSizeSetting="1 min", whatToShow="TRADES",
                    useRTH=True, formatDate=2, keepUpToDate=False, timeout=60)
                if not bars:
                    raise ValueError("empty historical response")
                frame = util.df(bars).set_index("date")
                frame.index = pd.to_datetime(frame.index, utc=True)
                frame = normalize_minutes(frame)
                path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_parquet(path)
                result.update(status="ok", rows=len(frame))
            except Exception as exc:
                result.update(status="error", error=str(exc))
            log["results"].append(result)
            (output / "fetch_log.json").write_text(json.dumps(log, indent=2))
            print(f"Historical minute sessions {number}/{len(todo)}: {symbol} {day} {result['status']}", flush=True)
            await asyncio.sleep(.4)
    finally:
        ib.disconnect()


def daily_data(output, fetch):
    import yfinance as yf
    yf.set_tz_cache_location(str(output / "yf_cache"))
    result = {}
    for symbol in WEIGHTS:
        path = output / f"{symbol}_raw_daily.parquet"
        if not path.exists() and fetch:
            data = yf.Ticker(symbol).history(start="2010-01-01", end="2026-08-29", auto_adjust=False, actions=True)
            if data.empty:
                raise ValueError(f"No daily data for {symbol}")
            data.columns = [c.lower().replace(" ", "_") for c in data.columns]
            data.to_parquet(path)
        data = pd.read_parquet(path)
        data.index = pd.DatetimeIndex(data.index).tz_convert(TZ).tz_localize(None).normalize()
        result[symbol] = data
    return result


def return_paths(minutes, trade):
    """Close marks and conservative minute-low marks, stopping at actual exit.

    An intrabar limit exit may follow that minute's low, so the lower path
    includes it. An opening fill or 10:30 time exit excludes later lows.
    """
    data = normalize_minutes(minutes)
    data.index = data.index.tz_convert(TZ)
    entry, exit_ = pd.Timestamp(trade["entry_ts"]), pd.Timestamp(trade["exit_ts"])
    grid = pd.date_range(entry, entry.normalize() + pd.Timedelta(hours=10, minutes=30), freq="1min")
    price, final = float(trade["entry_price"]), float(trade["exit_price"])
    close = pd.Series(final / price - 1, index=grid)
    lower = close.copy()
    held = grid[grid < exit_]
    close.loc[held] = data.loc[held, "close"] / price - 1
    lower.loc[held] = data.loc[held, "low"] / price - 1
    if trade["exit_reason"] == "etf_ema_limit" and float(data.loc[exit_, "open"]) < final:
        lower.loc[exit_] = min(float(data.loc[exit_, "low"]) / price - 1, final / price - 1)
    return close, lower


def research_atr(daily, entry_date, expected_last):
    prior = daily.loc[daily.index < pd.Timestamp(entry_date)]
    expected = session_labels(prior.index[0], expected_last).difference(KNOWN_CLOSURES)
    if not prior.index.equals(expected):
        raise ValueError("Research daily history has missing or unexpected sessions")
    # We validated the corrected grid above; preserve production ATR arithmetic.
    return prior_wilder_atr14(prior, as_of_date=entry_date)


def portfolio_paths(trades, paths, weight_column, cost_bps):
    rows = []
    nav = peak = intraday_peak = 1.0
    worst_intraday_dd = 0.0
    for day, group in trades.groupby("entry_date", sort=True):
        # Each ETF gets its own fraction of the SAME opening NAV. Unused weight is cash.
        weights = group[weight_column].to_numpy(float)
        cost = float(weights.sum() * cost_bps / 10000)
        ret = float(np.dot(weights, group.gross_return_bps.to_numpy(float)) / 10000 - cost)
        close = sum(paths[int(i)][0] * w for i, w in zip(group.index, weights)) - cost
        lower = sum(paths[int(i)][1] * w for i, w in zip(group.index, weights)) - cost
        minimum = float(min(0, lower.min()))
        for close_mark, low_mark in zip(close, lower):
            # Only previously observed closing marks enter the running peak.
            worst_intraday_dd = min(worst_intraday_dd, nav * (1 + low_mark) / intraday_peak - 1)
            intraday_peak = max(intraday_peak, nav * (1 + close_mark))
        nav *= 1 + ret
        peak = max(peak, nav)
        rows.append({"date": day, "trades": len(group), "weight": weights.sum(),
                     "return": ret, "nav": nav, "closed_drawdown": nav / peak - 1,
                     "intraday_loss_bound": minimum})
    days = pd.DataFrame(rows)
    returns = days["return"]
    wins, losses = returns[returns > 0].sum(), -returns[returns < 0].sum()
    summary = {"trades": len(trades), "active_days": len(days), "both_days": int(days.trades.eq(2).sum()),
               "total_return_pct": (nav - 1) * 100,
               "closed_max_drawdown_pct": days.closed_drawdown.min() * 100,
               "intraday_drawdown_proxy_pct": worst_intraday_dd * 100,
               "worst_day_pct": returns.min() * 100,
               "worst_day": days.loc[returns.idxmin(), "date"],
               "worst_intraday_loss_bound_pct": days.intraday_loss_bound.min() * 100,
               "win_day_pct": returns.gt(0).mean() * 100,
               "profit_factor_days": wins / losses if losses else None,
               "mean_active_day_bps": returns.mean() * 10000,
               "without_best_day_return_pct": ((1 + returns.drop(returns.idxmax())).prod() - 1) * 100,
               "without_best_two_days_return_pct": ((1 + returns.drop(returns.nlargest(2).index)).prod() - 1) * 100}
    return days, summary


def write_report(output, payload):
    stats = payload["results"]
    rows = ["# Legend long sizing: SPY 40% / QQQ 30% NAV", "",
            "Research replay dated 2026-09-16. Signal sample: 2012-01-01 through 2026-08-28. No live settings changed.", "",
            "Each ETF receives its allocation only when it independently qualifies for a long entry. Both together use 70% of opening NAV; unused allocation stays in cash. Returns below are cumulative contributions to standalone NAV over the full sample, not annual returns.", "",
            "## Main comparison (2 bp assumed all-in round-trip cost)", "",
            "| Metric | Inherited ATR sizing | SPY 40% / QQQ 30% |", "|---|---:|---:|"]
    base, proposed = stats["baseline_2bps"], stats["proposed_2bps"]
    for label, key in [("Trades", "trades"), ("Active dates", "active_days"), ("Both ETFs trade", "both_days"),
                       ("Total NAV return (%)", "total_return_pct"), ("Closed-day maximum drawdown (%)", "closed_max_drawdown_pct"),
                       ("Worst closed day (%)", "worst_day_pct"), ("Worst intraday loss envelope (%)", "worst_intraday_loss_bound_pct"),
                       ("Intraday drawdown proxy (%)", "intraday_drawdown_proxy_pct"), ("Winning dates (%)", "win_day_pct"),
                       ("Daily profit factor", "profit_factor_days"), ("Return excluding best date (%)", "without_best_day_return_pct")]:
        rows.append(f"| {label} | {base[key]:.3f} | {proposed[key]:.3f} |")
    rows += ["", f"The full-sample result remains positive after removing the two best dates: {proposed['without_best_two_days_return_pct']:.2f}% total NAV return. However, the 2022-2026 result excluding its best date is only {stats['2022_2026']['without_best_day_return_pct']:.3f}%. This is a promising but sparse and concentrated historical result; it does not establish that 40/30 is optimal, that higher sizing is warranted, or that the measured drawdowns bound future losses."]
    rows += ["", "## ETF attribution", "", "| ETF | Trades | Mean net trade return (bp of notional) | Winning trades | NAV contribution, compounded alone | Mean inherited allocation |",
             "|---|---:|---:|---:|---:|---:|"]
    for symbol in WEIGHTS:
        x = stats[symbol]
        rows.append(f"| {symbol} | {x['trades']} | {x['average_trade_net_bps']:.2f} | {x['win_trade_pct']:.1f}% | {x['total_return_pct']:.2f}% | {x['mean_baseline_weight_pct']:.2f}% |")
    rows += ["", "ETF-alone compounded contributions do not sum exactly to the combined compounded return.", "",
             "## Cost and period sensitivity", "", "| Round-trip cost (bp of notional) | 40/30 total NAV return |", "|---|---:|"]
    for cost in (0, 1, 2, 5, 10):
        rows.append(f"| {cost} | {stats[f'proposed_{cost}bps']['total_return_pct']:.2f}% |")
    rows += ["", "| Period | Trades | 40/30 NAV return after 2 bp |", "|---|---:|---:|"]
    for era in ("2012_2019", "2020_2021", "2022_2026"):
        rows.append(f"| {era.replace('_', '-')} | {stats[era]['trades']} | {stats[era]['total_return_pct']:.2f}% |")
    interval = stats["date_bootstrap"]["mean_active_day_bps_95pct_interval"]
    rows += ["", f"Date-cluster bootstrap: mean active-day NAV return {proposed['mean_active_day_bps']:.2f} bp, with an illustrative 95% interval [{interval[0]:.2f}, {interval[1]:.2f}] bp. Both ETFs stay paired when resampling. Assumes independent dates; excludes regime persistence and research-selection bias.", "",
             "## Coverage and limitations", ""]
    exclusions = pd.DataFrame(payload["exclusions"])
    rows += [f"Verified {payload['verified_setups']} setups against the current native signal code and the archived signal hashes; admitted {proposed['trades']} long trades. Exclusions: " + ", ".join(f"{reason}={count}" for reason, count in exclusions.reason.value_counts().items()) + ".", ""]
    rows.extend(f"- {item}" for item in payload["assumptions"])
    rows += ["", "The local ATR calendar correction follows the [NYSE operator's closure notice](https://ir.theice.com/press/news-details/2024/The-New-York-Stock-Exchange-Will-Close-Markets-on-January-9-to-Honor-the-Passing-of-Former-President-Jimmy-Carter-on-National-Day-of-Mourning/default.aspx).", "", "A simultaneous 1% decline in both ETFs would cost approximately 0.70% of NAV at full 40/30 allocation, before costs. This is an illustrative stress, not an estimated probability or a maximum loss.", "",
             "## Reproduction", "", "Run `python research/legend_nav_sizing.py --source-root <archive-owning-checkout> --output <artifact-directory>`; add `--fetch` to populate missing historical inputs using read-only broker requests. `results.json` records input and source SHA-256 hashes. `trades.csv` records every simulated fill, exit, ATR, weight, and minute-low adverse-excursion envelope; `exclusions.csv`, `yearly.csv`, and sizing/cost daily CSVs provide the accounting audit trail.", ""]
    (output / "REPORT.md").write_text("\n".join(rows), encoding="utf-8")


def run(args):
    source, output = args.source_root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    candidates, inputs = load_candidates(source)
    print(f"Verified {len(candidates)} archived setups; minute data will determine long eligibility", flush=True)
    candidates.to_csv(output / "verified_candidates.csv", index=False)
    if args.fetch:
        asyncio.run(fetch_events(candidates, source, output, args.port, args.client_id))
    daily = daily_data(output, args.fetch)
    records, exclusions, paths = [], [], {}
    for row in candidates.to_dict("records"):
        symbol, day = row["etf"], row["entry_date"]
        path = event_path(source, output, symbol, day)
        if not path.exists():
            exclusions.append({"etf": symbol, "date": day, "reason": "missing_minutes"})
            continue
        inputs.append({"path": str(path), "sha256": sha(path)})
        minutes = pd.read_parquet(path)
        local = normalize_minutes(minutes)
        local.index = local.index.tz_convert(TZ)
        open_stamp = pd.Timestamp(f"{day} 09:30", tz=TZ)
        if open_stamp in local.index and float(local.loc[open_stamp, "open"]) >= row["initial_ema"]:
            exclusions.append({"etf": symbol, "date": day, "reason": "not_long"})
            continue
        dividends = daily[symbol]["dividends"]
        ex_dividend = bool(dividends.get(pd.Timestamp(day), 0) > 0)
        trade = simulate_etf_trade(minutes, entry_date=day, initial_ema=row["initial_ema"], ex_dividend=ex_dividend)
        if not trade["traded"]:
            exclusions.append({"etf": symbol, "date": day, "reason": trade["skip_reason"]})
            continue
        if trade["direction"] != 1:
            raise ValueError(f"Minute/15min opening side disagreement: {symbol} {day}")
        atr = research_atr(daily[symbol], day, row["setup_date"])
        base_weight = min(.25, .001 * trade["entry_price"] / (1.25 * atr))
        close, lower = return_paths(minutes, trade)
        paths[len(records)] = close, lower
        records.append({**row, **trade, "atr14": atr, "proposed_weight": WEIGHTS[symbol],
                        "baseline_weight": base_weight, "mae_bound_bps": min(0, lower.min()) * 10000})
    trades = pd.DataFrame(records)
    if trades.empty:
        raise ValueError("No trades available")
    trades.to_csv(output / "trades.csv", index=False)
    pd.DataFrame(exclusions).to_csv(output / "exclusions.csv", index=False)
    results = {}
    for sizing in ("baseline", "proposed"):
        for cost in (0, 1, 2, 5, 10):
            days, summary = portfolio_paths(trades, paths, f"{sizing}_weight", cost)
            results[f"{sizing}_{cost}bps"] = summary
            days.to_csv(output / f"{sizing}_{cost}bps_days.csv", index=False)
    primary = pd.read_csv(output / "proposed_2bps_days.csv")
    rng = np.random.default_rng(4030)
    # Dates, not ETF trades, are the resampling unit; overlapping ETF returns stay paired.
    means = rng.choice(primary["return"].to_numpy(), size=(10000, len(primary)), replace=True).mean(axis=1)
    results["date_bootstrap"] = {"mean_active_day_bps_95pct_interval": (np.quantile(means, [.025, .975]) * 10000).tolist(),
                                "replicates": 10000, "seed": 4030,
                                "limitation": "IID dates; does not account for regime persistence, selection or specification search."}
    for symbol in WEIGHTS:
        subset = trades.loc[trades.etf.eq(symbol)]
        _, results[symbol] = portfolio_paths(subset, paths, "proposed_weight", 2)
        results[symbol]["average_trade_net_bps"] = float(subset.gross_return_bps.mean() - 2)
        results[symbol]["win_trade_pct"] = float(subset.gross_return_bps.gt(2).mean() * 100)
        results[symbol]["mean_baseline_weight_pct"] = float(subset.baseline_weight.mean() * 100)
        results[symbol]["worst_trade_mae_bound_bps"] = float(subset.mae_bound_bps.min())
        results[symbol]["median_hold_minutes"] = float(((pd.to_datetime(subset.exit_ts, utc=True) - pd.to_datetime(subset.entry_ts, utc=True)).dt.total_seconds() / 60).median())
    years = []
    for year, subset in trades.groupby(trades.entry_date.str[:4]):
        _, stats = portfolio_paths(subset, paths, "proposed_weight", 2)
        years.append({"year": year, **stats})
    pd.DataFrame(years).to_csv(output / "yearly.csv", index=False)
    for label, start, end in (("2012_2019", "2012", "2019-12-31"), ("2020_2021", "2020", "2021-12-31"), ("2022_2026", "2022", "2026-12-31")):
        subset = trades.loc[trades.entry_date.ge(start) & trades.entry_date.le(end)]
        _, results[label] = portfolio_paths(subset, paths, "proposed_weight", 2)
    payload = {"period": ["2012-01-01", "2026-08-28"], "verified_setups": len(candidates),
               "exclusions": exclusions, "results": results,
               "assumptions": ["Long signals only; SPY 40%, QQQ 30%, no redistribution, fractions of same start-of-day NAV.",
                 "Cost is assumed all-in ROUND-TRIP basis points of traded notional; charged at entry for drawdown accounting.",
                 "No stops; production 09:31 entry, dynamic EMA limit exits, 10:30 time exit.",
                 "Daily return compounds at sleeve NAV; idle cash earns zero. Other portfolio positions are excluded.",
                 "Baseline: 10bps ATR risk and 25% notional cap; fractional shares, no 5000-share or external capacity caps.",
                 "Intraday loss uses a conservative synchronized minute-low envelope. Intraday drawdown proxy measures those lows against earlier CLOSE-mark peaks; it is not a bound on tick-exact maximum drawdown.",
                 "Historical sample omits 320 symbol-days with incomplete archived 15min signal history; not an out-of-sample validation.",
                 "ATR comparator validates the daily calendar with the announced 2025-01-09 market closure corrected locally; the archived signal sample itself is unchanged.",
                 "One-minute touches assume fills and no operational delays; costs do not model missed fills or outages."],
               "input_files": inputs + [{"path": str(output / f"{s}_raw_daily.parquet"), "sha256": sha(output / f"{s}_raw_daily.parquet")} for s in WEIGHTS],
               "source_hashes": {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), ROOT / "legend_etf/core.py", ROOT / "legend_etf/etf_source.py"]}}
    (output / "results.json").write_text(json.dumps(payload, indent=2, default=str))
    write_report(output, payload)
    print(json.dumps({"exclusions": pd.DataFrame(exclusions).reason.value_counts().to_dict(), "results": results}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/legend-sizing")
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--port", type=int, default=7496)
    parser.add_argument("--client-id", type=int, default=186)
    run(parser.parse_args())
