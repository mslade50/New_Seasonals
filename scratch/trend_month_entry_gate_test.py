"""Test a month-entry fragility gate on the production trend sleeve.

Decision rule under test
------------------------
At the prior month's final available risk-dial observation, compute the
10-session moving average of the 63-day dial.  If that value is above a
threshold (50 for the primary test), hold the trend sleeve in cash for the
following calendar month.

The trend model mirrors trend_sleeve.py: production 12-ETF universe, 12-1
momentum AND close above the 10-month moving average, inverse-63d-vol slots,
20% slot cap, and off slots in cash.  Signals are observed at month-end close
and traded at the next session's open.  Results include 5 bps per side on
target-weight turnover, including the extra exit/re-entry turnover caused by
the dial gate.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps


ROOT = Path(__file__).resolve().parents[1]
UNIVERSE = [
    "SPY", "QQQ", "IWM", "EFA", "EEM", "FXI", "VNQ",
    "GLD", "SLV", "DBC", "TLT", "LQD",
]
COST_PER_SIDE = 0.0005
WEIGHT_CAP = 0.20
VOL_FLOOR = 0.04
MIN_MONTHLY_CLOSES = 13


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    mp = pd.read_parquet(
        ROOT / "data" / "master_prices.parquet",
        columns=["ticker", "date", "Open", "Close"],
    )
    mp["date"] = pd.to_datetime(mp["date"])
    sub = mp[mp["ticker"].isin(UNIVERSE + ["^IRX"])]
    close = sub.pivot(index="date", columns="ticker", values="Close").sort_index()
    open_px = sub.pivot(index="date", columns="ticker", values="Open").sort_index()
    irx = close.pop("^IRX").ffill()
    open_px = open_px.drop(columns=["^IRX"])
    close = close[UNIVERSE].astype(float)
    open_px = open_px[UNIVERSE].astype(float)

    frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet").sort_index()
    frag.index = pd.to_datetime(frag.index).tz_localize(None).normalize()
    dial = frag["63d"].dropna().rolling(10, min_periods=1).mean()
    return close, open_px, irx, dial


def build_targets(close: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly = close.resample("ME").last()
    mom12_1 = monthly.shift(1) / monthly.shift(12) - 1.0
    ma10 = monthly - monthly.rolling(10).mean()
    eligible = monthly.notna().rolling(MIN_MONTHLY_CLOSES).count() >= MIN_MONTHLY_CLOSES
    signal = (mom12_1 > 0) & (ma10 > 0) & eligible

    vol63 = close.pct_change().rolling(63).std() * np.sqrt(252)
    vol_monthly = vol63.resample("ME").last().clip(lower=VOL_FLOOR)
    inv = (1.0 / vol_monthly).where(eligible, 0.0)
    slots = inv.div(inv.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    slots = slots.clip(upper=WEIGHT_CAP)
    targets = slots * signal.astype(float)
    return targets, signal


def monthly_inputs(
    close: pd.DataFrame,
    open_px: pd.DataFrame,
    irx: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    monthly_close = close.resample("ME").last()
    close_returns = monthly_close.pct_change()

    first_open = open_px.resample("ME").first()
    # Row M is the return from the first open of M to the first open of M+1.
    # It is earned by weights chosen at the prior month-end and labels month M.
    next_open_returns = first_open.shift(-1) / first_open - 1.0

    rf_signal_month = (irx.resample("ME").last() / 100.0).ffill() / 12.0
    rf_held_month = rf_signal_month.shift(1)
    return next_open_returns, close_returns, rf_held_month


def sleeve_returns(
    targets: pd.DataFrame,
    asset_returns: pd.DataFrame,
    rf_held_month: pd.Series,
    gate_on_signal_month: pd.Series | None = None,
) -> pd.Series:
    effective = targets.copy()
    if gate_on_signal_month is not None:
        gate = gate_on_signal_month.reindex(effective.index).fillna(False)
        effective = effective.mask(gate, 0.0)

    held = effective.shift(1).fillna(0.0)
    invested = held.sum(axis=1)
    asset_pnl = (held * asset_returns.reindex(held.index)).sum(axis=1)
    cash_pnl = (1.0 - invested) * rf_held_month.reindex(held.index).fillna(0.0)
    turnover = effective.sub(effective.shift(1)).abs().sum(axis=1).shift(1).fillna(0.0)
    return (asset_pnl + cash_pnl - COST_PER_SIDE * turnover).rename("return")


def summary(r: pd.Series, rf: pd.Series) -> dict[str, float]:
    r = r.dropna()
    curve = (1.0 + r).cumprod()
    years = len(r) / 12.0
    excess = r - rf.reindex(r.index).fillna(0.0)
    return {
        "N": len(r),
        "avg": r.mean(),
        "median": r.median(),
        "hit": (r > 0).mean(),
        "compound": curve.iloc[-1] - 1.0,
        "CAGR": curve.iloc[-1] ** (1.0 / years) - 1.0,
        "vol": r.std() * np.sqrt(12),
        "Sharpe": excess.mean() / r.std() * np.sqrt(12),
        "maxDD": (curve / curve.cummax() - 1.0).min(),
        "worst": r.min(),
        "best": r.max(),
    }


def episode_ids(periods: pd.PeriodIndex) -> np.ndarray:
    ords = periods.astype("int64")
    return np.cumsum(np.r_[True, np.diff(ords) != 1])


def conditional_table(
    r: pd.Series,
    score_entering: pd.Series,
    threshold: float,
) -> tuple[pd.DataFrame, float, float]:
    d = pd.concat([r.rename("return"), score_entering.rename("score")], axis=1).dropna()
    high = d[d["score"] > threshold]["return"]
    low = d[d["score"] <= threshold]["return"]
    welch = sps.ttest_ind(high, low, equal_var=False)
    one = sps.ttest_1samp(high, 0.0)
    rows = []
    for label, x in [(f">{threshold:g}", high), (f"<={threshold:g}", low)]:
        rows.append({
            "state": label,
            "N": len(x),
            "avg%": x.mean() * 100,
            "median%": x.median() * 100,
            "hit%": (x > 0).mean() * 100,
            "compound%": ((1 + x).prod() - 1) * 100,
            "worst%": x.min() * 100,
            "best%": x.max() * 100,
        })
    return pd.DataFrame(rows), float(welch.pvalue), float(one.pvalue)


def main() -> None:
    close, open_px, irx, dial = load_inputs()
    targets, signal = build_targets(close)
    next_open, close_to_close, rf = monthly_inputs(close, open_px, irx)

    dial_month_end = dial.groupby(pd.Grouper(freq="ME")).last()
    dial_month_end = dial_month_end.reindex(targets.index)
    score_entering = dial_month_end.shift(1).rename("score_entering")

    base_open = sleeve_returns(targets, next_open, rf)
    base_close = sleeve_returns(targets, close_to_close, rf)

    # A return month is usable only if the next month's first open exists.
    valid_open = next_open.notna().any(axis=1)
    base_open = base_open[valid_open.reindex(base_open.index).fillna(False)]
    score_entering = score_entering.reindex(base_open.index)
    rf = rf.reindex(base_open.index)

    print("PRIMARY: production 12-ETF sleeve, true next-open execution")
    table, p_diff, p_zero = conditional_table(base_open, score_entering, 50.0)
    print(table.round(3).to_string(index=False))
    print(f"Welch p(high vs low)={p_diff:.4f}; one-sample p(high vs 0)={p_zero:.4f}")

    joined = pd.concat([base_open, score_entering], axis=1).dropna()
    high = joined[joined["score_entering"] > 50].copy()
    high.index = high.index.to_period("M")
    high["episode"] = episode_ids(high.index)
    eps = high.groupby("episode")["return"].agg(
        months="size", avg="mean", compound=lambda x: (1 + x).prod() - 1,
    )
    print(f"High-dial month list ({len(high)} months / {len(eps)} episodes):")
    print(", ".join(str(p) for p in high.index))
    print("Episodes (months, avg%, compounded%):")
    print((eps.assign(avg=lambda x: x["avg"] * 100,
                      compound=lambda x: x["compound"] * 100)
           .round(3).to_string()))

    print("\nHigh-dial months by return year:")
    by_year = high.groupby(high.index.year)["return"].agg(N="size", avg="mean", total=lambda x: (1+x).prod()-1)
    print((by_year.assign(avg=lambda x: x["avg"] * 100,
                          total=lambda x: x["total"] * 100)
           .round(3).to_string()))
    print("Leave-one-high-year-out averages:")
    for year in sorted(high.index.year.unique()):
        x = high.loc[high.index.year != year, "return"]
        print(f"  drop {year}: {x.mean()*100:+.3f}%/mo (N={len(x)})")

    print("\nThreshold sensitivity (next-open conditional return):")
    sens_rows = []
    for threshold in [45.0, 50.0, 55.0, 60.0]:
        d = pd.concat([base_open, score_entering], axis=1).dropna()
        hi = d[d["score_entering"] > threshold]["return"]
        lo = d[d["score_entering"] <= threshold]["return"]
        test = sps.ttest_ind(hi, lo, equal_var=False)
        sens_rows.append({
            "threshold": threshold, "N_hi": len(hi),
            "hi_avg%": hi.mean() * 100, "lo_avg%": lo.mean() * 100,
            "gap_pp": (hi.mean() - lo.mean()) * 100,
            "hi_hit%": (hi > 0).mean() * 100, "p_diff": test.pvalue,
        })
    print(pd.DataFrame(sens_rows).round(3).to_string(index=False))

    print("\nSame-close robustness at threshold >50:")
    same = pd.concat([base_close.rename("return"), score_entering], axis=1).dropna()
    hi_c = same[same["score_entering"] > 50]["return"]
    lo_c = same[same["score_entering"] <= 50]["return"]
    print(f"N_hi={len(hi_c)} high avg={hi_c.mean()*100:+.3f}% vs low={lo_c.mean()*100:+.3f}%")

    gate_signal = dial_month_end > 50.0
    gated_open = sleeve_returns(targets, next_open, rf, gate_on_signal_month=gate_signal)
    gated_open = gated_open.reindex(base_open.index)
    common = pd.concat([base_open.rename("base"), gated_open.rename("gate"), rf], axis=1).dropna()
    common = common[score_entering.reindex(common.index).notna()]
    s_base = summary(common["base"], common[rf.name] if rf.name in common else rf)
    s_gate = summary(common["gate"], common[rf.name] if rf.name in common else rf)
    print("\nFull counterfactual from first available month-entry dial (extra turnover included):")
    rows = []
    for label, s in [("always run", s_base), ("cash when >50", s_gate)]:
        rows.append({
            "policy": label, "N": s["N"], "CAGR%": s["CAGR"]*100,
            "vol%": s["vol"]*100, "Sharpe": s["Sharpe"],
            "maxDD%": s["maxDD"]*100, "worst%": s["worst"]*100,
        })
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    hi_idx = joined.index[joined["score_entering"] > 50]
    cash_alt = rf.reindex(hi_idx)
    avoided = cash_alt - base_open.reindex(hi_idx)
    print(f"At the live 0.30x NAV sleeve scale, historical avg benefit of cash in a >50 month: "
          f"{avoided.mean()*0.30*100:+.3f}% NAV ({avoided.mean()*0.30*10_000:+.1f} bps).")

    current_score = dial.iloc[-1]
    oldest_in_window = dial.index[-10]
    raw_floor = 500.0 - (dial.iloc[-1] * 10.0 -
                         pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
                         .loc[oldest_in_window, "63d"])
    latest_target = targets.iloc[-1]
    latest_on = latest_target[latest_target > 0]
    print("\nCurrent state:")
    print(f"dial as of {dial.index[-1].date()}: {current_score:.3f} "
          f"({'ABOVE' if current_score > 50 else 'NOT ABOVE'} 50)")
    print(f"A missing next-session raw 63d reading would have to finish below "
          f"{raw_floor:.3f} to pull the 10-day mean back under 50.")
    print(f"trend target as of {targets.index[-1].date()}: {len(latest_on)} assets ON, "
          f"{latest_on.sum()*100:.2f}% of sleeve / {latest_on.sum()*30:.2f}% of account NAV")
    print("ON weights: " + ", ".join(f"{t} {w*100:.2f}%" for t, w in latest_on.items()))


if __name__ == "__main__":
    main()
