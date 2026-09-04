"""Intraday book excursion study (2026-07-28): per-day worst/best intraday
book mark from daily OHLC of open positions, flat $750k basis.

Method: for each day and each position open on it, mark the position at its
worst intraday print (Low for longs, High for shorts) vs its reference
(entry price on entry day, prior close after), and sum across positions.
Close marks use the same refs, so they sum to the daily MTM convention.

Bounds, stated up front:
- Per-ticker extremes are NOT simultaneous -> the summed trough is a
  PESSIMISTIC bound on the true book trough (tightest on correlated selloff
  days, overstated on mixed days).
- Entry days are near-tight for limit entries (a long limit fills at first
  touch, lower lows are post-fill; symmetric for shorts). T+1-Open entries
  use the full day range.
- Intraday stop/target exits (minority; time exits go MOC) apply the full
  day range even though the position died intraday -> pessimistic.
- Drawups (best marks) carry the same caveats in the OPTIMISTIC direction
  plus entry-timing ambiguity — treat as an upper bound only.
"""
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import data_provider

NAV = 750_000.0
LEDGER = os.path.join(_ROOT, "data", "_r2_ledger_check.parquet")


def main():
    df = pd.read_parquet(LEDGER)
    tickers = sorted(set(df["Ticker"].astype(str).str.replace(".", "-", regex=False)))
    md = data_provider.get_history(tickers, start="2002-06-01")

    px = {}
    for t, f in md.items():
        if f is None or f.empty:
            continue
        g = f.copy()
        if isinstance(g.columns, pd.MultiIndex):
            g.columns = g.columns.get_level_values(0)
        g.columns = [str(c).capitalize() for c in g.columns]
        px[t] = g[["High", "Low", "Close"]]

    worst = defaultdict(float)   # date -> summed worst intraday mark
    best = defaultdict(float)    # date -> summed best intraday mark
    closes = defaultdict(float)  # date -> summed close mark
    skipped = 0
    best_clean = defaultdict(float)  # drawup lower bound: no entry-day credit,
                                     # stop/target exit days capped at the fill
    cols = ["Ticker", "Direction", "Entry Date", "Exit Date", "Entry Price",
            "Shares_flat", "PnL_flat_750k", "Exit Price", "Exit Type"]
    for tick, direction, en, ex, entry_px, shares, pnl_flat, exit_px, exit_ty \
            in df[cols].values:
        t = str(tick).replace(".", "-")
        p = px.get(t)
        if p is None or pd.isna(shares) or not shares:
            skipped += 1
            continue
        en, ex = pd.Timestamp(en), pd.Timestamp(ex)
        days = p.loc[en:ex]
        if days.empty:
            skipped += 1
            continue
        sign = -1.0 if str(direction) == "Short" else 1.0
        sh = float(shares)
        refs = days["Close"].shift(1)
        refs.iloc[0] = float(entry_px)
        trade_close_sum = 0.0
        last_d = days.index[-1]
        fill_capped = str(exit_ty) in ("Stop", "Target") and not pd.isna(exit_px)
        for i, (d, hi, lo, cl, ref) in enumerate(zip(
                days.index, days["High"].values, days["Low"].values,
                days["Close"].values, refs.values)):
            if pd.isna(ref) or pd.isna(lo) or pd.isna(hi):
                continue
            adverse = (lo - ref) if sign > 0 else (ref - hi)
            favorable = (hi - ref) if sign > 0 else (ref - lo)
            worst[d] += min(0.0, adverse * sh)
            best[d] += max(0.0, favorable * sh)
            # clean drawup: entry-day HOD/LOD may predate the fill -> no credit;
            # a stop/target exit leaves the position before the day's extreme ->
            # cap the credit at the realized fill move.
            fav_clean = favorable
            if i == 0:
                fav_clean = 0.0
            elif d == last_d and fill_capped:
                fav_clean = min(favorable, (float(exit_px) - ref) * sign)
            best_clean[d] += max(0.0, fav_clean * sh)
            mark = (cl - ref) * sign * sh
            closes[d] += mark
            trade_close_sum += mark
        # Exit-day reconciliation to the realized fill — the same convention
        # get_daily_mtm_series uses, so close marks sum to booked PnL.
        if not pd.isna(pnl_flat):
            closes[days.index[-1]] += float(pnl_flat) - trade_close_sum

    days = pd.DataFrame({
        "worst": pd.Series(worst), "best": pd.Series(best),
        "best_clean": pd.Series(best_clean).reindex(pd.Series(worst).index).fillna(0.0),
        "close": pd.Series(closes)}).sort_index()
    days = days[days.index >= "2003-01-01"]
    yrs = len(days) / 252.0
    print(f"{len(days)} position-days, {skipped} trades skipped (no prices)")
    print(f"note: 'worst' sums each position's own worst minute -> pessimistic bound\n")

    print("=== INTRADAY TROUGH FREQUENCY (flat $750k) ===")
    for pct in (1.0, 1.5, 2.0, 3.0, 4.0):
        thr = -pct / 100 * NAV
        m = days[days["worst"] <= thr]
        if not len(m):
            print(f"touched -{pct}% intraday: never")
            continue
        rec = m["close"] - m["worst"]
        fin_pos = (m["close"] > 0).mean() * 100
        fin_half = (m["close"] > thr / 2).mean() * 100
        fin_worse = (m["close"] <= thr).mean() * 100
        print(f"touched -{pct}% intraday: {len(m)}x ({len(m) / yrs:.1f}/yr) | "
              f"finish: median ${m['close'].median():,.0f}, "
              f"{fin_pos:.0f}% green, {fin_half:.0f}% recovered >half, "
              f"{fin_worse:.0f}% closed at/below the touch level | "
              f"median bounce off low ${rec.median():,.0f}")

    print("\n=== CLOSE vs TROUGH on -2% touch days ===")
    m2 = days[days["worst"] <= -0.02 * NAV].copy()
    if len(m2):
        m2["fin_bucket"] = pd.cut(m2["close"] / NAV * 100,
                                  [-99, -3, -2, -1, 0, 99],
                                  labels=["<-3%", "-3..-2%", "-2..-1%", "-1..0%", "green"])
        print(m2["fin_bucket"].value_counts().reindex(
            ["green", "-1..0%", "-2..-1%", "-3..-2%", "<-3%"]).to_string())
        w5 = m2.nsmallest(5, "worst")
        print("\ndeepest troughs:")
        for d, r in w5.iterrows():
            print(f"  {d.date()}  trough ${r['worst']:,.0f} ({r['worst'] / NAV * 100:.1f}%) "
                  f"-> close ${r['close']:,.0f} ({r['close'] / NAV * 100:.1f}%)")

    print("\n=== DRAWUPS: naive upper bound vs clean lower bound ===")
    print("(clean = no entry-day credit, stop/target exit days capped at fill;")
    print(" truth is between the two, closer to clean for the dip-buy book)")
    for pct in (1.5, 2.0, 3.0):
        thr = pct / 100 * NAV
        for label, col in (("naive", "best"), ("clean", "best_clean")):
            m = days[days[col] >= thr]
            if not len(m):
                print(f"  +{pct}% {label}: never")
                continue
            giveback = (m[col] - m["close"])
            print(f"  +{pct}% {label}: {len(m)}x ({len(m) / yrs:.1f}/yr) | "
                  f"median close ${m['close'].median():,.0f} | "
                  f"median giveback ${giveback.median():,.0f}")

    print("\n=== SANITY ===")
    print(f"sum of close marks ${days['close'].sum():,.0f} vs booked flat "
          f"${df['PnL_flat_750k'].sum():,.0f} (close-basis diff = stop/target "
          f"fill vs close-mark on exit days; expected small)")
    days.to_csv(os.path.join(_HERE, "intraday_excursion_days.csv"))
    print("wrote scratch/intraday_excursion_days.csv")


if __name__ == "__main__":
    main()
