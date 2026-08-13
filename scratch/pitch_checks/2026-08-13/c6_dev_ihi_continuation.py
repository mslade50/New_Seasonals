"""c6_dev - round 3 on the surviving direction: LONG IHI on the thrust-out-of-
a-base state (r21 >= 99 while >= 10% below the 52w high).

NOT the pitched direction. The pitched fade is dead (short -0.95%/h3,
-1.50%/h5, negative in 9 of 9 years). This develops the continuation so the
composer can decide with numbers.

1. horizon scan h=1..10 -> the recommended time_td comes from here
2. entry form: MOC vs a close-anchored LIMIT at k*ATR (Wilder-14), WHOLE
   variants with fill rates (never a marginal-fill decomposition)
3. exit: target/stop sensitivity, or the line on why time-only fits
4. loser paths for what_kills_it
5. tomorrow-specific: what lands inside the hold (vix_expiry 08-19, opex
   08-21, retail cluster 08-18..20) and any IHI top-holding earnings
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

TK = ["IHI", "XLV", "SPY"]
px = close_panel(TK).dropna()
raw = load_prices(TK)
ihi_df = raw["IHI"]
ihi, xlv, spy = ihi_df["Close"], raw["XLV"]["Close"], raw["SPY"]["Close"]

r21 = pct_rank(ihi, 21)
dd = ihi / ihi.rolling(252).max() - 1.0
m = ((r21 >= 99) & (dd <= -0.10)).reindex(px.index).fillna(False)
trig = px.index[m.values]

print("=== 1. HORIZON SCAN (episodes, min_gap 5) ===")
show(horizon_scan(px, trig, [("IHI", 1.0)], hs=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
                  min_gap=5), "long IHI outright")
show(horizon_scan(px, trig, [("IHI", 1.0), ("XLV", -0.995)],
                  hs=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10), min_gap=5),
     "IHI residual vs XLV (b=0.995)")

print("\n=== 2. ENTRY FORM: MOC vs close-anchored LIMIT at k*ATR, WHOLE variants ===")
atr = pd.Series(wilder_atr(ihi_df["High"], ihi_df["Low"], ihi_df["Close"], 14),
                index=ihi_df.index).reindex(px.index)
highp = ihi_df['High'].reindex(px.index)
lowp = ihi_df['Low'].reindex(px.index)
openp = ihi_df['Open'].reindex(px.index)
idx = px.index
pos = pd.Series(range(len(idx)), index=idx)
epi = declusters(trig, 5, idx)
H = 5
rows = []
# MOC variant: enter at close D+1, exit close D+1+H
moc = []
for d in epi:
    p = pos[d]
    if p + 1 + H >= len(idx):
        continue
    moc.append(ihi.iloc[p + 1 + H] / ihi.iloc[p + 1] - 1.0)
rows.append({"variant": "MOC at close D+1", "fill_rate_pct": 100.0,
             "n": len(moc), "mean_pct": round(100 * np.mean(moc), 3),
             "hit": round(100 * np.mean(np.array(moc) > 0), 1),
             "worst_pct": round(100 * np.min(moc), 2)})
# LIMIT variants: limit at close(D) - k*ATR(D), live D+1..D+1 (1 session) or D+1..D+2
for k in (0.10, 0.25, 0.50):
    for live in (1, 2):
        vals, fills = [], 0
        for d in epi:
            p = pos[d]
            if p + live + H >= len(idx):
                continue
            lim = ihi.iloc[p] - k * atr.iloc[p]
            fill_i = None
            for j in range(1, live + 1):
                if lowp.iloc[p + j] <= lim:
                    fill_i = p + j
                    break
            if fill_i is None:
                continue  # no fill = no trade (whole-variant comparison)
            fills += 1
            entry = min(lim, openp.iloc[fill_i])  # gap-through fills at open
            vals.append(ihi.iloc[fill_i + H] / entry - 1.0)
        rows.append({"variant": f"LIMIT close-{k}ATR, live {live}td",
                     "fill_rate_pct": round(100 * fills / len(epi), 1),
                     "n": len(vals),
                     "mean_pct": round(100 * np.mean(vals), 3) if vals else None,
                     "hit": round(100 * np.mean(np.array(vals) > 0), 1) if vals else None,
                     "worst_pct": round(100 * np.min(vals), 2) if vals else None})
print(pd.DataFrame(rows).to_string(index=False))
print("  NOTE: unfilled limits book NOTHING (whole-variant comparison). Expected")
print("  value per SIGNAL for a limit variant = mean_pct * fill_rate.")
for r in rows[1:]:
    if r["mean_pct"] is not None:
        print(f"   {r['variant']}: per-signal EV = "
              f"{r['mean_pct'] * r['fill_rate_pct'] / 100:+.3f}% vs MOC "
              f"{rows[0]['mean_pct']:+.3f}%")
print(f"  today's ATR14 = {atr.iloc[-1]:.3f} ({100*atr.iloc[-1]/ihi.iloc[-1]:.2f}% of "
      f"price {ihi.iloc[-1]:.2f}); close-0.25ATR = {ihi.iloc[-1]-0.25*atr.iloc[-1]:.2f}")

print("\n=== 3. EXIT: target / stop sensitivity on the MOC h=5 form ===")
rows = []
for tgt in (None, 1.0, 1.5, 2.0, 3.0):
    for stp in (None, 1.0, 1.5, 2.0):
        vals = []
        for d in epi:
            p = pos[d]
            if p + 1 + H >= len(idx):
                continue
            e = ihi.iloc[p + 1]
            a = atr.iloc[p + 1]
            out = None
            for j in range(2, H + 2):          # stops arm day 2 (book convention)
                hi, lo, cl = (highp.iloc[p + j], lowp.iloc[p + j],
                              ihi.iloc[p + j])
                if tgt and hi >= e + tgt * a and stp and lo <= e - stp * a:
                    out = -stp * a / e          # pessimistic: stop books
                    break
                if tgt and hi >= e + tgt * a:
                    out = tgt * a / e
                    break
                if stp and lo <= e - stp * a:
                    out = min(-stp * a, openp.iloc[p + j] - e) / e
                    break
            if out is None:
                out = ihi.iloc[p + 1 + H] / e - 1.0
            vals.append(out)
        rows.append({"tgt_atr": tgt, "stop_atr": stp, "n": len(vals),
                     "mean_pct": round(100 * np.mean(vals), 3),
                     "hit": round(100 * np.mean(np.array(vals) > 0), 1),
                     "worst_pct": round(100 * np.min(vals), 2)})
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== 4. LOSER PATHS (episode_paths, h=5, MOC) ===")
paths = episode_paths(px, epi, [("IHI", 1.0)], 5)
fin = paths[5]
losers = paths[fin <= 0]
print("all episodes, cumulative % by day:")
print((100 * paths).round(2).to_string())
print(f"\nlosers (N={len(losers)}):")
print((100 * losers).round(2).to_string())
print(f"\nworst single episode finish {100*fin.min():.2f}% on "
      f"{fin.idxmin().date()}; worst intra-hold trough "
      f"{100*paths.min().min():.2f}%; median trough across all episodes "
      f"{100*paths.min(axis=1).median():.2f}%")
print(f"day-1 sign predicts finish? corr = {paths[1].corr(fin):+.3f}; "
      f"of {int((paths[1] <= 0).sum())} episodes down on day 1, "
      f"{int(((paths[1] <= 0) & (fin > 0)).sum())} still finished up")

print("\n=== 5. TOMORROW-SPECIFIC: what lands inside an 08-13 MOC entry, h=5 (exit 08-20) ===")
ev = load_events()
w = ev[(ev["date"] >= "2026-08-13") & (ev["date"] <= "2026-08-25")]
print(w.to_string(index=False))
print("\n  event-in-hold historical split (opex/vix_expiry inside the window):")
for kinds in [("vix_expiry",), ("opex",), ("cpi", "ppi", "nfp")]:
    fl = event_in_window(epi, idx, 5, 1, kinds)
    r = vehicle_ret(px, [("IHI", 1.0)], 5)
    show([summarize(r.loc[epi[fl]].values, f"{'+'.join(kinds)} IN (N={int(fl.sum())})"),
          summarize(r.loc[epi[~fl]].values, f"{'+'.join(kinds)} OUT")])

print("\n  IHI top-holding earnings inside 08-13..08-21:")
try:
    ec = pd.read_parquet(Path(__file__).resolve().parents[3] / "data" /
                         "earnings_calendar.parquet")
    dcol = [c for c in ec.columns if "date" in c.lower()][0]
    tcol = [c for c in ec.columns if "tick" in c.lower() or c.lower() == "symbol"][0]
    ec[dcol] = pd.to_datetime(ec[dcol])
    hold = ["ABT", "ISRG", "BSX", "MDT", "SYK", "EW", "BDX", "ZBH", "DXCM", "RMD",
            "PODD", "STE", "COO", "HOLX", "BAX"]
    sel = ec[(ec[tcol].isin(hold)) & (ec[dcol] >= "2026-08-13")
             & (ec[dcol] <= "2026-08-21")]
    print(sel[[tcol, dcol]].to_string(index=False) if len(sel)
          else "    NONE of the 15 largest medtech names report inside the window.")
except Exception as e:  # noqa: BLE001
    print("    earnings parquet unavailable:", e)
