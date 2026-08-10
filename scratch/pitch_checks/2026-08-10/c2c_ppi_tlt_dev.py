"""C2 round 3 (dev) -- develop the one survivor: long TLT across the PPI session.

Round 1 killed the stated 52w-floor gate (a filter that does not filter:
+0.115% ungated -> +0.117% gated, N 286 -> 28). Round 2 cleared the four
obvious confounds. This script does the three things round 3 owes:

  1. EXECUTABLE FORM. Today is Mon 2026-08-10; CPI is Wed 08-12 and PPI is
     Thu 08-13. A pitch can enter MOC 08-10 or MOC 08-11 only. So the
     entry/exit grid around the print gets measured directly, not assumed.
  2. ENTRY FORM as WHOLE variants: MOC vs a close-anchored LIMIT at k ATR
     (fill rate + stats of the whole variant, never a marginal-fill split).
  3. LOSER PATHS: episode_paths on the losing episodes so the thesis has a
     concrete invalidation number.

Plus the two remaining hostile tests:
  4. Is the mechanism the PRINT or the mid-quarter REFUNDING AUCTION? (PPI
     lands right after the 10y/30y auctions.) Split refunding months.
  5. Book overlap: does the systematic book already hold duration?
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

TK = ["TLT", "IEF", "^TNX"]
raw = load_prices(TK)
px = close_panel(TK).dropna(subset=["TLT"])
idx = px.index
ev = load_events(["ppi", "cpi"])

ppi_pos, ppi_sess = [], []
for x in ev[ev.event == "ppi"]["date"]:
    p = int(idx.searchsorted(x, side="left"))
    if 0 < p < len(idx):
        ppi_pos.append(p)
        ppi_sess.append(idx[p])
ppi_pos = np.array(ppi_pos)
cpi_sess = set()
for x in ev[ev.event == "cpi"]["date"]:
    p = int(idx.searchsorted(x, side="left"))
    if 0 <= p < len(idx):
        cpi_sess.add(idx[p])

c = px["TLT"].values
allr1 = (px["TLT"].shift(-1) / px["TLT"] - 1.0).dropna()

print("this week: CPI", ev[(ev.event == "cpi") & (ev.date >= "2026-08-01")].head(1)
      ["date"].dt.date.tolist(), " PPI",
      ev[(ev.event == "ppi") & (ev.date >= "2026-08-01")].head(1)["date"].dt.date.tolist())

# ------------------------------------------------------------------ 1. grid
print("\n" + "=" * 92)
print("1. ENTRY/EXIT GRID around the PPI session (entry MOC p-k, exit MOC p+j)")
print("   TODAY'S EXECUTABLE ROWS: k=2 (enter Tue 08-11) and k=3 (enter Mon 08-10)")
print("=" * 92)
rows = []
for k in (1, 2, 3, 4):
    for j in (0, 1, 2):
        v, dts = [], []
        for p in ppi_pos:
            if p - k < 0 or p + j >= len(idx):
                continue
            v.append(c[p + j] / c[p - k] - 1.0)
            dts.append(idx[p])
        v = np.array(v)
        h = k + j
        base = (px["TLT"].shift(-h) / px["TLT"] - 1.0).dropna()
        s = summarize(v, f"enter p-{k} -> exit p+{j}  (hold {h}td)")
        s["edge_pp"] = round(s["mean_pct"] - 100 * base.mean(), 3)
        s["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
        s["boot"] = round(bootstrap_p_le0(v), 3)
        s["today"] = "<== ENTER MON" if (k, j) == (3, 0) else (
            "<== ENTER TUE" if (k, j) == (2, 0) else "")
        rows.append(s)
show(rows, "TLT long, PPI-session grid (N=286-287 each)")

# ------------------------------------------------------- 2. CPI-in-window
print("\n" + "=" * 92)
print("2. THIS WEEK'S SHAPE: CPI falls INSIDE the p-2 -> p window. Historically")
print("   PPI usually PRECEDES CPI, so most windows do not contain one. Split it.")
print("=" * 92)
for k, j in [(2, 0), (3, 0), (1, 0)]:
    v, has_cpi, dts = [], [], []
    for p in ppi_pos:
        if p - k < 0 or p + j >= len(idx):
            continue
        win = set(idx[p - k + 1: p + j + 1])
        v.append(c[p + j] / c[p - k] - 1.0)
        has_cpi.append(bool(win & cpi_sess))
        dts.append(idx[p])
    v, has_cpi = np.array(v), np.array(has_cpi)
    print(f"\n  enter p-{k} -> exit p+{j}:")
    for lbl, m in [("CPI inside the window (THIS WEEK)", has_cpi),
                   ("no CPI inside", ~has_cpi)]:
        if m.sum() == 0:
            continue
        s = summarize(v[m], lbl)
        w = int((v[m] > 0).sum())
        print(f"    {lbl:36s} N={s['n']:3d} mean {s['mean_pct']:+.3f}% "
              f"hit {s['hit']:5.1f}% t {s['t']:+5.2f} worst {s['worst_pct']:+6.2f}% "
              f"sign p {sign_test(w, int(m.sum())):.4f}")
        d = pd.DatetimeIndex(np.array(dts)[m])
        mid = d.year % 4 == 2
        if mid.sum():
            print(f"      midterm N={int(mid.sum())} {100*v[m][mid].mean():+.3f}%  "
                  f"non-midterm N={int((~mid).sum())} {100*v[m][~mid].mean():+.3f}%")

# ------------------------------------------------- 3. horizon scan (formal)
print("\n" + "=" * 92)
print("3. HORIZON SCAN off the pre-specified anchor (p-2, lag=1) -- pick h HERE")
print("=" * 92)
anc = pd.DatetimeIndex([idx[p - 2] for p in ppi_pos if p >= 2])
show(horizon_scan(px, anc, [("TLT", 1.0)], hs=(1, 2, 3, 4, 5, 7, 10),
                  min_gap=5), "TLT long, PPI anchor")

# ------------------------------------------------ 4. refunding-auction test
print("\n" + "=" * 92)
print("4. MECHANISM: the print, or the mid-quarter refunding auction?")
print("   Feb/May/Aug/Nov are refunding months (10y+30y auctions the 9th-12th).")
print("=" * 92)
v, dts = [], []
for p in ppi_pos:
    if p < 1:
        continue
    v.append(c[p] / c[p - 1] - 1.0)
    dts.append(idx[p])
v, d = np.array(v), pd.DatetimeIndex(dts)
ref = np.isin(d.month, [2, 5, 8, 11])
for lbl, m in [("refunding months (Feb/May/Aug/Nov)", ref),
               ("other 8 months", ~ref)]:
    s = summarize(v[m], lbl)
    w = int((v[m] > 0).sum())
    print(f"  {lbl:36s} N={s['n']:3d} mean {s['mean_pct']:+.3f}% "
          f"hit {s['hit']:5.1f}% t {s['t']:+5.2f} sign p "
          f"{sign_test(w, int(m.sum())):.4f}")
print("  -> if the edge lived only in refunding months it would be the auction,")
print("     not the print. August IS a refunding month, so today's trade sits in")
print("     whichever bucket wins.")

# --------------------------------------------- 5. entry form: MOC vs LIMIT
print("\n" + "=" * 92)
print("5. ENTRY FORM as WHOLE variants -- MOC vs close-anchored LIMIT at k ATR")
print("   (fill rate + stats of the ENTIRE variant; unfilled = 0, no marginal")
print("    decomposition -- registry rule)")
print("=" * 92)
tl = raw["TLT"].copy()
atr = wilder_atr(tl["High"], tl["Low"], tl["Close"], 14)
tl = tl.assign(atr=atr)
tl = tl.reindex(idx)
lo, hi, cl, op = tl["Low"].values, tl["High"].values, tl["Close"].values, tl["Open"].values
av = tl["atr"].values

for kmult in (0.0, 0.10, 0.25, 0.40):
    fills, rets = 0, []
    for p in ppi_pos:
        e = p - 2                      # entry session (MOC on p-2, or limit on p-1)
        if e < 1 or p >= len(idx) or np.isnan(av[e]):
            continue
        if kmult == 0.0:
            rets.append(cl[p] / cl[e] - 1.0)
            fills += 1
            continue
        lim = cl[e] - kmult * av[e]     # buy the dip, live the NEXT session only
        if lo[e + 1] <= lim:
            fill = min(lim, op[e + 1])
            rets.append(cl[p] / fill - 1.0)
            fills += 1
        else:
            rets.append(0.0)            # no fill = flat, counted in the variant
    r = np.array(rets)
    n = len(r)
    s = summarize(r, "")
    print(f"  {'MOC p-2' if kmult == 0 else f'LIMIT close-{kmult:.2f}ATR on p-1':26s}"
          f" fill {100*fills/n:5.1f}%  variant mean {s['mean_pct']:+.3f}%  "
          f"sd {s['sd_pct']:.3f}%  worst {s['worst_pct']:+.2f}%  "
          f"t {s['t']:+.2f}")
print("  (LIMIT variants pay for their better fills with a big unfilled share;")
print("   the whole-variant mean is what decides.)")

# ------------------------------------------------------- 6. loser paths
print("\n" + "=" * 92)
print("6. LOSER PATHS -- what invalidates the thesis, with a number")
print("=" * 92)
paths = episode_paths(px, anc, [("TLT", 1.0)], h=2, lag=1)
tot = paths[2]
losers = paths.loc[tot < 0]
print(f"episodes {len(paths)}, losers {len(losers)} ({100*len(losers)/len(paths):.1f}%)")
print(f"loser mean {100*tot[tot < 0].mean():+.3f}%   "
      f"winner mean {100*tot[tot > 0].mean():+.3f}%   "
      f"loser 5th pctile {100*np.percentile(tot[tot < 0], 5):+.2f}%")
print(f"day-1 (the CPI-equivalent session) of the eventual losers: mean "
      f"{100*losers[1].mean():+.3f}%; of the winners "
      f"{100*paths.loc[tot > 0, 1].mean():+.3f}%")
bad = losers[1] < -0.005
print(f"  P(final loss | day-1 already < -0.50%) = "
      f"{100*float((paths.loc[paths[1] < -0.005, 2] < 0).mean()):.1f}%  "
      f"(N={int((paths[1] < -0.005).sum())}) vs base "
      f"{100*len(losers)/len(paths):.1f}%")
print("worst 8 episodes:")
for dt in tot.nsmallest(8).index:
    print(f"  {dt.date()}  d1 {100*paths.loc[dt,1]:+.2f}%  "
          f"d2 {100*paths.loc[dt,2]:+.2f}%")

# ----------------------------------------------------- 7. book overlap
print("\n" + "=" * 92)
print("7. BOOK OVERLAP -- does the systematic book already hold duration?")
print("=" * 92)
sma200 = px["TLT"].rolling(200).mean()
mom = px["TLT"] / px["TLT"].shift(252) - 1.0
print(f"  TLT close {px['TLT'].iloc[-1]:.2f}  200d SMA {sma200.iloc[-1]:.2f}  "
      f"above? {bool(px['TLT'].iloc[-1] > sma200.iloc[-1])}")
print(f"  TLT 12-1 momentum {100*mom.iloc[-1]:+.2f}%")
print("  Trend sleeve requires 12-1 momentum > 0 AND above the 10-month MA ->"
      f" TLT slot is {'HELD' if (px['TLT'].iloc[-1] > sma200.iloc[-1] and mom.iloc[-1] > 0) else 'FLAT (cash)'}.")
tlt_state = Path(__file__).resolve().parents[3] / "data" / "trend_sleeve_state.json"
if tlt_state.exists():
    import json
    st = json.loads(tlt_state.read_text())
    print("  trend_sleeve_state.json holdings:", st.get("holdings", st))
