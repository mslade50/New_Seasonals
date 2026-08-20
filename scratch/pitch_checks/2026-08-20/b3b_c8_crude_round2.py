"""C8 round 2. Round 1 left the h=10 cell alive: USO +2.145% over 20 with the
true JH-6 anchor ranking 1 of 16, CL=F +1.628% over 25 also ranking 1 of 16,
and both beating a matched August tdom-band control by ~+1.5pp.

This round attacks (a) the trading-day-of-month decomposition the round-1
control could not resolve, (b) the midterm conditioner against a midterm
control, (c) LOYO and drop-best, (d) the LIVE price state, (e) multiplicity
across the vehicle x horizon grid, (f) what the book already does here.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
warnings.filterwarnings("ignore")
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

px = close_panel(["USO", "XLE", "CL=F", "XOP", "SPY"])
d = px.index
jh = pd.DatetimeIndex(sorted(set(load_events(["jackson_hole"])["date"])
                             & set(d)))
jh = jh[jh <= d[-1]]
OFF = -6


def anchor_at(off):
    pos = d.get_indexer(jh) + off
    pos = pos[(pos >= 0) & (pos < len(d))]
    return d[pos]


entry = anchor_at(OFF)
tmap = {}
for y in sorted(set(d[d.month == 8].year)):
    m = d[(d.year == y) & (d.month == 8)]
    for i, x in enumerate(m):
        tmap[x] = i + 1
anchor_tdom = pd.Series([tmap[x] for x in entry], index=entry)
print("Anchor tdom by year:")
print(anchor_tdom.groupby(anchor_tdom.index.year).first().to_string())

# ------------------------------------------------------ tdom decomposition
print("\n\n" + "=" * 78)
print("1. TRADING-DAY-OF-MONTH DECOMPOSITION. Is the JH-6 anchor a tdom "
      "slice? Every August start, by tdom, no event involved.")
print("=" * 78)
aug = d[d.month == 8]
for tkr in ("CL=F", "USO"):
    for h in (6, 10):
        rows = []
        for t in range(1, 23):
            days = pd.DatetimeIndex([x for x in aug if tmap[x] == t])
            s = fwd_lag(px[tkr], h, lag=0).reindex(days).dropna()
            if len(s) < 8:
                continue
            rows.append({"tdom": t, "n": len(s), "mean_pct": 100 * s.mean(),
                         "hit": 100 * (s > 0).mean(),
                         "n_anchors": int((anchor_tdom == t).sum())})
        df = pd.DataFrame(rows)
        print(f"\n--- {tkr} h={h}: August forward return by tdom "
              f"(unconditional) ---")
        print(df.round(3).to_string(index=False))
        # anchor-weighted tdom expectation: weight each tdom by how often the
        # anchor lands there. This is the RIGHT control.
        w = anchor_tdom.value_counts()
        w = w / w.sum()
        exp = sum(df[df.tdom == t].mean_pct.iloc[0] * wt
                  for t, wt in w.items() if (df.tdom == t).any())
        obs = 100 * fwd_lag(px[tkr], h, lag=0).reindex(entry).dropna().mean()
        print(f"  ANCHOR-TDOM-WEIGHTED unconditional expectation: "
              f"{exp:+.3f}%   observed anchor {obs:+.3f}%   "
              f"EXCESS {obs-exp:+.3f}pp")

# ------------------------------------------------------------ midterm
print("\n\n" + "=" * 78)
print("2. MIDTERM against a MIDTERM-restricted August control (2026 is "
      "midterm)")
print("=" * 78)
lo, hi = int(anchor_tdom.min()), int(anchor_tdom.max())
band = pd.DatetimeIndex([x for x in aug if lo <= tmap[x] <= hi])
for tkr in ("USO", "CL=F", "XLE"):
    for h in (6, 10):
        s = fwd_lag(px[tkr], h, lag=0)
        e = s.reindex(entry).dropna()
        b = s.reindex(band).dropna()
        em, bm = e[e.index.year % 4 == 2], b[b.index.year % 4 == 2]
        wins = int((em > 0).sum())
        print(f"  {tkr:6s} h={h:2d}  MIDTERM anchor {100*em.mean():+.3f}% "
              f"(N={len(em)}, hit {100*(em>0).mean():.0f}%, sign p "
              f"{sign_test(wins, len(em)):.3f})  |  midterm Aug tdom "
              f"{lo}-{hi} {100*bm.mean():+.3f}% (N={len(bm)})  ->  EXCESS "
              f"{100*(em.mean()-bm.mean()):+.3f}pp")
    print()

# ------------------------------------------------------------ LOYO / drop
print("\n" + "=" * 78)
print("3. LOYO and drop-best against the matched control")
print("=" * 78)
for tkr in ("USO", "CL=F"):
    for h in (6, 10):
        s = fwd_lag(px[tkr], h, lag=0).reindex(entry).dropna()
        b = 100 * fwd_lag(px[tkr], h, lag=0).reindex(band).dropna().mean()
        loyo = {int(y): 100 * s[s.index.year != y].mean()
                for y in sorted(set(s.index.year))}
        lo_v = min(loyo.values())
        lo_y = [y for y, v in loyo.items() if v == lo_v][0]
        v = np.sort(s.values)[::-1]
        print(f"  {tkr:6s} h={h:2d}  full {100*s.mean():+.3f}%  LOYO floor "
              f"{lo_v:+.3f}% (drop {lo_y})  drop-3 {100*v[3:].mean():+.3f}%  "
              f"|  matched Aug control {b:+.3f}%  ->  drop-3 EXCESS "
              f"{100*v[3:].mean()-b:+.3f}pp,  LOYO-floor EXCESS "
              f"{lo_v-b:+.3f}pp")

# ------------------------------------------------------------- live state
print("\n\n" + "=" * 78)
print("4. THE LIVE PRICE STATE: XLE at its 52w high (-0.16%), z10 +2.22, "
      "USO 63d rank 4.4. Does the cell survive inside it?")
print("=" * 78)
r63 = pct_rank(px["USO"], 63)
xhi = rolling_on_valid(px["XLE"], lambda x: x.rolling(252).max())
xoff = (px["XLE"] / xhi - 1.0) * 100
print(f"  live: USO 63d rank {r63.dropna().iloc[-1]:.1f}, XLE off-high "
      f"{xoff.dropna().iloc[-1]:.2f}%")
for tkr in ("USO", "CL=F"):
    for h in (6, 10):
        s = fwd_lag(px[tkr], h, lag=0).reindex(entry).dropna()
        rr = r63.reindex(s.index)
        m = (rr <= 25).values
        print(f"  {tkr:6s} h={h:2d}  anchors with USO 63d rank <= 25: "
              f"{100*s[m].mean():+.3f}% N={int(m.sum())} "
              f"{sorted(set(s.index[m].year))}  |  rest "
              f"{100*s[~m].mean():+.3f}% N={int((~m).sum())}")
        xo = xoff.reindex(s.index)
        mh = (xo >= -2.0).values
        print(f"          anchors with XLE within 2% of its 52w high: "
              f"{100*s[mh].mean():+.3f}% N={int(mh.sum())} "
              f"{sorted(set(s.index[mh].year))}  |  rest "
              f"{100*s[~mh].mean():+.3f}% N={int((~mh).sum())}")

# ---------------------------------------------------------- multiplicity
print("\n\n" + "=" * 78)
print("5. MULTIPLICITY: the JH x vehicle x horizon grid this cell won")
print("=" * 78)
cells = []
for tkr in ("USO", "XLE", "CL=F", "XOP", "SPY"):
    for h in (1, 2, 3, 5, 6, 8, 10):
        s = fwd_lag(px[tkr], h, lag=0)
        e = s.reindex(entry).dropna()
        b = s.reindex(band).dropna()
        if len(e) < 10:
            continue
        cells.append({"tkr": tkr, "h": h, "n": len(e),
                      "excess_pp": 100 * (e.mean() - b.mean())})
g = pd.DataFrame(cells).sort_values("excess_pp", ascending=False)
print(g.round(3).to_string(index=False))
print(f"\n  {len(g)} cells; sd of excess {g.excess_pp.std():.3f}pp; "
      f"the pitched USO h=10 cell ranks "
      f"{int((g.excess_pp > g[(g.tkr=='USO')&(g.h==10)].excess_pp.iloc[0]).sum())+1} "
      f"of {len(g)}")
print(f"  cells with excess >= +1.0pp: {int((g.excess_pp >= 1.0).sum())}; "
      f"cells <= -1.0pp: {int((g.excess_pp <= -1.0).sum())}")

# ---------------------------------------------------------- book overlap
print("\n\n" + "=" * 78)
print("6. BOOK OVERLAP: what does the systematic book already do in this "
      "window?")
print("=" * 78)
p = Path("data/backtest_trades_full.parquet")
if not p.exists():
    print("  ledger absent")
else:
    tr = pd.read_parquet(p)
    dc = "Signal_Date" if "Signal_Date" in tr.columns else tr.columns[0]
    tr[dc] = pd.to_datetime(tr[dc])
    ENERGY = {"USO", "XLE", "XOP", "OIH", "XOM", "CVX", "COP", "OXY", "SLB",
              "EOG", "PSX", "VLO", "MPC", "HAL", "DVN", "FANG", "APA", "HES",
              "MRO", "PXD", "KMI", "WMB", "OKE", "BKR", "ERX", "ERY", "GUSH",
              "DRIP", "DIG", "DUG", "UCO", "SCO", "BNO", "UNG", "NRGU"}
    pos = pd.Series(range(len(d)), index=d)
    win = set()
    for a in entry:
        pa = pos.get(a)
        if pa is None:
            continue
        for k in range(0, 12):
            if pa + k < len(d):
                win.add(d[pa + k])
    sub = tr[tr[dc].isin(win) & tr["Ticker"].isin(ENERGY)]
    print(f"  ledger rows {len(tr)}; energy-ticker signals inside a JH-6..+10 "
          f"window: {len(sub)}")
    if len(sub):
        dcol = "Direction" if "Direction" in sub.columns else None
        print(sub.groupby(["Strategy_Name"] + ([dcol] if dcol else []))
              .agg(n=("Ticker", "size"),
                   avgR=("R_Multiple", "mean") if "R_Multiple" in sub.columns
                   else ("Ticker", "size")).round(3).to_string())
        if dcol:
            print(f"\n  direction split: "
                  f"{sub[dcol].value_counts().to_dict()}")
