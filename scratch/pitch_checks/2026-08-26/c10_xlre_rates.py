"""C10 round 1: XLRE thrusting out of a 21-day trough while the long end rallies.

Today: XLRE z10 +1.25, XLRE 21d PIT rank 23.8, TLT 5d PIT rank 96.8.
Mechanism claim: XLRE is a duration proxy, so a bond rally should reprice a
sector that has not yet adjusted.

The registry has ZERO XLRE mentions but the SHAPE is closed repeatedly on
other sectors (XLI 08-25, XLU 08-25, watchlist #25 as a FAMILY effect with
Cochran Q 4.70 / p 0.789 / I-squared 0).  So the 9-SPDR reference class is the
deciding test, and specifically: does the DURATION-RALLY gate do work on XLRE
that it does not do on the other eight?
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

SECT = ["XLRE", "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
DEEP = ["IYR", "VNQ"]
ALL = SECT + DEEP + ["TLT", "SPY"]
px = load_prices(ALL)
idx = px["TLT"].index
for t in ALL:
    idx = idx.intersection(px[t].index) if t == "XLRE" else idx
# keep the full TLT/SPY calendar; align each sector on its own valid rows
CAL = px["SPY"].index.intersection(px["TLT"].index)
P = pd.DataFrame({t: px[t]["Close"].reindex(CAL) for t in ALL})

tlt_r5 = pct_rank(P["TLT"], 5)
tlt_r21 = pct_rank(P["TLT"], 21)

print(f"TODAY  XLRE z10={zscore(P['XLRE'],10).iloc[-1]:+.2f}  "
      f"rank21={pct_rank(P['XLRE'],21).iloc[-1]:.1f}   "
      f"TLT r5={tlt_r5.iloc[-1]:.1f}  r21={tlt_r21.iloc[-1]:.1f}")

Z_MIN, R21_MAX, TLT_MIN = 1.0, 30.0, 90.0


def cells(tkr: str):
    z = zscore(P[tkr], 10)
    r21 = pct_rank(P[tkr], 21)
    bare = (z >= Z_MIN) & (r21 <= R21_MAX)
    joint = bare & (tlt_r5 >= TLT_MIN)
    return bare, joint, (bare & (tlt_r5 < TLT_MIN))


def stats(tkr, mask, h, gap=5):
    ret = fwd_lag(P[tkr], h)
    sig = P.index[mask.fillna(False).values & ret.notna().values
                  & P[tkr].notna().values]
    if len(sig) == 0:
        return None, pd.DatetimeIndex([])
    epi = declusters(sig, gap, P.index[P[tkr].notna()])
    return ret.loc[epi].values, epi


print("\n" + "=" * 78)
print("1. XLRE: the joint cell against its controls and its own gate-off parent")
print("=" * 78)
bare, joint, offgate = cells("XLRE")
print(f"XLRE sample {P['XLRE'].dropna().index[0].date()} .. "
      f"{P['XLRE'].dropna().index[-1].date()}  "
      f"({int(P['XLRE'].notna().sum())} sessions)")
print(f"  z10>={Z_MIN} & r21<={R21_MAX}          : {int(bare.sum())} days")
print(f"  + TLT r5 >= {TLT_MIN}                  : {int(joint.sum())} days")
for h in (3, 5, 7, 10):
    j, ej = stats("XLRE", joint, h)
    b, eb = stats("XLRE", bare, h)
    o, eo = stats("XLRE", offgate, h)
    base = fwd_lag(P["XLRE"], h).dropna()
    rows = [summarize(j, f"JOINT h={h} (N={len(ej)} epi)"),
            summarize(b, f"BARE thrust-from-trough (N={len(eb)})"),
            summarize(o, f"BARE with TLT NOT rallying (N={len(eo)})"),
            summarize(base.values, "CTRL XLRE all days")]
    show(rows)
    if j is not None and b is not None and len(j):
        print(f"  RATES GATE VALUE = {100*(j.mean()-b.mean()):+.3f}pp "
              f"(discards {len(eb)-len(ej)} of {len(eb)} episodes); "
              f"joint excess over own drift {100*(j.mean()-base.mean()):+.3f}pp "
              f"= {100*(j.mean()-base.mean())*100/6:.1f}x a 6 bp round trip")
        w = int((j > 0).sum())
        up = float((base > 0).mean())
        print(f"  record {w}-{len(j)-w}; sign p vs XLRE's own up-rate "
              f"{100*up:.1f}% = {sign_test(w, len(j), up):.4f}")
        print(f"  episodes: {[str(d.date()) for d in ej]}\n")

print("=" * 78)
print("2. REFERENCE CLASS: identical rule on 10 sectors.  Does the RATES gate")
print("   do work on XLRE that it does not do on the other nine?")
print("=" * 78)
for h in (3, 5, 7):
    rows = []
    for t in SECT:
        b_, eb = stats(t, cells(t)[0], h)
        j_, ej = stats(t, cells(t)[1], h)
        base = fwd_lag(P[t], h).dropna()
        if j_ is None or len(j_) < 3:
            rows.append({"label": t, "n": 0 if j_ is None else len(j_)})
            continue
        r = summarize(j_, t)
        r["bare_pct"] = round(100 * b_.mean(), 3)
        r["n_bare"] = len(eb)
        r["gate_pp"] = round(100 * (j_.mean() - b_.mean()), 3)
        r["excess_pp"] = round(100 * (j_.mean() - base.mean()), 3)
        rows.append(r)
    df = pd.DataFrame(rows).sort_values("excess_pp", ascending=False)
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].round(3)
    print(f"\n--- h={h}, joint cell, sorted by excess over own drift ---")
    print(df.to_string(index=False))
    have_xlre = ("excess_pp" in df
                 and df.loc[df["label"] == "XLRE", "excess_pp"].notna().any())
    if not have_xlre:
        print("  XLRE has < 3 episodes at this rung -> NOT RANKABLE here; "
              "see c10b/c10c for the loosened, era-matched reference class")
    if "excess_pp" in df and df["excess_pp"].notna().any() and have_xlre:
        ex = df["excess_pp"].dropna().values
        xr = float(df.loc[df["label"] == "XLRE", "excess_pp"].iloc[0])
        gp = df["gate_pp"].dropna().values
        xg = float(df.loc[df["label"] == "XLRE", "gate_pp"].iloc[0])
        print(f"  XLRE excess {xr:+.3f}pp ranks {1+int((ex>xr).sum())} of {len(ex)}")
        print(f"  XLRE rates-GATE value {xg:+.3f}pp ranks "
              f"{1+int((gp>xg).sum())} of {len(gp)}; family mean gate value "
              f"{gp.mean():+.3f}pp, {int((gp>0).sum())} of {len(gp)} positive")
    if "gate_pp" in df and df["gate_pp"].notna().any():
        gp = df["gate_pp"].dropna().values
        print(f"  family mean gate value {gp.mean():+.3f}pp, "
              f"{int((gp>0).sum())} of {len(gp)} positive")
        # Cochran Q on the joint cells
        dq = df.dropna(subset=["mean_pct", "sd_pct"])
        mus = dq["mean_pct"].values / 100
        sds = dq["sd_pct"].values / 100
        ns = dq["n"].values.astype(float)
        w = ns / sds ** 2
        mfe = (w * mus).sum() / w.sum()
        Qs = (w * (mus - mfe) ** 2).sum()
        print(f"  fixed-effect common excess {100*mfe:+.3f}%, Cochran Q {Qs:.2f} "
              f"on {len(mus)-1} df")

print("\n" + "=" * 78)
print("3. WHICH OF THE THREE VARIANTS AM I MEASURING?  (watchlist #26 taxonomy)")
print("=" * 78)
for t in ("XLRE", "XLU"):
    z = zscore(P[t], 10); r21 = pct_rank(P[t], 21)
    vs = {
        "A today: lagging base + thrust + TLT RALLYING": (z >= 1) & (r21 <= 30) & (tlt_r5 >= 90),
        "B wl#26: sector washout r21<=5 + TLT ALSO HIT (r21<25)": (r21 <= 5) & (tlt_r21 < 25),
        "C mid-range: washout r21<=5, TLT mid": (r21 <= 5) & (tlt_r21 >= 25) & (tlt_r21 <= 75),
        "D bare washout r21<=5": (r21 <= 5),
        "E bare thrust-from-trough (no rates)": (z >= 1) & (r21 <= 30),
    }
    print(f"\n### {t}")
    for h in (3, 5):
        rows = []
        for lbl, m in vs.items():
            v, e = stats(t, m, h)
            rows.append(summarize(v, f"h={h} {lbl}") if v is not None
                        else {"label": lbl, "n": 0})
        base = fwd_lag(P[t], h).dropna()
        rows.append(summarize(base.values, f"h={h} CTRL all days"))
        show(rows)

print("\n" + "=" * 78)
print("4. DEEP HISTORY: IYR / VNQ, which pre-date XLRE's 2015 start")
print("=" * 78)
for t in ("IYR", "VNQ", "XLRE"):
    for h in (3, 5, 7):
        j, ej = stats(t, cells(t)[1], h)
        b, eb = stats(t, cells(t)[0], h)
        base = fwd_lag(P[t], h).dropna()
        if j is None:
            continue
        mid = np.array([d.year % 4 == 2 for d in ej])
        print(f"{t} h={h}: JOINT {100*j.mean():+.3f}% N={len(j)} "
              f"hit {100*(j>0).mean():.0f}% | BARE {100*b.mean():+.3f}% N={len(b)} "
              f"| drift {100*base.mean():+.3f}% | gate "
              f"{100*(j.mean()-b.mean()):+.3f}pp | midterm "
              f"{100*j[mid].mean() if mid.any() else float('nan'):+.3f}% "
              f"(N={int(mid.sum())})")
    print()
