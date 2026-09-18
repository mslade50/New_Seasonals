"""VIX3M/VIX term structure on 2026-09-08.

PREMISE CHECK FIRST. The cell was commissioned as "cheap spot vol under a steep and
STEEPENING curve". Against data/master_prices.parquet that is not what happened:

    ^VIX    14.53 -> 15.72   = +8.19% TODAY   (5d +5.36%, 21d +5.50%)
    ^VIX3M  17.61 -> 18.39   = +4.43% TODAY   (5d +4.91%)
    ratio   1.2120 -> 1.1698 = -3.48% TODAY   -> a FLATTENING, 11.5th pctile of daily moves

The ratio's trailing-252d rank is 55.6 (middle of its own year), not a top decile, and
its 5d change is -0.43% against a top-5% threshold of +10.46%. So the two cells as
specified DO NOT FIRE TODAY. They are run anyway, in full, because "the state is not
live" is itself the answer; then Cell C runs the state the tape ACTUALLY printed.

Cell A: ratio in the top decile of its trailing 252 sessions AND ^VIX < 17.
Cell B: 5-session CHANGE in the ratio in the top 5% of its full-history distribution.
Cell C: 1-session ratio change in the BOTTOM decile AND ^VIX < 17  <- today's shape.

CAREFUL: contango is the NORMAL state (ratio > 1 on 89% of sessions), so the
unconditional distribution and the local control matter more here than usual.

Convention: market-context product -> fwd_ret, lag=0, close-to-close.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

TK = ["^VIX", "^VIX3M", "^GSPC", "SPY", "^VVIX"]
px = close_panel(TK)
px = px.dropna(subset=["^VIX", "^VIX3M", "^GSPC"])
print("panel", px.index.min().date(), "->", px.index.max().date(), len(px), "rows")

ratio = px["^VIX3M"] / px["^VIX"]
vix = px["^VIX"]
ALLD = px.index
TODAY = pd.Timestamp("2026-09-08")


def rowify(f: pd.Series, d, label: str) -> dict:
    base = f.dropna()
    dd = pd.DatetimeIndex(d).intersection(base.index)
    v = base.loc[dd].values
    r = summarize(v, label)
    if r["n"]:
        up = int((v > 0).sum())
        r["record"] = f"{up}-{r['n'] - up}"
        r["sign_p"] = round(sign_test(up, r["n"]), 4)
        r["ctl_all_pct"] = round(100 * base.mean(), 3)
        r["edge_pct"] = round(r["mean_pct"] - 100 * base.mean(), 3)
    for k in ("sd_pct", "worst_pct", "best_pct"):
        r.pop(k, None)
    return r


def horizons(sub: str, dset, hs=(1, 5, 10, 21), tag: str = "") -> None:
    show([rowify(fwd_ret(px[sub], h), dset, f"h={h}") for h in hs],
         f"{sub} forward  {tag}")


# =========================================================== 0. PREMISE CHECK
print("\n" + "=" * 78)
print("PREMISE CHECK - what the cache says the term structure actually did today")
print("=" * 78)
d1r = ratio / ratio.shift(1) - 1.0
d5r = rolling_on_valid(ratio, lambda x: x / x.shift(5) - 1.0)
v1 = vix / vix.shift(1) - 1.0
v5 = rolling_on_valid(vix, lambda x: x / x.shift(5) - 1.0)
m1 = px["^VIX3M"] / px["^VIX3M"].shift(1) - 1.0
m5 = rolling_on_valid(px["^VIX3M"], lambda x: x / x.shift(5) - 1.0)

print(f"  ^VIX    {vix.iloc[-2]:.2f} -> {vix.iloc[-1]:.2f}   "
      f"1d {100*v1.iloc[-1]:+.2f}%   5d {100*v5.iloc[-1]:+.2f}%")
print(f"  ^VIX3M  {px['^VIX3M'].iloc[-2]:.2f} -> {px['^VIX3M'].iloc[-1]:.2f}   "
      f"1d {100*m1.iloc[-1]:+.2f}%   5d {100*m5.iloc[-1]:+.2f}%")
print(f"  ratio   {ratio.iloc[-2]:.4f} -> {ratio.iloc[-1]:.4f}  "
      f"1d {100*d1r.iloc[-1]:+.2f}%   5d {100*d5r.iloc[-1]:+.2f}%")
print(f"  -> the curve FLATTENED today. 1d ratio change sits at the "
      f"{100*(d1r.dropna() <= d1r.iloc[-1]).mean():.1f}th percentile of all daily moves.")
print("  -> the brief's stated '^VIX 5d -3.79%' is NOT in this cache; VIX 5d is "
      f"{100*v5.iloc[-1]:+.2f}%.")

r252 = rolling_on_valid(ratio, lambda x: x.rolling(252).rank(pct=True) * 100.0)
today = ratio.iloc[-1]
full_pct = 100.0 * (ratio.dropna() <= today).mean()
print(f"\n  ratio level {today:.4f}: full-history pctile {full_pct:.1f} "
      f"(n={ratio.notna().sum()} since {ratio.dropna().index[0].date()}); "
      f"trailing-252d rank {r252.iloc[-1]:.1f}")

print("\nUNCONDITIONAL ratio distribution (contango IS the normal state):")
print(ratio.dropna().quantile([.01, .05, .10, .25, .50, .75, .90, .95, .99])
      .round(4).to_string())
print(f"  mean {ratio.mean():.4f}  sd {ratio.std():.4f}  "
      f"share > 1.00: {100*(ratio.dropna() > 1).mean():.1f}%  "
      f"share > 1.17: {100*(ratio.dropna() > 1.17).mean():.1f}%  "
      f"share ^VIX < 17: {100*(vix.dropna() < 17).mean():.1f}%")

# ================================================================== CELL A
print("\n\n" + "=" * 78)
print("CELL A: ratio in trailing-252d TOP DECILE and ^VIX < 17")
print("=" * 78)
cellA = ((r252 >= 90.0) & (vix < 17.0)).fillna(False)
A = ALLD[cellA.values]
print(f"raw days n = {len(A)}   {A.min().date()} -> {A.max().date()}")
print("by year:", dict(pd.Series(1, index=A).groupby(A.year).sum()))
print(f">>> TODAY IN THE MASK: {TODAY in A}   (r252 rank {r252.iloc[-1]:.1f} "
      f"needs >= 90; VIX {vix.iloc[-1]:.2f} needs < 17)")

GAP_A = 21  # the state persists for weeks; less counts one regime many times
epiA = declusters(A, GAP_A, ALLD)
print(f"declustered at {GAP_A}td: {len(epiA)} episodes")
print("episodes:", ", ".join(str(d.date()) for d in epiA))

for tag, dset in (("[RAW DAYS]", A), (f"[EPISODES {GAP_A}td]", epiA)):
    horizons("^GSPC", dset, tag=tag)
horizons("^VIX", epiA, tag=f"[EPISODES {GAP_A}td]")

print("\n-- CELL A decomposition: which gate is doing the work? --")
only_decile = ALLD[(r252 >= 90.0).fillna(False).values]
only_lowvix = ALLD[(vix < 17.0).fillna(False).values]
for h in (1, 5, 10, 21):
    f = fwd_ret(px["^GSPC"], h)
    base = f.dropna()
    dA = pd.DatetimeIndex(A).intersection(base.index)
    dE = pd.DatetimeIndex(epiA).intersection(base.index)
    loc = local_control(base.index, dA, 126)
    show([rowify(f, dA, "COND raw days"),
          rowify(f, dE, "COND episodes"),
          rowify(f, loc, "CTRL local +/-126td ex-trigger"),
          rowify(f, only_decile, "CTRL ratio top decile ONLY"),
          rowify(f, only_lowvix, "CTRL ^VIX < 17 ONLY"),
          rowify(f, base.index, "CTRL all days")],
         f"CELL A, ^GSPC h={h}")

# ================================================================== CELL B
print("\n\n" + "=" * 78)
print("CELL B: 5d CHANGE in the ratio in the top 5%")
print("=" * 78)
thr = d5r.dropna().quantile(0.95)
print(f"today's 5d ratio change: {100*d5r.iloc[-1]:+.2f}%   "
      f"top-5% threshold = {100*thr:+.2f}%")
print(f">>> TODAY IN THE MASK: {bool(d5r.iloc[-1] >= thr)}")
print("5d-change distribution:",
      {k: round(v, 4) for k, v in
       d5r.dropna().quantile([.05, .25, .5, .75, .95, .99]).items()})

cellB = (d5r >= thr).fillna(False)
B = ALLD[cellB.values]
print(f"\nraw days n = {len(B)}   {B.min().date()} -> {B.max().date()}")
print("by year:", dict(pd.Series(1, index=B).groupby(B.year).sum()))
GAP_B = 10
epiB = declusters(B, GAP_B, ALLD)
print(f"declustered at {GAP_B}td: {len(epiB)} episodes")

for tag, dset in (("[RAW DAYS]", B), (f"[EPISODES {GAP_B}td]", epiB)):
    horizons("^GSPC", dset, tag=f"after a top-5% 5d ratio JUMP {tag}")
    horizons("^VIX", dset, tag=f"after a top-5% 5d ratio JUMP {tag}")

print("\n-- CELL B controls, ^GSPC --")
for h in (1, 5, 10, 21):
    f = fwd_ret(px["^GSPC"], h)
    base = f.dropna()
    dB = pd.DatetimeIndex(B).intersection(base.index)
    dE = pd.DatetimeIndex(epiB).intersection(base.index)
    show([rowify(f, dB, "COND raw days"),
          rowify(f, dE, "COND episodes"),
          rowify(f, local_control(base.index, dB, 126), "CTRL local +/-126td"),
          rowify(f, base.index, "CTRL all days")],
         f"CELL B, ^GSPC h={h}")

# ---------------- CELL B-prime: the shape the brief DESCRIBED (jump, VIX falling)
print("\n\n" + "=" * 78)
print("CELL B-prime: top-5% 5d ratio jump AND ^VIX 5d return < 0")
print("(the shape the brief described - VIX3M up while VIX falls - which the cache "
      "does not show today)")
print("=" * 78)
cellBp = (cellB & (v5 < 0)).fillna(False)
Bp = ALLD[cellBp.values]
epiBp = declusters(Bp, GAP_B, ALLD)
print(f"raw days n = {len(Bp)}, episodes {len(epiBp)}   "
      f">>> TODAY IN THE MASK: {TODAY in Bp}")
print("by year:", dict(pd.Series(1, index=Bp).groupby(Bp.year).sum()))
if len(epiBp) > 1:
    print("episodes:", ", ".join(str(d.date()) for d in epiBp))
    horizons("^GSPC", epiBp, tag="[B-prime EPISODES]")
    horizons("^VIX", epiBp, tag="[B-prime EPISODES]")

# ================================================================== CELL C
print("\n\n" + "=" * 78)
print("CELL C: the state TODAY ACTUALLY PRINTED - 1d ratio change in the BOTTOM "
      "DECILE and ^VIX < 17")
print("(spot vol pops off a low base faster than 3-month; the curve compresses)")
print("=" * 78)
thrC = d1r.dropna().quantile(0.10)
print(f"today's 1d ratio change {100*d1r.iloc[-1]:+.2f}%   "
      f"bottom-decile threshold {100*thrC:+.2f}%")
cellC = ((d1r <= thrC) & (vix < 17.0)).fillna(False)
C = ALLD[cellC.values]
print(f">>> TODAY IN THE MASK: {TODAY in C}")
print(f"raw days n = {len(C)}   {C.min().date()} -> {C.max().date()}")
print("by year:", dict(pd.Series(1, index=C).groupby(C.year).sum()))
GAP_C = 5
epiC = declusters(C, GAP_C, ALLD)
print(f"declustered at {GAP_C}td: {len(epiC)} episodes")

for tag, dset in (("[RAW DAYS]", C), (f"[EPISODES {GAP_C}td]", epiC)):
    horizons("^GSPC", dset, tag=f"after a bottom-decile 1d FLATTENING, VIX<17 {tag}")
horizons("^VIX", epiC, tag=f"[EPISODES {GAP_C}td]")

print("\n-- CELL C controls, ^GSPC --")
only_flat = ALLD[(d1r <= thrC).fillna(False).values]
for h in (1, 5, 10, 21):
    f = fwd_ret(px["^GSPC"], h)
    base = f.dropna()
    dC = pd.DatetimeIndex(C).intersection(base.index)
    dE = pd.DatetimeIndex(epiC).intersection(base.index)
    show([rowify(f, dC, "COND raw days"),
          rowify(f, dE, "COND episodes"),
          rowify(f, local_control(base.index, dC, 126), "CTRL local +/-126td"),
          rowify(f, only_flat, "CTRL bottom-decile flatten ONLY (no VIX gate)"),
          rowify(f, only_lowvix, "CTRL ^VIX < 17 ONLY"),
          rowify(f, base.index, "CTRL all days")],
         f"CELL C, ^GSPC h={h}")

# ============================================== era split + concentration
print("\n\n" + "=" * 78)
print("era split (2018) + concentration")
print("=" * 78)
for name, dset in (("CELL A", epiA), ("CELL B", epiB), ("CELL C", epiC)):
    for h in (5, 21):
        f = fwd_ret(px["^GSPC"], h).dropna()
        de = pd.DatetimeIndex(dset).intersection(f.index)
        if len(de) < 3:
            continue
        v = f.loc[de].values
        show(era_split(de, v), f"{name} ^GSPC h={h} era split (episodes, n={len(de)})")
        print(f"  concentration k=2: {cluster_note(de, v, 2)}")
        print(f"  concentration k=4: {cluster_note(de, v, 4)}")

# Cell A's real result is on ^VIX, not ^GSPC - split that too.
for h in (5, 21):
    f = fwd_ret(px["^VIX"], h).dropna()
    de = pd.DatetimeIndex(epiA).intersection(f.index)
    v = f.loc[de].values
    show(era_split(de, v), f"CELL A ^VIX h={h} era split (episodes, n={len(de)})")
    print(f"  concentration k=2: {cluster_note(de, v, 2)}")
    up = int((v > 0).sum())
    print(f"  record {up}-{len(v)-up}, sign p = {sign_test(up, len(v)):.5f}")
