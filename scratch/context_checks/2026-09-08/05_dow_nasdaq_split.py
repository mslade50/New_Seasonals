"""Dow down hard while the Nasdaq holds: what happens next, and does the SPREAD persist?

Live state 2026-09-08: ^DJI -1.18%, QQQ -0.08% -> QQQ-minus-DJI spread +1.10pp on a
session the Dow lost more than 1%. Not a one-bar thing: ^DJI 21d rank 9.1 vs QQQ 30.6,
^DJI 21d -2.31% vs QQQ 21d -0.65%.

Cell: QQQ_ret - DJI_ret >= +1.0pp AND DJI_ret <= -1.0%.
Mirror control: DJI_ret - QQQ_ret >= +1.0pp AND QQQ_ret <= -1.0% (is the effect about the
SPREAD, or just about "somebody was down a lot today"?).

Convention: market-context product -> fwd_ret, lag=0, close-to-close.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

TK = ["^DJI", "QQQ", "^GSPC", "IWM"]
px = close_panel(TK)
px = px.dropna(subset=["^DJI", "QQQ", "^GSPC"])
print("panel", px.index.min().date(), "->", px.index.max().date(), len(px), "rows")

ret = {t: px[t] / px[t].shift(1) - 1.0 for t in TK}
spread = ret["QQQ"] - ret["^DJI"]          # + means QQQ beat DJI

SPREAD_MIN = 0.010
DJI_MAX = -0.010

cell = (spread >= SPREAD_MIN) & (ret["^DJI"] <= DJI_MAX)
cell = cell.fillna(False)
dates = px.index[cell.values]

mirror = ((-spread) >= SPREAD_MIN) & (ret["QQQ"] <= DJI_MAX)
mirror = mirror.fillna(False)
mdates = px.index[mirror.values]

ALLD = px.index


def rowify(f: pd.Series, d: pd.DatetimeIndex, label: str) -> dict:
    """summarize + record + exact sign p + all-days control + edge."""
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


# ---------------------------------------------------------------- 1. the cell
print(f"\n############ CELL: QQQ-DJI spread >= {100*SPREAD_MIN:.1f}pp "
      f"AND DJI <= {100*DJI_MAX:.1f}% ############")
print(f"raw days n = {len(dates)}   {dates.min().date()} -> {dates.max().date()}")
by_yr = pd.Series(1, index=dates).groupby(dates.year).sum()
print("\nby year:")
print(by_yr.to_string())
print(f"\n2000-2002 share: {by_yr.reindex(range(2000, 2003)).fillna(0).sum():.0f} "
      f"of {len(dates)}   2020 share: {by_yr.get(2020, 0):.0f}")
print("\nfull date list:")
for y, g in pd.Series(dates).groupby(dates.year):
    print(f"  {y}: " + ", ".join(str(d.date()) for d in g))

print("\n>>> today's session in the mask:",
      pd.Timestamp("2026-09-08") in dates,
      f"(spread {100*spread.iloc[-1]:+.2f}pp, DJI {100*ret['^DJI'].iloc[-1]:+.2f}%)")

GAP = 10  # td: 2000-02 and 2020 fire on consecutive/near sessions
epi = declusters(dates, GAP, ALLD)
mepi = declusters(mdates, GAP, ALLD)
print(f"\ndeclustered at {GAP}td: {len(epi)} episodes (from {len(dates)} raw days)")
print("episode dates:", ", ".join(str(d.date()) for d in epi))

# ------------------------------------------- 2. forward returns per instrument
for tag, dset in (("RAW DAYS", dates), (f"EPISODES ({GAP}td gap)", epi)):
    for sub in ["^DJI", "QQQ", "^GSPC", "IWM"]:
        rows = []
        for h in (1, 5, 10):
            rows.append(rowify(fwd_ret(px[sub], h), dset, f"h={h}"))
        show(rows, f"{sub} forward after the split day  [{tag}]")

# ---------------------------------------------- 3. THE SPREAD as the subject
print("\n\n############ THE SPREAD ITSELF: does QQQ-minus-DJI continue or revert? "
      "############")
for tag, dset in (("RAW DAYS", dates), (f"EPISODES ({GAP}td gap)", epi)):
    rows = []
    for h in (1, 5, 10):
        f = fwd_ret(px["QQQ"], h) - fwd_ret(px["^DJI"], h)
        r = rowify(f, dset, f"h={h} fwd spread (pp)")
        rows.append(r)
    show(rows, f"QQQ - DJI forward spread, percentage points  [{tag}]")
    print("  (positive = QQQ keeps outperforming = CONTINUATION; "
          "negative = Dow catches up = REVERSAL)")

# same for SPY-ish breadth read: IWM - QQQ
rows = []
for h in (1, 5, 10):
    f = fwd_ret(px["IWM"], h) - fwd_ret(px["QQQ"], h)
    rows.append(rowify(f, epi, f"h={h}"))
show(rows, "side look: IWM - QQQ forward spread (pp), episodes")

# ------------------------------------------------ 4. era split + concentration
HEAD_H = 5
print(f"\n\n############ era split + concentration, headline h={HEAD_H} ############")
fsp = fwd_ret(px["QQQ"], HEAD_H) - fwd_ret(px["^DJI"], HEAD_H)
base = fsp.dropna()

d_raw = pd.DatetimeIndex(dates).intersection(base.index)
d_epi = pd.DatetimeIndex(epi).intersection(base.index)
show(era_split(d_raw, base.loc[d_raw].values), "SPREAD h5 era split - raw days")
show(era_split(d_epi, base.loc[d_epi].values), "SPREAD h5 era split - episodes")
print("\nconcentration (episodes, k=2):", cluster_note(d_epi, base.loc[d_epi].values, 2))
print("concentration (episodes, k=4):", cluster_note(d_epi, base.loc[d_epi].values, 4))

for sub in ["^GSPC", "^DJI", "QQQ"]:
    f = fwd_ret(px[sub], HEAD_H).dropna()
    de = pd.DatetimeIndex(epi).intersection(f.index)
    show(era_split(de, f.loc[de].values), f"{sub} h{HEAD_H} era split - episodes")
    print(f"  concentration k=2: {cluster_note(de, f.loc[de].values, 2)}")

# local control on the headline
loc = local_control(base.index, d_raw, 126)
show([rowify(fsp, d_raw, "COND raw days"),
      rowify(fsp, d_epi, "COND episodes"),
      rowify(fsp, loc, "CTRL local +/-126td ex-trigger")],
     f"SPREAD h{HEAD_H}: conditional vs local control")

# ------------------------------------------------------- 5. the MIRROR control
print(f"\n\n############ MIRROR CELL: DJI-QQQ spread >= {100*SPREAD_MIN:.1f}pp "
      f"AND QQQ <= {100*DJI_MAX:.1f}% ############")
print(f"raw days n = {len(mdates)}   {mdates.min().date()} -> {mdates.max().date()}")
mby = pd.Series(1, index=mdates).groupby(mdates.year).sum()
print("by year:", dict(mby))
print(f"declustered at {GAP}td: {len(mepi)} episodes")

for sub in ["^DJI", "QQQ", "^GSPC"]:
    rows = []
    for h in (1, 5, 10):
        rows.append(rowify(fwd_ret(px[sub], h), mepi, f"h={h}"))
    show(rows, f"MIRROR: {sub} forward, episodes")

rows = []
for h in (1, 5, 10):
    f = fwd_ret(px["QQQ"], h) - fwd_ret(px["^DJI"], h)
    rows.append(rowify(f, mepi, f"h={h} fwd QQQ-DJI spread (pp)"))
show(rows, "MIRROR: QQQ - DJI forward spread, episodes")
print("  (if the main cell shows spread continuation and the mirror shows the SAME sign,")
print("   the finding is a QQQ drift artifact, not a spread effect.)")

# a plain 'any down day' control for scale
plain = px.index[(ret["^DJI"] <= DJI_MAX).fillna(False).values]
plain_epi = declusters(plain, GAP, ALLD)
rows = []
for h in (1, 5, 10):
    rows.append(rowify(fwd_ret(px["^GSPC"], h), plain_epi, f"h={h}"))
show(rows, f"CONTROL: ANY ^DJI <= -1.0% day (n_raw={len(plain)}, "
           f"n_epi={len(plain_epi)}) -> ^GSPC forward")
rows = []
for h in (1, 5, 10):
    f = fwd_ret(px["QQQ"], h) - fwd_ret(px["^DJI"], h)
    rows.append(rowify(f, plain_epi, f"h={h}"))
show(rows, "CONTROL: ANY ^DJI <= -1.0% day -> QQQ-DJI forward spread (pp)")

# unconditional spread scale, so 'mean spread' has a yardstick
for h in (1, 5, 10):
    f = (fwd_ret(px["QQQ"], h) - fwd_ret(px["^DJI"], h)).dropna()
    print(f"unconditional QQQ-DJI h={h} spread: mean {100*f.mean():+.3f}pp, "
          f"median {100*f.median():+.3f}pp, sd {100*f.std():.3f}pp, "
          f"pos {100*(f > 0).mean():.1f}%, n={len(f)}")

# --------------------------------------------- 6. the 2018+ sub-cell in detail
# The only thing in the whole file that carries a t-stat is the modern era.
# Check whether it is just 2020.
print("\n\n############ 2018+ SUB-CELL DETAIL ############")
cut = pd.Timestamp("2018-01-01")
epi_mod = pd.DatetimeIndex([d for d in epi if d >= cut])
raw_mod = pd.DatetimeIndex([d for d in dates if d >= cut])
print(f"2018+ episodes n={len(epi_mod)}: "
      + ", ".join(str(d.date()) for d in epi_mod))
for sub in ["^GSPC", "^DJI", "QQQ", "IWM"]:
    rows = []
    for h in (1, 5, 10):
        rows.append(rowify(fwd_ret(px[sub], h), epi_mod, f"h={h}"))
    show(rows, f"2018+ episodes: {sub} forward")

f5 = fwd_ret(px["^GSPC"], 5).dropna()
dm = pd.DatetimeIndex(epi_mod).intersection(f5.index)
vm = f5.loc[dm].values
print(f"\n^GSPC h5 2018+ concentration k=2: {cluster_note(dm, vm, 2)}")
print(f"^GSPC h5 2018+ concentration k=4: {cluster_note(dm, vm, 4)}")
ex2020 = pd.DatetimeIndex([d for d in dm if d.year != 2020])
print("\nex-2020 check:")
show([summarize(vm, "2018+ all"),
      summarize(f5.loc[ex2020].values, "2018+ ex-2020")],
     "^GSPC h5, 2018+ with and without 2020")
up = int((f5.loc[ex2020].values > 0).sum())
n = len(ex2020)
print(f"  ex-2020 record {up}-{n-up}, sign p = {sign_test(up, n):.4f}")

# and the same-era plain-down-day control, the honest comparison
plain_mod = pd.DatetimeIndex([d for d in plain_epi if d >= cut])
rows = []
for h in (1, 5, 10):
    rows.append(rowify(fwd_ret(px["^GSPC"], h), plain_mod, f"h={h}"))
show(rows, f"2018+ CONTROL: any ^DJI <= -1.0% episode (n={len(plain_mod)}) -> ^GSPC")
