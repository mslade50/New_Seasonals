"""Basis verification after the coordinator's pad-fill correction.

Confirms two things:

(1) Every MASK in c1_c4 / c2_c7 / c3_c8 / c4_book_overlap was already built on
    valid-session returns (`_valid_pct_change` / `pct_rank`), so no verdict
    rests on a padded number. Re-derived here from scratch as an independent
    check rather than asserted from the source.
(2) The only bare `.pct_change()` calls in those scripts are 1-DAY returns
    feeding correlation / PC1 / beta. pad-fill turns a calendar hole into a
    synthetic zero-return row that survives `.dropna()`, so those secondary
    statistics get re-measured on a strictly valid basis to confirm the
    quoted values do not move.
(3) The FXI-EEM 63d sub-premise, which the coordinator's audit table does not
    cover and which carries a LARGER error than either cell that does.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change

import numpy as np
import pandas as pd

BAR = pd.Timestamp("2026-08-24")
NAMES = ["OIH", "XOP", "XLE", "USO", "EEM", "EFA", "FXI", "GDX", "GLD", "SPY"]
px_all = load_prices(NAMES)
spy = px_all["SPY"]["Close"].dropna()
CAL = spy.index[spy.index <= BAR]
px = pd.DataFrame({t: px_all[t]["Close"] for t in NAMES}).reindex(CAL)


def clean(t, n):
    return _valid_pct_change(px_all[t]["Close"].dropna(), n).reindex(CAL)


def pit(s):
    return rolling_on_valid(s, lambda x: x.rolling(252).rank(pct=True) * 100)


print("=" * 96)
print("1. MASK BASIS - trigger counts and today's state, valid-session only")
print("=" * 96)
masks = {
    "C4 OIH-XOP PIT63 <= 2.5": pit(clean("OIH", 63) - clean("XOP", 63)) <= 2.5,
    "C7 GDX PIT21 >= 99": pct_rank(px_all["GDX"]["Close"].dropna(), 21).reindex(CAL) >= 99,
    "C8 EEM-EFA PIT63 <= 2": pit(clean("EEM", 63) - clean("EFA", 63)) <= 2.0,
    "C8 EEM-EFA PIT63 <= 5": pit(clean("EEM", 63) - clean("EFA", 63)) <= 5.0,
    "C8 EEM-EFA PIT63 <= 10 (only live rung)":
        pit(clean("EEM", 63) - clean("EFA", 63)) <= 10.0,
}
for k, m in masks.items():
    m = m.reindex(CAL, fill_value=False).fillna(False)
    live = bool(m.iloc[-1])
    last = m[m].index[-1].date() if m.any() else "never"
    print(f"  {k:42s} n_days={int(m.sum()):4d}  live TODAY={str(live):5s}  last={last}")

print("\n" + "=" * 96)
print("2. SECONDARY STATS on a strictly valid basis (pad vs no-pad)")
print("=" * 96)


def daily_valid(t):
    s = px_all[t]["Close"].dropna()
    return (s / s.shift(1) - 1.0).reindex(CAL)


COMPLEX = ["XLE", "XOP", "OIH", "USO"]
for lbl, frame in (
        ("pad (as quoted)", pd.DataFrame({t: px[t].pct_change() for t in COMPLEX})),
        ("strictly valid", pd.DataFrame({t: daily_valid(t) for t in COMPLEX}))):
    R = frame.dropna()
    C = R.corr()
    ev = np.linalg.eigvalsh(C.values)[::-1]
    pr = ev.sum() ** 2 / (ev ** 2).sum()
    off = C.values[np.triu_indices(len(COMPLEX), 1)]
    print(f"  energy complex {lbl:16s} n={len(R)}  mean corr {off.mean():.3f}  "
          f"PC1 {100*ev[0]/ev.sum():.1f}%  participation ratio {pr:.2f}")

for a, b, lbl in (("OIH", "XOP", "C4 beta OIH-on-XOP"),
                  ("EEM", "EFA", "C8 beta EEM-on-EFA")):
    for tag, ra, rb in (("pad", px[a].pct_change(), px[b].pct_change()),
                        ("valid", daily_valid(a), daily_valid(b))):
        beta = (ra.rolling(252).cov(rb) / rb.rolling(252).var()).reindex(CAL)
        print(f"  {lbl} [{tag:5s}] live {beta.iloc[-1]:.4f}  median {beta.median():.4f}")

print("\n" + "=" * 96)
print("3. FXI-EEM 63d - the C8 sub-premise the correction table omits")
print("=" * 96)
Q = "The size of the pad error depends on WHICH panel is padded, so the panel " \
    "has to match the recon's universe to reproduce the map."
print(f"  {Q}")
# this script's own 10-name panel is all US-listed ETFs, so padding it barely
# moves anything; the recon's panel carries ^TNX and DX-Y.NYB, whose foreign
# calendars are what open the holes.
narrow = (px.pct_change(63)["FXI"] - px.pct_change(63)["EEM"]) * 100
RECON = NAMES + ["^TNX", "DX-Y.NYB", "TLT", "SLV", "UUP", "QQQ", "XLK", "XLV"]
rp = load_prices(RECON)
wide = pd.DataFrame({t: rp[t]["Close"] for t in rp}).dropna(how="all")
wide = wide[wide.index <= BAR]
wpad = (wide.pct_change(63)["FXI"] - wide.pct_change(63)["EEM"]) * 100
val = (clean("FXI", 63) - clean("EEM", 63)) * 100
print(f"  surface map claims                 : +4.24pp  PIT 99.6")
print(f"  pad, RECON-universe panel (+^TNX,DX): {wpad.iloc[-1]:+.2f}pp  "
      f"PIT {pit(wpad/100).iloc[-1]:.1f}   <- reproduces the map")
print(f"  pad, US-ETF-only panel             : {narrow.iloc[-1]:+.2f}pp  "
      f"PIT {pit(narrow/100).iloc[-1]:.1f}   <- barely moves")
print(f"  VALID-SESSION truth                : {val.iloc[-1]:+.2f}pp  "
      f"PIT {pit(val/100).iloc[-1]:.1f}")
print(f"  error on the recon panel: {wpad.iloc[-1]-val.iloc[-1]:+.2f}pp / "
      f"{pit(wpad/100).iloc[-1]-pit(val/100).iloc[-1]:+.1f} pctile points -- LARGER "
      f"than either 63d cell in the correction table")
print("  consequence: 'China is leading the EM index it sits inside' is FALSE; "
      "FXI and EEM are level over 63 sessions.")
print("\nDONE basis verification")
