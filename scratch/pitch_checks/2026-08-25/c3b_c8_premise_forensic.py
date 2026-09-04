"""Forensic on the C4 / C8 / C7 premise numbers in 00_surface_map.md.

The map's recon (00_recon_premises.py) builds ONE panel over
["...","^TNX","DX-Y.NYB",...] and calls `px.pct_change(63)` on it. That is the
union-calendar + pad trap `pitch_lab._valid_pct_change` exists to prevent
(2026-08-19 registry: the pad form moved pct_rank by up to 29.4 percentile
points). This script reproduces the map's number, the clean number, and the
gap, for every premise the three candidates are named after.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change

import pandas as pd

BAR = pd.Timestamp("2026-08-24")
MAP_UNIVERSE = ["SPY", "QQQ", "IWM", "XLK", "XLV", "XLP", "XLU", "XLI", "XLE",
                "SMH", "TLT", "IEF", "^TNX", "HYG", "LQD", "GLD", "GDX", "SLV",
                "UUP", "DX-Y.NYB", "EFA", "EEM", "FXI", "OIH", "XOP", "USO"]
have = load_prices(MAP_UNIVERSE)
panel = pd.DataFrame({t: have[t]["Close"] for t in have}).dropna(how="all")
panel = panel[panel.index <= BAR]
print(f"panel rows {len(panel)}  cols {len(panel.columns)}")
nan_rows = int(panel[["EEM", "EFA"]].notna().all(axis=1).sum())
print(f"rows where BOTH EEM and EFA print: {nan_rows}")
for t in ["EEM", "EFA", "OIH", "XOP", "GDX", "GLD", "^TNX", "DX-Y.NYB"]:
    if t in panel:
        s = panel[t]
        v = s.dropna()
        print(f"  {t}: first {v.index[0].date()}  n_valid {len(v)}  "
              f"NaN rows inside its own span "
              f"{int(s.loc[v.index[0]:].isna().sum())}")

print("\n" + "=" * 96)
print("PREMISE COMPARISON: map method (panel pct_change + pad) vs clean method")
print("=" * 96)
map_r = {n: panel.pct_change(n) for n in (5, 21, 63)}


def clean_r(t, n):
    return _valid_pct_change(have[t]["Close"].dropna(), n).reindex(panel.index)


def pit_of(s):
    return rolling_on_valid(s, lambda x: x.rolling(252).rank(pct=True) * 100).iloc[-1]


CASES = [
    ("C8  EEM-EFA 63d", "EEM", "EFA", 63, -7.57, 1.6),
    ("C4  OIH-XOP 63d", "OIH", "XOP", 63, -18.52, 0.4),
    ("C4  OIH-XLE  5d", "OIH", "XLE", 5, -5.04, 2.4),
    ("C8  FXI-EEM 63d", "FXI", "EEM", 63, +4.24, 99.6),
    ("C7  GDX-GLD 21d", "GDX", "GLD", 21, +22.90, 99.2),
]
for lbl, a, b, n, map_val, map_pit in CASES:
    m = (map_r[n][a] - map_r[n][b]).dropna()
    c = (clean_r(a, n) - clean_r(b, n)).dropna()
    print(f"\n{lbl}")
    print(f"   map-as-written : {100*m.iloc[-1]:+8.2f}pp   PIT {pit_of(map_r[n][a]-map_r[n][b]):6.2f}"
          f"   (surface map claims {map_val:+.2f}pp / PIT {map_pit})")
    print(f"   clean (valid-session, per-ticker) : {100*c.iloc[-1]:+8.2f}pp   "
          f"PIT {pit_of(clean_r(a,n)-clean_r(b,n)):6.2f}")
    print(f"   DELTA level {100*(c.iloc[-1]-m.iloc[-1]):+.2f}pp   "
          f"DELTA pctile {pit_of(clean_r(a,n)-clean_r(b,n)) - pit_of(map_r[n][a]-map_r[n][b]):+.2f}pt")

print("\n" + "=" * 96)
print("SINGLE-NAME premises")
print("=" * 96)
for t, n, claim in (("GDX", 21, 37.63), ("OIH", 63, None), ("EEM", 63, None)):
    m = map_r[n][t]
    c = clean_r(t, n)
    print(f"  {t} {n}d: map {100*m.iloc[-1]:+.2f}%  clean {100*c.iloc[-1]:+.2f}%  "
          f"PIT map {pit_of(m):.2f} clean {pit_of(c):.2f}"
          + (f"   (map claims {claim:+.2f}%)" if claim else ""))

print("\n" + "=" * 96)
print("DOES THE PITCHED C8 TRIGGER FIRE TODAY?")
print("=" * 96)
sp = clean_r("EEM", 63) - clean_r("EFA", 63)
pit = rolling_on_valid(sp, lambda x: x.rolling(252).rank(pct=True) * 100)
for rung in (1, 2, 5, 10, 16):
    print(f"   PIT63 <= {rung:2d}:  today {pit.iloc[-1]:.2f} -> "
          f"{'FIRES' if pit.iloc[-1] <= rung else 'does NOT fire'}")
last = pit[(pit <= 2)].index
print(f"   last day the pitched rung was live: {last[-1].date() if len(last) else 'never'}"
      f"  ({(pit.index[-1] - last[-1]).days} calendar days ago)")
print("\nDONE forensic")
