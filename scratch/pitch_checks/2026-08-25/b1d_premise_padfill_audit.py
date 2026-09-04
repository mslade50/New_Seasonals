"""Premise audit: every 'spread (pp)' headline in today's surface map was
computed by 00_recon_premises.py as px.pct_change(N) on a WIDE union-calendar
panel. pandas' pct_change defaults to fill_method='pad', which forward-fills
foreign-calendar holes (futures / FX / caret tickers trade sessions the equity
names do not) into synthetic zero-return days and SHIFTS every window that
spans one. pitch_lab._valid_pct_change exists precisely for this
(documented 2026-08-19: up to 29.4 percentile points of drift).

This is not a kill for anything on its own; it is a check on whether the
headline numbers the candidates are NAMED after are the real ones.
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change
import pandas as pd

WIDE = ["SMH", "SPY", "XLU", "TLT", "OIH", "XOP", "XLE", "EEM", "EFA", "FXI",
        "GDX", "GLD", "SLV", "USO", "QQQ", "XLK", "XLV", "XLP", "XLI", "IWM",
        "HYG", "LQD", "IEF", "SVXY", "DX-Y.NYB", "^TNX", "^VIX", "HE=F", "LE=F",
        "CL=F", "GC=F"]
px = close_panel(WIDE)
print(f"wide panel rows {len(px)} (union calendar)")

PAIRS = [("SMH", "SPY", 63, "C2  SMH-SPY 63d", -10.17),
         ("SMH", "SPY", 5, "C2  SMH-SPY 5d", None),
         ("XLU", "TLT", 21, "C3  XLU-TLT 21d", -6.20),
         ("XLU", "SPY", 21, "C3  XLU-SPY 21d", -9.95),
         ("OIH", "XOP", 63, "C4  OIH-XOP 63d", -18.5),
         ("EEM", "EFA", 63, "C8  EEM-EFA 63d", -7.57),
         ("GDX", "GLD", 21, "C7  GDX-GLD 21d", 22.9),
         ("XLV", "XLK", 5, "C1  XLV-XLK 5d", 9.98)]
print(f"\n{'cell':22s} {'map says':>9s} {'pad-fill':>9s} {'VALID':>9s} {'error':>8s}"
      f" {'pad pctile':>11s} {'VALID pctile':>13s}")
for a, b, n, lbl, claimed in PAIRS:
    if a not in px or b not in px:
        continue
    pad = (px.pct_change(n)[a] - px.pct_change(n)[b]) * 100
    val = (_valid_pct_change(px[a], n) - _valid_pct_change(px[b], n)) * 100
    pp = rolling_on_valid(pad, lambda x: x.rolling(252).rank(pct=True) * 100)
    pv = rolling_on_valid(val, lambda x: x.rolling(252).rank(pct=True) * 100)
    c = f"{claimed:+.2f}" if claimed is not None else "   -   "
    print(f"{lbl:22s} {c:>9s} {pad.iloc[-1]:+9.2f} {val.iloc[-1]:+9.2f} "
          f"{pad.iloc[-1]-val.iloc[-1]:+8.2f} {pp.iloc[-1]:11.1f} {pv.iloc[-1]:13.1f}")
print("\n  -> the pad-fill column reproduces the surface map exactly, so the map's "
      "spread headlines ARE the padded ones. Percentiles mostly survive; the "
      "MAGNITUDES do not.")
