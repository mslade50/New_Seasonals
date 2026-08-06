"""Post-event SVXY entries: is there a tradeable short-vol window AFTER
each macro event (or holding through it from the event-day open)?

Instrument: synthetic -0.5x short-vol legs from UVXY OHLC (validated
0.9967 daily corr vs real SVXY in the -0.5x era). Legs:
  overnight_t = -(1/3) x (UVXY open_t / close_{t-1} - 1)
  intraday_t  = -(1/3) x (UVXY close_t / open_t - 1)
Windows compound legs. 2011-10+ throughout (UVXY inception bound).

Grid per event: entries {open td0, close td0, close td+1},
holds to {close +1, +2, +3, +5} (open0 also exits close0).
Events: fomc_decision (ALL + ex-midterm), cpi, nfp, opex, quad_witching,
vix_expiry. Note: CPI/NFP release 8:30 -> open td0 is POST-event;
FOMC 2pm -> open td0 holds THROUGH the announcement.

~120 cells: expect ~6 lone |t|>=2 by chance. Look for coherent event
patterns, not isolated cells. Decay baseline: +12.5 bps/day long-SVXY.

Run: python scratch/svxy_postevent_grid.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from macro_calendar import event_dates  # noqa: E402


def load_uvxy() -> pd.DataFrame:
    mp = pd.read_parquet(ROOT / "data" / "master_prices.parquet",
                         columns=["ticker", "date", "Open", "Close"])
    df = mp[mp["ticker"] == "UVXY"].set_index("date").sort_index()
    df.index = pd.to_datetime(df.index).normalize()
    df = df[~df.index.duplicated(keep="last")][["Open", "Close"]]
    return df


u = load_uvxy()
ovn = -(u["Open"] / u["Close"].shift(1) - 1) / 3.0     # short-vol overnight
intra = -(u["Close"] / u["Open"] - 1) / 3.0            # short-vol intraday
idx = u.index


def window(p: int, entry: str, exit_k: int) -> float:
    """Compound short-vol return from the entry point to close of p+exit_k.

    entry: 'open0' (open of event session p), 'close0', 'close1'.
    """
    start = {"open0": p, "close0": p + 1, "close1": p + 2}[entry]
    hi = p + exit_k
    if hi >= len(idx) or start > hi or p < 1:
        return np.nan
    total = 1.0
    if entry == "open0":
        total *= 1 + intra.iloc[p]
        start = p + 1
    for i in range(start, hi + 1):
        total *= (1 + ovn.iloc[i]) * (1 + intra.iloc[i])
    return total - 1


def cell(vals: list[float]) -> tuple[float, float, int, float]:
    x = pd.Series(vals).dropna()
    if len(x) < 8:
        return (np.nan, np.nan, len(x), np.nan)
    t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    return (x.mean() * 1e4, t, len(x), (x > 0).mean())


EVENTS = ["fomc_decision", "cpi", "nfp", "opex", "quad_witching", "vix_expiry"]
ENTRIES = ["open0", "close0", "close1"]
EXITS = [0, 1, 2, 3, 5]

for ev in EVENTS:
    dts = event_dates(ev)
    dts = dts[(dts >= idx.min()) & (dts <= idx.max() - pd.Timedelta(days=12))]
    variants = [("ALL", dts)]
    if ev == "fomc_decision":
        variants.append(("ex-midterm", dts[[d.year % 4 != 2 for d in dts]]))
    for tag, sub in variants:
        print(f"\n===== {ev} ({tag}, N={len(sub)}) — long SVXY, bps / t =====")
        print(f"{'entry':8s}" + "".join(f"  ->close+{k:<7d}" for k in EXITS))
        for entry in ENTRIES:
            cells = []
            for k in EXITS:
                if (entry, k) in (("close0", 0), ("close1", 0), ("close1", 1)):
                    cells.append(f"{'—':>15s}")
                    continue
                vals = [window(idx.searchsorted(d), entry, k) for d in sub
                        if idx.searchsorted(d) < len(idx)]
                m, t, n, hit = cell(vals)
                flag = "*" if abs(t) >= 2 else " "
                cells.append(f"{m:+7.0f}/{t:+5.1f}{flag}  ")
            print(f"{entry:8s}" + "".join(cells))
