"""C1 neighbours: ATR rung 1.5/2/2.5/3, KRE rung (-0.5%, 0, +0.5%), hedge
XLF-beta / KRE-beta / SPY-beta, h=1,2,3,5; plus non-earnings subset of the
parent (the live BAC session is a non-earnings, intraday-led slide) and FOMC
in-window split. All date-averaged, declustered gap=max(h,5)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pitch_lab import *  # noqa
from k2_common import *  # noqa

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
BANKS = ["JPM", "BAC", "C", "WFC", "GS", "MS", "BNY", "STT"]
px, cal, P = load_panels(BANKS + ["SPY", "XLF", "KRE"])
D = derive(px, cal)
C = P["Close"]
ret1, sh, gap = D["ret1"], D["shock"], D["gap"]
beta = {b: rolling_beta(ret1, b) for b in ["XLF", "KRE", "SPY"]}
spy1, kre1 = ret1["SPY"], ret1["KRE"]

ern = pd.read_parquet(ROOT / "data" / "earnings_calendar.parquet", columns=["ticker", "date"])
ern["date"] = pd.to_datetime(ern["date"])
EARN = pd.DataFrame(False, index=cal, columns=BANKS)
for t in BANKS:
    pos = cal.searchsorted(pd.DatetimeIndex(ern.loc[ern["ticker"] == t, "date"]))
    for p in pos:
        if p < len(cal):
            EARN.iloc[p, BANKS.index(t)] = True
            if p + 1 < len(cal):
                EARN.iloc[p + 1, BANKS.index(t)] = True


def pan(h: int, hedge: str) -> pd.DataFrame:
    F = fwd_panel(C, h)
    return F[BANKS].sub(beta[hedge][BANKS].mul(F[hedge], axis=0))


def mask(atr: float, kre_min, spy_min: float = -0.01) -> pd.DataFrame:
    m = sh[BANKS].le(-atr)
    cond = (spy1 > spy_min)
    if kre_min is not None:
        cond = cond & (kre1 >= kre_min)
    return m.apply(lambda c: c & cond.fillna(False))


def row(M: pd.DataFrame, h: int, hedge: str, label: str) -> dict:
    e = events_from_mask(M, BANKS)
    p = pan(h, hedge)
    e["v"] = lookup(p, e["date"], e["name"])
    ctl = p.loc[cal >= pd.Timestamp("2006-06-22")].mean(axis=1).mean()
    return date_stats(e, "v", cal, max(h, 5), label, ctl)


for h in (1, 2, 3, 5):
    rows = []
    for atr in (1.5, 2.0, 2.5, 3.0):
        for km, kl in [(None, "no KRE gate"), (-0.005, "KRE>=-0.5%"), (0.0, "KRE>=0"), (0.005, "KRE>=+0.5%")]:
            rows.append(row(mask(atr, km), h, "XLF", f"h={h} atr>={atr} {kl}"))
    show(rows, f"neighbours, XLF-beta hedge, h={h}")

rows = []
for h in (1, 2, 3, 5):
    for hedge in ("XLF", "KRE", "SPY"):
        rows.append(row(mask(2.0, 0.0), h, hedge, f"h={h} CELL hedge {hedge}-beta"))
show(rows, "CELL by hedge")

# non-earnings shocks (the live BAC type), with and without KRE gate
rows = []
for h in (1, 2, 3, 5):
    for atr in (1.5, 2.0):
        for km, kl in [(None, "no KRE"), (0.0, "KRE>=0")]:
            M = mask(atr, km) & ~EARN
            rows.append(row(M, h, "XLF", f"h={h} NON-EARN atr>={atr} {kl}"))
            G = gap[BANKS] / ret1[BANKS]
            Mi = M & (G < 0.5)
            rows.append(row(Mi, h, "XLF", f"h={h} NON-EARN intraday-led atr>={atr} {kl}"))
show(rows, "non-earnings shocks (live BAC 09-14 is non-earnings, gap share 0.11)")

# FOMC decision inside h-window for the 1.5 ATR no-KRE parent (largest sample)
for h in (1, 5):
    e = events_from_mask(mask(1.5, None), BANKS)
    p = pan(h, "XLF")
    e["v"] = lookup(p, e["date"], e["name"])
    s = date_series(e, "v")
    keep = declusters(s.index, max(h, 5), cal)
    fl = event_in_window(keep, cal, h, 1, ("fomc_decision",))
    v = s.loc[keep].values
    show([summarize(v[fl], f"h={h} parent1.5 FOMC in window"),
          summarize(v[~fl], f"h={h} parent1.5 FOMC out")], "FOMC split")

print("\nBAC Wilder ATR 09-14:", round(D["atr"]["BAC"].iloc[-1], 3),
      " % of price:", round(100 * D["atr"]["BAC"].iloc[-1] / C["BAC"].iloc[-1], 2),
      " beta vs XLF:", round(beta["XLF"]["BAC"].iloc[-1], 2),
      " XLF ATR%:", round(100 * D["atrp"]["XLF"].iloc[-1], 2))
