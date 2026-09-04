"""B1 recon: every live calendar event x every asset class, at TODAY'S offset.

Today's data date (2026-08-10) sits at a fixed session offset from each live
event.  The historical analogue of "publish today, enter MOC tonight" is a mask
on dates at the SAME offset, with lag=1 entry.  So this table is literally the
set of trades today's calendar makes available, priced against a
trading-day-of-month matched control (the 2026-08-10 lesson: an all-days
control is not a control when the anchor has a fixed month position).

Verdict feeds scratch/pitch_checks/2026-08-11/00_surface_map.md.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, load_events, fwd_lag, declusters, summarize, sign_test,
)

warnings.filterwarnings("ignore")

CLASSES = {
    "us_large": ["SPY", "QQQ"],
    "us_small": ["IWM"],
    "rates": ["TLT", "IEF"],
    "credit": ["HYG", "LQD"],
    "gold_miners": ["GLD", "GDX"],
    "other_metals": ["SLV"],
    "energy": ["USO", "UNG", "DBC"],
    "dollar_fx": ["UUP", "DX-Y.NYB"],
    "international": ["EFA", "EEM", "FXI"],
    "volatility": ["SVXY"],
}
TICKERS = [t for v in CLASSES.values() for t in v]
CLS_OF = {t: c for c, v in CLASSES.items() for t in v}

# today's data bar is 2026-08-10; sessions BEFORE each event from that bar
TODAY_OFFSETS = {"cpi": 2, "ppi": 3, "vix_expiry": 7, "opex": 9, "jackson_hole": 14}
HORIZONS = (1, 2, 3, 5, 10)

px = close_panel(TICKERS)
ev = load_events(list(TODAY_OFFSETS))
all_dates = px.index

pos = pd.Series(np.arange(len(all_dates)), index=all_dates)


def tdom_of(idx: pd.DatetimeIndex) -> pd.Series:
    """1-indexed trading day of month for every date in the panel."""
    s = pd.Series(idx, index=idx)
    return s.groupby([idx.year, idx.month]).cumcount() + 1


TDOM = tdom_of(all_dates)


def anchor_dates(kind: str, offset: int) -> pd.DatetimeIndex:
    """Dates exactly `offset` sessions before an event of `kind`."""
    dates = pd.DatetimeIndex(sorted(ev.loc[ev["event"] == kind, "date"].unique()))
    out = []
    for d in dates:
        loc = all_dates.searchsorted(d)
        if loc >= len(all_dates):
            continue
        # the event's own session (or the next session if it fell on a holiday)
        j = loc - offset
        if 0 <= j < len(all_dates):
            out.append(all_dates[j])
    return pd.DatetimeIndex(sorted(set(out)))


rows = []
for kind, off in TODAY_OFFSETS.items():
    anch = anchor_dates(kind, off)
    anch = declusters(anch, 5, all_dates)
    for tkr in TICKERS:
        s = px[tkr].dropna()
        if len(s) < 500:
            continue
        a = anch[anch.isin(s.index)]
        if len(a) < 8:
            continue
        # tdom of the ENTRY session (anchor + 1)
        ent_tdom = TDOM.reindex(all_dates)[pos[a].values + 1].values
        for h in HORIZONS:
            f = fwd_lag(s, h, lag=1)
            v = f.reindex(a).dropna()
            if len(v) < 8:
                continue
            fa = f.reindex(s.index).dropna()
            # tdom-matched control: same entry-month-position, not a trigger
            ctl_mask = TDOM.reindex(fa.index).isin(set(ent_tdom.tolist())) & ~fa.index.isin(a)
            ctl = fa[ctl_mask]
            own = fa
            st = summarize(v.values)
            wins = int((v.values > 0).sum())
            rows.append(dict(
                event=kind, tkr=tkr, cls=CLS_OF[tkr], h=h, n=len(v),
                mean=st["mean_pct"], hit=st["hit"], t=st["t"],
                ctl=ctl.mean() * 100 if len(ctl) else np.nan,
                own=own.mean() * 100,
                excess=st["mean_pct"] - (ctl.mean() * 100 if len(ctl) else np.nan),
                signp=sign_test(wins, len(v)),
            ))

df = pd.DataFrame(rows)
df["absx"] = df["excess"].abs()

print("=== every cell, excess over tdom-matched control, sorted by |excess| ===")
top = df.sort_values("absx", ascending=False).head(45)
for _, r in top.iterrows():
    print(f"{r['event']:>13} {r['tkr']:>9} h={r['h']:<3} N={r['n']:<4} "
          f"mean={r['mean']:+7.3f} ctl={r['ctl']:+7.3f} own={r['own']:+7.3f} "
          f"excess={r['excess']:+7.3f} hit={r['hit']:5.1f} signp={r['signp']:.4f}")

print("\n=== CPI row only (today is the session before CPI) ===")
c = df[df.event == "cpi"].sort_values(["h", "excess"], ascending=[True, False])
for h in HORIZONS:
    print(f"-- h={h} --")
    for _, r in c[c.h == h].iterrows():
        print(f"   {r['tkr']:>9} ({r['cls']:>13}) N={r['n']:<4} mean={r['mean']:+7.3f} "
              f"ctl={r['ctl']:+7.3f} excess={r['excess']:+7.3f} hit={r['hit']:5.1f} p={r['signp']:.4f}")

df.to_csv(Path(__file__).with_name("01_event_class_recon.csv"), index=False)
print("\nwrote 01_event_class_recon.csv")
