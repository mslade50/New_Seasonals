"""Jackson Hole (2026-08-28, 4 td out): what do SPY / TLT / GLD do into and
through the symposium Friday? Anchors: the session k td before the JH date,
lag-1 entries per posts doctrine."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
import pitch_lab as pl

ev = pl.load_events(["jackson_hole"])
jh = pd.DatetimeIndex(sorted(ev["date"].unique()))
px = pl.load_prices(["SPY", "TLT", "GLD", "^VIX"])

for tkr in ["SPY", "TLT", "GLD", "^VIX"]:
    df = px[tkr]
    close = df["Close"]
    dates = close.index
    print(f"\n===== {tkr} (history {dates[0].date()}..{dates[-1].date()})")
    # anchor = session k td BEFORE the JH Friday; entry at anchor close (lag=0
    # here because the anchor is itself in the past relative to the event),
    # but the tradeable read for us tonight is: enter Monday close (k=4),
    # exit the JH close (h=4 from that anchor).
    for k, h, label in [(4, 4, "Mon close -> JH Fri close"),
                        (5, 4, "prior Fri close -> JH Fri close"),
                        (1, 1, "Thu close -> JH Fri close (speech day)"),
                        (0, 5, "JH Fri close -> next week")]:
        rows = []
        for d in jh:
            pos = dates.searchsorted(d)
            if pos >= len(dates) or pos - k < 0:
                continue
            # nearest session on/before the JH date
            if dates[pos] != d:
                pos -= 1
            a = pos - k
            b = a + h
            if b >= len(dates) or a < 0:
                continue
            rows.append((dates[a], close.iloc[b] / close.iloc[a] - 1.0))
        vals = np.array([r[1] for r in rows])
        if not len(vals):
            continue
        s = pl.summarize(vals, label)
        wins = int((vals > 0).sum())
        print(f"  k={k} h={h} {label}: N={s['n']} mean={s['mean_pct']:+.2f}% "
              f"med={s['median_pct']:+.2f}% hit={s['hit']:.0f}% "
              f"sign_p={pl.sign_test(wins, len(vals)):.4f} "
              f"worst={vals.min()*100:+.1f}% best={vals.max()*100:+.1f}%")
        for e in pl.era_split(pd.DatetimeIndex([r[0] for r in rows]), vals):
            print(f"      era {e}")
