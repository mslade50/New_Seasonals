"""Round-1 tail-risk closer for all three candidates.

Jackson Hole is 2026-08-28 = JH-4, so ANY hold of 4 td or more from a Monday
entry (lag=1 -> entry Tuesday 08-25, so h>=3 spans it) carries the speech. And
the fragility dial's ma10(63d) is 89.5, near the top of its range.

Two splits, on the only cells with enough history to split:
  (1) jackson_hole IN the hold window vs OUT, for
        C4's plain XLI r5-washout, C5's duration-neutral curve leg, and
        C9's "gold thrust + GLD >10% off its high"
  (2) the fragility dial: episodes where the 10d-MA of the 63d dial was >= 80
      against the rest. The dial parquet only starts 2016 so this is a
      small-sample read and is reported as one.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pd.set_option("display.width", 220)

# ------------------------------------------------------------------ the dial
try:
    frag = pd.read_parquet(Path(__file__).resolve().parents[3] / "data" / "rd2_fragility.parquet")
    frag.index = pd.to_datetime(frag.index)
    dial = frag["63d"].rolling(10).mean()
    print("fragility ma10(63d): %s .. %s, live %.1f"
          % (dial.dropna().index[0].date(), dial.dropna().index[-1].date(),
             dial.dropna().iloc[-1]))
    print("  live reading is the %.0fth percentile of the series"
          % (100 * (dial.dropna() <= dial.dropna().iloc[-1]).mean()))
except Exception as e:                                            # pragma: no cover
    dial = None
    print("dial unavailable:", e)

JH = load_events(["jackson_hole"])["date"]
print("jackson_hole dates on file: %d, next %s"
      % (len(JH), str(JH[JH > pd.Timestamp("2026-08-22")].min().date())))


def jh_split(px, idx, mask, legs, h, label, min_gap=None):
    ret = vehicle_ret(px, legs, h, 1)
    sig = idx[np.asarray(mask.reindex(idx, fill_value=False).values, bool)
              & ret.notna().values]
    if len(sig) == 0:
        print("  %s: N=0" % label)
        return
    e = declusters(sig, min_gap or max(h, 5), idx)
    fl = event_in_window(e, idx, h, 1, ("jackson_hole",))
    v = ret.loc[e].values
    rows = [summarize(v[fl], f"{label} h={h}: JH IN hold"),
            summarize(v[~fl], f"{label} h={h}: JH OUT")]
    show(rows, None)
    if dial is not None:
        dv = dial.reindex(pd.DatetimeIndex(e)).values
        hi = dv >= 80
        ok = ~np.isnan(dv)
        rows = [summarize(v[hi & ok], f"{label} h={h}: dial ma10(63d) >= 80 (LIVE 89.5)"),
                summarize(v[ok & ~hi], f"{label} h={h}: dial < 80"),
                summarize(v[~ok], f"{label} h={h}: pre-2016, no dial")]
        show(rows, None)


print("\n" + "=" * 100)
print("C4 -- plain XLI r5rank<=5, long XLI  (the gated cell is N=4 and cannot be split)")
print("=" * 100)
p4 = close_panel(["XLI", "XLB", "XLE"]).dropna(how="any")
r5 = pct_rank(p4["XLI"], 5, 252)
for h in (3, 5):
    jh_split(p4, p4.index, r5 <= 5, [("XLI", 1.0)], h, "XLI washout")

print("\n" + "=" * 100)
print("C5 -- duration-neutral curve leg (long IEF / short 0.523 TLT) on TNX at a 52wh")
print("=" * 100)
p5 = close_panel(["^TNX", "TLT", "IEF"]).dropna(how="any")
tnx = p5["^TNX"]
LEVEL = (tnx / rolling_on_valid(tnx, lambda x: x.rolling(252).max()) - 1.0) >= -0.0025
POS = [("IEF", 1.0), ("TLT", -0.5223509038594482)]
for h in (5, 8, 10):
    jh_split(p5, p5.index, LEVEL, POS, h, "curve leg", min_gap=10)

print("\n" + "=" * 100)
print("C9 -- gold thrust WITH GLD >10%% off its 52w high (today's only populated read)")
print("=" * 100)
p9 = close_panel(["GLD", "GDX", "^TNX"]).dropna(how="any")
g = p9["GLD"]
M9 = (pct_rank(g, 21, 252) >= 95) & ((g / rolling_on_valid(g, lambda x: x.rolling(252).max()) - 1.0) <= -0.10)
for h in (3, 5, 10):
    jh_split(p9, p9.index, M9, [("GLD", 1.0)], h, "gold thrust in drawdown")
