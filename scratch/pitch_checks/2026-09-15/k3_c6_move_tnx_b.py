import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

# C6 round 1b: lookback and level neighbours of the joint bond-vol + yield thrust,
# each with its own placebo ladder, and concentration of the 90/90 cell.
pd.set_option("future.no_silent_downcasting", True)
TK = ["^MOVE", "^TNX", "TLT", "IEF"]
raw = close_panel(TK)
cal = raw["TLT"].dropna().index
px = raw.reindex(cal)
mv = raw["^MOVE"]
tx = raw["^TNX"]
lvl_pct = rolling_on_valid(mv, lambda x: x.rolling(252).rank(pct=True) * 100).reindex(cal).ffill(limit=1)
R = {}
for n in (3, 5, 10):
    R[f"M{n}"] = pct_rank(mv, n).reindex(cal).ffill(limit=1)
    R[f"T{n}"] = pct_rank(tx, n).reindex(cal).ffill(limit=1)
live = pd.Timestamp("2026-09-14")
print("LIVE", {k: round(v.loc[live], 1) for k, v in R.items()}, "MOVE level pctile", round(lvl_pct.loc[live], 1))

fomc = pd.DatetimeIndex(load_events(["fomc_decision"])["date"])
fpos, fkept = anchor_positions(cal, fomc, 0)
fpos = np.array(fpos)
N = len(cal)

DEFS = {
    "r5 90/90 (pitched)": lambda d: (R["M5"].reindex(d) >= 90) & (R["T5"].reindex(d) >= 90),
    "r5 85/85": lambda d: (R["M5"].reindex(d) >= 85) & (R["T5"].reindex(d) >= 85),
    "r5 80/80": lambda d: (R["M5"].reindex(d) >= 80) & (R["T5"].reindex(d) >= 80),
    "r3 90/90": lambda d: (R["M3"].reindex(d) >= 90) & (R["T3"].reindex(d) >= 90),
    "r10 90/90": lambda d: (R["M10"].reindex(d) >= 90) & (R["T10"].reindex(d) >= 90),
    "r10 80/80": lambda d: (R["M10"].reindex(d) >= 80) & (R["T10"].reindex(d) >= 80),
    "MOVE level pct>=90 & TNX r5>=90": lambda d: (lvl_pct.reindex(d) >= 90) & (R["T5"].reindex(d) >= 90),
    "MOVE level pct>=80 & TNX r5>=80": lambda d: (lvl_pct.reindex(d) >= 80) & (R["T5"].reindex(d) >= 80),
    "MOVE r5>=90 & TNX r10>=90": lambda d: (R["M5"].reindex(d) >= 90) & (R["T10"].reindex(d) >= 90),
}
walked = 0
for V in ["TLT", "IEF"]:
    for h in (1, 3):
        ret = fwd_lag(px[V], h, 1)
        rows = []
        for name, fn in DEFS.items():
            means = {}
            for shift in range(-5, 6):
                sp = fpos - 2 + shift
                sp = sp[(sp >= 0) & (sp < N)]
                d = cal[sp]
                m = fn(d).values
                x = ret.reindex(d).values[m]
                x = x[~np.isnan(x)]
                means[shift] = (x.mean() if len(x) else np.nan, x)
                walked += 1
            x0 = means[0][1]
            t0 = means[0][0]
            ladder = np.array([means[s][0] for s in range(-5, 6)])
            rank = int(np.nansum(ladder > t0)) + 1 if len(x0) else np.nan
            r = summarize(x0, name)
            if r["n"]:
                w = int((x0 > 0).sum())
                r["rec"] = f"{w}-{len(x0)-w}"
                r["sign_p"] = round(sign_test(w, len(x0)), 4)
                srt = np.sort(x0)[::-1]
                r["top1_share"] = round(100 * srt[0] / x0.sum(), 0) if x0.sum() != 0 else np.nan
            r["placebo_rank"] = f"{rank} of {int(np.isfinite(ladder).sum())}"
            rows.append(r)
        show(rows, f"{V} h={h}: neighbour definitions at k=-2, lag 1 (placebo rank over k=-7..+3)")
print(f"\ncells walked: {walked}")
