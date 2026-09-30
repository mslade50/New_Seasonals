"""sb1 TRV round-3 dev: horizon scan, MOC vs close-anchored -0.5 ATR limit (whole variants), stop sensitivity, loser paths."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sb1_engine as E

import numpy as np
import pandas as pd

T = "TRV"
E.PX = load_prices([T, "XLF", "SPY"])
f = E.PX[T]
idx = f.index
O, H, L, C = (f[k].values for k in ("Open", "High", "Low", "Close"))
atr = wilder_atr(f["High"], f["Low"], f["Close"])
anc = E.anchors(idx)
A = pd.DatetimeIndex([idx[p] for p in anc.values()])
px = pd.DataFrame({T: f["Close"]})

for lag in (1, 2):
    show(horizon_scan(px, A, [(T, 1.0)], hs=tuple(range(1, 22)), lag=lag, min_gap=100), f"horizon scan lag=T+{lag}")

# entry variants, exit fixed at anchor+2+21 close
HOLD = 21
rows = []
for name in ("MOC T+1", "MOC T+2", "LMT -0.5ATR (A..A+2)", "LMT -0.5ATR (A+1..A+2)"):
    res, fills = [], 0
    for y, p in anc.items():
        ex = p + 2 + HOLD
        if ex >= len(idx):
            continue
        b = p - 1
        if name.startswith("MOC"):
            e = C[p + int(name[-1])]
        else:
            lim = C[b] - 0.5 * atr[b]
            days = range(p, p + 3) if "(A.." in name else range(p + 1, p + 3)
            e = None
            for d in days:
                if L[d] <= lim:
                    e = min(O[d], lim)
                    break
            if e is None:
                res.append(0.0)
                continue
        fills += 1
        res.append(C[ex] / e - 1)
    r = np.array(res)
    rows.append({"variant": name, "signals": len(r), "fills": fills, "fill%": round(100 * fills / len(r), 0),
                 "mean_per_signal%": round(100 * r.mean(), 2),
                 "mean_if_filled%": round(100 * r[r != 0].mean(), 2) if fills else np.nan,
                 "hit_if_filled": f"{int((r>0).sum())}/{fills}"})
show(rows, "entry variants (whole variants, unfilled = 0)")

# stop sensitivity from MOC T+2 entry; stop = entry - k*ATR(board bar); optional target at +2.79 ATR (board blended)
srows = []
for k in (0.8, 1.0, 1.3, 1.6, 2.0, None):
    for tgt in (None, 2.79):
        out, stopped, hit_t = [], 0, 0
        for y, p in anc.items():
            e0 = p + 2
            ex = e0 + HOLD
            if ex >= len(idx):
                continue
            a = atr[p - 1]
            e = C[e0]
            r = C[ex] / e - 1
            for d in range(e0 + 1, ex + 1):
                if k is not None and L[d] <= e - k * a:
                    r = min(O[d], e - k * a) / e - 1; stopped += 1; break
                if tgt is not None and H[d] >= e + tgt * a:
                    r = max(O[d], e + tgt * a) / e - 1; hit_t += 1; break
            out.append(r)
        o = np.array(out)
        srows.append({"stop_atr": k, "target_atr": tgt, "n": len(o), "stopped": stopped, "tgt_hit": hit_t,
                      "mean%": round(100 * o.mean(), 2), "hit": f"{int((o>0).sum())}/{len(o)}", "worst%": round(100 * o.min(), 2)})
show(srows, "stop / target sensitivity (MOC T+2, 21td time exit)")

paths = episode_paths(px, A, [(T, 1.0)], 21, lag=2)
fin = paths[21]
los = paths[fin < 0]
print("\nloser paths (%), day 1/2/5/10/21 and min:")
print((100 * los[[1, 2, 5, 10, 21]]).round(2).assign(min=(100 * los.min(axis=1)).round(2)).to_string())
print(f"winners day-2 mean {100*paths[fin>0][2].mean():+.2f}%  day-5 {100*paths[fin>0][5].mean():+.2f}%  "
      f"min-path mean {100*paths[fin>0].min(axis=1).mean():+.2f}%")
print(f"losers day-2 mean {100*los[2].mean():+.2f}%  day-5 {100*los[5].mean():+.2f}%")
print(f"MAE (min path) all: median {100*paths.min(axis=1).median():.2f}%, in ATR units median "
      f"{np.median([paths.loc[d].min() / (atr[idx.get_loc(d)-1]/C[idx.get_loc(d)-1]) for d in paths.index]):.2f}")
print(f"\ntoday: close {C[-1]:.2f} ({idx[-1].date()}), Wilder ATR {atr[-1]:.2f} ({100*atr[-1]/C[-1]:.2f}%), "
      f"-0.5ATR limit {C[-1]-0.5*atr[-1]:.2f}, 1.6 ATR stop from close {C[-1]-1.6*atr[-1]:.2f}")
