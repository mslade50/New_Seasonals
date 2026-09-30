"""kA c2 round 1+2 - long duration when ^TNX sits at a WHOLE-PERCENT yield after a thrust.

Pre-specified cell (surface map c2): ^TNX closes within 5 bp BELOW a whole-percent level
(or first crosses one) after a 63d rise >= 50 bp while at/near its 252 max; long TLT/IEF
h=5/10. Built-in placebo: the identical rule on the .25/.50/.75 grids.

Definitions (fixed before reading any forward return):
  rise63     ^TNX - ^TNX 63 valid sessions ago >= 0.50 (yield points; 0.50 = 50 bp)
  near_max   ^TNX >= its 252 rolling max (incl. today) - 0.05
  parent     rise63 & near_max
  grid f     levels L = k + f, f in {0, .25, .5, .75}
  below(f)   0 <= Lup - TNX <= 0.05, Lup = next grid level at/above the close
  xfirst(f)  TNX >= Ldn and the prior 252-session max < Ldn (first close through Ldn in a year)
  cell(f)    parent & (below(f) | xfirst(f))

Live: 2026-09-18 ^TNX 4.998 (0.2 bp below 5.00); 5.006 on 09-16 was the first close
through 5.00 (the FOMC decision close). Entry lag 1 = the 09-21 close.
Forward measures: TLT, IEF (2002-07-30+) and -d(^TNX) in bp (2000+, vehicle-free).
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

px = close_panel(["^TNX", "TLT", "IEF", "SPY"])
idx = px.index
y = px["^TNX"]
yv = y.dropna()
rise63 = (yv - yv.shift(63)).reindex(idx)
max252 = rolling_on_valid(y, lambda x: x.rolling(252).max())
prior_max = rolling_on_valid(y, lambda x: x.shift(1).rolling(252).max())
parent = (rise63 >= 0.50) & (y >= max252 - 0.05)
parent = parent.reindex(idx, fill_value=False)


def grid_masks(f):
    lup = np.ceil(y - f - 1e-9) + f
    ldn = np.floor(y - f + 1e-9) + f
    below = ((lup - y) <= 0.05) & ((lup - y) >= 0)
    xfirst = (y >= ldn) & (prior_max < ldn)
    return below.fillna(False), xfirst.fillna(False), lup, ldn


# forward measures, lag 1
def fwd_dy_bp(h):
    """long-duration sign: -(yield change) in bp from close D+1 to D+1+h."""
    return -100 * (y.shift(-(1 + h)) - y.shift(-1))


FWD = {}
for h in (3, 5, 10):
    FWD[("TLT", h)] = vehicle_ret(px, [("TLT", 1.0)], h)
    FWD[("IEF", h)] = vehicle_ret(px, [("IEF", 1.0)], h)
    FWD[("dy", h)] = fwd_dy_bp(h)


def ep_rows(mask, lbl, gap=21):
    rows = []
    m = mask.reindex(idx, fill_value=False)
    d_all = idx[m.values]
    for key in [("TLT", 5), ("TLT", 10), ("IEF", 5), ("IEF", 10), ("dy", 5), ("dy", 10)]:
        r = FWD[key]
        d = d_all.intersection(r.dropna().index)
        if len(d) == 0:
            rows.append({"label": f"{lbl} {key}", "n": 0})
            continue
        e = declusters(d, gap, idx)
        v = r.loc[e].values
        if key[0] == "dy":
            s = {"label": f"{lbl} -dTNX bp h={key[1]}", "n": len(v),
                 "mean": v.mean(), "median": float(np.median(v)), "hit": 100 * (v > 0).mean(),
                 "base": r.dropna().mean(), "sign_p": sign_test(int((v > 0).sum()), len(v))}
        else:
            s = summarize(v, f"{lbl} {key[0]} h={key[1]}")
            s["base"] = 100 * r.dropna().mean()
            s["sign_p"] = sign_test(int((v > 0).sum()), len(v))
        s["n_days"] = len(d)
        rows.append(s)
    return rows


print("=== live ===")
for f in (0.0, 0.25, 0.5, 0.75):
    b, x, lup, ldn = grid_masks(f)
    print(f"  grid {f:.2f}: below {bool(b.iloc[-1])} (Lup {lup.iloc[-1]:.2f}, {100*(lup.iloc[-1]-y.iloc[-1]):.1f} bp)  "
          f"xfirst {bool(x.iloc[-1])}")
print(f"  rise63 {100*rise63.iloc[-1]:+.1f} bp, max252 {max252.iloc[-1]:.3f}, parent {bool(parent.iloc[-1])}")
b0, x0, _, _ = grid_masks(0.0)
print("  last 6 sessions  TNX / below0 / xfirst0 / parent:")
for d in idx[-6:]:
    print(f"   {d.date()} {y[d]:.3f} {bool(b0[d])} {bool(x0[d])} {bool(parent[d])}")

# ------------------------------------------------------------- whole vs placebo grids
cells = {}
for f in (0.0, 0.25, 0.5, 0.75):
    b, x, _, _ = grid_masks(f)
    cells[f] = (parent & (b | x)).reindex(idx, fill_value=False)
    show(ep_rows(cells[f], f"grid {f:.2f}"), f"CELL on grid {f:.2f} (parent & (below|xfirst)), 21td episodes")
plac = (cells[0.25] | cells[0.5] | cells[0.75])
show(ep_rows(plac, "PLACEBO pooled .25/.50/.75"), "PLACEBO pooled")
show(ep_rows(parent, "PARENT (rise63>=50bp & near 252 max)"), "PARENT")
show(ep_rows(parent & ~cells[0.0], "PARENT minus whole-percent cell"), "COMPLEMENT (discarded by the round gate)")
show(ep_rows(parent & ~cells[0.0] & ~plac, "PARENT off every grid"), "PARENT off every grid level")

# below-only and xfirst-only splits on the whole grid
b, x, lup, ldn = grid_masks(0.0)
show(ep_rows(parent & b, "whole BELOW only"), "whole-percent: within 5bp below only (TODAY's form)")
show(ep_rows(parent & x, "whole XFIRST only"), "whole-percent: first cross only (fired 09-16)")

# ------------------------------------------------------------- explicit episodes
print("\n=== whole-percent cell episodes (21td decluster), with level and forwards ===")
d = idx[cells[0.0].values]
e = declusters(d, 21, idx)
recs = []
for a in e:
    lv = lup[a] if b[a] else ldn[a]
    recs.append({"date": a.date(), "TNX": round(y[a], 3), "level": lv, "form": "below" if b[a] else "xfirst",
                 "rise63bp": round(100 * rise63[a], 1),
                 "TLT5": round(100 * FWD[("TLT", 5)].get(a, np.nan), 2),
                 "TLT10": round(100 * FWD[("TLT", 10)].get(a, np.nan), 2),
                 "IEF10": round(100 * FWD[("IEF", 10)].get(a, np.nan), 2),
                 "-dy5bp": round(FWD[("dy", 5)].get(a, np.nan), 1),
                 "-dy10bp": round(FWD[("dy", 10)].get(a, np.nan), 1),
                 "midterm": a.year % 4 == 2})
print(pd.DataFrame(recs).to_string(index=False))

# ------------------------------------------------------------- regime / NFP splits
print("\n=== NFP-in-hold split on the whole-percent cell (TLT h=10; live NFP at +9) ===")
r = FWD[("TLT", 10)]
dd = d.intersection(r.dropna().index)
ee = declusters(dd, 21, idx)
fl = event_in_window(ee, idx, 10, 1, ("nfp",))
show([summarize(r.loc[ee].values[fl], f"NFP IN (N={fl.sum()})"),
      summarize(r.loc[ee].values[~fl], f"NFP OUT (N={(~fl).sum()})")])
print("\n=== parent: NFP-in-hold split, TLT h=10 ===")
dd = idx[parent.values].intersection(r.dropna().index)
ee = declusters(dd, 21, idx)
fl = event_in_window(ee, idx, 10, 1, ("nfp",))
show([summarize(r.loc[ee].values[fl], f"NFP IN (N={fl.sum()})"),
      summarize(r.loc[ee].values[~fl], f"NFP OUT (N={(~fl).sum()})")])

# ------------------------------------------------------------- definition neighbours
print("\n=== definition neighbours on the whole grid vs pooled placebo (TLT h=5 / h=10, -dy10) ===")
rows = []
for band in (0.03, 0.05, 0.10):
    for rise in (0.30, 0.50, 0.75):
        par = ((rise63 >= rise) & (y >= max252 - 0.05)).reindex(idx, fill_value=False)
        for f_lbl, fs in (("whole", (0.0,)), ("placebo", (0.25, 0.5, 0.75))):
            m = pd.Series(False, index=idx)
            for f in fs:
                lup_ = np.ceil(y - f - 1e-9) + f
                ldn_ = np.floor(y - f + 1e-9) + f
                bb = ((lup_ - y) <= band) & ((lup_ - y) >= 0)
                xx = (y >= ldn_) & (prior_max < ldn_)
                m = m | (par & (bb | xx)).reindex(idx, fill_value=False)
            out = {"band_bp": int(band * 100), "rise_bp": int(rise * 100), "grid": f_lbl}
            for key in [("TLT", 5), ("TLT", 10), ("dy", 10)]:
                rr = FWD[key]
                dd = idx[m.values].intersection(rr.dropna().index)
                ee = declusters(dd, 21, idx)
                v = rr.loc[ee].values
                out[f"n_{key[0]}{key[1]}"] = len(v)
                out[f"{key[0]}{key[1]}"] = round((100 if key[0] != "dy" else 1) * np.mean(v), 3) if len(v) else np.nan
                out[f"hit_{key[0]}{key[1]}"] = round(100 * np.mean(v > 0), 0) if len(v) else np.nan
            rows.append(out)
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== base rates ===")
for key in [("TLT", 5), ("TLT", 10), ("IEF", 10), ("dy", 10)]:
    rr = FWD[key].dropna()
    print(f"  {key}: all-days mean {(100 if key[0] != 'dy' else 1)*rr.mean():+.3f}  hit {100*(rr>0).mean():.1f}%")
