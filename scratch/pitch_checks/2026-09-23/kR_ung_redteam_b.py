"""RED-TEAM round b: cluster robustness, ATR regime of today vs episodes,
era x EIA cross, expiry-week subset, book-ledger UNG rows."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
d = load_prices(["UNG"])["UNG"].astype(float)
c, v = d["Close"], d["Volume"]
idx = c.index
r1 = c.pct_change()
vr = v / v.rolling(63).mean()
pc = c.shift(1)
tr = pd.concat([d["High"] - d["Low"], (d["High"] - pc).abs(),
                (d["Low"] - pc).abs()], axis=1).max(axis=1)
atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
h2 = c.shift(-3) / c.shift(-1) - 1
trig = ((r1 >= 0.05) & (vr >= 3)).fillna(False)
sig = idx[trig.values & h2.notna().values]


def line(vals, lab):
    vals = np.asarray(vals, float)
    vals = vals[~np.isnan(vals)]
    w, n = int((vals > 0).sum()), len(vals)
    return (f"{lab}: N={n} mean {100*vals.mean():+.2f}% median "
            f"{100*np.median(vals):+.2f}% rec {w}-{n-w} sign p {sign_test(w, n):.4f}"
            f" boot {bootstrap_p_le0(vals):.3f}")


for gap in (2, 10, 21):
    e = declusters(sig, gap, idx)
    print(line(h2.loc[e].values, f"decluster {gap}td"))
e = declusters(sig, 2, idx)
x = h2.loc[e]
mo = x.groupby([x.index.year, x.index.month]).mean()
print(line(mo.values, "month-cluster means (one obs per y-m)"))

# ATR regime
atrp = (atr / c).loc[e]
print(f"\nATR% at signal: episodes median {100*atrp.median():.2f}% "
      f"min {100*atrp.min():.2f}% max {100*atrp.max():.2f}%; today "
      f"{100*atr.iloc[-1]/c.iloc[-1]:.2f}% (pctile among episodes "
      f"{100*(atrp < atr.iloc[-1]/c.iloc[-1]).mean():.0f})")
mv = ((c - pc) / atr).loc[e]
print(f"thrust size in signal-ATR: episodes median {mv.median():.2f}, today "
      f"{(c.iloc[-1]-pc.iloc[-1])/atr.iloc[-1]:.2f}")
today_atrp = atr.iloc[-1] / c.iloc[-1]
print("worst h2 mapped to TODAY's ATR% (pct / today ATR%):",
      {k.date(): round(val / today_atrp, 2) for k, val in x.nsmallest(3).items()})
for k in (1.5, 2.0, 2.5):
    print(f"  k={k}: worst pct-mapped {x.min()/today_atrp/k:+.2f}R")
# ATR-% regime split
lo = atrp.values <= np.median(atrp.values)
print(line(x.values[lo], "episodes with ATR% <= median (today-like low vol)"))
print(line(x.values[~lo], "episodes with ATR% > median"))
low3 = atrp.values < 0.035
print(line(x.values[low3], "episodes ATR% < 3.5%"))

# EIA inside x era
thu = np.array([any(idx[idx.get_loc(dd) + k].weekday() == 3 for k in (2, 3)) for dd in e])
yr = e.year
for lab, m in [("EIA-in 2010+", thu & (yr >= 2010)), ("EIA-in pre-2010", thu & (yr < 2010)),
               ("EIA-out 2010+", ~thu & (yr >= 2010)), ("EIA-in 2018+", thu & (yr >= 2018)),
               ("EIA-out 2018+", ~thu & (yr >= 2018))]:
    print(line(x.values[m], lab))
# chance that 3 worst all land in the inside group
from math import comb
n_in = int(thu.sum())
print(f"P(3 worst all in EIA-in group | random) = {comb(n_in,3)/comb(len(e),3):.3f}")

# t+2 checkpoint vs signal close
t2 = (c.shift(-2) / c - 1).loc[e]
for thr in (-0.02, -0.03, -0.04):
    m = t2.values < thr
    print(line(x.values[m], f"t+2 close < signal close {100*thr:+.0f}%"),
          "| complement:", line(x.values[~m], "")[:60])
print(f"levels on 10.86: -2% {10.86*0.98:.2f}, -3% {10.86*0.97:.2f}, -4% {10.86*0.96:.2f}")

# book ledger UNG rows
bt = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
tcol = [cc for cc in bt.columns if cc.lower() in ("ticker", "symbol")][0]
u = bt[bt[tcol].astype(str).str.upper() == "UNG"]
cols = [cc for cc in bt.columns if any(k in cc.lower() for k in
        ("strategy", "date", "entry", "exit", "direction", "r"))][:14]
print("\nledger UNG rows:")
print(u[cols].to_string())
