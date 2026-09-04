"""Crisis-alpha track: final summary stats for the writeup."""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
NAV = 750_000.0

sleeves = pd.read_parquet(HERE / "ca_sleeves.parquet")
frag = pd.read_parquet(HERE / "ca_frag.parquet")["frag63_ma10"]
book_mon = pd.read_parquet(HERE / "ca_book_monthly.parquet")
book_mon.index = pd.PeriodIndex.from_ordinals(book_mon.index.values, freq="M")

win = (pd.Period("2016-08"), pd.Period("2026-06"))

def mstats(mon, label):
    mon = mon[(mon.index >= win[0]) & (mon.index <= win[1])]
    eq = (1 + mon).cumprod()
    yrs = len(mon) / 12
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    vol = mon.std() * np.sqrt(12)
    sharpe = mon.mean() / mon.std() * np.sqrt(12)
    dd = (eq / eq.cummax() - 1).min()
    print(f"{label:<28} CAGR {cagr*100:6.2f}%  vol {vol*100:5.2f}%  Sharpe(0%) {sharpe:5.2f}  "
          f"maxDD {dd*100:6.2f}%  N={len(mon)}")
    return mon

b = book_mon["ret"]
bm = mstats(b, "book (flat 750k, exit-mo)")

for col in ["vxxp2_55", "vxxp5_55", "put_55", "putspread_55"]:
    s = sleeves[col].groupby(sleeves[col].index.to_period("M")).sum()
    sm = mstats(s, f"sleeve {col}")
    cm = mstats((b.reindex(s.index).fillna(0) + s), f"book + {col}")
    print(f"   corr(book, {col}) = {b.reindex(sm.index).corr(sm):+.3f}")

# sleeve standalone dollar drawdown
print("\nsleeve $ drawdowns (cum PnL, flat NAV):")
for col in sleeves.columns:
    eq = sleeves[col].fillna(0).cumsum() * NAV
    dd = (eq - eq.cummax()).min()
    print(f"  {col:<14} total ${eq.iloc[-1]:+10,.0f}  maxDD ${dd:+10,.0f}")

# book worst months and whether gate was on
print("\n12 worst book months and gate state (frag mo-mean, vxxp5 sleeve that month $):")
s5 = sleeves["vxxp5_55"].groupby(sleeves["vxxp5_55"].index.to_period("M")).sum() * NAV
fm = frag.groupby(frag.index.to_period("M")).mean()
worst = b[(b.index >= win[0])].nsmallest(12)
for p, r in worst.items():
    print(f"  {p}  book {r*100:+6.2f}%  frag {fm.get(p, np.nan):5.1f}  hedge ${s5.get(p, 0):+8,.0f}")

# turnover / trade count of the tactical vol sleeve
pos_col = pd.read_parquet(HERE / "ca_frag.parquet")  # recompute gate
f = frag
raw_on = pd.Series(np.nan, index=f.index)
raw_on[f >= 55] = 1.0
raw_on[f < 50] = 0.0
gate = raw_on.ffill().fillna(0.0)
switches = gate.diff().abs().sum()
print(f"\ngate switches 2016-07..2026-07: {switches:.0f} (round trips ~{switches/2:.0f}), "
      f"days on: {(gate==1).sum()} of {len(gate)}")
