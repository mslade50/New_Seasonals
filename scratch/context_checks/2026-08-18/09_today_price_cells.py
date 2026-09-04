"""The today lane, after three tape-derived crosses died.

Drills 05, 06 and 08 all killed their own cells, so the today lane needs a
candidate with cleaner provenance: something the ENGINE fired, era-stable,
where the drill adds conditioning rather than inventing the cell.

Two left standing from the price lane verdicts:
  ^BVSP  5+ consecutive down closes   n=148, +0.386%, 85-63, t 1.84, era-stable
  PL=F   a 2-ATR down session          n=221, +0.143%, 126-94, t 1.06

Plus one tape state no trigger caught: silver fell 4.08% today, 1.6 ATR, after
a 21-day gain of 11.65%.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, load_prices, summarize, show, sign_test, era_split,
    cluster_note, declusters, local_control, wilder_atr,
)

# ---------------------------------------------------------------- BVSP
px = close_panel(["^BVSP", "SPY", "EEM"])
idx = px.index
r = px.pct_change(fill_method=None)
b = px["^BVSP"]
down = (r["^BVSP"] < 0).astype(int)
streak = down * 0
run = 0
vals = []
for x in down.values:
    run = run + 1 if x == 1 else 0
    vals.append(run)
streak = pd.Series(vals, index=idx)
print(f"^BVSP current down streak: {int(streak.iloc[-1])} sessions, "
      f"last close {b.iloc[-1]:,.0f}, 1d {100*r['^BVSP'].iloc[-1]:+.2f}%")

print("\n" + "=" * 74)
print("A. ^BVSP after N consecutive down closes")
print("=" * 74)
for n in (5, 6):
    m = streak >= n
    trig = idx[m.values]
    trig = trig[trig < idx[-1]]
    epi = declusters(pd.DatetimeIndex(trig), 3, idx)
    out = []
    for h in (1, 3, 5, 10):
        f = b.shift(-h) / b - 1.0
        v = f.loc[f.index.intersection(epi)].dropna().values
        row = summarize(v, f"h={h}")
        row["rec"] = f"{int((v > 0).sum())}-{int((v <= 0).sum())}"
        row["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
        base = f.dropna()
        row["ctl_all_pct"] = round(100 * base.mean(), 3)
        row["edge_pp"] = round(row["mean_pct"] - 100 * base.mean(), 3)
        out.append(row)
    show(out, f"^BVSP, streak >= {n} (episodes {len(epi)})")

m5 = streak >= 5
trig5 = idx[m5.values]
epi5 = declusters(pd.DatetimeIndex(trig5[trig5 < idx[-1]]), 3, idx)
f1 = b.shift(-1) / b - 1.0
s = f1.loc[f1.index.intersection(epi5)].dropna()
show(era_split(s.index, s.values), "^BVSP h=1, era split")
print(" ", cluster_note(s.index, s.values, k=2))
valid = f1.dropna().index
loc = local_control(valid, pd.DatetimeIndex(epi5).intersection(valid), win=126)
print(f"  CTRL local +/-126td ex-trigger: {100*f1.loc[loc].mean():+.3f}% (n={len(loc)})")
order = np.argsort(-np.abs(s.values))[:2]
keep = np.ones(len(s), bool)
keep[order] = False
print(f"  ex the 2 largest episodes: {100*s.values[keep].mean():+.3f}% "
      f"({int((s.values[keep]>0).sum())}-{int((s.values[keep]<=0).sum())})")
by_yr = pd.Series(s.values).groupby(s.index.year.values).mean()
print(f"  positive in {int((by_yr>0).sum())} of {len(by_yr)} years with an episode")

print("\n" + "=" * 74)
print("B. is it a Brazil fact or an EM fact? same streak, EEM's forward return")
print("=" * 74)
out = []
for t in ["^BVSP", "EEM", "SPY"]:
    f = px[t].shift(-1) / px[t] - 1.0
    v = f.loc[f.index.intersection(epi5)].dropna().values
    row = summarize(v, f"{t} h=1 after a ^BVSP 5-day slide")
    row["rec"] = f"{int((v > 0).sum())}-{int((v <= 0).sum())}"
    base = f.dropna()
    row["edge_pp"] = round(row["mean_pct"] - 100 * base.mean(), 3)
    out.append(row)
show(out, "does the bounce travel?")

# ---------------------------------------------------------------- PL=F
print("\n" + "=" * 74)
print("C. PL=F after a 2-ATR down session")
print("=" * 74)
raw = load_prices(["PL=F"])["PL=F"].dropna(subset=["Close"])
atr = pd.Series(np.asarray(wilder_atr(raw["High"], raw["Low"], raw["Close"], 14)),
                index=raw.index)
c = raw["Close"]
chg = c.diff()
m = chg <= -2.0 * atr.shift(1)
trig = raw.index[m.fillna(False).values]
epi = declusters(pd.DatetimeIndex(trig[trig < raw.index[-1]]), 5, raw.index)
print(f"episodes: {len(epi)}")
out = []
for h in (1, 3, 5, 10):
    f = c.shift(-h) / c - 1.0
    v = f.loc[f.index.intersection(epi)].dropna().values
    row = summarize(v, f"h={h}")
    row["rec"] = f"{int((v > 0).sum())}-{int((v <= 0).sum())}"
    row["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
    base = f.dropna()
    row["edge_pp"] = round(row["mean_pct"] - 100 * base.mean(), 3)
    out.append(row)
show(out, "platinum after a 2-ATR down day")
f1 = c.shift(-1) / c - 1.0
s = f1.loc[f1.index.intersection(epi)].dropna()
show(era_split(s.index, s.values), "PL=F h=1, era split")
print(" ", cluster_note(s.index, s.values, k=2))

# ---------------------------------------------------------------- silver
print("\n" + "=" * 74)
print("D. SI=F: a sharp drop inside a strong run (no trigger caught this)")
print("=" * 74)
si = close_panel(["SI=F"])["SI=F"].dropna()
sr = si.pct_change(fill_method=None)
r21 = si.pct_change(21)
sraw = load_prices(["SI=F"])["SI=F"].dropna(subset=["Close"])
satr = pd.Series(np.asarray(wilder_atr(sraw["High"], sraw["Low"], sraw["Close"], 14)),
                 index=sraw.index)
atrpct = (satr / sraw["Close"]).reindex(si.index).ffill()
print(f"today: SI=F {100*sr.iloc[-1]:+.2f}%, 21d {100*r21.iloc[-1]:+.2f}%, "
      f"drop in ATR units {sr.iloc[-1]/atrpct.iloc[-1]:.2f}")
mask = (sr <= -0.03) & (r21 >= 0.08)
trig = si.index[mask.fillna(False).values]
epi = declusters(pd.DatetimeIndex(trig[trig < si.index[-1]]), 5, si.index)
print(f"episodes (drop >=3% while 21d >= +8%): {len(epi)}")
out = []
for h in (1, 3, 5, 10, 21):
    f = si.shift(-h) / si - 1.0
    v = f.loc[f.index.intersection(epi)].dropna().values
    row = summarize(v, f"h={h}")
    row["rec"] = f"{int((v > 0).sum())}-{int((v <= 0).sum())}"
    row["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
    base = f.dropna()
    row["ctl_all_pct"] = round(100 * base.mean(), 3)
    row["edge_pp"] = round(row["mean_pct"] - 100 * base.mean(), 3)
    out.append(row)
show(out, "silver: sharp drop inside an uptrend")
f5 = si.shift(-5) / si - 1.0
s = f5.loc[f5.index.intersection(epi)].dropna()
show(era_split(s.index, s.values), "SI=F h=5, era split")
print(" ", cluster_note(s.index, s.values, k=2))
