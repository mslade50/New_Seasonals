"""b1 — C3 round 1: long copper on a five-day thrust to a fresh 52-week high.

Live state: FCX +15.30% over 5 sessions, close == its 252d max, 2.0x volume.

Order of operations follows the registry:
  0. is this even a COPPER state, or a single-name FCX state?
  1. GATE-OFF FIRST (2026-08-19, four cells in one morning): thrust alone,
     fresh-high alone, then the intersection. If the gate subtracts, done.
  2. battery() on the intersection.
  3. denominator-roll / magnitude-vs-rank contrast.
  4. behaviour AT today's extreme (2026-08-18: "the loud state is usually the
     poisoning conditioner").
  5. book overlap, measured from the ledger rather than asserted.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    battery, close_panel, cluster_note, declusters, era_split, load_prices,
    local_control, rolling_on_valid, show, sign_test, summarize, vehicle_ret,
)
from pitch_lab import _valid_pct_change as vpc  # noqa: E402

pd.set_option("display.width", 220)

EQ = ["FCX", "COPX", "XME", "XLB", "SCCO", "TECK", "SPY"]
px = close_panel(EQ)
raw = load_prices(["HG=F"])

THRUST = 0.15          # today's reading is +15.30%
H = 5

r5 = {t: vpc(px[t], 5) for t in EQ}
hi252 = {t: rolling_on_valid(px[t], lambda x: x.rolling(252).max()) for t in EQ}
at_high = {t: (px[t] >= hi252[t] * (1 - 1e-9)) for t in EQ}

# ---------------------------------------------------------------------------
print("=== 0. is the live state COPPER, or is it FCX? ===")
hg = raw["HG=F"]["Close"]
print(f"  HG=F 5d {hg.pct_change(5).iloc[-1]:+.2%}  21d {hg.pct_change(21).iloc[-1]:+.2%}  "
      f"dist 52w high {hg.iloc[-1] / hg.rolling(252).max().iloc[-1] - 1:+.2%}")
for t in ["FCX", "COPX", "XME", "SCCO", "TECK", "XLB"]:
    print(f"  {t:6s} 5d {r5[t].iloc[-1]:+.2%}   dist 52w high "
          f"{px[t].iloc[-1] / hi252[t].iloc[-1] - 1:+.2%}   at_high={bool(at_high[t].iloc[-1])}")
corr = pd.concat([px["FCX"].pct_change(), hg.pct_change()], axis=1).dropna().corr().iloc[0, 1]
print(f"  FCX vs HG=F daily return corr (full history): {corr:.3f}")

# ---------------------------------------------------------------------------
print("\n=== 1. GATE-OFF FIRST: the two halves and the intersection (FCX leg) ===")
valid = vehicle_ret(px, [("FCX", 1.0)], H).notna()
ret = vehicle_ret(px, [("FCX", 1.0)], H)
masks = {
    "A thrust only  r5>=15%": (r5["FCX"] >= THRUST),
    "B fresh 52w high only ": at_high["FCX"],
    "C BOTH (the cell)     ": (r5["FCX"] >= THRUST) & at_high["FCX"],
}
rows = []
for lbl, m in masks.items():
    d = px.index[m.fillna(False).values & valid.values]
    epi = declusters(d, H, px.index)
    r = summarize(ret.loc[d].values, f"{lbl} day-level")
    r["n_epi"] = len(epi)
    rows.append(r)
    rows.append(summarize(ret.loc[epi].values, f"{lbl} episodes"))
rows.append(summarize(ret[valid].values, "CTRL-b FCX all days"))
show(rows, f"gate attribution, h={H}")

mA = masks["A thrust only  r5>=15%"].fillna(False)
mB = masks["B fresh 52w high only "].fillna(False)
mC = masks["C BOTH (the cell)     "].fillna(False)
print(f"  overlap: thrust days {int(mA.sum())}, high days {int(mB.sum())}, "
      f"both {int(mC.sum())}  -> the gate keeps "
      f"{100*mC.sum()/max(mA.sum(),1):.1f}% of thrust days")
dA = px.index[mA.values & valid.values]
dC = px.index[mC.values & valid.values]
dAnotC = dA.difference(dC)
show([summarize(ret.loc[dC].values, "thrust AND high"),
      summarize(ret.loc[dAnotC].values, "thrust WITHOUT high"),
      summarize(ret.loc[px.index[mB.values & valid.values].difference(dC)].values,
                "high WITHOUT thrust")],
     "gate marginal (day level)")

# ---------------------------------------------------------------------------
print("\n=== 2. battery on the cell ===")
variants = {
    "r5>=10% & high": (r5["FCX"] >= 0.10) & at_high["FCX"],
    "r5>=12% & high": (r5["FCX"] >= 0.12) & at_high["FCX"],
    "r5>=15% & high": mC,
    "r5>=18% & high": (r5["FCX"] >= 0.18) & at_high["FCX"],
    "r5>=20% & high": (r5["FCX"] >= 0.20) & at_high["FCX"],
    "r5>=15% & within 1% of high": (r5["FCX"] >= 0.15) & (px["FCX"] >= hi252["FCX"] * 0.99),
    "r5>=15% & within 2% of high": (r5["FCX"] >= 0.15) & (px["FCX"] >= hi252["FCX"] * 0.98),
    "r5>=15%, no high gate": mA,
}
battery(px, mC, [("FCX", 1.0)], H, "C3 FCX: r5>=15% into a fresh 52w high",
        cost_bps=10.0, variants=variants)

# ---------------------------------------------------------------------------
print("\n=== 3. denominator roll / magnitude vs rank ===")
d1 = vpc(px["FCX"], 1)
roll = d1.shift(4)          # the bar that leaves the 5d window next session
own_dom = (d1.abs() > roll.abs())
sub = px.index[mC.values & valid.values]
print(f"  on trigger days the OWN day move exceeds the rolling-off bar on "
      f"{100*own_dom.loc[sub].mean():.1f}% of them (registry warns the high tail "
      f"is where the roll dominates)")
rk5 = rolling_on_valid(vpc(px["FCX"], 5), lambda x: x.rolling(252).rank(pct=True) * 100)
mRank = (rk5 >= 99.0) & at_high["FCX"]
dR = px.index[mRank.fillna(False).values & valid.values]
show([summarize(ret.loc[declusters(dC, H, px.index)].values, "MAGNITUDE form (r5>=15%)"),
      summarize(ret.loc[declusters(dR, H, px.index)].values, "RANK form (rk5>=99)")],
     "magnitude vs rank form, episodes")

# ---------------------------------------------------------------------------
print("\n=== 4. behaviour AT today's extreme (gradient inside the trigger set) ===")
tr = px.index[mC.values & valid.values]
x = r5["FCX"].loc[tr].values * 100
y = ret.loc[tr].values * 100
if len(tr) > 3:
    b, a = np.polyfit(x, y, 1)
    print(f"  fwd{H} = {a:+.3f} {b:+.4f} * r5%   ->  fitted at today's 15.30%: "
          f"{a + b*15.30:+.3f}%   (trigger-set mean {y.mean():+.3f}%)")
    print(f"  trigger r5 quartiles: {np.percentile(x, [25, 50, 75]).round(2)}  "
          f"today 15.30 sits at pctile {100*(x < 15.30).mean():.0f}")
    lo, hi = x <= np.median(x), x > np.median(x)
    show([summarize(y[lo] / 100, "r5 BELOW trigger median"),
          summarize(y[hi] / 100, "r5 ABOVE trigger median (today's half)")],
         "split at the trigger-set median thrust")
# distance above the 200d, today +29.95%
sma200 = rolling_on_valid(px["FCX"], lambda x: x.rolling(200).mean())
ext = (px["FCX"] / sma200 - 1).loc[tr].values * 100
if len(tr) > 3:
    b2, a2 = np.polyfit(ext, y, 1)
    print(f"  fwd{H} = {a2:+.3f} {b2:+.4f} * ext200%  -> fitted at today's "
          f"+29.95%: {a2 + b2*29.95:+.3f}%   (trigger ext median {np.median(ext):.1f}%)")

# ---------------------------------------------------------------------------
print("\n=== 5. book overlap (measured from the ledger) ===")
try:
    led = pd.read_parquet("data/backtest_trades_full.parquet")
    dcol = "Signal_Date" if "Signal_Date" in led.columns else led.columns[0]
    led[dcol] = pd.to_datetime(led[dcol])
    f = led[led["Ticker"] == "FCX"]
    print(f"  FCX trades in the 23y ledger: {len(f)}")
    if len(f):
        print(f["Strategy_Name"].value_counts().to_string())
        trig = set(pd.DatetimeIndex(tr).normalize())
        # a book trade whose signal lands on or within 3 td after a trigger
        pos = pd.Series(range(len(px.index)), index=px.index)
        near = set()
        for d in tr:
            p = pos[d]
            near |= set(px.index[p:min(p + 4, len(px.index))])
        hit = f[f[dcol].isin(near)]
        print(f"  FCX ledger trades inside a trigger window (0..3 td): {len(hit)}")
        if len(hit):
            print(hit[[dcol, "Strategy_Name", "Direction", "R_Multiple"]]
                  .to_string(index=False))
except Exception as exc:  # noqa: BLE001
    print("  ledger read failed:", exc)
