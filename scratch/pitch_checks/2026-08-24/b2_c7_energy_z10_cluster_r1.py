"""b2 — C7 round 1: the energy z10 >= 2 CLUSTER, five names thrusting at once.

COMPLEX DECLARED UP FRONT AND NOT CHANGED AFTER SEEING RESULTS. It is the set
of energy instruments carried in the pitch tape universe, all eleven:

    XLE XOP USO COP CVX VLO OXY SLB EOG HAL WMB

TRIGGER: count(z10 >= 2.0) >= 5 on a day where all eleven have a valid z10
(first such day 2006-07-24, 5052 usable sessions). z10 is the TAPE definition,
10d return / (21d sd * sqrt(10)), verified against the tape file in b0.

VEHICLE: long XLE, lag=1 MOC. Round-1 horizon h=5.

Order, per the registry:
  1. registry collision MEASURED — day overlap against the two masks the
     2026-08-17 kill actually used, not asserted.
  2. GATE-OFF FIRST — is the COUNT doing anything over XLE's own z10?
  3. battery on the trigger.
  4. today's reading inside the historical support; the count gradient.
  5. book overlap from the ledger; JH-in-hold.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    battery, close_panel, cluster_note, declusters, era_split, load_prices,
    local_control, pct_rank, rolling_on_valid, show, sign_test, summarize,
    vehicle_ret, wilder_atr, event_in_window, bootstrap_p_le0,
)

pd.set_option("display.width", 230)

COMPLEX = ["XLE", "XOP", "USO", "COP", "CVX", "VLO", "OXY", "SLB", "EOG",
           "HAL", "WMB"]
K = 5
H = 5


def tape_z10(close: pd.Series, n: int = 10) -> pd.Series:
    r = close.pct_change(n)
    v = close.pct_change().rolling(21).std()
    return r / (v * np.sqrt(n))


raw = load_prices(sorted(set(COMPLEX + ["SPY"])))
pan = close_panel(sorted(set(COMPLEX + ["SPY"]))).dropna(subset=["XLE", "SPY"])
IDX = pan.index

z = pd.DataFrame({t: tape_z10(raw[t]["Close"]) for t in COMPLEX}).reindex(IDX)
allvalid = z.notna().all(axis=1)
cnt = (z >= 2.0).sum(axis=1).where(allvalid)
TRIG = (cnt >= K).fillna(False)

print(f"panel {IDX[0].date()} .. {IDX[-1].date()}  N={len(IDX)}")
print(f"usable (all 11 valid) = {int(allvalid.sum())}, first {IDX[allvalid][0].date()}")
print(f"TODAY count={int(cnt.iloc[-1])}: "
      f"{sorted(z.columns[(z.iloc[-1] >= 2.0).values])}")
print(f"trigger days (count>={K}) = {int(TRIG.sum())}, "
      f"episodes(10td) = {len(declusters(IDX[TRIG.values], 10, IDX))}")

# ---------------------------------------------------------------------------
print("\n=== 1. REGISTRY COLLISION, MEASURED (2026-08-17 XLE thrust-to-high) ===")
xle = pan["XLE"]
ohlc = raw["XLE"].reindex(IDX)
atr = pd.Series(np.asarray(wilder_atr(ohlc["High"], ohlc["Low"], ohlc["Close"], 14),
                           dtype=float), index=IDX)
r5rk = pct_rank(xle, 5)
r63rk = pct_rank(xle, 63)
hi52 = rolling_on_valid(xle, lambda x: x.rolling(252).max())
off_hi = xle / hi52 - 1.0
mv5_atr = (xle - xle.shift(5)) / atr
NEAR_HI = off_hi >= -0.01
MID63 = (r63rk >= 30) & (r63rk <= 75)
RANK_KILLED = ((r5rk >= 98) & NEAR_HI & MID63).fillna(False)
MAG_KILLED = ((mv5_atr >= 2.5) & NEAR_HI & MID63).fillna(False)
FRESH_HIGH = (off_hi >= -1e-9).fillna(False)

for nm, m in (("08-17 RANK mask", RANK_KILLED), ("08-17 MAG mask", MAG_KILLED),
              ("XLE fresh 52w high", FRESH_HIGH), ("XLE near-high (<=1%)", NEAR_HI.fillna(False)),
              ("XLE z10>=2 alone", (z["XLE"] >= 2).fillna(False))):
    a = set(IDX[TRIG.values])
    b = set(IDX[m.values])
    inter = a & b
    print(f"  {nm:24s} N={len(b):5d}  overlap with count>={K}: {len(inter):4d} "
          f"= {100*len(inter)/max(len(a),1):5.1f}% of trigger days, "
          f"{100*len(inter)/max(len(b),1):5.1f}% of its own")
print(f"  today's XLE state: rank5={r5rk.iloc[-1]:.1f} rank63={r63rk.iloc[-1]:.1f} "
      f"off52wh={100*off_hi.iloc[-1]:+.2f}% 5d move={mv5_atr.iloc[-1]:+.2f} ATR  "
      f"-> 08-17 RANK mask fires today: {bool(RANK_KILLED.iloc[-1])}, "
      f"MAG mask: {bool(MAG_KILLED.iloc[-1])}")

# ---------------------------------------------------------------------------
print("\n=== 2. GATE-OFF FIRST: does the COUNT add over XLE's own thrust? ===")
ret = vehicle_ret(pan, [("XLE", 1.0)], H)
valid = ret.notna()
xz = (z["XLE"] >= 2).fillna(False) & allvalid.fillna(False)


def cell(mask, label):
    d = IDX[mask.values & valid.values]
    if len(d) == 0:
        return {"label": label, "n": 0}
    e = declusters(d, 10, IDX)
    r = summarize(ret.loc[e].values, label)
    r["n_days"] = len(d)
    return r


rows = [cell((cnt >= k).fillna(False), f"count >= {k}") for k in range(1, 10)]
rows += [
    cell(xz, "XLE's OWN z10>=2 (no count)"),
    cell(xz & TRIG, "XLE z10>=2 AND count>=5"),
    cell(xz & ~TRIG, "XLE z10>=2 but count<5"),
    cell(~xz & TRIG, "count>=5 but XLE NOT thrusting"),
    cell(allvalid.fillna(False), "CTRL all usable days"),
]
show(rows, f"count ladder + gate attribution, long XLE h={H} (episodes, 10td gap)")

# ---------------------------------------------------------------------------
print("\n=== 3. battery on count>=5 ===")
variants = {f"count>={k}": (cnt >= k).fillna(False) for k in (3, 4, 5, 6, 7)}
variants["XLE z10>=2 alone"] = xz
variants["count>=5, z thr 1.75"] = (((z >= 1.75).sum(axis=1).where(allvalid)) >= 5).fillna(False)
variants["count>=5, z thr 2.25"] = (((z >= 2.25).sum(axis=1).where(allvalid)) >= 5).fillna(False)
variants["count>=5, z thr 2.5"] = (((z >= 2.5).sum(axis=1).where(allvalid)) >= 5).fillna(False)
variants["frac>=45% of complex"] = ((cnt / 11.0) >= 0.45).fillna(False)
battery(pan, TRIG, [("XLE", 1.0)], H, f"C7 long XLE on count(z10>=2)>={K}",
        cost_bps=4.0, variants=variants, min_gap=10,
        event_kinds=("opex", "jackson_hole", "cpi"))

# ---------------------------------------------------------------------------
print("\n=== 4. is today's reading inside the historical support? ===")
tr = IDX[TRIG.values & valid.values]
epi = declusters(tr, 10, IDX)
sub = pd.DataFrame({
    "cnt": cnt.loc[tr], "off_hi": off_hi.loc[tr] * 100,
    "r63rk": r63rk.loc[tr], "xle_z": z["XLE"].loc[tr],
    "fwd": ret.loc[tr] * 100,
})
print(sub.describe().round(2).to_string())
print(f"  today: cnt=5 off_hi=-0.17% r63rk={r63rk.iloc[-1]:.1f} xle_z={z['XLE'].iloc[-1]:.2f}")
print(f"  trigger days with XLE within 1% of its 52w high: "
      f"{int((sub['off_hi'] >= -1).sum())} of {len(sub)} "
      f"({100*(sub['off_hi'] >= -1).mean():.1f}%)")
show([summarize(sub.loc[sub["off_hi"] >= -1, "fwd"].values / 100,
                "trigger AND XLE within 1% of high (today's state)"),
      summarize(sub.loc[sub["off_hi"] < -1, "fwd"].values / 100,
                "trigger, XLE NOT near a high")],
     "the live sub-state (day level)")
b, a = np.polyfit(sub["off_hi"].values, sub["fwd"].values, 1)
print(f"  fwd{H} = {a:+.3f} {b:+.4f} * off_hi%  -> fitted at today's -0.17%: "
      f"{a + b*(-0.17):+.3f}%  (trigger-set mean {sub['fwd'].mean():+.3f}%)")

print("\n  --- count gradient (day level) ---")
for k in range(1, 10):
    m = (cnt == k).fillna(False) & valid
    d = IDX[m.values]
    if len(d) >= 5:
        print(f"   count=={k}: N={len(d):4d}  mean {100*ret.loc[d].mean():+.3f}%  "
              f"hit {100*(ret.loc[d] > 0).mean():.1f}%")

# ---------------------------------------------------------------------------
print("\n=== 5. book overlap (ledger) + Jackson Hole in the hold ===")
try:
    led = pd.read_parquet("data/backtest_trades_full.parquet")
    led["Signal Date"] = pd.to_datetime(led["Signal Date"])
    ENERGY_TKRS = set(COMPLEX) | {"CL=F", "OIH", "PSX", "MPC", "DVN", "FANG",
                                  "KMI", "OKE", "XES", "MRO", "APA", "HES"}
    pos = pd.Series(range(len(IDX)), index=IDX)
    win = set()
    for d in tr:
        p = pos[d]
        win |= set(IDX[p:min(p + 4, len(IDX))])
    hit = led[led["Signal Date"].isin(win) & led["Ticker"].isin(ENERGY_TKRS)]
    print(f"  energy-ticker book trades signalled inside a trigger window (0..3 td): {len(hit)}")
    if len(hit):
        print(hit.groupby(["Strategy", "Direction"])
              .agg(n=("R_Multiple", "size"), avgR=("R_Multiple", "mean"),
                   pnl=("PnL_flat_750k", "sum")).round(3).to_string())
        print(f"  direction split: {hit['Direction'].value_counts().to_dict()}")
    allhit = led[led["Signal Date"].isin(win)]
    print(f"  ALL book trades in trigger windows: {len(allhit)}, "
          f"{allhit['Direction'].value_counts().to_dict()}, "
          f"avgR {allhit['R_Multiple'].mean():.3f}")
except Exception as exc:  # noqa: BLE001
    print("  ledger read failed:", exc)

jh = event_in_window(epi, IDX, H, 1, ("jackson_hole",))
show([summarize(ret.loc[epi].values[jh], f"JH inside the hold (N={int(jh.sum())})"),
      summarize(ret.loc[epi].values[~jh], f"JH outside (N={int((~jh).sum())})")],
     "Jackson Hole in the hold (today's h>=4 hold contains 2026-08-28)")
