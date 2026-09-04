"""C7 round 1 - short GDX after a 99.6th-pctile 21-day run.

Live premise: GDX 21d +37.63%, PIT trailing-252 pctile 99.6; GDX-GLD 21d
+22.90pp, PIT 99.2; GDX -10.62% off its 52w high; GLD -13.96% off its own.

Round-1 obligations discharged here:
  0. PREMISE re-derivation: PIT vs FULL-HISTORY percentile (the 2026-08-18
     lookahead trap), and the count of days that ever ran hotter.
  1. REGISTRY COLLISION, measured in BOTH directions against two corpses:
       - 2026-08-17 e2_c10: GDX 21d rank == 100 maximal thrust (+ magnitude
         gates 20/26/30%), closed by the reference class.
       - 2026-08-18 b5_c5: GDX-GLD 21d spread PIT >= 97, whose OUTRIGHT
         vehicle is literally `-r_gdx`, i.e. THIS candidate, measured at
         -1.153% at h=5.
     Overlap is P(A|B) and P(B|A) on trigger DAYS (2026-08-24 rule: 91% = the
     same object).
  2. battery() on the short, h=5, rung ladder 95/97/99/99.5 + magnitude gates.
  3. W4 RECONCILIATION: the long side of this exact state.
  4. BOOK OVERLAP: what the systematic ledger does on GDX thrust days.
  5. cost at a 5 bps single-leg round trip.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BAR = pd.Timestamp("2026-08-24")
NAMES = ["GDX", "GLD", "SLV", "GDXJ", "NEM", "SPY"]

px_all = load_prices(NAMES)
spy = px_all["SPY"]["Close"].dropna()
CAL = spy.index[spy.index <= BAR]
px = pd.DataFrame({t: px_all[t]["Close"] for t in NAMES}).reindex(CAL)
for t in NAMES:
    s = px_all[t]["Close"].dropna()
    print(f"  {t}: {s.index[0].date()} .. {s.index[-1].date()}  n={len(s)}")

gdx = px_all["GDX"]["Close"].dropna()
gld = px_all["GLD"]["Close"].dropna()

r21_g = _valid_pct_change(gdx, 21).reindex(CAL)
pit21_g = pct_rank(gdx, 21).reindex(CAL)
full21_g = rolling_on_valid(r21_g, lambda x: x.expanding(252).rank(pct=True) * 100)
rk21_g_100 = pct_rank(gdx, 21).reindex(CAL)          # 08-17 corpse basis
sp21 = r21_g - _valid_pct_change(gld, 21).reindex(CAL)
pit_sp = rolling_on_valid(sp21, lambda x: x.rolling(252).rank(pct=True) * 100)


def dist_hi(c, look=252):
    return rolling_on_valid(c, lambda x: x / x.rolling(look).max() - 1.0).reindex(CAL)


dd_g, dd_gld = dist_hi(gdx), dist_hi(gld)

print("\n" + "=" * 100)
print("0. PREMISE re-derivation")
print("=" * 100)
print(f"  GDX 21d ret {100*r21_g.iloc[-1]:+.2f}%   PIT-252 pctile {pit21_g.iloc[-1]:.2f}"
      f"   FULL-HISTORY pctile {100*(r21_g <= r21_g.iloc[-1]).mean():.2f}")
hotter = int((r21_g > r21_g.iloc[-1]).sum())
print(f"  days in GDX's whole history that ran HOTTER over 21d: {hotter} of "
      f"{int(r21_g.notna().sum())} ({100*hotter/r21_g.notna().sum():.2f}%)")
hot_dates = CAL[(r21_g > r21_g.iloc[-1]).values]
if len(hot_dates):
    print(f"     they cluster: {pd.Series(hot_dates.year).value_counts().sort_index().to_dict()}")
print(f"  GDX-GLD 21d spread {100*sp21.iloc[-1]:+.2f}pp   PIT {pit_sp.iloc[-1]:.2f}"
      f"   FULL {100*(sp21 <= sp21.iloc[-1]).mean():.2f}")
print(f"  GDX off 52wh {100*dd_g.iloc[-1]:+.2f}%   GLD off 52wh {100*dd_gld.iloc[-1]:+.2f}%")

# ------------------------------------------------------------ 1. collisions
print("\n" + "=" * 100)
print("1. REGISTRY COLLISION, measured both directions")
print("=" * 100)
A = (pit21_g >= 99.0).fillna(False)                       # C7 as pitched
B1 = (rk21_g_100 >= 100.0).fillna(False)                  # 08-17 maximal thrust
B2 = (pit_sp >= 97.0).fillna(False)                       # 08-18 ratio corpse
for lbl, B in (("08-17 GDX rk21==100", B1), ("08-18 GDX-GLD spread PIT>=97", B2)):
    inter = int((A & B).sum())
    print(f"  A=C7(PIT21>=99) n={int(A.sum())}   B={lbl} n={int(B.sum())}   "
          f"|A&B|={inter}   P(B|A)={inter/max(A.sum(),1):.3f}   "
          f"P(A|B)={inter/max(B.sum(),1):.3f}")
print(f"  today: C7 fires={bool(A.iloc[-1])}  08-17 mask fires={bool(B1.iloc[-1])}  "
      f"08-18 mask fires={bool(B2.iloc[-1])}")

# -------------------------------------------------------------- 2. battery
TRIG = {
    "PIT21 >= 95": (pit21_g >= 95).fillna(False),
    "PIT21 >= 97": (pit21_g >= 97).fillna(False),
    "PIT21 >= 99 (pitched)": A,
    "PIT21 >= 99.5": (pit21_g >= 99.5).fillna(False),
    "PIT21>=99 & r21 >= 20%": A & (r21_g >= 0.20),
    "PIT21>=99 & r21 >= 30%": A & (r21_g >= 0.30),
    "PIT21>=99 & r21 >= 37% (LIVE)": A & (r21_g >= 0.376),
    "PIT21>=99 & GDX off-52wh >= -12% (LIVE)": A & (dd_g >= -0.12),
    "PIT21>=99 & GLD off-52wh <= -10% (LIVE)": A & (dd_gld <= -0.10),
}
print("\n  trigger day counts:")
for k, m in TRIG.items():
    m = m.reindex(CAL, fill_value=False).fillna(False)
    print(f"    {k:45s} n_days={int(m.sum()):4d}  live={bool(m.iloc[-1])}")

battery(px, A, [("GDX", -1.0)], h=5,
        title="C7 SHORT GDX, PIT21 >= 99", cost_bps=5.0, variants=TRIG,
        event_kinds=("jackson_hole", "cpi", "fomc_decision"))

# ------------------------------------------------------- 3. horizon + sign
print("\n" + "=" * 100)
print("3. HORIZON SCAN on the short (does the sign ever go our way?)")
print("=" * 100)
trigA = CAL[A.values]
show(horizon_scan(px, trigA, [("GDX", -1.0)], hs=(1, 2, 3, 4, 5, 6, 8, 10)),
     "short GDX, PIT21>=99 episodes")
show(horizon_scan(px, trigA, [("GDX", -1.0), ("GLD", 1.0)], hs=(1, 2, 3, 5, 10)),
     "short GDX / long GLD (dollar), same trigger")

# ------------------------------------------------------ 4. W4 reconciliation
print("\n" + "=" * 100)
print("4. W4 reconciliation - the LONG side of the same state")
print("=" * 100)
r5_g = _valid_pct_change(gdx, 5).reindex(CAL)
for lo, hi, lbl in ((0.10, 9.9, "GDX 5d > +10% (W4's caveat cell)"),
                    (-9.9, 0.10, "GDX 5d <= +10%")):
    m = ((r5_g > lo) & (r5_g <= hi)).fillna(False)
    ser = vehicle_ret(px, [("GLD", 1.0)], 5)
    v = ser.notna()
    t = CAL[m.values & v.values]
    e = declusters(t, 5, CAL[v.values])
    print(f"  LONG GLD | {lbl}: ", summarize(ser.loc[e].values, "")["mean_pct"].__round__(3),
          f"% hit {summarize(ser.loc[e].values,'')['hit']:.1f} N={len(e)}")
rows = []
for lbl, legs in (("LONG GDX", [("GDX", 1.0)]), ("SHORT GDX", [("GDX", -1.0)]),
                  ("LONG GLD", [("GLD", 1.0)])):
    ser = vehicle_ret(px, legs, 5)
    v = ser.notna()
    e = declusters(CAL[A.values & v.values], 5, CAL[v.values])
    r = summarize(ser.loc[e].values, f"{lbl} on PIT21>=99")
    r["ctl_pct"] = round(100 * ser[v].mean(), 3)
    r["edge_pct"] = round(r["mean_pct"] - 100 * ser[v].mean(), 3)
    rows.append(r)
show(rows, "4. both sides of the live state, h=5 episodes")

# ---------------------------------------------------------- 5. book overlap
print("\n" + "=" * 100)
print("5. BOOK OVERLAP - what does the systematic ledger do on GDX thrust days?")
print("=" * 100)
lp = ROOT / "data" / "backtest_trades_full.parquet"
if lp.exists():
    led = pd.read_parquet(lp)
    dcol = "Signal_Date" if "Signal_Date" in led.columns else led.columns[0]
    led[dcol] = pd.to_datetime(led[dcol])
    trig_days = set(pd.DatetimeIndex(CAL[A.values]).normalize())
    win = set()
    pos = pd.Series(range(len(CAL)), index=CAL)
    for d in CAL[A.values]:
        p = pos[d]
        for k in range(0, 6):
            if p + k < len(CAL):
                win.add(CAL[p + k])
    sub = led[led[dcol].isin(win)]
    print(f"  ledger rows on/within 5td of a PIT21>=99 GDX thrust day: {len(sub)}")
    if len(sub):
        sc = "Strategy_Name" if "Strategy_Name" in led.columns else "Strategy"
        dirc = "Direction" if "Direction" in led.columns else None
        print(sub.groupby([sc] + ([dirc] if dirc else [])).size()
              .sort_values(ascending=False).head(12).to_string())
        miners = ["GDX", "GDXJ", "NEM", "AEM", "AU", "KGC", "AGI", "GOLD", "RGLD",
                  "PAAS", "HL", "EGO", "IAG", "BTG", "WPM", "FNV", "SSRM"]
        tc = "Ticker" if "Ticker" in led.columns else "Symbol"
        m2 = sub[sub[tc].isin(miners)]
        print(f"  of which MINER names: {len(m2)}")
        if len(m2):
            print(m2.groupby([sc] + ([dirc] if dirc else [])).size().to_string())
else:
    print("  ledger parquet absent")
print("\nDONE C7 round 1")
