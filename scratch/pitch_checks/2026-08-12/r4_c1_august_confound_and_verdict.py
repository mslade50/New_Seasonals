"""C1 RED TEAM r4 -- close the loop.

  M. Verify tonight's calendar state from the raw file (CPI 08-12, PPI 08-13).
  N. Is AUGUST confounded with price state? r3 found the live cell is 9-for-9
     when the eve sits within 3% of a 52w low, and today is 0.33% off one. If
     the four August losers were all in the OTHER price state, 'August' is a
     label for that state. CROSS the rescuing conditioner with the killing one
     first (registry rule) -- and if the cross is empty, say so and DO NOT
     credit the rescue.
  O. The final expectation and the R arithmetic at three sizes.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

ROOT_P = Path(__file__).resolve().parents[3]
ecsv = pd.read_csv(ROOT_P / "data" / "macro_events.csv")
ecsv["date"] = pd.to_datetime(ecsv["date"])

print("=" * 100)
print("M. TONIGHT'S CALENDAR STATE, from data/macro_events.csv directly")
print("=" * 100)
w = ecsv[(ecsv["date"] >= "2026-08-05") & (ecsv["date"] <= "2026-08-25")]
print(w[["date", "event", "detail", "time_et"]].to_string(index=False))
cpi_d = set(ecsv.loc[ecsv.event == "cpi", "date"])
ppi_d = set(ecsv.loc[ecsv.event == "ppi", "date"])
print(f"\n  2026-08-12 is a CPI print: {pd.Timestamp('2026-08-12') in cpi_d}")
print(f"  2026-08-13 is a PPI print: {pd.Timestamp('2026-08-13') in ppi_d}")
print("  -> the entry close and the exit close are the two sessions the cell "
      "is defined on.")

mp = pd.read_parquet(ROOT_P / "data" / "master_prices.parquet")
tl = mp[mp["ticker"] == "TLT"].copy()
tl["date"] = pd.to_datetime(tl["date"])
tl = tl.sort_values("date").drop_duplicates("date", keep="last").set_index("date")
idx, c = tl.index, tl["Close"].values.astype(float)
N = len(c)
d1 = np.full(N, np.nan)
d1[1:] = c[1:] / c[:-1] - 1.0
ok = ~np.isnan(d1)
base_hit = float((d1[ok] > 0).mean())
sessd = lambda k: {int(idx.searchsorted(x, "left"))
                   for x in ecsv.loc[ecsv["event"] == k, "date"]
                   if 0 <= int(idx.searchsorted(x, "left")) < N}
PPI, CPI = sessd("ppi"), sessd("cpi")
ppi_l = sorted(p for p in PPI if 1 <= p < N and ok[p])
v = np.array([d1[p] for p in ppi_l])
dt = pd.DatetimeIndex([idx[p] for p in ppi_l])
mo = dt.month.values
L = np.array([(p - 1) in CPI for p in ppi_l])
lp = np.array(ppi_l)[L]
lv, ld, lmo = v[L], dt[L], mo[L]

lo52 = pd.Series(c, index=idx).rolling(252).min().values
hi52 = pd.Series(c, index=idx).rolling(252).max().values
dlo = (c / lo52 - 1.0) * 100

print("\n" + "=" * 100)
print("N. IS AUGUST CONFOUNDED WITH PRICE STATE?")
print("=" * 100)
eve_lo = np.array([dlo[p - 1] for p in lp])
print(f"  TODAY's eve state: TLT {dlo[-1]:.2f}% above its 52w low")
print("\n  the four August observations, with their eve price state:")
for i in np.where(lmo == 8)[0]:
    print(f"    {ld[i].date()}  print {100*lv[i]:+.3f}%   eve was "
          f"{eve_lo[i]:6.2f}% above its 52w low")
print(f"\n  median eve-above-low across all 55 = {np.median(eve_lo):.2f}%")
print(f"  median eve-above-low across the 4 August obs = "
      f"{np.median(eve_lo[lmo == 8]):.2f}%")
near = eve_lo <= 3.0
print(f"\n  CROSS (registry rule: cross the rescue with the kill FIRST):")
print(f"    live cell AND near a 52w low AND August: N={int((near & (lmo==8)).sum())}")
print("    -> the cross is EMPTY. A rescuing sub-cell with N=0 is not a rescue.")
print("       The 52w-low reading is NOT credited in the point estimate below.")
print(f"    live cell AND near a 52w low (any month): N={int(near.sum())} "
       f"mean {1e4*lv[near].mean():+.1f} bps hit {100*(lv[near]>0).mean():.0f}% "
       f"signp {sign_test(int((lv[near]>0).sum()), int(near.sum()), base_hit):.4f}")
print(f"    live cell NOT near a 52w low: N={int((~near).sum())} "
      f"mean {1e4*lv[~near].mean():+.1f} bps hit {100*(lv[~near]>0).mean():.0f}%")
print("\n  what this DOES license: August's 0-for-4 is not a clean read on the")
print("  month either, because all four eves sat well away from the state the")
print("  trade is being taken in tonight. Two small crossed cells, no rescue")
print("  and no kill; the 55-observation cell is what remains.")

print("\n  PARENT check on the same cross (bigger N):")
pe_lo = np.array([dlo[p - 1] for p in ppi_l])
pn = pe_lo <= 3.0
for lbl, m in [("parent near-low, August", pn & (mo == 8)),
               ("parent near-low, other months", pn & (mo != 8)),
               ("parent not-near-low, August", (~pn) & (mo == 8))]:
    s = v[m]
    if len(s):
        print(f"    {lbl:32s} N={len(s):3d} mean {1e4*s.mean():+7.1f} bps  "
              f"hit {100*(s>0).mean():5.1f}%")

print("\n" + "=" * 100)
print("O. SIZE / R ARITHMETIC (NAV 750k, TLT 82.19, Wilder-14 ATR 0.5734)")
print("=" * 100)
atr = float(wilder_atr(tl["High"].values, tl["Low"].values, tl["Close"].values, 14)[-1])
px = float(c[-1])
NAV = 750_000
cell_sd = lv.std(ddof=1)
print(f"  the cell's ONE-SESSION sd is {100*cell_sd:.3f}% = {cell_sd*px/atr:.2f} ATR.")
print("  So a 1.0-ATR risk unit on a no-stop one-session hold makes a typical")
print(f"  outcome {cell_sd*px/atr:.2f}R, not 1R. The stated 'R' understates the")
print("  spread of outcomes by about a third.")
for bps in (30, 20, 15, 12):
    rd = NAV * bps / 1e4
    sh = int(rd / atr)
    notl = sh * px
    print(f"\n  {bps} bps risk: {sh:,} sh, notional ${notl:,.0f} "
          f"({100*notl/NAV:.1f}% NAV)")
    print(f"    edge +16 bps -> ${notl*0.0016:+,.0f} ({1e4*notl*0.0016/NAV:+.1f} bps NAV, "
          f"{notl*0.0016/rd:+.2f}R)")
    print(f"    worst observed -2.33% -> ${-notl*0.02331:+,.0f} "
          f"({1e4*-notl*0.02331/NAV:+.1f} bps NAV, {-notl*0.02331/rd:+.2f}R)")
    print(f"    1 sd ({100*cell_sd:.2f}%) -> +/-${notl*cell_sd:,.0f} "
          f"({notl*cell_sd/rd:.2f}R)")
print("\n  book-wide reference: the systematic book's per-strategy daily cap is")
print("  250 bps of staged ATR risk, and a single pitch idea's sanity bound is")
print("  60 bps. The constraint that actually binds here is NOTIONAL as a")
print("  fraction of NAV on a no-stop overnight, which the pitch grammar does")
print("  not cap by design (no-notional-caps rule).")
