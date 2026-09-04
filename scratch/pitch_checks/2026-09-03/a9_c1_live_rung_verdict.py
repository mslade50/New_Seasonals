"""C1 -- the decisive live-state question, isolated.

a8 section 0 found that the compression gate's dose response is NOT monotone:
on SVXY-covered (matched) anchors the tightest rung is dead on BOTH legs
  (0, 5]  SVXY -0.096% / 52.0% hit    short ^VIX -0.046% / 60.0%   n=25
  (5,10]  SVXY +1.465% / 82.4%        short ^VIX +4.457% / 70.6%   n=17
 (10,15]  SVXY +2.034% / 78.6%        short ^VIX +3.846% / 71.4%   n=14
and TODAY reads 3.57, inside the dead rung.

This script does three things and then stops:
 1. states today's exact cell on MATCHED days for both legs, explicitly,
    instead of by hand off a printed table;
 2. tests the MECHANISM that would explain the inversion -- at extreme
    compression the VIX is already at a low LEVEL with the front future in
    heavy contango, so there is little premium left to crush and SVXY has to
    overcome roll. If that is right, the dead rung should be explained by VIX
    LEVEL / contango rather than by the range percentile per se, and the live
    values (VIX level pctile 12.3, contango +16.6%) sit on the wrong side;
 3. prices the arm: what rel-range reading, or what accompanying state, turns
    the cell back on. That number goes on the watchlist verbatim.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa: F401,F403,E402
from pitch_lab import (close_panel, fwd_lag, summarize, sign_test, load_events,
                       rolling_on_valid, show, anchor_positions, bootstrap_p_le0)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)

px = close_panel(["SVXY", "UVXY", "^VIX", "^VIX3M", "SPY"])
cal = px["SPY"].dropna().index
vix = px["^VIX"]
rng21 = (rolling_on_valid(vix, lambda x: x.rolling(21).max())
         - rolling_on_valid(vix, lambda x: x.rolling(21).min()))
REL = rolling_on_valid(rng21 / rolling_on_valid(vix, lambda x: x.rolling(21).mean()),
                       lambda x: x.rolling(252).rank(pct=True) * 100)
VLP = rolling_on_valid(vix, lambda x: x.rolling(252).rank(pct=True) * 100)
TS = px["^VIX3M"] / px["^VIX"] - 1.0
sma20 = rolling_on_valid(vix, lambda x: x.rolling(20, min_periods=16).mean())

KINDS = ("nfp", "cpi", "ppi", "fomc_decision")
EV = {k: load_events([k])["date"] for k in KINDS}
ALL_PRINTS = pd.DatetimeIndex(sorted(pd.concat(list(EV.values())).unique()))
pos = pd.Series(range(len(cal)), index=cal)
rows = []
for kind in KINDS:
    p, kept = anchor_positions(cal, EV[kind], -2)
    for i, ap in enumerate(p):
        d0 = kept[i]
        nxt = ALL_PRINTS[ALL_PRINTS > d0]
        rw = 99 if len(nxt) == 0 else int(
            pos.get(nxt[0], int(cal.searchsorted(nxt[0])))
            - pos.get(d0, int(cal.searchsorted(d0))))
        rows.append({"anchor": cal[ap], "kind": kind, "runway_td": rw})
F = pd.DataFrame(rows).set_index("anchor").sort_index()
g = F.groupby(level=0)
F = F[~F.index.duplicated(keep="first")].assign(
    runway_td=g["runway_td"].min(),
    kind=g["kind"].apply(lambda x: "+".join(sorted(set(x)))))
for c, s in (("rel", REL), ("vlp", VLP), ("ts", TS), ("vix", vix), ("sma", sma20)):
    F[c] = s.reindex(F.index).values
F["svxy"] = fwd_lag(px["SVXY"].dropna(), 1, lag=1).reindex(F.index).values
F["nvix"] = (-fwd_lag(px["^VIX"].dropna(), 1, lag=1)).reindex(F.index).values
CL = F[F["runway_td"] >= 3].copy()
MATCH = CL[CL["svxy"].notna()]            # SVXY-covered, i.e. 2011-10-04+

LIVE = dict(rel=float(REL.iloc[-1]), vlp=float(VLP.iloc[-1]),
            ts=float(TS.iloc[-1]), vix=float(vix.iloc[-1]),
            sma=float(sma20.iloc[-1]))
print("=" * 118)
print(f"LIVE 2026-09-02: rel-range pctile {LIVE['rel']:.2f}  VIX {LIVE['vix']:.2f} "
      f"(level pctile {LIVE['vlp']:.1f})  contango {100*LIVE['ts']:+.2f}%  "
      f"20d SMA {LIVE['sma']:.2f}")
print(f"clear-calendar anchors: all {len(CL)}, SVXY-covered {len(MATCH)}")
print("=" * 118)


def cc(v, label):
    v = pd.Series(v).dropna()
    st = summarize(v.values, label)
    if st["n"]:
        st["signp"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
        st["rec"] = f"{int((v>0).sum())}-{int((v<0).sum())}"
    return st


# ---------------------------------------------------------------------------
print("\n1. TODAY'S EXACT CELL, BOTH LEGS, ON MATCHED DAYS ONLY")
live_mask = ((MATCH["rel"] <= 5) & (MATCH["vix"] > 13) & (MATCH["vix"] > MATCH["sma"]))
S = MATCH[live_mask]
print(f"   rel<=5 AND clear calendar AND VIX>13 AND VIX>20d SMA, SVXY-covered:"
      f" n={len(S)}")
print(S.assign(svxy=(100*S['svxy']).round(2), nvix=(100*S['nvix']).round(2),
               ts=(100*S['ts']).round(1), rel=S['rel'].round(2),
               vlp=S['vlp'].round(1))[
    ["kind", "rel", "vix", "vlp", "ts", "svxy", "nvix"]].to_string())
show([cc(S["svxy"].values, "long SVXY, today's cell (matched)"),
      cc(S["nvix"].values, "short ^VIX, SAME days"),
      cc(MATCH.loc[MATCH["rel"] > 5, "svxy"].values, "long SVXY, rel>5 (matched)"),
      cc(MATCH.loc[MATCH["rel"] > 5, "nvix"].values, "short ^VIX, rel>5 (matched)")],
     "today's rung vs everything above it")
# unmatched ^VIX version, for the record
allS = CL[(CL["rel"] <= 5) & (CL["vix"] > 13) & (CL["vix"] > CL["sma"])]
print(f"   for the record, the FULL-history ^VIX version of today's cell "
      f"(n={len(allS)}, back to 2001) reads "
      f"{100*allS['nvix'].mean():+.3f}% at "
      f"{100*(allS['nvix']>0).mean():.1f}% hit, sign p "
      f"{sign_test(int((allS['nvix']>0).sum()), len(allS)):.4f} -- it is carried")
print("   by the pre-SVXY years, which is exactly the coverage asymmetry a8 flagged.")

# ---------------------------------------------------------------------------
print("\n" + "=" * 118)
print("2. THE MECHANISM TEST -- is the dead rung about the RANGE, or about")
print("   there being no premium left to crush (low VIX level / heavy contango)?")
print("=" * 118)
print("\n2a. split the matched clear-calendar cell on VIX LEVEL percentile")
rows = []
for lo, hi in ((0, 15), (15, 30), (30, 50), (50, 101)):
    m = (MATCH["vlp"] >= lo) & (MATCH["vlp"] < hi)
    rows.append(cc(MATCH.loc[m, "svxy"].values, f"SVXY | VIX lvl pctile [{lo},{hi})"))
    rows.append(cc(MATCH.loc[m, "nvix"].values, f"-^VIX | same"))
show(rows, f"live VIX level pctile is {LIVE['vlp']:.1f} -> the FIRST bucket")

print("\n2b. split on CONTANGO (VIX3M/VIX - 1). live = "
      f"{100*LIVE['ts']:+.2f}%")
rows = []
for lo, hi in ((-1, 0.05), (0.05, 0.12), (0.12, 0.18), (0.18, 9)):
    m = (MATCH["ts"] >= lo) & (MATCH["ts"] < hi)
    rows.append(cc(MATCH.loc[m, "svxy"].values,
                   f"SVXY | contango [{100*lo:.0f},{100*hi:.0f})%"))
    rows.append(cc(MATCH.loc[m, "nvix"].values, "-^VIX | same"))
show(rows, "contango buckets")

print("\n2c. 2x2: rel-range rung x VIX level -- which one carries the deadness?")
tbl = []
for rlo, rhi in ((0, 5), (5, 15), (15, 101)):
    for vlo, vhi in ((0, 25), (25, 101)):
        m = ((MATCH["rel"] > rlo if rlo else MATCH["rel"] >= 0)
             & (MATCH["rel"] <= rhi) & (MATCH["vlp"] >= vlo) & (MATCH["vlp"] < vhi))
        s = cc(MATCH.loc[m, "svxy"].values, "")
        v = cc(MATCH.loc[m, "nvix"].values, "")
        tbl.append({"rel": f"({rlo},{rhi}]", "vix_lvl_pct": f"[{vlo},{vhi})",
                    "n": s.get("n", 0),
                    "SVXY_mean": round(s.get("mean_pct", np.nan), 3),
                    "SVXY_hit": round(s.get("hit", np.nan), 1),
                    "VIX_mean": round(v.get("mean_pct", np.nan), 3),
                    "VIX_hit": round(v.get("hit", np.nan), 1)})
print(pd.DataFrame(tbl).to_string(index=False))
print(f"   LIVE lands in rel (0,5] x VIX lvl [0,25).")

print("\n2d. the SVXY-vs-spot WEDGE by contango (does roll explain the gap?)")
w = MATCH.dropna(subset=["svxy", "nvix"]).copy()
w["wedge"] = w["svxy"] - 0.5 * w["nvix"]      # SVXY is -0.5x; spot-equivalent gap
rows = []
for lo, hi in ((-1, 0.10), (0.10, 0.16), (0.16, 9)):
    m = (w["ts"] >= lo) & (w["ts"] < hi)
    rows.append(summarize(w.loc[m, "wedge"].values,
                          f"wedge | contango [{100*lo:.0f},{100*hi:.0f})%"))
rows.append(summarize(w["wedge"].values, "wedge | all"))
show(rows, "SVXY return minus 0.5 x (short-spot) return, by contango")
print(f"   corr(contango, wedge) = {w['ts'].corr(w['wedge']):+.4f} on n={len(w)}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 118)
print("3. THE ARM -- what turns this cell back on")
print("=" * 118)
rows = []
for lo, hi in ((5, 15), (5, 20), (5, 25), (10, 20)):
    m = (MATCH["rel"] > lo) & (MATCH["rel"] <= hi)
    rows.append(cc(MATCH.loc[m, "svxy"].values, f"SVXY | rel in ({lo},{hi}]"))
    rows.append(cc(MATCH.loc[m, "nvix"].values, f"-^VIX | rel in ({lo},{hi}]"))
show(rows, "candidate arms (matched, clear-calendar, k=-2, h=1)")
best = MATCH[(MATCH["rel"] > 5) & (MATCH["rel"] <= 15)]
print(f"   the (5,15] arm: SVXY n={int(best['svxy'].notna().sum())} "
      f"mean {100*best['svxy'].mean():+.3f}% "
      f"record {int((best['svxy']>0).sum())}-{int((best['svxy']<0).sum())} "
      f"sign p {sign_test(int((best['svxy']>0).sum()), int(best['svxy'].notna().sum())):.4f} "
      f"bootP(mean<=0) {bootstrap_p_le0(best['svxy'].dropna().values):.4f}")
print(f"   today's rel is {LIVE['rel']:.2f}; the arm needs rel > 5.0 on a k=-2")
print(f"   anchor whose print has >= 3 sessions of clear calendar behind it.")
print("\n   NOTE the direction of travel: a 21-day range percentile RISES as the")
print("   range widens, so this arm fires on the FIRST stirring out of the dead")
print("   zone, not on further compression. The next scheduled prints are")
print("   2026-09-10 (PPI, runway 1 -> DISQUALIFIED) and 2026-09-11 (CPI,")
# forward runway needs a FORWARD session calendar -- cal ends 2026-09-02, so
# searchsorted pins every future date to the same slot (all runways read 0).
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
fut = pd.DatetimeIndex(pd.date_range(cal[-1], periods=90, freq=bd))
ev_kind = load_events(list(KINDS)).set_index("date")["event"]
for d in ALL_PRINTS[ALL_PRINTS > pd.Timestamp('2026-09-02')][:5]:
    o = ALL_PRINTS[ALL_PRINTS > d]
    if len(o) == 0:
        rw = 99
    else:
        rw = int(fut.searchsorted(o[0]) - fut.searchsorted(d))
    k = ev_kind.get(d, "?")
    k = k if isinstance(k, str) else "+".join(sorted(set(k)))
    print(f"     {d.date()} {k:14s} -> runway {rw:2d} td  "
          f"{'CLEAR, cell can arm' if rw >= 3 else 'crowded, disqualified'}")
