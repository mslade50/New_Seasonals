"""r4 - RED TEAM attack 4: live-state honesty + the composer's sizing inputs.

 a) cluster depth 3: what did depth-matched entries do, and is depth a real
    conditioner or a 4-observation carve?
 b) MAGNITUDE not rank (registry trap): what ret21 does "rank 100" mean on
    each historical episode, and where does today's +13.94% sit?
 c) COMPOSITION: is the ETF trigger just "its top holdings all thrust at
    once", and does the holdings-thrust state have different forward stats?
 d) event/earnings risk inside a 2026-08-14 entry .. ~2026-08-21 exit.
 e) cheap verifications the composer needs: Wilder-14 ATR, last close,
    median 21d dollar volume.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

H = 5
HOLD = ["ABT", "ISRG", "BSX", "MDT", "SYK", "EW", "BDX", "ZBH", "DXCM",
        "RMD", "PODD", "STE", "COO", "HOLX", "BAX", "GEHC", "ALGN", "MASI"]
px_map = load_prices(["IHI"] + HOLD)
ihi_df = px_map["IHI"]
c = ihi_df["Close"].dropna()
r21 = pct_rank(c, 21)
r21raw = c.pct_change(21)
dd = c / c.rolling(252).max() - 1.0
m = ((r21 >= 99) & (dd <= -0.10)).fillna(False)
ret = fwd_lag(c, H)
trig = c.index[m.values & ret.notna().values]
epi = declusters(trig, 5, c.index)
epi = epi[ret.reindex(epi).notna().values]
v = ret.loc[epi].values
span = (c.index >= trig[0]) & (c.index <= trig[-1]) & ret.notna().values
ctrl = ret[span].values
base = float((ctrl > 0).mean())

print("=== a) CLUSTER DEPTH (today = 3) ===")
mv = m.reindex(c.index).fillna(False).values
dep = np.zeros(len(mv), int)
for i in range(len(mv)):
    dep[i] = (dep[i-1] + 1) if (mv[i] and i > 0) else (1 if mv[i] else 0)
dser = pd.Series(dep, index=c.index)
print(f"  today depth = {int(dser.iloc[-1])} (last bar {c.index[-1].date()})")
rows = []
alltrig = c.index[mv & ret.notna().values]
for lo, hi in [(1, 1), (2, 2), (3, 3), (4, 99), (1, 2), (3, 99)]:
    sel = alltrig[(dser.loc[alltrig] >= lo) & (dser.loc[alltrig] <= hi)]
    if len(sel) == 0:
        continue
    vv = ret.loc[sel].values
    w = int((vv > 0).sum())
    rows.append({"depth": f"{lo}-{hi}", "n_days": len(vv),
                 "mean_pct": round(100*vv.mean(), 3),
                 "hit": round(100*(vv > 0).mean(), 1),
                 "excess_pp": round(100*(vv.mean()-ctrl.mean()), 3),
                 "sign_p_base": round(sign_test(w, len(vv), base), 4),
                 "worst_pct": round(100*vv.min(), 2)})
print(pd.DataFrame(rows).to_string(index=False))
d3 = alltrig[dser.loc[alltrig] == 3]
print(f"  depth-3 DAYS: N={len(d3)} -> {', '.join(str(x.date()) for x in d3)}")
print("  depth is a DAY-LEVEL cut of 34 trigger days; the pitched statistic is")
print("  the 16-EPISODE cell, which by construction takes depth-1 days only:")
print(f"  depth of the 16 episodes: {sorted(dser.loc[epi].values.tolist())}")
print("  -> today enters at depth 3, i.e. NOT the state the +1.499% was measured on.")
vE = ret.loc[epi].values
sel3 = alltrig[dser.loc[alltrig] >= 3]
print(f"  episodes (depth-1 by construction) mean {100*vE.mean():+.3f}%  vs  "
      f"depth>=3 days mean {100*ret.loc[sel3].mean():+.3f}% (N={len(sel3)})")

print("\n=== b) MAGNITUDE vs RANK (registry trap) ===")
mag = 100 * r21raw.loc[epi].values
tod = 100 * r21raw.iloc[-1]
tab = pd.DataFrame({"episode": [str(d.date()) for d in epi],
                    "ret21_pct": np.round(mag, 2),
                    "r21_rank": np.round(r21.loc[epi].values, 1),
                    "dd52wh_pct": np.round(100*dd.loc[epi].values, 2),
                    "fwd5_pct": np.round(100*v, 2)}).sort_values("ret21_pct")
print(tab.to_string(index=False))
print(f"  TODAY ret21 = {tod:+.2f}%, rank {r21.iloc[-1]:.1f}, dd {100*dd.iloc[-1]:.2f}%")
print(f"  episode ret21 distribution: min {mag.min():+.2f} p25 "
      f"{np.percentile(mag,25):+.2f} median {np.median(mag):+.2f} p75 "
      f"{np.percentile(mag,75):+.2f} max {mag.max():+.2f}")
print(f"  today's {tod:+.2f}% sits at the "
      f"{100*(mag < tod).mean():.0f}th percentile of the winning-cell magnitudes "
      f"({int((mag >= tod).sum())} of {len(mag)} episodes were as large or larger)")
w = mag >= 10.0
print(f"  episodes with ret21 >= 10%: N={int(w.sum())} mean {100*v[w].mean():+.3f}% "
      f"hit {100*(v[w]>0).mean():.1f}%  |  < 10%: N={int((~w).sum())} mean "
      f"{100*v[~w].mean():+.3f}% hit {100*(v[~w]>0).mean():.1f}%")
print(f"  corr(ret21 magnitude, fwd5) = {np.corrcoef(mag, 100*v)[0,1]:+.3f}")
w2 = mag >= tod
if w2.sum():
    print(f"  episodes AT LEAST as extreme as today (>= {tod:.2f}%): N={int(w2.sum())} "
          f"mean {100*v[w2].mean():+.3f}% hit {100*(v[w2]>0).mean():.1f}% "
          f"dates {[str(d.date()) for d in epi[w2]]}")

print("\n=== c) COMPOSITION: is it just 'the top holdings all thrust'? ===")
have = [t for t in HOLD if t in px_map]
today = {}
for t in have:
    s = px_map[t]["Close"].dropna()
    if s.index[-1] < c.index[-1] - pd.Timedelta(days=7):
        continue
    today[t] = (pct_rank(s, 21).iloc[-1], 100*(s.iloc[-1]/s.iloc[-22]-1))
print("  today's holding r21 rank / ret21:")
print("   " + "  ".join(f"{t}:{a:.0f}/{b:+.1f}%" for t, (a, b) in
                        sorted(today.items(), key=lambda kv: -kv[1][0])))
# breadth series: fraction of available holdings with r21 >= 95
rk = pd.DataFrame({t: pct_rank(px_map[t]["Close"], 21) for t in have})
rk = rk.reindex(c.index)
avail = rk.notna().sum(axis=1)
brd = (rk >= 95).sum(axis=1) / avail.replace(0, np.nan)
print(f"  breadth (frac of holdings with r21>=95): today "
      f"{brd.iloc[-1]:.2f} ({int((rk.iloc[-1] >= 95).sum())} of "
      f"{int(avail.iloc[-1])} names), full-history median {brd.median():.2f}, "
      f"p95 {brd.quantile(0.95):.2f}")
print(f"  breadth on the 16 episodes: median {brd.loc[epi].median():.2f}, "
      f"values {np.round(brd.loc[epi].values, 2).tolist()}")
# a HOLDINGS-thrust trigger, no ETF rank at all
rows = []
for thr in (0.4, 0.5, 0.6):
    mm = ((brd >= thr) & (dd <= -0.10)).fillna(False)
    tt = c.index[mm.values & ret.notna().values]
    if len(tt) == 0:
        rows.append({"breadth>=": thr, "n": 0})
        continue
    e = declusters(tt, 5, c.index)
    e = e[ret.reindex(e).notna().values]
    vv = ret.loc[e].values
    rows.append({"breadth>=": thr, "n_days": len(tt), "n_epi": len(vv),
                 "mean_pct": round(100*vv.mean(), 3),
                 "hit": round(100*(vv > 0).mean(), 1),
                 "excess_pp": round(100*(vv.mean()-ctrl.mean()), 3)})
print("  HOLDINGS-BREADTH trigger (no ETF rank gate), long IHI h=5:")
print(pd.DataFrame(rows).to_string(index=False))
ov = ((brd >= 0.5) & m).fillna(False)
print(f"  overlap: of {int(m.sum())} ETF-trigger days, "
      f"{int(ov.sum())} also have breadth>=0.5 "
      f"({100*ov.sum()/max(1,m.sum()):.0f}%)")

print("\n=== d) EVENTS / EARNINGS inside entry 2026-08-14 .. exit ~2026-08-21 ===")
ev = load_events()
print(ev[(ev["date"] >= "2026-08-13") & (ev["date"] <= "2026-08-24")].to_string(index=False))
try:
    ec = pd.read_parquet(Path(__file__).resolve().parents[3] / "data" /
                         "earnings_calendar.parquet")
    dc = [x for x in ec.columns if "date" in x.lower()][0]
    tc = [x for x in ec.columns if "tick" in x.lower() or x.lower() == "symbol"][0]
    ec[dc] = pd.to_datetime(ec[dc])
    sel = ec[(ec[tc].isin(HOLD)) & (ec[dc] >= "2026-08-13") & (ec[dc] <= "2026-08-24")]
    print("  holding earnings in window:",
          sel[[tc, dc]].to_string(index=False) if len(sel) else "NONE of 18 names")
except Exception as e:  # noqa: BLE001
    print("  earnings parquet unavailable:", e)
for kinds in [("vix_expiry",), ("opex",)]:
    fl = event_in_window(epi, c.index, H, 1, kinds)
    show([summarize(v[fl], f"{kinds[0]} IN hold (N={int(fl.sum())})"),
          summarize(v[~fl], f"{kinds[0]} OUT (N={int((~fl).sum())})")])

print("\n=== e) COMPOSER INPUTS ===")
atr = pd.Series(wilder_atr(ihi_df["High"], ihi_df["Low"], ihi_df["Close"], 14),
                index=ihi_df.index).dropna()
last = c.iloc[-1]
print(f"  last bar         : {c.index[-1].date()}")
print(f"  close            : {last:.2f}")
print(f"  Wilder-14 ATR    : {atr.iloc[-1]:.4f}  ({100*atr.iloc[-1]/last:.2f}% of price)")
dv = (ihi_df["Close"] * ihi_df["Volume"]).dropna()
print(f"  median 21d $vol  : ${dv.iloc[-21:].median():,.0f}  "
      f"(median share vol {ihi_df['Volume'].iloc[-21:].median():,.0f})")
print(f"  min 21d $vol     : ${dv.iloc[-21:].min():,.0f}")
sz = 0.0030 * 750_000 / (1.0 * atr.iloc[-1])   # 30 bps risk, 1 ATR risk unit
print(f"  30bps of $750k at a 1.0-ATR risk unit = {sz:,.0f} shares = "
      f"${sz*last:,.0f} notional = "
      f"{100*sz*last/dv.iloc[-21:].median():.2f}% of median daily $vol")
