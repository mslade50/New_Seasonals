"""kD round 3 (development): c10 LONG dollar after the FOMC decision.

1. today's state (regime126 on the 2026-09-15 eve)
2. month+tdom control drop on DX (R1 +0.172 tdomX -> +0.036 month+tdom):
   same-subset comparison + calendar-month-matched control
3. horizon_scan h=1..10, DX-Y.NYB and UUP, entry D close (lag=0 on D)
4. UUP entry variants as WHOLE variants per decision (unfilled = 0):
   MOC on D close vs LIMIT(CLOSE anchor = eve close, -k Wilder-14 ATR)
   filled on D (fill = min(open, limit)), exit close D+h
5. exits: time-only vs stop k ATR armed day 2, pessimistic (gap -> open
   minus 13bp), whole-variant mean
6. loser paths (episode_paths), what kills it
7. 2022 / 2025 / 2026 episodes individually
8. SPY link: post-FOMC dollar vs SPY same window
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_grammar import wilder_atr

import numpy as np
import pandas as pd

pxd = load_prices(["DX-Y.NYB", "UUP", "^IRX", "SPY"])
irx = pxd["^IRX"]["Close"].dropna()
fomc = load_events(["fomc_decision"])["date"]
fomc = pd.DatetimeIndex(fomc[fomc <= pd.Timestamp("2026-09-15")])


def tdom_of(ix):
    ym = pd.Series(ix.year * 100 + ix.month, index=ix)
    return ym.groupby(ym.values).cumcount().values + 1


# 1. today's state
e = irx.index[-1]
ic = irx - irx.shift(126)
print(f"1. eve {e.date()}: ^IRX {irx[e]:.3f}  126d chg {ic[e]:+.3f}  "
      f"-> regime126 {'zirp' if irx[e] < 0.3 else ('hike' if ic[e] > 0.25 else ('cut' if ic[e] < -0.25 else 'flat'))}"
      f"   (63d chg {irx[e] - irx.shift(63)[e]:+.3f})")

dx = pxd["DX-Y.NYB"]
s = dx["Close"].dropna()
idx = s.index
TD = tdom_of(idx)
MON = idx.month.values
POS = pd.Series(np.arange(len(idx)), index=idx)
dp = np.array([POS[d] for d in fomc if d in POS.index])
EX = np.zeros(len(idx), bool)
for p in dp:
    EX[max(0, p - 5):p + 6] = True
r3 = (s.shift(-3) / s - 1.0).values
ok = ~np.isnan(r3)
dp = dp[ok[dp]]
b_td = {j: np.nanmean(r3[(TD == j) & ~EX & ok]) for j in np.unique(TD)}
b_mon = {m: np.nanmean(r3[(MON == m) & ~EX & ok]) for m in range(1, 13)}
xt, xm, xmt, nmt = [], [], [], []
for p in dp:
    xt.append(r3[p] - b_td[TD[p]])
    xm.append(r3[p] - b_mon[MON[p]])
    mm = (MON == MON[p]) & (TD == TD[p]) & ~EX & ok
    nmt.append(mm.sum())
    xmt.append(r3[p] - np.nanmean(r3[mm]) if mm.sum() >= 8 else np.nan)
xt, xm, xmt, nmt = map(np.array, (xt, xm, xmt, nmt))
sub = ~np.isnan(xmt)
print(f"\n2. controls on DX h3: tdomX all {100 * xt.mean():+.3f}% (n {len(xt)}); month-matched X "
      f"{100 * xm.mean():+.3f}% (t {xm.mean() / (xm.std(ddof=1) / np.sqrt(len(xm))):.2f}); "
      f"month+tdom X {100 * np.nanmean(xmt):+.3f}% on its subset n {sub.sum()} "
      f"(median matched days {int(np.median(nmt))}); tdomX on that SAME subset {100 * xt[sub].mean():+.3f}%, "
      f"raw on subset {100 * r3[dp][sub].mean():+.3f}%, excluded rows raw {100 * r3[dp][~sub].mean():+.3f}% n {(~sub).sum()}")
print("   month bucket means h3 (%, non-FOMC windows):", {m: round(100 * b_mon[m], 3) for m in range(1, 13)})

# 3. horizon scans
for tk in ("DX-Y.NYB", "UUP"):
    px = pd.DataFrame({tk: pxd[tk]["Close"]}).dropna()
    show(horizon_scan(px, fomc, [(tk, 1.0)], hs=tuple(range(1, 11)), lag=0, min_gap=1),
         f"3. horizon_scan {tk} long, entry D close (lag=0 on D)")

# 4/5. UUP entry + exit variants
u = pxd["UUP"].dropna()
uo, uh, ul, uc = (u[c].values for c in ("Open", "High", "Low", "Close"))
atr = wilder_atr(uh, ul, uc)
up = pd.Series(np.arange(len(u)), index=u.index)
ud = [d for d in fomc if d in up.index and up[d] >= 20 and up[d] + 10 < len(u)]


def trade(p, h, k=None, stop=None):
    """Return fraction for one decision or np.nan if no fill."""
    a = atr[p - 1]
    if k is None:
        entry = uc[p]
    else:
        lim = uc[p - 1] - k * a
        if ul[p] > lim:
            return np.nan
        entry = min(uo[p], lim)
    if stop is not None:
        sp = entry - stop * a
        for j in range(p + 1, p + h + 1):  # fill on D (MOC or intraday limit) -> stop arms D+1
            if uo[j] <= sp:
                return (uo[j] * (1 - 13e-4)) / entry - 1
            if ul[j] <= sp:
                return (sp * (1 - 3e-4)) / entry - 1
    return uc[p + h] / entry - 1


rows = []
for h in (3, 5):
    for k in (None, 0.0, 0.1, 0.25, 0.5):
        for stop in (None, 1.0, 1.5):
            v = np.array([trade(up[d], h, k, stop) for d in ud])
            f = ~np.isnan(v)
            whole = np.where(f, v, 0.0)
            rows.append({"h": h, "entry": "MOC D" if k is None else f"LIMIT eveC-{k}ATR",
                         "stop": stop or "none", "fills": f"{f.sum()}/{len(v)}",
                         "fill_mean%": round(100 * v[f].mean(), 3),
                         "fill_hit": round(100 * (v[f] > 0).mean(), 1),
                         "whole_mean%": round(100 * whole.mean(), 3),
                         "whole_t": round(whole.mean() / (whole.std(ddof=1) / np.sqrt(len(whole))), 2),
                         "worst%": round(100 * v[f].min(), 2)})
print("\n4/5. UUP whole variants (unfilled = 0), exit close D+h; stop pessimistic, arms day 2")
print(pd.DataFrame(rows).to_string(index=False))
print(f"   UUP ATR% of price on eve 2026-09-15: {100 * atr[-1] / uc[-1]:.3f}%  close {uc[-1]:.2f} ATR {atr[-1]:.4f}")

# 6. loser paths DX h=3
px = pd.DataFrame({"DX": s})
dd = pd.DatetimeIndex([d for d in fomc if d in POS.index])
paths = episode_paths(px, dd, [("DX", 1.0)], 5, lag=0)
fin3 = paths[3]
losers = paths[fin3 < 0]
print(f"\n6. DX loser paths (h=3 final < 0): n {len(losers)}/{len(paths)}; loser mean final "
      f"{100 * losers[3].mean():+.3f}%; loser mean day1 {100 * losers[1].mean():+.3f}%")
d1 = paths[1]
for thr in (-0.003, -0.005, -0.0075):
    m = d1 <= thr
    print(f"   day1 <= {100 * thr:.2f}%: n {m.sum()}, h3 final mean {100 * paths[3][m].mean():+.3f}%, "
          f"h3 hit {100 * (paths[3][m] > 0).mean():.0f}%, h5 mean {100 * paths[5][m].mean():+.3f}%")
m = d1 > 0
print(f"   day1 > 0: n {m.sum()}, h3 final mean {100 * paths[3][m].mean():+.3f}%")
print("   worst 6 h3 episodes (cum % day1..5):")
print((100 * paths.loc[fin3.nsmallest(6).index]).round(2).to_string())
q = np.percentile(100 * fin3, [5, 10, 25, 50])
print(f"   h3 final pct 5/10/25/50: {q.round(3)}")

# 7. episodes 2022 / 2025 / 2026
r1 = (s / s.shift(1) - 1.0).values
sp = pxd["SPY"]["Close"].reindex(idx)
tab = []
for d in dd:
    if d.year not in (2022, 2025, 2026):
        continue
    p = POS[d]
    tab.append({"date": d.date(), "eve->D%": round(100 * r1[p], 3),
                "h1%": round(100 * (s.iloc[p + 1] / s.iloc[p] - 1), 3),
                "h3%": round(100 * (s.iloc[p + 3] / s.iloc[p] - 1), 3),
                "h3_tdomX%": round(100 * (r3[p] - b_td[TD[p]]), 3),
                "h5%": round(100 * (s.iloc[p + 5] / s.iloc[p] - 1), 3) if p + 5 < len(s) else np.nan,
                "SPY_h3%": round(100 * (sp.iloc[p + 3] / sp.iloc[p] - 1), 2)})
print("\n7. 2022 / 2025 / 2026 decisions, DX long")
t7 = pd.DataFrame(tab)
print(t7.to_string(index=False))
for y in (2022, 2025, 2026):
    g = t7[pd.to_datetime(t7.date).dt.year == y]
    print(f"   {y}: n {len(g)} h3 mean {g['h3%'].mean():+.3f}% rec {(g['h3%'] > 0).sum()}-{(g['h3%'] <= 0).sum()}")

# 8. SPY link
spr = (sp.shift(-3) / sp - 1.0).values
sp1 = (sp / sp.shift(1) - 1.0).values
okk = ~np.isnan(spr[dp])
print(f"\n8. FOMC: SPY eve->D {100 * np.nanmean(sp1[dp]):+.3f}%  SPY D->D+3 {100 * np.nanmean(spr[dp]):+.3f}%  "
      f"corr(DX h3, SPY h3) {np.corrcoef(r3[dp][okk], spr[dp][okk])[0, 1]:+.3f}")
for lbl, m in (("SPY h3 < 0", spr[dp] < 0), ("SPY h3 >= 0", spr[dp] >= 0)):
    m = m & okk
    print(f"   {lbl}: n {m.sum()} DX h3 mean {100 * r3[dp][m].mean():+.3f}%")

# 9. subsets matching today at h=3 and h=5 (DX tdomX)
ic126 = irx - irx.shift(126)
print("\n9. DX subsets, tdom-matched excess (today: Sept, midterm, hike regime, quad witch in hold)")
out = []
for h in (3, 5):
    r = (s.shift(-h) / s - 1.0).values
    okh = ~np.isnan(r)
    bt = {j: np.nanmean(r[(TD == j) & ~EX & okh]) for j in np.unique(TD)}
    dph = np.array([POS[d] for d in fomc if d in POS.index])
    dph = dph[okh[dph]]
    dts = idx[dph]
    x = np.array([r[p] - bt[TD[p]] for p in dph])
    yrs = dts.year.values
    qw = event_in_window(dts, idx, h, lag=0, kinds=("quad_witching",))
    reg = np.array([("hike" if ic126.asof(idx[p - 1]) > 0.25 and irx.asof(idx[p - 1]) >= 0.3 else "other")
                    for p in dph])
    subs = {"all": np.ones(len(x), bool), "pre-2008": yrs < 2008,
            "2008-2017": (yrs >= 2008) & (yrs < 2018), "2018+": yrs >= 2018,
            "midterm": yrs % 4 == 2, "quad witch IN": qw, "quad witch OUT": ~qw,
            "September": dts.month.values == 9, "hike regime": reg == "hike",
            "QW IN & hike": qw & (reg == "hike")}
    for nm, m in subs.items():
        v = x[m]
        wn = int((v > 0).sum())
        out.append({"h": h, "subset": nm, "n": len(v), "tdomX%": round(100 * v.mean(), 3),
                    "t": round(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))), 2) if len(v) > 2 else np.nan,
                    "rec": f"{wn}-{len(v) - wn}", "sign_p": round(sign_test(wn, len(v)), 3)})
print(pd.DataFrame(out).to_string(index=False))
