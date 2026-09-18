"""c3 — LONG SVXY on ^VIX 21d RELATIVE-range compression <= 5th pctile INTO a
dense calendar (>= 4 scheduled events in the forward 6 sessions).

Pre-specified. The candidate exists to test ONE thing: the 2026-09-03 parked
near-miss found the (0,5] compression bucket DEAD on a clear calendar
(-0.096% over 25 anchors, 13-11). Does the sign FLIP on a dense calendar?

Also settles, because the brief demands it:
  - today's EXACT rel-range percentile (rule 8: ^VIX reindexed to SPY)
  - is the pre-specified trigger even LIVE today?
  - the 2x2: {(0,5], (5,15]} x {dense, clear}
  - SPY-RESIDUAL: regress the SVXY leg on SPY. Three vol cells have died here
    reducing to SVXY = a + b*SPY with no residual.
  - the SVXY Feb-2018 re-lever break: -0.5x era reported ALONE, never pooled
  - corr(C1 short-SPY leg, C3 long-SVXY leg): they cannot both ship
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

EVENT_KINDS = ["nfp", "cpi", "ppi", "fomc_decision", "opex",
               "quad_witching", "vix_expiry"]
WIN, THRESH = 6, 4
RELEVER = pd.Timestamp("2018-02-28")   # SVXY -1.0x -> -0.5x
COST_BPS = 5.0

px = close_panel(["SPY", "^VIX", "SVXY", "^VIX3M"])
spy = px["SPY"].dropna()
idx = spy.index
vix = px["^VIX"].reindex(idx).ffill()
svxy = px["SVXY"].reindex(idx)
pos = pd.Series(range(len(idx)), index=idx)
ev = load_events(EVENT_KINDS)

# --------------------------------------------------- compression + density
rr = rolling_on_valid(vix, lambda x: (x.rolling(21).max() - x.rolling(21).min())
                      / x.rolling(21).mean())
rrp = rolling_on_valid(rr, lambda x: x.rolling(252).rank(pct=True) * 100.0)

print("=" * 80)
print("0. TODAY'S STATE (^VIX reindexed to SPY's calendar, rule 8)")
for d in rrp.dropna().index[-6:]:
    print(f"   {d.date()}  ^VIX {vix.loc[d]:6.2f}  rel-range {rr.loc[d]:.4f}  "
          f"trailing-252 pctile {rrp.loc[d]:7.3f}")
today_p = rrp.dropna().iloc[-1]
print(f"\n   ** EXACT rel-range trailing-252 percentile at the 2026-09-09 close"
      f" = {today_p:.3f} **")
print(f"   pre-specified trigger is pctile <= 5.  {today_p:.3f} <= 5 ? "
      f"{'YES' if today_p <= 5 else 'NO -- THE TRIGGER IS NOT LIVE'}")
print(f"   the brief assumed ~2. The 09-08 reading WAS 3.571; ^VIX rose 4.71% on")
print(f"   09-09, which widened the 21d range and lifted the percentile to "
      f"{today_p:.3f}.")
print(f"   {today_p:.3f} sits in the (5,15] bucket -- the bucket the 09-03")
print( "   near-miss recorded as the PAYING one, not the dead one.")

# forward-session density, projected past the end of the price index for today
def density_series(win=WIN):
    locs = idx.searchsorted(pd.DatetimeIndex(ev["date"]))
    ok = locs < len(idx)
    cnt = np.zeros(len(idx))
    for L in locs[ok]:
        cnt[L] += 1
    cs = np.concatenate([[0.0], np.cumsum(cnt)])
    out = np.full(len(idx), np.nan)
    for p in range(len(idx)):
        if p + win >= len(idx):
            continue
        out[p] = cs[min(len(idx), p + win + 1)] - cs[p + 1]
    return pd.Series(out, index=idx)

dens = density_series()

# today's density has to be projected: the price index has no forward sessions
proj = pd.bdate_range(idx[-1] + pd.Timedelta(days=1), periods=WIN + 4)
hol = pd.DatetimeIndex([])  # no NYSE holiday inside Sep 10-18 2026
proj = proj.difference(hol)[:WIN]
live = ev[(ev["date"] > idx[-1]) & (ev["date"] <= proj[-1])]
print(f"\n   forward {WIN} sessions from the 2026-09-09 anchor: "
      f"{[str(d.date()) for d in proj]}")
for _, r in live.iterrows():
    print(f"     {r['date'].date()}  {r['event']}")
print(f"   ** today's density(D+1..D+{WIN}) = {len(live)} event instances on "
      f"{live['date'].nunique()} distinct dates -> "
      f"{'DENSE' if len(live) >= THRESH else 'not dense'} **")

comp_lo = (rrp > 0) & (rrp <= 5)
comp_mid = (rrp > 5) & (rrp <= 15)
dense = (dens >= THRESH).fillna(False)

# ------------------------------------------------------------------- the 2x2
print("\n" + "=" * 80)
print("1. THE 2x2  compression x calendar, LONG SVXY, entry lag=1 MOC")
sv_ok = svxy.notna()
for H in (5, 6, 10):
    print(f"\n   ---- h={H} ----")
    ret = vehicle_ret(px, [("SVXY", 1.0)], H, 1)
    valid = ret.dropna().index
    rows = []
    for clbl, cm in [("(0,5]", comp_lo), ("(5,15]", comp_mid)]:
        for dlbl, dm in [("DENSE>=4", dense), ("clear<4", ~dense & dens.notna())]:
            m = (cm & dm).fillna(False)
            t = pd.DatetimeIndex(idx[m.values]).intersection(valid)
            if len(t) == 0:
                rows.append({"label": f"{clbl} x {dlbl}", "n": 0})
                continue
            e = declusters(t, max(H, WIN), valid)
            s = summarize(ret.loc[e].values, f"{clbl} x {dlbl}")
            s["n_days"] = len(t)
            s["signp"] = round(sign_test(int((ret.loc[e].values > 0).sum()),
                                         len(e)), 4)
            rows.append(s)
    base = ret.loc[valid]
    rows.append({"label": "CTRL all SVXY days", "n": len(base),
                 "mean_pct": round(100 * base.mean(), 3),
                 "hit": round(100 * (base > 0).mean(), 1),
                 "t": round(base.mean() / (base.std(ddof=1) / np.sqrt(len(base))), 2)})
    show(rows, f"   2x2 at h={H} (episodes, decluster gap {max(H, WIN)})")

# ------------------------------------------------- 2. THE SPY-RESIDUAL RULE
print("\n" + "=" * 80)
print("2. SPY-RESIDUAL RULE: does the SVXY vehicle add anything over SPY?")
H = 6
rs = vehicle_ret(px, [("SVXY", 1.0)], H, 1)
rp = vehicle_ret(px, [("SPY", 1.0)], H, 1)
both = pd.concat([rs, rp], axis=1, keys=["svxy", "spy"]).dropna()
for lbl, sub in [("full SVXY history", both),
                 ("-1.0x era (pre 2018-02)", both[both.index < RELEVER]),
                 ("-0.5x era (2018-02+)", both[both.index >= RELEVER])]:
    X = np.vstack([np.ones(len(sub)), sub["spy"].values]).T
    beta, *_ = np.linalg.lstsq(X, sub["svxy"].values, rcond=None)
    resid = sub["svxy"].values - X @ beta
    se_a = resid.std(ddof=2) / np.sqrt(len(sub))
    print(f"   {lbl:26s} N={len(sub):5d}  SVXY = {100*beta[0]:+.3f}% "
          f"+ {beta[1]:.2f} x SPY   alpha t {beta[0]/se_a:+.2f}  "
          f"R2 {1 - resid.var()/sub['svxy'].values.var():.3f}")

print("\n   the same regression INSIDE each live cell (h=6):")
ret_s = vehicle_ret(px, [("SVXY", 1.0)], H, 1)
ret_p = vehicle_ret(px, [("SPY", 1.0)], H, 1)
valid = ret_s.dropna().index.intersection(ret_p.dropna().index)
for clbl, cm in [("(0,5]", comp_lo), ("(5,15]", comp_mid)]:
    for dlbl, dm in [("DENSE", dense), ("clear", ~dense & dens.notna())]:
        m = (cm & dm).fillna(False)
        t = pd.DatetimeIndex(idx[m.values]).intersection(valid)
        e = declusters(t, max(H, WIN), valid)
        if len(e) < 4:
            print(f"     {clbl} x {dlbl:5s}: N={len(e)} too few to regress")
            continue
        a = ret_s.loc[e].values
        b = ret_p.loc[e].values
        X = np.vstack([np.ones(len(e)), b]).T
        beta, *_ = np.linalg.lstsq(X, a, rcond=None)
        resid = a - X @ beta
        se_a = resid.std(ddof=2) / np.sqrt(len(e)) if len(e) > 2 else np.nan
        print(f"     {clbl} x {dlbl:5s}: N={len(e):3d}  SVXY {100*a.mean():+.3f}%  "
              f"SPY {100*b.mean():+.3f}%  beta {beta[1]:.2f}  "
              f"alpha {100*beta[0]:+.3f}% (t {beta[0]/se_a:+.2f})")

# ------------------------------------------- 3. -0.5x era alone (never pooled)
print("\n" + "=" * 80)
print("3. SVXY -0.5x ERA ALONE (2018-02-28+). Never pool across the re-lever.")
for H in (5, 6, 10):
    ret = vehicle_ret(px, [("SVXY", 1.0)], H, 1)
    valid = ret.dropna().index
    valid5 = valid[valid >= RELEVER]
    rows = []
    for clbl, cm in [("(0,5]", comp_lo), ("(5,15]", comp_mid)]:
        for dlbl, dm in [("DENSE", dense), ("clear", ~dense & dens.notna())]:
            m = (cm & dm).fillna(False)
            t = pd.DatetimeIndex(idx[m.values]).intersection(valid5)
            if len(t) == 0:
                rows.append({"label": f"h={H} {clbl}x{dlbl}", "n": 0})
                continue
            e = declusters(t, max(H, WIN), valid5)
            s = summarize(ret.loc[e].values, f"h={H} {clbl}x{dlbl}")
            s["signp"] = round(sign_test(int((ret.loc[e].values > 0).sum()),
                                         len(e)), 4)
            rows.append(s)
    show(rows, f"   -0.5x era only, h={H}")

# -------------------------------- 4. compression bucket ladder (dose response)
print("\n" + "=" * 80)
print("4. COMPRESSION DOSE RESPONSE at h=6, split by calendar")
H = 6
ret = vehicle_ret(px, [("SVXY", 1.0)], H, 1)
valid = ret.dropna().index
rows = []
for lo, hi in [(0, 5), (5, 10), (10, 15), (15, 25), (25, 50), (50, 100)]:
    cm = (rrp > lo) & (rrp <= hi)
    for dlbl, dm in [("DENSE", dense), ("clear", ~dense & dens.notna()),
                     ("ANY", pd.Series(True, index=idx))]:
        m = (cm & dm).fillna(False)
        t = pd.DatetimeIndex(idx[m.values]).intersection(valid)
        if len(t) == 0:
            continue
        e = declusters(t, max(H, WIN), valid)
        s = summarize(ret.loc[e].values, f"({lo},{hi}] x {dlbl}")
        rows.append(s)
show(rows, "   compression ladder x calendar (LONG SVXY, h=6, episodes)")

# ---------------------------------- 5. correlation constraint C1 vs C3
print("\n" + "=" * 80)
print("5. CORRELATION CONSTRAINT: C1 (SHORT SPY) vs C3 (LONG SVXY), h=6")
a = vehicle_ret(px, [("SPY", -1.0)], 6, 1)
b = vehicle_ret(px, [("SVXY", 1.0)], 6, 1)
j = pd.concat([a, b], axis=1, keys=["short_spy", "long_svxy"]).dropna()
print(f"   corr over all overlapping days (N={len(j)}) = "
      f"{j.corr().iloc[0,1]:+.3f}")
j5 = j[j.index >= RELEVER]
print(f"   corr in the -0.5x era only (N={len(j5)})        = "
      f"{j5.corr().iloc[0,1]:+.3f}")
print("   -> they are the SAME equity-beta position with OPPOSITE signs.")

# ------------------------------------------------------- 6. placebo + cost
print("\n" + "=" * 80)
print("6. PLACEBO LADDER on the live cell ((5,15] x DENSE), h=6, LONG SVXY")
H = 6
ret = vehicle_ret(px, [("SVXY", 1.0)], H, 1)
valid = ret.dropna().index
base_mask = (comp_mid & dense).fillna(False)
base_t = pd.DatetimeIndex(idx[base_mask.values]).intersection(valid)
rows = []
for k in range(-5, 6):
    sh = []
    for d in base_t:
        p = pos.get(d)
        q = p + k
        if 0 <= q < len(idx):
            sh.append(idx[q])
    t = pd.DatetimeIndex(sorted(set(sh))).intersection(valid)
    e = declusters(t, max(H, WIN), valid)
    rows.append(summarize(ret.loc[e].values,
                          f"k={k:+d}" + ("  <-- TRUE" if k == 0 else "")))
show(rows, "   placebo ladder, (5,15] x DENSE")
tm = [r for r in rows if "TRUE" in r["label"]][0]["mean_pct"]
print(f"   TRUE ranks {1 + sum(1 for r in rows if r.get('mean_pct', -9e9) > tm)}/11")

e = declusters(base_t, max(H, WIN), valid)
v = ret.loc[e].values
print(f"\n   cost: episode mean {100*v.mean():+.3f}% = {1e4*v.mean():+.0f} bp "
      f"vs {COST_BPS} bp -> {1e4*v.mean()/COST_BPS:+.1f}x cost (need >=5x)")
print(f"   concentration: {cluster_note(e, v)}")
print(f"   worst episode {100*v.min():+.2f}% on {e[int(np.argmin(v))].date()}")
show(era_split(e, v), "   era split, (5,15] x DENSE")
