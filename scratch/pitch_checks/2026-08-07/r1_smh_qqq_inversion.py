"""RED TEAM / INVERSION 1: SHORT SMH / LONG QQQ on the laggard-snapback trigger.

The prior checker (c10) measured LONG SMH / SHORT QQQ and found it dead at h=10
but reported h=5 at -0.447% t=-2.30 day-level, i.e. the SHORT SMH side made
money. This script asks whether that sign flip is a trade or an artifact.

TWO CORRECTIONS vs c10:
  * c10 used `fwd_ret` = lag 0 (signal close -> close). The REAL order is entry
    MOC on D+1 (today 2026-08-07 = trigger day + 1). Everything here is lag=1.
    lag=0 is reported alongside so the discrepancy is visible.
  * everything is reported in TRADE terms: trade = short SMH / long QQQ, so
    trade_ret = r_QQQ - r_SMH. Positive = the inversion makes money.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import *  # noqa

import numpy as np
import pandas as pd

px = load_prices(["SMH", "QQQ", "SPY"])
cal = px["SMH"]["Close"].dropna().index
for t in ("QQQ", "SPY"):
    cal = cal.intersection(px[t]["Close"].dropna().index)
smh = px["SMH"]["Close"].reindex(cal)
qqq = px["QQQ"]["Close"].reindex(cal)
spy = px["SPY"]["Close"].reindex(cal)
pos = pd.Series(range(len(cal)), index=cal)

r63 = pct_rank(smh, 63)
r5 = pct_rank(smh, 5)
spy200 = spy.rolling(200).mean()
spy_above = spy > spy200
spy_dist = spy / spy200 - 1.0


def fwd_lag(s, h, lag=1):
    return s.shift(-(lag + h)) / s.shift(-lag) - 1.0


def trade(h, lag=1):
    """SHORT SMH / LONG QQQ, equal dollar. Positive = inversion wins."""
    return fwd_lag(qqq, h, lag) - fwd_lag(smh, h, lag)


TRIG_MASK = (r63 <= 10) & (r5 >= 80)


def trig_dates(h, lag=1, extra=None):
    m = TRIG_MASK.copy()
    if extra is not None:
        m = m & extra
    ok = trade(h, lag).notna()
    return cal[m.fillna(False).values & ok.reindex(cal).fillna(False).values]


print("=" * 78)
print("INVERSION 1 -- SHORT SMH / LONG QQQ  (trigger: SMH rank63<=10 & rank5>=80)")
print("=" * 78)
print(f"common calendar {cal[0].date()} .. {cal[-1].date()}  n={len(cal)}")
print(f"TODAY (2026-08-06 close): SMH rank63={r63.iloc[-1]:.2f}  rank5={r5.iloc[-1]:.2f}  "
      f"fires={bool(TRIG_MASK.iloc[-1])}")
print(f"TODAY SPY: {'ABOVE' if spy_above.iloc[-1] else 'BELOW'} 200d SMA, "
      f"{100*spy_dist.iloc[-1]:+.2f}% vs 200d")

t5 = trig_dates(5)
print(f"\ntrigger days with a valid lag=1 h=5 window: {len(t5)}  "
      f"({t5[0].date()} .. {t5[-1].date()})")
print("  by year:", dict(pd.Series(1, index=t5).groupby(t5.year).sum()))

# ---------------------------------------------------------------- 1) h=5 cell
print("\n" + "=" * 78)
print("1) h=5 CELL: day-level vs episodes, lag=1 (real order) and lag=0 (c10 basis)")
print("=" * 78)
rows = []
for lag in (1, 0):
    tt = trig_dates(5, lag)
    v = trade(5, lag).reindex(tt).to_numpy()
    rows.append(summarize(v, f"day-level lag={lag}"))
    for gap in (5, 10):
        e = declusters(tt, gap, cal)
        ev = trade(5, lag).reindex(e).to_numpy()
        s = summarize(ev, f"EPISODES gap={gap} lag={lag}")
        s["boot_p_le0"] = bootstrap_p_le0(ev)
        rows.append(s)
show(rows, "h=5, trade = SHORT SMH / LONG QQQ (positive = inversion wins)")

# unconditional control for the same trade
for lag in (1,):
    u = trade(5, lag)
    tt = trig_dates(5, lag)
    print(f"\n  CTRL all-days lag={lag}: mean={100*u.mean():+.4f}%  n={u.notna().sum()}")
    insp = u[(cal >= tt[0]) & (cal <= tt[-1])]
    print(f"  CTRL same-span    : mean={100*insp.mean():+.4f}%  n={insp.notna().sum()}")

EPI5 = declusters(trig_dates(5, 1), 5, cal)
EV5 = trade(5, 1).reindex(EPI5).to_numpy()
print(f"\n  episode dates (gap=5, lag=1), N={len(EPI5)}:")
for d, x in zip(EPI5, EV5):
    print(f"    {d.date()}  {100*x:+7.2f}%   SPY {'above' if spy_above.get(d) else 'BELOW'} 200d "
          f"({100*spy_dist.get(d, np.nan):+.1f}%)")

# ---------------------------------------------------------------- 2) era split
print("\n" + "=" * 78)
print("2) ERA SPLIT of the h=5 cell (pre-2018 vs 2018+)")
print("=" * 78)
tt = trig_dates(5, 1)
show(era_split(tt, trade(5, 1).reindex(tt).to_numpy()), "day-level, lag=1")
show(era_split(EPI5, EV5), "episodes gap=5, lag=1")
# and the h=10 cell for cross-reference to c10's claimed era flip
t10 = trig_dates(10, 1)
show(era_split(t10, trade(10, 1).reindex(t10).to_numpy()), "h=10 day-level, lag=1 (cross-ref)")

# --------------------------------------------------- 3) drop best/worst episode
print("\n" + "=" * 78)
print("3) DROP-BEST / DROP-WORST EPISODE (h=5, gap=5, lag=1)")
print("=" * 78)
o = np.argsort(EV5)
rows = [summarize(EV5, "all episodes"),
        summarize(np.delete(EV5, o[-1]), f"drop BEST ({EPI5[o[-1]].date()} {100*EV5[o[-1]]:+.2f}%)"),
        summarize(np.delete(EV5, o[0]), f"drop WORST ({EPI5[o[0]].date()} {100*EV5[o[0]]:+.2f}%)"),
        summarize(np.delete(EV5, [o[-1], o[-2]]), "drop BEST 2"),
        summarize(np.delete(EV5, [o[0], o[1]]), "drop WORST 2")]
for r in rows:
    r["boot_p_le0"] = np.nan
show(rows)
# leave one year out on episodes
loyo = []
for y in sorted(set(EPI5.year)):
    m = EPI5.year != y
    s = summarize(EV5[m], f"drop {y} (n_out={int((~m).sum())})")
    loyo.append(s)
show(loyo, "leave-one-YEAR-out (episodes)")

# ---------------------------------------------------------------- 4) bootstrap
print("\n" + "=" * 78)
print("4) BOOTSTRAP on the TRADE's episode returns (short SMH / long QQQ)")
print("=" * 78)
print(f"  P(mean <= 0) = {bootstrap_p_le0(EV5):.4f}   (episodes gap=5, lag=1, N={len(EV5)})")
e10 = declusters(trig_dates(5, 1), 10, cal)
v10g = trade(5, 1).reindex(e10).to_numpy()
print(f"  P(mean <= 0) = {bootstrap_p_le0(v10g):.4f}   (episodes gap=10, N={len(v10g)})")

# ------------------------------------------------------------ 5) horizon curve
print("\n" + "=" * 78)
print("5) HORIZON CURVE h=1..21 (lag=1), trade = short SMH / long QQQ")
print("=" * 78)
rows = []
for h in range(1, 22):
    tt = trig_dates(h, 1)
    if len(tt) == 0:
        continue
    dv = trade(h, 1).reindex(tt).to_numpy()
    e = declusters(tt, max(h, 5), cal)
    evv = trade(h, 1).reindex(e).to_numpy()
    d = summarize(dv, f"h={h}")
    s = {"h": h, "n_day": d["n"], "day_mean_pct": d["mean_pct"], "day_t": d["t"],
         "n_epi": len(evv)}
    ee = summarize(evv, "")
    s["epi_mean_pct"] = ee["mean_pct"]
    s["epi_t"] = ee["t"]
    s["epi_hit"] = ee["hit"]
    rows.append(s)
show(rows, "horizon curve")

# -------------------------------------------------------------- 6) grid at h=5
print("\n" + "=" * 78)
print("6) THRESHOLD GRID at h=5 (lag=1), episodes gap=5")
print("=" * 78)
grid = []
for a in (5, 10, 15, 20):
    for b in (70, 80, 90):
        m = (r63 <= a) & (r5 >= b)
        ok = trade(5, 1).notna()
        tt = cal[m.fillna(False).values & ok.reindex(cal).fillna(False).values]
        if len(tt) == 0:
            grid.append({"cell": f"r63<={a} r5>={b}", "n_day": 0})
            continue
        e = declusters(tt, 5, cal)
        ev = trade(5, 1).reindex(e).to_numpy()
        dd = summarize(trade(5, 1).reindex(tt).to_numpy(), "")
        ss = summarize(ev, "")
        grid.append({"cell": f"r63<={a} r5>={b}", "n_day": dd["n"], "day_mean_pct": dd["mean_pct"],
                     "day_t": dd["t"], "n_epi": ss["n"], "epi_mean_pct": ss["mean_pct"],
                     "epi_t": ss["t"], "epi_hit": ss["hit"]})
show(grid, "grid (PITCHED CELL = r63<=10 r5>=80)")

# ------------------------------------------------------- 7) leg attribution
print("\n" + "=" * 78)
print("7) LEG ATTRIBUTION + REGIME TEST at h=5 (lag=1)")
print("=" * 78)
tt = trig_dates(5, 1)
e = declusters(tt, 5, cal)
rows = [summarize(fwd_lag(smh, 5).reindex(tt).to_numpy(), "SMH leg, trigger days"),
        summarize(fwd_lag(qqq, 5).reindex(tt).to_numpy(), "QQQ leg, trigger days"),
        summarize(fwd_lag(spy, 5).reindex(tt).to_numpy(), "SPY,     trigger days"),
        summarize(fwd_lag(smh, 5).reindex(cal).to_numpy(), "SMH all-days"),
        summarize(fwd_lag(qqq, 5).reindex(cal).to_numpy(), "QQQ all-days"),
        summarize(fwd_lag(spy, 5).reindex(cal).to_numpy(), "SPY all-days")]
show(rows, "legs, day-level")
rows = [summarize(fwd_lag(smh, 5).reindex(e).to_numpy(), "SMH leg, episodes"),
        summarize(fwd_lag(qqq, 5).reindex(e).to_numpy(), "QQQ leg, episodes"),
        summarize(fwd_lag(spy, 5).reindex(e).to_numpy(), "SPY,     episodes")]
show(rows, "legs, episodes")

ab = spy_above.reindex(tt).fillna(False).to_numpy()
print(f"\n  REGIME: trigger days with SPY ABOVE 200d SMA: {ab.sum()}/{len(ab)} = {100*ab.mean():.1f}%")
print(f"          trigger days with SPY BELOW 200d SMA: {(~ab).sum()}/{len(ab)} = {100*(~ab).mean():.1f}%")
sd = spy_dist.reindex(tt).to_numpy()
print(f"  SPY dist to 200d on trigger days: mean={100*np.nanmean(sd):+.2f}%  "
      f"median={100*np.nanmedian(sd):+.2f}%  p90={100*np.nanpercentile(sd,90):+.2f}%  "
      f"max={100*np.nanmax(sd):+.2f}%")
print(f"  TODAY: {100*spy_dist.iloc[-1]:+.2f}%  -> percentile of trigger-day distribution = "
      f"{100*(sd < spy_dist.iloc[-1]).mean():.1f}%")
# 52w high check
hi52 = spy.rolling(252).max()
at_hi = (spy / hi52 - 1.0) >= -0.005
ah = at_hi.reindex(tt).fillna(False).to_numpy()
print(f"  trigger days with SPY within 0.5% of its 52w high: {ah.sum()}/{len(ah)}")
print(f"  TODAY SPY vs 52w high: {100*(spy.iloc[-1]/hi52.iloc[-1]-1):+.2f}%")

# ------------------------------------------- 8) restrict to SPY above 200d SMA
print("\n" + "=" * 78)
print("8) TODAY'S REGIME ONLY: trigger days with SPY ABOVE its 200d SMA, h=5")
print("=" * 78)
for lbl, extra in (("SPY ABOVE 200d", spy_above), ("SPY BELOW 200d", ~spy_above)):
    tt2 = trig_dates(5, 1, extra=extra)
    if len(tt2) == 0:
        print(f"  {lbl}: N=0")
        continue
    e2 = declusters(tt2, 5, cal)
    v2 = trade(5, 1).reindex(e2).to_numpy()
    d2 = summarize(trade(5, 1).reindex(tt2).to_numpy(), f"{lbl} day-level")
    s2 = summarize(v2, f"{lbl} episodes")
    s2["boot_p_le0"] = bootstrap_p_le0(v2)
    show([d2, s2])
    print(f"    dates: {', '.join(str(d.date()) for d in e2)}")
    print(f"    years: {sorted(set(e2.year))}")

# stricter: SPY above 200d by >5%
strict = spy_dist > 0.05
tt3 = trig_dates(5, 1, extra=strict)
print(f"\n  trigger days with SPY >5% ABOVE 200d (today is +{100*spy_dist.iloc[-1]:.1f}%): N={len(tt3)}")
if len(tt3):
    print(f"    dates: {', '.join(str(d.date()) for d in tt3)}")
    show([summarize(trade(5, 1).reindex(tt3).to_numpy(), "SPY>+5% vs 200d, day-level")])

print("\n5x cost hurdle: SMH ~2bp + QQQ ~1bp per side = ~6 bps round trip -> need >=0.30%")
