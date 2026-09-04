"""C6 -- kill "short UNG through the CPI window".

The only number that matters: the short's EXCESS over UNG's own unconditional
drift. UNG bleeds ~-0.90%/10td structurally, so a naked short looks good on
any anchor whatsoever. If the CPI anchor adds nothing on top of that, it is a
filter that does not filter and the trade is just "be short UNG", which needs
no morning and no print.

Then: UNG is 5.3% off its 52w LOW after +4.11% in one session. A short at a
52w low into a squeeze is the tail that kills it -- quantify the worst episode
and the near-52w-low conditional. Then borrow cost, honestly.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (close_panel, load_events, fwd_lag, declusters, summarize,  # noqa: E402
                       sign_test, bootstrap_p_le0, cluster_note, local_control)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)

px = close_panel(["UNG", "USO", "DBC", "SPY"])
idx = px.index
u = px["UNG"].dropna()
uidx = u.index
tdom = pd.Series(pd.Series(uidx, index=uidx).groupby([uidx.year, uidx.month]).cumcount().values + 1,
                 index=uidx)


def anchors_for(kind: str, offset: int = 2) -> pd.DatetimeIndex:
    ev = load_events([kind])["date"]
    out = []
    for d in ev:
        p = uidx.searchsorted(d, side="left") - offset
        if 0 <= p < len(uidx):
            out.append(uidx[p])
    return pd.DatetimeIndex(sorted(set(out)))


cpi_a = anchors_for("cpi", 2)
cpi_a = cpi_a.intersection(uidx)
print(f"UNG history {uidx[0].date()} .. {uidx[-1].date()}  ({len(uidx)} bars)")
print(f"CPI anchors inside UNG history: {len(cpi_a)}")

# ---------------------------------------------------------------------------
# 1. THE number: short excess over UNG's own drift, every horizon
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("1. SHORT UNG on the CPI anchor vs SHORT UNG ALWAYS (the only number)")
print("=" * 100)
rows = []
for h in (1, 2, 3, 5, 7, 10):
    f = fwd_lag(u, h, lag=1)
    short_all = -f.dropna()
    v = -f.reindex(cpi_a).dropna()
    # tdom-matched control
    want = set(tdom.reindex(v.index).dropna().astype(int))
    ctl_idx = tdom[tdom.isin(want)].index.intersection(short_all.index).difference(v.index)
    ctl = short_all.reindex(ctl_idx).dropna()
    st = summarize(v.values)
    stc = summarize(short_all.values)
    # welch on cond vs all-days
    se = np.sqrt(v.values.var(ddof=1) / len(v) + short_all.values.var(ddof=1) / len(short_all))
    rows.append({"h": h, "n": st["n"],
                 "cpi_short_pct": round(st["mean_pct"], 3),
                 "always_short_pct": round(stc["mean_pct"], 3),
                 "excess": round(st["mean_pct"] - stc["mean_pct"], 3),
                 "excess_tdom": round(st["mean_pct"] - 100 * ctl.mean(), 3),
                 "welch_t": round((v.values.mean() - short_all.values.mean()) / se, 2),
                 "hit": round(st["hit"], 1),
                 "hit_always": round(stc["hit"], 1),
                 "signp": round(sign_test(int((v.values > 0).sum()), len(v)), 4),
                 "worst": round(st["worst_pct"], 2)})
print(pd.DataFrame(rows).to_string(index=False))
print("\nREAD: 'always_short_pct' is what a naked, always-on UNG short earns over the")
print("same horizon with NO anchor at all. 'excess' is everything the CPI print adds.")

# declustered episodes at h=5 (the pitched horizon)
H = 5
f5 = fwd_lag(u, H, lag=1)
epi = declusters(cpi_a, 5, uidx)
v5 = -f5.reindex(epi).dropna()
epi = v5.index
short_all5 = -f5.dropna()
print(f"\ndeclustered (5td) h=5 episodes: n={len(v5)}")
st = summarize(v5.values)
print(f"  short mean {st['mean_pct']:+.3f}%  always-short {100*short_all5.mean():+.3f}%  "
      f"excess {st['mean_pct']-100*short_all5.mean():+.3f}%")
print(f"  hit {st['hit']:.1f}  t {st['t']:+.2f}  signp {sign_test(int((v5.values>0).sum()), len(v5)):.4f}  "
      f"bootstrap P(mean<=0) {bootstrap_p_le0(v5.values):.3f}")
print(f"  worst {st['worst_pct']:+.2f}%  best {st['best_pct']:+.2f}%  sd {st['sd_pct']:.2f}")
print(f"  concentration: {cluster_note(epi, v5.values)}")

# ---------------------------------------------------------------------------
# 2. era split -- the structural bleed is not constant
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("2. ERA SPLIT -- the bleed is the whole trade, so is the bleed stable?")
print("=" * 100)
for lo, hi in ((2007, 2012), (2013, 2017), (2018, 2021), (2022, 2026)):
    sel = (epi.year >= lo) & (epi.year <= hi)
    vv = v5.values[sel]
    if len(vv) < 3:
        continue
    aa = short_all5[(short_all5.index.year >= lo) & (short_all5.index.year <= hi)]
    st = summarize(vv)
    print(f"  {lo}-{hi}: n={st['n']:<3} cpi-short {st['mean_pct']:+.3f}% "
          f"always-short {100*aa.mean():+.3f}%  excess {st['mean_pct']-100*aa.mean():+.3f}%  "
          f"hit {st['hit']:.1f}  signp {sign_test(int((vv>0).sum()), len(vv)):.4f}")

print("\n  UNG buy-and-hold by era (what the bleed actually was):")
for lo, hi in ((2007, 2012), (2013, 2017), (2018, 2021), (2022, 2026)):
    seg = u[(u.index.year >= lo) & (u.index.year <= hi)]
    if len(seg) < 2:
        continue
    yrs = (seg.index[-1] - seg.index[0]).days / 365.25
    print(f"    {lo}-{hi}: {100*(seg.iloc[-1]/seg.iloc[0]-1):+8.1f}% total, "
          f"{100*((seg.iloc[-1]/seg.iloc[0])**(1/yrs)-1):+7.1f}%/yr")

# ---------------------------------------------------------------------------
# 3. the squeeze tail -- short at a 52w low
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("3. SQUEEZE TAIL -- UNG is 5.3% off its 52w LOW today after +4.11% in a session")
print("=" * 100)
lo52 = u / u.rolling(252).min() - 1.0
hi52 = u / u.rolling(252).max() - 1.0
near_low = lo52 <= 0.10          # within 10% of the 52w low (today: +5.3%)
one_day = u.pct_change()

print(f"today: {100*lo52.iloc[-1]:.1f}% above the 52w low, "
      f"{100*hi52.iloc[-1]:.1f}% from the 52w high, 1d {100*one_day.iloc[-1]:+.2f}%")

print("\n(a) ALL days, short UNG h=5, split by distance above the 52w low:")
for lbl, sel in (("within 10% of 52w low  <-- TODAY", near_low.reindex(short_all5.index).fillna(False).values),
                 ("more than 10% above", ~near_low.reindex(short_all5.index).fillna(False).values)):
    vv = short_all5.values[sel]
    st = summarize(vv)
    print(f"  {lbl:<36} n={st['n']:<5} short mean {st['mean_pct']:+.3f}% "
          f"hit {st['hit']:.1f} t {st['t']:+.2f} worst {st['worst_pct']:+.2f}% "
          f"p5 {100*np.percentile(vv, 5):+.2f}%")

print("\n(b) CPI anchor AND within 10% of the 52w low (today's exact cell):")
sel = near_low.reindex(epi).fillna(False).values
vv = v5.values[sel]
if len(vv) >= 3:
    st = summarize(vv)
    print(f"  n={st['n']:<3} short mean {st['mean_pct']:+.3f}% hit {st['hit']:.1f} "
          f"signp {sign_test(int((vv>0).sum()), len(vv)):.4f} worst {st['worst_pct']:+.2f}%")
    aa = short_all5[near_low.reindex(short_all5.index).fillna(False).values]
    print(f"  always-short in the same state {100*aa.mean():+.3f}% -> "
          f"CPI excess {st['mean_pct']-100*aa.mean():+.3f}%")
    print(f"  dates: {[str(d.date()) for d in epi[sel]]}")
else:
    print(f"  n={len(vv)} -- too few")

print("\n(c) worst 5-day SHORT losses in UNG history (the squeeze tail):")
worst = short_all5.nsmallest(12)
print(pd.Series((100 * worst.values).round(2), index=[d.date() for d in worst.index]).to_string())
print(f"\n  worst 5d short loss ever: {100*short_all5.min():.2f}%")
print(f"  worst 5d short loss on a CPI anchor: {100*v5.min():.2f}% on {epi[int(np.argmin(v5.values))].date()}")
print(f"  P(short loses > 5% in 5d), all days: {100*float((short_all5 < -0.05).mean()):.2f}%")
print(f"  P(short loses > 10% in 5d), all days: {100*float((short_all5 < -0.10).mean()):.2f}%")
print(f"  ratio of worst loss to mean gain: {abs(100*short_all5.min())/ (100*v5.mean()):.0f}x the edge")

# also: after a >+4% one-day pop, is a short better or worse?
print("\n(d) short UNG h=5 AFTER a >= +4% one-day pop (today printed +4.11%):")
pop = (one_day >= 0.04).reindex(short_all5.index).fillna(False)
vv = short_all5.values[pop.values]
st = summarize(vv)
print(f"  n={st['n']:<4} short mean {st['mean_pct']:+.3f}% (vs always {100*short_all5.mean():+.3f}%) "
      f"hit {st['hit']:.1f} t {st['t']:+.2f} worst {st['worst_pct']:+.2f}%")

# ---------------------------------------------------------------------------
# 4. cost
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("4. COST -- what the short has to pay before it keeps anything")
print("=" * 100)
edge_bps = 100 * (st_e := summarize(v5.values))["mean_pct"]
excess_bps = 100 * (st_e["mean_pct"] - 100 * short_all5.mean())
print(f"  gross short edge h=5 = {edge_bps:.1f} bps; ANCHOR-ATTRIBUTABLE excess = {excess_bps:.1f} bps")
print(f"  UNG round trip (spread + commission), 1 leg ~ 6-10 bps")
print(f"  borrow: UNG is a commodity-pool ETF; borrow is usually available but the fee is")
print(f"    variable and has run hard-to-borrow during natgas squeezes -- the exact state")
print(f"    where the short is most at risk. At a 5td hold even 5%/yr borrow = ~10 bps.")
print(f"  --> anchor excess {excess_bps:.1f} bps vs ~16-20 bps of round trip + borrow")
