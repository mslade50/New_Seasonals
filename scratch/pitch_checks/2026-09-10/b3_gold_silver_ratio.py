"""B3 round 1 -- "Silver leads, gold is washed out".

PRE-SPECIFIED: LONG GLD / SHORT SLV, equal dollar weight, entry lag=1 MOC.
TRIGGER: SLV 5d return - GLD 5d return >= +2.5pp  AND  zscore(GLD close, 10) <= -0.75.

Eighth metals cell examined in three weeks -> the bar is deliberately higher.
Rule 5: legs BEFORE the spread. Rule 7: filter first, then decluster.
CTRL-a is the RATIO's own drift, because the gold/silver ratio trends for years.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change  # private: n-VALID-session return

pd.set_option("display.width", 210)

TICKS = ["SPY", "GLD", "SLV", "GDX", "DX-Y.NYB", "^TNX"]
raw = load_prices(TICKS)
cal = raw["SPY"].index
px = pd.DataFrame({t: raw[t]["Close"].reindex(cal) for t in TICKS})
for t in TICKS:
    v = px[t].dropna()
    print(f"  {t:10s} valid={len(v):5d}  {v.index[0].date()} .. {v.index[-1].date()}")

gld, slv = px["GLD"], px["SLV"]
r5_g = _valid_pct_change(gld, 5)
r5_s = _valid_pct_change(slv, 5)
spread5 = r5_s - r5_g
z10_g = zscore(gld, 10)

print("\n--- LIVE STATE CHECK (2026-09-09 close) ---")
print(f"  GLD 5d {100*r5_g.iloc[-1]:+.2f}% (brief +1.66)   SLV 5d {100*r5_s.iloc[-1]:+.2f}% (brief +4.83)")
print(f"  spread {100*spread5.iloc[-1]:+.2f}pp (brief +3.17)   GLD z10 {z10_g.iloc[-1]:+.2f} (brief -1.07)")

TRIG = ((spread5 >= 0.025) & (z10_g <= -0.75)).fillna(False)
BARE = (spread5 >= 0.025).fillna(False)
print(f"\n  trigger live today = {bool(TRIG.iloc[-1])}")
print(f"  full trigger days N={int(TRIG.sum())} | bare 5d-spread days N={int(BARE.sum())}")
print("  by year:", dict(TRIG[TRIG].groupby(TRIG[TRIG].index.year).size()))

H = 5
PAIR = [("GLD", 1.0), ("SLV", -1.0)]

battery(px, TRIG, PAIR, H, "B3 LONG GLD / SHORT SLV  h=5", cost_bps=3.0,
        variants={
            "spread>=1.5pp & z<=-0.75": ((spread5 >= 0.015) & (z10_g <= -0.75)).fillna(False),
            "spread>=2.0pp & z<=-0.75": ((spread5 >= 0.020) & (z10_g <= -0.75)).fillna(False),
            "spread>=2.5pp & z<=-0.75 (DEFENDED)": TRIG,
            "spread>=3.5pp & z<=-0.75": ((spread5 >= 0.035) & (z10_g <= -0.75)).fillna(False),
            "spread>=2.5pp & z<=-0.25": ((spread5 >= 0.025) & (z10_g <= -0.25)).fillna(False),
            "spread>=2.5pp & z<=-1.25": ((spread5 >= 0.025) & (z10_g <= -1.25)).fillna(False),
            "spread>=2.5pp, NO z gate (BARE)": BARE,
        },
        event_kinds=("cpi",))


def cell(mask, legs, h=H, lbl="", min_gap=None):
    ret = vehicle_ret(px, legs, h, 1)
    valid = ret.notna()
    d = px.index[mask.reindex(px.index, fill_value=False).values & valid.values]
    if len(d) == 0:
        return {"label": lbl, "n": 0}, pd.DatetimeIndex([]), np.array([])
    e = declusters(d, min_gap or h, px.index)
    v = ret.loc[e].values
    s = summarize(v, lbl)
    s["n_days"] = len(d)
    w = int((v > 0).sum())
    s["record"] = f"{w}-{len(v)-w}"
    s["sign_p"] = round(sign_test(w, len(v)), 4)
    return s, e, v


print("\n\n" + "=" * 78)
print("2. PRICE THE LEGS BEFORE THE SPREAD (rule 5)")
print("=" * 78)
rows = []
for legs, lbl in (([("GLD", 1.0)], "LONG GLD alone"),
                  ([("SLV", -1.0)], "SHORT SLV alone"),
                  ([("SLV", 1.0)], "  (LONG SLV, reference)"),
                  (PAIR, "PAIR long GLD / short SLV")):
    s, _, _ = cell(TRIG, legs, lbl=lbl)
    rows.append(s)
for legs, lbl in (([("GLD", 1.0)], "GLD uncond 5d"),
                  ([("SLV", -1.0)], "-SLV uncond 5d"),
                  (PAIR, "PAIR uncond 5d = RATIO DRIFT (CTRL-a)")):
    r = vehicle_ret(px, legs, H, 1).dropna()
    rows.append(summarize(r.values, lbl))
show(rows, "legs vs spread")

print("\n  --- ratio drift by era (is a 1-10td 'mean reversion' claim riding a multi-year trend?) ---")
pr = vehicle_ret(px, PAIR, H, 1).dropna()
byyr = pr.groupby(pr.index.year).agg(["count", "mean"])
byyr["mean"] = (100 * byyr["mean"]).round(3)
print(byyr.to_string())

print("\n\n" + "=" * 78)
print("3. GATE ATTRIBUTION: does the gold-washout gate FILTER? (rule 4)")
print("=" * 78)
rows = []
gates = {
    "BOTH (DEFENDED)": TRIG,
    "BARE spread>=2.5pp, no z gate": BARE,
    "DISCARDED COMPLEMENT: spread>=2.5pp & z > -0.75": (BARE & (z10_g > -0.75)).fillna(False),
    "z<=-0.75 ALONE (no spread gate)": (z10_g <= -0.75).fillna(False),
    "  complement z > -0.75": (z10_g > -0.75).fillna(False),
    "ALL DAYS": pd.Series(True, index=px.index),
}
for lbl, m in gates.items():
    s, _, _ = cell(m, PAIR, lbl=lbl)
    rows.append(s)
show(rows, "gate attribution, pair h=5")

print("\n\n" + "=" * 78)
print("4. RISK-NEUTRALITY: equal DOLLAR is not equal RISK (SLV vol >> GLD vol)")
print("=" * 78)
rg = gld.pct_change()
rs = slv.pct_change()
vg = rg.rolling(63).std()
vs = rs.rolling(63).std()
print(f"  full-sample daily vol: GLD {100*rg.std():.3f}%  SLV {100*rs.std():.3f}%  "
      f"ratio {rs.std()/rg.std():.2f}x")
_, epi, v_eq = cell(TRIG, PAIR)
# vol-scaled: short SLV at (vol_gld/vol_slv) weight so the two legs carry equal risk
w = (vg / vs).reindex(px.index)
retg = fwd_lag(gld, H, 1)
rets = fwd_lag(slv, H, 1)
vs_scaled = retg - w * rets
d = px.index[TRIG.values & vs_scaled.notna().values]
e = declusters(d, H, px.index)
show([summarize(v_eq, f"equal-DOLLAR pair (N={len(v_eq)})"),
      summarize(vs_scaled.loc[e].values, f"vol-SCALED pair, short-SLV wt=volG/volS (N={len(e)})"),
      summarize(vs_scaled.dropna().values, "vol-scaled pair, all days")],
     "equal dollar vs equal risk")
print(f"  mean short-SLV weight on trigger days = {w.loc[e].mean():.3f}")
print("  >>> every number elsewhere in this script refers to the EQUAL-DOLLAR pair.")

print("\n\n" + "=" * 78)
print("5. ERA + CONCENTRATION + EPISODE DATES")
print("=" * 78)
s, epi, v = cell(TRIG, PAIR)
print(f"  episodes N={len(epi)}  mean {100*v.mean():+.3f}%  record {s['record']}  "
      f"sign p {s['sign_p']}  bootstrap P(mean<=0) {bootstrap_p_le0(v):.3f}")
show(era_split(epi, v), "era split 2018")
show(era_split(epi, v, cut="2013-01-01"), "era split 2013")
print("  ", cluster_note(epi, v))
if len(v) >= 3:
    o = np.argsort(-v)
    print(f"  drop-best-1 mean {100*np.delete(v, o[0]).mean():+.3f}% "
          f"(record {(np.delete(v,o[0])>0).sum()}-{(np.delete(v,o[0])<=0).sum()})")
    print(f"  drop-best-2 mean {100*np.delete(v, o[:2]).mean():+.3f}% "
          f"(record {(np.delete(v,o[:2])>0).sum()}-{(np.delete(v,o[:2])<=0).sum()})")
print("\n  episodes:")
for d_, x in zip(epi, v):
    print(f"    {d_.date()}  {100*x:+7.3f}%")

print("\n\n" + "=" * 78)
print("6. HORIZON: is 5d the right window, or is the reversion at 1-2d / 10-21d?")
print("=" * 78)
show(horizon_scan(px, epi, PAIR, hs=(1, 2, 3, 5, 7, 10, 21)), "horizon scan, pair")

print("\n\n" + "=" * 78)
print("7. PERMUTATION (rule 1) against the DEFENDED cell + the grid I walked")
print("=" * 78)
rng = np.random.default_rng(23)
retp = vehicle_ret(px, PAIR, H, 1)
idx = px.index
m0 = TRIG.reindex(idx, fill_value=False).values.astype(bool)
GRID = [((spread5 >= a) & (z10_g <= b)).fillna(False).reindex(idx, fill_value=False).values
        for a in (0.015, 0.020, 0.025, 0.035)
        for b in (-0.25, -0.75, -1.25)]


def stat(mv):
    d_ = idx[mv & retp.notna().values]
    if len(d_) == 0:
        return -np.inf
    return retp.loc[declusters(d_, H, idx)].mean()


obs = stat(m0)
nd, nm = [], []
for _ in range(3000):
    k = int(rng.integers(1, len(idx)))
    nd.append(stat(np.roll(m0, k)))
    nm.append(max(stat(np.roll(g, k)) for g in GRID))
nd, nm = np.array(nd, float), np.array(nm, float)
print(f"  statistic tested = DEFENDED cell episode MEAN {H}d pair return = {100*obs:+.3f}%")
print(f"  uncharged p (defended cell only)                = {(nd >= obs).mean():.4f}")
print(f"  charged p vs max over the 12-cell grid I walked = {(nm >= obs).mean():.4f}")

print("\n\n" + "=" * 78)
print("8. COST")
print("=" * 78)
edge = 100 * 100 * v.mean()
print(f"  GLD ~1bp + SLV ~2bp = 6 bps pair round trip (in+out).")
print(f"  episode mean {edge:.1f} bps -> {edge/6.0:.1f}x cost (floor 5x)")
