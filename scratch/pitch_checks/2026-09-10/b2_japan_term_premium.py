"""B2 round 1 -- "Japan is the one market a term-premium repricing helps".

PRE-SPECIFIED: LONG EWJ / SHORT EFA, equal dollar weight, entry lag=1 MOC.
TRIGGER: ^TNX closes at a trailing-252d HIGH.

Order convention (rule 7): FILTER first, THEN decluster.
Calendar (rule 8): everything reindexed to SPY's NYSE calendar; ^TNX carries
bars the equity complex lacks.
Rule 5: legs priced BEFORE the spread. Rule 6: pre-declared reference class.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

pd.set_option("display.width", 210)

# reference class PRE-DECLARED before any result is seen: every country/region
# equity ETF in the cache with >= 10y history, plus SPY as the US member.
CLASS = ["EWJ", "EWT", "EWW", "EWY", "EWZ", "FXI", "VGK", "INDA", "KWEB",
         "EEM", "SPY"]
TICKS = ["SPY", "EFA", "^TNX", "DX-Y.NYB", "JPY=X", "EURUSD=X", "^N225"] + CLASS
TICKS = list(dict.fromkeys(TICKS))
raw = load_prices(TICKS)
cal = raw["SPY"].index
px = pd.DataFrame({t: raw[t]["Close"].reindex(cal) for t in TICKS})
for t in TICKS:
    v = px[t].dropna()
    print(f"  {t:10s} valid={len(v):5d}  {v.index[0].date()} .. {v.index[-1].date()}")

tnx = px["^TNX"]


def at_high(s, n):
    hi = rolling_on_valid(s, lambda x: x.rolling(n).max())
    return (s >= hi - 1e-9) & s.notna() & hi.notna()


TRIG = at_high(tnx, 252).fillna(False)
H = 5
print(f"\n^TNX at 252d high: {int(TRIG.sum())} days; live today = {bool(TRIG.iloc[-1])}")
print("  by year:", dict(TRIG[TRIG].groupby(TRIG[TRIG].index.year).size()))

PAIR = [("EWJ", 1.0), ("EFA", -1.0)]

# ------------------------------------------------------- 1. battery on pair
battery(px, TRIG, PAIR, H, "B2 LONG EWJ / SHORT EFA  h=5", cost_bps=4.0,
        variants={
            "TNX 63d high": at_high(tnx, 63).fillna(False),
            "TNX 126d high": at_high(tnx, 126).fillna(False),
            "TNX 252d high (DEFENDED)": TRIG,
            "TNX 252d high & new by 5bp": (TRIG & (tnx >= rolling_on_valid(
                tnx.shift(1), lambda x: x.rolling(252).max()) + 0.05)).fillna(False),
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
for legs, lbl in (([("EWJ", 1.0)], "LONG EWJ alone"),
                  ([("EFA", -1.0)], "SHORT EFA alone"),
                  ([("EFA", 1.0)], "  (LONG EFA, for reference)"),
                  (PAIR, "PAIR long EWJ / short EFA")):
    s, _, _ = cell(TRIG, legs, lbl=lbl)
    rows.append(s)
# unconditional drift of each leg over the same span
for legs, lbl in (([("EWJ", 1.0)], "EWJ uncond drift 5d"),
                  ([("EFA", -1.0)], "-EFA uncond drift 5d"),
                  (PAIR, "PAIR uncond drift 5d")):
    r = vehicle_ret(px, legs, H, 1).dropna()
    rows.append(summarize(r.values, lbl))
show(rows, "legs vs spread, TNX 252d high")

print("\n\n" + "=" * 78)
print("3. THE CURRENCY TEST (the mechanism test)")
print("=" * 78)
_, epi, v = cell(TRIG, PAIR)
print(f"  pair episodes N={len(epi)}, mean {100*v.mean():+.3f}%")
# contemporaneous FX move over the SAME entry->exit span
for fx, lbl, sgn in (("DX-Y.NYB", "DXY", 1.0), ("JPY=X", "USDJPY", 1.0),
                     ("EURUSD=X", "EURUSD", 1.0)):
    fr = vehicle_ret(px, [(fx, sgn)], H, 1)
    ok = fr.loc[epi].notna().values
    if ok.sum() < 5:
        print(f"  {lbl}: too few overlapping episodes ({int(ok.sum())})")
        continue
    x = fr.loc[epi].values[ok]
    y = v[ok]
    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x)
    r2 = 1 - resid.var() / y.var() if y.var() > 0 else np.nan
    tt = (a) / (resid.std(ddof=1) / np.sqrt(len(y))) if len(y) > 2 else np.nan
    print(f"  regress pair on {lbl:7s}: beta={b:+.3f}  R2={r2:.3f}  "
          f"ALPHA(intercept)={100*a:+.3f}%  resid-t on alpha={tt:+.2f}  N={len(y)}")

# yen-hedged proxy: does the pair still work after removing the USDJPY move?
print("\n  --- yen-hedged proxy: EWJ return minus USDJPY move (approx local JPY equity) ---")
jr = vehicle_ret(px, [("JPY=X", 1.0)], H, 1)     # USDJPY up = yen weaker
ewj = vehicle_ret(px, [("EWJ", 1.0)], H, 1)
efa = vehicle_ret(px, [("EFA", -1.0)], H, 1)
hedged = ewj + jr                                # add back yen depreciation
hp = (hedged + efa)
d = px.index[TRIG.values & hp.notna().values]
e = declusters(d, H, px.index)
show([summarize(hp.loc[e].values, f"yen-HEDGED pair episodes (N={len(e)})"),
      summarize(hp.dropna().values, "yen-hedged pair, all days"),
      summarize((ewj + jr).loc[e].values, "yen-hedged EWJ leg alone"),
      summarize(v, f"unhedged pair (N={len(v)})")], "currency decomposition")

print("\n  --- ^N225 (local-currency Japan) vs EFA-in-USD is NOT clean; use N225 vs VGK+FX ---")
n2 = vehicle_ret(px, [("^N225", 1.0)], H, 1)
d2 = px.index[TRIG.values & n2.notna().values]
e2 = declusters(d2, H, px.index)
show([summarize(n2.loc[e2].values, f"^N225 local ccy alone, TNX high (N={len(e2)})"),
      summarize(n2.dropna().values, "^N225 all days")], "Japan in its own currency")

print("\n\n" + "=" * 78)
print("4. REFERENCE CLASS (rule 6): LONG(country) / SHORT EFA on the same trigger")
print("   class PRE-DECLARED above, K =", len(CLASS))
print("=" * 78)
rows, stats = [], {}
for c in CLASS:
    s, e_, v_ = cell(TRIG, [(c, 1.0), ("EFA", -1.0)], lbl=f"{c} vs EFA")
    rows.append(s)
    if s.get("n"):
        stats[c] = s["mean_pct"]
show(rows, "reference class, episode level h=5")
if stats:
    order = sorted(stats.items(), key=lambda kv: -kv[1])
    rank = [k for k, _ in order].index("EWJ") + 1
    print(f"\n  EWJ rank in the class: {rank} of {len(order)}   "
          f"ordering: {[f'{k} {v:+.2f}' for k, v in order]}")

print("\n  max-of-K permutation P, tested against EWJ-vs-EFA's OWN episode mean")
rng = np.random.default_rng(11)
retmap = {c: vehicle_ret(px, [(c, 1.0), ("EFA", -1.0)], H, 1) for c in CLASS}
m0 = TRIG.reindex(px.index, fill_value=False).values.astype(bool)
idx = px.index


def stat_for(c, mv):
    r = retmap[c]
    d = idx[mv & r.notna().values]
    if len(d) == 0:
        return -np.inf
    return r.loc[declusters(d, H, idx)].mean()


obs_ewj = stat_for("EWJ", m0)
nullmax = []
for _ in range(2000):
    k = int(rng.integers(1, len(idx)))
    mv = np.roll(m0, k)
    nullmax.append(max(stat_for(c, mv) for c in CLASS))
nullmax = np.array(nullmax, float)
own = np.array([stat_for("EWJ", np.roll(m0, int(rng.integers(1, len(idx)))))
                for _ in range(2000)], float)
print(f"    statistic tested = EWJ-vs-EFA episode mean = {100*obs_ewj:+.3f}%")
print(f"    uncharged rotation p (EWJ cell only)        = {(own >= obs_ewj).mean():.4f}")
print(f"    charged max-of-K={len(CLASS)} rotation p    = {(nullmax >= obs_ewj).mean():.4f}")

print("\n\n" + "=" * 78)
print("5. GATE ATTRIBUTION + DISCARDED COMPLEMENT (rule 4)")
print("=" * 78)
rows = []
for lbl, m in (("TNX 252d high (DEFENDED)", TRIG),
               ("NO yield condition (all days)", pd.Series(True, index=px.index)),
               ("complement: NOT at 252d high", (~TRIG) & tnx.notna())):
    s, _, _ = cell(m.fillna(False) if m.dtype == bool else m, PAIR, lbl=lbl)
    rows.append(s)
show(rows, "gate attribution, pair")

print("\n\n" + "=" * 78)
print("6. ERA + CONCENTRATION + COST")
print("=" * 78)
s, epi, v = cell(TRIG, PAIR)
print(f"  episodes N={len(epi)}  mean {100*v.mean():+.3f}%  record {s['record']}  "
      f"sign p {s['sign_p']}  bootstrap P(mean<=0) {bootstrap_p_le0(v):.3f}")
show(era_split(epi, v), "era split 2018")
show(era_split(epi, v, cut="2013-01-01"), "era split 2013 (Abenomics start)")
show(era_split(epi, v, cut="2021-01-01"), "era split 2021")
print("  ", cluster_note(epi, v))
if len(v) >= 3:
    o = np.argsort(-v)
    print(f"  drop-best-1 mean {100*np.delete(v, o[0]).mean():+.3f}%  "
          f"drop-best-2 mean {100*np.delete(v, o[:2]).mean():+.3f}%")
by_yr = pd.Series(v, index=[d.year for d in epi]).groupby(level=0).agg(["count", "mean"])
by_yr["mean"] = (100 * by_yr["mean"]).round(3)
print("\n  by year:\n", by_yr.to_string())
edge = 100 * 100 * v.mean()
print(f"\n  cost: 2 legs x ~4 bps = 8 bps round trip; edge {edge:.1f} bps -> "
      f"{edge/8.0:.1f}x cost (floor 5x)")
