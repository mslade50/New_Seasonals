"""C6 round 1 -- Long the defense complex on a coordinated five-name z10
washout with the index near its 52-week high.

Live 2026-09-01 z10: ITA -2.55, RTX -1.97, GD -1.88, LMT -1.70, NOC -1.65
(four of five at -1.5 or worse); SPY -2.07% off its 52-week high.
8 declustered episodes over 5 years, THREE of them 2026.

The registry closed "the whole defensive complex washed out while the index
sits at a 52-week high" on 2026-08-28 as a post-presidential-election rotation
wearing a sector-breadth label (62.5% of 16 trigger days within 60 calendar
days of a presidential election against a 9.1% base). That was XLP/XLU/XLV;
this is aerospace/defense, a different basket, so the election test is
re-run here rather than inherited.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pitch_lab import *  # noqa
from _rc import cochran, dial_series, per_name, perm_max_of_n, welch  # noqa

pd.set_option("display.width", 250)

DEF5 = ["ITA", "RTX", "GD", "LMT", "NOC"]
DEF_EQ = ["RTX", "GD", "LMT", "NOC", "LHX"]      # 5 singles, all back to 2000
COMPLEXES = {
    "defense    ": ["RTX", "GD", "LMT", "NOC", "LHX"],
    "big banks  ": ["JPM", "BAC", "WFC", "C", "GS"],
    "semis      ": ["INTC", "MU", "AMD", "TXN", "QCOM"],
    "oil majors ": ["XOM", "CVX", "COP", "SLB", "OXY"],
    "staples    ": ["PG", "KO", "PEP", "CL", "WMT"],
    "pharma     ": ["JNJ", "PFE", "MRK", "ABT", "BMY"],
    "utilities  ": ["DUK", "SO", "D", "NEE", "AEP"],
    "rails/machy": ["UNP", "CSX", "NSC", "CAT", "DE"],
    "multi-inds ": ["HON", "GE", "MMM", "EMR", "ITW"],
    "semicap    ": ["AMAT", "KLAC", "LRCX", "ADI", "TSM"],
}
ALL = sorted({t for v in COMPLEXES.values() for t in v} | set(DEF5) | {"SPY"})
px = close_panel(ALL)
spy_hi = rolling_on_valid(px["SPY"], lambda x: x.rolling(252).max())
SPY_DD = px["SPY"] / spy_hi - 1.0
SPY_NEAR = SPY_DD >= -0.03

Z = {t: zscore(px[t], 10) for t in ALL}
print("LIVE z10 2026-09-01: " + "  ".join(f"{t} {Z[t].iloc[-1]:+.2f}" for t in DEF5))
print("LIVE z10 (5 singles): " + "  ".join(f"{t} {Z[t].iloc[-1]:+.2f}" for t in DEF_EQ))
print(f"SPY off its 52w high {100*SPY_DD.iloc[-1]:+.2f}%")


def washout(names, k=4, thr=-1.5):
    cnt = sum((Z[t] <= thr).astype(float).where(Z[t].notna(), np.nan)
              for t in names)
    return (cnt >= k).fillna(False)


BARE = washout(DEF5)
MAIN = (BARE & SPY_NEAR).fillna(False)
BARE_EQ = washout(DEF_EQ)
MAIN_EQ = (BARE_EQ & SPY_NEAR).fillna(False)
print("MAIN fires today (ITA form):", bool(MAIN.iloc[-1]),
      "| 5-singles form:", bool(MAIN_EQ.iloc[-1]))

# vehicles
EQW = [(t, 0.2) for t in DEF_EQ]
ITA = [("ITA", 1.0)]

print("\n########## 0. HORIZON SCAN ##########")
for lbl, legs, m in [("ITA vehicle", ITA, MAIN), ("EW 5-single basket", EQW, MAIN)]:
    show(horizon_scan(px, px.index[m.values], legs, hs=(1, 2, 3, 5, 7, 10)),
         f"{lbl}, episode level, lag=1")
print("  NOTE: 6-point horizon grid x 2 vehicles walked. x12 multiplicity on "
      "any best-cell claim.")

H, GAP = 5, 5

battery(px, MAIN, ITA, H, "C6 long ITA on a 4-of-5 defense z10 washout x SPY near high",
        cost_bps=12.0, min_gap=GAP,
        variants={
            "5-of-5 at z10<=-1.5": (washout(DEF5, 5) & SPY_NEAR).fillna(False),
            "3-of-5 at z10<=-1.5": (washout(DEF5, 3) & SPY_NEAR).fillna(False),
            "4-of-5 at z10<=-2.0": (washout(DEF5, 4, -2.0) & SPY_NEAR).fillna(False),
            "4-of-5 at z10<=-1.0": (washout(DEF5, 4, -1.0) & SPY_NEAR).fillna(False),
            "SPY within 5% of high": (BARE & (SPY_DD >= -0.05)).fillna(False),
            "5-singles complex (no ITA)": MAIN_EQ,
        }, event_kinds=("nfp",))

ret = vehicle_ret(px, ITA, H)
valid = ret.dropna().index
epi = declusters(px.index[MAIN.values].intersection(valid), GAP, valid)
vals = ret.loc[epi].values

# ------------------------------------------------------- 1. CONCENTRATION
print("\n########## 1. CONCENTRATION (the headline risk) ##########")
print("  " + cluster_note(epi, vals, k=2))
yrs = pd.DatetimeIndex(epi).year
for y in sorted(set(yrs)):
    m = yrs == y
    print(f"   {y}: N={int(m.sum())} mean {100*vals[m].mean():+.3f}% "
          f"total {100*vals[m].sum():+.3f}pp")
m26 = yrs == 2026
print(f"  2026 share of episodes: {int(m26.sum())}/{len(vals)} = "
      f"{100*m26.mean():.0f}%; 2026 share of TOTAL return "
      f"{100*vals[m26].sum()/vals.sum() if vals.sum() else float('nan'):.0f}%")
if (~m26).sum() >= 2:
    w = int((vals[~m26] > 0).sum())
    print(f"  EX-2026: N={int((~m26).sum())} mean {100*vals[~m26].mean():+.3f}% "
          f"record {w}-{int((~m26).sum())-w} sign p "
          f"{sign_test(w, int((~m26).sum())):.4f}")
    print(f"           vs ITA all-days drift {100*ret.loc[valid].mean():+.3f}% "
          f"-> edge {100*(vals[~m26].mean()-ret.loc[valid].mean()):+.3f}pp")
top2 = np.sort(np.abs(vals))[-2:].sum()
print(f"  top-2 |episode| = {100*top2:.2f}pp against a total of "
      f"{100*vals.sum():.2f}pp")

# ------------------------------------------------------ 2. GATE ATTRIBUTION
print("\n########## 2. GATE ATTRIBUTION ##########")
rows = []
for lbl, m in [("FULL: 4of5 washout & SPY within 3%", MAIN),
               ("BARE 4-of-5 washout (drop SPY leg)", BARE),
               ("washout & SPY MORE than 3% off (complement)", (BARE & ~SPY_NEAR)),
               ("SPY near-high alone", SPY_NEAR)]:
    t = px.index[m.fillna(False).values].intersection(valid)
    if len(t) == 0:
        rows.append({"label": lbl, "n": 0})
        continue
    e = declusters(t, GAP, valid)
    r = summarize(ret.loc[e].values, lbl)
    r["n_days"] = len(t)
    rows.append(r)
show(rows, f"gate attribution, ITA, h={H}")
print(f"  SPY-leg dose = {rows[0]['mean_pct'] - rows[1]['mean_pct']:+.3f} pp "
      f"(full {rows[0]['mean_pct']:+.3f} on {rows[0]['n']} vs bare "
      f"{rows[1]['mean_pct']:+.3f} on {rows[1]['n']})")
sma200 = rolling_on_valid(px["SPY"], lambda x: x.rolling(200).mean())
above = (px["SPY"] > sma200).dropna()
td = px.index[MAIN.values].intersection(above.index)
print(f"  bull-tape selector: {100*above.loc[td].mean():.1f}% of {len(td)} trigger "
      f"days above SPY's 200d, base {100*above.mean():.1f}%")

# ------------------------------------------------------------ 3. BETA / ALPHA
print("\n########## 3. IS IT JUST BETA? regress basket fwd on SPY fwd ##########")
for lbl, legs in [("ITA", ITA), ("EW 5-single basket", EQW)]:
    y = vehicle_ret(px, legs, H)
    x = vehicle_ret(px, [("SPY", 1.0)], H)
    both = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    b, a = np.polyfit(both["x"], both["y"], 1)
    e = pd.DatetimeIndex(epi).intersection(both.index)
    resid = both.loc[e, "y"] - (a + b * both.loc[e, "x"])
    print(f"  {lbl}: full-sample beta on SPY(h={H}) = {b:.3f}, intercept "
          f"{100*a:+.3f}%")
    print(f"     episode raw mean {100*both.loc[e,'y'].mean():+.3f}%, SPY over the "
          f"same windows {100*both.loc[e,'x'].mean():+.3f}%")
    print(f"     ALPHA (beta-adjusted residual) = {100*resid.mean():+.3f}% "
          f"on N={len(resid)}, t = "
          f"{resid.mean()/(resid.std(ddof=1)/np.sqrt(len(resid))):+.2f}, "
          f"record {int((resid>0).sum())}-{int((resid<=0).sum())}")

# --------------------------------------------------- 4. VEHICLE COMPARISON
print("\n########## 4. VEHICLES as WHOLE variants ##########")
rows = []
for lbl, legs, cost in [("ITA (1 leg, ~12 bps rt)", ITA, 12.0),
                        ("EW basket RTX/GD/LMT/NOC/LHX (5 legs, ~5 bps each)", EQW, 25.0),
                        ("EW basket ex-LHX + ITA", [(t, 0.2) for t in DEF5], 30.0)]:
    r_ = vehicle_ret(px, legs, H)
    v_ = r_.dropna().index
    e_ = declusters(px.index[MAIN.values].intersection(v_), GAP, v_)
    r = summarize(r_.loc[e_].values, lbl)
    if r["n"]:
        r["drift_pct"] = round(100 * r_.loc[v_].mean(), 3)
        r["edge_pct"] = round(r["mean_pct"] - 100 * r_.loc[v_].mean(), 3)
        r["x_cost"] = round(100 * r["mean_pct"] / cost, 1)
    rows.append(r)
show(rows, f"vehicle comparison, h={H}")

# ------------------------------------------ 5. REFERENCE CLASS OF COMPLEXES
print("\n########## 5. REFERENCE CLASS -- the same rule on 10 complexes ##########")
rows, stats = [], []
for name, names in COMPLEXES.items():
    m = (washout(names) & SPY_NEAR).fillna(False)
    legs = [(t, 0.2) for t in names]
    r_ = vehicle_ret(px, legs, H)
    v_ = r_.dropna().index
    t_ = px.index[m.values].intersection(v_)
    if len(t_) < 3:
        rows.append({"complex": name, "n_days": len(t_), "n_epi": 0})
        continue
    e_ = declusters(t_, GAP, v_)
    v = r_.loc[e_].values
    v = v[~np.isnan(v)]
    span = (v_ >= t_[0]) & (v_ <= t_[-1])
    ctrl = r_.loc[v_[span]].dropna().values
    exc = v.mean() - ctrl.mean()
    se_d = np.sqrt(v.var(ddof=1) / len(v) + ctrl.var(ddof=1) / len(ctrl))
    rows.append({"complex": name, "n_days": len(t_), "n_epi": len(v),
                 "mean_pct": 100 * v.mean(), "hit": 100 * (v > 0).mean(),
                 "drift_pct": 100 * ctrl.mean(), "excess_pct": 100 * exc,
                 "t_excess": exc / se_d, "se_d_pct": 100 * se_d,
                 "worst_pct": 100 * v.min()})
    stats.append((name, v, exc, se_d))
df = pd.DataFrame(rows).sort_values("t_excess", ascending=False)
show(df.to_dict("records"), f"equal-weight basket per complex, h={H}")
co = cochran(df)
if co:
    print(f"  Cochran Q = {co['Q']:.2f} on {co['df']} df, p = {co['p']:.4f}, "
          f"I-squared = {co['I2_pct']:.1f}%")
    print(f"  fixed-effect COMMON excess = {co['fe_common_pct']:+.3f} pp "
          f"(se {co['fe_se_pct']:.3f}, t {co['fe_t']:+.2f})")
ok = df.dropna(subset=["t_excess"])
names_ranked = list(ok["complex"])
if any("defense" in n for n in names_ranked):
    i = [j for j, n in enumerate(names_ranked) if "defense" in n][0]
    print(f"  DEFENSE ranks {i+1} of {len(names_ranked)} by excess-t "
          f"(leader {names_ranked[0].strip()})")

# permutation max-of-10 with a common circular offset
rng = np.random.default_rng(7)
prep = []
for name, names in COMPLEXES.items():
    m = (washout(names) & SPY_NEAR).fillna(False)
    legs = [(t, 0.2) for t in names]
    r_ = vehicle_ret(px, legs, H)
    v_ = r_.dropna().index
    t_ = px.index[m.values].intersection(v_)
    if len(t_) < 3:
        continue
    e_ = declusters(t_, GAP, v_)
    pos = pd.Series(range(len(v_)), index=v_)
    ip = np.array([pos[d] for d in e_ if d in pos.index])
    prep.append((name, r_.loc[v_].values, ip))
obs = {}
for name, rv, ip in prep:
    v = rv[ip]
    obs[name] = (v.mean() - rv.mean(),
                 (v.mean() - rv.mean()) / (v.std(ddof=1) / np.sqrt(len(v))))
maxlen = min(len(rv) for _, rv, _ in prep)
nullE, nullT = [], []
for _ in range(2000):
    off = int(rng.integers(0, maxlen))
    me, mt = -1e9, -1e9
    for name, rv, ip in prep:
        v = rv[(ip + off) % len(rv)]
        sd = v.std(ddof=1)
        e_ = v.mean() - rv.mean()
        me = max(me, e_)
        mt = max(mt, abs(e_ / (sd / np.sqrt(len(v)))) if sd > 0 else 0.0)
    nullE.append(me)
    nullT.append(mt)
nullE, nullT = np.array(nullE), np.array(nullT)
dexc, dt = obs.get("defense    ", (np.nan, np.nan))
best = max(obs.items(), key=lambda kv: kv[1][0])
print(f"  permutation max-of-{len(prep)} (common circular offset, 2000 draws):")
print(f"    observed best excess {100*best[1][0]:+.3f}pp ({best[0].strip()}); "
      f"defense excess {100*dexc:+.3f}pp, |t| {abs(dt):.2f}")
print(f"    P(max excess >= best) = {(nullE >= best[1][0]).mean():.4f}   "
      f"P(max excess >= defense's) = {(nullE >= dexc).mean():.4f}   "
      f"P(max|t| >= defense's) = {(nullT >= abs(dt)).mean():.4f}")

# ----------------------------------------------- 6. ELECTION / DIAL / EVENTS
print("\n########## 6. ELECTION PROXIMITY (the 2026-08-28 registry shape) ##########")
elec = load_events(["election"])["date"] if "election" in set(
    load_events()["event"]) else pd.Series(dtype="datetime64[ns]")
pres = pd.DatetimeIndex([f"{y}-11-05" for y in range(2000, 2027, 4)])
tdays = px.index[MAIN.values]
allv = valid


def within(d, cal, days):
    return bool((np.abs((cal - d).days) <= days).any())


hit = np.array([within(d, pres, 60) for d in tdays])
base = np.array([within(d, pres, 60) for d in allv])
print(f"  trigger days within 60 calendar days of a presidential election: "
      f"{100*hit.mean():.1f}% of {len(tdays)}, base {100*base.mean():.1f}%")
ehit = np.array([within(d, pres, 60) for d in epi])
print(f"  episodes within 60d of a presidential election: {int(ehit.sum())} of "
      f"{len(epi)}: {[str(d.date()) for d, f in zip(epi, ehit) if f]}")

print("\n########## 7. FRAGILITY DIAL ##########")
d = dial_series()
dv = d.reindex(epi).dropna()
print(f"  live ma10-63d dial {d.iloc[-1]:.1f}; episodes with a reading "
      f"{len(dv)} of {len(epi)}")
if len(dv):
    print("  " + ", ".join(f"{str(k.date())}={v:.1f}" for k, v in dv.items()))
    print(f"  MAX episode dial {dv.max():.1f} vs today {d.iloc[-1]:.1f} -> today "
          f"{'INSIDE' if d.iloc[-1] <= dv.max() else 'OUTSIDE'} the population")

print("\n########## 8. BOOK OVERLAP + COST ##########")
tr = pd.read_parquet(Path(__file__).resolve().parents[3] / "data"
                     / "backtest_trades_full.parquet")
tr["Signal Date"] = pd.to_datetime(tr["Signal Date"])
pos = pd.Series(range(len(px.index)), index=px.index)
win = set()
for dte in epi:
    p = pos[dte]
    win |= set(px.index[max(0, p - 1):min(len(px.index), p + H + 2)])
sub = tr[tr["Signal Date"].isin(win)]
print(f"  book signals in a [-1,+{H+1}] td window: {len(sub)}")
if len(sub):
    print(sub.groupby(["Strategy", "Direction"]).agg(
        n=("R_Multiple", "size"), avgR=("R_Multiple", "mean")).round(2).to_string())
print(f"  ITA is a ~$7bn ETF; assume ~12 bps round trip incl. spread. episode "
      f"mean {100*vals.mean():.3f}% = {10000*vals.mean():.1f} bps -> "
      f"{10000*vals.mean()/12:.1f}x cost")
