"""ROUND 2 close-out. r2 found the live state (TLT AT its 52w low) is the
DEAD subcell: N=9 +0.189% at a 44.4% hit against +1.129% for the 40 triggers
more than 3% off the low. An N=9 bucket is not a kill on its own (house
rule), so turn it into a FULL-POPULATION statement:

 (1) continuous regression of the forward return on distance-to-52w-low,
     over the 59 triggers AND over the 288 ungated anchor days
 (2) cut sensitivity - is the wrong-way gradient monotone and stable
 (3) the ungated month-end anchor's own placebo ladder (is the anchor real
     independent of the gate) and its at-the-low behaviour
 (4) the exact live joint state - how many historical days ever matched it
 (5) book overlap: STRATEGY_BOOK universes, the ledger's TLT trades, and the
     trend sleeve's month-end rebalance colliding with this trade's exit
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np, pandas as pd

LAG, H = 1, 9
TK = ["SPY", "TLT", "IEF"]
raw = load_prices(TK + ["^TNX"])
idx = raw["SPY"]["Close"].index
for t in TK[1:]:
    idx = idx.intersection(raw[t]["Close"].index)
px = pd.DataFrame({t: raw[t]["Close"].reindex(idx) for t in TK}).dropna()
idx = px.index
tnx = raw["^TNX"]["Close"].reindex(idx).ffill()
is_last = pd.Series(idx.year * 100 + idx.month, index=idx).ne(
    pd.Series(idx.year * 100 + idx.month, index=idx).shift(-1)).values
is_last[-1] = False
pos = pd.Series(range(len(idx)), index=idx)


def anchor(h=H, k=0):
    t = pos.values + LAG + h + k
    m = np.zeros(len(idx), bool); ok = t < len(idx)
    m[ok] = is_last[t[ok]]
    return m


rT = fwd_lag(px["TLT"], H, LAG)
rS = fwd_lag(px["SPY"], H, LAG)
SP = rT - rS
T21 = px["TLT"].pct_change(21)
low52 = px["TLT"].rolling(252).min()
dist = px["TLT"] / low52 - 1.0
A = anchor(); G = (T21 <= -0.025).fillna(False).values; v = rT.notna().values
D = idx[A & G & v]; DA = idx[A & v]


def ols(xs, ys, lab):
    m = ~np.isnan(xs) & ~np.isnan(ys)
    xs, ys = xs[m], ys[m]
    n = len(xs)
    b, a = np.polyfit(xs, ys, 1)
    yh = a + b * xs
    se = np.sqrt(((ys - yh) ** 2).sum() / (n - 2) / ((xs - xs.mean()) ** 2).sum())
    print("  %-42s n=%3d slope %+.4f pp per 1%% off-the-low, t=%+.2f, "
          "fitted at TODAY (0.00%% off low) = %+.3f%%, at the median (7.5%%) = %+.3f%%"
          % (lab, n, 100 * b / 100, b / se, 100 * a, 100 * (a + b * 0.075)))


print("===== 1. CONTINUOUS gradient: fwd return vs distance to the 52w low =====")
ols(dist.loc[D].values, rT.loc[D].values, "TLT outright, the 59 triggers")
ols(dist.loc[D].values, SP.loc[D].values, "TLT-SPY spread, the 59 triggers")
ols(dist.loc[DA].values, rT.loc[DA].values, "TLT outright, 288 ungated anchor days")
ols(dist.loc[idx[v]].values, rT.loc[idx[v]].values, "TLT outright, ALL 6041 days (baseline)")

print("\n===== 2. cut sensitivity of the at-the-low subcell (TLT outright) =====")
rows = []
for c in (0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10):
    inn = D[(dist.loc[D].values <= c)]
    out = D[(dist.loc[D].values > c)]
    r = summarize(rT.loc[inn].values, f"within {100*c:.1f}% of 52w low (LIVE=0.0%)")
    r["out_pct"] = round(100 * rT.loc[out].mean(), 3)
    r["out_n"] = len(out)
    rows.append(r)
show(rows, "2a. TLT outright: inside the live rung vs outside it")
rows = []
for c in (0.01, 0.03, 0.05):
    inn = D[(dist.loc[D].values <= c)]
    out = D[(dist.loc[D].values > c)]
    r = summarize(SP.loc[inn].values, f"SPREAD within {100*c:.1f}% of low")
    r["out_pct"] = round(100 * SP.loc[out].mean(), 3)
    rows.append(r)
show(rows, "2b. same cuts, TLT-SPY spread")

print("\n===== 3. the UNGATED month-end anchor (the parent), placebo ladder =====")
rows = []
for k in range(0, 11):
    Ak = anchor(H, k)
    d = idx[Ak & v]
    rows.append(summarize(rT.loc[d].values, f"exit k={k} sessions BEFORE month-end"))
show(rows, "3a. ungated anchor exit-offset ladder, TLT outright, h=9")
dl = dist.loc[DA].values
rows = [summarize(rT.loc[DA[dl <= 0.01]].values, "ungated anchor, TLT within 1% of 52w low"),
        summarize(rT.loc[DA[(dl > 0.01) & (dl <= 0.05)]].values, "ungated anchor, 1-5% off low"),
        summarize(rT.loc[DA[dl > 0.05]].values, "ungated anchor, >5% off low")]
show(rows, "3b. is at-the-low poison a property of the GATE or of the ANCHOR?")

print("\n===== 4. how much precedent does the exact LIVE state have? =====")
live_thr = -0.0336
tnx_rk = (tnx.diff(21)).rolling(252).rank(pct=True) * 100
joint = A & (T21 <= live_thr).fillna(False).values & (dist <= 0.01).fillna(False).values & v
dj = idx[joint]
print("month-end anchor + TLT21d<=-3.36%% + within 1%% of 52w low : N=%d  %s"
      % (len(dj), ", ".join(str(x.date()) for x in dj)))
if len(dj):
    print("   TLT %+.3f%% hit %.1f%% | spread %+.3f%% | sign p(record %d-%d) = %.4f"
          % (100 * rT.loc[dj].mean(), 100 * (rT.loc[dj] > 0).mean(),
             100 * SP.loc[dj].mean(), int((rT.loc[dj] > 0).sum()),
             int((rT.loc[dj] <= 0).sum()),
             sign_test(int((rT.loc[dj] > 0).sum()), len(dj))))
print("   live ^TNX 21d-change rank = %.1f (brief said 86.5)" % tnx_rk.iloc[-1])
hi = D[(tnx_rk.loc[D].values >= 70)]
lo = D[(tnx_rk.loc[D].values < 70)]
print("   triggers with ^TNX 21d-chg rank >= 70 (live side): N=%d %+.3f%% hit %.1f%% | rank<70 N=%d %+.3f%%"
      % (len(hi), 100 * rT.loc[hi].mean(), 100 * (rT.loc[hi] > 0).mean(),
         len(lo), 100 * rT.loc[lo].mean()))

print("\n===== 5. BOOK OVERLAP =====")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import strategy_config as sc
for uname in ("LIQUID_PLUS_COMMODITIES", "CSV_UNIVERSE"):
    u = getattr(sc, uname, None)
    if u is not None:
        print("  TLT in %s: %s (universe size %d)" % (uname, "TLT" in set(u), len(set(u))))
# ---- item 5 fixed: STRATEGY_BOOK is a LIST of dicts with universe_tickers ----
print("\n===== 5b. BOOK OVERLAP (corrected) =====")
for s in sc.STRATEGY_BOOK:
    u = s.get("universe_tickers") or []
    if "TLT" in set(u):
        print("  STRATEGY_BOOK carrier: %-34s dir=%s hold=%s entry=%s"
              % (s["name"], s.get("settings", {}).get("direction"),
                 s.get("execution", {}).get("hold_days"),
                 s.get("execution", {}).get("entry_type")))
print("  strategies with TLT in universe_tickers: %d of %d"
      % (sum("TLT" in set(s.get("universe_tickers") or []) for s in sc.STRATEGY_BOOK),
         len(sc.STRATEGY_BOOK)))
L = pd.read_parquet("data/backtest_trades_full.parquet")
print("  ledger cols:", list(L.columns)[:14])
tc = "Ticker" if "Ticker" in L.columns else [c for c in L.columns if "icker" in c][0]
t = L[L[tc] == "TLT"]
print("  ledger TLT trades: %d of %d total" % (len(t), len(L)))
if len(t):
    print("   ", dict(t["Strategy_Name"].value_counts()) if "Strategy_Name" in t.columns else "")
import json, os
p = "data/trend_sleeve_state.json"
print("  trend_sleeve_state.json exists:", os.path.exists(p))
if os.path.exists(p):
    st = json.load(open(p))
    print("   state:", json.dumps(st)[:400])
import trend_sleeve as ts
print("  trend sleeve universe:", getattr(ts, "UNIVERSE", getattr(ts, "TICKERS", "?")))

print("\n===== 5c. would the TREND SLEEVE stage a TLT BUY into this trade's exit? =====")
m = px["TLT"].resample("ME").last()
mom = m / m.shift(12) - 1.0          # 12-1 momentum, skip most recent month
mom121 = m.shift(1) / m.shift(12) - 1.0
ma10 = m.rolling(10).mean()
print("  TLT monthly close %.2f | 10-month MA %.2f -> above MA? %s"
      % (m.iloc[-1], ma10.iloc[-1], bool(m.iloc[-1] > ma10.iloc[-1])))
print("  TLT 12-1 momentum %+.2f%% -> positive? %s"
      % (100 * mom121.iloc[-1], bool(mom121.iloc[-1] > 0)))
print("  combo (mom>0 AND above 10m MA) = %s -> sleeve %s stage TLT at the 08-31 close"
      % (bool((mom121.iloc[-1] > 0) and (m.iloc[-1] > ma10.iloc[-1])),
         "WOULD" if (mom121.iloc[-1] > 0 and m.iloc[-1] > ma10.iloc[-1]) else "would NOT"))
print("  note: sleeve signals on the month-end CLOSE and executes MOO the NEXT session,")
print("        so even a sleeve BUY would not print against this trade's 08-31 MOC exit.")
