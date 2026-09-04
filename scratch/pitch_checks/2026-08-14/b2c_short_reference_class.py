"""C2 round 2, part 3 — the ONE form that survived b2b's era test: SHORT the
insurance basket at h=10 (+1.055%, 9-3, sign p 0.073, both eras positive,
top-2 concentration only 10%).

Three things decide it, and all three are pre-registered here before the run:

  (A) TRADEABILITY. The 14-name equal-weight basket is NOT a legal pitch (the
      grammar caps an idea at 4 legs). Every 4-name selection rule available at
      signal time is measured. If no 4-name subset carries the basket's number,
      it is a kill on tradeability.
  (B) REFERENCE CLASS. The short direction was found by flipping the sign on a
      losing long, and insurance is the WORST of 10 industry groups on the
      long side at h=5 (excess -1.106pp against a cross-group mean of -0.236pp,
      sd 0.640pp). That is the definition of an extreme draw from a family. The
      identical short is run on all 10 groups at h=10, with a homogeneity test
      and a family-wise rank.
  (C) COST. A 4-leg equity short round trip against the realised edge.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pitch_lab import (bootstrap_p_le0, cluster_note, declusters, fwd_lag,  # noqa: E402
                       load_prices, pct_rank, show, sign_test, summarize)

ASOF = pd.Timestamp("2026-08-13")
H = 10

GROUPS = {
    "insurance": ["HIG", "ALL", "TRV", "AIG", "MET", "PGR", "CB", "AFL",
                  "PRU", "LNC", "WRB", "CINF", "L", "GL"],
    "banks": ["JPM", "BAC", "WFC", "C", "USB", "PNC", "TFC", "MTB", "FITB",
              "KEY", "RF", "HBAN", "ZION"],
    "semis": ["NVDA", "AMD", "INTC", "MU", "TXN", "ADI", "AMAT", "LRCX",
              "KLAC", "MCHP", "NXPI", "SWKS", "QCOM", "AVGO"],
    "retail": ["WMT", "TGT", "COST", "HD", "LOW", "TJX", "ROST", "DG",
               "DLTR", "BBY", "KSS", "M"],
    "energy": ["XOM", "CVX", "COP", "EOG", "SLB", "HAL", "OXY", "PSX",
               "VLO", "MPC", "DVN"],
    "utilities": ["NEE", "DUK", "SO", "D", "AEP", "EXC", "XEL", "ED", "WEC",
                  "ES", "PEG", "PNW", "ETR"],
    "transports": ["UNP", "CSX", "NSC", "UPS", "FDX", "DAL", "UAL", "LUV",
                   "ODFL", "CHRW", "EXPD"],
    "staples": ["KO", "PEP", "PG", "MO", "PM", "MDLZ", "GIS", "CL",
                "KMB", "SYY", "HSY"],
    "machinery": ["CAT", "DE", "HON", "GE", "MMM", "EMR", "ITW", "PH",
                  "ROK", "DOV", "ETN"],
    "homebuild_disc": ["DHI", "LEN", "PHM", "WHR", "F", "GM", "APTV",
                       "BWA", "LKQ", "GPC"],
}
ALL = sorted({t for v in GROUPS.values() for t in v} | {"SPY"})
raw = load_prices(ALL)
spy = raw["SPY"]["Close"]
BASE_IDX = spy[spy.index <= ASOF].index


def state(names):
    have = [t for t in names if t in raw]
    panel = pd.DataFrame({t: raw[t]["Close"] for t in have}).reindex(BASE_IDX)
    r5 = pd.DataFrame({t: pct_rank(panel[t].dropna(), 5).reindex(BASE_IDX)
                       for t in panel})
    r63 = pd.DataFrame({t: pct_rank(panel[t].dropna(), 63).reindex(BASE_IDX)
                        for t in panel})
    nav = r5.notna().sum(axis=1)
    ew = (1.0 + panel.pct_change().mean(axis=1, skipna=True).fillna(0.0)).cumprod()
    m = (((r5 <= 20).sum(axis=1) / nav >= 0.70) & (nav >= 8)
         & (r63.median(axis=1) >= 70)).fillna(False)
    return panel, r5, r63, ew, m, have


panel, r5, r63, insew, m0, have = state(GROUPS["insurance"])
ret = fwd_lag(insew, H)
trig = BASE_IDX[m0.values & ret.notna().values]
epi = declusters(trig, H, BASE_IDX)
print(f"insurance triggers {len(trig)} days -> {len(epi)} episodes at h={H}")

print("\n" + "=" * 78)
print("A. TRADEABILITY — every 4-name selection rule available at signal time")
print("=" * 78)
pos = pd.Series(range(len(BASE_IDX)), index=BASE_IDX)


def sel_short(dates, k, key_fn, lag=1):
    d_out, v_out, picks = [], [], []
    for d in dates:
        p = pos.get(d)
        if p is None or p + lag + H >= len(BASE_IDX):
            continue
        key = key_fn(d).dropna()
        if len(key) < k:
            continue
        sel = list(key.sort_values().index[:k])
        e, x = panel.iloc[p + lag][sel], panel.iloc[p + lag + H][sel]
        if e.isna().any() or x.isna().any():
            continue
        d_out.append(d)
        v_out.append(float(-(x / e - 1.0).mean()))  # SHORT
        picks.append(sel)
    return pd.DatetimeIndex(d_out), np.asarray(v_out), picks


RULES = {
    "4 most-washed (lowest rank5)": lambda d: r5.loc[d],
    "4 strongest (highest rank63)": lambda d: -r63.loc[d],
    "4 weakest 63d (lowest rank63)": lambda d: r63.loc[d],
    "4 alphabetically first (placebo)": lambda d: pd.Series(
        range(len(panel.columns)), index=panel.columns).where(
        panel.loc[d].notna()),
}
rows = []
for lbl, fn in RULES.items():
    d, v, pk = sel_short(epi, 4, fn)
    if len(v) == 0:
        continue
    s = summarize(v, f"SHORT {lbl} (N={len(v)})")
    s["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
    s["boot"] = round(bootstrap_p_le0(v), 3)
    s["x_cost_40bp"] = round(100 * v.mean() * 100 / 40, 2)
    rows.append(s)
d14, v14 = pd.DatetimeIndex(epi), -ret.loc[epi].values
s = summarize(v14, f"SHORT full 14-name basket (NOT tradeable) (N={len(v14)})")
s["sign_p"] = round(sign_test(int((v14 > 0).sum()), len(v14)), 4)
s["boot"] = round(bootstrap_p_le0(v14), 3)
s["x_cost_40bp"] = round(100 * v14.mean() * 100 / 40, 2)
rows.append(s)
show(rows, "4-leg tradeable forms vs the untradeable 14-name basket")
print("  (x_cost_40bp: edge / a 4-leg equity short round trip at ~10 bps/leg;")
print("   the book's bar is >= 5x)")

print("\n" + "=" * 78)
print("B. REFERENCE CLASS — the identical SHORT on 10 industry groups, h=10")
print("=" * 78)
res = []
for g, names in GROUPS.items():
    _, _, _, gew, gm, ghave = state(names)
    if len(ghave) < 8:
        continue
    r = fwd_lag(gew, H)
    t = BASE_IDX[gm.values & r.notna().values]
    if len(t) < 3:
        res.append({"group": g, "n_epi": 0})
        continue
    e = declusters(t, H, BASE_IDX)
    v = -r.loc[e].values
    drift = -r.dropna()
    se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
    res.append({"group": g, "n_names": len(ghave), "n_epi": len(e),
                "short_mean_pct": round(100 * v.mean(), 3),
                "hit": round(100 * (v > 0).mean(), 1),
                "short_drift_pct": round(100 * drift.mean(), 3),
                "excess_pp": round(100 * (v.mean() - drift.mean()), 3),
                "se_pp": round(100 * se, 3),
                "sign_p": round(sign_test(int((v > 0).sum()), len(v)), 3)})
df = pd.DataFrame(res).dropna(subset=["excess_pp"]).sort_values(
    "excess_pp", ascending=False)
print(df.to_string(index=False))

ex = df["excess_pp"].values / 100.0
se = df["se_pp"].values / 100.0
w = 1.0 / se**2
fe = float((w * ex).sum() / w.sum())
Q = float((w * (ex - fe) ** 2).sum())
dfree = len(ex) - 1
try:
    from scipy.stats import chi2
    pQ = float(chi2.sf(Q, dfree))
except Exception:
    pQ = np.nan
I2 = max(0.0, (Q - dfree) / Q) if Q > 0 else 0.0
ins = df[df["group"] == "insurance"]["excess_pp"].iloc[0]
rank = int((df["excess_pp"] >= ins).sum())
print(f"\n  fixed-effect COMMON short excess across the family = {100*fe:+.3f}pp")
print(f"  Cochran Q = {Q:.2f} on {dfree} df, p = {pQ:.3f}, I^2 = {100*I2:.1f}%")
print(f"  insurance excess {ins:+.3f}pp ranks {rank} of {len(ex)} "
      f"(family-wise p for 'best group under no effect' = {rank/len(ex):.3f})")

print("\n" + "=" * 78)
print("C. what the family-wise number means, spelled out")
print("=" * 78)
print("  The short was reached by flipping the sign on a LOSING long, then")
print("  choosing the horizon (3/5/10) that scored best. Direction x horizon")
print(f"  x 10 groups is the search; the family shows a common effect of "
      f"{100*fe:+.3f}pp with Q p={pQ:.3f}.")
