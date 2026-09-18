"""K3 c8 round 1 - Long IEF against 0.523 TLT after a BELLY-LED five-day
selloff (IEF 5d pct_rank <= 5 while TLT 5d pct_rank >= 20). Signal 2026-09-16.

  T0 live state
  T1 cell vs controls h=1..10 (all days, own drift in span, local +/-126,
     TDOM-MATCHED), fixed 0.523 hedge and realised rolling-252 beta hedge
  T2 leg attribution (IEF leg vs own drift, short TLT leg vs own drift)
  T3 filter_vs_reanchor vs parent A (IEF 5d rank <= 5) and parent B
     (^TNX within 0.25% of its trailing-252 max, watchlist 19's level)
  T4 era: pre-2018 / 2018+ / 2022-2023 hiking era; midterm
  T5 cost: 4.423 bp two-leg round trip incl borrow (watchlist 19's figure)
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import rolling_on_valid

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
BAR = pd.Timestamp("2026-09-16")
px = close_panel(["IEF", "TLT", "^TNX"])
px = px[px.index <= BAR].dropna(subset=["IEF", "TLT"])
D = px.index
COST = 4.423
HEDGE = 0.523
LEGS = [("IEF", 1.0), ("TLT", -HEDGE)]
ief5 = pct_rank(px["IEF"], 5)
tlt5 = pct_rank(px["TLT"], 5)
tnx = px["^TNX"]
tnx_hi = rolling_on_valid(tnx, lambda x: x.rolling(252).max())
tnx_at_max = (tnx / tnx_hi - 1.0) >= -0.0025

CELL = (ief5 <= 5) & (tlt5 >= 20)
PA = ief5 <= 5
PB = tnx_at_max.reindex(D, fill_value=False)
print(f"LIVE: IEF r5 {ief5.iloc[-1]:.1f}  TLT r5 {tlt5.iloc[-1]:.1f}  "
      f"cell {bool(CELL.iloc[-1])}  PA {bool(PA.iloc[-1])}  "
      f"TNX {tnx.iloc[-1]:.3f} hi {tnx_hi.iloc[-1]:.3f} PB {bool(PB.iloc[-1])}")
print(f"days: cell {int(CELL.sum())}  PA {int(PA.sum())}  PB {int(PB.sum())}  "
      f"cell&PB {int((CELL & PB).sum())}")

r1 = px[["IEF", "TLT"]].pct_change()
b_real = (r1["IEF"].rolling(252).cov(r1["TLT"]) / r1["TLT"].rolling(252).var())
print(f"realised beta IEF on TLT live {b_real.iloc[-1]:.3f}  "
      f"median {b_real.median():.3f}")

tdom = pd.Series(D, index=D).groupby([D.year, D.month]).rank().astype(int)
tdom = pd.Series(tdom.values, index=D)


def fixed(h, lag=1):
    return vehicle_ret(px, LEGS, h, lag)


def realised(h, lag=1):
    return fwd_lag(px["IEF"], h, lag) - b_real * fwd_lag(px["TLT"], h, lag)


def cellrow(mask, ret, h, lab, gap=None):
    valid = ret.notna()
    sig = D[(mask.reindex(D, fill_value=False) & valid).values]
    epi = declusters(sig, gap or max(h, 5), D)
    v = ret.loc[epi].values
    r = summarize(v, lab)
    if not r["n"]:
        return r, epi, v
    w = int((v > 0).sum())
    r["rec"] = f"{w}-{r['n']-w}"
    r["sign_p"] = round(sign_test(w, r["n"]), 4)
    base = ret[valid & ~mask.reindex(D, fill_value=False)]
    r["ctl_all_bp"] = round(1e4 * base.mean(), 1)
    tm = base.groupby(tdom.reindex(base.index)).mean()
    r["tdom_ctl_bp"] = round(1e4 * np.nanmean(tm.reindex(tdom.loc[epi]).values), 1)
    r["edge_tdom_bp"] = round(1e4 * v.mean() - r["tdom_ctl_bp"], 1)
    loc = local_control(D[valid.values], sig)
    r["local_bp"] = round(1e4 * ret.loc[loc].mean(), 1)
    r["mean_bp"] = round(1e4 * v.mean(), 1)
    r["cost_x"] = round(1e4 * v.mean() / COST, 2)
    return r, epi, v


cols = ["label", "n", "mean_bp", "hit", "t", "rec", "sign_p", "ctl_all_bp",
        "tdom_ctl_bp", "edge_tdom_bp", "local_bp", "cost_x", "worst_pct"]
for name, fn in [("FIXED 0.523", fixed), ("REALISED beta", realised)]:
    rows = []
    for h in range(1, 11):
        rows.append(cellrow(CELL, fn(h), h, f"{name} h={h}")[0])
    print(f"\n=== T1 cell {name} ===")
    print(pd.DataFrame(rows)[cols].round(3).to_string(index=False))

# T2 leg attribution
rows = []
for h in (1, 3, 5, 10):
    for lab, r in [("IEF leg", fwd_lag(px["IEF"], h)),
                   ("-0.523 TLT leg", -HEDGE * fwd_lag(px["TLT"], h))]:
        rows.append(cellrow(CELL, r, h, f"{lab} h={h}")[0])
print("\n=== T2 leg attribution (edge vs own all-days / tdom) ===")
print(pd.DataFrame(rows)[cols].round(3).to_string(index=False))

# parents measured on their own
rows = []
for h in (1, 3, 5, 8, 10):
    for lab, m in [("PA IEF r5<=5", PA), ("PB TNX at 252max", PB),
                   ("PA & TLT r5<20 (discard)", PA & ~CELL),
                   ("cell & PB", CELL & PB), ("cell & ~PB", CELL & ~PB)]:
        rows.append(cellrow(m, fixed(h), h, f"{lab} h={h}")[0])
print("\n=== T3a parents and complements (fixed hedge) ===")
print(pd.DataFrame(rows)[cols].round(3).to_string(index=False))

# T3 filter vs reanchor (declustered anchors)
for h in (3, 5, 8):
    ret = fixed(h)
    valid = ret.notna()
    for lab, par in [("PA IEF r5<=5", PA), ("PB TNX 252max", PB)]:
        pd_ = declusters(D[(par & valid).values], max(h, 5), D)
        cd_ = declusters(D[(CELL & valid).values], max(h, 5), D)
        pm = pd.Series(D.isin(pd_), index=D)
        cm = pd.Series(D.isin(cd_), index=D)
        out = filter_vs_reanchor(ret, pm, cm, D, 21, f"h={h} vs {lab}")
        if out["shifts"]:
            rn = reanchor_null(ret, [a for a, _, _ in out["pairs"]],
                               out["shifts"], D, out["kept_at_child_pct"] / 100)
            print(f"  reanchor_null p {rn['p']:.3f}")

# T4 era / regime on fixed hedge
for h in (3, 5, 10):
    r, epi, v = cellrow(CELL, fixed(h), h, "x")
    y = epi.year
    segs = [("pre-2018", y < 2018), ("2018+", y >= 2018),
            ("2022-2023", (y >= 2022) & (y <= 2023)),
            ("2018+ ex 22-23", (y >= 2018) & ~((y >= 2022) & (y <= 2023))),
            ("midterm", y % 4 == 2), ("non-midterm", y % 4 != 2)]
    rows = []
    for lab, m in segs:
        s = summarize(v[m], lab)
        if s["n"]:
            w = int((v[m] > 0).sum())
            s["rec"] = f"{w}-{s['n']-w}"
            s["sign_p"] = round(sign_test(w, s["n"]), 4)
            s["mean_bp"] = round(100 * s["mean_pct"], 1)
        rows.append(s)
    show(rows, f"T4 era h={h}")
    print(cluster_note(epi, v))
