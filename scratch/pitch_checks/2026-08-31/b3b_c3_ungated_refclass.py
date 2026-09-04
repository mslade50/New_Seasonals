"""C3 round 2b: the LAST open door.

Round 2 killed the pitched object (the "deeply lagging" gate SUBTRACTS 1.867pp
on AVGO itself). What survived the wreck is AVGO's UNGATED pre-print session:
+0.513% on 33-22, beta-neutral residual +0.525% at t +2.66, era-stable. That is
not the pitched trade, but a checker owes it the same reference class before
calling it a near-miss, because a single name selected out of 962 for having a
pretty pre-print cell is the textbook max-of-N draw.

Runs: ungated reference class (Cochran Q / I2 / common excess / AVGO's rank /
permutation max-of-N), drop-2-best, per-quarter breakdown, and the September
sub-cell that today actually is.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd
from scipy import stats as st

OUT = Path(__file__).resolve().parent
close = pd.read_pickle(OUT / "_c3_panel.pkl")
D = pd.read_pickle(OUT / "_c3_anchors.pkl")

# ---------------------------------------------------------------------------
# 1. ungated reference class
# ---------------------------------------------------------------------------
rows = []
for t, g in D.groupby("ticker"):
    if len(g) < 20:
        continue
    e = g["excess"].values
    sd = e.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        continue
    rows.append({"ticker": t, "n": len(e), "excess_pp": 100 * e.mean(),
                 "se_pp": 100 * sd / np.sqrt(len(e))})
RC = pd.DataFrame(rows)
w = 1.0 / RC["se_pp"] ** 2
common = float((w * RC["excess_pp"]).sum() / w.sum())
se_c = float(np.sqrt(1.0 / w.sum()))
Q = float((w * (RC["excess_pp"] - common) ** 2).sum())
dfq = len(RC) - 1
I2 = max(0.0, 100 * (Q - dfq) / Q) if Q > 0 else 0.0
RC["t"] = RC["excess_pp"] / RC["se_pp"]
RC = RC.sort_values("excess_pp", ascending=False).reset_index(drop=True)

print("=" * 78)
print("1. UNGATED pre-print reference class (names with >= 20 prints)")
print("=" * 78)
print(f"  class size {len(RC)} names, {int(RC['n'].sum())} anchors")
print(f"  common fixed-effect excess {common:+.4f}pp (se {se_c:.4f}, z {common/se_c:+.2f})")
print(f"  Cochran Q {Q:.2f} on {dfq} df, p {st.chi2.sf(Q, dfq):.4f}, I-squared {I2:.1f}%")
rk = int(RC.index[RC["ticker"] == "AVGO"][0]) + 1
a = RC[RC["ticker"] == "AVGO"].iloc[0]
print(f"  AVGO excess {a['excess_pp']:+.4f}pp on n={int(a['n'])}, t {a['t']:+.2f}, "
      f"RANK {rk} of {len(RC)}  -> empirical family-wise share {rk/len(RC):.4f}")
print("\n  top 10:")
print(RC.head(10).round(4).to_string(index=False))

rng = np.random.default_rng(11)
nulls = np.array([(RC["t"].values * rng.choice([-1.0, 1.0], size=len(RC))).max()
                  for _ in range(4000)])
print(f"\n  permutation (sign-flip) null over {len(RC)} names:")
print(f"    P(max t >= AVGO's t {a['t']:.2f}) = {(nulls >= a['t']).mean():.4f}")
print(f"    P(max t >= class max {RC['t'].max():.2f}) = "
      f"{(nulls >= RC['t'].max()).mean():.4f}")
print(f"    null median max t {np.median(nulls):.2f}, observed class max "
      f"{RC['t'].max():.2f}")

# ---------------------------------------------------------------------------
# 2. AVGO ungated: robustness
# ---------------------------------------------------------------------------
A = D[D["ticker"] == "AVGO"].sort_values("report").copy()
r = A["ret"].values
print("\n" + "=" * 78)
print("2. AVGO ungated pre-print session, robustness")
print("=" * 78)
print(f"  raw mean {100*r.mean():+.3f}%  t {r.mean()/(r.std(ddof=1)/np.sqrt(len(r))):+.2f}")
order = np.argsort(-np.abs(r))
for k in (1, 2, 3):
    drop = np.delete(r, order[:k])
    print(f"  drop-{k}-largest-|r|: mean {100*drop.mean():+.3f}% "
          f"(N={len(drop)})  -> {100*drop.mean()*100/8:.2f}x an 8 bp round trip")
print(f"  concentration: {cluster_note(pd.DatetimeIndex(A['report']), r, k=2)}")

A["q"] = A["report"].dt.month.map({12: "Dec", 1: "Dec", 2: "Q1", 3: "Q1",
                                   5: "Q2", 6: "Q2", 8: "Q3", 9: "Q3"})
print("\n  by reporting month:")
for m, g in A.groupby(A["report"].dt.month):
    w_ = int((g["ret"] > 0).sum())
    print(f"    month {m:2d}: N={len(g):2d} mean {100*g['ret'].mean():+7.3f}% "
          f"record {w_}-{len(g)-w_}")

sep = A[A["report"].dt.month.isin([8, 9])]
w_ = int((sep["ret"] > 0).sum())
up = float((close["AVGO"].dropna().pct_change() > 0).mean())
print(f"\n  the AUG/SEP print (today's instance): N={len(sep)} "
      f"mean {100*sep['ret'].mean():+.3f}% record {w_}-{len(sep)-w_} "
      f"sign p {sign_test(w_, len(sep), p=up):.4f}")
print(sep[["report", "r63", "ret"]].assign(
    ret=lambda x: (100 * x["ret"]).round(3)).to_string(index=False))

# ---------------------------------------------------------------------------
# 3. is the ungated cell a MEGA-CAP effect or a small-cap one?
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("3. does the ungated pre-print session survive in liquid large caps?")
print("=" * 78)
sys.path.insert(0, str(ROOT))
from strategy_config import LIQUID_PLUS_COMMODITIES  # noqa
liq = set(LIQUID_PLUS_COMMODITIES)
L = D[D["ticker"].isin(liq)]
NL = D[~D["ticker"].isin(liq)]
show([summarize(L["excess"].values, f"LIQUID names (N={len(L)}, "
                f"{L['ticker'].nunique()} tickers)"),
      summarize(NL["excess"].values, f"overflow/illiquid (N={len(NL)})")],
     "liquidity split, ungated excess")
Lp = L[L["report"] < "2018-01-01"]
Ls = L[L["report"] >= "2018-01-01"]
show([summarize(Lp["excess"].values, f"liquid pre-2018 (N={len(Lp)})"),
      summarize(Ls["excess"].values, f"liquid 2018+ (N={len(Ls)})")],
     "liquid era split")
e = 100 * L["excess"].mean()
print(f"  liquid-name excess {e:.3f}pp = {e*100:.2f} bps -> "
      f"{e*100/8:.2f}x an 8 bp single-name round trip (bar 5x)")
