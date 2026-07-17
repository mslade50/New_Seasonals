"""Dial interpretation, decay-aware.

McKinley's point: HORIZON_DECAY_DD stands the dial down as drawdowns deepen,
so low-dial days mix 'genuinely calm' with 'mid-crash, dial intentionally
silent'. The dial's design jurisdiction is NEAR HIGHS ('cheap time to hedge
when things are going well but not under the surface').

1) Composition: per dial bucket, share of days by drawdown state + avg VIX.
2) Jurisdiction table: near-high days only (within 2% / 5% of 52w high) —
   fwd returns + P(>=10% drawdown within 63td) by dial bucket.
3) Cheapness: VIX3M level on high-dial-near-high days vs all days (is the
   hedge actually cheap when the dial says hedge?).
"""
import os

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

frag = pd.read_parquet(os.path.join(_ROOT, "data", "rd2_fragility.parquet"))
s63 = frag["63d"].dropna().sort_index()
ma10 = s63.rolling(10, min_periods=1).mean()

mp = pd.read_parquet(os.path.join(_ROOT, "data", "master_prices.parquet"),
                     filters=[("ticker", "in", ["SPY", "^VIX", "^VIX3M"])])
mp["date"] = pd.to_datetime(mp["date"])


def ser(tkr):
    return (mp[mp["ticker"] == tkr].set_index("date")["Close"]
            .sort_index().reindex(s63.index).ffill())


spy, vix, vix3m = ser("SPY"), ser("^VIX"), ser("^VIX3M")
hi52 = spy.rolling(252, min_periods=60).max()
dd = spy / hi52 - 1

fwd = {h: spy.shift(-h) / spy - 1 for h in (10, 21, 63)}
n = len(spy)
spy_np = spy.to_numpy()
worst63 = np.full(n, np.nan)
for i in range(n - 63):
    worst63[i] = spy_np[i + 1: i + 64].min() / spy_np[i] - 1

df = pd.DataFrame({
    "score": ma10, "dd": dd, "vix": vix, "vix3m": vix3m,
    "f10": fwd[10], "f21": fwd[21], "f63": fwd[63],
    "worst63": worst63,
}).dropna(subset=["score", "dd"])

BUCKETS = [(0, 5), (5, 15), (15, 27), (27, 40), (40, 50), (50, 200)]


def blab(lo, hi):
    return f"{lo}-{hi if hi < 200 else '+'}"


pd.set_option("display.width", 170)

print("== 1) What lives inside each dial bucket (all days) ==")
rows = []
for lo, hi in BUCKETS:
    g = df[(df["score"] >= lo) & (df["score"] < hi)]
    rows.append({
        "dial": blab(lo, hi), "n": len(g),
        "%near_high(<2%dd)": round((g["dd"] >= -0.02).mean() * 100),
        "%mild(2-10%)": round(((g["dd"] < -0.02) & (g["dd"] >= -0.10)).mean() * 100),
        "%deep(>10%dd)": round((g["dd"] < -0.10).mean() * 100),
        "avg_vix": round(g["vix"].mean(), 1),
        "f63_mean%": round(g["f63"].mean() * 100, 2),
    })
print(pd.DataFrame(rows).set_index("dial").to_string())

for band, thresh in [("within 2% of 52w high", -0.02), ("within 5%", -0.05),
                     ("within 10%", -0.10)]:
    sub = df[df["dd"] >= thresh]
    print(f"\n== 2) Jurisdiction: {band} only (N={len(sub)}) ==")
    rows = []
    for lo, hi in BUCKETS:
        g = sub[(sub["score"] >= lo) & (sub["score"] < hi)]
        if len(g) < 30:
            rows.append({"dial": blab(lo, hi), "n": len(g)})
            continue
        w = g["worst63"].dropna()
        rows.append({
            "dial": blab(lo, hi), "n": len(g),
            "f10_mean%": round(g["f10"].mean() * 100, 2),
            "f21_mean%": round(g["f21"].mean() * 100, 2),
            "f63_mean%": round(g["f63"].mean() * 100, 2),
            "f63_med%": round(g["f63"].median() * 100, 2),
            "%neg63": round((g["f63"] < 0).mean() * 100),
            "P(dd>=5%)": round((w <= -0.05).mean(), 2),
            "P(dd>=10%)": round((w <= -0.10).mean(), 2),
            "avg_vix3m": round(g["vix3m"].mean(), 1),
        })
    print(pd.DataFrame(rows).set_index("dial").to_string())
    corr = sub["score"].corr(sub["f63"], method="spearman")
    print(f"spearman(score, fwd63) inside jurisdiction: {corr:+.3f}")

print("\n== 3) Cheapness when the dial says hedge (near-high, dial >= 40) ==")
sig = df[(df["dd"] >= -0.02) & (df["score"] >= 40)]
print(f"N={len(sig)} days | VIX3M mean {sig['vix3m'].mean():.1f} "
      f"(pctile of all days: {(df['vix3m'] < sig['vix3m'].mean()).mean() * 100:.0f}th) | "
      f"VIX mean {sig['vix'].mean():.1f}")
w = sig["worst63"].dropna()
print(f"fwd63 mean {sig['f63'].mean() * 100:+.2f}%, median {sig['f63'].median() * 100:+.2f}%, "
      f"P(dd>=5%) {(w <= -0.05).mean():.2f}, P(dd>=10%) {(w <= -0.10).mean():.2f} "
      f"vs all-days {(df['worst63'].dropna() <= -0.10).mean():.2f}")
