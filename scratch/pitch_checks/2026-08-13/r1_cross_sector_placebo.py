"""r1 - RED TEAM attack 1 (the decisive one): cross-sector placebo.

Question: is the C6 survivor an IHI fact, or is "21d rank >= 99 while still
>= 10% below the 52w high" a generic sector-momentum-continuation fact that
IHI happens to be today's instance of?

Identical trigger, identical h=5 lag-1 MOC long, on every liquid
sector/industry ETF in the cache with >= 2010 history. Each instrument's
rolling 252d max is computed on its OWN series (registry: close_panel unions
dates and silently moves a 52w window).

Reports per ticker: episodes N, mean, hit, own-drift control over the SAME
trigger span, excess, sign p vs the ticker's own base rate. Then the pooled
cell, the ex-IHI pooled cell, IHI's cross-sectional rank, and a null for
"max of K draws".
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

H = 5
R21_MIN = 99.0
DD_MAX = -0.10

TK = ["XLV", "XLK", "XLE", "XLF", "XLI", "XLB", "XLP", "XLU", "XLY",
      "SMH", "XBI", "KRE", "IHI", "VNQ", "XOP", "OIH", "GDX", "XME",
      "ITA", "ITB", "IYR", "IYT", "XRT", "XHB", "IBB", "GDXJ", "COPX"]

px_map = load_prices(TK)
TK = [t for t in TK if t in px_map]

DEFENSIVE = {"XLV", "XLP", "XLU", "IHI", "IBB", "XBI"}


def cell(c: pd.Series, r21_min=R21_MIN, dd_max=DD_MAX, h=H, min_gap=5):
    c = c.dropna()
    r21 = pct_rank(c, 21)
    dd = c / c.rolling(252).max() - 1.0
    m = ((r21 >= r21_min) & (dd <= dd_max)).fillna(False)
    ret = fwd_lag(c, h)
    valid = ret.notna()
    trig = c.index[m.values & valid.values]
    if len(trig) == 0:
        return None
    epi = declusters(trig, min_gap, c.index)
    epi = epi[ret.reindex(epi).notna().values]
    if len(epi) == 0:
        return None
    v = ret.loc[epi].values
    span = (c.index >= trig[0]) & (c.index <= trig[-1]) & valid.values
    ctrl = ret[span].values
    base_hit = float((ctrl > 0).mean())
    wins = int((v > 0).sum())
    return {
        "ticker": None, "n_days": int((m.values & valid.values).sum()),
        "n_epi": len(epi), "mean_pct": 100 * v.mean(),
        "hit": 100 * (v > 0).mean(),
        "drift_pct": 100 * ctrl.mean(), "base_hit": 100 * base_hit,
        "excess_pp": 100 * (v.mean() - ctrl.mean()),
        "sign_p_coin": sign_test(wins, len(v)),
        "sign_p_base": sign_test(wins, len(v), base_hit),
        "record": f"{wins}-{len(v)-wins}",
        "first": str(epi[0].date()), "last": str(epi[-1].date()),
        "_v": v, "_epi": epi, "_ctrl_mean": ctrl.mean(),
    }


print(f"=== 1. CROSS-SECTOR PLACEBO: r21>={R21_MIN} & dd<={DD_MAX:.0%}, "
      f"long h={H} lag=1, episodes min_gap 5 ===")
rows, keep = [], {}
for t in TK:
    r = cell(px_map[t]["Close"])
    if r is None:
        rows.append({"ticker": t, "n_epi": 0})
        continue
    r["ticker"] = t
    keep[t] = r
    rows.append({k: v for k, v in r.items() if not k.startswith("_")})
df = pd.DataFrame(rows).sort_values("excess_pp", ascending=False)
cols = ["ticker", "n_days", "n_epi", "mean_pct", "hit", "drift_pct",
        "base_hit", "excess_pp", "sign_p_coin", "sign_p_base", "record",
        "first", "last"]
print(df[[c for c in cols if c in df.columns]].to_string(index=False,
                                                         float_format=lambda x: f"{x:8.3f}"))

n_fire = len(keep)
pos = sum(1 for r in keep.values() if r["excess_pp"] > 0)
print(f"\n  tickers with the state ever: {n_fire} of {len(TK)}")
print(f"  positive excess-over-own-drift: {pos}/{n_fire} "
      f"({100*pos/n_fire:.0f}%)")
print(f"  median excess across tickers: "
      f"{np.median([r['excess_pp'] for r in keep.values()]):+.3f}pp")
print(f"  mean   excess across tickers: "
      f"{np.mean([r['excess_pp'] for r in keep.values()]):+.3f}pp")

# ------------------------------------------------------------------ pooled
print("\n=== 2. POOLED cell (every ticker's episodes stacked) ===")
allv = np.concatenate([r["_v"] for r in keep.values()])
allx = np.concatenate([r["_v"] - r["_ctrl_mean"] for r in keep.values()])
exihi = np.concatenate([r["_v"] for t, r in keep.items() if t != "IHI"])
exihix = np.concatenate([r["_v"] - r["_ctrl_mean"]
                         for t, r in keep.items() if t != "IHI"])
show([summarize(allv, f"POOLED raw (K={n_fire} tickers)"),
      summarize(allx, "POOLED excess over each ticker's own drift"),
      summarize(exihi, "POOLED raw EX-IHI"),
      summarize(exihix, "POOLED excess EX-IHI")])
for lbl, v in [("pooled excess", allx), ("pooled excess ex-IHI", exihix)]:
    w = int((v > 0).sum())
    print(f"  {lbl}: N={len(v)} record {w}-{len(v)-w} sign p(coin) "
          f"{sign_test(w, len(v)):.4f}  bootstrap P(mean<=0) "
          f"{bootstrap_p_le0(v):.4f}")

# ------------------------------------------------------------------ IHI rank
print("\n=== 3. IHI's position in the cross-section ===")
ihi = keep["IHI"]
ex = pd.Series({t: r["excess_pp"] for t, r in keep.items()}).sort_values(ascending=False)
print("  excess_pp ranking (desc):")
print("   " + "  ".join(f"{t}:{v:+.2f}" for t, v in ex.items()))
rk = int((ex > ihi["excess_pp"]).sum()) + 1
print(f"  IHI excess {ihi['excess_pp']:+.3f}pp ranks {rk} of {len(ex)}")
mn = pd.Series({t: r["mean_pct"] for t, r in keep.items()}).sort_values(ascending=False)
print(f"  IHI raw mean {ihi['mean_pct']:+.3f}% ranks "
      f"{int((mn > ihi['mean_pct']).sum())+1} of {len(mn)}")
# how often does a max-of-K draw look this good, given the pooled distribution?
rng = np.random.default_rng(7)
sizes = [len(r["_v"]) for r in keep.values()]
maxes = []
for _ in range(4000):
    best = -1e9
    for s in sizes:
        samp = rng.choice(exihix, size=s, replace=True)
        best = max(best, samp.mean())
    maxes.append(best)
maxes = np.array(maxes)
print(f"  MAX-OF-K NULL: resample the pooled ex-IHI excess into the same "
      f"{len(sizes)} sample sizes; P(max >= IHI's {ihi['excess_pp']:+.3f}pp) = "
      f"{(maxes >= ihi['excess_pp']).mean():.4f}  "
      f"(null max mean {maxes.mean():+.3f}pp, p90 {np.percentile(maxes,90):+.3f})")

# ------------------------------------------------------------------ subgroup
print("\n=== 4. SUBGROUP: defensives/healthcare vs cyclicals ===")
for lbl, sel in [("DEFENSIVE/HC " + str(sorted(DEFENSIVE & set(keep))),
                  [t for t in keep if t in DEFENSIVE]),
                 ("CYCLICAL/other", [t for t in keep if t not in DEFENSIVE])]:
    v = np.concatenate([keep[t]["_v"] - keep[t]["_ctrl_mean"] for t in sel])
    r = summarize(v, lbl)
    r["n_tickers"] = len(sel)
    show([r])
    tick = {t: round(keep[t]["excess_pp"], 2) for t in sel}
    print("    per-ticker excess:", tick)

# ------------------------------------------------------------------ robustness
print("\n=== 5. cross-section under NEIGHBOURING gates (excess_pp, h=5) ===")
grid = []
for rq in (97, 99, 100):
    for dm in (-0.05, -0.10):
        vals = {}
        for t in TK:
            r = cell(px_map[t]["Close"], rq, dm)
            vals[t] = round(r["excess_pp"], 2) if r else None
        v2 = [x for x in vals.values() if x is not None]
        grid.append({"r21>=": rq, "dd<=": dm, "k_fire": len(v2),
                     "median_excess": round(float(np.median(v2)), 3),
                     "frac_pos": round(float(np.mean(np.array(v2) > 0)), 2),
                     "IHI": vals.get("IHI"),
                     "IHI_rank": (sorted(v2, reverse=True).index(vals["IHI"]) + 1)
                     if vals.get("IHI") is not None else None})
print(pd.DataFrame(grid).to_string(index=False))

# ------------------------------------------------------------------ anti-rip-off
print("\n=== 6. ANTI-RIP-OFF: does the state overlap the book's breakout strats? ===")
c = px_map["IHI"]["Close"].dropna()
dd = c / c.rolling(252).max() - 1.0
r252 = pct_rank(c, 252)
m = ((pct_rank(c, 21) >= R21_MIN) & (dd <= DD_MAX)).fillna(False)
new52 = (c >= c.rolling(252).max() - 1e-12)
print(f"  Sector BO / 52wh Breakout both require use_52w 'New 52w High'.")
print(f"  trigger days that ARE a new 52w high: {int((m & new52).sum())} of "
      f"{int(m.sum())}  -> disjoint by construction (dd<=-10% forbids it)")
print(f"  IHI 252d rank on trigger days: median {r252.loc[c.index[m.values]].median():.1f} "
      f"(Sector BO wants 65-90, 52wh wants 50-90); today {r252.iloc[-1]:.1f}")
print(f"  IHI dd today {100*dd.iloc[-1]:.2f}% -> neither book strat can fire.")
