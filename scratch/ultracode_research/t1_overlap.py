"""Test 1: overlap matrix across dials at their own top-decile cutoffs
(day-level and trade-level), plus Test 2: 'any dial top decile' vs 63d rule."""
import numpy as np
import pandas as pd

from dial_setup import DIALS, clustered_t, load_frag, load_trades_joined, smooth

frag = load_frag()
sm = smooth(frag, 10).dropna()

# top-decile cutoffs of each dial's smoothed daily distribution
p90 = {d: sm[d].quantile(0.90) for d in DIALS}
print("p90 cutoffs (10d MA daily distribution):",
      {d: round(v, 1) for d, v in p90.items()})

flags = pd.DataFrame({d: sm[d] >= p90[d] for d in DIALS})

print("\n=== DAY-LEVEL OVERLAP (N days =", len(flags), ") ===")
print("days flagged per dial:", {d: int(flags[d].sum()) for d in DIALS})
print("\nP(col flagged | row flagged):")
cond = pd.DataFrame(index=DIALS, columns=DIALS, dtype=float)
jac = pd.DataFrame(index=DIALS, columns=DIALS, dtype=float)
for a in DIALS:
    for b in DIALS:
        cond.loc[a, b] = flags[flags[a]][b].mean()
        jac.loc[a, b] = (flags[a] & flags[b]).sum() / max((flags[a] | flags[b]).sum(), 1)
print(cond.round(2).to_string())
print("\nJaccard (intersection/union):")
print(jac.round(2).to_string())
n_any = (flags.any(axis=1)).sum()
n_all = (flags.all(axis=1)).sum()
print(f"\ndays any-dial flagged: {n_any} ({n_any/len(flags)*100:.1f}%), "
      f"all three: {n_all} ({n_all/len(flags)*100:.1f}%)")

# ---- trade-level ----
t = load_trades_joined(10)
nb = t[~t.is_ovs].copy()
print(f"\n=== TRADE-LEVEL (non-OVS, N={len(nb)}) ===")
for d in DIALS:
    nb[f"flag_{d}"] = nb[f"frag_{d}"] >= p90[d]
nb["flag_any"] = nb[[f"flag_{d}" for d in DIALS]].any(axis=1)
nb["flag_all"] = nb[[f"flag_{d}" for d in DIALS]].all(axis=1)

print("trades flagged:", {d: int(nb[f"flag_{d}"].sum()) for d in DIALS},
      "any:", int(nb.flag_any.sum()), "all:", int(nb.flag_all.sum()))
cond_t = pd.DataFrame(index=DIALS, columns=DIALS, dtype=float)
for a in DIALS:
    for b in DIALS:
        cond_t.loc[a, b] = nb[nb[f"flag_{a}"]][f"flag_{b}"].mean()
print("\nP(col flagged | row flagged), trades:")
print(cond_t.round(2).to_string())

# ---- Test 2: rules head-to-head ----
print("\n=== RULES HEAD-TO-HEAD (non-OVS, monthly-clustered) ===")
rules = {
    "63d only (p90)": nb.flag_63d,
    "21d only (p90)": nb.flag_21d,
    "5d only (p90)": nb.flag_5d,
    "any dial p90": nb.flag_any,
    "all dials p90": nb.flag_all,
    "63d>=50 (established)": nb.frag_63d >= 50,
    "63d>=50 OR 21d p90": (nb.frag_63d >= 50) | nb.flag_21d,
    "63d>=50 OR 5d p90": (nb.frag_63d >= 50) | nb.flag_5d,
    "2-of-3 p90": (nb[[f"flag_{d}" for d in DIALS]].sum(axis=1) >= 2),
}
rows = []
for name, mask in rules.items():
    hi, lo = nb[mask], nb[~mask]
    ts, p, nmh, nml = clustered_t(hi, lo)
    rows.append({
        "rule": name, "N_hi": len(hi), "avgR_hi": hi.R_Multiple.mean(),
        "avgR_lo": lo.R_Multiple.mean(),
        "spread": hi.R_Multiple.mean() - lo.R_Multiple.mean(),
        "t": ts, "p": p, "mo_hi": nmh,
        "pct_trades": len(hi) / len(nb) * 100,
    })
print(pd.DataFrame(rows).round(3).to_string(index=False))

# LOYO for the top contenders
print("\n=== LOYO (drop each year, recompute clustered t) ===")
for name in ["63d>=50 (established)", "any dial p90", "63d>=50 OR 21d p90", "21d only (p90)"]:
    mask = rules[name]
    ts_all = []
    for yr in sorted(nb.yr.unique()):
        sub = nb[nb.yr != yr]
        m = mask.loc[sub.index]
        ts, p, _, _ = clustered_t(sub[m], sub[~m])
        ts_all.append((yr, ts))
    worst = max(ts_all, key=lambda x: x[1])
    print(f"{name}: t range [{min(x[1] for x in ts_all):+.2f}, {max(x[1] for x in ts_all):+.2f}], "
          f"weakest when dropping {worst[0]} (t={worst[1]:+.2f})")
