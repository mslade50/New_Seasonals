"""Test 3: threshold stability curves per dial (sweep with N per point).
Test 4: MA window sensitivity 5..21 per dial (percentile-matched coverage)."""
import numpy as np
import pandas as pd

from dial_setup import DIALS, clustered_t, load_frag, load_trades_joined, smooth

frag = load_frag()

print("=== TEST 3: threshold sweep per dial (10d MA, non-OVS, clustered t) ===")
t = load_trades_joined(10)
nb = t[~t.is_ovs].copy()
sm10 = smooth(frag, 10).dropna()

for d in DIALS:
    print(f"\n--- dial {d} ---  (daily pctiles: "
          f"p70={sm10[d].quantile(.7):.0f} p80={sm10[d].quantile(.8):.0f} "
          f"p90={sm10[d].quantile(.9):.0f} p95={sm10[d].quantile(.95):.0f})")
    col = f"frag_{d}"
    # sweep across daily-distribution percentiles p60..p95
    rows = []
    for pct in [60, 65, 70, 75, 80, 85, 90, 95]:
        thr = sm10[d].quantile(pct / 100)
        mask = nb[col] >= thr
        hi, lo = nb[mask], nb[~mask]
        if len(hi) < 15:
            continue
        ts, p, nmh, _ = clustered_t(hi, lo)
        rows.append({"pctile": pct, "thr": round(thr, 1), "N_hi": len(hi),
                     "avgR_hi": hi.R_Multiple.mean(), "avgR_lo": lo.R_Multiple.mean(),
                     "t": ts, "p": p, "mo_hi": nmh})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

# fixed-value sweep for 63d around the candidate zone (40..65 step 2.5)
print("\n--- 63d fixed-value sweep 35..65 ---")
rows = []
for thr in np.arange(35, 67.5, 2.5):
    mask = nb.frag_63d >= thr
    hi, lo = nb[mask], nb[~mask]
    if len(hi) < 15:
        continue
    ts, p, nmh, _ = clustered_t(hi, lo)
    rows.append({"thr": thr, "N_hi": len(hi), "avgR_hi": hi.R_Multiple.mean(),
                 "avgR_lo": lo.R_Multiple.mean(), "t": ts, "p": p, "mo_hi": nmh})
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\n=== TEST 4: MA window sensitivity (threshold = each window's own daily p-tile) ===")
# For 63d hold coverage constant two ways: percentile-matched (p79 ~ thr 50 zone)
# and fixed-value 50. For 21d/5d use p90.
for d, pcts in [("63d", [79, 90]), ("21d", [90]), ("5d", [90])]:
    for pct in pcts:
        rows = []
        for w in [1, 3, 5, 8, 10, 13, 15, 18, 21]:
            tw = load_trades_joined(w)
            nbw = tw[~tw.is_ovs]
            smw = smooth(frag, w).dropna()
            thr = smw[d].quantile(pct / 100)
            mask = nbw[f"frag_{d}"] >= thr
            hi, lo = nbw[mask], nbw[~mask]
            ts, p, nmh, _ = clustered_t(hi, lo)
            rows.append({"MA_w": w, "thr": round(thr, 1), "N_hi": len(hi),
                         "avgR_hi": hi.R_Multiple.mean(), "t": ts, "p": p})
        print(f"\n--- {d}, threshold at daily p{pct} of each window's own distribution ---")
        print(pd.DataFrame(rows).round(3).to_string(index=False))
