"""MA-window sweep for the exact production candidate: 63d dial, fixed thr=50."""
import pandas as pd

from dial_setup import clustered_t, load_trades_joined

rows = []
for w in [1, 3, 5, 8, 10, 13, 15, 18, 21]:
    tw = load_trades_joined(w)
    nbw = tw[~tw.is_ovs]
    mask = nbw["frag_63d"] >= 50
    hi, lo = nbw[mask], nbw[~mask]
    ts, p, nmh, _ = clustered_t(hi, lo)
    rows.append({"MA_w": w, "N_hi": len(hi), "avgR_hi": hi.R_Multiple.mean(),
                 "avgR_lo": lo.R_Multiple.mean(), "t": ts, "p": p, "mo_hi": nmh})
print("63d dial, fixed thr=50, MA window sweep (non-OVS, clustered):")
print(pd.DataFrame(rows).round(3).to_string(index=False))
