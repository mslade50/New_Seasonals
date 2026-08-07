"""Power + regime context for the four 2026-08-07 candidates.

(a) What effect size could each study even DETECT at its episode N? A candidate
    whose measured edge is far inside its own MDE band is untestable, not proven.
(b) What does the book's own live risk state say about a long utilities dip-buy
    on 2026-08-07 (fragility dial + P/C fear state)?
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import *  # noqa
from _engine import vehicle_ret

import numpy as np
import pandas as pd

px = close_panel(["SPY", "XLU", "XLP", "TLT", "IEF", "^TNX"]).dropna()
z = zscore(px["XLU"], 10)
spy_off = px["SPY"] / px["SPY"].rolling(252).max() - 1.0
sp21 = px["XLU"].pct_change(21) - px["XLP"].pct_change(21)
tlt_off = px["TLT"] / px["TLT"].rolling(252).min() - 1.0
tnx63 = pct_rank(px["^TNX"], 63)

CAND = {
    "C1 XLU long":      ((z <= -2.0) & (spy_off >= -0.015), [("XLU", 1.0)], 5, 2),
    "C2 XLU-XLP":       ((sp21 <= -0.04) & (z <= -1.5), [("XLU", 1.0), ("XLP", -1.0)], 5, 4),
    "C3 TLT long h5":   ((tlt_off <= 0.015) & (tnx63 >= 85), [("TLT", 1.0)], 5, 2),
    "C3 TLT long h10":  ((tlt_off <= 0.015) & (tnx63 >= 85), [("TLT", 1.0)], 10, 2),
    "C4 XLU-SPY":       ((z <= -2.0) & (spy_off >= -0.015), [("XLU", 1.0), ("SPY", -1.0)], 5, 4),
}

rows = []
for name, (m, legs, h, cost) in CAND.items():
    r = vehicle_ret(px, legs, h, 1)
    s = px.index[m.fillna(False).values & r.notna().values]
    e = declusters(s, h, px.index)
    v = r.loc[e].values
    a = summarize(v, name)
    a["n_days"] = len(s)
    a["mde_pct"] = 2.0 * a["sd_pct"] / np.sqrt(a["n"])       # effect needed for t=2
    a["edge_vs_mde"] = a["mean_pct"] / a["mde_pct"]
    a["cost_bps"] = cost
    a["edge_bps"] = 100 * a["mean_pct"]
    a["x_cost"] = a["edge_bps"] / cost
    a["boot_p_le0"] = bootstrap_p_le0(v)
    rows.append(a)
show(rows, "power: measured edge vs minimum detectable effect (t=2)")
print("\nread: |edge| < MDE means the sample cannot resolve the claim either way.")
print("every candidate's MEASURED episode mean is <= 0 except C2 (+0.05%, 1/60th of its MDE).")

# --- book regime context -----------------------------------------------------
print("\n=== live book regime on 2026-08-07 ===")
try:
    fr = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
    fr.index = pd.to_datetime(fr.index)
    ma = fr["63d"].rolling(10).mean()
    print(f"  fragility 10d-MA(63d) = {ma.iloc[-1]:.1f} (last row {fr.index[-1].date()}), "
          f"raw 21d = {fr['21d'].iloc[-1]:.1f}")
    print(f"  dial >= 50 -> FAMILY4 dip-buy band territory")
except Exception as exc:  # pragma: no cover
    print("  fragility parquet unavailable:", exc)
try:
    sys.path.insert(0, str(ROOT))
    import pc_fear
    st = pc_fear.fear_state_asof(pd.Timestamp("2026-08-07"))
    print("  pc_fear state:", st)
except Exception as exc:
    print("  pc_fear probe failed:", exc)

# --- seasonal/state sanity: how often is XLU this washed out in August? ------
sig = px.index[((z <= -2.0) & (spy_off >= -0.015)).fillna(False).values]
print("\nC1/C4 trigger days by calendar month:")
print(pd.Series(1, index=sig).groupby(sig.month).sum().to_string())
print("August trigger days in 25y:", list(sig[sig.month == 8].date))

# --- C2 sample-window dependence (the panel start date flips its sign) -------
print("\n=== C2 sign flips with the sample start date ===")
for tk, lbl in [(["SPY", "XLU", "XLP"], "2001-start panel (XLU/XLP/SPY only)"),
                (["SPY", "XLU", "XLP", "TLT", "IEF", "^TNX"], "2002-08-start panel (+TLT/IEF)")]:
    p = close_panel(tk).dropna()
    zz = zscore(p["XLU"], 10)
    ss = p["XLU"].pct_change(21) - p["XLP"].pct_change(21)
    m = (ss <= -0.04) & (zz <= -1.5)
    r = vehicle_ret(p, [("XLU", 1.0), ("XLP", -1.0)], 5, 1)
    s = p.index[m.fillna(False).values & r.notna().values]
    e = declusters(s, 5, p.index)
    a = summarize(r.loc[e].values, lbl)
    print(f"  {lbl}: N={a['n']} mean={a['mean_pct']:+.3f}% t={a['t']:+.2f} "
          f"first={s[0].date()}")
