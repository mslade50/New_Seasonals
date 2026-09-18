"""c2 round 1: SHORT SPY from a monthly opex close when ^VIX fell >= 10% over
the three sessions into expiry (vanna/charm tailwind ends at expiry), h=1..5.

Trigger forms (anchor = opex date, entry MOC on the opex close, lag=0):
  A tradeable : ^VIX close(opex-1)/close(opex-4) - 1 <= -10%   (known before the MOC)
  B definition: ^VIX close(opex)/close(opex-3) - 1 <= -10%     (VIX settles 16:15, after
                the 16:00 MOC -- NOT exactly tradeable; reported for contrast)
  C one-day   : any one-day ^VIX change <= -10% on opex-3..opex-1
Live 2026-09-18: A = 09-14 17.10 -> 09-17 15.44 = -9.71% (misses by 0.29pp);
B needs today's ^VIX <= 15.48; C live (-12.82% on 09-17).
Mechanism honesty: data/option_positioning_history.parquet starts 2026-08-05,
so dealer vanna/charm exposure is NOT measurable historically. Only the VIX
proxy is.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pitch_lab import *  # noqa
from kA_common import *  # noqa
import numpy as np
import pandas as pd

px = build_panel()
cal = px.index
vix = px["^VIX"]
opex = pd.DatetimeIndex(sorted(set(load_events(["opex"])["date"]) & set(cal)))
opex = opex[opex < pd.Timestamp("2026-09-18")]
COST = 3.0
v3_lag = vix.shift(1) / vix.shift(4) - 1       # on date X: change opex-4 -> opex-1
v3_def = vix / vix.shift(3) - 1                 # change X-3 -> X
v1 = vix / vix.shift(1) - 1
min1d_prior3 = v1.shift(1).rolling(3).min()     # min one-day change on X-3..X-1
live_A = vix.loc["2026-09-17"] / vix.loc["2026-09-14"] - 1
print(f"live A (09-14 -> 09-17) {100*live_A:+.2f}%   B needs ^VIX(09-18) <= {0.9*vix.loc['2026-09-15']:.2f}")

try:
    pos_hist = pd.read_parquet(Path(__file__).resolve().parents[3] / "data" / "option_positioning_history.parquet")
    dc = [c for c in pos_hist.columns if "date" in c.lower()]
    print("option_positioning_history rows", len(pos_hist), "date cols", dc,
          (pos_hist[dc[0]].min(), pos_hist[dc[0]].max()) if dc else pos_hist.index[[0, -1]].tolist())
except Exception as e:
    print("positioning file read:", e)


def sshort(h):
    return -vehicle_ret(px, [("SPY", 1.0)], h, lag=0)   # short SPY, entry close X, exit X+h


is_opex = pd.Series(cal.isin(opex), index=cal)
mid = pd.Series(cal.year % 4 == 2, index=cal)

for h in (1, 2, 3, 5):
    r = sshort(h)
    ok = r.notna()
    rows = []
    for lbl, m in [("A opex & VIX3 lag <= -10%", is_opex & (v3_lag <= -0.10)),
                   ("B opex & VIX3 def <= -10% (not tradeable)", is_opex & (v3_def <= -0.10)),
                   ("C opex & 1d crush in opex-3..-1", is_opex & (min1d_prior3 <= -0.10)),
                   ("A complement: opex & VIX3 lag > -10%", is_opex & (v3_lag > -0.10)),
                   ("CTRL all opex (unconditioned post-opex)", is_opex),
                   ("PLACEBO A non-opex same crush", ~is_opex & (v3_lag <= -0.10)),
                   ("PLACEBO C non-opex 1d crush in prior 3", ~is_opex & (min1d_prior3 <= -0.10)),
                   ("CTRL all days short SPY", pd.Series(True, index=cal)),
                   ("A September", is_opex & (v3_lag <= -0.10) & pd.Series(cal.month == 9, index=cal)),
                   ("CTRL September opex", is_opex & pd.Series(cal.month == 9, index=cal)),
                   ("A midterm", is_opex & (v3_lag <= -0.10) & mid),
                   ("A non-midterm", is_opex & (v3_lag <= -0.10) & ~mid)]:
        mm = (m & ok).reindex(cal, fill_value=False)
        d = cal[mm.values]
        if "PLACEBO" in lbl:
            d = declusters(d, h, cal)
        rows.append(rec_row(r.loc[d].values, lbl, COST))
    show(rows, f"1. short SPY from the opex close, h={h}")

# neighbours (threshold / window) on the tradeable form
print("\n=== 2. definition neighbours, tradeable lag form, short SPY ===")
rows = []
for win in (2, 3, 4, 5):
    vw = vix.shift(1) / vix.shift(1 + win) - 1
    for thr in (-0.08, -0.10, -0.12, -0.15):
        for h in (1, 3, 5):
            r = sshort(h)
            d = cal[(is_opex & (vw <= thr) & r.notna()).values]
            x = rec_row(r.loc[d].values, f"win{win} thr{int(thr*100)} h{h}", COST)
            rows.append({k: x.get(k) for k in ("label", "n", "mean_pct", "hit", "t", "rec", "sign_p", "x_cost")})
print(pd.DataFrame(rows).round(3).to_string(index=False))

# offset ladder, A definition relocated to anchor opex+k
print("\n=== 3. placebo offset ladder: anchor = opex + k, VIX3 (lag form) <= -10% at the anchor ===")
for h in (1, 3, 5):
    r = sshort(h)
    rows = []
    for k in range(-5, 6):
        p = cal.get_indexer(opex) + k
        p = p[(p >= 0) & (p < len(cal))]
        a = cal[p]
        a = a[(v3_lag.reindex(a) <= -0.10).values]
        v = r.reindex(a).dropna().values
        w = int((v > 0).sum())
        rows.append({"k": k, "n": len(v), "mean_pct": 100 * v.mean() if len(v) else np.nan,
                     "hit": 100 * (v > 0).mean() if len(v) else np.nan, "rec": f"{w}-{len(v)-w}"})
    df = pd.DataFrame(rows)
    df["rank"] = df["mean_pct"].rank(ascending=False).astype(int)
    print(f"h={h}")
    print(df.round(3).to_string(index=False))
    print(f"  TRUE k=0 ranks {int(df.loc[df.k == 0, 'rank'].iloc[0])} of {len(df)}")

# era + concentration for A at h=1,3,5
print("\n=== 4. A: era, concentration, dates ===")
for h in (1, 3, 5):
    r = sshort(h)
    d = cal[(is_opex & (v3_lag <= -0.10) & r.notna()).values]
    v = r.loc[d]
    show([rec_row(v[v.index < "2018-01-01"].values, f"h{h} pre-2018", COST),
          rec_row(v[v.index >= "2018-01-01"].values, f"h{h} 2018+", COST)])
    print("  ", signed_concentration(v.index, v.values))
r1, r3, r5 = sshort(1), sshort(3), sshort(5)
d = cal[(is_opex & (v3_lag <= -0.10)).values]
for x in d:
    print(f"   {x.date()} VIX3lag {100*v3_lag[x]:+.1f}% VIX {vix[x]:.1f}  shortSPY h1 {100*r1.get(x, np.nan):+.2f}% "
          f"h3 {100*r3.get(x, np.nan):+.2f}% h5 {100*r5.get(x, np.nan):+.2f}%")
