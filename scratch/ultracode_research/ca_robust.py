"""Crisis-alpha track: robustness checks on the finalist overlays.

1. Leave-one-episode-out totals (does one crash pay for everything?)
2. Entry-lag sensitivity: act t+1 (base) vs t+2.
3. Monthly-clustered t-stats on sleeve returns (all months, hi-frag months).
4. Always-on VXX-proxy bleed over the SAME 2016-07+ window.
5. Hysteresis 0 vs 5.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
NAV = 750_000.0

import ca_overlays as ca  # reuse builders (runs its __main__ only when executed directly)

frag = ca.frag
sleeves = pd.read_parquet(HERE / "ca_sleeves.parquet")

# episodes (same construction as ca_episodes)
THR_ON, THR_OFF = 55, 50
episodes = []
in_ep = False
for d, f in frag.items():
    if not in_ep and f >= THR_ON:
        in_ep, start, peak = True, d, f
    elif in_ep:
        peak = max(peak, f)
        if f < THR_OFF:
            episodes.append((start, d, peak))
            in_ep = False
if in_ep:
    episodes.append((start, frag.index[-1], peak))

print("=== 1. Leave-one-episode-out (sleeve $ total, 2016-07+) ===")
for col in ["vxxp5_55", "put_55", "putspread_55"]:
    tot = sleeves[col].sum() * NAV
    print(f"\n{col}: full total ${tot:+,.0f}")
    for (s, e, p) in episodes:
        ep_pnl = sleeves[col].loc[s:e].sum() * NAV
        if abs(ep_pnl) > 2000:
            print(f"  drop {s.date()}..{e.date()} (peak {p:.0f}, ep ${ep_pnl:+,.0f}) -> ${tot-ep_pnl:+,.0f}")

print("\n=== 2. Entry-lag sensitivity (VXX-proxy 5% NAV thr55) ===")
def lagged_sleeve(lag):
    a_ret = ca.ret["VXXP"].dropna()
    idx = a_ret.index.intersection(frag.index)
    f = frag.reindex(idx).ffill()
    raw_on = pd.Series(np.nan, index=idx)
    raw_on[f >= 55] = 1.0
    raw_on[f < 50] = 0.0
    gate = raw_on.ffill().fillna(0.0)
    pos = gate.shift(lag).fillna(0.0)
    a = a_ret.reindex(pos.index).fillna(0.0)
    sleeve = pos * a * 0.05
    turn = pos.diff().abs().fillna(pos.abs()) * 0.05
    return sleeve - turn * 10 / 1e4

for lag in (1, 2, 3):
    s = lagged_sleeve(lag)
    print(f"  lag t+{lag}: total ${s.sum()*NAV:+,.0f}")

print("\n=== 3. Monthly-clustered t-stats (sleeve monthly returns) ===")
fm = frag.groupby(frag.index.to_period("M")).mean()
hi_m = fm[fm >= 50].index
for col in ["vxxp2_55", "vxxp5_55", "put_55", "putspread_55"]:
    mon = sleeves[col].groupby(sleeves[col].index.to_period("M")).sum().dropna()
    mon = mon[(mon.index >= pd.Period("2016-08")) & (mon.index <= pd.Period("2026-06"))]
    t_all = stats.ttest_1samp(mon, 0)
    hi = mon.reindex(hi_m.intersection(mon.index)).dropna()
    t_hi = stats.ttest_1samp(hi, 0) if len(hi) > 3 else None
    print(f"  {col:<14} all: mean {mon.mean()*100:+.3f}%/mo t={t_all.statistic:+.2f} p={t_all.pvalue:.3f} (N={len(mon)})"
          f" | hiFrag: mean {hi.mean()*100:+.3f}%/mo t={t_hi.statistic:+.2f} p={t_hi.pvalue:.3f} (N={len(hi)})")

print("\n=== 4. Always-on VXX-proxy over 2016-07+ (same window as tactical) ===")
ao = ca.ret["VXXP"].dropna()
ao = ao[ao.index >= frag.index.min()]
eq = (1 + ao * 0.05).cumprod()  # constant-rebalanced 5% NAV
print(f"  5% NAV constant: total drag {(eq.iloc[-1]-1)*100:+.1f}% of NAV over "
      f"{len(ao)/252:.1f}y = {((eq.iloc[-1])**(252/len(ao))-1)*100:+.2f}%/yr of NAV")
gate_days = (frag >= 55).sum()
print(f"  tactical was in-market {gate_days} of {len(frag)} days ({gate_days/len(frag)*100:.0f}%)")

print("\n=== 5. Hysteresis sensitivity (VXX-proxy 5% thr55) ===")
for hyst in (0, 5, 10):
    s = ca.tactical_sleeve("VXXP", 55, 0.05, 10, hysteresis=hyst)
    print(f"  hysteresis {hyst}: total ${s.sum()*NAV:+,.0f}")

print("\n=== 6. VIX level at gate-on (was vol already expensive?) ===")
vix = ca.close["^VIX"].reindex(frag.index).ffill()
for (s, e, p) in episodes:
    print(f"  {s.date()} gate-on VIX={vix.loc[s]:.1f}  peak-frag {p:.0f}")
