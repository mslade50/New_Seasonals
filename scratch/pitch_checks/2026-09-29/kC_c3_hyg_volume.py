"""kC C3 round 1: long HYG across the Q4 turn after a down day on >= 2.5x its
63-day volume. Gate attribution: volume spike vs plain 5d washout vs plain
calendar window. Volume ratio matches build_pitch_state (today / rolling-63
mean INCLUDING today). Adjusted closes: month-turn HYG returns are total returns.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
P = load_prices(["HYG", "IEF", "LQD", "SPY"])
px = close_panel(["HYG", "IEF", "LQD", "SPY"])
px = px[px["HYG"].notna()].copy()
idx = px.index
vol = P["HYG"]["Volume"].reindex(idx)
vr = vol / vol.rolling(63).mean()
r1 = px["HYG"].pct_change(fill_method=None)
dn = r1 < 0
r5 = pct_rank(px["HYG"], 5)
z10 = zscore(px["HYG"], 10)
lo252 = rolling_on_valid(px["HYG"], lambda x: x.rolling(252).min())
d_lo = px["HYG"] / lo252 - 1

# sessions remaining to the month's last session (0 = month-end close)
ym = idx.year * 100 + idx.month
g = pd.Series(1, index=idx).groupby(ym)
to_me = (g.transform("size") - g.cumcount() - 1).astype(int)
cur = ym == 202609
to_me[cur] = to_me[cur] + 2  # Sept 2026 still has 09-29, 09-30 to print
isq = pd.Series(np.isin(idx.month, [3, 6, 9, 12]), index=idx)
issep = pd.Series(idx.month == 9, index=idx)

d0 = idx[-1]
print(f"last bar {d0.date()}: HYG r1 {100*r1[d0]:+.2f}%  vr {vr[d0]:.2f}x  r5 {r5[d0]:.1f}  "
      f"z10 {z10[d0]:.2f}  over 252lo {100*d_lo[d0]:.2f}%  to_me {to_me[d0]}")
print(f"HYG history {idx[0].date()} .. {d0.date()}; vr defined from {vr.first_valid_index().date()}")

spike = dn & (vr >= 2.5)
print(f"\nvr>=2.5 days: {(vr >= 2.5).sum()} (down {spike.sum()}, up/flat {((vr >= 2.5) & ~dn).sum()})")
# does volume cluster at the month turn?
tm = to_me.clip(upper=6)
print("share of vr>=2.5 days by sessions-to-ME (6 = 6+), vs share of all days:")
a = pd.DataFrame({"spike": (vr >= 2.5)[vr.notna()].groupby(tm[vr.notna()]).mean() * 100,
                  "all_days_share": tm[vr.notna()].value_counts(normalize=True).sort_index() * 100})
print(a.round(2).to_string())
print("mean vr by sessions-to-ME:", vr.groupby(tm).mean().round(2).to_dict())

me_win = to_me.isin([1, 2, 3])   # signal ME-3..ME-1, hold spans the turn
cells = {
    "ALL DAYS": pd.Series(True, index=idx),
    "A spike any date": spike,
    "A2 vr>=2.5 any dir": vr >= 2.5,
    "B spike & ME-3..-1": spike & me_win,
    "C spike & QE-3..-1": spike & me_win & isq,
    "C' spike & nonQE ME-3..-1": spike & me_win & ~isq,
    "D spike & ME-2 exact": spike & (to_me == 2),
    "D' spike & QE-2 exact": spike & (to_me == 2) & isq,
    "E cal ME-2 all": to_me == 2,
    "E cal QE-2 all": (to_me == 2) & isq,
    "E cal SepQE-2": (to_me == 2) & issep,
    "E cal ME-2 & down": (to_me == 2) & dn,
    "F wash r5<=5 any": r5 <= 5,
    "F wash r5<=5 & ME-2": (r5 <= 5) & (to_me == 2),
    "F wash r5<=5 & ME-3..-1": (r5 <= 5) & me_win,
    "G spike & r5<=5": spike & (r5 <= 5),
    "G' spike & r5>5": spike & (r5 > 5),
    "H spike & within2% 252lo": spike & (d_lo <= 0.02),
    "LIVE spike&QE-2&r5<=5": spike & (to_me == 2) & isq & (r5 <= 5),
}
rows = []
for h in (1, 2, 3, 5, 10):
    ret = fwd_lag(px["HYG"], h)
    base = ret.dropna().mean()
    for lbl, m in cells.items():
        m = m.reindex(idx, fill_value=False) & ret.notna()
        dts = idx[m.values]
        if lbl == "ALL DAYS":
            v = ret[m].values
        else:
            dts = declusters(dts, h, idx)
            v = ret.loc[dts].values
        if len(v) == 0:
            rows.append({"h": h, "cell": lbl, "n": 0})
            continue
        w = int((v > 0).sum())
        rows.append({"h": h, "cell": lbl, "n": len(v), "mean": round(100 * v.mean(), 3),
                     "xs": round(100 * (v.mean() - base), 3), "rec": f"{w}-{len(v)-w}",
                     "sign_p": round(sign_test(w, len(v)), 4), "med": round(100 * np.median(v), 3),
                     "worst": round(100 * v.min(), 2)})
print("\n=== episodes (declustered at h), long HYG lag=1, % ===")
print(pd.DataFrame(rows).to_string(index=False))

# listing: the month-turn spike episodes
h = 5
ret5 = fwd_lag(px["HYG"], 5)
ret3 = fwd_lag(px["HYG"], 3)
dts = idx[(spike & me_win).values]
print("\nspike & ME-3..-1 days (all, not declustered):")
print(pd.DataFrame({"to_me": to_me[dts], "r1%": (100 * r1[dts]).round(2), "vr": vr[dts].round(2),
                    "r5": r5[dts].round(1), "h3%": (100 * ret3[dts]).round(2),
                    "h5%": (100 * ret5[dts]).round(2)}).to_string())

# round-1 batteries
nf = ("nfp",)
battery(px, spike, [("HYG", 1.0)], 5, "C3-A spike any date, long HYG", 2.0,
        variants={"vr>=2.0 dn": dn & (vr >= 2.0), "vr>=3.0 dn": dn & (vr >= 3.0),
                  "vr>=2.5 any dir": vr >= 2.5, "r5<=5 washout alone": r5 <= 5},
        event_kinds=nf)
battery(px, spike & me_win, [("HYG", 1.0)], 5, "C3-B spike & ME-3..-1, long HYG", 2.0,
        variants={"QE only": spike & me_win & isq, "non-QE": spike & me_win & ~isq,
                  "ME-2 exact": spike & (to_me == 2), "cal ME-3..-1 no spike": me_win,
                  "cal ME-2 no spike": to_me == 2},
        event_kinds=nf)
battery(px, spike & me_win, [("HYG", 1.0)], 3, "C3-B spike & ME-3..-1, long HYG", 2.0,
        event_kinds=nf)
