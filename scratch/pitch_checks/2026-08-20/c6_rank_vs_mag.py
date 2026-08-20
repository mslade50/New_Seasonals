"""C6: long the dollar after a 21-day rank washout.

Attack 1 - finish the rank-vs-magnitude adjudication cleanly (registry entry).
Attack 2 - distance-from-the-extreme gradient WITHIN the rank<=2 trigger set,
           fitted at today's -2.32%, plus today's percentile inside that set.
Attack 4 - era + concentration on the rank form.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change

px = close_panel(["DX-Y.NYB", "UUP"])
d = px.index
dx = px["DX-Y.NYB"].dropna()
r21 = (dx / dx.shift(21) - 1.0)
rk21 = pct_rank(px["DX-Y.NYB"], 21).reindex(dx.index)

TODAY_MAG = float(r21.iloc[-1])
print(f"panel {dx.index[0].date()} .. {dx.index[-1].date()}  N={len(dx)}")
print(f"TODAY  21d ret {100*TODAY_MAG:+.3f}%  rank252 {rk21.iloc[-1]:.2f}  "
      f"full-hist pctile {100*(r21.dropna() < TODAY_MAG).mean():.2f}  "
      f"z vs full-sample sd {TODAY_MAG / r21.std():+.2f}")

# ------------------------------------------------------------------ 1. forms
print("\n\n######## 1. RANK FORM vs MAGNITUDE FORM (DXY spot, lag=1) ########")
forms = {
    "rank<=2 (pitched)": rk21 <= 2,
    "rank<=5": rk21 <= 5,
    "rank<=10": rk21 <= 10,
    "mag<=-2.32% (today)": r21 <= TODAY_MAG,
    "mag<=-3%": r21 <= -0.03,
    "mag<=-4%": r21 <= -0.04,
    "mag<=-5%": r21 <= -0.05,
}
rows = []
for lbl, m in forms.items():
    m = m.fillna(False)
    e = declusters(dx.index[m.values], 21, dx.index)
    for h in (3, 5, 10):
        f = (dx.shift(-(1 + h)) / dx.shift(-1) - 1.0)
        v = f.reindex(e).dropna()
        c = f.dropna()
        if len(v) < 3:
            continue
        rows.append({"form": lbl, "h": h, "days": int(m.sum()), "N_ep": len(v),
                     "mean_pct": round(100 * v.mean(), 3),
                     "hit": round(100 * (v > 0).mean(), 1),
                     "alldays_pct": round(100 * c.mean(), 3),
                     "excess_pp": round(100 * (v.mean() - c.mean()), 3),
                     "t": round(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))), 2),
                     "signp": round(sign_test(int((v > 0).sum()), len(v)), 3)})
show(rows, "rank vs magnitude, episode level (gap 21)")

# overlap: how much of the rank set IS a magnitude extreme?
mr = (rk21 <= 2).fillna(False)
mm = (r21 <= TODAY_MAG).fillna(False)
print(f"\n  overlap: rank<=2 days {int(mr.sum())}, mag<=today days {int(mm.sum())}, "
      f"both {int((mr & mm).sum())}")
print(f"  share of rank<=2 days that are ALSO mag<=today: "
      f"{100*(mr & mm).sum()/mr.sum():.1f}%")
print(f"  magnitude distribution INSIDE rank<=2 days: "
      f"min {100*r21[mr].min():+.2f}% p25 {100*r21[mr].quantile(.25):+.2f}% "
      f"median {100*r21[mr].median():+.2f}% p75 {100*r21[mr].quantile(.75):+.2f}% "
      f"max {100*r21[mr].max():+.2f}%")
print(f"  TODAY {100*TODAY_MAG:+.2f}% sits at the "
      f"{100*(r21[mr] < TODAY_MAG).mean():.1f}th percentile of that trigger set")

# ------------------------------------------------- 2. distance gradient (OLS)
print("\n\n######## 2. DOSE RESPONSE INSIDE THE rank<=2 SET ########")
epi = declusters(dx.index[mr.values], 21, dx.index)
for h in (3, 5, 10):
    f = (dx.shift(-(1 + h)) / dx.shift(-1) - 1.0)
    sub = pd.DataFrame({"mag": r21.reindex(epi), "fwd": f.reindex(epi)}).dropna()
    x = sub["mag"].values
    y = sub["fwd"].values
    b, a = np.polyfit(x, y, 1)
    fit_today = a + b * TODAY_MAG
    r = np.corrcoef(x, y)[0, 1]
    # bucket ladder: deeper half vs shallower half of the trigger set
    med = np.median(x)
    deep, shal = y[x <= med], y[x > med]
    print(f"  h={h:<3} N={len(sub)}  slope={b:+.4f} (per 1.0 of 21d ret) "
          f"corr={r:+.3f}  intercept={100*a:+.3f}%")
    print(f"        fitted at TODAY ({100*TODAY_MAG:+.2f}%) = {100*fit_today:+.3f}%   "
          f"| observed set mean {100*y.mean():+.3f}%")
    print(f"        DEEP half (<= {100*med:+.2f}%) N={len(deep)} {100*deep.mean():+.3f}% "
          f"hit {100*(deep>0).mean():.0f}%  |  SHALLOW half N={len(shal)} "
          f"{100*shal.mean():+.3f}% hit {100*(shal>0).mean():.0f}%")

# --------------------------------------------------- 4. era + concentration
print("\n\n######## 4. ERA + CONCENTRATION, rank<=2 form ########")
for h in (3, 5, 10):
    f = (dx.shift(-(1 + h)) / dx.shift(-1) - 1.0)
    v = f.reindex(epi).dropna()
    e = v.index
    print(f"\n  h={h}  N={len(v)} mean {100*v.mean():+.3f}%")
    show(era_split(e, v.values), f"   era split h={h}")
    print("   ", cluster_note(e, v.values))
    by = v.groupby(v.index.year).agg(["count", "mean", "sum"])
    by[["mean", "sum"]] = (by[["mean", "sum"]] * 100).round(3)
    print(by.to_string())

# --------------------------------------------------------------- 3. cost
print("\n\n######## 3. COST ########")
f5 = (dx.shift(-(1 + 3)) / dx.shift(-1) - 1.0).reindex(epi).dropna()
print(f"  rank<=2 h=3 episode mean {100*f5.mean():+.3f}% = {10000*f5.mean():.1f} bps")
print(f"  DX futures round trip ~1.5 bps -> {10000*f5.mean()/1.5:.1f}x   "
      f"UUP round trip ~6 bps -> {10000*f5.mean()/6:.1f}x")
print("  cost is NOT the binding constraint here; see the sections above.")
