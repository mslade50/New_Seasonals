"""C6 addendum for the registry entry: isolate WHY the rank form and the
magnitude form disagree in sign.

A rank extreme in a quiet year and a magnitude extreme select different
populations. Measure the population difference directly (realised vol of the
21-day return at trigger time), then cut the rank set by magnitude to see
which half carries the positive sign, and cut the magnitude set by rank.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

px = close_panel(["DX-Y.NYB"])
dx = px["DX-Y.NYB"].dropna()
r21 = (dx / dx.shift(21) - 1.0)
rk21 = pct_rank(px["DX-Y.NYB"], 21).reindex(dx.index)
vol252 = r21.rolling(252).std()          # trailing-year sd of the 21d return
TODAY = float(r21.iloc[-1])

print(f"TODAY 21d ret {100*TODAY:+.3f}%  rank {rk21.iloc[-1]:.2f}  "
      f"trailing-252 sd of 21d ret {100*vol252.iloc[-1]:.3f}%  "
      f"(full-sample sd {100*r21.std():.3f}%)")
print(f"  today's move in trailing-year sd units: {TODAY/vol252.iloc[-1]:+.2f}")
print(f"  today's trailing-year sd percentile: "
      f"{100*(vol252.dropna() < vol252.iloc[-1]).mean():.1f} "
      f"-> the dollar year has been QUIET, which is what makes rank 0.8 cheap")

mr = (rk21 <= 2).fillna(False)
mm = (r21 <= TODAY).fillna(False)
print("\n=== population difference between the two forms ===")
for lbl, m in [("rank<=2", mr), ("mag<=-2.32%", mm), ("mag<=-4%", (r21 <= -.04).fillna(False))]:
    print(f"  {lbl:<14} N_days={int(m.sum()):>4}  "
          f"median 21d move {100*r21[m].median():+.2f}%  "
          f"median trailing-yr sd {100*vol252[m].median():.2f}%  "
          f"median |move|/sd {(r21[m]/vol252[m]).median():+.2f}")
print(f"  TODAY                    move {100*TODAY:+.2f}%  "
      f"trailing-yr sd {100*vol252.iloc[-1]:.2f}%  move/sd {TODAY/vol252.iloc[-1]:+.2f}")


def cell(mask, h, lbl):
    m = mask.fillna(False)
    e = declusters(dx.index[m.values], 21, dx.index)
    f = (dx.shift(-(1 + h)) / dx.shift(-1) - 1.0)
    v = f.reindex(e).dropna()
    c = f.dropna()
    if len(v) < 4:
        return None
    return {"cell": lbl, "h": h, "N": len(v),
            "mean_pct": round(100 * v.mean(), 3),
            "excess_pp": round(100 * (v.mean() - c.mean()), 3),
            "hit": round(100 * (v > 0).mean(), 1),
            "signp": round(sign_test(int((v > 0).sum()), len(v)), 3),
            "t": round(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))), 2)}


print("\n=== 2x2: rank gate x magnitude gate ===")
rows = []
cells = {
    "rank<=2  &  mag<=-4%  (both extreme)": mr & (r21 <= -.04),
    "rank<=2  &  mag>-4%   (rank only, today's kind)": mr & (r21 > -.04),
    "rank>2   &  mag<=-4%  (magnitude only)": (~mr) & (r21 <= -.04),
    "rank<=2 ALL": mr,
    "mag<=-4% ALL": (r21 <= -.04),
}
for lbl, m in cells.items():
    for h in (3, 5, 10):
        r = cell(m, h, lbl)
        if r:
            rows.append(r)
show(rows, "which half of the rank set carries the positive sign?")

print("\n=== the quiet-year conditioner, stated directly ===")
quiet = vol252 <= vol252.quantile(0.35)
rows = []
for lbl, m in [("rank<=2 in a QUIET dollar year", mr & quiet),
               ("rank<=2 in a LOUD dollar year", mr & ~quiet)]:
    for h in (3, 5, 10):
        r = cell(m, h, lbl)
        if r:
            rows.append(r)
show(rows, "rank<=2 split by the trailing-year sd of the 21d return")
print(f"  today's trailing-yr sd {100*vol252.iloc[-1]:.2f}% vs the 35th-pctile "
      f"cut of {100*vol252.quantile(0.35):.2f}% -> today is "
      f"{'QUIET' if vol252.iloc[-1] <= vol252.quantile(0.35) else 'LOUD'}")
