"""C10 round 2: close out the only tradeable form.

Three loose ends from round 1:
 (a) the no-earnings control's MEAN is contaminated (best +35,254%, a bad
     split in the 1120-name panel) -- redo it winsorized and on medians
 (b) the live names are all MEGA-CAP BMO reporters, the weakest cell half;
     measure that cell directly instead of the pooled one
 (c) the live instances were dropped from round 1's display table by the
     p+12 forward-window guard -- locate them in the trigger distribution
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from strategy_config import LIQUID_PLUS_COMMODITIES

pd.set_option("display.width", 250)
HS = (1, 2, 3, 5, 10)
df = pd.read_parquet(Path(__file__).parent / "_c10_events.parquet")
BEAT = df.surp > 0
cell = df[BEAT & (df.react <= -0.05)]


def wstats(sub, h, label, lo=-0.5, hi=0.5):
    x = (sub[f"f{h}"] - sub[f"s{h}"]).dropna()
    if len(x) < 3:
        return {"label": label, "n": len(x)}
    xw = x.clip(lo, hi)
    byd = sub.assign(v=(sub[f"f{h}"] - sub[f"s{h}"]).clip(lo, hi)).dropna(subset=["v"]) \
             .groupby("rdate")["v"].mean()
    return {"label": label, "n": len(x),
            "mean_pct": round(100 * x.mean(), 3),
            "winsor_mean_pct": round(100 * xw.mean(), 3),
            "median_pct": round(100 * x.median(), 3),
            "hit": round(100 * (x > 0).mean(), 1),
            "n_dates": len(byd),
            "clust_t": round(float(byd.mean() / (byd.std(ddof=1) / np.sqrt(len(byd)))), 2)}


LIQ = set(LIQUID_PLUS_COMMODITIES)
print("===== (b) THE LIVE SHAPE: liquid, BMO, 2013+, mega-cap =====")
rows = []
for nm, sub in [
    ("all triggers", cell),
    ("liquid", cell[cell.liq]),
    ("liquid & BMO", cell[cell.liq & cell.bmo]),
    ("liquid & BMO & 2013+", cell[cell.liq & cell.bmo & (cell.rdate >= "2013-01-01")]),
    ("liquid & BMO & 2018+", cell[cell.liq & cell.bmo & (cell.rdate >= "2018-01-01")]),
    ("liquid & 2018+", cell[cell.liq & (cell.rdate >= "2018-01-01")]),
]:
    for h in (1, 3, 5, 10):
        rows.append(wstats(sub, h, f"{nm} h={h}"))
show(rows, "market-relative, entry lag=1")

print("\n===== the revenue-beat leg (WMT/ROST/TGT/TJX all beat BOTH lines) =====")
both = cell[(cell.rsurp > 0)]
show([wstats(both[both.liq], h, f"liquid, EPS+ and REV+ beat, sold <=-5% h={h}")
      for h in (1, 3, 5, 10)], "")
show([wstats(both[both.liq & (both.rdate >= "2013-01-01")], h,
             f"  same, 2013+ h={h}") for h in (1, 3, 5, 10)], "")

print("\n===== (a) no-earnings control, winsorized, LIQUID names only =====")
print("  round 1's mean was unusable (best +35,254% = a bad split in the panel).")
print("  medians, cell vs control, market-relative:")
for h in (1, 3, 5, 10):
    a = wstats(cell[cell.liq], h, "")
    print(f"   h={h}: cell liquid median {a['median_pct']:+.3f}% "
          f"winsor mean {a['winsor_mean_pct']:+.3f}%  (n={a['n']}, clust_t {a['clust_t']})")

print("\n===== (c) the live instances in the trigger distribution =====")
raw = pd.read_parquet(ROOT / "data" / "master_prices.parquet")
raw["date"] = pd.to_datetime(raw["date"])
for t, rd in [("WMT", "2026-08-20"), ("ROST", "2026-08-20"), ("TJX", "2026-08-19"),
              ("TGT", "2026-08-19")]:
    g = raw[raw.ticker == t].sort_values("date").set_index("date")
    c = g["Close"]
    i = c.index.get_indexer([pd.Timestamp(rd)], method="nearest")[0]
    r = c.iloc[i] / c.iloc[i - 1] - 1.0
    print(f"  {t} {rd}: session return {100*r:+.2f}%  -> "
          f"{'QUALIFIES' if r <= -0.05 else 'does NOT qualify at -5%'}")
pp = cell.react.dropna()
for nm, x in [("WMT -9.15%", -0.0915)]:
    print(f"  {nm} sits at the {100*(pp<=x).mean():.1f}th percentile by depth of a "
          f"population whose median is {100*pp.median():.2f}%")
print(f"  and the DEEP half of that population is the losing half at h=10: "
      f"react<=-9% ATR-form -3.0xATR h=10 = -0.164% (round 1 table)")

print("\n===== summary line =====")
a = wstats(cell[cell.liq & (cell.rdate >= '2013-01-01')], 5, "")
print(f" tradeable form (liquid, 2013+, h=5): n={a['n']} over {a['n_dates']} dates, "
      f"{a['mean_pct']:+.3f}% market-relative, clustered t {a['clust_t']}, "
      f"= {a['mean_pct']*100:+.1f} bps against a 10 bps single-name round trip.")
