"""C2 round 2: the h=2 cell is the only live one, so attack it specifically.

(a) where inside the hold does the return land, (b) the 1.0-1.5% thrust band
that the parent definition contains and the child excludes, (c) episode-year
drops, (d) tdom + month-of-year controls, (e) the duration translation to IEF,
(f) a rotation permutation priced on the grid actually walked.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

pd.set_option("display.width", 220)

raw = load_prices(["TLT", "IEF", "^TNX"])
tlt = raw["TLT"]["Close"].dropna()
d = tlt.index
px = pd.DataFrame({"TLT": tlt, "IEF": raw["IEF"]["Close"].reindex(d),
                   "TNX": raw["^TNX"]["Close"].reindex(d)})
SHORT = [("TLT", -1.0)]
d1 = tlt.pct_change(fill_method=None)
dist = tlt / tlt.rolling(252).min() - 1.0

def cell(thr, lo):
    return ((d1 >= thr) & (dist <= lo)).fillna(False).astype(bool)

m = cell(0.015, 0.04)
epi = declusters(d[m], 10, d)

# ------------------------------------------------------------- a) where the return lands
print("===== a) per-session decomposition of the h=2 hold (entry = close D+1) =====")
paths = episode_paths(px, epi, SHORT, 4)
print((paths * 100).round(3).to_string())
inc = paths.diff(axis=1)
inc[1] = paths[1]
print("\n per-session increments, mean pp / hit%:")
for k in paths.columns:
    v = inc[k].dropna().values
    print(f"  session +{k}: mean {100*v.mean():+.3f}pp  hit {(v>0).mean()*100:.1f}%  "
          f"N={len(v)}  t={v.mean()/(v.std(ddof=1)/np.sqrt(len(v))):+.2f}")

# ------------------------------------------------------------- b) the excluded band
print("\n\n===== b) the 1.0-1.5% thrust band the parent contains and the child drops =====")
for h in (2, 3):
    ret = vehicle_ret(px, SHORT, h)
    valid = ret.dropna().index
    for lbl, mm in [("thrust >= 1.5% (the cell)", cell(0.015, 0.04)),
                    ("thrust in [1.0%, 1.5%)", ((d1 >= 0.010) & (d1 < 0.015) & (dist <= 0.04)).fillna(False).astype(bool)),
                    ("thrust >= 1.0% (the PARENT)", cell(0.010, 0.04)),
                    ("thrust in [1.25%, 1.5%)", ((d1 >= 0.0125) & (d1 < 0.015) & (dist <= 0.04)).fillna(False).astype(bool)),
                    ("thrust >= 1.75%", cell(0.0175, 0.04)),
                    ("thrust >= 2.0%", cell(0.020, 0.04))]:
        e = declusters(pd.DatetimeIndex(d[mm]).intersection(valid), 10, valid)
        v = ret.loc[e].values
        if len(v) == 0:
            continue
        print(f" h={h}  {lbl:<28} N={len(v):>3} mean {100*v.mean():+.3f}% "
              f"hit {(v>0).mean()*100:.0f}% signp {sign_test(int((v>0).sum()), len(v)):.4f}")

# ------------------------------------------------------------- c) year drops
print("\n\n===== c) episode-year concentration =====")
for h in (2,):
    ret = vehicle_ret(px, SHORT, h)
    e = pd.DatetimeIndex(epi).intersection(ret.dropna().index)
    v = ret.loc[e]
    by = v.groupby(v.index.year).agg(["count", "mean", "sum"])
    by[["mean", "sum"]] = (by[["mean", "sum"]] * 100).round(3)
    print(by.to_string())
    for drop in ([2022], [2022, 2023], [2021, 2022, 2023], [2013], [2013, 2022]):
        k = v[~v.index.year.isin(drop)]
        w = int((k > 0).sum())
        t = k.mean() / (k.std(ddof=1) / np.sqrt(len(k))) if len(k) > 1 else np.nan
        print(f"   ex-{drop}: N={len(k)} mean {100*k.mean():+.3f}% record {w}-{len(k)-w} "
              f"t={t:+.2f} signp {sign_test(w, len(k)):.4f}")
    # LOYO
    yrs = sorted(set(v.index.year))
    loyo = []
    for y in yrs:
        k = v[v.index.year != y]
        loyo.append((y, 100 * k.mean()))
    print("   LOYO means:", {y: round(mm, 3) for y, mm in loyo})
    print(f"   LOYO floor: {min(mm for _, mm in loyo):+.3f}%")

# ------------------------------------------------------------- d) tdom / month controls
print("\n\n===== d) trading-day-of-month and month-of-year controls (2026-08-10 / 08-13 traps) =====")
tdom = pd.Series(index=d, dtype=float)
for (y, mo), g in pd.Series(range(len(d)), index=d).groupby([d.year, d.month]):
    tdom.loc[g.index] = np.arange(1, len(g) + 1)
ret2 = vehicle_ret(px, SHORT, 2)
valid = ret2.dropna().index
tri = pd.DatetimeIndex(epi).intersection(valid)
print(f" trigger tdoms: {sorted(tdom.reindex(tri).astype(int).tolist())}")
print(f" trigger months: {sorted(tri.month.tolist())}")
# tdom-matched control
ctrl = []
for t0 in tri:
    same = valid[(tdom.reindex(valid) == tdom.loc[t0]).values]
    ctrl.append(ret2.loc[same].mean())
ctrl = np.array(ctrl)
print(f" tdom-matched control mean {100*np.nanmean(ctrl):+.3f}%  -> tdom-matched excess "
      f"{100*(ret2.loc[tri].mean() - np.nanmean(ctrl)):+.3f}pp")
# month-matched
ctrl_m = []
for t0 in tri:
    same = valid[valid.month == t0.month]
    ctrl_m.append(ret2.loc[same].mean())
print(f" month-matched control mean {100*np.nanmean(ctrl_m):+.3f}%  -> month-matched excess "
      f"{100*(ret2.loc[tri].mean() - np.nanmean(ctrl_m)):+.3f}pp")
byday = ret2.loc[tri]
print(f" trigger weekday histogram (entry is D+1): "
      f"{pd.Series((tri + pd.Timedelta(days=0)).dayofweek).value_counts().sort_index().to_dict()}")

# ------------------------------------------------------------- e) duration translation
print("\n\n===== e) duration translation: does the SAME cell exist in IEF? =====")
for h in (2, 3):
    for tkr in ("TLT", "IEF"):
        ret = vehicle_ret(px, [(tkr, -1.0)], h)
        e = pd.DatetimeIndex(epi).intersection(ret.dropna().index)
        v = ret.loc[e].values
        base = ret.dropna()
        print(f" h={h} short {tkr}: N={len(v)} mean {100*v.mean():+.3f}% "
              f"hit {(v>0).mean()*100:.0f}%  own drift {100*base.mean():+.3f}%  "
              f"excess {100*(v.mean()-base.mean()):+.3f}pp")
    # ratio vs sd ratio
    sd_ratio = (tlt.pct_change(fill_method=None).std() /
                px["IEF"].pct_change(fill_method=None).std())
    print(f"    TLT/IEF daily-sd ratio {sd_ratio:.2f}")

# ------------------------------------------------------------- f) rotation permutation
print("\n\n===== f) rotation permutation over the grid actually walked =====")
# grid: 2 signs x 5 horizons x 9 definitions = 90 cells
THR = [0.010, 0.0125, 0.015, 0.0175, 0.020]
LOW = [0.02, 0.03, 0.04, 0.06, 0.08]
HS = [1, 2, 3, 5, 10]
masks = {}
for th in THR:
    for lo in LOW:
        masks[(th, lo)] = cell(th, lo)

def grid_max_t(shift):
    """Rotate the RETURN series by `shift` sessions, keeping masks fixed."""
    best = 0.0
    for h in HS:
        ret = vehicle_ret(px, SHORT, h)
        arr = ret.values.copy()
        arr = np.roll(arr, shift)
        rr = pd.Series(arr, index=d)
        valid = rr.dropna().index
        for key, mm in masks.items():
            e = declusters(pd.DatetimeIndex(d[mm]).intersection(valid), 10, valid)
            if len(e) < 8:
                continue
            v = rr.loc[e].values
            sd = v.std(ddof=1)
            if sd == 0:
                continue
            t = abs(v.mean() / (sd / np.sqrt(len(v))))   # both signs => |t|
            best = max(best, t)
    return best

obs = grid_max_t(0)
rng = np.random.default_rng(42)
shifts = rng.integers(60, len(d) - 60, size=120)
null = np.array([grid_max_t(int(s)) for s in shifts])
print(f" observed grid max |t| = {obs:.2f}")
print(f" null grid max |t|: mean {null.mean():.2f}, p90 {np.percentile(null,90):.2f}, "
      f"max {null.max():.2f}")
print(f" P(grid max |t| >= observed) = {(null >= obs).mean():.3f}  (120 rotations, 125-cell grid)")
