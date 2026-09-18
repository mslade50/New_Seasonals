import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

px = close_panel(["HYG", "IEF", "SPY", "^TNX"])
px = px[px["HYG"].notna() & px["IEF"].notna() & px["SPY"].notna()]
hyg, ief = px["HYG"], px["IEF"]
ratio = hyg / ief
H = 5

ret_h = vehicle_ret(px, [("HYG", 1.0)], H)
ret_i = vehicle_ret(px, [("IEF", 1.0)], H)
ret_s = vehicle_ret(px, [("SPY", 1.0)], H)
valid = ret_h.notna() & ret_i.notna() & ret_s.notna()

# joint 5d betas (non-overlapping) HYG on IEF + SPY
r = pd.concat([hyg.pct_change(5), ief.pct_change(5), px["SPY"].pct_change(5)], axis=1).iloc[::5].dropna()
X = np.column_stack([np.ones(len(r)), r.iloc[:, 1], r.iloc[:, 2]])
b = np.linalg.lstsq(X, r.iloc[:, 0].values, rcond=None)[0]
r18 = r[r.index >= "2018"]
X18 = np.column_stack([np.ones(len(r18)), r18.iloc[:, 1], r18.iloc[:, 2]])
b18 = np.linalg.lstsq(X18, r18.iloc[:, 0].values, rcond=None)[0]
print(f"joint beta HYG ~ IEF + SPY: full IEF {b[1]:.3f} SPY {b[2]:.3f} | 2018+ IEF {b18[1]:.3f} SPY {b18[2]:.3f}")
resid = ret_h - b18[1] * ret_i - b18[2] * ret_s

hyg_r5 = pct_rank(hyg, 5); rat_r5 = pct_rank(ratio, 5)
child = (hyg_r5 <= 10) & (rat_r5 >= 80)
sig = px.index[child.reindex(px.index, fill_value=False).values & valid.values]
epi = declusters(sig, H, px.index)
tab = pd.DataFrame({"HYG": 100 * ret_h.loc[epi], "IEF": 100 * ret_i.loc[epi], "SPY": 100 * ret_s.loc[epi],
                    "resid18": 100 * resid.loc[epi], "ratio_r5": rat_r5.loc[epi], "hyg_r5": hyg_r5.loc[epi]})
print("\n=== child episodes h=5 ===")
print(tab.round(3).to_string())
v = tab["HYG"].values / 100
srt = np.sort(v)[::-1]
print(f"mean {100*v.mean():+.3f}%  drop-best1 {100*srt[1:].mean():+.3f}%  drop-best2 {100*srt[2:].mean():+.3f}%  "
      f"ex-2022 {tab.loc[tab.index.year != 2022, 'HYG'].mean():+.3f}% (n={int((tab.index.year != 2022).sum())})")
print(f"resid (HYG - {b18[1]:.2f} IEF - {b18[2]:.2f} SPY) child mean {tab['resid18'].mean():+.3f}%  "
      f"all-days resid {100*resid[valid].mean():+.3f}%  hit {100*(tab['resid18']>0).mean():.0f}%")

# neighbour grid: HYG rank lookback n x ratio lookback n x thresholds
print("\n=== neighbour grid, long HYG h=5 episodes (excess vs own all-days since 2008) ===")
ctl = 100 * ret_h[valid].mean()
rows = []
for n in (5, 10):
    hr = pct_rank(hyg, n); rr = pct_rank(ratio, n)
    for thr in (5, 10, 15):
        for rthr in (70, 80, 90):
            m = (hr <= thr) & (rr >= rthr)
            s = px.index[m.reindex(px.index, fill_value=False).values & valid.values]
            if len(s) == 0:
                rows.append({"cell": f"n{n} r<={thr} rat>={rthr}", "n": 0}); continue
            e = declusters(s, H, px.index)
            vv = ret_h.loc[e].values
            ss = np.sort(vv)[::-1]
            rows.append({"cell": f"n{n} r<={thr} rat>={rthr}", "n": len(e), "mean": 100 * vv.mean(),
                         "median": 100 * np.median(vv), "hit": 100 * (vv > 0).mean(),
                         "excess": 100 * vv.mean() - ctl,
                         "dropbest2": 100 * ss[2:].mean() if len(ss) > 2 else np.nan,
                         "ex2022": 100 * np.nanmean(vv[pd.DatetimeIndex(e).year != 2022]) if (pd.DatetimeIndex(e).year != 2022).any() else np.nan,
                         "pre18_n": int((pd.DatetimeIndex(e) < "2018").sum())})
show(rows)

# rate-driven defined directly: HYG r5<=10 and IEF r5<=10 (both falling)
print("\n=== alternative definition of 'duration-driven flush' ===")
ief_r5 = pct_rank(ief, 5)
alts = {"HYG r5<=10 & IEF r5<=10": (hyg_r5 <= 10) & (ief_r5 <= 10),
        "HYG r5<=10 & IEF r5<=20": (hyg_r5 <= 10) & (ief_r5 <= 20),
        "HYG r5<=10 & IEF r5>=50 (spread-only)": (hyg_r5 <= 10) & (ief_r5 >= 50),
        "HYG r5<=10 & rat r5>=80 & IEF r5<=10": child & (ief_r5 <= 10)}
rows = []
for lbl, m in alts.items():
    s = px.index[m.reindex(px.index, fill_value=False).values & valid.values]
    e = declusters(s, H, px.index)
    for veh, rr_ in (("HYG", ret_h), ("IEF", ret_i), ("resid", resid)):
        vv = rr_.loc[e].values
        d = pd.DatetimeIndex(e)
        rows.append({"cell": f"{lbl} | {veh}", "n": len(e), "mean": 100 * vv.mean(), "median": 100 * np.median(vv),
                     "hit": 100 * (vv > 0).mean(), "pre18": 100 * np.nanmean(vv[d < "2018"]) if (d < "2018").any() else np.nan,
                     "post18": 100 * np.nanmean(vv[d >= "2018"]) if (d >= "2018").any() else np.nan})
show(rows)

# regime splits on the child (day-level, too few episodes) and on the wider neighbour r5<=15 & rat>=70
print("\n=== regime splits (episodes) ===")
tnx = px["^TNX"]
tnx_r63 = pct_rank(tnx, 63)
wide = (hyg_r5 <= 15) & (rat_r5 >= 70)
for lbl, m in (("child", child), ("wide r5<=15 rat>=70", wide)):
    s = px.index[m.reindex(px.index, fill_value=False).values & valid.values]
    e = declusters(s, H, px.index)
    vv = ret_h.loc[e]
    d = pd.DatetimeIndex(e)
    tr = tnx_r63.reindex(d).values
    rows = [summarize(vv.values, f"{lbl} all"),
            summarize(vv[d.year % 4 == 2].values, "midterm"),
            summarize(vv[d.year % 4 != 2].values, "non-midterm"),
            summarize(vv[tr >= 80].values, "TNX r63>=80 (rising rates)"),
            summarize(vv[tr < 80].values, "TNX r63<80"),
            summarize(vv[d < "2018"].values, "pre-2018"),
            summarize(vv[d >= "2018"].values, "2018+"),
            summarize(vv[d >= "2021"].values, "2021+")]
    fl = event_in_window(d, px.index, H, 1, ("fomc_decision",))
    rows += [summarize(vv[fl].values, "FOMC in hold"), summarize(vv[~fl].values, "FOMC out")]
    show(rows, lbl)

# live TNX r63
print(f"\nlive TNX r63 {tnx_r63.dropna().iloc[-1]:.1f}  IEF r5 {ief_r5.iloc[-1]:.1f}")
