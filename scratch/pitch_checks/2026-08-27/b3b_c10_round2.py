"""C10 round 2 — the pooled cell's era/midterm behaviour at h=5 (where the
pool looked strongest), TJX's own concentration, and TJX's rank inside the
reference class at h=5 as well as h=10.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

ASOF = pd.Timestamp("2026-08-26")
RNG = np.random.default_rng(23)
Z_MAX, R21_MAX, LOW_PCT, MIN_HIST = -2.0, 2.0, 2.0, 1500

tape = sorted(json.load(open(ROOT / "data/pitch_tape.json"))["tickers"])
frames = load_prices(tape)
built = {}
for t in tape:
    if t not in frames:
        continue
    s = frames[t]["Close"].dropna()
    s = s[s.index <= ASOF]
    if len(s) < MIN_HIST:
        continue
    z, r21 = zscore(s, 10), pct_rank(s, 21)
    dl = 100 * (s / s.rolling(252).min() - 1.0)
    built[t] = {"s": s, "mask": ((z <= Z_MAX) & (r21 <= R21_MAX) & (dl <= LOW_PCT)).fillna(False)}
print(f"names {len(built)}")

for H in (5, 10):
    vals, dts, names, per = [], [], [], {}
    for t, b in built.items():
        s = b["s"]
        f = fwd_lag(s, H, 1)
        valid = f.dropna().index
        trig = pd.DatetimeIndex(s.index[b["mask"].reindex(s.index, fill_value=False).values]).intersection(valid)
        if len(trig) == 0:
            continue
        epi = declusters(trig, H, valid)
        ex = (f.loc[epi] - f.loc[valid].mean()).values
        per[t] = {"v": f.loc[epi].values, "ex": ex, "f": f.loc[valid],
                  "n": len(epi)}
        vals.extend(ex.tolist()); dts.extend(list(epi)); names.extend([t] * len(epi))
    ex = np.array(vals); dts = pd.DatetimeIndex(dts)
    print(f"\n{'='*70}\nPOOLED name-matched EXCESS, h={H}  (N={len(ex)} episodes, "
          f"{len(per)} names)\n{'='*70}")
    print(f"  mean {100*ex.mean():+.3f}pp  t {ex.mean()/(ex.std(ddof=1)/np.sqrt(len(ex))):+.2f}  "
          f"hit {100*(ex>0).mean():.1f}%  median {100*np.median(ex):+.3f}pp")
    for lbl, m in [("pre-2018", dts < pd.Timestamp("2018-01-01")),
                   ("2018+", dts >= pd.Timestamp("2018-01-01"))]:
        m = np.asarray(m)
        print(f"    {lbl:<10} N={int(m.sum()):>4}  mean {100*ex[m].mean():+7.3f}pp  "
              f"t {ex[m].mean()/(ex[m].std(ddof=1)/np.sqrt(m.sum())):+6.2f}  "
              f"hit {100*(ex[m]>0).mean():5.1f}%")
    mid = np.array([d.year % 4 == 2 for d in dts])
    for lbl, m in [("midterm", mid), ("non-midterm", ~mid)]:
        print(f"    {lbl:<12} N={int(m.sum()):>4}  mean {100*ex[m].mean():+7.3f}pp  "
              f"t {ex[m].mean()/(ex[m].std(ddof=1)/np.sqrt(m.sum())):+6.2f}")
    # crisis attribution
    cri = np.array([d.year in (2008, 2009, 2020) for d in dts])
    print(f"    2008/09/20 crisis years  N={int(cri.sum()):>4}  mean {100*ex[cri].mean():+7.3f}pp | "
          f"ex-crisis N={int((~cri).sum()):>4}  mean {100*ex[~cri].mean():+7.3f}pp  "
          f"t {ex[~cri].mean()/(ex[~cri].std(ddof=1)/np.sqrt((~cri).sum())):+.2f}")
    # 2018+ AND ex-crisis: the modern, non-crisis cell
    mod = np.asarray(dts >= pd.Timestamp("2018-01-01")) & ~cri
    print(f"    2018+ AND ex-crisis      N={int(mod.sum()):>4}  mean {100*ex[mod].mean():+7.3f}pp  "
          f"t {ex[mod].mean()/(ex[mod].std(ddof=1)/np.sqrt(mod.sum())):+.2f}")

    # TJX rank in the class
    M = {t: 100 * d["ex"].mean() for t, d in per.items() if d["n"] >= 5}
    S = pd.Series(M).sort_values(ascending=False)
    if "TJX" in S.index:
        print(f"  TJX excess {S['TJX']:+.3f}pp (n={per['TJX']['n']}) -> rank "
              f"{int((S >= S['TJX']).sum())} of {len(S)};  class median {S.median():+.3f}pp")
    # TJX concentration
    d = per.get("TJX")
    if d:
        yrs = pd.DatetimeIndex([x for x in
                                declusters(pd.DatetimeIndex(built['TJX']['s'].index[built['TJX']['mask'].values]),
                                           H, d["f"].index)]).year
        vv = d["v"]
        if len(yrs) == len(vv):
            by = pd.Series(vv).groupby(yrs.values).agg(["size", "sum", "mean"])
            print("  TJX by year:", {int(y): (int(r['size']), round(100 * r['mean'], 2))
                                     for y, r in by.iterrows()})
            for drop in ([2008], [2008, 2020]):
                m = ~np.isin(yrs, drop)
                if m.sum():
                    print(f"    drop {drop}: N={int(m.sum())} mean {100*vv[m].mean():+.3f}% "
                          f"vs drift {100*d['f'].mean():+.3f}%")
