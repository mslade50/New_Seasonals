"""C5 round 2b — close the last door: the state-matched cell (PIT floor AND
spread <= -15pp) looked best on SMH-XLV. Read it under the regime today is
actually in (SPY above its 200d) and under the midterm conditioner, and check
whether the interaction is stable across the four pitched pairs.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

ASOF = pd.Timestamp("2026-08-26")
H = 10
CLASS = ["XLK", "XLV", "SMH", "IBB", "QQQ", "SPY"]
PITCHED = [("XLK", "XLV"), ("SMH", "XLV"), ("SMH", "IBB"), ("QQQ", "XLV")]
px = close_panel(CLASS)
px = px[px.index <= ASOF]
spy = px["SPY"].dropna()
above = (spy >= rolling_on_valid(spy, lambda x: x.rolling(200).mean()))
frag = pd.read_parquet(ROOT / "data/rd2_fragility.parquet")
ma10 = frag["63d"].rolling(10).mean()


def vret(s, n):
    v = s.dropna()
    return (v / v.shift(n) - 1.0).reindex(s.index)


for a, b in PITCHED:
    sp = (vret(px[a], 63) - vret(px[b], 63)).dropna()
    pit = sp.rolling(252).apply(lambda w: (w[:-1] < w[-1]).mean() * 100.0, raw=True)
    m = ((pit <= 2.5) & (sp <= -0.15))
    ret = fwd_lag(px[a], H, 1) - fwd_lag(px[b], H, 1)
    valid = ret.dropna().index
    trig = pd.DatetimeIndex(px.index[m.reindex(px.index, fill_value=False).fillna(False).values]).intersection(valid)
    if len(trig) < 3:
        print(f"\n{a}-{b}: n<3")
        continue
    epi = declusters(trig, H, valid)
    v = ret.loc[epi].values
    drift = ret.loc[valid].mean()
    ab = above.reindex(epi).fillna(False).values
    mid = np.array([d.year % 4 == 2 for d in epi])
    print(f"\n=== {a}-{b}  state cell (PIT<=2.5 AND spread<=-15pp), h={H} ===")
    print(f"  ALL              N={len(v):>3}  mean {100*v.mean():+7.3f}%  "
          f"excess {100*(v.mean()-drift):+7.3f}pp  hit {100*(v>0).mean():5.1f}%")
    for lbl, msk in [("SPY>=200d <-TODAY", ab), ("SPY<200d", ~ab),
                     ("MIDTERM <-TODAY", mid), ("non-midterm", ~mid),
                     ("SPY>=200d AND midterm <-TODAY", ab & mid)]:
        if msk.sum() == 0:
            print(f"  {lbl:<30} N=  0")
            continue
        w = v[msk]
        print(f"  {lbl:<30} N={int(msk.sum()):>3}  mean {100*w.mean():+7.3f}%  "
              f"excess {100*(w.mean()-drift):+7.3f}pp  hit {100*(w>0).mean():5.1f}%  "
              f"record {int((w>0).sum())}-{int((w<=0).sum())}  sign p "
              f"{sign_test(int((w>0).sum()), len(w)):.4f}")
    yrs = pd.Series(v).groupby(pd.DatetimeIndex(epi).year.values).agg(["size", "mean"])
    print("  by year:", {int(y): (int(r['size']), round(100 * r['mean'], 2)) for y, r in yrs.iterrows()})
    dl = ma10.reindex(epi).dropna()
    print(f"  dial ma10(63d): n {len(dl)} max {dl.max():.1f} >=85 {int((dl>=85).sum())} [today 88.6]")
