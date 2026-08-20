"""C7 round 2b: the only vehicle with a pulse is short KRE (outright or against
XLF). Tear it down: concentration, crisis years, era, beta-neutrality, and the
gate-off parent.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

pd.set_option("display.width", 240)

BANKS = ["JPM", "BAC", "C", "WFC", "GS", "MS", "USB", "KEY", "RF", "STT", "SCHW"]
raw = load_prices(BANKS + ["XLF", "KRE", "SPY"])
d = raw["SPY"]["Close"].dropna().index
close = pd.DataFrame({t: raw[t]["Close"].reindex(d) for t in raw})
R5 = pd.DataFrame({t: pct_rank(raw[t]["Close"].dropna(), 5).reindex(d) for t in BANKS})
R63 = pd.DataFrame({t: pct_rank(raw[t]["Close"].dropna(), 63).reindex(d) for t in BANKS})
nv = R5.notna().sum(axis=1)
brd = (R5 <= 20).sum(axis=1) / nv.replace(0, np.nan)
med = R63.median(axis=1)
m_int = ((brd >= 0.70) & (med >= 70) & (nv >= 8)).fillna(False).astype(bool)
m_brd = ((brd >= 0.70) & (nv >= 8)).fillna(False).astype(bool)
epi = declusters(d[m_int], 10, d)

PAIR = [("KRE", -1.0), ("XLF", 1.0)]
KRE = [("KRE", -1.0)]

print("===== KRE availability =====")
print(f" KRE first bar {raw['KRE']['Close'].dropna().index[0].date()}")

for legs, nm in [(KRE, "short KRE"), (PAIR, "short KRE / long XLF")]:
    print(f"\n\n########## {nm} ##########")
    for h in (3, 10):
        ret = vehicle_ret(close, legs, h)
        valid = ret.dropna().index
        e = pd.DatetimeIndex(epi).intersection(valid)
        v = ret.loc[e]
        print(f"\n h={h}  N={len(v)}  mean {100*v.mean():+.3f}%  hit {100*(v>0).mean():.1f}%  "
              f"signp {sign_test(int((v>0).sum()), len(v)):.4f}  median {100*v.median():+.3f}%")
        print(f"   {cluster_note(e, v.values)}")
        by = v.groupby(v.index.year).agg(["count", "mean", "sum"])
        by[["mean", "sum"]] = (by[["mean", "sum"]] * 100).round(2)
        print(by.to_string())
        for drop in ([2008, 2009], [2008, 2009, 2020], [2008, 2009, 2011, 2020]):
            k = v[~v.index.year.isin(drop)]
            if len(k) < 2:
                print(f"   ex-{drop}: N={len(k)}")
                continue
            w = int((k > 0).sum())
            print(f"   ex-{drop}: N={len(k)} mean {100*k.mean():+.3f}% record {w}-{len(k)-w} "
                  f"signp {sign_test(w, len(k)):.4f}")
        # drop top-2 by absolute size
        order = np.argsort(-np.abs(v.values))
        k = v.drop(v.index[order[:2]])
        print(f"   drop-2-largest: N={len(k)} mean {100*k.mean():+.3f}%")
        # era
        show(era_split(e, v.values), f"   era split h={h}")
        # gate-off parent
        e0 = declusters(pd.DatetimeIndex(d[m_brd]).intersection(valid), 10, valid)
        v0 = ret.loc[e0]
        print(f"   GATE-OFF parent (breadth alone): N={len(v0)} mean {100*v0.mean():+.3f}% "
              f"hit {100*(v0>0).mean():.1f}%  -> the intact gate adds "
              f"{100*(v.mean()-v0.mean()):+.3f}pp")

# ---------------------------------------------------------------- beta
print("\n\n===== beta-neutrality of the KRE/XLF pair (registry 2026-08-10) =====")
for h in (3, 10):
    rk = fwd_lag(close["KRE"], h).dropna()
    rx = fwd_lag(close["XLF"], h).reindex(rk.index)
    ok = rx.notna()
    b = np.polyfit(rx[ok].values, rk[ok].values, 1)[0]
    e = pd.DatetimeIndex(epi).intersection(rk.index)
    resid = (rk - b * rx).loc[e]
    print(f" h={h}: KRE-on-XLF beta {b:.2f}; equal-dollar pair short "
          f"{-100*(rk-rx).loc[e].mean():+.3f}%; BETA-NEUTRAL short residual "
          f"{-100*resid.mean():+.3f}% at a {100*(-resid>0).mean():.0f}% hit "
          f"(both legs on trigger days: KRE {100*rk.loc[e].mean():+.3f}%, "
          f"XLF {100*rx.loc[e].mean():+.3f}%)")
