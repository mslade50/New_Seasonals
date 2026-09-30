"""Cross-checks before composing:
 A. the 20 episodes of a 12bp+ 10y jump to a 52w high: same-day IWM/SPY, next-day IWM-SPY and 10y, per episode
 B. MOVE-jump-in-calm-VIX threshold sensitivity (10/12/15%, VIX < 18/20/22)
 C. 10y seasonal doy (18 of 26 lower on this date): FOMC contamination and the structural slot"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from seasonal_edge import seasonal_window_returns  # noqa

px = close_panel(["^TNX", "SPY", "IWM", "^MOVE", "^VIX", "TLT"])
tnx = px["^TNX"].dropna()
idx = tnx.index
bp = tnx.diff() * 100
spy, iwm = px["SPY"].reindex(idx), px["IWM"].reindex(idx)
hi = tnx >= tnx.rolling(252, min_periods=200).max() - 1e-9
trig = idx[((bp >= 12) & hi).values]
epi = declusters(trig[trig < idx[-1]], 5, idx)
rows = []
for d in epi:
    p = idx.get_loc(d)
    rows.append({"date": d.date(), "tnx": round(tnx.iloc[p], 2), "bp": round(bp.iloc[p], 1),
                 "spy_d": round(100 * (spy.iloc[p] / spy.iloc[p - 1] - 1), 2), "iwm_d": round(100 * (iwm.iloc[p] / iwm.iloc[p - 1] - 1), 2),
                 "rel_d": round(100 * (iwm.iloc[p] / iwm.iloc[p - 1] - spy.iloc[p] / spy.iloc[p - 1]), 2),
                 "rel_h1": round(100 * (iwm.iloc[p + 1] / iwm.iloc[p] - spy.iloc[p + 1] / spy.iloc[p]), 2),
                 "bp_h1": round(bp.iloc[p + 1], 1), "spy_h1": round(100 * (spy.iloc[p + 1] / spy.iloc[p] - 1), 2)})
df = pd.DataFrame(rows)
print("A.\n", df.to_string(index=False))
sub = df[(df.iwm_d <= -1.0)]
print("A. subset IWM down 1%+ same day: n", len(sub), "IWM trailed next day", int((sub.rel_h1 < 0).sum()), "mean rel_h1", round(sub.rel_h1.mean(), 3),
      "| bp_h1 up", int((sub.bp_h1 > 0).sum()))
sub2 = df[(df.iwm_d > -1.0)]
print("A. subset IWM not down 1%: n", len(sub2), "trailed", int((sub2.rel_h1 < 0).sum()), "mean", round(sub2.rel_h1.mean(), 3))
print("A. prior-year max 10y jump: largest bp since 2025-04-07:", round(bp.loc["2025-04-08":idx[-2]].max(), 1))

mv = px["^MOVE"].dropna()
mi = mv.index
mr = mv.pct_change()
vix = px["^VIX"].reindex(mi)
spy_m = px["SPY"].reindex(mi)
f5 = fwd_ret(spy_m, 5)
fm5 = fwd_ret(mv, 5)
print("\nB. MOVE jump x VIX ceiling -> SPY h5 (declustered 5)")
for th in (0.10, 0.12, 0.15):
    for vc in (18, 20, 22):
        t = mi[((mr >= th) & (vix < vc)).values]
        e = declusters(t[t < mi[-1]], 5, mi)
        v = f5.loc[e].dropna()
        ctl = local_control(mi, e, 126)
        m5 = fm5.loc[e].dropna()
        print(f"   MOVE>={th:.0%} VIX<{vc}: n {len(v)} SPY h5 {100 * v.mean():+.2f}% up {(v > 0).sum()}/{len(v)} p {sign_test(int((v > 0).sum()), len(v)):.3f} "
              f"local {100 * f5.loc[ctl].mean():+.2f}% | MOVE h5 down {(m5 < 0).sum()}/{len(m5)} mean {100 * m5.mean():+.2f}%")
print("   2026 MOVE jumps >= 21%:", [(str(d.date()), round(100 * x, 1)) for d, x in mr[mr >= 0.21].items() if d.year == 2026])
print("   count of 21%+ jumps by year:", mr[mr >= 0.21].groupby(mr[mr >= 0.21].index.year).size().to_dict())

print("\nC. 10y seasonal doy")
prices = load_prices(["^TNX"])
st = seasonal_window_returns(prices["^TNX"], pd.Timestamp("2026-09-23"), 1)
fomc = set(pd.DatetimeIndex(load_events(["fomc_decision"])["date"]).normalize())
lv = prices["^TNX"]["Close"].dropna()
li = lv.index
tdoy = {}
for y, rr in zip(st["years"], st["rets"]):
    # find the h1 session: the day after the pick; recover by matching the return
    yr = lv[li.year == y]
    ch = yr.pct_change().shift(-1)
    d = (ch - rr).abs().idxmin()
    nxt = li[li.get_loc(d) + 1]
    tdoy[y] = (d.date(), nxt.date(), round(100 * rr, 2), round((lv.loc[nxt] - lv.loc[d]) * 100, 1), nxt in fomc)
for y, v in tdoy.items():
    print("  ", y, v)
nf = [v for v in tdoy.values() if not v[4]]
print("   ex-FOMC: n", len(nf), "down", sum(1 for v in nf if v[3] < 0), "mean bp", round(np.mean([v[3] for v in nf]), 2))
