"""c2 round 2 (kill confirmation): is the Hillenbrand post-decision yield decline
present in its OWN window (eve ^TNX at/near a 252 max)? Episode lists for the
TNX gate and the floor gate on XLU, placebo ladder for the TNX-gated XLU cell,
and the reference-class sign table at h=2/3/5 with the TNX gate.
"""
from kB_common import *  # noqa

s = ser("XLU")
idx, c = s.index, s.values
tnx = ser(TNX).reindex(idx).ffill()
tmax = rolling_on_valid(tnx, lambda x: x.rolling(252).max())
tn = tnx.values
tlt = ser("TLT").reindex(idx).values
dp = dec_pos(idx, hmax=16, pre=252)
ps = np.array([p for _, p in dp])
ds = pd.DatetimeIndex([d for d, _ in dp])
r21 = pct_rank(s, 21).values
r63 = pct_rank(s, 63).values
r5 = pct_rank(s, 5).values

print("=== MECHANISM: post-decision ^TNX change by eve yield state (2001+) ===")
states = {"all": np.ones(len(idx), bool),
          "TNX @252max": (tnx >= tmax - 1e-9).values,
          "TNX within 15bp": (tnx >= tmax - 0.15).values,
          "TNX >15bp below max": (tnx < tmax - 0.15).values}
for lbl, g in states.items():
    m = g[ps - 1]
    for h in (1, 2, 3, 5):
        dy = np.array([tn[p + h] - tn[p] for p in ps[m]])
        tr = np.array([tlt[p + h] / tlt[p] - 1 for p in ps[m] if not np.isnan(tlt[p])])
        print(f"  {lbl:<22} h={h} n={m.sum():3d} dTNX {100*np.nanmean(dy):+6.1f}bp "
              f"falls {int((dy<0).sum())}-{int((dy>=0).sum())}  TLT {100*np.nanmean(tr):+.3f}%")

print("\n=== XLU at decision close, eve TNX within 15bp of 252 max ===")
m = states["TNX within 15bp"][ps - 1]
for i in np.where(m)[0]:
    p = ps[i]
    print(f"  {ds[i].date()} XLU r5/21/63 {r5[p-1]:5.1f}/{r21[p-1]:5.1f}/{r63[p-1]:5.1f} "
          f"dTNX h3 {100*(tn[p+3]-tn[p]):+5.1f}bp  XLU " + " ".join(
              f"h{h} {100*(c[p+h]/c[p]-1):+.2f}" for h in (1, 2, 3, 5, 10)))
v = {h: np.array([c[p + h] / c[p] - 1 for p in ps[m]]) for h in (2, 3, 5)}
for h in (2, 3, 5):
    print(line(f"  XLU FOMC & TNX within 15bp h={h}", v[h]))
    df, r0 = placebo(c, ps[m], h)
    print(f"   PLACEBO k=0 rank {r0}/11: " + " ".join(
        f"{k:+d}:{x:+.2f}" for k, x in zip(df.k, df.mean_pct)))

print("\n=== XLU at decision close, eve XLU r21<=10 (any yield state) ===")
m2 = (r21[ps - 1] <= 10)
for i in np.where(m2)[0]:
    p = ps[i]
    print(f"  {ds[i].date()} r5/21/63 {r5[p-1]:5.1f}/{r21[p-1]:5.1f}/{r63[p-1]:5.1f} "
          f"TNX-gap {100*(tn[p-1]-tmax.values[p-1]):+6.1f}bp  XLU " + " ".join(
              f"h{h} {100*(c[p+h]/c[p]-1):+.2f}" for h in (2, 3, 5)))

print("\n=== reference class sign table: FOMC & eve TNX within 15bp of 252 max ===")
for t in ["XLU", "XLRE", "IYR", "VNQ", "XHB", "ITB", "XLP", "SPY"]:
    st = ser(t)
    ix = st.index
    cc = st.values
    tt = ser(TNX).reindex(ix).ffill()
    tm = rolling_on_valid(tt, lambda x: x.rolling(252).max())
    g = (tt >= tm - 0.15).values
    dpp = dec_pos(ix, hmax=10, pre=252)
    pp = np.array([p for _, p in dpp])
    mm = g[pp - 1]
    row = []
    for h in (2, 3, 5):
        vv = np.array([cc[p + h] / cc[p] - 1 for p in pp[mm]])
        tc = tdom_ctrl(ix, cc, pp[mm], h)
        row.append(f"h{h} {100*vv.mean():+.2f}% {int((vv>0).sum())}-{int((vv<=0).sum())}"
                   f" tdomexc {100*np.nanmean(vv-tc):+.2f}")
    print(f"  {t:<5} n={mm.sum():2d}  " + " | ".join(row))
