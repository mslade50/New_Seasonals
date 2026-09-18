"""c4 round 1: long HYG MOC on the decision close after a five-day flush.

Gate on the EVE close. Live: HYG tape z10 -2.27 (10d ret / (21d sd*sqrt10)),
HYG r5 2.0, IEF r5 1.2 (a DURATION-driven flush), HYG 1.98% above 252 low.
Also: mechanism check in its own window (does ^TNX fall / IEF rise after the
decision when the eve carries these states?), and filter_vs_reanchor against
the non-event flush cell killed 09-14.
"""
from kB_common import *  # noqa

s = ser("HYG")
idx = s.index
verify_alignment(idx)
ief = ser("IEF").reindex(idx).ffill()
tz = tape_z10(s)
pz = zscore(s, 10)
r5 = pct_rank(s, 5)
ir5 = pct_rank(ief, 5)
lo252 = rolling_on_valid(s, lambda x: x.rolling(252).min())
near_lo = s <= lo252 * 1.03
print(f"live eve 2026-09-15: tape z10 {tz.iloc[-1]:+.2f}  pitch_lab z10 "
      f"{pz.iloc[-1]:+.2f}  HYG r5 {r5.iloc[-1]:.1f}  IEF r5 {ir5.iloc[-1]:.1f} "
      f" near 252 low(3%) {bool(near_lo.iloc[-1])}")

gates = {
    "tape z10<=-2 [LIVE]": tz <= -2,
    "tape z10<=-1.5": tz <= -1.5,
    "tape z10<=-2.5": tz <= -2.5,
    "pitch_lab z10<=-2": pz <= -2,
    "HYG r5<=5 & IEF r5<=10 (duration flush)": (r5 <= 5) & (ir5 <= 10),
    "tape z10<=-2 & IEF r5<=10": (tz <= -2) & (ir5 <= 10),
    "tape z10<=-2 & IEF r5>=50 (spread flush)": (tz <= -2) & (ir5 >= 50),
    "HYG r5<=10": r5 <= 10,
}
res = event_state_cell("HYG", s, gates, hs=(1, 2, 3, 5, 10),
                       headline="tape z10<=-2 [LIVE]", cost_bps=4)

# placebo + episodes for the duration-flush neighbour too
dp = dec_pos(idx, hmax=16, pre=252)
ps = np.array([p for _, p in dp])
ds = pd.DatetimeIndex([d for d, _ in dp])
c = s.values
for lbl in ["HYG r5<=5 & IEF r5<=10 (duration flush)", "tape z10<=-1.5"]:
    m = gates[lbl].reindex(idx).fillna(False).values.astype(bool)[ps - 1]
    print(f"\nepisodes [{lbl}] n={m.sum()}")
    for i in np.where(m)[0]:
        p = ps[i]
        print(f"   {ds[i].date()} tz {tz.iloc[p-1]:+.2f} " + " ".join(
            f"h{h} {100*(c[p+h]/c[p]-1):+.2f}" for h in (1, 2, 3, 5, 10)))
    for h in (3, 5):
        df, r0 = placebo(c, ps[m], h)
        print(f"  PLACEBO h={h} k=0 rank {r0}/11: " + " ".join(
            f"{k:+d}:{x:+.2f}" for k, x in zip(df.k, df.mean_pct)))

print("\n=========== MECHANISM in its own window (post-decision yields) ===========")
tnx = ser(TNX).reindex(idx).ffill()
tmax = rolling_on_valid(tnx, lambda x: x.rolling(252).max())
tn = tnx.values
iv = ief.values
for lbl, g in [("all decisions", pd.Series(True, index=idx)),
               ("HYG tape z10<=-2", gates["tape z10<=-2 [LIVE]"]),
               ("duration flush", gates["HYG r5<=5 & IEF r5<=10 (duration flush)"]),
               ("TNX within 15bp of 252 max", tnx >= tmax - 0.15)]:
    m = g.reindex(idx).fillna(False).values.astype(bool)[ps - 1]
    for h in (1, 3, 5):
        dy = np.array([tn[p + h] - tn[p] for p in ps[m]])
        ri = np.array([iv[p + h] / iv[p] - 1 for p in ps[m]])
        print(f"  {lbl:<28} h={h} n={m.sum():3d} dTNX {100*dy.mean():+6.1f}bp "
              f"(falls {int((dy<0).sum())}-{int((dy>=0).sum())})  IEF {100*ri.mean():+.3f}%")

print("\n=========== filter vs re-anchor: non-event flush -> FOMC-eve flush ===========")
for h in (3, 5):
    ret = fwd_lag(s, h, 1)
    flush = (tz <= -2).fillna(False)
    fd = declusters(idx[flush.values], 10, idx)
    parent = pd.Series(idx.isin(fd), index=idx)
    eve_m = pd.Series(False, index=idx)
    eve_m.iloc[ps - 1] = True
    child = flush & eve_m
    out = filter_vs_reanchor(ret, parent, child, idx, window_td=21,
                             label=f"HYG tape z10<=-2 h={h}")
    if out["n_matched"]:
        rn = reanchor_null(ret, [a for a, _, _ in out["pairs"]], out["shifts"], idx,
                           np.nanmean(ret.reindex([b for _, b, _ in out["pairs"]]).values))
        print("  reanchor_null:", {k: (round(v, 3) if isinstance(v, float) else v)
                                   for k, v in rn.items()})
