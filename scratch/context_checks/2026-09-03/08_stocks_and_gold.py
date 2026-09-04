"""No cross-asset trigger fired today, but the shape of the session is the most
distinctive thing about it: ^GSPC +1.06%, gold +3.53%, silver +4.42%, the whole
curve lower, VIX -5.79%, the dollar -0.57%. Everything rallied except the
dollar.

`P9:stocks_bonds_up` missed it because TLT only managed +0.15% (the co-move
floor is 50bp) and `P9c:dollar_gold_up` correctly did not fire because the
dollar fell. So this cell has to be built by hand.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

px = close_panel(["^GSPC", "SPY", "GC=F", "DX-Y.NYB", "^TNX", "TLT", "^VIX"])
ref = px["^GSPC"].dropna().index
r = {t: px[t].reindex(ref).pct_change() for t in px.columns}

spx, gold, dxy, tnx = r["^GSPC"], r["GC=F"], r["DX-Y.NYB"], r["^TNX"]
print("today: ^GSPC %+.2f%%  GC=F %+.2f%%  DXY %+.2f%%  ^TNX %+.2f%%"
      % (100*spx.iloc[-1], 100*gold.iloc[-1], 100*dxy.iloc[-1], 100*tnx.iloc[-1]))

mask = (spx >= 0.01) & (gold >= 0.02) & (dxy < 0) & (tnx < 0)
mask = mask.fillna(False)
fires = ref[mask.to_numpy()]
dec = declusters(fires, 5, ref)
print(f"\nstocks +1%, gold +2%, dollar down, yields down: "
      f"{len(fires)} days, {len(dec)} episodes")
print("  ", [str(d.date()) for d in dec])

# Relax one leg at a time to see which condition is doing the work.
for label, m in (("stocks +1% & gold +2%", (spx >= 0.01) & (gold >= 0.02)),
                 ("stocks +1% & gold +2% & dollar dn",
                  (spx >= 0.01) & (gold >= 0.02) & (dxy < 0)),
                 ("all four legs", mask),
                 ("stocks +1% (any gold)", spx >= 0.01)):
    f = ref[m.fillna(False).to_numpy()]
    d = declusters(f, 5, ref)
    print(f"  {label:34s} {len(f):5d} days / {len(d):4d} episodes")


def report(dates, tick, h, label, note=False):
    f = fwd_ret(px[tick].dropna(), h).reindex(dates).dropna()
    if len(f) < 4:
        print(f"  {label:26s} {tick:9s} n={len(f)} (too few)")
        return
    s = summarize(f.to_numpy())
    up = int((f > 0).sum())
    print(f"  {label:26s} {tick:9s} h{h} n={s['n']:4d} mean={s['mean_pct']:+7.3f}% "
          f"med={s['median_pct']:+7.3f}% hit={s['hit']:5.1f}% t={s['t']:+6.2f} "
          f"{up}-{len(f)-up} up p={sign_test(max(up,len(f)-up),len(f)):.4f}")
    if note:
        print("      ", cluster_note(f.index, f.to_numpy()))
        for e in era_split(f.index, f.to_numpy()):
            print(f"       {e['label']:9s} n={e['n']:4d} mean={e['mean_pct']:+6.3f}% "
                  f"hit={e['hit']:5.1f}%")


ctl = local_control(ref, dec, 126)
for h in (1, 5):
    for tick in ("^GSPC", "GC=F", "TLT", "^VIX"):
        report(dec, tick, h, "4-leg episodes", note=(h == 1 and tick in ("^GSPC", "GC=F")))
        report(ctl, tick, h, "local control 126td")
    print()
