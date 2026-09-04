"""BTC +5.38% today with its 21d return at the 100th percentile of its year.
The engine tags this `solid` (n=303, +0.74% next day, 172-131, t 2.91, BH pass).
Crypto momentum cells are usually a handful of episodes wearing 300 days'
clothing, so: decluster to episodes, check concentration, check the era.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

px = close_panel(["BTC-USD", "ETH-USD", "^GSPC"])
btc = px["BTC-USD"].dropna()
ref = btc.index
rank21 = pct_rank(btc, 21, 252)
mask = (rank21 >= 95).fillna(False)
fires = ref[mask.reindex(ref).fillna(False).to_numpy()]
dec = declusters(fires, 10, ref)
print(f"BTC 21d-return rank >= 95: {len(fires)} days, {len(dec)} episodes at 10td")
print(f"today's rank21: {rank21.iloc[-1]:.1f}, 21d return "
      f"{100*(btc.iloc[-1]/btc.iloc[-22]-1):+.1f}%")


def report(dates, h, label, tick="BTC-USD", note=False):
    f = fwd_ret(px[tick].dropna(), h).reindex(dates).dropna()
    if len(f) < 4:
        print(f"  {label:24s} n={len(f)} (too few)")
        return
    r = summarize(f.to_numpy())
    up = int((f > 0).sum())
    print(f"  {label:24s} {tick:8s} h{h} n={r['n']:4d} mean={r['mean_pct']:+7.3f}% "
          f"med={r['median_pct']:+7.3f}% hit={r['hit']:5.1f}% t={r['t']:+6.2f} "
          f"{up}-{len(f)-up} up p={sign_test(max(up,len(f)-up),len(f)):.4f}")
    if note:
        print("      ", cluster_note(f.index, f.to_numpy()))
        for e in era_split(f.index, f.to_numpy(), cut="2021-01-01"):
            print(f"       {e['label']:12s} n={e['n']:4d} mean={e['mean_pct']:+6.3f}% "
                  f"hit={e['hit']:5.1f}%")


ctl = local_control(ref, dec, 126)
for h in (1, 5):
    report(fires, h, "all rank>=95 days", note=(h == 1))
    report(dec, h, "episodes (10td gap)", note=(h == 1))
    report(ctl, h, "local control 126td")
    report(ref, h, "all BTC days")
    print()
print("episode dates:", [str(d.date()) for d in dec[-14:]])
