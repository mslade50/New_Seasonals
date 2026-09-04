"""03 left the sharpest result unexamined: after a 2+ ATR USDJPY DOWN session,
TLT fell the next day 23 of 32 times (sign p 0.0100) against a 52.6% local
control. That is the opposite of the reflex "yen up = risk off = bonds bid"
story, so it needs an era split, a concentration check and a look at whether
it is just the 2008 cluster.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

px = close_panel(["JPY=X", "TLT", "IEF", "^TNX", "^GSPC"])
raw = load_prices(["JPY=X"])["JPY=X"]
ref = px["^GSPC"].dropna().index
atr = pd.Series(wilder_atr(raw["High"], raw["Low"], raw["Close"]), index=raw.index)
fires = pd.DatetimeIndex(
    [d for d in raw.index[(raw["Close"].diff() <= -2 * atr.shift(1)).to_numpy()]
     if d in ref])
dec = declusters(fires, 5, ref)


def report(tick, dates, h=1):
    f = fwd_ret(px[tick].dropna(), h).reindex(dates).dropna()
    r = summarize(f.to_numpy())
    up = int((f > 0).sum())
    print(f"  {tick:6s} h{h}  n={r['n']:3d} mean={r['mean_pct']:+7.3f}% "
          f"med={r['median_pct']:+7.3f}% hit={r['hit']:5.1f}% t={r['t']:+6.2f} "
          f"{up}-{len(f)-up} up  sign_p={sign_test(max(up, len(f)-up), len(f)):.4f}")
    print("        ", cluster_note(f.index, f.to_numpy()))
    for e in era_split(f.index, f.to_numpy()):
        u = int((f[f.index >= '2018-01-01'] > 0).sum()) if e['label'] == '2018+' \
            else int((f[f.index < '2018-01-01'] > 0).sum())
        print(f"         {e['label']:9s} n={e['n']:3d} mean={e['mean_pct']:+6.3f}% "
              f"hit={e['hit']:5.1f}%  {u}-{e['n']-u} up")
    return f


print("declustered USDJPY 2+ATR down fires, forward from the fire close")
for tick in ("TLT", "IEF", "^TNX"):
    report(tick, dec, 1)
    print()

f = fwd_ret(px["TLT"].dropna(), 1).reindex(dec).dropna()
print("TLT next-session move, episode by episode:")
for d, v in f.items():
    print(f"  {d.date()}  {100*v:+6.2f}%")

# Drop the GFC entirely: does it survive?
keep = pd.DatetimeIndex([d for d in dec if not ('2008-01-01' <= str(d.date()) <= '2009-12-31')])
print(f"\nex-2008/09 ({len(keep)} of {len(dec)} episodes)")
for tick in ("TLT", "IEF"):
    report(tick, keep, 1)

# And is the yen spike even needed, or is any big risk day enough?
print("\ncontrol: TLT next-session move on ALL sessions in the TLT sample")
s = fwd_ret(px["TLT"].dropna(), 1).dropna()
up = int((s > 0).sum())
print(f"  n={len(s)}  mean={100*s.mean():+.3f}%  hit={100*up/len(s):.1f}%  "
      f"{up}-{len(s)-up} up")
