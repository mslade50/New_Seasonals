"""^MOVE fell 6.31% today, a 2+ ATR down session for bond-implied vol, on the
eve of payrolls. The engine's cell says MOVE keeps falling (116-177 down,
sign p 0.0009, BH pass). A vol index mean-reverts by construction, so the
question is whether this beats MOVE's own unconditional drift after ANY down
day, and what the RATES market does next.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

px = close_panel(["^MOVE", "TLT", "IEF", "^TNX", "^GSPC", "^VIX"])
raw = load_prices(["^MOVE"])["^MOVE"]
ref = px["^GSPC"].dropna().index
mv = px["^MOVE"].dropna()
atr = pd.Series(wilder_atr(raw["High"], raw["Low"], raw["Close"]), index=raw.index)
chg = raw["Close"].diff()
fires = pd.DatetimeIndex([d for d in raw.index[(chg <= -2 * atr.shift(1)).to_numpy()]
                          if d in ref and d in mv.index])
dec = declusters(fires, 5, ref)
print(f"^MOVE 2+ATR down sessions: {len(fires)} raw, {len(dec)} declustered")
print(f"today: {100*mv.pct_change().iloc[-1]:+.2f}% to {mv.iloc[-1]:.2f}, "
      f"{abs(chg.iloc[-1])/atr.shift(1).iloc[-1]:.2f} ATR")
print(f"^MOVE level percentile (trailing 252d): "
      f"{100*(mv.tail(252) <= mv.iloc[-1]).mean():.1f}")


def report(tick, dates, h=1, label="", note=False):
    f = fwd_ret(px[tick].dropna(), h).reindex(dates).dropna()
    if len(f) < 4:
        print(f"  {label:26s} {tick:6s} n={len(f)} (too few)")
        return None
    r = summarize(f.to_numpy())
    up = int((f > 0).sum())
    print(f"  {label:26s} {tick:6s} h{h} n={r['n']:4d} mean={r['mean_pct']:+7.3f}% "
          f"med={r['median_pct']:+7.3f}% hit={r['hit']:5.1f}% t={r['t']:+6.2f} "
          f"{up}-{len(f)-up} up p={sign_test(max(up,len(f)-up),len(f)):.4f}")
    if note:
        print("      ", cluster_note(f.index, f.to_numpy()))
        for e in era_split(f.index, f.to_numpy()):
            print(f"       {e['label']:9s} n={e['n']:3d} mean={e['mean_pct']:+6.3f}% "
                  f"hit={e['hit']:5.1f}%")
    return f


# The control that matters: MOVE's own behaviour after ANY down session.
mvret = mv.pct_change()
anydown = mv.index[(mvret < 0).to_numpy()]
big = mv.index[(mvret <= -0.05).to_numpy()]
print()
for h in (1, 5):
    report("^MOVE", dec, h, "2ATR down (declustered)", note=(h == 1))
    report("^MOVE", pd.DatetimeIndex(big), h, "any -5% day")
    report("^MOVE", pd.DatetimeIndex(anydown), h, "any down day")
    report("^MOVE", mv.index, h, "all days")
    print()

print("what the rates + equity market did after the 2ATR down fires")
for tick in ("TLT", "IEF", "^TNX", "^GSPC", "^VIX"):
    report(tick, dec, 1, "2ATR down (declustered)", note=(tick == "TLT"))
