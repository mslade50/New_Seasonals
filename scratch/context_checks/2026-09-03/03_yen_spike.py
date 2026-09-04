"""Today's yen surge: USDJPY -2.75%, a 2+ ATR down session, with EURJPY -2.45,
CHFJPY -2.30, CADJPY -2.11, NZDJPY -3.04 all firing the same trigger.

Four subjects firing on one event is one event. Decluster it, era-split it,
and check what it means for the thing Scott actually cares about (US equities
and vol), not just for the cross itself.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

SUBJ = ["JPY=X", "EURJPY=X", "^GSPC", "SPY", "^VIX", "TLT", "GC=F"]
px = close_panel(SUBJ)
raw = load_prices(["JPY=X"])["JPY=X"]
ref = px["^GSPC"].dropna().index

atr = pd.Series(wilder_atr(raw["High"], raw["Low"], raw["Close"]), index=raw.index)
move = raw["Close"].diff()
big_down = (move <= -2 * atr.shift(1)) & atr.shift(1).notna()
fires = pd.DatetimeIndex([d for d in big_down.index[big_down.to_numpy()] if d in ref])
print(f"USDJPY 2+ ATR DOWN sessions: {len(fires)} raw")
today = ref[-1]
print(f"today's move: {100*raw['Close'].pct_change().iloc[-1]:+.2f}%  "
      f"= {abs(move.iloc[-1]) / atr.shift(1).iloc[-1]:.2f} ATR")

dec = declusters(fires, 5, ref)
print(f"after declustering at 5 td:  {len(dec)}")
print("  dates:", [str(d.date()) for d in dec[-12:]])
print()


def block(label, dates, tick, h=1, note=False):
    f = fwd_ret(px[tick].dropna(), h).reindex(dates).dropna()
    if len(f) < 4:
        print(f"    {label:26s} {tick:9s} n={len(f)} (too few)")
        return None
    r = summarize(f.to_numpy())
    up = int((f > 0).sum())
    p = sign_test(max(up, len(f) - up), len(f))
    print(f"    {label:26s} {tick:9s} n={r['n']:3d} mean={r['mean_pct']:+7.3f}% "
          f"med={r['median_pct']:+7.3f}% hit={r['hit']:5.1f}% t={r['t']:+6.2f} "
          f"{up}-{len(f)-up} up p={p:.4f}")
    if note:
        print("      ", cluster_note(f.index, f.to_numpy()))
        for e in era_split(f.index, f.to_numpy()):
            print(f"       {e['label']:9s} n={e['n']:3d} mean={e['mean_pct']:+6.3f}% "
                  f"hit={e['hit']:5.1f}%")
    return f


ctl = local_control(ref, dec, 126)
for h in (1, 5):
    print(f"--- h{h} ---")
    for tick in SUBJ:
        block("raw fires", fires, tick, h)
        block("declustered", dec, tick, h, note=(tick in ("JPY=X", "^GSPC")))
        block("local control +/-126td", ctl, tick, h)
        print()

# Does the size of the yen move matter? Rank today's move in history.
pct = raw["Close"].pct_change()
print(f"today's USDJPY move percentile (all sessions): "
      f"{100*(pct.dropna() <= pct.iloc[-1]).mean():.2f}%  "
      f"({(pct.dropna() <= pct.iloc[-1]).sum()} of {pct.dropna().size})")

# The rarer cut: a 2 ATR yen rally the session BEFORE a top-tier US print.
ev = load_events(["nfp", "cpi", "fomc_decision"])
evd = set(pd.DatetimeIndex(ev["date"]))
pos = {d: i for i, d in enumerate(ref)}
before = pd.DatetimeIndex([d for d in dec
                           if pos.get(d, len(ref) - 1) + 1 < len(ref)
                           and ref[pos[d] + 1] in evd])
print(f"\ndeclustered fires whose NEXT session is NFP/CPI/FOMC: {len(before)}")
print("  ", [str(d.date()) for d in before])
for tick in ("JPY=X", "^GSPC", "^VIX"):
    block("fire, event next session", before, tick, 1)
