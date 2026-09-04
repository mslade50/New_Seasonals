"""Two loose ends. (1) ETH closed +18.91% with its 5d return at the 100th
percentile of its year and BTC +7.64%; the engine's BTC 5d-rank cell is the only
`solid` tag_hint in the whole price lane, so check whether the joint state
survives its own control. (2) Jackson Hole opens in seven sessions and no cell
fired for it; test the run-in before letting it stay a calendar line."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, cluster_note, declusters, era_split, fwd_ret, load_events,
    local_control, sign_test, summarize,
)

print("########## 1. crypto ##########")
px = close_panel(["BTC-USD", "ETH-USD", "SPY"])
btc, eth = px["BTC-USD"].dropna(), px["ETH-USD"].dropna()
for nm, s in (("BTC-USD", btc), ("ETH-USD", eth)):
    r1 = s.pct_change()
    print(f"{nm} {s.index[0].date()} -> {s.index[-1].date()} n={len(s)}  "
          f"today {float(r1.iloc[-1])*100:+.2f}%")

r5 = eth.pct_change(5)
rank5 = r5.rolling(252).rank(pct=True) * 100
print("ETH 5d rank today", round(float(rank5.iloc[-1]), 1))

# ETH single-session tail
re = eth.pct_change()
today = float(re.iloc[-1])
hist = re.iloc[:-1].dropna()
print("ETH sessions larger than today:", int((hist > today).sum()), "of", len(hist))
for thr in (0.10, 0.12, 0.15):
    idx = pd.DatetimeIndex([d for d in re.index[(re >= thr).fillna(False)]
                            if d < eth.index[-1]])
    trig = declusters(idx, 5, eth.index)
    print(f"\n--- ETH single session >= {thr*100:.0f}%  (raw {len(idx)}, "
          f"decl {len(trig)}) ---")
    for h in (1, 5, 10):
        f = fwd_ret(eth, h).reindex(trig).dropna()
        if len(f) == 0:
            continue
        st = summarize(f.values)
        nup = int((f > 0).sum())
        allc = summarize(fwd_ret(eth, h).dropna().values)
        print(f"  h{h:<3} n={st['n']:<3} mean={st['mean_pct']:+.3f}%  "
              f"med={st['median_pct']:+.3f}%  {nup}-{len(f)-nup} up  "
              f"t={st['t']:+.2f}  sign_p={sign_test(nup, len(f)):.4f}  "
              f"all {allc['mean_pct']:+.3f}%")
    if thr == 0.12 and len(trig):
        print("  episodes:", [d.date().isoformat() for d in trig])

# the joint state: ETH >= 12% and BTC >= 5% on the same session
print("\n--- ETH >= 12% AND BTC >= 5% same session ---")
rb = btc.pct_change().reindex(re.index)
j = pd.DatetimeIndex([d for d in re.index[((re >= 0.12) & (rb >= 0.05)).fillna(False)]
                      if d < eth.index[-1]])
jt = declusters(j, 5, eth.index)
print("  raw", len(j), "decl", len(jt), [d.date().isoformat() for d in jt])
for h in (1, 5, 21):
    f = fwd_ret(eth, h).reindex(jt).dropna()
    if len(f) == 0:
        continue
    st = summarize(f.values)
    nup = int((f > 0).sum())
    print(f"  ETH h{h:<3} n={st['n']:<3} mean={st['mean_pct']:+.3f}%  "
          f"{nup}-{len(f)-nup} up  sign_p={sign_test(nup, len(f)):.4f}")

# the engine's BTC 5d-rank cell, with its control
print("\n--- BTC 5d return in the top 5% of its year (engine's `solid` hint) ---")
rb5 = btc.pct_change(5)
brank = rb5.rolling(252).rank(pct=True) * 100
idx = pd.DatetimeIndex([d for d in btc.index[(brank >= 95).fillna(False)]
                        if d < btc.index[-1]])
trig = declusters(idx, 5, btc.index)
print("  raw", len(idx), "decl", len(trig))
for h in (1, 5):
    f = fwd_ret(btc, h).reindex(trig).dropna()
    st = summarize(f.values)
    nup = int((f > 0).sum())
    allc = summarize(fwd_ret(btc, h).dropna().values)
    loc = summarize(fwd_ret(btc, h).reindex(
        local_control(btc.index, trig, 126)).dropna().values)
    print(f"  h{h:<3} n={st['n']:<3} mean={st['mean_pct']:+.3f}%  "
          f"{nup}-{len(f)-nup} up  t={st['t']:+.2f}  sign_p={sign_test(nup, len(f)):.4f}"
          f"  | all {allc['mean_pct']:+.3f}%  local {loc['mean_pct']:+.3f}%")
f1 = fwd_ret(btc, 1).reindex(trig).dropna()
print("  era:", [(e["label"], e["n"], round(e["mean_pct"], 3))
                 for e in era_split(f1.index, f1.values)])
print(" ", cluster_note(f1.index, f1.values))

print("\n########## 2. Jackson Hole run-in ##########")
ev = load_events()
print("kinds:", sorted(ev["kind"].unique()) if "kind" in ev.columns else ev.columns.tolist())
jh = ev[ev.get("kind", ev.columns[0]) == "jackson_hole"] if "kind" in ev.columns else None
if jh is None or len(jh) == 0:
    print("  no jackson_hole rows in macro_events; nothing to test")
else:
    dates = pd.DatetimeIndex(pd.to_datetime(jh["date"]))
    print("  events:", [d.date().isoformat() for d in dates])
    mkt = close_panel(["SPY", "GC=F", "DX-Y.NYB", "^VIX", "TLT"])
    all_idx = mkt.index
    pos = {d: i for i, d in enumerate(all_idx)}
    anchors = []
    for d in dates:
        prior = all_idx[all_idx < d]
        if len(prior) < 10:
            continue
        a = prior[-1]
        i = pos[a] - 6            # the session 7 td before the event
        if i >= 0:
            anchors.append(all_idx[i])
    anchors = pd.DatetimeIndex([a for a in anchors if a < all_idx[-1]])
    print("  anchors (7 td before):", len(anchors))
    for nm in ("SPY", "GC=F", "DX-Y.NYB", "^VIX", "TLT"):
        s = mkt[nm].dropna()
        for h in (7,):
            f = fwd_ret(s, h).reindex(anchors).dropna()
            if len(f) < 5:
                continue
            st = summarize(f.values)
            nup = int((f > 0).sum())
            allc = summarize(fwd_ret(s, h).dropna().values)
            print(f"  {nm:<10} run-in h{h} n={st['n']:<3} mean={st['mean_pct']:+.3f}%  "
                  f"{nup}-{len(f)-nup} up  t={st['t']:+.2f}  "
                  f"sign_p={sign_test(nup, len(f)):.4f}  all {allc['mean_pct']:+.3f}%")
