"""Two follow-ons the robustness pass turned up.

(a) A control in drill 07 was more interesting than the cell it was controlling:
    IWM after ^NYA prints a 52w high, n=527 raw, -0.116%, 45.0% hit, t=-2.99,
    against an all-days IWM of +0.102%. ^NYA closed AT a 52w high today. Needs
    declustering, an era split and a look at whether it is a small-cap statement
    or just a "the index is extended" statement.
(b) Is the CPI-eve VIX bid a CPI fact or a scheduled-event fact? If NFP and FOMC
    eves do the same thing at a 52w high, the general version is the better
    nugget and the CPI framing is too narrow.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, load_events, fwd_ret, declusters, local_control, summarize,
    era_split, sign_test, cluster_note,
)

TIX = ["SPY", "QQQ", "IWM", "^RUT", "^NYA", "^VIX", "^GSPC"]
px = close_panel(TIX)
px = px[px.index >= "1999-01-01"]
dates = px.index
nya_dh = px["^NYA"] / px["^NYA"].rolling(252).max() - 1.0
spy_dh = px["SPY"] / px["SPY"].rolling(252).max() - 1.0
print(f"today ^NYA dist-high {100*nya_dh.iloc[-1]:+.4f}%  SPY {100*spy_dh.iloc[-1]:+.4f}%")


def stat(sub, idx, h, label, indent="   ", extra=False):
    f = fwd_ret(px[sub].dropna(), h)
    v = f.reindex(pd.DatetimeIndex(sorted(set(idx)))).dropna()
    if len(v) < 3:
        print(f"{indent}{label:56s} n={len(v)} too few")
        return None
    r = summarize(v.values, "")
    up = int((v.values > 0).sum())
    print(f"{indent}{label:56s} n={r['n']:4d} mean {r['mean_pct']:+.3f}% "
          f"med {r['median_pct']:+.3f}% hit {r['hit']:.1f}% t={r['t']:+.2f} | "
          f"{up}-{len(v)-up} up-p {sign_test(up, len(v)):.4f} "
          f"dn-p {sign_test(len(v)-up, len(v)):.4f}")
    if extra:
        print(f"        {cluster_note(v.index, v.values)}")
        for e in era_split(v.index, v.values):
            if e.get("n", 0):
                print(f"        era n={e['n']:4d} mean {e['mean_pct']:+.3f}% "
                      f"hit {e['hit']:.1f}% t={e['t']:+.2f}")
        keep = v.iloc[np.argsort(-np.abs(v.values))[2:]]
        rk = summarize(keep.values, "")
        upk = int((keep.values > 0).sum())
        print(f"        ex the 2 largest: n={rk['n']} mean {rk['mean_pct']:+.3f}% "
              f"hit {rk['hit']:.1f}% {upk}-{len(keep)-upk}")
    return v


print("\n" + "=" * 78)
print("(a) small caps after the NYSE composite prints a 52w high")
print("=" * 78)
hi = dates[(nya_dh > -0.0005).fillna(False).values]
hi = hi[hi <= dates[-2]]
print(f"   raw sessions at a ^NYA 52w high: {len(hi)}")
for gap, lab in ((1, "raw"), (5, "5td declustered"), (21, "21td declustered")):
    dc = declusters(hi, gap, dates) if gap > 1 else hi
    print(f"\n   --- {lab} ({len(dc)} anchors) ---")
    for sub in ("IWM", "SPY", "QQQ"):
        stat(sub, dc, 1, f"{sub} h1", extra=(sub == "IWM" and gap == 21))
    f_i = fwd_ret(px["IWM"].dropna(), 1).reindex(dc)
    f_s = fwd_ret(px["SPY"].dropna(), 1).reindex(dc)
    d = (f_i - f_s).dropna()
    r = summarize(d.values, "")
    up = int((d.values > 0).sum())
    print(f"   {'IWM minus SPY h1':56s} n={r['n']:4d} mean {r['mean_pct']:+.3f}% "
          f"med {r['median_pct']:+.3f}% hit {r['hit']:.1f}% t={r['t']:+.2f} | "
          f"{up}-{len(d)-up} up-p {sign_test(up, len(d)):.4f} "
          f"dn-p {sign_test(len(d)-up, len(d)):.4f}")

print("\n   controls:")
stat("IWM", dates, 1, "IWM, ALL sessions")
stat("IWM", dates[(spy_dh > -0.0005).fillna(False).values], 1,
     "IWM when SPY (not ^NYA) is at a 52w high")
f_i = fwd_ret(px["IWM"].dropna(), 1)
f_s = fwd_ret(px["SPY"].dropna(), 1)
d_all = (f_i - f_s).dropna()
r = summarize(d_all.values, "")
up = int((d_all.values > 0).sum())
print(f"   {'IWM minus SPY, ALL sessions':56s} n={r['n']:4d} mean {r['mean_pct']:+.3f}% "
      f"med {r['median_pct']:+.3f}% hit {r['hit']:.1f}% t={r['t']:+.2f} | {up}-{len(d_all)-up}")

print("\n" + "=" * 78)
print("(b) is the eve vol bid a CPI fact or a scheduled-event fact?")
print("=" * 78)
at_high = (spy_dh > -0.005)


def eves(kind, k=2):
    e = load_events([kind])
    out = []
    for dd in e["date"]:
        pos = dates.searchsorted(pd.Timestamp(dd))
        if pos >= len(dates) or dates[pos] != pd.Timestamp(dd) or pos - k < 0:
            continue
        out.append(dates[pos - k])
    return pd.DatetimeIndex(out)


for kind in ("cpi", "nfp", "fomc_decision", "ppi"):
    ev_idx = eves(kind)
    sel = ev_idx[at_high.reindex(ev_idx).fillna(False).values]
    stat("^VIX", sel, 1, f"^VIX, {kind} eve (k=2) with SPY at a 52w high",
         extra=(kind in ("cpi", "nfp")))
    stat("^VIX", ev_idx, 1, f"   ...{kind} eve, tape unconditional")

print("\n   pooled: any of CPI / NFP / FOMC, 2 td out, SPY at a 52w high")
pool = pd.DatetimeIndex(sorted(set().union(*[set(eves(k)) for k in ("cpi", "nfp", "fomc_decision")])))
sel = pool[at_high.reindex(pool).fillna(False).values]
stat("^VIX", sel, 1, "^VIX h1", extra=True)
stat("SPY", sel, 1, "SPY h1")
stat("^VIX", pool, 1, "   ...same events, tape unconditional")
rest = dates[at_high.reindex(dates).fillna(False).values].difference(pool)
stat("^VIX", rest, 1, "SPY at a 52w high with NO top-tier event 2 td out")
