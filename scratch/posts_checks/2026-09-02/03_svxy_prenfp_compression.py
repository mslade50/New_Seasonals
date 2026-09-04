"""Idea candidate for Thursday 2026-09-03: long SVXY at Thursday's close, out
at Friday's close (the payrolls print), out of a dead 21-day VIX range.

This morning's pitch killed the same cell as "the right cell, one session
early": its signal date was 09-01 (k=-3), so its MOC entry landed at k=-2.
Tonight IS k=-2, so a MOC entry tomorrow (k=-1) buys the cell the pitch said
pays: SVXY lag-1 h=1 from k=-2 anchors gated on the VIX 21-day relative range
in the bottom 15% of its trailing year. Reproduced here with the pitch's
definitions (pitch_checks/2026-09-02/c9b_vol_nfp_round2.py), then the
controls the post needs: ungated anchors, the gate on non-anchor days, eras
(SVXY re-levered 2018-02-28), midterm, September, concentration, worst, and
the same cell on ^VIX for the longer history.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    anchor_positions, bootstrap_p_le0, close_panel, cluster_note, declusters,
    era_split, fwd_lag, load_events, load_prices, rolling_on_valid, sign_test,
    summarize, wilder_atr,
)

warnings.filterwarnings("ignore")
ASOF = pd.Timestamp("2026-09-02")
px = close_panel(["^VIX", "SVXY", "SPY"])
cal = px["SPY"].dropna().index
vix = px["^VIX"]
rng21 = (rolling_on_valid(vix, lambda x: x.rolling(21).max())
         - rolling_on_valid(vix, lambda x: x.rolling(21).min()))
rel = rng21 / rolling_on_valid(vix, lambda x: x.rolling(21).mean())
RNG = rolling_on_valid(rel, lambda x: x.rolling(252).rank(pct=True) * 100)
print(f"tonight {ASOF.date()}: VIX {vix.iloc[-1]:.2f}  21d rel-range pctile {RNG.iloc[-1]:.1f}")

sv = load_prices(["SVXY"])["SVXY"]
svc = sv["Close"].dropna()
atr = pd.Series(wilder_atr(sv["High"], sv["Low"], sv["Close"]), index=sv.index).reindex(svc.index)
print(f"SVXY {svc.iloc[-1]:.2f} bar {svc.index[-1].date()}  Wilder-14 ATR {atr.iloc[-1]:.4f} ({100*atr.iloc[-1]/svc.iloc[-1]:.2f}%)")

nfp = load_events(["nfp"])["date"]
pos, _ = anchor_positions(cal, nfp, -2)
A_all = pd.DatetimeIndex([cal[i] for i in pos])
A_all = A_all[A_all <= ASOF]
print(f"tonight is a k=-2 payrolls anchor: {ASOF in set(A_all)}   (next NFP {nfp[nfp > ASOF].iloc[0].date()})")
G = (RNG <= 15.0)
A = A_all[G.reindex(A_all).fillna(False).values]
A_off = A_all[~G.reindex(A_all).fillna(False).values]
print(f"k=-2 anchors: {len(A_all)}  gate ON: {len(A)}  gate OFF: {len(A_off)}")


def block(name, s, dates, h=1, lag=1):
    f = fwd_lag(s, h, lag)
    v = f.reindex(pd.DatetimeIndex(dates)).dropna()
    if len(v) == 0:
        print(f"  {name:<48} n=0")
        return v
    st = summarize(v.values)
    nup = int((v > 0).sum())
    drift = 100 * f.dropna().mean()
    print(f"  {name:<48} n={st['n']:<4} mean={st['mean_pct']:+.3f}%  med={st['median_pct']:+.3f}%  "
          f"{nup}-{len(v)-nup} ({st['hit']:.1f}%)  t={st['t']:+.2f}  sp={sign_test(nup, len(v)):.4f}  "
          f"| drift {drift:+.3f}%  | worst {st['worst_pct']:+.2f}% ({v.idxmin().date()})")
    return v


print("\n=== A. SVXY, MOC k=-1 -> MOC print day (lag-1 h1), the pitch's cell ===")
v = block("SVXY gate ON", svc, A)
print("    era:", [(e["label"], e["n"], round(e.get("mean_pct", np.nan), 3), round(e.get("hit", np.nan), 1))
                   for e in era_split(v.index, v.values)])
post = v[v.index >= "2018-03-01"]
print(f"    post re-lever (2018-03+): n={len(post)} mean={100*post.mean():+.3f}% "
      f"{int((post>0).sum())}-{int((post<=0).sum())} sp={sign_test(int((post>0).sum()), len(post)):.4f}")
print("    concentration:", cluster_note(v.index, v.values))
print(f"    bootstrap P(mean<=0): {bootstrap_p_le0(v.values):.4f}")
mid = v[[d.year % 4 == 2 for d in v.index]]
sep = v[[d.month == 9 for d in v.index]]
print(f"    midterm n={len(mid)} {int((mid>0).sum())}-{int((mid<=0).sum())} mean={100*mid.mean():+.3f}%   "
      f"september n={len(sep)} {int((sep>0).sum())}-{int((sep<=0).sum())} mean={100*sep.mean():+.3f}%")
print("    all episodes:", [(d.date().isoformat(), round(100 * x, 2)) for d, x in v.items()])
block("SVXY gate OFF (the discarded anchors)", svc, A_off)
block("SVXY ALL k=-2 anchors", svc, A_all)
# the gate on non-anchor days: every gated session that is not within 3 td of an NFP anchor
near = set()
for a in A_all:
    i = cal.get_loc(a)
    near.update(cal[max(0, i - 3): i + 4])
G_days = cal[G.reindex(cal).fillna(False).values]
G_days = G_days[(G_days <= ASOF) & (G_days >= svc.index[0])]
G_nonnfp = pd.DatetimeIndex([d for d in G_days if d not in near])
G_nonnfp_dc = declusters(G_nonnfp, 5, cal)
block("SVXY gate ON, NON-payrolls days (declustered 5)", svc, G_nonnfp_dc)
print("\n=== A2. horizon / anchor neighbours for SVXY (lag-1) ===")
for k in (-3, -2, -1):
    p_, _ = anchor_positions(cal, nfp, k)
    ak = pd.DatetimeIndex([cal[i] for i in p_])
    ak = ak[(ak <= ASOF) & G.reindex(ak).fillna(False).values]
    for h in (1, 2):
        block(f"SVXY k={k} h={h} gate ON", svc, ak, h)

print("\n=== B. the same cell on ^VIX (2000+), lag-1 h1: VIX lower = the trade works ===")
vv = block("VIX gate ON", vix.dropna(), A)
print("    era:", [(e["label"], e["n"], round(e.get("mean_pct", np.nan), 3), round(e.get("hit", np.nan), 1))
                   for e in era_split(vv.index, vv.values)])
block("VIX gate OFF", vix.dropna(), A_off)
block("VIX ALL k=-2 anchors", vix.dropna(), A_all)
block("VIX gate ON, NON-payrolls days (declustered 5)", vix.dropna(), G_nonnfp_dc)
print("\n=== C. SPY on the same cell, lag-1 h1 (is it just an up day?) ===")
block("SPY gate ON", px["SPY"].dropna(), A)
block("SPY ALL k=-2 anchors", px["SPY"].dropna(), A_all)
print("\n=== D. the loser paths: SVXY gate-ON episodes below -2% ===")
print("   ", [(d.date().isoformat(), round(100 * x, 2)) for d, x in v.sort_values().head(6).items()])
