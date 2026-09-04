"""C5 round 1 — 63d return spread at a PIT floor, long laggard / short leader.

Cell: XLK-XLV 63d spread at PIT pctile 0.0; SMH-XLV 1.2; SMH-IBB 1.2.
Registry collisions this must clear:
  - "sector-vs-index pairs on a crowding/leadership trigger" -> PRICE BOTH LEGS.
  - "laggard-snapback continuation (SMH/QQQ form)" -> that trigger required the
    laggard to be snapping back (r5 high). Here the trigger is the 63d SPREAD
    at a trailing-252 floor, no snapback condition. Reported either way.
  - the one-day XLV-XLK rotation gap (watchlist 15) -> check whether the 63d
    spread floor is carried by recent 1d/5d gaps.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

ASOF = pd.Timestamp("2026-08-26")
PAIRS = [("XLK", "XLV"), ("SMH", "XLV"), ("SMH", "IBB"), ("QQQ", "XLV")]
NAMES = sorted({t for p in PAIRS for t in p})

px = close_panel(NAMES)
px = px[px.index <= ASOF]
print("panel", px.shape, px.index[0].date(), "->", px.index[-1].date())


def vret(s: pd.Series, n: int) -> pd.Series:
    v = s.dropna()
    return (v / v.shift(n) - 1.0).reindex(s.index)


def pit_pctile(s: pd.Series, lookback: int = 252) -> pd.Series:
    v = s.dropna()
    r = v.rolling(lookback).apply(lambda w: (w[:-1] < w[-1]).mean() * 100.0, raw=True)
    return r.reindex(s.index)


print("\n" + "=" * 78)
print("TODAY'S STATE (asof 2026-08-26 close)")
print("=" * 78)
state = {}
for a, b in PAIRS:
    sp = (vret(px[a], 63) - vret(px[b], 63)).dropna()
    pit = pit_pctile(sp)
    state[(a, b)] = (sp, pit)
    print(f"  {a}-{b:<4} 63d spread {100*sp.iloc[-1]:+7.2f}pp   PIT pctile {pit.dropna().iloc[-1]:5.2f}"
          f"   |  {a} 63d {100*vret(px[a],63).iloc[-1]:+6.2f}%   {b} 63d {100*vret(px[b],63).iloc[-1]:+6.2f}%")

# ---- is today's floor a genuine 63d divergence or a recent-gap artifact? -----
print("\n--- registry check: is the 63d floor carried by recent 1d/5d gaps? ---")
for a, b in PAIRS:
    d1 = 100 * (px[a].pct_change(1) - px[b].pct_change(1)).iloc[-1]
    d5 = 100 * (vret(px[a], 5) - vret(px[b], 5)).iloc[-1]
    d21 = 100 * (vret(px[a], 21) - vret(px[b], 21)).iloc[-1]
    d63 = 100 * (vret(px[a], 63) - vret(px[b], 63)).iloc[-1]
    print(f"  {a}-{b:<4} 1d {d1:+6.2f}pp  5d {d5:+6.2f}pp  21d {d21:+7.2f}pp  63d {d63:+7.2f}pp"
          f"   -> last5 share of 63d: {100*d5/d63 if d63 else float('nan'):5.1f}%")

# ---- round 1 battery per pair --------------------------------------------
THR = 2.5
for a, b in PAIRS:
    sp, pit = state[(a, b)]
    mask = (pit <= THR)
    mask = mask.reindex(px.index, fill_value=False).fillna(False)
    n_days = int(mask.sum())
    print(f"\n\n{'#'*78}\n### C5 {a}-{b}: 63d spread PIT <= {THR}   ({n_days} trigger days)\n{'#'*78}")
    if n_days == 0:
        print("  never fires. dead.")
        continue
    variants = {
        "PIT<=1.0": (pit <= 1.0).reindex(px.index, fill_value=False).fillna(False),
        "PIT<=5.0": (pit <= 5.0).reindex(px.index, fill_value=False).fillna(False),
        "PIT<=10.0": (pit <= 10.0).reindex(px.index, fill_value=False).fillna(False),
        "spread<=-15pp": (sp <= -0.15).reindex(px.index, fill_value=False).fillna(False),
    }
    for h in (5, 10):
        battery(px, mask, [(a, 1.0), (b, -1.0)], h,
                f"C5 PAIR long {a} / short {b}", cost_bps=5.0,
                variants=variants if h == 10 else None,
                event_kinds=("cpi", "ppi", "nfp"))

    # ---- PER-LEG ATTRIBUTION (the registry's explicit requirement) --------
    trig = px.index[mask.values]
    for h in (5, 10):
        epi = declusters(trig, h, px.index)
        rows = []
        for lbl, legs in [(f"LONG {a} outright", [(a, 1.0)]),
                          (f"SHORT {b} outright", [(b, -1.0)]),
                          (f"PAIR {a}-{b}", [(a, 1.0), (b, -1.0)])]:
            r = vehicle_ret(px, legs, h, 1)
            val = r.dropna().index
            e = epi.intersection(val)
            row = summarize(r.loc[e].values, lbl)
            base = r.loc[val]
            row["ctl_alldays_pct"] = round(100 * base.mean(), 3)
            row["excess_pct"] = round(row["mean_pct"] - 100 * base.mean(), 3)
            rows.append(row)
        show(rows, f"PER-LEG ATTRIBUTION {a}/{b} h={h} (episodes, min_gap={h})")

    # ---- horizon scan 1..21 (pitch cap is 10) ----------------------------
    show(horizon_scan(px, trig, [(a, 1.0), (b, -1.0)],
                      hs=(1, 2, 3, 5, 10, 15, 21)),
         f"HORIZON SCAN pair {a}-{b} (1..21; pitch cap is 10)")
