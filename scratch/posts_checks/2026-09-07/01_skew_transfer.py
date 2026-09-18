"""^SKEW 21d-change rank >= 95 -> does it TRANSFER to a tradeable index leg?

The context product's drill 03 (scratch/context_checks/2026-09-07/03_skew_transfer.py)
scored the LAG-0 cell on ^GSPC: 362 raw sessions -> 124 episodes (10td gap),
h5 +0.435% at t 2.74 on 123 episodes. That number is not tradeable from here --
it enters on the anchor close, and the anchor close (Friday 2026-09-04) is gone.

This script re-runs the SAME mask, verbatim, on tradeable vehicles under the
posts/pitch lag-1 convention:

  A. close entry:  signal on anchor close D -> enter MOC on D+1 -> exit MOC on
     D+1+h.  h = 1, 2, 3, 5.  The h=5 form is the candidate: enter Tuesday
     2026-09-08 MOC, out Tuesday 2026-09-15 MOC.
  B. open entry:   enter MOO on D+1's open -> exit on the close of D+h.
     h=1 is the plain D+1 open-to-close; h=5 holds to the close of D+5.

Every cell prints a local +/-126td control (trigger days removed) and an
all-days control on the same form. Episodes are declustered at 10 td because
the state is persistent (a rich tail bid lasts weeks).

Two conditioning probes on the h=5 close-entry cell, both live today:
  - SPY's own 21d return rank < 50 (today: printed below, ~31)
  - ^SKEW ABSOLUTE 252d level percentile >= 80 (today: ~88), the context
    script's own second cut.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, cluster_note, declusters, era_split, fwd_lag, load_prices,
    local_control, pct_rank, sign_test, summarize,
)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)

ASOF = pd.Timestamp("2026-09-04")
VEH = ["SPY", "QQQ", "IWM", "EEM"]

# --------------------------------------------------------------- the mask
# Reproduced EXACTLY from the context drill: close_panel over the NYSE index,
# pct_rank(LEVEL, 21, 252) -- pitch_lab.pct_rank differences internally, so
# this is the 21d PERCENT CHANGE's trailing-252d percentile.
px = close_panel(["^SKEW", "^GSPC"] + VEH)
nyse = px["^GSPC"].dropna().index
px = px.reindex(nyse)
skew = px["^SKEW"].dropna()
print(f"^SKEW {skew.index[0].date()} .. {skew.index[-1].date()}  n={len(skew)}")
print(f"latest ^SKEW close {skew.iloc[-1]:.2f}")

rk = pct_rank(skew, 21, 252)
print(f"latest ^SKEW 21d-change rank {rk.iloc[-1]:.1f}   (context drill: 98.0)")

trig_all = rk.index[rk >= 95.0]
trig = declusters(trig_all, 10, skew.index)
print(f"\n21d rank >= 95: {len(trig_all)} raw sessions -> {len(trig)} episodes "
      f"(10td min gap)   (context drill: 362 raw -> 124 episodes)")
print(f"  most recent episodes: {[str(d.date()) for d in trig[-8:]]}")

lvl_rank = skew.rolling(252).rank(pct=True) * 100.0
print(f"latest ^SKEW ABSOLUTE 252d level percentile: {lvl_rank.iloc[-1]:.1f}   "
      f"(context drill: 87.7)")
both = rk.index[(rk >= 95.0) & (lvl_rank >= 80.0)]
both_ep = declusters(both, 10, skew.index)
print(f"  rank>=95 AND level>=80th: {len(both)} raw -> {len(both_ep)} episodes")

spy_rk21 = pct_rank(px["SPY"].dropna(), 21, 252)
print(f"latest SPY 21d return rank: {spy_rk21.iloc[-1]:.1f}   (< 50 today: "
      f"{bool(spy_rk21.iloc[-1] < 50)})")
weak_spy = spy_rk21.reindex(skew.index)
trig_weak = pd.DatetimeIndex([d for d in trig if pd.notna(weak_spy.get(d))
                              and weak_spy.get(d) < 50.0])
print(f"  episodes with SPY 21d rank < 50 at the anchor: {len(trig_weak)} "
      f"of {len(trig)}")

raw = load_prices(VEH)

# ------------------------------------------------------------- the freezes
print("\n=== FREEZE: Friday 2026-09-04 close + Wilder-14 ATR (pitch_lab) ===")
from pitch_lab import wilder_atr  # noqa: E402
for name in VEH:
    d = raw[name]
    c = d["Close"].dropna()
    a = pd.Series(wilder_atr(d["High"], d["Low"], d["Close"]),
                  index=d.index).reindex(c.index)
    print(f"  {name:<5} close {c.iloc[-1]:>10.4f}  bar {c.index[-1].date()}  "
          f"Wilder-14 ATR {a.iloc[-1]:>8.4f}  "
          f"({100 * a.iloc[-1] / c.iloc[-1]:.2f}% of close)")


# ----------------------------------------------------------------- helpers
def close_block(label, s, dates, h, ctrl_dates=None, notes=False):
    """lag-1 close entry: MOC on D+1, MOC out on D+1+h."""
    f = fwd_lag(s, h, 1)
    base = f.dropna()
    v = f.reindex(pd.DatetimeIndex(dates)).dropna()
    if len(v) == 0:
        print(f"  {label:<44} n=0")
        return v
    st = summarize(v.values)
    nup = int((v > 0).sum())
    line = (f"  {label:<44} n={st['n']:<4} mean={st['mean_pct']:+.3f}%  "
            f"med={st['median_pct']:+.3f}%  {nup}-{len(v) - nup} "
            f"({st['hit']:.1f}%)  t={st['t']:+.2f}  "
            f"sp={sign_test(nup, len(v)):.4f}")
    if ctrl_dates is not None:
        cv = f.reindex(pd.DatetimeIndex(ctrl_dates)).dropna()
        line += (f"  | LOCAL+-126 n={len(cv)} {100 * cv.mean():+.3f}% "
                 f"hit {100 * (cv > 0).mean():.1f}%")
    line += (f"  | ALL-DAYS n={len(base)} {100 * base.mean():+.3f}% "
             f"hit {100 * (base > 0).mean():.1f}%")
    print(line)
    print(f"      worst {st['worst_pct']:+.2f}% ({v.idxmin().date()})   "
          f"best {st['best_pct']:+.2f}% ({v.idxmax().date()})")
    if notes:
        print("      era:", [(e["label"], e["n"],
                              round(e.get("mean_pct", np.nan), 3),
                              round(e.get("hit", np.nan), 1))
                             for e in era_split(v.index, v.values)])
        print("      conc:", cluster_note(v.index, v.values))
    return v


def open_entry(d, dates, h):
    """MOO on D+1's open -> close of D+h. h=1 is D+1 open-to-close."""
    c = d["Close"].dropna()
    o = d["Open"].reindex(c.index).where(lambda x: x > 0)
    pos = {x: i for i, x in enumerate(c.index)}
    out, gap = {}, {}
    for a in pd.DatetimeIndex(dates):
        p = pos.get(a)
        if p is None or p + h >= len(c) or p + 1 >= len(c):
            continue
        op = o.iloc[p + 1]
        if not np.isfinite(op):
            continue
        out[a] = c.iloc[p + h] / op - 1.0
        gap[a] = op / c.iloc[p] - 1.0
    return pd.Series(out, dtype=float), pd.Series(gap, dtype=float)


def open_block(label, d, dates, h, ctrl_dates=None, notes=False):
    v, g = open_entry(d, dates, h)
    if len(v) == 0:
        print(f"  {label:<44} n=0")
        return v
    c = d["Close"].dropna()
    allv, _ = open_entry(d, c.index[252:-(h + 2)], h)
    st = summarize(v.values)
    nup = int((v > 0).sum())
    line = (f"  {label:<44} n={st['n']:<4} mean={st['mean_pct']:+.3f}%  "
            f"med={st['median_pct']:+.3f}%  {nup}-{len(v) - nup} "
            f"({st['hit']:.1f}%)  t={st['t']:+.2f}  "
            f"sp={sign_test(nup, len(v)):.4f}")
    if ctrl_dates is not None:
        cv, _ = open_entry(d, ctrl_dates, h)
        if len(cv):
            line += (f"  | LOCAL+-126 n={len(cv)} {100 * cv.mean():+.3f}% "
                     f"hit {100 * (cv > 0).mean():.1f}%")
    line += (f"  | ALL-DAYS n={len(allv)} {100 * allv.mean():+.3f}% "
             f"hit {100 * (allv > 0).mean():.1f}%")
    print(line)
    print(f"      gap forfeited (anchor close -> D+1 open) mean "
          f"{100 * g.mean():+.3f}% med {100 * g.median():+.3f}%   |   "
          f"worst {st['worst_pct']:+.2f}% ({v.idxmin().date()})   "
          f"best {st['best_pct']:+.2f}% ({v.idxmax().date()})")
    if notes:
        print("      era:", [(e["label"], e["n"],
                              round(e.get("mean_pct", np.nan), 3),
                              round(e.get("hit", np.nan), 1))
                             for e in era_split(v.index, v.values)])
        print("      conc:", cluster_note(v.index, v.values))
    return v


# --------------------------------------------------------------- per vehicle
for name in VEH:
    d = raw[name]
    c = d["Close"].dropna()
    ep = trig.intersection(c.index)
    ctrl = local_control(c.index, ep, 126)
    print(f"\n================ {name}  (bars {c.index[0].date()} .. "
          f"{c.index[-1].date()};  {len(ep)} of {len(trig)} episodes covered) "
          f"================")
    print("  -- A. lag-1 CLOSE entry (MOC D+1 -> MOC D+1+h), episodes --")
    for h in (1, 2, 3, 5):
        close_block(f"A. episodes h={h}", c, ep, h, ctrl, notes=(h == 5))
    print("  -- A. day-level (all raw sessions in state), for contrast --")
    raw_days = trig_all.intersection(c.index)
    for h in (1, 5):
        close_block(f"A. raw days h={h}", c, raw_days, h)
    print("  -- B. lag-1 OPEN entry (MOO D+1 -> close D+h), episodes --")
    for h in (1, 5):
        open_block(f"B. episodes MOO h={h}", d, ep, h, ctrl, notes=(h == 5))

# ------------------------------------------------------- conditioning probes
print("\n\n================ CONDITIONING PROBES on the h=5 CLOSE-ENTRY cell "
      "================")
print("probe 1: episodes where SPY's own 21d return rank < 50 at the anchor "
      f"(live today: {spy_rk21.iloc[-1]:.1f})")
print("probe 2: episodes where ^SKEW's ABSOLUTE 252d level percentile >= 80 "
      f"(live today: {lvl_rank.iloc[-1]:.1f})")
for name in VEH:
    d = raw[name]
    c = d["Close"].dropna()
    print(f"\n--- {name} ---")
    ep = trig.intersection(c.index)
    ctrl = local_control(c.index, ep, 126)
    close_block("baseline: all episodes h=5", c, ep, 5, ctrl)
    ep_w = trig_weak.intersection(c.index)
    close_block("probe 1: SPY 21d rank < 50", c, ep_w, 5, ctrl, notes=True)
    ep_l = both_ep.intersection(c.index)
    close_block("probe 2: SKEW level pctile >= 80", c, ep_l, 5, ctrl, notes=True)
    ep_b = trig_weak.intersection(both_ep).intersection(c.index)
    close_block("probe 1 AND 2", c, ep_b, 5, ctrl)

# ------------------------------------------------------ episode date listing
print("\n\n=== episode anchor dates (10td decluster, rank>=95) ===")
print(", ".join(str(d.date()) for d in trig))
print(f"\n=== SPY-weak subset ({len(trig_weak)}) ===")
print(", ".join(str(d.date()) for d in trig_weak))
print(f"\n=== SKEW-level>=80 subset ({len(both_ep)}) ===")
print(", ".join(str(d.date()) for d in both_ep))

print("\nDONE.")
