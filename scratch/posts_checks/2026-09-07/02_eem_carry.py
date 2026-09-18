"""Yen rips, EM rallies anyway -- the tradeable (lag-1) version of the cell.

Context drill 06 (scratch/context_checks/2026-09-07/06_carry_unwind_em.py)
scored the disagreement cell LAG-0: USDJPY (JPY=X) <= -1.5% on the session AND
EEM > 0 the same session, declustered at 5 td -> EEM h5 n=30, mean +0.694%,
hit 73.3%, t ~1.1. That form enters on the anchor close, which for us is
Friday 2026-09-04 and is gone.

This re-runs the IDENTICAL mask under the lag-1 convention: enter the NEXT
session (Tuesday 2026-09-08), MOC or MOO, and measure from there.

Mask reproduced verbatim from the context drill:
  panel = close_panel(["JPY=X","EEM","FXI","^GSPC","EWJ","AUDJPY=X"])
          reindexed to the ^GSPC (NYSE) calendar,
          dropna on JPY=X and EEM, truncated at 2026-09-04;
  jpy = panel["JPY=X"].pct_change();  eem = panel["EEM"].pct_change()
  trigger = (jpy <= -0.015) & (eem > 0)

EEM's first bar is 2003-04-14, so the cell is 2003+ by construction.

Declustering: the context drill used a 5 td gap (that is what makes n=30).
Both 5 td and 10 td are reported here; the 10 td set is the house rule for a
persistent trigger, and this trigger is a one-session event, so 5 td is the
one that reconciles.

Controls: the plain washout (JPY=X <= -1.5% regardless of EEM's sign), EEM up
on any day, the local +/-126td neighbourhood ex-trigger, and all days.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, cluster_note, declusters, era_split, fwd_lag, load_prices,
    local_control, sign_test, summarize, wilder_atr,
)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)

ASOF = pd.Timestamp("2026-09-04")

TK = ["JPY=X", "EEM", "FXI", "^GSPC", "EWJ", "AUDJPY=X"]
panel = close_panel(TK)
nyse = panel["^GSPC"].dropna().index
px = panel.reindex(nyse).dropna(subset=["JPY=X", "EEM"], how="any")
px = px[px.index <= ASOF]
print(f"panel {px.index[0].date()} .. {px.index[-1].date()}  n={len(px)}")

jpy = px["JPY=X"].pct_change()      # negative = yen STRONGER
eem = px["EEM"].pct_change()
last = px.index[-1]
print(f"latest {last.date()}: USDJPY {100 * jpy.loc[last]:+.2f}%, "
      f"EEM {100 * eem.loc[last]:+.2f}%   (context drill: -2.05% / +1.82%)")

YEN = -0.015
strong_yen = jpy <= YEN
trig_mask = (strong_yen & (eem > 0)).fillna(False)
washout_mask = strong_yen.fillna(False)
eem_up_mask = (eem > 0).fillna(False)

trig_all = px.index[trig_mask]
trig5 = declusters(trig_all, 5, px.index)
trig10 = declusters(trig_all, 10, px.index)
print(f"\nUSDJPY <= -1.5% sessions: {int(washout_mask.sum())}")
print(f"disagreement cell (yen up 1.5%+ AND EEM up): {len(trig_all)} raw -> "
      f"{len(trig5)} episodes @5td / {len(trig10)} episodes @10td")
print(f"  (context drill h5 cell was n=30 at 5td decluster)")
print(f"  today {last.date()} in the raw state: {bool(trig_mask.loc[last])}")

wash5 = declusters(px.index[washout_mask], 5, px.index)
wash10 = declusters(px.index[washout_mask], 10, px.index)
print(f"washout control (yen up 1.5%+, either EEM sign): "
      f"{int(washout_mask.sum())} raw -> {len(wash5)} @5td / {len(wash10)} @10td")
print(f"EEM-up control (any session EEM > 0): {int(eem_up_mask.sum())} raw days")

# ------------------------------------------------------------------ freeze
raw = load_prices(["EEM"])
d = raw["EEM"]
c = d["Close"].dropna()
a = pd.Series(wilder_atr(d["High"], d["Low"], d["Close"]),
              index=d.index).reindex(c.index)
print("\n=== FREEZE: Friday 2026-09-04 close + Wilder-14 ATR ===")
print(f"  EEM   close {c.iloc[-1]:>10.4f}  bar {c.index[-1].date()}  "
      f"Wilder-14 ATR {a.iloc[-1]:>8.4f}  "
      f"({100 * a.iloc[-1] / c.iloc[-1]:.2f}% of close)")

ser = px["EEM"]
loc_ctrl = local_control(px.index, trig5, 126)
print(f"local +/-126td control (ex-trigger): n={len(loc_ctrl)} days")


# ----------------------------------------------------------------- helpers
def close_block(label, dates, h, notes=False):
    f = fwd_lag(ser, h, 1)
    base = f.dropna()
    v = f.reindex(pd.DatetimeIndex(dates)).dropna()
    if len(v) == 0:
        print(f"  {label:<46} n=0")
        return v
    st = summarize(v.values)
    nup = int((v > 0).sum())
    cv = f.reindex(loc_ctrl).dropna()
    print(f"  {label:<46} n={st['n']:<5} mean={st['mean_pct']:+.3f}%  "
          f"med={st['median_pct']:+.3f}%  {nup}-{len(v) - nup} "
          f"({st['hit']:.1f}%)  t={st['t']:+.2f}  "
          f"sp={sign_test(nup, len(v)):.4f}  | LOCAL+-126 n={len(cv)} "
          f"{100 * cv.mean():+.3f}% hit {100 * (cv > 0).mean():.1f}%"
          f"  | ALL-DAYS n={len(base)} {100 * base.mean():+.3f}% "
          f"hit {100 * (base > 0).mean():.1f}%")
    print(f"      worst {st['worst_pct']:+.2f}% ({v.idxmin().date()})   "
          f"best {st['best_pct']:+.2f}% ({v.idxmax().date()})")
    if notes:
        print("      era:", [(e["label"], e["n"],
                              round(e.get("mean_pct", np.nan), 3),
                              round(e.get("hit", np.nan), 1))
                             for e in era_split(v.index, v.values)])
        print("      conc:", cluster_note(v.index, v.values))
    return v


def open_entry(dates, h):
    """MOO on D+1's open -> close of D+h."""
    o = d["Open"].reindex(c.index).where(lambda x: x > 0)
    pos = {x: i for i, x in enumerate(c.index)}
    out, gap = {}, {}
    for x in pd.DatetimeIndex(dates):
        p = pos.get(x)
        if p is None or p + h >= len(c) or p + 1 >= len(c):
            continue
        op = o.iloc[p + 1]
        if not np.isfinite(op):
            continue
        out[x] = c.iloc[p + h] / op - 1.0
        gap[x] = op / c.iloc[p] - 1.0
    return pd.Series(out, dtype=float), pd.Series(gap, dtype=float)


def open_block(label, dates, h, notes=False):
    v, g = open_entry(dates, h)
    if len(v) == 0:
        print(f"  {label:<46} n=0")
        return v
    allv, _ = open_entry(c.index[252:-(h + 2)], h)
    cv, _ = open_entry(loc_ctrl, h)
    st = summarize(v.values)
    nup = int((v > 0).sum())
    print(f"  {label:<46} n={st['n']:<5} mean={st['mean_pct']:+.3f}%  "
          f"med={st['median_pct']:+.3f}%  {nup}-{len(v) - nup} "
          f"({st['hit']:.1f}%)  t={st['t']:+.2f}  "
          f"sp={sign_test(nup, len(v)):.4f}  | LOCAL+-126 n={len(cv)} "
          f"{100 * cv.mean():+.3f}% hit {100 * (cv > 0).mean():.1f}%"
          f"  | ALL-DAYS n={len(allv)} {100 * allv.mean():+.3f}% "
          f"hit {100 * (allv > 0).mean():.1f}%")
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


# ------------------------------------------------- reconcile with the context
print("\n\n================ RECONCILIATION: the context drill's LAG-0 cell "
      "================")
from pitch_lab import fwd_ret  # noqa: E402
for h in (1, 5):
    v = fwd_ret(ser, h).reindex(trig5).dropna()
    st = summarize(v.values)
    nup = int((v > 0).sum())
    print(f"  EEM lag0 h={h} (5td episodes)  n={st['n']} mean="
          f"{st['mean_pct']:+.3f}%  hit {st['hit']:.1f}%  t={st['t']:+.2f}  "
          f"{nup}-{len(v) - nup}")
print("  context drill reported: h5 n=30, mean +0.694%, hit 73.3%, t 1.1")

# ---------------------------------------------------------- the tradeable form
print("\n\n================ EEM, LAG-1 (enter Tuesday 2026-09-08) "
      "================")
print("  -- A. CLOSE entry: MOC on D+1 -> MOC on D+1+h, 5td episodes --")
for h in (1, 3, 5, 10):
    close_block(f"A. disagreement episodes h={h}", trig5, h, notes=(h == 5))

print("  -- A. same cell, 10td decluster --")
for h in (1, 5):
    close_block(f"A. disagreement episodes @10td h={h}", trig10, h,
                notes=(h == 5))

print("  -- A. day-level (all raw sessions in state) --")
for h in (1, 5):
    close_block(f"A. raw days h={h}", trig_all, h)

print("\n  -- B. OPEN entry: MOO on D+1 -> close of D+h, 5td episodes --")
open_block("B. disagreement episodes MOO h=5", trig5, 5, notes=True)
open_block("B. disagreement episodes MOO h=1", trig5, 1)

# -------------------------------------------------------------- the controls
print("\n\n================ CONTROLS (same lag-1 close form) ================")
print("  -- CTRL 1: the plain washout, USDJPY <= -1.5% regardless of EEM --")
for h in (1, 5):
    close_block(f"CTRL washout episodes @5td h={h}", wash5, h, notes=(h == 5))
    close_block(f"CTRL washout raw days h={h}", px.index[washout_mask], h)

print("  -- CTRL 2: EEM simply up on the session (no yen condition) --")
for h in (1, 5):
    close_block(f"CTRL EEM-up raw days h={h}", px.index[eem_up_mask], h)

print("  -- CTRL 3: the textbook pairing, yen up 1.5%+ AND EEM DOWN --")
textbook = px.index[(strong_yen & (eem < 0)).fillna(False)]
tb5 = declusters(textbook, 5, px.index)
print(f"     ({len(textbook)} raw -> {len(tb5)} episodes @5td)")
for h in (1, 5):
    close_block(f"CTRL textbook episodes h={h}", tb5, h, notes=(h == 5))

# ------------------------------------------------------------ episode listing
print("\n\n=== disagreement episode anchor dates (5td decluster) ===")
for x in trig5:
    print(f"  {x.date()}  USDJPY {100 * jpy.loc[x]:+.2f}%  "
          f"EEM {100 * eem.loc[x]:+.2f}%")
print("\n=== same, 10td decluster ===")
print(", ".join(str(x.date()) for x in trig10))

print("\nDONE.")
