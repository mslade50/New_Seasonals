"""The NYSE composite closed up six sessions running, at a 52-week high, while the tape block
says the share of the panel above its 200d mean is LOWER than 21 sessions ago (65.5 vs 70.1).

Two jobs. First reconcile the engine's breadth number, because its panel is all 98 subjects,
FX pairs and VIX included, which is not what a reader hears in the word breadth. Then run the
streak cell on an equity-only panel: the 11 sector ETFs plus the US index subjects.

Base cell for reference: `P7:up_streak|^NYA` n=240 raw days, h1 -0.105%.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from pitch_lab import close_panel, fwd_ret, summarize, sign_test, era_split, cluster_note, declusters  # noqa
from build_context_state import CONTEXT_UNIVERSE, BREADTH_ONLY  # noqa

ALL = sorted({t for v in CONTEXT_UNIVERSE.values() for t in v} | set(BREADTH_ONLY))
EQUITY = sorted(set(CONTEXT_UNIVERSE["us_index"]) | set(BREADTH_ONLY))
print(f"engine panel {len(ALL)} names; equity-only panel {len(EQUITY)}: {EQUITY}")

px = close_panel(ALL + ["^NYA"])
nya = px["^NYA"].dropna()
idx = nya.index


def breadth_of(cols: list[str]) -> pd.Series:
    sub = px[[c for c in cols if c in px.columns]].reindex(idx)
    ma = sub.rolling(200).mean()
    ok = sub.notna() & ma.notna()
    return 100.0 * ((sub > ma) & ok).sum(axis=1) / ok.sum(axis=1).replace(0, np.nan)


b_all = breadth_of(ALL)
b_eq = breadth_of(EQUITY)
print(f"engine-panel breadth  today {b_all.iloc[-1]:.1f}  21 sessions ago {b_all.iloc[-22]:.1f}"
      f"   (tape block: 65.5 / 70.1)")
print(f"equity-only breadth   today {b_eq.iloc[-1]:.1f}  21 sessions ago {b_eq.iloc[-22]:.1f}"
      f"   63 sessions ago {b_eq.iloc[-64]:.1f}")

r = nya.pct_change()
run, vals = 0, []
for x in (r > 0).astype(int).values:
    run = run + 1 if x else 0
    vals.append(run)
streak = pd.Series(vals, index=idx)
at_high = nya >= nya.rolling(252).max() * 0.999
narrowing = b_eq < b_eq.shift(21)
print(f"\ntonight: streak {int(streak.iloc[-1])}, at a 52w high {bool(at_high.iloc[-1])}, "
      f"equity breadth narrowing {bool(narrowing.iloc[-1])}")

trig = declusters(idx[(streak >= 5).values], 5, idx)
hi = pd.DatetimeIndex([d for d in trig if bool(at_high.get(d, False))])
hi_nar = pd.DatetimeIndex([d for d in hi if bool(narrowing.get(d, False))])
hi_wid = hi.difference(hi_nar)
print(f"5+ up closes: {len(trig)} declustered; at a 52w high {len(hi)}; "
      f"of those, breadth narrowing {len(hi_nar)}, broadening {len(hi_wid)}")


def block(name, dates):
    print(f"\n{name}  n {len(dates)}")
    for h in (1, 5, 10, 21):
        v = fwd_ret(nya, h).reindex(dates).dropna()
        if len(v) < 4:
            continue
        s = summarize(v.values, "")
        up = int((v > 0).sum())
        print(f"  h{h:<3d} n {s['n']:3d}  {up}-{s['n']-up}  mean {s['mean_pct']:+.2f}%  "
              f"med {s['median_pct']:+.2f}%  t {s['t']:+.2f}  signp {sign_test(up, s['n']):.4f}")
    v = fwd_ret(nya, 21).reindex(dates).dropna()
    if len(v) >= 5:
        print("   h21 era:", [f"{e['label']} n {e['n']} mean {e['mean_pct']:+.2f}% up {e['hit']:.1f}%"
                              for e in era_split(v.index, v.values)])
        print("   h21 cluster:", cluster_note(v.index, v.values, k=2))


block("all declustered 5+ up streaks", trig)
block("streak ending at a 52-week high", hi)
block("... equity breadth narrowing over 21 sessions", hi_nar)
block("... equity breadth broadening", hi_wid)
for h in (1, 5, 10, 21):
    base = fwd_ret(nya, h).dropna()
    print(f"control h{h:<3d}: n {len(base)} mean {100*base.mean():+.2f}% up {100*(base>0).mean():.1f}%")

if len(hi_nar):
    v = fwd_ret(nya, 21).reindex(hi_nar).dropna()
    print("\nnarrowing episodes h21:", ", ".join(f"{d.date()}:{100*x:+.1f}" for d, x in zip(v.index, v.values)))

# how unusual is the divergence itself, index at a high with equity breadth 21d lower
div = at_high & narrowing
divd = declusters(idx[div.fillna(False).values], 10, idx)
print(f"\n^NYA at a 52-week high with equity breadth below its 21-sessions-ago reading: "
      f"{int(div.sum())} raw days, {len(divd)} declustered episodes")
for h in (5, 10, 21, 63):
    v = fwd_ret(nya, h).reindex(divd).dropna()
    s = summarize(v.values, "")
    up = int((v > 0).sum())
    base = fwd_ret(nya, h).dropna()
    print(f"  h{h:<3d} n {s['n']:3d}  {up}-{s['n']-up}  mean {s['mean_pct']:+.2f}%  "
          f"med {s['median_pct']:+.2f}%  t {s['t']:+.2f}  signp {sign_test(up, s['n']):.4f}"
          f"   control {100*base.mean():+.2f}% / {100*(base>0).mean():.1f}%")
v = fwd_ret(nya, 21).reindex(divd).dropna()
print("  h21 era:", [f"{e['label']} n {e['n']} mean {e['mean_pct']:+.2f}% up {e['hit']:.1f}%"
                     for e in era_split(v.index, v.values)])
print("  h21 cluster:", cluster_note(v.index, v.values, k=2))
