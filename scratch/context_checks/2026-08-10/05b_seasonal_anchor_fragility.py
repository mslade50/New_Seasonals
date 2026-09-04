"""The engine's own seasonal function, with the anchor walked a few days.

My calendar-day reconstruction in 05 gave SPY 13-13 where the engine reported
18-8. The two use different anchor rules (engine: closest TRADING day-of-year
within +/-2, one per prior year; mine: calendar Aug 9-13, middle session), so
the honest test is to call the engine's own function and slide the anchor.
A seasonal that survives is one where the neighbours agree.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import load_prices, sign_test  # noqa: E402
from scripts.seasonal_edge import seasonal_window_returns  # noqa: E402

TIX = ["SPY", "^GSPC", "QQQ", "IWM", "NG=F"]
px = load_prices(TIX)
spy = px["SPY"]
idx = spy.index[spy.index <= "2026-08-10"]
print(f"SPY frame {spy.index.min().date()} -> {spy.index.max().date()}")

base = seasonal_window_returns(spy, "2026-08-10", 1)
print(f"\nengine cell as shipped, asof 2026-08-10, h1: n={base['n']} "
      f"mean {100*base['mean']:+.3f}% median {100*base['median']:+.3f}% "
      f"{base['n_up']}-{base['n_down']} up "
      f"sign p {sign_test(max(base['n_up'], base['n_down']), base['n']):.4f}")

print("\n--- walk the anchor +/- 5 trading days (SPY, h1) ---")
print(f"{'anchor':12s} {'n':>3s} {'mean%':>8s} {'med%':>8s} {'rec':>8s} {'sign p':>8s}")
rows = []
for off in range(-5, 6):
    pos = len(idx) - 1 + off
    if pos < 0 or pos >= len(idx):
        continue
    a = idx[pos]
    st = seasonal_window_returns(spy, a, 1)
    if not st or st.get("insufficient"):
        continue
    p = sign_test(max(st["n_up"], st["n_down"]), st["n"])
    rows.append((off, st))
    print(f"{str(a.date()):12s} {st['n']:3d} {100*st['mean']:+8.3f} "
          f"{100*st['median']:+8.3f} {st['n_up']:3d}-{st['n_down']:<3d} {p:8.4f}")

ups = [st["n_up"] for _, st in rows]
means = [100 * st["mean"] for _, st in rows]
print(f"\n  up-count across the 11 neighbouring anchors: min {min(ups)} max {max(ups)}")
print(f"  mean%% across them: min {min(means):+.3f} max {max(means):+.3f} "
      f"median {np.median(means):+.3f}")
print(f"  anchors with sign p < 0.10: "
      f"{sum(1 for _, st in rows if sign_test(max(st['n_up'], st['n_down']), st['n']) < 0.10)}"
      f" of {len(rows)}")

print("\n--- same walk for the other index subjects, h1 ---")
for t in ("^GSPC", "QQQ", "IWM"):
    line = []
    for off in range(-3, 4):
        a = idx[len(idx) - 1 + off]
        st = seasonal_window_returns(px[t], a, 1)
        if st and not st.get("insufficient"):
            line.append(f"{st['n_up']}-{st['n_down']}")
    print(f"  {t:8s} {'  '.join(line)}")

print("\n--- and NG=F, whose seasonal cell I am also using ---")
for off in range(-3, 4):
    a = idx[len(idx) - 1 + off]
    st = seasonal_window_returns(px["NG=F"], a, 1)
    if st and not st.get("insufficient"):
        p = sign_test(max(st["n_up"], st["n_down"]), st["n"])
        print(f"  {str(a.date())} n={st['n']:3d} mean {100*st['mean']:+.3f}% "
              f"{st['n_up']}-{st['n_down']} up  sign p {p:.4f}")
