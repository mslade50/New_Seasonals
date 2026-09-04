"""Verification pass: every number the brief quotes that a drill did not
already print, plus the era check the [solid] tag on the VIX Monday cell
needs before it can carry that tag."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import load_prices, load_events, summarize, sign_test, cluster_note

px = load_prices(["^VIX", "QQQ"])
ASOF = pd.Timestamp("2026-08-21")

print("########## ^VIX all Mondays, era split (the [solid] gate) ##########")
r = px["^VIX"]["Close"].astype(float).loc[:ASOF].pct_change(fill_method=None).dropna()
mon = r[r.index.dayofweek == 0]
cut = pd.Timestamp("2018-01-01")
for lab, v in [("all", mon), ("pre-2018", mon[mon.index < cut]), ("2018+", mon[mon.index >= cut])]:
    up = int((v.values > 0).sum())
    st = summarize(v.values, lab)
    print(f"   {lab:9s} n={st['n']:5d} mean={st['mean_pct']:+.3f}% med={st['median_pct']:+.3f}% "
          f"hit={st['hit']:.1f}% t={st['t']:+.2f} record {up}-{st['n']-up}")
print("   concentration:", cluster_note(mon.index, mon.values, k=2))
dec = pd.Series(mon.values, index=mon.index).groupby((mon.index.year // 5) * 5).mean() * 100
print("   by half-decade:", {int(k): round(v, 2) for k, v in dec.items()})

print("\n########## Jackson Hole VIX sign test ##########")
jh = pd.DatetimeIndex(load_events(["jackson_hole"])["date"])
jh = jh[jh <= ASOF]
v = r.reindex(jh).dropna()
down = int((v.values < 0).sum())
print(f"   n={len(v)} down={down} up={len(v)-down} "
      f"sign_p(down)={sign_test(down, len(v)):.4f} "
      f"mean={100*v.mean():+.3f}% median={100*np.median(v.values):+.3f}%")

print("\n########## QQQ Monday base rates, restated ##########")
q = px["QQQ"]["Close"].astype(float).loc[:ASOF].pct_change(fill_method=None).dropna()
i = q.index
for lab, m in [("all Mondays", i.dayofweek == 0),
               ("August Mondays", (i.month == 8) & (i.dayofweek == 0)),
               ("Aug Mondays pre-2018", (i.month == 8) & (i.dayofweek == 0) & (i < cut)),
               ("Aug Mondays 2018+", (i.month == 8) & (i.dayofweek == 0) & (i >= cut))]:
    v = q[m]
    up = int((v.values > 0).sum())
    st = summarize(v.values, lab)
    print(f"   {lab:22s} n={st['n']:5d} mean={st['mean_pct']:+.3f}% "
          f"hit={st['hit']:.2f}% record {up}-{st['n']-up}")
aug = q[(i.month == 8) & (i.dayofweek == 0)]
byyear = pd.Series(aug.values, index=aug.index).groupby(aug.index.year)
lose = [int(y) for y, x in byyear if (x > 0).sum() <= (x <= 0).sum()]
print(f"   losing-majority Augusts: {lose}")
print(f"   of the {sum(1 for y,_ in byyear if y >= 2018)} Augusts since 2018, "
      f"{sum(1 for y in lose if y >= 2018)} had a losing majority "
      f"(2026 is partial: {len(aug[aug.index.year == 2026])} Mondays so far)")
