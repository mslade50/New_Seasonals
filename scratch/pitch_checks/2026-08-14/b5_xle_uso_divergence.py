"""C5 round 1 — producers against the barrel. XLE has beaten USO by ~19pp over
63 sessions (today: XLE ret63 +6.72%, USO ret63 -11.98%, spread +18.69pp, the
91.7th percentile of the spread's own history).

Direction is NOT assumed. All three readings are measured and the data picks:
  A  continuation: long XLE / short USO
  B  mean reversion: short XLE / long USO
  C  each leg outright (long XLE; long USO; short USO)

USO's roll decay must be priced explicitly, so the unconditional drift of every
leg over the same horizon is printed before any conditional number.

Declustering: a 63d spread is a slow state, so the standard min_gap=h leaves
heavy within-regime overlap. Both min_gap=h and min_gap=63 are reported.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pitch_lab import (battery, bootstrap_p_le0, cluster_note, declusters,  # noqa: E402
                       fwd_lag, load_prices, pct_rank, show, sign_test,
                       summarize, vehicle_ret)

ASOF = pd.Timestamp("2026-08-13")
TK = ["XLE", "USO", "SPY", "DBC", "XOP"]
raw = load_prices(TK)
px = pd.DataFrame({t: raw[t]["Close"] for t in TK}).dropna()
px = px[px.index <= ASOF]
print(f"common span {px.index[0].date()} .. {px.index[-1].date()}  n={len(px)}")

spread63 = px["XLE"].pct_change(63) - px["USO"].pct_change(63)
print(f"today spread63 = {100*spread63.iloc[-1]:+.2f}pp  "
      f"(pctile {100*(spread63.iloc[-1] > spread63.dropna()).mean():.1f})")

print("\n" + "=" * 78)
print("0. UNCONDITIONAL DRIFT of every leg (the carry accounting USO owes)")
print("=" * 78)
rows = []
for h in (3, 5, 10):
    for t in ("XLE", "USO"):
        r = fwd_lag(px[t], h).dropna()
        s = summarize(r.values, f"{t} h={h} all days")
        s["ann_pct"] = round(100 * ((1 + r.mean()) ** (252 / h) - 1), 2)
        rows.append(s)
    sp = (fwd_lag(px["XLE"], h) - fwd_lag(px["USO"], h)).dropna()
    rows.append(summarize(sp.values, f"XLE-USO h={h} all days"))
show(rows, "unconditional")
print(f"  USO total return over the sample: "
      f"{100*(px['USO'].iloc[-1]/px['USO'].iloc[0]-1):+.1f}%  vs XLE "
      f"{100*(px['XLE'].iloc[-1]/px['XLE'].iloc[0]-1):+.1f}%")

BASE = 0.19
m0 = (spread63 >= BASE).reindex(px.index, fill_value=False).fillna(False)
print(f"\nTRIGGER spread63 >= {100*BASE:.0f}pp: N={int(m0.sum())} days, "
      f"yrs={sorted(set(m0[m0].index.year))}, live today="
      f"{bool(m0.reindex([ASOF]).fillna(False).iloc[0])}")

VARIANTS = {
    "spread>=10pp": (spread63 >= 0.10),
    "spread>=15pp": (spread63 >= 0.15),
    "spread>=25pp": (spread63 >= 0.25),
    "spread>=30pp": (spread63 >= 0.30),
    "+ XLE ret63>0": (spread63 >= BASE) & (px["XLE"].pct_change(63) > 0),
    "+ USO rank63<10": (spread63 >= BASE) & (pct_rank(px["USO"], 63) < 10),
}
VARIANTS = {k: v.reindex(px.index, fill_value=False).fillna(False)
            for k, v in VARIANTS.items()}

FORMS = [
    ("A continuation: long XLE / short USO", [("XLE", 1.0), ("USO", -1.0)], 15),
    ("B reversion:   short XLE / long USO", [("XLE", -1.0), ("USO", 1.0)], 15),
    ("C1 long XLE outright", [("XLE", 1.0)], 5),
    ("C2 long USO outright", [("USO", 1.0)], 10),
    ("C3 short USO outright", [("USO", -1.0)], 10),
]

for h in (5, 10):
    for lbl, legs, cost in FORMS:
        battery(px, m0, legs, h, f"C5 {lbl}", cost_bps=cost,
                variants=VARIANTS if lbl.startswith("A") else None)

print("\n" + "=" * 78)
print("REGIME-LEVEL declustering (min_gap 63 td): distinct divergence episodes")
print("=" * 78)
trig = px.index[m0.values]
for h in (5, 10):
    epi = declusters(trig, 63, px.index)
    for lbl, legs, cost in FORMS:
        r = vehicle_ret(px, legs, h).loc[epi].dropna()
        if len(r) == 0:
            continue
        wins = int((r > 0).sum())
        s = summarize(r.values, f"{lbl} h={h} (regimes N={len(r)})")
        s["sign_p"] = round(sign_test(wins, len(r)), 4)
        show([s])
    print(f"  regime dates: {[str(d.date()) for d in epi]}")

print("\n" + "=" * 78)
print("MECHANISM: is conditional XLE just levered crude? (beta residual)")
print("=" * 78)
for h in (5, 10):
    xr = fwd_lag(px["XLE"], h)
    ur = fwd_lag(px["USO"], h)
    ok = xr.notna() & ur.notna()
    beta = np.polyfit(ur[ok], xr[ok], 1)[0]
    resid = xr - beta * ur
    epi = declusters(px.index[m0.values], h, px.index)
    show([summarize(resid.loc[epi].dropna().values,
                    f"XLE residual vs {beta:.3f}x USO, h={h} (episodes)"),
          summarize(resid[ok].values, f"residual all days h={h}")],
         f"h={h}")
