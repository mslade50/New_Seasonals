"""C5 round 2 — the producer-over-barrel divergence, after round 1 showed the
continuation spread is 79% two April-2020 episodes.

Questions:
  (1) LIVENESS — today's spread is +18.69pp. The 19pp base used in round 1 is
      NOT live. Re-base to the largest threshold that IS live and re-measure.
  (2) THRESHOLD MONOTONICITY — a real state moves smoothly with its cut.
  (3) DROP APRIL 2020 — the negative-oil-price crash is a structural break in
      USO (contract roll to the June future, an 8:1 reverse split that August).
      What is left without it?
  (4) CYCLE + ERA on regime-declustered episodes.
  (5) TAPE over-selection.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pitch_lab import (bootstrap_p_le0, cluster_note, declusters, fwd_lag,  # noqa: E402
                       load_prices, show, sign_test, summarize, vehicle_ret)

ASOF = pd.Timestamp("2026-08-13")
raw = load_prices(["XLE", "USO", "SPY"])
px = pd.DataFrame({t: raw[t]["Close"] for t in ["XLE", "USO", "SPY"]}).dropna()
px = px[px.index <= ASOF]
spread63 = px["XLE"].pct_change(63) - px["USO"].pct_change(63)
today = spread63.iloc[-1]
print(f"today spread63 = {100*today:+.3f}pp")

FORMS = {
    "A long XLE / short USO": [("XLE", 1.0), ("USO", -1.0)],
    "B short XLE / long USO": [("XLE", -1.0), ("USO", 1.0)],
    "C1 long XLE": [("XLE", 1.0)],
    "C3 short USO": [("USO", -1.0)],
}

print("\n" + "=" * 78)
print("1+2. THRESHOLD LADDER, regime-declustered (min_gap 63 td), h=5")
print("     '*' marks thresholds that are LIVE on today's +18.69pp")
print("=" * 78)
for thr in (0.08, 0.10, 0.12, 0.15, 0.17, 0.18, 0.19, 0.21, 0.25, 0.30):
    m = (spread63 >= thr).reindex(px.index, fill_value=False).fillna(False)
    trig = px.index[m.values]
    if len(trig) < 3:
        print(f"  {100*thr:5.0f}pp: too few triggers")
        continue
    epi = declusters(trig, 63, px.index)
    live = "*" if today >= thr else " "
    out = []
    for lbl, legs in FORMS.items():
        r = vehicle_ret(px, legs, 5).loc[epi].dropna()
        out.append(f"{lbl.split()[0]} {100*r.mean():+6.2f}%/med {100*r.median():+6.2f}%")
    print(f" {live}{100*thr:5.0f}pp  N_regimes={len(epi):3d}  " + "  ".join(out))

print("\n" + "=" * 78)
print("3. DROP APRIL-JULY 2020 (the negative-oil / USO restructuring break)")
print("=" * 78)
BASE = 0.15  # the largest round threshold that is live today
m = (spread63 >= BASE).reindex(px.index, fill_value=False).fillna(False)
trig = px.index[m.values]
for h in (5, 10):
    epi = declusters(trig, 63, px.index)
    keep = ~((epi >= "2020-03-01") & (epi <= "2020-08-31"))
    for lbl, legs in FORMS.items():
        r = vehicle_ret(px, legs, h).loc[epi].dropna()
        d = r.index
        k = ~((d >= "2020-03-01") & (d <= "2020-08-31"))
        rows = [summarize(r.values, f"{lbl} h={h} ALL (N={len(r)})"),
                summarize(r.values[k], f"{lbl} h={h} ex-2020 (N={int(k.sum())})")]
        rows[0]["sign_p"] = round(sign_test(int((r.values > 0).sum()), len(r)), 3)
        rows[1]["sign_p"] = round(
            sign_test(int((r.values[k] > 0).sum()), int(k.sum())), 3)
        show(rows)

print("\n" + "=" * 78)
print("4. CYCLE + ERA on regime episodes (h=5, base 15pp)")
print("=" * 78)
epi = declusters(trig, 63, px.index)
for lbl, legs in FORMS.items():
    r = vehicle_ret(px, legs, 5).loc[epi].dropna()
    d = pd.DatetimeIndex(r.index)
    v = r.values
    rows = [summarize(v, f"{lbl} all (N={len(v)})")]
    for sub, mm in (("pre-2018", np.asarray(d < pd.Timestamp("2018-01-01"))),
                    ("2018+", np.asarray(d >= pd.Timestamp("2018-01-01"))),
                    ("midterm", np.asarray(d.year % 4 == 2)),
                    ("non-midterm", np.asarray(d.year % 4 != 2))):
        if mm.sum():
            rows.append(summarize(v[mm], f"  {sub} (N={int(mm.sum())})"))
    show(rows)
    print(f"  {cluster_note(d, v)}")

print("\n" + "=" * 78)
print("5. TAPE over-selection (SPY below 200d on trigger days)")
print("=" * 78)
below = (px["SPY"] < px["SPY"].rolling(200).mean())
ok = below.notna()
print(f"  base rate {100*below[ok].mean():.1f}%   trigger days "
      f"{100*below[m.values & ok.values].mean():.1f}%  "
      f"(N={int((m.values & ok.values).sum())})")
