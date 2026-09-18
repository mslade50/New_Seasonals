"""K3R c6b - definition neighbours for the c6 cell (built from today's tape),
reported as a table rather than the best occupant.

Pair: short refiner vs its own lagged rolling-252 beta x XLE, lag=1 MOC.
Grid: USO 1d <= -2/-3/-4% x refiner within 0/1/3% of its trailing-252 close
high x refiner set {VLO} / {VLO,MPC,PSX} (any member qualifies; the pooled
pair shorts each qualifying member equally). Episodes filter-then-decluster
gap 10. h = 3, 5, 10. Also the refiner-vs-USO mechanism at h=5 per rung.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pd.set_option("display.width", 260)
BAR = pd.Timestamp("2026-09-16")
REF = ["VLO", "MPC", "PSX"]
full = close_panel(REF + ["XLE", "USO"])
full = full[full.index <= BAR]
D = full.index[full[["VLO", "XLE", "USO"]].notna().all(axis=1).values]
full = full.loc[D]
r1 = full.pct_change()
beta = {n: (r1[n].rolling(252).cov(r1["XLE"]) / r1["XLE"].rolling(252).var()).shift(1) for n in REF}
uso1 = full["USO"].pct_change()
prox = {n: full[n] / full[n].rolling(252).max() - 1.0 for n in REF}
COST = 7.0

rows = []
for h in (3, 5, 10):
    legret = {n: -fwd_lag(full[n], h) + beta[n] * fwd_lag(full["XLE"], h) for n in REF}
    mech = {n: fwd_lag(full[n], h) - fwd_lag(full["USO"], h) for n in REF}
    for uth in (-0.02, -0.03, -0.04):
        for pth in (0.0, -0.01, -0.03):
            for setlbl, names in (("VLO", ["VLO"]), ("VLO|MPC|PSX", REF)):
                q = pd.DataFrame({n: (prox[n] >= pth - 1e-12) for n in names})
                crude = uso1 <= uth
                trig = q.any(axis=1) & crude
                # pooled return: mean over qualifying members that day
                vals = pd.DataFrame({n: legret[n].where(q[n]) for n in names}).mean(axis=1)
                mv = pd.DataFrame({n: mech[n].where(q[n]) for n in names}).mean(axis=1)
                t = D[trig.values].intersection(vals.dropna().index)
                e = declusters(t, 10, D)
                v = vals.loc[e].values
                r = {"h": h, "uso<=": uth, "prox>=": pth, "set": setlbl, "n": len(v)}
                if len(v):
                    w = int((v > 0).sum())
                    r.update({"mean_pct": round(100 * v.mean(), 3), "rec": f"{w}-{len(v)-w}",
                              "sign_p": round(sign_test(w, len(v)), 3),
                              "x_cost": round(1e4 * v.mean() / COST, 2),
                              "mech_ref_minus_uso_pct": round(100 * np.nanmean(mv.loc[e].values), 3)})
                rows.append(r)
df = pd.DataFrame(rows)
for h in (3, 5, 10):
    print(f"\n=== neighbours h={h} (short refiner vs beta*XLE; mech>0 = refiner kept outrunning crude) ===")
    print(df[df["h"] == h].to_string(index=False))
sub = df[df["n"] > 0]
for h in (3, 5, 10):
    s = sub[sub["h"] == h]
    print(f"h={h}: {int((s['mean_pct'] > 0).sum())} of {len(s)} non-empty neighbour cells positive; "
          f"mechanism positive (refiner outran crude) in {int((s['mech_ref_minus_uso_pct'] > 0).sum())} of {len(s)}")
