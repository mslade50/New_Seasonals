"""C5 mandatory blockers: separability from the dial + VIX Range Compression,
the reference class (other 7 fragility signals), the SVXY leverage break,
definition fragility (five neighbour definitions), year histogram, midterm."""
import sys
from math import erf, sqrt
from pathlib import Path

ROOTP = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOTP))
sys.path.insert(0, str(ROOTP / "scripts"))
sys.path.insert(0, str(ROOTP / "pages"))
import numpy as np
import pandas as pd
from pitch_lab import *  # noqa

DAY = Path(__file__).resolve().parent
masks = pd.read_parquet(DAY / "_sigmask_cache.parquet")
pctd = pd.read_parquet(DAY / "_disp_pctile.parquet")
comp = pctd["composite"].dropna()

px = close_panel(["SVXY", "^VIX", "SPY"])
idx = px.index
disp_raw = (comp > 85).reindex(idx, fill_value=False)
disp_prod = masks["Dispersion"].reindex(idx, fill_value=False).astype(bool)
vrc = masks["VIX Range Compression"].reindex(idx, fill_value=False).astype(bool)
ret10 = vehicle_ret(px, [("SVXY", 1.0)], 10, 1)

# ---------------------------------------------------------------- (a) dial
fr = pd.read_parquet(ROOTP / "data" / "rd2_fragility.parquet")
dial = fr["63d"].rolling(10).mean().reindex(idx)   # THE sizing statistic
print("=== BLOCKER A: separability from the fragility dial (ma10 of the 63d col) ===")
print("VINTAGE: rd2_fragility.parquet is point-in-time append-only only since")
print("2026-07-02; earlier rows are the recompute vintage. Both used. Dial starts")
print("2016-07 so this covers 2016+ triggers only.")
ok = dial.notna()
for lbl, m in [("Dispersion raw comp>85", disp_raw), ("Dispersion PROD signal", disp_prod)]:
    sel = m & ok
    if sel.sum() == 0:
        print(f"  {lbl}: no overlap window")
        continue
    d = dial[sel]
    base = dial[ok]
    print(f"  {lbl}: N={int(sel.sum())} trigger days with a dial | mean dial "
          f"{d.mean():.1f} vs all-days {base.mean():.1f} | P(dial>=50|trig)="
          f"{100*(d>=50).mean():.1f}% vs base {100*(base>=50).mean():.1f}% | "
          f"P(dial>=85|trig)={100*(d>=85).mean():.1f}% vs base "
          f"{100*(base>=85).mean():.1f}%")
print(f"  today dial ma10-63d = {dial.dropna().iloc[-1]:.1f} (surface map 87.8)")

sel = disp_raw & ok & ret10.notna()
hi = sel & (dial >= 50)
lo = sel & (dial < 50)
nd = ok & ret10.notna() & ~disp_raw
show([summarize(ret10[hi].values, f"disp & dial>=50 (N={int(hi.sum())})"),
      summarize(ret10[lo].values, f"disp & dial<50  (N={int(lo.sum())})"),
      summarize(ret10[nd & (dial >= 50)].values, "no disp & dial>=50"),
      summarize(ret10[nd & (dial < 50)].values, "no disp & dial<50")],
     "long SVXY h=10, day level, dial x dispersion 2x2 (2016+)")

# ---------------------------------------------------------------- (b) VRC
print("\n=== BLOCKER B: separability from VIX Range Compression ===")
both = disp_raw & vrc
print(f"  disp raw>85 days={int(disp_raw.sum())} VRC days={int(vrc.sum())} "
      f"both={int(both.sum())} -> P(VRC|disp)="
      f"{100*both.sum()/max(1, disp_raw.sum()):.1f}%  P(disp|VRC)="
      f"{100*both.sum()/max(1, vrc.sum()):.1f}%")
print(f"  PROD disp days={int(disp_prod.sum())} both={int((disp_prod & vrc).sum())} "
      f"-> P(VRC|PROD disp)="
      f"{100*(disp_prod & vrc).sum()/max(1, disp_prod.sum()):.1f}%")
v = ret10.notna()
show([summarize(ret10[disp_raw & vrc & v].values, "disp & VRC"),
      summarize(ret10[disp_raw & ~vrc & v].values, "disp & not VRC"),
      summarize(ret10[~disp_raw & vrc & v].values, "VRC only"),
      summarize(ret10[~disp_raw & ~vrc & v].values, "neither")],
     "long SVXY h=10, day level, dispersion x VIX-range-compression 2x2")

# --------------------------------------------------- (c) the reference class
print("\n=== BLOCKER C: reference class = the other seven fragility signals ===")
print("Identical rule for every member: LONG SVXY, lag=1, h=10, episodes min_gap=10")
rows, eff, var, names = [], [], [], []
base_all = ret10.dropna()
for name in masks.columns:
    m = masks[name].reindex(idx, fill_value=False).astype(bool)
    sig = idx[m.values & ret10.notna().values]
    if len(sig) < 3:
        rows.append({"signal": name, "n_epi": len(sig)})
        continue
    epi = declusters(sig, 10, idx)
    s = summarize(ret10.loc[epi].values, name)
    se = s["sd_pct"] / np.sqrt(s["n"])
    rows.append({"signal": name, "n_epi": s["n"], "mean_pct": round(s["mean_pct"], 3),
                 "edge_pp": round(s["mean_pct"] - 100 * base_all.mean(), 3),
                 "se": round(se, 3), "hit": round(s["hit"], 1), "t": round(s["t"], 2)})
    eff.append(s["mean_pct"])
    var.append(se ** 2)
    names.append(name)
df = pd.DataFrame(rows).sort_values("mean_pct", ascending=False)
print(df.to_string(index=False))
eff = np.array(eff)
var = np.array(var)
w = 1 / var
mu = (w * eff).sum() / w.sum()
Q = float((w * (eff - mu) ** 2).sum())
dfree = len(eff) - 1
I2 = max(0.0, (Q - dfree) / Q) * 100 if Q > 0 else 0.0
pQ = 1 - 0.5 * (1 + erf((Q - dfree) / sqrt(2 * 2 * dfree)))
dm = df.loc[df["signal"] == "Dispersion", "mean_pct"].iloc[0]
rank = int((df["mean_pct"] > dm).sum()) + 1
print(f"  pooled {mu:.3f}%  Cochran Q={Q:.2f} on {dfree} df (normal-approx "
      f"p~{pQ:.3f})  I^2={I2:.1f}%  Dispersion ranks {rank} of "
      f"{int(df['mean_pct'].notna().sum())}")

# ------------------------------------------- (d) SVXY leverage break + years
print("\n=== BLOCKER D: SVXY leverage break 2018-02-28 (-1x -> -0.5x) ===")
sig = idx[disp_raw.values & ret10.notna().values]
epi = declusters(sig, 10, idx)
cut = pd.Timestamp("2018-02-28")
pre, post = epi[epi < cut], epi[epi >= cut]
show([summarize(ret10.loc[pre].values, f"pre 2018-02-28 (-1x) N={len(pre)}"),
      summarize(ret10.loc[post].values, f"post 2018-02-28 (-0.5x) N={len(post)}")],
     "long SVXY h=10 episodes across the leverage change")
ctrl_post = ret10[idx >= cut].dropna()
print(f"  post-change all-days control {100*ctrl_post.mean():.3f}% -> post-change "
      f"edge vs drift {100*(ret10.loc[post].mean()-ctrl_post.mean()):+.3f}pp")
yrs = pd.Series(1, index=epi).groupby(epi.year).sum()
print("  episode year histogram:", yrs.to_dict())
mid = epi[epi.year % 4 == 2]
non = epi[epi.year % 4 != 2]
print(f"\n=== BLOCKER E: midterm cross === midterm N={len(mid)} "
      f"{100*ret10.loc[mid].mean():+.3f}% vs non-midterm N={len(non)} "
      f"{100*ret10.loc[non].mean():+.3f}%")

# ---------------------------------------------- (f) definition fragility
print("\n=== BLOCKER F: definition fragility, five neighbour definitions ===")
from build_atr_downside_stats import build_inputs_from_master  # noqa: E402
spy_df, closes, sp500 = build_inputs_from_master()
spy_close = spy_df["Close"]
crv = sp500.pct_change().rolling(21, min_periods=11).std() * np.sqrt(252)
avail = crv.notna().sum(axis=1)
acrv = crv.mean(axis=1).where(avail >= 50)
srv = spy_close.pct_change().rolling(21, min_periods=11).std() * np.sqrt(252)
ci = acrv.dropna().index.intersection(srv.dropna().index)
ratio = acrv.reindex(ci) / srv.reindex(ci).replace(0, np.nan)
gap = acrv.reindex(ci) - srv.reindex(ci)


def rp(s, lb):
    return s.dropna().rolling(lb).rank(pct=True) * 100


defs = {
    "D1 prod composite (ratio+gap)/2 lb504": comp,
    "D2 ratio-only pctile lb504": rp(ratio, 504),
    "D3 composite lb252": (rp(ratio, 252) + rp(gap, 252)) / 2,
    "D4 MEDIAN component RV ratio lb504":
        rp(crv.median(axis=1).where(avail >= 50).reindex(ci)
           / srv.reindex(ci).replace(0, np.nan), 504),
}
r21 = sp500.pct_change(21).abs()
nom = (r21.mean(axis=1).where(r21.count(axis=1) >= 50)
       - spy_close.pct_change(21).abs())
defs["D5 Nomura abs-return dispersion lb504"] = rp(nom, 504)
out = []
for lbl, s in defs.items():
    m = (s.reindex(idx) > 85).fillna(False)
    sg = idx[m.values & ret10.notna().values]
    if len(sg) < 3:
        out.append({"label": lbl, "n": 0})
        continue
    e = declusters(sg, 10, idx)
    r = summarize(ret10.loc[e].values, lbl)
    r["n_days"] = len(sg)
    r["edge_pp"] = round(r["mean_pct"] - 100 * base_all.mean(), 3)
    r["today"] = round(float(s.dropna().iloc[-1]), 1)
    out.append(r)
show(out, "long SVXY h=10 episodes, one row per dispersion DEFINITION (>85)")
