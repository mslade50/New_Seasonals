"""C9 round 2: reference class, decluster/concentration, era & midterm split,
and the definition-neighbour sweep on BOTH sides of the 52w-high rule.

The brief's requirement: "testing the reference class (XOP, OIH, and the top
XLE components) so a single ETF's number has to survive its peer group."

Two forms are tested per vehicle:
  (i)  the vehicle's OWN 52w high + crude floor (the honest peer test)
  (ii) XLE's 52w high + crude floor, traded in the peer (the co-movement test)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

ASOF = pd.Timestamp("2026-08-18")
PEERS = ["XLE", "XOP", "OIH", "XOM", "CVX", "COP", "EOG", "SLB", "VLO",
         "OXY", "HAL", "DVN", "WMB", "OKE", "BKR", "MPC", "PSX", "KMI", "FANG"]
px = close_panel(PEERS + ["USO", "CL=F", "SPY"])
idx = px.index
H, GAP = 10, 21

crude_cl = pct_rank(px["CL=F"].dropna(), 63).reindex(idx)
crude_uso = pct_rank(px["USO"].dropna(), 63).reindex(idx)


def at_52wh(t):
    s = px[t].dropna()
    hi = s.rolling(252).max().reindex(idx)
    return (px[t] >= hi * 0.99999).fillna(False)


xle_hi = at_52wh("XLE")

print("=" * 78)
print("REFERENCE CLASS (i): each peer at ITS OWN 52w high + CL=F 63d rank<=15")
print("=" * 78)
rows = []
for t in PEERS:
    r = vehicle_ret(px, [(t, 1.0)], H, 1)
    m = at_52wh(t) & (crude_cl <= 15).fillna(False)
    d = idx[m.values & r.notna().values]
    e = declusters(d, GAP, idx)
    # ungated 52w-high baseline for the same peer
    d0 = idx[at_52wh(t).values & r.notna().values]
    e0 = declusters(d0, GAP, idx)
    if len(e) == 0:
        rows.append({"vehicle": t, "n": 0, "n_ungated": len(e0)})
        continue
    s = summarize(r.loc[e].values)
    w = int((r.loc[e] > 0).sum())
    rows.append({"vehicle": t, "n": len(e), "mean_pct": round(s["mean_pct"], 3),
                 "hit": round(s["hit"], 1),
                 "n_ungated": len(e0),
                 "ungated_mean_pct": round(100 * r.loc[e0].mean(), 3),
                 "gate_adds_pp": round(s["mean_pct"] - 100 * r.loc[e0].mean(), 3),
                 "sign_p": round(sign_test(w, len(e)), 4)})
df = pd.DataFrame(rows)
print(df.to_string(index=False))
ok = df.dropna(subset=["mean_pct"]) if "mean_pct" in df else df
if len(ok):
    print(f"\npeers with a POSITIVE gated mean: "
          f"{int((ok['mean_pct'] > 0).sum())} of {len(ok)} "
          f"({100*(ok['mean_pct'] > 0).mean():.1f}%)")
    print(f"peers where the crude gate ADDS to their own 52w-high state: "
          f"{int((ok['gate_adds_pp'] > 0).sum())} of {len(ok)} "
          f"({100*(ok['gate_adds_pp'] > 0).mean():.1f}%)   "
          f"median add {ok['gate_adds_pp'].median():+.3f}pp")

print("\n" + "=" * 78)
print("REFERENCE CLASS (ii): XLE's signal, traded in each peer")
print("=" * 78)
rows = []
base = xle_hi & (crude_cl <= 15).fillna(False)
for t in PEERS:
    r = vehicle_ret(px, [(t, 1.0)], H, 1)
    d = idx[base.values & r.notna().values]
    e = declusters(d, GAP, idx)
    if len(e) == 0:
        rows.append({"vehicle": t, "n": 0})
        continue
    s = summarize(r.loc[e].values)
    w = int((r.loc[e] > 0).sum())
    rows.append({"vehicle": t, "n": len(e), "mean_pct": round(s["mean_pct"], 3),
                 "hit": round(s["hit"], 1), "worst_pct": round(s["worst_pct"], 2),
                 "sign_p": round(sign_test(w, len(e)), 4)})
df2 = pd.DataFrame(rows)
print(df2.to_string(index=False))
ok2 = df2.dropna(subset=["mean_pct"]) if "mean_pct" in df2 else df2
if len(ok2):
    print(f"\npeers with a POSITIVE mean on XLE's signal: "
          f"{int((ok2['mean_pct'] > 0).sum())} of {len(ok2)} "
          f"({100*(ok2['mean_pct'] > 0).mean():.1f}%)")

print("\n" + "=" * 78)
print("CONCENTRATION / ERA / MIDTERM on the XLE gated cell")
print("=" * 78)
r = vehicle_ret(px, [("XLE", 1.0)], H, 1)
e = declusters(idx[base.values & r.notna().values], GAP, idx)
v = r.loc[e].values
print(f"episodes N={len(v)}  dates: {', '.join(str(d.date()) for d in e)}")
print(" ", cluster_note(e, v))
print(f"  bootstrap P(mean<=0) = {bootstrap_p_le0(v):.3f}")
mid = np.array([d.year % 4 == 2 for d in e])
show([summarize(v[mid], f"midterm (N={int(mid.sum())})"),
      summarize(v[~mid], f"non-midterm (N={int((~mid).sum())})")], "midterm split")

print("\n" + "=" * 78)
print("DEFINITION NEIGHBOURS on the 52w-high side (crude gate held at CL<=15)")
print("=" * 78)
s_xle = px["XLE"].dropna()
rows = []
for lbk in (126, 189, 252, 378, 504):
    hi = s_xle.rolling(lbk).max().reindex(idx)
    m = (px["XLE"] >= hi * 0.99999).fillna(False) & (crude_cl <= 15).fillna(False)
    d = idx[m.values & r.notna().values]
    ee = declusters(d, GAP, idx)
    if len(ee) == 0:
        rows.append({"lookback_high": lbk, "n": 0})
        continue
    st = summarize(r.loc[ee].values)
    rows.append({"lookback_high": lbk, "n": len(ee),
                 "mean_pct": round(st["mean_pct"], 3), "hit": round(st["hit"], 1)})
show(rows, "high lookback sensitivity")

rows = []
for g in (5, 10, 15, 21, 42, 63):
    m = xle_hi & (crude_cl <= 15).fillna(False)
    d = idx[m.values & r.notna().values]
    ee = declusters(d, g, idx)
    st = summarize(r.loc[ee].values)
    rows.append({"decluster_gap_td": g, "n": len(ee),
                 "mean_pct": round(st["mean_pct"], 3), "hit": round(st["hit"], 1)})
show(rows, "decluster-gap sensitivity")

print("\n" + "=" * 78)
print("CRUDE-PROXY SENSITIVITY: the thesis says 'the barrel', the tape read")
print("says USO. They disagree TODAY.")
print("=" * 78)
print(f"  CL=F 63d rank {crude_cl.loc[ASOF]:.1f}  ->  today FAILS a <=15 gate")
print(f"  USO  63d rank {crude_uso.loc[ASOF]:.1f}  ->  today PASSES a <=15 gate")
agree = ((crude_cl <= 15) == (crude_uso <= 15)).reindex(idx)
both = crude_cl.notna() & crude_uso.notna()
print(f"  the two proxies agree on the <=15 gate on "
      f"{100*agree[both].mean():.1f}% of {int(both.sum())} shared days")
