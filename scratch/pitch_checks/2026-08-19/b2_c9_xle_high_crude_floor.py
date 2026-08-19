"""C9 round 1: LONG XLE at a fresh 52-week high while crude's 63-day return
rank sits at a floor (~6 today).

Thesis to falsify: producers making new highs while the barrel is at a 63-day
floor means the equity is pricing margin / capital return rather than the
commodity, and that leadership persists.

The falsification design the brief demands:
  (a) price XLE's own unconditional drift AT a 52-week high -- the 52w-high
      state is a documented momentum state, so the CRUDE GATE has to EARN
      its place. Run it gate-off.
  (b) magnitude gradient AT today's reading, not the pooled mean.
  (c) reference class: XOP / OIH / top XLE components, so one ETF's number
      has to survive its peer group. (b2b.)

Crude read: CL=F (the barrel, history from 2000-08) is primary; USO
(2006-04) is the robustness copy, and is what the surface map quotes.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

ASOF = pd.Timestamp("2026-08-18")
NAMES = ["XLE", "USO", "CL=F", "XOP", "OIH", "XOM", "CVX", "COP", "EOG",
         "SLB", "VLO", "OXY", "HAL", "DVN", "WMB", "OKE", "BKR", "MPC",
         "PSX", "KMI", "FANG", "SPY"]
px = close_panel(NAMES)
idx = px.index

# NOTE: close_panel's index is the UNION of every ticker's calendar, so a
# ticker with a different calendar (CL=F trades some equity holidays) leaves
# NaN rows in XLE. rolling(252).max() with default min_periods returns NaN
# for any window containing one -- which silently reported "XLE is not at a
# 52-week high" on a day it closed exactly at one. Every rolling statistic
# here is computed on the DROPPED series and reindexed back.
xle = px["XLE"]
xle_c = xle.dropna()
roll_hi = xle_c.rolling(252).max().reindex(idx)
at_hi = (xle >= roll_hi * 0.99999).fillna(False)           # strict fresh 52w high
near_hi = {p: (xle >= roll_hi * (1 - p)).fillna(False) for p in (0.005, 0.01, 0.02)}

crude_cl = pct_rank(px["CL=F"].dropna(), 63).reindex(idx)
crude_uso = pct_rank(px["USO"].dropna(), 63).reindex(idx)
xle_r63 = xle_c.pct_change(63).reindex(idx)
uso_r63 = px["USO"].dropna().pct_change(63).reindex(idx)
cl_r63 = px["CL=F"].dropna().pct_change(63).reindex(idx)

print("== live readings", ASOF.date(), "==")
print(f"XLE close {xle.loc[ASOF]:.2f}  252d max {roll_hi.loc[ASOF]:.2f}  "
      f"at 52w high = {bool(at_hi.loc[ASOF])}")
print(f"CL=F 63d rank {crude_cl.loc[ASOF]:.1f}   USO 63d rank {crude_uso.loc[ASOF]:.1f}")
print(f"XLE 63d {100*xle_r63.loc[ASOF]:+.2f}%  "
      f"CL=F 63d {100*cl_r63.loc[ASOF]:+.2f}%  "
      f"USO 63d {100*uso_r63.loc[ASOF]:+.2f}%")
print(f"n days XLE at a fresh 52w high in history: {int(at_hi.sum())}")

H = 10
LEGS = [("XLE", 1.0)]
GAP = 21

base = at_hi & (crude_cl <= 15).fillna(False)

variants = {}
for g in (5, 10, 15, 20, 25, 33, 50, 100):
    variants[f"52wh + CL63rank<={g}"] = at_hi & (crude_cl <= g).fillna(False)
variants["52wh, NO crude gate"] = at_hi
for p, m in near_hi.items():
    variants[f"within {p*100:.1f}% of 52wh + CL<=15"] = m & (crude_cl <= 15).fillna(False)
variants["52wh + USO63rank<=15"] = at_hi & (crude_uso <= 15).fillna(False)
variants["52wh + USO63rank<=25"] = at_hi & (crude_uso <= 25).fillna(False)

battery(px, base, LEGS, H,
        "C9  LONG XLE at a fresh 52w high, CL=F 63d rank <= 15",
        cost_bps=2.0, variants=variants, min_gap=GAP)

# ------------------------------------------------------- gate attribution
print("\n" + "=" * 78)
print("GATE ATTRIBUTION: what does the crude floor ADD to plain 52w-high XLE?")
print("=" * 78)
ret = vehicle_ret(px, LEGS, H, 1)
valid = ret.notna()
hi_days = idx[at_hi.values & valid.values]
hi_epi = declusters(hi_days, GAP, idx)
base_mean = ret.loc[hi_epi].mean()
rows = [{"cell": "ALL 52w-high episodes (no crude gate)", "n": len(hi_epi),
         "mean_pct": round(100 * base_mean, 3),
         "hit": round(100 * (ret.loc[hi_epi] > 0).mean(), 1),
         "t": round(summarize(ret.loc[hi_epi].values)["t"], 2),
         "vs_52wh_pp": 0.0}]
for g in (5, 10, 15, 20, 25, 33, 50):
    d = idx[(at_hi & (crude_cl <= g).fillna(False)).values & valid.values]
    e = declusters(d, GAP, idx)
    if len(e) == 0:
        rows.append({"cell": f"52wh + CL63<= {g}", "n": 0})
        continue
    s = summarize(ret.loc[e].values)
    w = int((ret.loc[e] > 0).sum())
    rows.append({"cell": f"52wh + CL63<={g}", "n": len(e),
                 "mean_pct": round(s["mean_pct"], 3), "hit": round(s["hit"], 1),
                 "t": round(s["t"], 2) if not np.isnan(s["t"]) else np.nan,
                 "vs_52wh_pp": round(s["mean_pct"] - 100 * base_mean, 3),
                 "sign_p": round(sign_test(w, len(e)), 4)})
show(rows, "crude gate vs the plain 52w-high momentum state")

# ------------------------------------------- magnitude gradient at TODAY
print("\n" + "=" * 78)
print("MAGNITUDE GRADIENT: XLE 52w-high episodes bucketed by the crude rank")
print("(today's reading is CL=F rank %.1f / USO rank %.1f -- read the LOWEST bucket,"
      % (crude_cl.loc[ASOF], crude_uso.loc[ASOF]))
print(" not the pooled mean)")
print("=" * 78)
for lbl, cr in [("CL=F", crude_cl), ("USO", crude_uso)]:
    rows = []
    edges = [(0, 10), (10, 20), (20, 35), (35, 50), (50, 65), (65, 80), (80, 101)]
    for lo, hi in edges:
        m = at_hi & ((cr >= lo) & (cr < hi)).fillna(False)
        d = idx[m.values & valid.values]
        e = declusters(d, GAP, idx)
        if len(e) == 0:
            rows.append({"bucket": f"[{lo},{hi})", "n": 0})
            continue
        s = summarize(ret.loc[e].values, f"[{lo},{hi})")
        w = int((ret.loc[e] > 0).sum())
        rows.append({"bucket": f"[{lo},{hi})", "n": len(e),
                     "mean_pct": round(s["mean_pct"], 3), "hit": round(s["hit"], 1),
                     "worst_pct": round(s["worst_pct"], 2),
                     "sign_p": round(sign_test(w, len(e)), 4)})
    show(rows, f"crude read = {lbl} 63d rank, XLE 52w-high episodes, h={H}")

# ---------------------------------------------------- spread form contrast
print("\n" + "=" * 78)
print("THE SPREAD FORM (killed 2026-08-14) vs the OUTRIGHT, same days")
print("=" * 78)
sp = (xle_r63 - uso_r63)
print(f"today's XLE-USO 63d spread = {100*sp.loc[ASOF]:+.2f}pp")
for thr in (10, 15, 17, 18, 20, 25):
    m = at_hi & (sp >= thr / 100.0).fillna(False)
    d = idx[m.values & valid.values]
    e = declusters(d, GAP, idx)
    if len(e) == 0:
        print(f"  spread>={thr}pp & 52wh: no episodes")
        continue
    s = summarize(ret.loc[e].values)
    print(f"  spread>={thr}pp & 52wh: N={len(e):3d} outright-XLE mean {s['mean_pct']:+.3f}% "
          f"hit {s['hit']:.1f}% t={s['t']:+.2f}")

# ------------------------------------------------------------ horizon peek
print()
show(horizon_scan(px, declusters(idx[base.values & valid.values], GAP, idx),
                  LEGS, hs=(1, 2, 3, 5, 7, 10), min_gap=GAP),
     "horizon peek, gated cell (CL63<=15 + 52wh)")
