"""C7 round 1+2: FADE the USO thrust out of a deep 63d base.

State: USO 5d rank >= 90 AND 63d rank <= 20 (trailing 252d ranks), which is
today's tape (5d rank 90.1, 63d rank 9.5). Recon: N=101 days / 43 episodes,
forward excess -0.350 / -0.544 / -0.664 at h=1/3/5, hit 46.5/46.5/48.8%.

Kill angles:
  A. GATE ATTRIBUTION -- run it WITHOUT the 63d gate, and with the 63d gate
     alone. A nested subset that reverses its parent's sign is a partition of
     noise (registry).
  B. ROLL DECAY, the brief's own question. USO's own-drift control removes the
     AVERAGE roll drag. Test whether the cell survives in vehicles with NO
     roll: XLE and XOP traded on the SAME USO-defined trigger days.
  C. definition neighbours: nudge both thresholds.
  D. concentration, year histogram, era, midterm split.
  E. the event inside the hold. Tonight's entry holds tomorrow's PPI at every
     horizon; the registry's own finding is that the event flips the crude
     cell's sign.
  F. book overlap: the systematic book is short crude-spike names already.
  G. cost + the tail of a naked short on a commodity thrust.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

px = close_panel(["USO", "XLE", "XOP", "DBC"])
idx = px.index

r5 = pct_rank(px["USO"], 5)
r63 = pct_rank(px["USO"], 63)
print(f"today's bar 2026-08-11: USO rank5={r5.iloc[-1]:.1f}  "
      f"rank63={r63.iloc[-1]:.1f}  1d={100*(px['USO'].iloc[-1]/px['USO'].iloc[-2]-1):+.2f}%  "
      f"5d={100*(px['USO'].iloc[-1]/px['USO'].iloc[-6]-1):+.2f}%")

M = (r5 >= 90) & (r63 <= 20)
M = M.fillna(False)
print(f"trigger days: {int(M.sum())}  today fires: {bool(M.iloc[-1])}")

# ------------------------------------------------------- A. gate attribution
print("\n=== A. GATE ATTRIBUTION (episode level, min_gap=5) ===")
gates = {
    "BOTH: r5>=90 & r63<=20 (the cell)": M,
    "PARENT: r5>=90 alone (no 63d gate)": (r5 >= 90).fillna(False),
    "PARENT: r63<=20 alone (no thrust)": (r63 <= 20).fillna(False),
    "COMPLEMENT: r5>=90 & r63>20": ((r5 >= 90) & (r63 > 20)).fillna(False),
}
for h in (1, 3, 5):
    rows = []
    ret = vehicle_ret(px, [("USO", -1.0)], h, 1)
    base = ret.dropna()
    for lbl, g in gates.items():
        t = idx[g.reindex(idx, fill_value=False).values].intersection(base.index)
        epi = declusters(t, 5, base.index)
        v = ret.loc[epi].values
        if len(v) < 3:
            rows.append({"label": lbl, "n": len(v)})
            continue
        w = int((v > 0).sum())
        rows.append({"label": lbl, "n_days": len(t), "n_epi": len(epi),
                     "short_mean_pct": round(100*v.mean(), 3),
                     "excess_pct": round(100*(v.mean()-base.mean()), 3),
                     "hit": round(100*(v > 0).mean(), 1),
                     "sign_p": round(sign_test(w, len(v),
                                               float((base > 0).mean())), 4),
                     "worst_pct": round(100*v.min(), 2)})
    show(rows, f"SHORT USO, h={h}")

# --------------------------------------------------------- B. roll decay
print("\n=== B. ROLL DECAY: same USO-defined trigger, no-roll vehicles ===")
print("  (if the fade is a real market effect it survives in XLE/XOP; if it is")
print("   USO's contango bleed it does not)")
for h in (1, 3, 5):
    rows = []
    for tkr in ("USO", "XLE", "XOP", "DBC"):
        ret = vehicle_ret(px, [(tkr, -1.0)], h, 1)
        base = ret.dropna()
        t = idx[M.reindex(idx, fill_value=False).values].intersection(base.index)
        epi = declusters(t, 5, base.index)
        v = ret.loc[epi].values
        if len(v) < 3:
            continue
        w = int((v > 0).sum())
        rows.append({"vehicle": tkr, "n_epi": len(epi),
                     "short_mean_pct": round(100*v.mean(), 3),
                     "own_drift_pct": round(100*base.mean(), 3),
                     "excess_pct": round(100*(v.mean()-base.mean()), 3),
                     "hit": round(100*(v > 0).mean(), 1),
                     "sign_p": round(sign_test(w, len(v),
                                               float((base > 0).mean())), 4)})
    show(rows, f"h={h}")
# how big is USO's structural bleed vs XLE over the same span?
for tkr in ("USO", "XLE", "XOP", "DBC"):
    s = px[tkr].dropna()
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    print(f"  {tkr}: {100*((s.iloc[-1]/s.iloc[0])**(1/yrs)-1):+.2f}%/yr "
          f"structural since {s.index[0].date()}")

# ------------------------------------------------------------ full battery
variants = {
    "r5>=85 & r63<=20": ((r5 >= 85) & (r63 <= 20)).fillna(False),
    "r5>=95 & r63<=20": ((r5 >= 95) & (r63 <= 20)).fillna(False),
    "r5>=90 & r63<=10": ((r5 >= 90) & (r63 <= 10)).fillna(False),
    "r5>=90 & r63<=30": ((r5 >= 90) & (r63 <= 30)).fillna(False),
    "r5>=90 & r63<=15": ((r5 >= 90) & (r63 <= 15)).fillna(False),
    "r5>=90 alone": (r5 >= 90).fillna(False),
}
for h in (1, 3, 5):
    battery(px, M, [("USO", -1.0)], h,
            f"C7 SHORT USO on the deep-base thrust, h={h}",
            cost_bps=8.0, variants=variants, min_gap=5,
            event_kinds=("cpi", "ppi"))

# ----------------------------------------- D. concentration / era / midterm
print("\n=== D. episode detail (h=3, SHORT USO) ===")
ret = vehicle_ret(px, [("USO", -1.0)], 3, 1)
t = idx[M.values].intersection(ret.dropna().index)
epi = declusters(t, 5, ret.dropna().index)
v = ret.loc[epi].values
yr = pd.Series(100*v, index=epi).groupby(epi.year).agg(["sum", "count", "mean"])
print(yr.round(2).to_string())
print(f"  positive years {int((yr['sum']>0).sum())}/{len(yr)}")
mid = np.array([y % 4 == 2 for y in epi.year])
show([summarize(v[mid], f"midterm (N={int(mid.sum())})"),
      summarize(v[~mid], f"non-midterm (N={int((~mid).sum())})")], "midterm split")
print(f"  {cluster_note(epi, v, k=3)}")
print("  episodes:", ", ".join(f"{d.date()}:{100*x:+.1f}"
                               for d, x in zip(epi, v)))

# ------------------------------------------------------------- F. book overlap
print("\n=== F. book overlap ===")
try:
    led = pd.read_parquet("data/backtest_trades_full.parquet")
    led["Signal_Date"] = pd.to_datetime(led["Signal_Date"])
    en = led[led["Ticker"].isin(["USO", "XLE", "XOP", "DBC"])]
    print(f"  ledger energy trades: {len(en)}")
    print(en.groupby(["Strategy_Name", "Direction"]).agg(
        n=("R_Multiple", "size"), avgR=("R_Multiple", "mean")).round(2).to_string())
    trig_set = set(idx[M.values])
    on_trig = en[en["Signal_Date"].isin(trig_set)]
    print(f"\n  book trades signalled ON a C7 trigger day: {len(on_trig)}")
    if len(on_trig):
        print(on_trig.groupby(["Strategy_Name", "Direction", "Ticker"]).agg(
            n=("R_Multiple", "size"), avgR=("R_Multiple", "mean")).round(2).to_string())
except Exception as e:  # noqa
    print(f"  ledger unavailable: {e}")
