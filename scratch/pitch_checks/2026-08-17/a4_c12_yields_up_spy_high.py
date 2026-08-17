"""C12 round 1 - yields rising while SPY sits at a 52w high.

This is a CONDITIONER check, so it is built as one: the question is not whether
SPY at a 52w high goes up, it is whether adding the rates state changes anything.
Every cell is therefore reported beside its own gate-OFF parent, and a gate that
does not move the parent is a gate that does not filter.

Two candidate definitions of "yields rising", both pre-registered here before
looking: (A) pct_rank(^TNX, 21) >= 70, (B) TLT within 1.0% of its 52w low. The
cleaner one is the one whose PARENT is not already the whole story.

Forward legs: SPY (the equity read) and TLT (the rates read).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

BAR = pd.Timestamp("2026-08-14")
px = close_panel(["SPY", "TLT", "^TNX"]).dropna()
IDX = px.index
spy, tlt, tnx = px["SPY"], px["TLT"], px["^TNX"]

off_hi = (spy / spy.rolling(252).max() - 1.0) * 100
off_lo = (tlt / tlt.rolling(252).min() - 1.0) * 100
r21 = pct_rank(tnx, 21)
print(f"today ({BAR.date()}): SPY {off_hi.loc[BAR]:.2f}% off 52w high | "
      f"^TNX {tnx.loc[BAR]:.2f}, 21d rank {r21.loc[BAR]:.1f} | "
      f"TLT {off_lo.loc[BAR]:.2f}% off 52w low")

NEAR_HI = off_hi >= -0.5
GATE_A = r21 >= 70
GATE_B = off_lo <= 1.0
print(f"\nday counts: near-high {int(NEAR_HI.sum())} | gateA {int(GATE_A.sum())} | "
      f"gateB {int(GATE_B.sum())} | A&near {int((NEAR_HI & GATE_A).sum())} | "
      f"B&near {int((NEAR_HI & GATE_B).sum())} | A&B&near "
      f"{int((NEAR_HI & GATE_A & GATE_B).sum())}")

# ---------------------------------------------------------------- the grid
print("\n" + "=" * 96)
print("1. GATE ATTRIBUTION GRID - parent, gated, and gate-alone, both legs")
print("=" * 96)
for leg, lbl in ((("SPY", 1.0), "long SPY"), (("TLT", 1.0), "long TLT")):
    for h in (5, 10):
        f = fwd_lag(px[leg[0]], h, 1)
        valid = f.notna()
        rows = []
        for name, m in (("CTRL all days", pd.Series(True, index=IDX)),
                        ("PARENT SPY near 52w high", NEAR_HI),
                        ("GATE-A alone (TNX r21>=70)", GATE_A),
                        ("GATE-B alone (TLT<=1% of low)", GATE_B),
                        ("near-high x GATE-A", NEAR_HI & GATE_A),
                        ("near-high x GATE-B", NEAR_HI & GATE_B),
                        ("near-high x A x B (live)", NEAR_HI & GATE_A & GATE_B),
                        ("near-high x NOT A", NEAR_HI & ~GATE_A)):
            sel = IDX[(m & valid).values]
            e = declusters(sel, 10, IDX)
            r = summarize(f.reindex(e).dropna().values, f"{name} [epi]")
            r["n_days"] = len(sel)
            rows.append(r)
        show(rows, f"{lbl} h={h} (episodes, min_gap 10td)")
        par = f.reindex(declusters(IDX[(NEAR_HI & valid).values], 10, IDX)).dropna()
        for name, m in (("A", NEAR_HI & GATE_A), ("B", NEAR_HI & GATE_B),
                        ("AxB", NEAR_HI & GATE_A & GATE_B)):
            v = f.reindex(declusters(IDX[(m & valid).values], 10, IDX)).dropna()
            if len(v) < 2:
                print(f"  gate {name}: N={len(v)} - nothing to attribute")
                continue
            w = int((v > 0).sum())
            print(f"  gate {name}: {100*v.mean():+.3f}% vs parent "
                  f"{100*par.mean():+.3f}% -> gate moves it "
                  f"{100*(v.mean()-par.mean()):+.3f}pp | {w}-{len(v)-w} "
                  f"sign p {sign_test(w, len(v)):.4f} | "
                  f"bootstrap P(mean<=0) {bootstrap_p_le0(v.values):.3f}")

# ------------------------------------------------------------ 2. era split
print("\n" + "=" * 96)
print("2. ERA + concentration of the live cell (near-high x A x B)")
print("=" * 96)
live = IDX[(NEAR_HI & GATE_A & GATE_B).values]
epi_live = declusters(live, 10, IDX)
print(f"live-cell days N={len(live)}, episodes N={len(epi_live)}")
print("  episode dates:", ", ".join(str(d.date()) for d in epi_live))
print("  years:", sorted({d.year for d in live}))
for leg in ("SPY", "TLT"):
    f = fwd_lag(px[leg], 10, 1)
    v = f.reindex(epi_live).dropna()
    if len(v) >= 2:
        show(era_split(v.index, v.values), f"{leg} h=10 live cell era split")
        print("  ", cluster_note(v.index, v.values))

# --------------------------------------- 3. is this just C2 in other clothes?
print("\n" + "=" * 96)
print("3. OVERLAP with C2 (the late-August equity short from a high)")
print("=" * 96)
aug = pd.Series((IDX.month == 8) & (IDX.day >= 15) & (IDX.day <= 19), index=IDX)
print(f"  live-cell days that are also Aug 15-19: {int((aug & NEAR_HI & GATE_A & GATE_B).sum())}"
      f" of {len(live)}")
print(f"  near-high days that are Aug 15-19: "
      f"{int((aug & NEAR_HI).sum())} of {int(NEAR_HI.sum())}")

# ----------------------------------- 4. threshold neighbours (definition test)
print("\n" + "=" * 96)
print("4. DEFINITION NEIGHBOURS on the rates gate (forward SPY h=10)")
print("=" * 96)
f = fwd_lag(spy, 10, 1)
par = f.reindex(declusters(IDX[(NEAR_HI & f.notna()).values], 10, IDX)).dropna()
rows = [{"gate": "none (parent)", "n_epi": len(par),
         "mean_pct": round(100 * par.mean(), 3),
         "hit": round(100 * (par > 0).mean(), 1)}]
for thr in (50, 60, 70, 80, 90):
    m = NEAR_HI & (r21 >= thr)
    v = f.reindex(declusters(IDX[(m & f.notna()).values], 10, IDX)).dropna()
    rows.append({"gate": f"TNX r21>={thr}", "n_epi": len(v),
                 "mean_pct": round(100 * v.mean(), 3) if len(v) else np.nan,
                 "hit": round(100 * (v > 0).mean(), 1) if len(v) else np.nan})
for thr in (0.5, 1.0, 2.0, 5.0):
    m = NEAR_HI & (off_lo <= thr)
    v = f.reindex(declusters(IDX[(m & f.notna()).values], 10, IDX)).dropna()
    rows.append({"gate": f"TLT<={thr}% of low", "n_epi": len(v),
                 "mean_pct": round(100 * v.mean(), 3) if len(v) else np.nan,
                 "hit": round(100 * (v > 0).mean(), 1) if len(v) else np.nan})
for thr in (-0.25, -0.5, -1.0, -2.0):
    m = (off_hi >= thr) & GATE_A
    v = f.reindex(declusters(IDX[(m & f.notna()).values], 10, IDX)).dropna()
    rows.append({"gate": f"SPY>={thr}% of high x A", "n_epi": len(v),
                 "mean_pct": round(100 * v.mean(), 3) if len(v) else np.nan,
                 "hit": round(100 * (v > 0).mean(), 1) if len(v) else np.nan})
show(rows, "definition ladder, forward SPY h=10")
