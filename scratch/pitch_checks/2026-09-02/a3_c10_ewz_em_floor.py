"""C10 round 1 - LONG EWZ at the tape's z10 extreme (+2.03) inside an EM
complex at a 63-day rank floor (EEM r63 0.4).  Signal 2026-09-01.

STANDING REGISTRY BAR, quoted: the country family is CLOSED across five
members (EWZ twice, FXI, SMH/QQQ, EFA, KWEB, EWJ) and the entry that closed it
says "a new country instance needs P(max-of-K) below 0.05 on the residual
before it is worth a check."  This is the mirror direction (EWZ thrusting
while the complex is at a floor rather than EWZ breaking while the complex is
firm), so it is a NEW object by trigger, and it inherits that bar by family.

FIRST FINDING, before anything else: THIS REPO HOLDS TWO z10 DEFINITIONS and
they disagree about the live reading by 0.57 z-units.
  - build_pitch_state._metrics_for: ret10 / (vol21 * sqrt(10))   -> the +2.03
    the tape prints and the number that put this candidate on the map
  - pitch_lab.zscore: the 10d return z-scored against its own trailing-252
    mean and sd                                                  -> +1.46
CLAUDE.md already records that these differ ("pitch_lab.zscore, whose
docstring claims the same definition but computes something else").  The
STATE definition is primary here because it is what selection used; the lab
definition is carried as the definition neighbour.

  U0. Count-first + live verify under BOTH definitions.
  U1. Outright long EWZ, the pair (equal-dollar AND beta-neutral), inversion.
  U2. LEG ATTRIBUTION on the pair.
  U3. Gate attribution: does the EEM-floor leg add over the bare thrust?
  U4. REFERENCE CLASS - identical rule across the international complex,
      Cochran Q / I^2 / permutation max-of-K on the residual.
  U5. Era, concentration, in-sample check, cost.
  U6. Lag profile and horizon scan.
"""
import sys
import warnings
from math import erfc, sqrt
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change  # noqa

import numpy as np
import pandas as pd

pd.set_option("display.width", 260)

BAR = pd.Timestamp("2026-09-01")
GAP = 5
INTL = ["EWZ", "EWW", "EWY", "EWT", "EWJ", "FXI", "EFA", "INDA", "KWEB",
        "VGK", "RSX"]
BASE = ["EEM", "SPY"] + INTL

px = close_panel(BASE)
px = px[px.index <= BAR].dropna(subset=["EWZ", "EEM"])
D = px.index
print(f"panel {D[0].date()} .. {D[-1].date()}  n={len(D)}")


def z10_state(s: pd.Series) -> pd.Series:
    """build_pitch_state._metrics_for: 10d return / (21d sd * sqrt(10))."""
    v = s.dropna()
    r10 = v.pct_change(10)
    vol21 = v.pct_change().rolling(21).std()
    return (r10 / (vol21 * np.sqrt(10))).reindex(s.index)


zS = {t: z10_state(px[t]) for t in INTL}
zL = {t: zscore(px[t], 10).reindex(D) for t in INTL}
eem_r63 = pct_rank(px["EEM"], 63).reindex(D)
ewz_r5 = pct_rank(px["EWZ"], 5).reindex(D)

print("\n" + "=" * 100)
print("U0. LIVE VERIFY under BOTH z10 definitions + COUNT FIRST")
print("=" * 100)
print(f"  live {D[-1].date()}: EWZ z10 STATE {zS['EWZ'].iloc[-1]:+.3f}  "
      f"| LAB {zL['EWZ'].iloc[-1]:+.3f}   <- 0.57 z-units apart")
print(f"    EEM r63 {eem_r63.iloc[-1]:.1f}  EWZ r5 {ewz_r5.iloc[-1]:.1f}  "
      f"EWZ 1d {100*(px['EWZ'].iloc[-1]/px['EWZ'].iloc[-2]-1):+.2f}%  "
      f"SPY 1d {100*(px['SPY'].iloc[-1]/px['SPY'].iloc[-2]-1):+.2f}%")

BARE = zS["EWZ"] >= 2.0
FLOOR = eem_r63 <= 5.0
TRIG = BARE & FLOOR
BARE_L = zL["EWZ"] >= 2.0
for lbl, m in [("STATE z10 >= 2.0 (bare thrust)", BARE),
               ("LAB   z10 >= 2.0 (definition neighbour)", BARE_L),
               ("EEM r63 <= 5 (bare floor)", FLOOR),
               ("BOTH, state def (the pitch)", TRIG),
               ("BOTH, lab def", BARE_L & FLOOR)]:
    mm = m.reindex(D).fillna(False)
    e = declusters(D[mm.values], GAP, D)
    yrs = sorted({d.year for d in e})
    print(f"  {lbl:40s} days {int(mm.sum()):5d}  episodes {len(e):4d}  "
          f"years {len(yrs):2d} {yrs[0] if yrs else '-'}-{yrs[-1] if yrs else '-'}"
          f"  LIVE {bool(mm.iloc[-1])}")
e_trig = declusters(D[TRIG.reindex(D).fillna(False).values], GAP, D)
print("  trigger episodes (state def):",
      ", ".join(str(d.date()) for d in e_trig))
# a floor ladder so the count-first stage is honest about what widening buys
for cut in (5, 10, 20, 30, 50):
    m = (BARE & (eem_r63 <= cut)).reindex(D).fillna(False)
    print(f"    EEM r63 <= {cut:2d}: days {int(m.sum()):4d}  episodes "
          f"{len(declusters(D[m.values], GAP, D)):3d}  LIVE {bool(m.iloc[-1])}")


def epi(mask, r, gap=GAP):
    m = mask.reindex(D, fill_value=False).fillna(False)
    return declusters(D[m.values & r.notna().values], gap, D)


def cellrow(mask, s, label, gap=GAP):
    e = epi(mask, s, gap)
    if len(e) == 0:
        return {"label": label, "n": 0}
    v = s.loc[e].values
    r = summarize(v, label)
    r["ctrl_pct"] = round(100 * s.dropna().mean(), 3)
    r["edge_pp"] = round(r["mean_pct"] - 100 * s.dropna().mean(), 3)
    w = int((v > 0).sum())
    r["rec"] = f"{w}-{len(v)-w}"
    r["p_vs_base"] = round(sign_test(w, len(v), float((s.dropna() > 0).mean())), 4)
    return r


de = px["EEM"].pct_change()
beta = ((px["EWZ"].pct_change().rolling(252).cov(de)
         / de.rolling(252).var()).shift(1))
print(f"\n  beta(EWZ~EEM) mean {beta.dropna().mean():.2f}  today "
      f"{beta.iloc[-1]:.2f}  -> an EQUAL-DOLLAR pair is a "
      f"{beta.iloc[-1]-1:+.2f}-unit EM bet today")

# ------------------------------------------------------------- U1 / U2 forms
print("\n" + "=" * 100)
print("U1/U2. FORMS and LEG ATTRIBUTION (widest countable cell used where the")
print("       pitched cell is uncountable - stated, not hidden)")
print("=" * 100)
CELLS = {"the pitch (EEM r63<=5)": TRIG,
         "loosened (EEM r63<=20)": BARE & (eem_r63 <= 20),
         "bare thrust (no EEM gate)": BARE}
for name, mask in CELLS.items():
    print(f"\n  ---- {name} ----")
    for h in (1, 3, 5, 10):
        fe, fw = fwd_lag(px["EEM"], h, 1), fwd_lag(px["EWZ"], h, 1)
        forms = {
            "OUTRIGHT long EWZ": fw,
            "INVERSION short EWZ": -fw,
            "EQUAL-DOLLAR long EWZ / short EEM": fw - fe,
            "BETA-NEUTRAL long EWZ / short beta*EEM": fw - beta * fe,
            "OUTRIGHT long EEM alone": fe,
        }
        show([cellrow(mask, s, lbl) for lbl, s in forms.items()], f"h={h}")
        s = fw - beta * fe
        e = epi(mask, s)
        if len(e) < 2:
            continue
        ewz_ex = 100 * (fw.loc[e].mean() - fw.dropna().mean())
        b = float(beta.loc[e].mean())
        eem_ex = -b * 100 * (fe.loc[e].mean() - fe.dropna().mean())
        tot = 100 * (s.loc[e].mean() - s.dropna().mean())
        print(f"    h={h} LEG ATTRIBUTION (N={len(e)}, beta {b:.2f}): spread "
              f"excess {tot:+.3f}pp | LONG EWZ leg {ewz_ex:+.3f}pp | "
              f"SHORT EEM leg {eem_ex:+.3f}pp -> EWZ leg carries "
              f"{100*ewz_ex/tot if tot else float('nan'):.0f}%")

# -------------------------------------------------------- U3. gate attribution
print("\n" + "=" * 100)
print("U3. GATE ATTRIBUTION - does the EEM-floor leg add anything?")
print("=" * 100)
for h in (1, 3, 5, 10):
    fe, fw = fwd_lag(px["EEM"], h, 1), fwd_lag(px["EWZ"], h, 1)
    s = fw - beta * fe
    lad = []
    for cut in (5, 10, 20, 30, 50, 100):
        lad.append(cellrow(BARE & (eem_r63 <= cut), s,
                           f"EEM r63<={cut} (today 0.4)"))
    lad.append(cellrow(BARE & (eem_r63 > 20), s, "EEM r63>20 (DISCARDS)"))
    show(lad, f"EEM-floor ladder, beta-neutral residual, h={h}")
    lad2 = []
    for cut in (5, 10, 20, 30, 50, 100):
        lad2.append(cellrow(BARE & (eem_r63 <= cut), fw,
                            f"EEM r63<={cut} OUTRIGHT"))
    show(lad2, f"EEM-floor ladder, OUTRIGHT long EWZ, h={h}")

# -------------------------------------------------------- U4. reference class
print("\n" + "=" * 100)
print("U4. REFERENCE CLASS - identical rule across the international complex")
print("    (run on the LOOSENED cell, r63<=20, since the pitched cell has N=1)")
print("=" * 100)
for H in (3, 5, 10):
    res = []
    for t in INTL:
        col = px[t]
        m = (zS[t] >= 2.0) & (eem_r63 <= 20)
        dt = col.pct_change()
        b = (dt.rolling(252).cov(de) / de.rolling(252).var()).shift(1)
        s = fwd_lag(col, H, 1) - b * fwd_lag(px["EEM"], H, 1)
        e = epi(m, s)
        if len(e) < 5:
            print(f"    {t}: only {len(e)} episodes, excluded")
            continue
        v = s.loc[e].values
        res.append({"name": t, "n": len(e),
                    "mean_pct": round(100 * v.mean(), 3),
                    "excess_pp": round(100 * (v.mean() - s.dropna().mean()), 3),
                    "se_pp": round(100 * v.std(ddof=1) / np.sqrt(len(v)), 3),
                    "rec": f"{int((v>0).sum())}-{len(v)-int((v>0).sum())}"})
    df = pd.DataFrame(res).sort_values("excess_pp", ascending=False)
    print(f"\n  h={H}: beta-neutral residual vs EEM, same rule on {len(df)} names")
    print(df.to_string(index=False))
    if "EWZ" not in set(df["name"]):
        continue
    obs = float(df[df["name"] == "EWZ"]["excess_pp"].iloc[0])
    w = 1.0 / df["se_pp"].values ** 2
    mu = float((df["excess_pp"].values * w).sum() / w.sum())
    Q = float((w * (df["excess_pp"].values - mu) ** 2).sum())
    dfree = len(df) - 1
    I2 = max(0.0, 100 * (Q - dfree) / Q) if Q > 0 else 0.0
    z_wh = ((Q / dfree) ** (1 / 3) - (1 - 2 / (9 * dfree))) / sqrt(2 / (9 * dfree))
    pQ = 0.5 * erfc(z_wh / sqrt(2))
    rng = np.random.default_rng(42)
    nmax = rng.normal(mu, df["se_pp"].values[None, :],
                      size=(20000, len(df))).max(axis=1)
    print(f"    common (inverse-variance) excess {mu:+.3f}pp   Cochran Q "
          f"{Q:.2f} on {dfree} df, p {pQ:.4f}, I^2 {I2:.1f}%")
    print(f"    EWZ {obs:+.3f}pp ranks {int((df['excess_pp']>obs).sum())+1} of "
          f"{len(df)};  permutation P(max-of-{len(df)} >= EWZ) = "
          f"{float((nmax >= obs).mean()):.4f}   null median best "
          f"{np.median(nmax):+.3f}pp")

# ------------------------------------------------- U5. era / concentration / cost
print("\n" + "=" * 100)
print("U5. ERA, CONCENTRATION, IN-SAMPLE CHECK, COST (loosened cell r63<=20)")
print("=" * 100)
MASK = BARE & (eem_r63 <= 20)
for H in (3, 5):
    fe, fw = fwd_lag(px["EEM"], H, 1), fwd_lag(px["EWZ"], H, 1)
    s = fw - beta * fe
    e = epi(MASK, s)
    v = s.loc[e].values
    print(f"\n  h={H}: N={len(e)} residual mean {100*v.mean():+.3f}%  median "
          f"{100*np.median(v):+.3f}%  bootstrap P(mean<=0) "
          f"{bootstrap_p_le0(v):.3f}  outright {100*fw.loc[e].mean():+.3f}%")
    show(era_split(e, v), f"era split (residual h={H})")
    print("  " + cluster_note(e, v))
    print("  episodes:", ", ".join(str(d.date()) for d in e))
print(f"\n  IN-SAMPLE CHECK: today's EWZ STATE z10 {zS['EWZ'].iloc[-1]:+.2f}; "
      f"episodes' z10 range [{zS['EWZ'].loc[e].min():+.2f}, "
      f"{zS['EWZ'].loc[e].max():+.2f}].  Today's EEM r63 {eem_r63.iloc[-1]:.1f}; "
      f"episode range [{eem_r63.loc[e].min():.1f}, {eem_r63.loc[e].max():.1f}]")
print(f"  COST: EWZ ~8 bp round trip (single-country ETF + FX); a pair adds "
      f"EEM ~6 bp = ~14 bp two-leg.")

# ------------------------------------------------------ U6. lag + horizon scan
print("\n" + "=" * 100)
print("U6. LAG PROFILE and HORIZON SCAN (loosened cell)")
print("=" * 100)
for h in (1, 3, 5):
    for lag in (0, 1, 2):
        ss = fwd_lag(px["EWZ"], h, lag) - beta * fwd_lag(px["EEM"], h, lag)
        ee = epi(MASK, ss)
        vv = ss.loc[ee].values
        oo = fwd_lag(px["EWZ"], h, lag)
        print(f"  h={h} lag={lag}: residual {100*vv.mean():+7.3f}% "
              f"({int((vv>0).sum())}-{len(vv)-int((vv>0).sum())})  "
              f"outright {100*oo.loc[ee].mean():+7.3f}%")
ee = epi(MASK, fwd_lag(px["EWZ"], 5, 1))
show(horizon_scan(px, ee, [("EWZ", 1.0)], hs=(1, 2, 3, 5, 7, 10), min_gap=GAP),
     "OUTRIGHT long EWZ, horizon scan (loosened cell)")
