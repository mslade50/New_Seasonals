"""C4 round 1 - LONG GDX against SHORT GLD on the three-name metals break out
of a parabolic miner run (the miners overshoot the metal).  Signal 2026-09-01.

MANDATORY charges from the registry, all answered here:
  R1. "Beta-neutralize a pair before crediting the spread" - equal-dollar
      GDX-GLD is a levered gold bet, not a spread (2026-08-10 c6).
  R2. Registry closes "Miner-versus-metal ratio reversion after a maximal
      thrust" (short GDX / long GLD, wrong-signed at ALL TEN horizons,
      -0.576% at h=5 over 51 episodes).  Today's is the SAME RATIO OBJECT in
      the mirror direction.  Measured here on the identical machinery.
  R3. "Adding a second metals leg beside a live one" - SLV correlates +0.708
      with a live GDX leg; correlation of the pair with the C3 short.
  R4. GDX thrust cells are closed by the reference class at permutation P
      (2026-08-17, P(max-of-15) = 0.582).  Reference class run here on 12
      miner-versus-metal pairs.

  T1. Beta (PIT trailing-252) of GDX on GLD, and the pair in three forms.
  T2. LEG ATTRIBUTION - each leg's excess over its OWN drift, separately.
      If the whole spread is one leg, that is the kill (2026-08-19 shape).
  T3. Lag profile 0/1/2 and horizon scan.
  T4. Concentration / era.
  T5. Reference class: the identical rule on every miner-vs-metal pair.
  T6. Cost - two legs.
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

BAR = pd.Timestamp("2026-09-01")
GAP = 5
MINERS = ["GDX", "NEM", "AEM", "KGC", "AU", "GFI", "RGLD", "WPM", "PAAS",
          "HL", "CDE", "AG"]
BASE = ["GLD", "SLV", "GDX"] + [m for m in MINERS if m != "GDX"]

px = close_panel(BASE)
px = px[px.index <= BAR].dropna(subset=["GLD", "SLV", "GDX"])
D = px.index
print(f"panel {D[0].date()} .. {D[-1].date()}  n={len(D)}")

r1 = {t: px[t] / px[t].shift(1) - 1.0 for t in px.columns}
gdx_r21 = pct_rank(px["GDX"], 21).reindex(D)
PARA = ((r1["SLV"] < -0.02) & (r1["GLD"] < -0.015) & (r1["GDX"] < -0.02)
        & (gdx_r21 >= 90))
BRK = (r1["SLV"] < -0.02) & (r1["GLD"] < -0.015) & (r1["GDX"] < -0.02)
print(f"parabolic days {int(PARA.sum())}  bare-break days {int(BRK.sum())}  "
      f"LIVE {bool(PARA.iloc[-1])}")
print(f"  live day: GDX {100*r1['GDX'].iloc[-1]:+.2f}%  "
      f"GLD {100*r1['GLD'].iloc[-1]:+.2f}%  -> miner overshoot "
      f"{100*(r1['GDX'].iloc[-1]-r1['GLD'].iloc[-1]):+.2f}pp")


def epi(mask, r, gap=GAP):
    m = mask.reindex(D, fill_value=False).fillna(False)
    return declusters(D[m.values & r.notna().values], gap, D)


# --------------------------------------------------------- T1. beta + 3 forms
print("\n" + "=" * 100)
print("T1. BETA of GDX on GLD (PIT trailing 252) and the pair in three forms")
print("=" * 100)
dg, dm = px["GLD"].pct_change(), px["GDX"].pct_change()
beta = (dm.rolling(252).cov(dg) / dg.rolling(252).var()).shift(1)
print(f"  beta(GDX~GLD) mean {beta.dropna().mean():.2f}  today {beta.iloc[-1]:.2f}"
      f"  (an EQUAL-DOLLAR GDX-GLD pair is therefore a "
      f"{beta.iloc[-1]-1:+.2f}-unit LONG gold bet)")

for h in (1, 2, 3, 5):
    fg, fm = fwd_lag(px["GLD"], h, 1), fwd_lag(px["GDX"], h, 1)
    forms = {
        "EQUAL-DOLLAR long GDX / short GLD": fm - fg,
        "BETA-NEUTRAL long GDX / short beta*GLD": fm - beta * fg,
        "OUTRIGHT long GDX": fm,
        "OUTRIGHT short GLD": -fg,
    }
    rows = []
    for lbl, s in forms.items():
        e = epi(PARA, s)
        r = summarize(s.loc[e].values, lbl)
        r["ctrl_pct"] = round(100 * s.dropna().mean(), 3)
        r["edge_pp"] = round(r["mean_pct"] - 100 * s.dropna().mean(), 3)
        w = int((s.loc[e].values > 0).sum())
        r["rec"] = f"{w}-{len(e)-w}"
        r["p_vs_base"] = round(sign_test(w, len(e),
                                         float((s.dropna() > 0).mean())), 4)
        rows.append(r)
    show(rows, f"pair forms, parabolic cell, h={h}")

# ------------------------------------------------------- T2. leg attribution
print("\n" + "=" * 100)
print("T2. LEG ATTRIBUTION - each leg's excess over its OWN drift, separately")
print("=" * 100)
for h in (1, 2, 3, 5):
    fg, fm = fwd_lag(px["GLD"], h, 1), fwd_lag(px["GDX"], h, 1)
    spread = fm - beta * fg
    e = epi(PARA, spread)
    gdx_ex = 100 * (fm.loc[e].mean() - fm.dropna().mean())
    gld_ex = -100 * (fg.loc[e].mean() - fg.dropna().mean())     # SHORT leg
    tot = 100 * (spread.loc[e].mean() - spread.dropna().mean())
    b = float(beta.loc[e].mean())
    print(f"  h={h} (N={len(e)}, mean beta on triggers {b:.2f}): "
          f"beta-neutral spread excess {tot:+.3f}pp"
          f"  |  LONG GDX leg excess {gdx_ex:+.3f}pp"
          f"  |  SHORT GLD leg excess (x beta {b:.2f}) {b*gld_ex:+.3f}pp"
          f"  -> GDX leg carries "
          f"{100*gdx_ex/(gdx_ex + b*gld_ex) if (gdx_ex + b*gld_ex) else float('nan'):.0f}%")

# --------------------------------------------------------- T3. lag + horizon
print("\n" + "=" * 100)
print("T3. LAG PROFILE and HORIZON SCAN (beta-neutral form)")
print("=" * 100)
for h in (1, 3):
    for lag in (0, 1, 2):
        fg, fm = fwd_lag(px["GLD"], h, lag), fwd_lag(px["GDX"], h, lag)
        s = fm - beta * fg
        e = epi(PARA, s)
        v = s.loc[e].values
        w = int((v > 0).sum())
        print(f"  h={h} lag={lag}: {100*v.mean():+7.3f}%  N={len(v)}  "
              f"record {w}-{len(v)-w}  median {100*np.median(v):+.3f}%  "
              f"ctrl {100*s.dropna().mean():+.3f}%")
fg1 = {h: fwd_lag(px["GLD"], h, 1) for h in range(1, 11)}
fm1 = {h: fwd_lag(px["GDX"], h, 1) for h in range(1, 11)}
rows = []
for h in range(1, 11):
    s = fm1[h] - beta * fg1[h]
    e = epi(PARA, s, gap=max(GAP, h))
    r = summarize(s.loc[e].values, f"h={h}")
    r["ctrl_pct"] = round(100 * s.dropna().mean(), 3)
    r["edge_pp"] = round(r["mean_pct"] - 100 * s.dropna().mean(), 3)
    rows.append(r)
show(rows, "beta-neutral spread, horizon scan h=1..10")

# --------------------------------------------------- T4. concentration / era
print("\n" + "=" * 100)
print("T4. CONCENTRATION AND ERA (beta-neutral, h=3)")
print("=" * 100)
H = 3
s = fwd_lag(px["GDX"], H, 1) - beta * fwd_lag(px["GLD"], H, 1)
e = epi(PARA, s)
v = s.loc[e].values
for d, x in zip(e, v):
    print(f"   {d.date()}  {100*x:+7.2f}%")
print(f"  mean {100*v.mean():+.3f}%  median {100*np.median(v):+.3f}%  "
      f"bootstrap P(mean<=0) {bootstrap_p_le0(v):.3f}")
yr = pd.DatetimeIndex(e).year
print(f"  ex-2026: {100*v[yr != 2026].mean():+.3f}% on N={int((yr!=2026).sum())}")
show(era_split(e, v), "era split")
print("  " + cluster_note(e, v))

# ------------------------------------------------------- T5. reference class
print("\n" + "=" * 100)
print("T5. REFERENCE CLASS - the identical rule on 12 miner-vs-metal pairs")
print("=" * 100)
for H in (1, 3, 5):
    fg = fwd_lag(px["GLD"], H, 1)
    res = []
    for m in MINERS:
        if m not in px.columns:
            continue
        dmm = px[m].pct_change()
        b = (dmm.rolling(252).cov(dg) / dg.rolling(252).var()).shift(1)
        s = fwd_lag(px[m], H, 1) - b * fg
        e = epi(PARA, s)
        if len(e) < 5:
            continue
        v = s.loc[e].values
        ex = 100 * (v.mean() - s.dropna().mean())
        res.append({"name": m, "n": len(e), "mean_pct": round(100 * v.mean(), 3),
                    "ctrl_pct": round(100 * s.dropna().mean(), 3),
                    "excess_pp": round(ex, 3),
                    "se_pp": round(100 * v.std(ddof=1) / np.sqrt(len(v)), 3),
                    "rec": f"{int((v>0).sum())}-{len(v)-int((v>0).sum())}"})
    df = pd.DataFrame(res).sort_values("excess_pp", ascending=False)
    print(f"\n  h={H}:")
    print(df.to_string(index=False))
    obs = float(df[df["name"] == "GDX"]["excess_pp"].iloc[0])
    k = len(df)
    rng = np.random.default_rng(42)
    nulls = rng.normal(0.0, df["se_pp"].values[None, :], size=(20000, k))
    nmax = nulls.max(axis=1)
    print(f"    GDX excess {obs:+.3f}pp ranks "
          f"{int((df['excess_pp'] > obs).sum())+1} of {k}; "
          f"cross-name mean {df['excess_pp'].mean():+.3f}pp, "
          f"sd {df['excess_pp'].std(ddof=1):.3f}pp vs mean SE "
          f"{df['se_pp'].mean():.3f}pp (dispersion ratio "
          f"{df['excess_pp'].std(ddof=1)/df['se_pp'].mean():.2f})")
    print(f"    permutation P(max-of-{k} >= GDX) = "
          f"{float((nmax >= obs).mean()):.4f}   null median best "
          f"{np.median(nmax):+.3f}pp")

# ---------------------------------------------------- R3 correlation with C3
print("\n" + "=" * 100)
print("R3. CORRELATION with the C3 short-SLV leg (registry: check before")
print("    pricing a second leg in the same complex)")
print("=" * 100)
for H in (1, 3):
    s = fwd_lag(px["GDX"], H, 1) - beta * fwd_lag(px["GLD"], H, 1)
    c3 = -fwd_lag(px["SLV"], H, 1)
    both = pd.concat([s, c3], axis=1).dropna()
    print(f"  h={H}: corr(pair, short SLV) = {both.corr().iloc[0,1]:+.3f} "
          f"(full history); on trigger episodes only: "
          f"{pd.concat([s.loc[epi(PARA,s)], c3.loc[epi(PARA,s)]], axis=1).corr().iloc[0,1]:+.3f}")

# --------------------------------------------------------------- T6. cost
print("\n" + "=" * 100)
print("T6. COST - two legs")
print("=" * 100)
for H in (1, 3, 5):
    s = fwd_lag(px["GDX"], H, 1) - beta * fwd_lag(px["GLD"], H, 1)
    e = epi(PARA, s)
    edge = 100 * 100 * s.loc[e].mean()
    print(f"  h={H}: spread mean {edge:+7.1f} bp against a 2-leg round trip of "
          f"~12 bp (GDX 6 + GLD 6) -> {edge/12.0:+.2f}x")
