"""C4 round 2 - the loose end left by a2's leg attribution.

a2 showed the beta-neutral GDX/GLD spread's excess is carried 232-495% by the
LONG GDX leg, i.e. the short-GLD hedge is a pure destroyer.  So the only thing
with content in C4 is OUTRIGHT LONG GDX on the parabolic break - which is
exactly the repetition-blocked fingerprint 749b2073856902b3 pitched 2026-08-27
("Buy the first flush inside a parabolic gold-miner run").

This script decides whether that underlying object is distinguishable at all,
so the composer knows whether a `changed_since` route even exists:

  W1. OUTRIGHT long-miner reference class - the identical rule on 12 miner
      names plus the two metals, permutation max-of-K.  (The registry already
      closed GDX thrust cells this way at P(max-of-15)=0.582 on 2026-08-17.)
  W2. Lag profile of the outright long - is the effect tradeable-lag shaped?
  W3. Concentration and era on the outright.
  W4. Fragility dial: where does today's 87.5 sit among trigger episodes?
  W5. Is today's state inside the historical support (GDX r5, drawdown)?
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
BASE = ["GLD", "SLV"] + MINERS

px = close_panel(BASE)
px = px[px.index <= BAR].dropna(subset=["GLD", "SLV", "GDX"])
D = px.index
r1 = {t: px[t] / px[t].shift(1) - 1.0 for t in px.columns}
gdx_r21 = pct_rank(px["GDX"], 21).reindex(D)
gdx_r5 = pct_rank(px["GDX"], 5).reindex(D)
PARA = ((r1["SLV"] < -0.02) & (r1["GLD"] < -0.015) & (r1["GDX"] < -0.02)
        & (gdx_r21 >= 90))
print(f"panel {D[0].date()} .. {D[-1].date()}  parabolic days "
      f"{int(PARA.sum())}  LIVE {bool(PARA.iloc[-1])}")


def epi(mask, r, gap=GAP):
    m = mask.reindex(D, fill_value=False).fillna(False)
    return declusters(D[m.values & r.notna().values], gap, D)


# ------------------------------------------------------ W1. reference class
print("\n" + "=" * 100)
print("W1. OUTRIGHT LONG-MINER REFERENCE CLASS (excess over each name's own")
print("    drift; the trade C4's leg attribution says is the only content)")
print("=" * 100)
for H in (1, 2, 3, 5, 10):
    res = []
    for t in MINERS + ["GLD", "SLV"]:
        s = fwd_lag(px[t], H, 1)
        e = epi(PARA, s)
        if len(e) < 5:
            continue
        v = s.loc[e].values
        res.append({"name": t, "n": len(e),
                    "mean_pct": round(100 * v.mean(), 3),
                    "ctrl_pct": round(100 * s.dropna().mean(), 3),
                    "excess_pp": round(100 * (v.mean() - s.dropna().mean()), 3),
                    "se_pp": round(100 * v.std(ddof=1) / np.sqrt(len(v)), 3),
                    "rec": f"{int((v>0).sum())}-{len(v)-int((v>0).sum())}"})
    df = pd.DataFrame(res).sort_values("excess_pp", ascending=False)
    obs = float(df[df["name"] == "GDX"]["excess_pp"].iloc[0])
    rng = np.random.default_rng(42)
    nmax = rng.normal(0.0, df["se_pp"].values[None, :],
                      size=(20000, len(df))).max(axis=1)
    # second null: impose the class's own common mean (the 2026-08-28 form)
    mu = float(df["excess_pp"].mean())
    nmax2 = rng.normal(mu, df["se_pp"].values[None, :],
                       size=(20000, len(df))).max(axis=1)
    print(f"\n  h={H}:")
    print(df.to_string(index=False))
    print(f"    GDX excess {obs:+.3f}pp ranks "
          f"{int((df['excess_pp']>obs).sum())+1} of {len(df)};  class mean "
          f"{mu:+.3f}pp, sd {df['excess_pp'].std(ddof=1):.3f}pp vs mean SE "
          f"{df['se_pp'].mean():.3f}pp (dispersion ratio "
          f"{df['excess_pp'].std(ddof=1)/df['se_pp'].mean():.2f})")
    print(f"    P(max-of-{len(df)} >= GDX) zero-null {float((nmax>=obs).mean()):.4f}"
          f"   common-mean null {float((nmax2>=obs).mean()):.4f}"
          f"   (null median best, common-mean, {np.median(nmax2):+.3f}pp)")
    pos = int((df["excess_pp"] > 0).sum())
    print(f"    {pos} of {len(df)} names positive, median excess "
          f"{df['excess_pp'].median():+.3f}pp -> "
          f"{'CLASS-WIDE effect' if pos >= 0.7*len(df) else 'name-specific'}")

# ---------------------------------------------------------- W2. lag profile
print("\n" + "=" * 100)
print("W2. LAG PROFILE of the OUTRIGHT long GDX")
print("=" * 100)
for h in (1, 2, 3, 5):
    line = []
    for lag in (0, 1, 2, 3):
        s = fwd_lag(px["GDX"], h, lag)
        e = epi(PARA, s)
        v = s.loc[e].values
        line.append(f"lag{lag} {100*v.mean():+6.2f}% ({int((v>0).sum())}-"
                    f"{len(v)-int((v>0).sum())})")
    print(f"  h={h}:  " + "   ".join(line))

# ----------------------------------------------------- W3. concentration/era
print("\n" + "=" * 100)
print("W3. CONCENTRATION AND ERA, outright long GDX")
print("=" * 100)
for H in (3, 5):
    s = fwd_lag(px["GDX"], H, 1)
    e = epi(PARA, s)
    v = s.loc[e].values
    print(f"\n  h={H}: N={len(e)} mean {100*v.mean():+.3f}% median "
          f"{100*np.median(v):+.3f}% bootstrap P(mean<=0) {bootstrap_p_le0(v):.3f}")
    for d, x in zip(e, v):
        print(f"     {d.date()}  {100*x:+7.2f}%")
    order = np.argsort(-v)
    print(f"    drop-best-2 {100*np.delete(v, order[:2]).mean():+.3f}%   "
          + cluster_note(e, v))
    show(era_split(e, v), f"era split h={H}")
    yr = pd.DatetimeIndex(e).year
    print(f"    ex-2026 {100*v[yr!=2026].mean():+.3f}% on N={int((yr!=2026).sum())}"
          f"  |  ex-2020+2026 {100*v[(yr!=2026)&(yr!=2020)].mean():+.3f}% on "
          f"N={int(((yr!=2026)&(yr!=2020)).sum())}")

# ------------------------------------------------------------- W4. dial / W5
print("\n" + "=" * 100)
print("W4/W5. FRAGILITY DIAL and IN-SAMPLE SUPPORT")
print("=" * 100)
fr = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
fr.index = pd.to_datetime(fr.index)
ma10 = fr["63d"].rolling(10).mean()
s = fwd_lag(px["GDX"], 5, 1)
e = epi(PARA, s)
print(f"  today's ma10-63d dial {ma10.iloc[-1]:.1f}")
for d in e:
    x = ma10.reindex([d]).iloc[0]
    print(f"     {d.date()}  dial {x:.1f}" if np.isfinite(x)
          else f"     {d.date()}  dial  (pre-2016, no reading)")
hi52 = rolling_on_valid(px["GDX"], lambda x: x.rolling(252).max())
dd = px["GDX"] / hi52 - 1.0
sma200 = rolling_on_valid(px["GDX"], lambda x: x.rolling(200).mean())
st = pd.DataFrame({"gdx_r5": gdx_r5, "gdx_r21": gdx_r21,
                   "dd_52wh_pct": 100 * dd,
                   "vs200d_pct": 100 * (px["GDX"] / sma200 - 1)}).loc[e]
print("\n  episode state distribution:")
print(st.describe().round(2).to_string())
print("  TODAY:", {k: round(float(v), 2) for k, v in
                   pd.DataFrame({"gdx_r5": gdx_r5, "gdx_r21": gdx_r21,
                                 "dd_52wh_pct": 100 * dd,
                                 "vs200d_pct": 100 * (px["GDX"] / sma200 - 1)}
                                ).iloc[-1].items()})
for c in st.columns:
    live = float(pd.DataFrame({"gdx_r5": gdx_r5, "gdx_r21": gdx_r21,
                               "dd_52wh_pct": 100 * dd,
                               "vs200d_pct": 100 * (px["GDX"] / sma200 - 1)}
                              )[c].iloc[-1])
    print(f"   today's {c:12s} {live:+7.2f} sits at the "
          f"{100.0*(st[c] <= live).mean():5.1f}th pctile of the {len(st)} "
          f"episodes (min {st[c].min():+.2f}, max {st[c].max():+.2f})")
