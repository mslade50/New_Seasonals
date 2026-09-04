"""C19 round 3 / development - SHORT SLV after a whole-metals-complex break.

Round 2 (b4c_c19_slv_short_teardown.py) left one live question: the cell is
strong unconditionally at h=1 (67-52, sign p 0.019 against SLV's own 46.3%
down-rate, +0.583pp over all days, welch t +2.21 against the local control)
and DEGRADES on every conditioner that matches today's tape, going wrong-
signed when all of them are imposed at once.  This script decides it, and if
it lives, develops it.

  D0. Is the state-matched cell DISTINGUISHABLE from the complement, or is
      the split just noise?  Welch t on the difference, per conditioner.
  D1. MULTIPLICITY: the conditioner x horizon grid searched in round 2, with
      a permutation max-of-N so the best cell is charged for the search.
  D2. horizon_scan h=1..10 - the horizon comes from the table.
  D3. ENTRY FORM: MOC vs a close-anchored LIMIT at k ATR, compared as WHOLE
      variants with fill rates (never a marginal-fill decomposition).
  D4. EXITS: target / stop sensitivity at the chosen horizon.
  D5. episode_paths on the LOSING episodes, so what_kills_it quotes a number.
  D6. SHRUNK magnitude from the 6-family reference class.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
BAR = pd.Timestamp("2026-08-28")
GAP = 5
BASE = ["GLD", "SLV", "GDX"]

raw = load_prices(["SLV"])["SLV"]
px = close_panel(BASE).dropna().loc[:BAR]
slv = px["SLV"]
r1 = {t: px[t] / px[t].shift(1) - 1.0 for t in BASE}
trig = (r1["GLD"] <= -0.02) & (r1["SLV"] <= -0.02) & (r1["GDX"] <= -0.02)

r63 = slv.pct_change(63)
sma200 = slv.rolling(200).mean()
hi52 = slv.rolling(252).max()

COND = {
    "unconditional": pd.Series(True, index=px.index),
    "r63<=-10% (today -12.2)": r63 <= -0.10,
    "below 200d by >5% (today -7.7)": (slv / sma200 - 1.0) <= -0.05,
    ">30% below 52wh (today -43.2)": (slv / hi52 - 1.0) <= -0.30,
    ">40% below 52wh (today -43.2)": (slv / hi52 - 1.0) <= -0.40,
    "SLV break <=-4% (today -4.38)": r1["SLV"] <= -0.04,
    "ALL FOUR of today's states": ((r63 <= -0.10)
                                   & ((slv / sma200 - 1.0) <= -0.05)
                                   & ((slv / hi52 - 1.0) <= -0.30)),
}


def ep(mask, h, gap=GAP):
    r = vehicle_ret(px, [("SLV", -1.0)], h, 1)
    v = r.notna()
    days = px.index[mask.reindex(px.index, fill_value=False).values
                    & trig.values & v.values]
    if len(days) == 0:
        return pd.DatetimeIndex([]), np.array([]), r[v]
    e = declusters(days, gap, px.index)
    return e, r.loc[e].values, r[v]


# ------------------------------------------------------ D0. is the split real?
print("=" * 100)
print("D0. STATE-MATCHED vs COMPLEMENT - is the difference distinguishable?")
print("=" * 100)
for h in (1, 3, 5):
    print(f"\n  h={h}")
    for lbl, m in COND.items():
        if lbl == "unconditional":
            continue
        _, a, base = ep(m, h)
        _, b, _ = ep(~m, h)
        if len(a) < 3 or len(b) < 3:
            continue
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        wa = int((a > 0).sum())
        wb = int((b > 0).sum())
        print(f"    {lbl:34s} ON {100*a.mean():+7.3f}% ({wa}-{len(a)-wa})  "
              f"OFF {100*b.mean():+7.3f}% ({wb}-{len(b)-wb})  "
              f"diff {100*(a.mean()-b.mean()):+7.3f}pp  welch t "
              f"{(a.mean()-b.mean())/se:+5.2f}")

# ----------------------------------------------------------- D1. multiplicity
print("\n" + "=" * 100)
print("D1. MULTIPLICITY over the conditioner x horizon grid searched in round 2")
print("=" * 100)
cells = []
for lbl, m in COND.items():
    for h in (1, 2, 3, 4, 5, 6, 8, 10):
        e, v, base = ep(m, h)
        if len(v) < 8:
            continue
        se = 100 * v.std(ddof=1) / np.sqrt(len(v))
        cells.append({"cell": f"{lbl} h={h}", "n": len(v),
                      "mean_pct": round(100 * v.mean(), 3),
                      "ctrl_pct": round(100 * base.mean(), 3),
                      "excess_pct": round(100 * (v.mean() - base.mean()), 3),
                      "se_pct": round(se, 3)})
df = pd.DataFrame(cells)
print(f"  grid size {len(df)} cells")
print(df.sort_values("excess_pct", ascending=False).head(10).to_string(index=False))
rng = np.random.default_rng(42)
nulls = rng.normal(0.0, df["se_pct"].values[None, :], size=(20000, len(df)))
nmax = nulls.max(axis=1)
obs = df["excess_pct"].max()
print(f"\n  permutation max-of-{len(df)} p = {float((nmax >= obs).mean()):.4f}"
      f"   (observed max {obs:+.3f}%, null max median {np.median(nmax):+.3f}%, "
      f"95th {np.percentile(nmax, 95):+.3f}%)")
uncond = df[df["cell"] == "unconditional h=1"].iloc[0]
print(f"  NOTE the pitched cell is the UNCONDITIONAL one: "
      f"{uncond['cell']} excess {uncond['excess_pct']:+.3f}% "
      f"-> it is not a max-of-grid draw; the grid charge applies to any "
      f"state-matched REFINEMENT.")
rk = int((df["excess_pct"] > uncond["excess_pct"]).sum()) + 1
print(f"  unconditional h=1 ranks {rk} of {len(df)} in the grid.")

# ------------------------------------------------------------- D2. horizon scan
print("\n" + "=" * 100)
print("D2. HORIZON SCAN (pitch_lab.horizon_scan, episodes gap 5)")
print("=" * 100)
sig = px.index[trig.values]
show(horizon_scan(px, sig, [("SLV", -1.0)], hs=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
                  min_gap=GAP), "SHORT SLV")
for h in (1, 2, 3):
    e, v, base = ep(COND["unconditional"], h)
    order = np.argsort(-v)
    w = int((v > 0).sum())
    print(f"  h={h}: {100*v.mean():+.3f}% record {w}-{len(v)-w}, "
          f"sign p vs down-rate {100*float((base>0).mean()):.1f}% = "
          f"{sign_test(w, len(v), float((base>0).mean())):.4f}, "
          f"drop-best-3 {100*np.delete(v, order[:3]).mean():+.3f}%, "
          f"median {100*np.median(v):+.3f}%")

# --------------------------------------------------------------- D3. entry form
print("\n" + "=" * 100)
print("D3. ENTRY FORM - MOC vs close-anchored LIMIT, WHOLE variants + fill rate")
print("=" * 100)
o = raw["Open"].reindex(px.index)
hi = raw["High"].reindex(px.index)
lo = raw["Low"].reindex(px.index)
cl = raw["Close"].reindex(px.index)
atr = pd.Series(wilder_atr(hi.values, lo.values, cl.values), index=px.index)
pos = pd.Series(range(len(px.index)), index=px.index)
print(f"  today ATR14 {atr.iloc[-1]:.4f} on SLV close {cl.iloc[-1]:.2f} "
      f"= {100*atr.iloc[-1]/cl.iloc[-1]:.2f}% of price")

H_DEV = 1
for k in (0.0, 0.15, 0.25, 0.40):
    fills, rets, nsig = [], [], 0
    e, _, _ = ep(COND["unconditional"], H_DEV)
    for d in e:
        p = pos[d]
        # entry day = p+1 (lag 1).  SHORT: a limit ABOVE the entry-day close
        if p + 1 + H_DEV >= len(px.index):
            continue
        nsig += 1
        ref = cl.iloc[p]                      # anchor = the SIGNAL close
        a = atr.iloc[p]
        if not np.isfinite(a):
            continue
        if k == 0.0:                          # MOC on the entry day
            entry = cl.iloc[p + 1]
            filled = True
        else:
            lim = ref + k * a                 # sell into strength
            filled = bool(hi.iloc[p + 1] >= lim)
            entry = lim if filled else np.nan
        if filled:
            exit_px = cl.iloc[p + 1 + H_DEV]
            rets.append(entry / exit_px - 1.0)   # short return
            fills.append(1)
        else:
            fills.append(0)
    fr = 100 * np.mean(fills) if fills else 0.0
    r = np.asarray(rets)
    w = int((r > 0).sum())
    print(f"  k={k:.2f} ATR {'(MOC)' if k==0 else '(LIMIT above sig close)'}: "
          f"fill {fr:5.1f}% ({len(r)}/{nsig})  mean {100*r.mean():+.3f}%  "
          f"median {100*np.median(r):+.3f}%  record {w}-{len(r)-w}  "
          f"TOTAL over all signals {100*r.sum()/max(nsig,1):+.3f}%/signal")

# ------------------------------------------------------------------- D4. exits
print("\n" + "=" * 100)
print("D4. EXITS - target / stop sensitivity at h=1 and h=3")
print("=" * 100)
for H in (1, 3):
    e, _, _ = ep(COND["unconditional"], H)
    for tgt, stp in [(None, None), (1.0, None), (None, 1.0), (1.0, 1.5),
                     (0.75, 1.25), (1.5, 2.0)]:
        rets = []
        for d in e:
            p = pos[d]
            if p + 1 + H >= len(px.index):
                continue
            a = atr.iloc[p]
            entry = cl.iloc[p + 1]
            if not np.isfinite(a):
                continue
            t_px = entry - tgt * a if tgt else None
            s_px = entry + stp * a if stp else None
            out = None
            for j in range(p + 2, p + 2 + H):
                if s_px is not None and hi.iloc[j] >= s_px:
                    out = entry / s_px - 1.0
                    break
                if t_px is not None and lo.iloc[j] <= t_px:
                    out = entry / t_px - 1.0
                    break
            if out is None:
                out = entry / cl.iloc[p + 1 + H] - 1.0
            rets.append(out)
        r = np.asarray(rets)
        w = int((r > 0).sum())
        print(f"  h={H} tgt={tgt} stop={stp}: mean {100*r.mean():+.3f}%  "
              f"median {100*np.median(r):+.3f}%  record {w}-{len(r)-w}  "
              f"worst {100*r.min():+.2f}%  best {100*r.max():+.2f}%")

# ----------------------------------------------------------- D5. loser anatomy
print("\n" + "=" * 100)
print("D5. LOSING EPISODES - what kills it, with a number")
print("=" * 100)
for H in (1, 3):
    e, v, _ = ep(COND["unconditional"], H)
    paths = episode_paths(px, e, [("SLV", -1.0)], H, lag=1)
    losers = paths[paths[H] < 0]
    winners = paths[paths[H] >= 0]
    print(f"\n  h={H}: {len(losers)} losers / {len(paths)} episodes")
    print(f"    losers  by day: "
          + "  ".join(f"d{c} {100*losers[c].mean():+.2f}%" for c in paths.columns))
    print(f"    winners by day: "
          + "  ".join(f"d{c} {100*winners[c].mean():+.2f}%" for c in paths.columns))
    print(f"    worst 5 losers: "
          + ", ".join(f"{str(pd.Timestamp(i).date())} {100*losers.loc[i, H]:+.2f}%"
                      for i in losers[H].nsmallest(5).index))
    yr = pd.DatetimeIndex(losers.index).year
    print(f"    loser years: {dict(pd.Series(yr).value_counts().head(6))}")

# ------------------------------------------------------------ D6. shrinkage
print("\n" + "=" * 100)
print("D6. SHRUNK MAGNITUDE from the 6-family reference class (b2_c3b S4)")
print("=" * 100)
print("  h=1 family table (from _out_c3b.txt): metals/SLV excess +0.566%, "
      "energy/USO +0.374,\n  semis/NVDA -0.153, banks/BAC +0.358, hombld/DHI "
      "-0.212, materls/FCX +0.171.")
print("  fixed-effect common excess +0.132% (se 0.106, t +1.25), I^2 37.1%.")
fam = np.array([0.566, 0.374, -0.153, 0.358, -0.212, 0.171])
se = np.array([0.271, 0.237, 0.251, 0.383, 0.210, 0.278])
tau2 = max(0.0, fam.var(ddof=1) - (se ** 2).mean())
shr = tau2 / (tau2 + se[0] ** 2)
common = float((fam / se ** 2).sum() / (1 / se ** 2).sum())
print(f"  method-of-moments tau^2 = {tau2:.4f}, shrinkage weight on the "
      f"metals estimate = {shr:.2f}")
print(f"  empirical-Bayes shrunk metals excess = "
      f"{shr*fam[0] + (1-shr)*common:+.3f}%  (raw {fam[0]:+.3f}%, family "
      f"{common:+.3f}%)")
print(f"  at 6 bp round trip that is "
      f"{100*(shr*fam[0] + (1-shr)*common)/6.0:.1f}x cost "
      f"(raw would be {100*fam[0]/6.0:.1f}x)")
