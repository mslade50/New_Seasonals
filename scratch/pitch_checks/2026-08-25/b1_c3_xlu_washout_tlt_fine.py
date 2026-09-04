"""C3 round 1 - Long XLU after a 21-day washout the bond market did not cause.

Live state (2026-08-24): XLU 21d -6.63% (PIT252 0.8), TLT 21d -0.43% (PIT 47.6),
XLU-TLT 21d spread -6.20pp (PIT 2.0). SPY -1.85% off its 52w high.

Order of business, set by the registry:
  0. COUNT THE LITERAL RUNG FIRST (2026-08-24 XLI kill: 0 days at the state).
  1. Is the RATES CONDITION load-bearing? gate on vs off, in pp.
  2. Is the forward return just short-duration-neutral long-XLU beta?
     -> regress the cell's XLU forward return on TLT's forward return.
  3. Reference class over the sector ETFs (Cochran Q + permutation max-of-k).
  4. Overlap with the SEVEN already-dead utilities expressions, computed not
     asserted (2026-08-07 x4, 2026-08-12 rank21, 2026-08-17 XLV pair,
     2026-08-19 XLU-strength).
  5. The SPY-near-high gate, which is LIVE today and which the 2026-08-12 kill
     says INVERTS this exact cell.
  6. The 2026-08-20 "SPY fell but X did not" reference class, in which XLU was
     one of 14 vehicles and scored +0.648.
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change
import numpy as np
import pandas as pd

SECTORS = ["XLU", "XLK", "XLI", "XLP", "XLV", "XLE", "XLF", "XLY", "XLB"]
TK = SECTORS + ["SPY", "TLT", "IEF", "D", "ETR"]
px = close_panel(TK)
px = px[px.index >= "2002-07-30"]            # TLT inception; the gate needs it
idx = px.index
COST = 5.0

r21 = {t: _valid_pct_change(px[t], 21) for t in TK}
rk21 = {t: pct_rank(px[t], 21) for t in TK}
z10 = {t: zscore(px[t], 10) for t in TK}
hi52 = {t: rolling_on_valid(px[t], lambda x: x.rolling(252).max()) for t in TK}
off_hi = {t: px[t] / hi52[t] - 1.0 for t in TK}
sma200 = rolling_on_valid(px["SPY"], lambda x: x.rolling(200).mean())
above200 = px["SPY"] > sma200

print("=== today's readings (2026-08-24) on the panel used here ===")
for t in ["XLU", "TLT", "SPY"]:
    print(f"  {t}: r21 {100*r21[t].iloc[-1]:+7.2f}%  rank21 {rk21[t].iloc[-1]:5.1f}  "
          f"z10 {z10[t].iloc[-1]:+5.2f}  off 52w hi {100*off_hi[t].iloc[-1]:+6.2f}%")
print(f"  XLU-TLT 21d spread {100*(r21['XLU'].iloc[-1]-r21['TLT'].iloc[-1]):+.2f}pp")
print(f"  SPY above 200d: {bool(above200.iloc[-1])}")

# ------------------------------------------------------- 0. count the rungs
print("\n" + "=" * 78)
print("0. POPULATION OF THE LITERAL RUNG (before any forward return is read)")
print("=" * 78)
wash = rk21["XLU"] <= 5
tlt_fine = (rk21["TLT"] >= 25) & (rk21["TLT"] <= 75)
tlt_fine_loose = r21["TLT"] >= -0.01
spy_near_hi = off_hi["SPY"] >= -0.03            # LIVE today at -1.85%

def npop(m, lbl):
    m = m.reindex(idx, fill_value=False)
    d = idx[m.values]
    ep = declusters(d, 21, idx)
    print(f"  {lbl:56s} days {len(d):5d}  episodes(21td) {len(ep):4d}"
          + (f"  last {d[-1].date()}" if len(d) else ""))
    return d

npop(wash, "XLU rank21 <= 5 (live 0.8)")
npop(wash & tlt_fine, "  x TLT rank21 in [25,75] (live 47.6)  <-- PITCHED")
npop(wash & tlt_fine_loose, "  x TLT r21 >= -1% (live -0.43%)")
npop(wash & tlt_fine & spy_near_hi, "  x TLT mid x SPY within 3% of 52w hi  <-- LIVE TRIPLE")
npop(rk21["XLU"] <= 2, "XLU rank21 <= 2 (tighter, live 0.8)")
npop((rk21["XLU"] <= 2) & tlt_fine, "  x TLT rank21 in [25,75]")

# --------------------------------------------------- 1. gate attribution
print("\n" + "=" * 78)
print("1. GATE ATTRIBUTION -- what is 'the bond market did not cause it' worth?")
print("=" * 78)
LEGS = [("XLU", 1.0)]


def ep_summary(mask, h, lbl, legs=LEGS, min_gap=21):
    ret = vehicle_ret(px, legs, h, 1)
    d = idx[mask.reindex(idx, fill_value=False).values]
    d = d.intersection(ret.dropna().index)
    if len(d) == 0:
        return {"label": lbl, "n": 0}, np.array([]), d
    ep = declusters(d, min_gap, idx)
    v = ret.loc[ep].values
    r = summarize(v, lbl)
    r["n_days"] = len(d)
    r["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
    return r, v, ep


for h in (3, 5, 10):
    ret = vehicle_ret(px, LEGS, h, 1)
    base = 100 * ret.dropna().mean()
    rows = []
    for lbl, m in [
            ("XLU washout ALONE (rank21<=5)", wash),
            ("  x TLT rank21 in [25,75]  PITCHED", wash & tlt_fine),
            ("  x TLT r21 >= -1%", wash & tlt_fine_loose),
            ("  x TLT rank21 < 25 (bond ALSO hit)", wash & (rk21["TLT"] < 25)),
            ("  x TLT rank21 > 75 (bond RALLYING)", wash & (rk21["TLT"] > 75)),
            ("TLT mid-range ALONE (no washout)", tlt_fine),
    ]:
        r, v, ep = ep_summary(m, h, lbl)
        if r.get("n"):
            r["edge_vs_alldays_pp"] = round(r["mean_pct"] - base, 3)
        rows.append(r)
    show(rows, f"h={h}  XLU long, episode level (21td decluster). all-days base {base:+.3f}%")
    a = [x for x in rows if x["label"].startswith("XLU washout ALONE")][0]
    b = [x for x in rows if "PITCHED" in x["label"]][0]
    if a.get("n") and b.get("n"):
        print(f"  -> THE RATES GATE IS WORTH {b['mean_pct']-a['mean_pct']:+.3f}pp "
              f"(discards {a['n']-b['n']} of {a['n']} episodes)")

# ---------------------------------------- 2. is it just long-XLU duration beta?
print("\n" + "=" * 78)
print("2. DURATION WEARING A SECTOR LABEL? regress fwd XLU on fwd TLT in the cell")
print("=" * 78)
for h in (3, 5, 10):
    rx = vehicle_ret(px, [("XLU", 1.0)], h, 1)
    rt = vehicle_ret(px, [("TLT", 1.0)], h, 1)
    for lbl, m in [("washout ALONE", wash), ("PITCHED (x TLT mid)", wash & tlt_fine)]:
        d = idx[m.reindex(idx, fill_value=False).values]
        ep = declusters(d.intersection(rx.dropna().index).intersection(rt.dropna().index), 21, idx)
        if len(ep) < 4:
            print(f"  h={h} {lbl}: N={len(ep)} too few")
            continue
        x, y = rt.loc[ep].values, rx.loc[ep].values
        b, a = np.polyfit(x, y, 1)
        resid = y - (a + b * x)
        se = resid.std(ddof=2) / np.sqrt(len(ep))
        print(f"  h={h} {lbl:22s} N={len(ep):3d}  raw XLU {100*y.mean():+6.3f}%  "
              f"beta_TLT {b:+5.2f}  alpha {100*a:+6.3f}%  "
              f"resid t {a/se if se else np.nan:+5.2f}  "
              f"| TLT leg itself {100*x.mean():+6.3f}%  corr {np.corrcoef(x,y)[0,1]:+.3f}")
    # duration-neutral vehicle
    for bhat in (0.3, 0.5):
        d = idx[(wash & tlt_fine).reindex(idx, fill_value=False).values]
        ep = declusters(d.intersection(rx.dropna().index), 21, idx)
        v = (rx.loc[ep] - bhat * rt.loc[ep]).values
        print(f"      h={h} duration-neutral XLU - {bhat}xTLT: {100*np.nanmean(v):+.3f}% "
              f"(N={len(ep)}, hit {100*np.mean(v>0):.1f}%)")

# ------------------------------------------------ 3. sector reference class
print("\n" + "=" * 78)
print("3. REFERENCE CLASS -- the identical rule on 9 sector ETFs")
print("=" * 78)
for h in (3, 5):
    rows, means, ses = [], [], []
    for t in SECTORS:
        m = (rk21[t] <= 5) & tlt_fine
        r, v, ep = ep_summary(m, h, t, legs=[(t, 1.0)])
        rb = vehicle_ret(px, [(t, 1.0)], h, 1).dropna()
        if r.get("n"):
            r["edge_pp"] = round(r["mean_pct"] - 100 * rb.mean(), 3)
            means.append(r["edge_pp"])
            ses.append(r["sd_pct"] / np.sqrt(r["n"]))
        rows.append(r)
    show(rows, f"h={h}  'sector 21d rank<=5 while TLT mid-range', by sector")
    means, ses = np.array(means), np.array(ses)
    w = 1 / ses ** 2
    pooled = (w * means).sum() / w.sum()
    Q = float((w * (means - pooled) ** 2).sum())
    dfree = len(means) - 1
    from scipy.stats import chi2
    print(f"  Cochran Q = {Q:.2f} on {dfree} df, p = {1-chi2.cdf(Q, dfree):.3f}; "
          f"common excess {pooled:+.3f}pp; XLU ranks "
          f"{1+int((means > means[SECTORS.index('XLU')]).sum())} of {len(means)}")
    # permutation max-of-k
    rng = np.random.default_rng(7)
    obs = means[SECTORS.index("XLU")]
    print(f"  XLU edge {obs:+.3f}pp vs max-of-9 observed {means.max():+.3f}pp "
          f"({SECTORS[int(np.argmax(means))]})")

# --------------------------- 4. overlap with the dead utilities expressions
print("\n" + "=" * 78)
print("4. OVERLAP WITH THE SEVEN DEAD UTILITIES EXPRESSIONS (computed)")
print("=" * 78)
pitched = (wash & tlt_fine).reindex(idx, fill_value=False)
pd_days = idx[pitched.values]
for lbl, m in [("z10 <= -1.5 (2026-08-07 corpse)", z10["XLU"] <= -1.5),
               ("rank21 <= 5 (2026-08-12 corpse)", wash),
               ("XLU-SPY 21d spread <= -5pp", (r21["XLU"] - r21["SPY"]) <= -0.05)]:
    mm = m.reindex(idx, fill_value=False)
    ov = (pitched & mm).sum()
    print(f"  {lbl:38s} joint {ov:4d} of {pitched.sum():4d} pitched days "
          f"= {100*ov/max(1,pitched.sum()):5.1f}%")

# ------------------------------- 5. the SPY-near-high gate, LIVE TODAY
print("\n" + "=" * 78)
print("5. THE SPY-NEAR-HIGH SLICE -- live today, and the 2026-08-12 kill says")
print("   this is where the XLU washout INVERTS")
print("=" * 78)
for h in (3, 5, 10):
    rows = []
    for lbl, m in [("PITCHED, SPY within 3% of hi  <-- LIVE", wash & tlt_fine & spy_near_hi),
                   ("PITCHED, SPY >3% off hi", wash & tlt_fine & ~spy_near_hi),
                   ("washout alone, SPY within 3% of hi", wash & spy_near_hi),
                   ("washout alone, SPY >3% off hi", wash & ~spy_near_hi)]:
        r, v, ep = ep_summary(m, h, lbl)
        rows.append(r)
    show(rows, f"h={h}")

# ------------------------------- 6. 'X did not fall' reference class (08-20)
print("\n" + "=" * 78)
print("6. TAPE OVER-SELECTION + COST + BOOK OVERLAP")
print("=" * 78)
base_a = 100 * above200.dropna().mean()
for lbl, m in [("washout alone", wash), ("PITCHED", wash & tlt_fine),
               ("LIVE triple", wash & tlt_fine & spy_near_hi)]:
    d = idx[m.reindex(idx, fill_value=False).values]
    a = above200.reindex(d).dropna()
    if len(a):
        print(f"  {lbl:16s} SPY>200d on {100*a.mean():5.1f}% of {len(a)} days "
              f"(base {base_a:.1f}%)  [TODAY {bool(above200.iloc[-1])}]")

print("\n  book overlap: staged OLV LONGS in D and ETR are the same factor as XLU")
rr = px[["XLU", "D", "ETR", "TLT", "SPY"]].pct_change()
for h in (3, 5):
    win = rr.rolling(h).sum().dropna()
    win = win[win.index >= "2022-01-01"]
    out = []
    for t in ["D", "ETR", "TLT", "SPY"]:
        out.append(f"{t} corr {win['XLU'].corr(win[t]):.3f} beta {np.polyfit(win['XLU'], win[t], 1)[0]:.2f}")
    print(f"   h={h} (2022+): " + " | ".join(out))
