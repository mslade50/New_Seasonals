"""C1 REFRAMED - the bare August duration seasonal, no event label on it.

The coordinator's correction: the ANCHOR is dead seven ways (Jackson Hole, opex
and VIX-expiry ladders are all plateaus, which the registry names as month
position rather than an event). What has never been tested AS THE TRADE is the
bare August month-position effect the registry established while killing the JH
cell: Aug 6-16 pays +1.025% at t=6.90 over 189 starts with no event involved.

Controls owed, in the order the registry demands:
  A. MONTH-OF-YEAR at matched trading-day-of-month - is AUGUST special, or is
     mid-month special in every month?
  B. the tdom profile inside August - today is tdom 11, the tail of the 6-16
     window. Does the effect reach it?
  C. era, on the bare seasonal (post-2013 arbitrage is a standing registry line)
  D. duration-neutral residual against IEF (the JH version was +0.122% at 50%)
  E. midterm control
  F. C1b reframed: does TLT AT A 52w LOW help or hurt the August cell? (the JH
     sample had zero such anchors; the 189-start August sample should have some)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

BAR = pd.Timestamp("2026-08-14")
px = close_panel(["TLT", "IEF"]).dropna()
IDX = px.index
tlt, ief = px["TLT"], px["IEF"]
tdom = pd.Series(IDX.to_series().groupby([IDX.year, IDX.month]).cumcount() + 1,
                 index=IDX)
F = {h: fwd_lag(tlt, h, 1) for h in (5, 10)}
FI = {h: fwd_lag(ief, h, 1) for h in (5, 10)}
MIDW = (tdom >= 4) & (tdom <= 12)          # today (tdom 11) is inside this
print(f"today = tdom 11. window under test: tdom 4-12 of the month.")

# ---------------------------------------------------- A. MONTH-OF-YEAR CONTROL
print("\n" + "=" * 96)
print("A. MONTH-OF-YEAR at MATCHED tdom 4-12: is AUGUST special?")
print("=" * 96)
for h in (5, 10):
    f = F[h]
    rows = []
    for mo in range(1, 13):
        sel = IDX[(IDX.month == mo) & MIDW.values & f.notna().values]
        v = f.reindex(sel).dropna()
        yr = pd.Series(v.values).groupby(v.index.year.values).mean()
        w = int((yr > 0).sum())
        rows.append({"month": mo, "n_days": len(v), "mean_pct": round(100 * v.mean(), 3),
                     "hit": round(100 * (v > 0).mean(), 1),
                     "yr_rec": f"{w}-{len(yr)-w}",
                     "yr_signp": round(sign_test(w, len(yr)), 4),
                     "mean_2018+": round(100 * v[v.index.year >= 2018].mean(), 3)})
    df = pd.DataFrame(rows).sort_values("mean_pct", ascending=False)
    print(f"\n--- h={h}, sorted by mean (all-days TLT drift = "
          f"{100*f.dropna().mean():+.3f}%) ---")
    print(df.to_string(index=False))
    aug_rank = list(df["month"]).index(8) + 1
    print(f"  AUGUST ranks {aug_rank} of 12 at h={h}")

# ------------------------------------------------- B. tdom PROFILE INSIDE AUGUST
print("\n" + "=" * 96)
print("B. tdom PROFILE INSIDE AUGUST - does the window reach today's tdom 11?")
print("=" * 96)
for h in (5, 10):
    f = F[h]
    rows = []
    for td in range(1, 20):
        sel = IDX[(IDX.month == 8) & (tdom.values == td) & f.notna().values]
        v = f.reindex(sel).dropna()
        if len(v) < 8:
            continue
        w = int((v > 0).sum())
        v18 = v[v.index.year >= 2018]
        rows.append({"aug_tdom": td, "n_yrs": len(v), "mean_pct": round(100 * v.mean(), 3),
                     "rec": f"{w}-{len(v)-w}", "signp": round(sign_test(w, len(v)), 4),
                     "n18": len(v18), "mean_2018+": round(100 * v18.mean(), 3),
                     "rec18": f"{int((v18>0).sum())}-{int((v18<=0).sum())}"})
    show(rows, f"August tdom profile h={h} (one obs per year)")

# ---------------------------------------------------------------- C. ERA
print("\n" + "=" * 96)
print("C. ERA on the bare cell (August tdom 4-12)")
print("=" * 96)
cell_days = IDX[(IDX.month == 8) & MIDW.values]
for h in (5, 10):
    f = F[h]
    v = f.reindex(cell_days).dropna()
    rows = []
    for lo, hi in ((2002, 2012), (2013, 2017), (2018, 2020), (2021, 2025)):
        s = v[(v.index.year >= lo) & (v.index.year <= hi)]
        yr = pd.Series(s.values).groupby(s.index.year.values).mean()
        r = summarize(s.values, f"{lo}-{hi}")
        r["yr_rec"] = f"{int((yr>0).sum())}-{int((yr<=0).sum())}"
        rows.append(r)
    s = v[v.index.year >= 2013]
    yr = pd.Series(s.values).groupby(s.index.year.values).mean()
    r = summarize(s.values, "2013+ (post-arbitrage line)")
    r["yr_rec"] = f"{int((yr>0).sum())}-{int((yr<=0).sum())}"
    rows.append(r)
    s = v[v.index.year >= 2018]
    yr = pd.Series(s.values).groupby(s.index.year.values).mean()
    r = summarize(s.values, "2018+")
    r["yr_rec"] = f"{int((yr>0).sum())}-{int((yr<=0).sum())}"
    rows.append(r)
    show(rows, f"h={h} era ladder")

# --------------------------------------------- D. DURATION-NEUTRAL vs IEF
print("\n" + "=" * 96)
print("D. DURATION-NEUTRAL RESIDUAL vs IEF - is this TLT, or is it duration?")
print("=" * 96)
d_tlt, d_ief = tlt.pct_change(), ief.pct_change()
m = d_tlt.notna() & d_ief.notna()
beta = np.polyfit(d_ief[m], d_tlt[m], 1)[0]
print(f"  daily beta(TLT on IEF) = {beta:.3f}")
for h in (5, 10):
    res = F[h] - beta * FI[h]
    v = res.reindex(cell_days).dropna()
    base = res.dropna()
    show([summarize(F[h].reindex(cell_days).dropna().values, f"TLT outright h={h}"),
          summarize(FI[h].reindex(cell_days).dropna().values, f"IEF outright h={h}"),
          summarize(v.values, f"TLT - {beta:.2f}*IEF residual h={h}"),
          summarize(base.values, f"residual, ALL DAYS h={h}")], f"h={h}")
    w = int((v > 0).sum())
    print(f"  residual excess {100*(v.mean()-base.mean()):+.3f}pp | {w}-{len(v)-w} "
          f"| IEF's own August excess "
          f"{100*(FI[h].reindex(cell_days).dropna().mean()-FI[h].dropna().mean()):+.3f}pp")

# ------------------------------------------------------------ E. MIDTERM
print("\n" + "=" * 96)
print("E. MIDTERM control on the bare cell")
print("=" * 96)
for h in (5, 10):
    f = F[h]
    v = f.reindex(cell_days).dropna()
    mid = v[v.index.year % 4 == 2]
    non = v[v.index.year % 4 != 2]
    show([summarize(mid.values, f"midterm h={h}"),
          summarize(non.values, f"non-midterm h={h}")], f"h={h}")
    yr = pd.Series(mid.values).groupby(mid.index.year.values).mean() * 100
    print("  midterm per-year:", ", ".join(f"{y}:{x:+.2f}" for y, x in yr.items()))

# ------------------------------------- F. C1b REFRAMED: 52w-low gate on August
print("\n" + "=" * 96)
print("F. C1b REFRAMED - does TLT AT a 52w low help or hurt the August cell?")
print("=" * 96)
off_lo = (tlt / tlt.rolling(252).min() - 1.0) * 100
print(f"  today TLT {off_lo.loc[BAR]:.2f}% off its 52w low")
for h in (5, 10):
    f = F[h]
    rows = []
    for thr, lbl in ((1.0, "<=1% of 52w low"), (2.0, "<=2%"), (5.0, "<=5%"),
                     (999, "any (parent)")):
        sel = pd.DatetimeIndex([d for d in cell_days if off_lo.get(d, 999) <= thr])
        v = f.reindex(sel).dropna()
        r = summarize(v.values, f"August tdom 4-12 x TLT {lbl}")
        r["yrs"] = len(set(v.index.year)) if len(v) else 0
        rows.append(r)
    # and the complement
    sel = pd.DatetimeIndex([d for d in cell_days if off_lo.get(d, 999) > 5.0])
    rows.append(summarize(f.reindex(sel).dropna().values,
                          "August tdom 4-12 x TLT >5% off its low"))
    show(rows, f"h={h} 52w-low gate attribution on the August cell")
    sel = pd.DatetimeIndex([d for d in cell_days if off_lo.get(d, 999) <= 1.0])
    v = f.reindex(sel).dropna()
    if len(v):
        print(f"  h={h} gated years: {sorted(set(v.index.year))}  "
              f"n_days={len(v)}  mean {100*v.mean():+.3f}%  "
              f"vs parent {100*f.reindex(cell_days).dropna().mean():+.3f}%  "
              f"-> gate moves it {100*(v.mean()-f.reindex(cell_days).dropna().mean()):+.3f}pp")

# ------------------------------------------------------------- G. LIVE READ
print("\n" + "=" * 96)
print("G. the live entry: tdom 11, 2026, rising-rate regime, TLT at a 52w low")
print("=" * 96)
h = 10
f = F[h]
live_like = pd.DatetimeIndex([d for d in cell_days
                              if d.month == 8 and tdom.loc[d] >= 10
                              and off_lo.get(d, 999) <= 2.0])
v = f.reindex(live_like).dropna()
print(f"  August tdom>=10 AND TLT within 2% of its 52w low: N={len(v)} days, "
      f"years {sorted(set(v.index.year)) if len(v) else '[]'}")
if len(v):
    print(f"  mean {100*v.mean():+.3f}%  hit {100*(v>0).mean():.1f}%  "
          f"worst {100*v.min():+.2f}%")
