"""C2 kill: long QQQ / short SPY into the CPI print (anchor = 2 sessions
before a CPI, entry lag=1 MOC, so h=1 IS the print session).

The five attacks, in order of expected lethality:
  1. BETA. Equal-dollar vs point-in-time beta-neutral (2026-08-10 GDX/GLD).
  2. Price the legs before the spread.
  3. Era + the spread's OWN unconditional drift (QQQ/SPY secular trend).
  4. Today's state is the INVERSE of the cell (QQQ is the laggard today).
  5. Cost: two legs.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, load_events, fwd_lag, declusters, summarize, sign_test,
    bootstrap_p_le0, local_control, cluster_note, battery, pct_rank,
)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)

TKRS = ["SPY", "QQQ"]
OFFSET = 2                 # today's data bar sits 2 sessions before CPI
HORIZONS = (1, 2, 3, 5)
BETA_WIN = 126             # rolling beta window (daily returns)

px = close_panel(TKRS).dropna()
all_dates = px.index
pos = pd.Series(np.arange(len(all_dates)), index=all_dates)
TDOM = pd.Series(all_dates, index=all_dates).groupby(
    [all_dates.year, all_dates.month]).cumcount() + 1

ev = load_events(["cpi"])
cpi_dates = pd.DatetimeIndex(sorted(ev["date"].unique()))
anch = []
for d in cpi_dates:
    loc = all_dates.searchsorted(d)
    j = loc - OFFSET
    if 0 <= j < len(all_dates):
        anch.append(all_dates[j])
anch = pd.DatetimeIndex(sorted(set(anch)))
anch = declusters(anch, 5, all_dates)
anch = anch[anch.isin(all_dates)]

# ---------------------------------------------------------------- beta, PIT
rs, rq = px["SPY"].pct_change(), px["QQQ"].pct_change()
cov = rq.rolling(BETA_WIN).cov(rs)
var = rs.rolling(BETA_WIN).var()
beta = (cov / var)                       # as of close D (point in time)
print(f"rolling {BETA_WIN}d beta QQQ~SPY: full-sample mean {beta.mean():.3f}, "
      f"median {beta.median():.3f}, today ({all_dates[-1].date()}) "
      f"{beta.iloc[-1]:.3f}")
print(f"beta on the trigger days: mean {beta.reindex(anch).mean():.3f}, "
      f"min {beta.reindex(anch).min():.3f}, max {beta.reindex(anch).max():.3f}")

# ------------------------------------------------- spreads, both constructions
def spreads(h):
    q = fwd_lag(px["QQQ"], h, 1)
    s = fwd_lag(px["SPY"], h, 1)
    eq = q - s                       # equal dollar
    bn = q - beta * s                # beta-neutral, PIT beta
    return q, s, eq, bn


# tdom-matched control set (entry session's trading-day-of-month)
ent_tdom = set(TDOM.reindex(all_dates)[pos[anch].values + 1].dropna().astype(int).tolist())

print("\n" + "=" * 100)
print("ATTACK 1+2: price the legs, then both spread constructions, vs three controls")
print("=" * 100)
rows = []
for h in HORIZONS:
    q, s, eq, bn = spreads(h)
    for name, ser in (("QQQ leg", q), ("SPY leg", s),
                      ("SPREAD equal-$", eq), ("SPREAD beta-neutral", bn)):
        v = ser.reindex(anch).dropna()
        base = ser.dropna()
        ctl_tdom = base[TDOM.reindex(base.index).isin(ent_tdom).values
                        & ~base.index.isin(anch)]
        loc = local_control(base.index, anch, 126)
        ctl_loc = base.reindex(loc).dropna()
        st = summarize(v.values)
        wins = int((v.values > 0).sum())
        rows.append(dict(
            h=h, leg=name, N=st["n"],
            mean_pct=round(st["mean_pct"], 3),
            all_days=round(100 * base.mean(), 3),
            tdom_ctl=round(100 * ctl_tdom.mean(), 3),
            local_ctl=round(100 * ctl_loc.mean(), 3),
            xs_alldays=round(st["mean_pct"] - 100 * base.mean(), 3),
            xs_tdom=round(st["mean_pct"] - 100 * ctl_tdom.mean(), 3),
            hit=round(st["hit"], 1),
            signp=round(sign_test(wins, st["n"]), 4),
            t=round(st["t"], 2),
        ))
print(pd.DataFrame(rows).to_string(index=False))

print("\n--- THE DECISIVE PAIR (h=3, the recon's best cell) ---")
q, s, eq, bn = spreads(3)
for name, ser in (("equal-$", eq), ("beta-neutral", bn)):
    v = ser.reindex(anch).dropna()
    base = ser.dropna()
    st = summarize(v.values)
    wins = int((v.values > 0).sum())
    print(f"{name:>14}: N={st['n']} mean={st['mean_pct']:+.3f}% "
          f"OWN drift={100*base.mean():+.3f}% "
          f"EXCESS={st['mean_pct'] - 100*base.mean():+.3f}% "
          f"hit={st['hit']:.1f}% t={st['t']:+.2f} "
          f"signp={sign_test(wins, st['n']):.4f} "
          f"bootP(mean<=0)={bootstrap_p_le0(v.values):.3f}")
    print(f"                 {cluster_note(v.index, v.values, 3)}")

print("\n" + "=" * 100)
print("ATTACK 3: era stability + is the CPI anchor adding anything over the")
print("          QQQ/SPY secular trend? (a filter that does not filter)")
print("=" * 100)
ERAS = {
    "pre-2018": lambda d: d < pd.Timestamp("2018-01-01"),
    "2018+": lambda d: d >= pd.Timestamp("2018-01-01"),
    "ex 2020-2021 (tech era)": lambda d: ~d.year.isin([2020, 2021]),
    "ex 2023-2025 (AI era)": lambda d: ~d.year.isin([2023, 2024, 2025]),
    "ex BOTH tech+AI eras": lambda d: ~d.year.isin([2020, 2021, 2023, 2024, 2025]),
    "2010-2019 only": lambda d: (d.year >= 2010) & (d.year <= 2019),
}
for h in (1, 3):
    q, s, eq, bn = spreads(h)
    print(f"\n-- h={h} --")
    er = []
    for lbl, f in ERAS.items():
        for name, ser in (("equal-$", eq), ("beta-neut", bn)):
            v = ser.reindex(anch).dropna()
            m = f(pd.DatetimeIndex(v.index))
            vv = v[m]
            base = ser.dropna()
            bm = f(pd.DatetimeIndex(base.index))
            st = summarize(vv.values)
            if not st["n"]:
                continue
            wins = int((vv.values > 0).sum())
            er.append(dict(era=lbl, build=name, N=st["n"],
                           mean=round(st["mean_pct"], 3),
                           own_drift=round(100 * base[bm].mean(), 3),
                           excess=round(st["mean_pct"] - 100 * base[bm].mean(), 3),
                           hit=round(st["hit"], 1),
                           signp=round(sign_test(wins, st["n"]), 4)))
    print(pd.DataFrame(er).to_string(index=False))

# year histogram of the equal-dollar spread at h=3
q, s, eq, bn = spreads(3)
v = eq.reindex(anch).dropna()
byyr = pd.DataFrame({"eq": v.values, "bn": bn.reindex(v.index).values},
                    index=v.index).groupby(v.index.year).agg(["sum", "count"])
print("\nyear histogram, h=3 (sum of spread return, pp):")
print((byyr * 100).round(2).to_string())

print("\n" + "=" * 100)
print("ATTACK 4: today's state is the INVERSE of the cell. QQQ is the LAGGARD")
print("          (21d rank 30.2, 63d rank 22.2, -3.28% off its 52w high) while")
print("          SPY is AT its high. Which bucket does today land in?")
print("=" * 100)
rel = px["QQQ"] / px["SPY"]
rel21 = pct_rank(rel, 21)             # 21d relative-strength rank of QQQ vs SPY
rel63 = pct_rank(rel, 63)
hi52_q = px["QQQ"] / px["QQQ"].rolling(252).max() - 1.0
hi52_s = px["SPY"] / px["SPY"].rolling(252).max() - 1.0
gap52 = 100 * (hi52_q - hi52_s)       # QQQ dist-to-high minus SPY's, in pp
r21q, r21s = pct_rank(px["QQQ"], 21), pct_rank(px["SPY"], 21)
rankgap = r21q - r21s

today = all_dates[-1]
print(f"today {today.date()}: rel21 rank {rel21.loc[today]:.1f}, "
      f"rel63 rank {rel63.loc[today]:.1f}, "
      f"QQQ-SPY 52wh gap {gap52.loc[today]:+.2f}pp, "
      f"QQQ 21d rank {r21q.loc[today]:.1f} vs SPY {r21s.loc[today]:.1f} "
      f"(rankgap {rankgap.loc[today]:+.1f})")

for h in (1, 3):
    q, s, eq, bn = spreads(h)
    print(f"\n-- h={h}, trigger days split by the state GOING IN --")
    out = []
    splits = {
        "rel21 <= 50 (QQQ lagging, TODAY)": rel21.reindex(anch) <= 50,
        "rel21 > 50 (QQQ leading)": rel21.reindex(anch) > 50,
        "rel63 <= 30 (TODAY 22.2-ish)": rel63.reindex(anch) <= 30,
        "rel63 > 30": rel63.reindex(anch) > 30,
        "52wh gap <= -2pp (QQQ well below, TODAY -3.3)": gap52.reindex(anch) <= -2,
        "52wh gap > -2pp": gap52.reindex(anch) > -2,
        "rankgap < 0 (QQQ momentum laggard, TODAY)": rankgap.reindex(anch) < 0,
        "rankgap >= 0": rankgap.reindex(anch) >= 0,
    }
    for lbl, m in splits.items():
        sel = pd.DatetimeIndex(m[m.fillna(False)].index)
        for name, ser in (("equal-$", eq), ("beta-neut", bn)):
            vv = ser.reindex(sel).dropna()
            st = summarize(vv.values)
            if not st["n"]:
                continue
            wins = int((vv.values > 0).sum())
            out.append(dict(bucket=lbl, build=name, N=st["n"],
                            mean=round(st["mean_pct"], 3),
                            hit=round(st["hit"], 1), t=round(st["t"], 2),
                            signp=round(sign_test(wins, st["n"]), 4),
                            worst=round(st["worst_pct"], 2)))
    print(pd.DataFrame(out).to_string(index=False))

print("\n" + "=" * 100)
print("ATTACK 5: cost. Two legs. QQQ+SPY are the cheapest ETFs alive; assume")
print("          1.5 bps all-in per leg round trip => ~3 bps for the pair.")
print("=" * 100)
for h in HORIZONS:
    q, s, eq, bn = spreads(h)
    for name, ser in (("equal-$", eq), ("beta-neutral", bn)):
        v = ser.reindex(anch).dropna()
        edge = 100 * v.mean() * 100
        print(f"  h={h} {name:>13}: edge {edge:+6.1f} bps vs 3.0 bps cost "
              f"-> {edge/3.0:+5.1f}x (need >= 5x)")

print("\n" + "=" * 100)
print("ROUND 1 BATTERY (equal-dollar spread, h=3) for the record")
print("=" * 100)
mask = pd.Series(False, index=all_dates)
mask.loc[anch] = True
battery(px, mask, [("QQQ", 1.0), ("SPY", -1.0)], 3,
        "C2 long QQQ / short SPY, 2 sessions before CPI", cost_bps=1.5,
        min_gap=5, event_kinds=("cpi",))
