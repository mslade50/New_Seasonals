"""C9 -- "the translation channel": long EFA against short SPY when the dollar
is washed out (DX-Y.NYB 21d return rank <= 15).

Claimed mechanism is a translation IDENTITY rather than a behavioural story:
EFA is unhedged, so a falling dollar mechanically adds to its USD return while
SPY gets nothing.

Adjacent closed cells: 2026-08-26 killed the EM version (EEM's excess is
negative at all five horizons, gate anti-selective); 2026-08-25 killed EEM-vs-
EFA on a false premise; watchlist 27 parks the bare dollar washout to a
non-midterm year.  So the DEVELOPED version is what is open.

Required probes:
 (i)   leg attribution + beta-neutral residual (the registry killed a SPY/QQQ
       pair on exactly this: "measured beta 0.617, so the equal-dollar pair
       carries -0.383 units of beta")
 (ii)  does the dollar gate do any work over the unconditional EFA-SPY spread?
 (iii) is the edge just the contemporaneous FX move -- run the dollar's OWN
       forward return over the same window and compare
 (iv)  definition neighbours: rank 5/10/15/20 x lookback 10/21/42
 (v)   era + midterm split
 (vi)  cost ~6-8 bps two-leg round trip, demand 30-40 bps
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pitch_lab import (load_prices, pct_rank, fwd_lag, summarize, show,  # noqa: E402
                       declusters, local_control, cluster_note, sign_test,
                       bootstrap_p_le0, era_split)

DXY, EFA, SPY = "DX-Y.NYB", "EFA", "SPY"
ASOF = pd.Timestamp("2026-08-27")
HS = (1, 2, 3, 5, 7, 10, 21)
RANK_MAX, LB_N = 15.0, 21

px = load_prices([DXY, EFA, SPY, "UUP", "EEM"])
S = {t: px[t]["Close"].dropna().loc[:ASOF] for t in px}
for t in S:
    print(f"  {t:<10} {S[t].index[0].date()} .. {S[t].index[-1].date()} "
          f"({len(S[t])} sessions)  last {S[t].iloc[-1]:.2f}")

# ranks on the dollar's OWN series, never on a union panel
d_rank = pct_rank(S[DXY], LB_N)
print(f"\n  live: DXY {LB_N}d rank = {d_rank.iloc[-1]:.1f}  "
      f"(gate <= {RANK_MAX})  PREMISE LIVE = {bool(d_rank.iloc[-1] <= RANK_MAX)}")
print(f"  EFA 21d rank {pct_rank(S[EFA],21).iloc[-1]:.1f}   "
      f"SPY 21d rank {pct_rank(S[SPY],21).iloc[-1]:.1f}")

cal = S[EFA].index.intersection(S[SPY].index).intersection(S[DXY].index)
PX = pd.DataFrame({t: S[t].reindex(cal) for t in [EFA, SPY, DXY]}).dropna()
print(f"  common calendar {PX.index[0].date()} .. {PX.index[-1].date()} "
      f"({len(PX)} sessions)")

MASK = (d_rank.reindex(PX.index) <= RANK_MAX)
print(f"  gate days on the common calendar: {int(MASK.sum())} of {len(PX)} "
      f"({100*MASK.mean():.1f}%)")

# beta of EFA on SPY
er = PX[EFA].pct_change()
sr = PX[SPY].pct_change()
ok = er.notna() & sr.notna()
BETA = float(np.polyfit(sr[ok], er[ok], 1)[0])
print(f"\n  measured EFA beta on SPY (daily, full common calendar) = {BETA:.3f}")
print(f"  -> an EQUAL-DOLLAR long-EFA/short-SPY pair carries {BETA-1:+.3f} units "
      f"of net SPY beta")


def leg_ret(legs, h):
    out = None
    for t, w in legs:
        r = fwd_lag(PX[t], h, 1)
        out = w * r if out is None else out + w * r
    return out


def cell(legs, h, mask=None, label="", min_gap=None):
    r = leg_ret(legs, h)
    m = (mask if mask is not None else MASK).reindex(PX.index, fill_value=False)
    dts = PX.index[m.values & r.notna().values]
    epi = declusters(dts, min_gap or h, PX.index)
    s = summarize(r.loc[epi].values, label)
    s["n_days"] = len(dts)
    return s, dts, epi, r


# --------------------------------------------------------------------------
print("\n=== C9.1  leg attribution at every horizon ===")
for h in HS:
    rows = []
    forms = [([(EFA, 1.0)], "leg A long EFA"),
             ([(SPY, -1.0)], "leg B short SPY"),
             ([(EFA, 1.0), (SPY, -1.0)], "equal-$ pair"),
             ([(EFA, 1.0), (SPY, -BETA)], f"beta-neutral (short {BETA:.2f}x)")]
    for legs, lbl in forms:
        s, dts, epi, r = cell(legs, h, label=f"{lbl} (epi {0})")
        s["label"] = lbl
        rows.append(s)
    # controls: same forms, ALL days
    for legs, lbl in forms:
        r = leg_ret(legs, h).dropna()
        rows.append(summarize(r.values, f"CTRL all days: {lbl}"))
    show(rows, f"h={h}")

# --------------------------------------------------------------------------
print("\n=== C9.2  does the dollar gate do any work? (gate ON vs gate OFF) ===")
for h in (3, 5, 10, 21):
    rows = []
    for legs, lbl in [([(EFA, 1.0), (SPY, -1.0)], "equal-$ pair"),
                      ([(EFA, 1.0), (SPY, -BETA)], "beta-neutral")]:
        r = leg_ret(legs, h)
        on = MASK.reindex(PX.index, fill_value=False).values & r.notna().values
        off = (~MASK.reindex(PX.index, fill_value=True).values) & r.notna().values
        e_on = declusters(PX.index[on], h, PX.index)
        rows.append(summarize(r.loc[e_on].values, f"{lbl} GATE ON (epi {len(e_on)})"))
        rows.append(summarize(r[off].values, f"{lbl} GATE OFF (days {int(off.sum())})"))
        rows.append(summarize(r.dropna().values, f"{lbl} ALL days"))
        # local control
        loc = local_control(PX.index[r.notna().values], PX.index[on])
        rows.append(summarize(r.loc[loc].values, f"{lbl} CTRL-c local +/-126td"))
    show(rows, f"gate attribution, h={h}")

# --------------------------------------------------------------------------
print("\n=== C9.3  is the pair's edge just the currency move it claims to harvest? ===")
print("  (dollar forward return over the SAME entry window vs the pair's edge)")
for h in (3, 5, 10, 21):
    s_pair, dts, epi, rp = cell([(EFA, 1.0), (SPY, -1.0)], h, label="pair")
    rd = fwd_lag(PX[DXY], h, 1)
    rows = [summarize(rp.loc[epi].values, f"equal-$ pair, gate episodes h={h}"),
            summarize(rd.loc[epi].dropna().values, f"DXY fwd, same episodes h={h}"),
            summarize(rd.dropna().values, f"DXY fwd, all days h={h}")]
    show(rows, "")
    a = 100 * rp.loc[epi].mean()
    b = 100 * rd.loc[epi].dropna().mean()
    print(f"    pair {a:+.3f}%  vs  dollar move {b:+.3f}%  -> the pair captures "
          f"{a/(-b):.2f}x of the dollar's OWN move (sign-flipped) "
          if abs(b) > 1e-9 else "")
    # regression of the pair's forward return on the dollar's forward return
    j = pd.concat([rp, rd], axis=1).dropna()
    j.columns = ["pair", "dxy"]
    bb = np.polyfit(j["dxy"], j["pair"], 1)
    resid = j["pair"] - (bb[0] * j["dxy"] + bb[1])
    ei = pd.DatetimeIndex(epi).intersection(j.index)
    show([summarize(resid.loc[ei].values, f"pair EX-DOLLAR residual, episodes h={h}"),
          summarize(resid.values, f"residual all days h={h}")],
         f"    pair ~ dxy slope {bb[0]:.3f} (all days)")

# --------------------------------------------------------------------------
print("\n=== C9.4  definition neighbours: rank threshold x lookback ===")
for h in (5, 10):
    rows = []
    for lb in (10, 21, 42):
        dr = pct_rank(S[DXY], lb)
        for thr in (5, 10, 15, 20):
            m = (dr.reindex(PX.index) <= thr)
            s, dts, epi, _ = cell([(EFA, 1.0), (SPY, -1.0)], h, mask=m,
                                  label=f"lb={lb} rank<={thr}")
            s["label"] = f"lb={lb} rank<={thr}"
            rows.append(s)
    show(rows, f"equal-$ pair neighbours, h={h}")

# --------------------------------------------------------------------------
print("\n=== C9.5  era + midterm split (equal-$ pair and beta-neutral) ===")
for h in (5, 10):
    for legs, lbl in [([(EFA, 1.0), (SPY, -1.0)], "equal-$"),
                      ([(EFA, 1.0), (SPY, -BETA)], "beta-neutral")]:
        s, dts, epi, r = cell(legs, h, label=lbl)
        v = r.loc[epi].values
        yrs = pd.DatetimeIndex(epi).year
        show([summarize(v[yrs < 2018], "pre-2018"), summarize(v[yrs >= 2018], "2018+"),
              summarize(v[yrs % 4 == 2], "midterm"), summarize(v[yrs % 4 != 2], "non-mid")],
             f"{lbl} h={h}  (episodes N={len(v)})")
        w = int((v > 0).sum())
        print(f"    record {w}-{len(v)-w}  sign p="
              f"{sign_test(max(w,len(v)-w), len(v)):.3f}  "
              f"bootstrap P(mean<=0)={bootstrap_p_le0(v):.3f}")
        print(f"    {cluster_note(epi, v)}")

# --------------------------------------------------------------------------
print("\n=== C9.6  cost check ===")
for h in (3, 5, 10, 21):
    s, _, epi, r = cell([(EFA, 1.0), (SPY, -1.0)], h, label="pair")
    sn, _, _, rn = cell([(EFA, 1.0), (SPY, -BETA)], h, label="bn")
    print(f"  h={h:<3} equal-$ {s['mean_pct']:+.3f}% = {100*s['mean_pct']:.1f} bps "
          f"-> {100*s['mean_pct']/8:.2f}x an 8 bps two-leg round trip  |  "
          f"beta-neutral {sn['mean_pct']:+.3f}%  (need >= 30-40 bps)")

print("\n=== C9.7  sanity: the EM version the registry already killed, for contrast ===")
cal2 = S["EEM"].index.intersection(PX.index)
PX2 = pd.DataFrame({"EEM": S["EEM"].reindex(cal2), "SPY": S[SPY].reindex(cal2)}).dropna()
for h in (5, 10):
    r = fwd_lag(PX2["EEM"], h, 1) - fwd_lag(PX2["SPY"], h, 1)
    m = (d_rank.reindex(PX2.index) <= RANK_MAX)
    dts = PX2.index[m.fillna(False).values & r.notna().values]
    epi = declusters(dts, h, PX2.index)
    show([summarize(r.loc[epi].values, f"EEM-SPY gate episodes h={h}"),
          summarize(r.dropna().values, f"EEM-SPY all days h={h}")], "")
