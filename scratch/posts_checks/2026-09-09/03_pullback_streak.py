"""SPY: "108 sessions since the last 5% pullback" -- verify, then price it.

DEFINITIONS (stated once, used everywhere below)
-----------------------------------------------
  close      = SPY adjusted close from data/master_prices.parquet (pitch_lab
               basis).  Cache starts 2000-01-03, so the running max is SEEDED
               at 2000-01-03: there is no pre-2000 high in this series.
  drawdown   = close / running_max(close) - 1        (running = EXPANDING max,
               i.e. the all-time-high-to-date on this series, as the question
               specifies.  A rolling-252d variant is printed in section 0 for
               reconciliation because the production risk page uses that one.)
  breach day = a session whose drawdown <= -5% (or <= -10%).
  streak_t   = sessions SINCE the last breach day = t - i(last breach).  A
               breach day itself has streak 0; the next session has streak 1.
               This is exactly risk_dashboard_v2's days_since_5pct arithmetic
               (len - 1 - get_loc(last breach)).
  completed streak = a maximal RUN of non-breach sessions bounded by a breach
               day on both sides; its length = the streak counter on the run's
               last session (>= 1).  Two ADJACENT breach days therefore make
               no streak at all (a length-0 gap); those are counted and
               reported separately, never folded into the median.
  left-censored: the run from 2000-01-03 to the first breach is NOT completed
               on the left (no preceding breach) and is excluded, reported.
  current streak: ongoing, hence excluded from the completed distribution and
               instead RANKED against it.

Because the literal definition restarts a streak on the day after the LAST
breach day of a correction -- typically weeks before the index recovers -- a
supplementary EPISODE definition is also reported in section 1b: a 5% episode
runs from the running-max peak to the day the close makes a new all-time high,
and the episode streak is the gap between the end of one episode and the first
breach of the next.  The two medians differ a lot and the difference is the
whole story of the distribution, so both are shown.

Sections:
  0  verify the two reported counts, both max definitions, with a reconciliation
  1  distribution of completed 5% streaks (+1b episode view)
  2  streaks that reached 100: what happened after, and the pullback that ended them
  3  lag-1 forward-return cell anchored on the first session a streak crosses 100
  4  same as 1 for 10% streaks
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa: F401,F403  (load_prices, summarize, show, sign_test, fwd_lag)

import numpy as np
import pandas as pd

ASOF = pd.Timestamp("2026-09-09")
REPORTED_5 = 108
REPORTED_10 = 344
HORIZONS = (5, 21, 63)
CROSS = 100

px = load_prices(["SPY"])
SPY = px["SPY"]
SPY = SPY[SPY.index <= ASOF]
C = SPY["Close"].astype("float64").dropna()
N = len(C)

RUNMAX = C.cummax()
DD = C / RUNMAX - 1.0
ROLLMAX = C.rolling(252).max()
DD252 = C / ROLLMAX - 1.0


def fmt(x, d=2):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


# ---------------------------------------------------------------------------
# 0. verification
# ---------------------------------------------------------------------------
print("=" * 100)
print("0. VERIFY THE REPORTED COUNTS")
print("=" * 100)
print(f"  SPY closes: {N} sessions  {C.index[0].date()} .. {C.index[-1].date()}"
      f"   last close {C.iloc[-1]:.2f}   prior close {C.iloc[-2]:.2f} "
      f"({C.index[-2].date()})")
print(f"  running max today {RUNMAX.iloc[-1]:.2f}   drawdown today "
      f"{100*DD.iloc[-1]:.2f}%")

for lbl, dser in (("running (expanding) max", DD), ("rolling 252d max", DD252)):
    print(f"\n  -- drawdown vs {lbl}")
    for th, rep in ((0.05, REPORTED_5), (0.10, REPORTED_10)):
        br = dser[dser <= -th]
        last = br.index[-1]
        i = C.index.get_loc(last)
        since_today = N - 1 - i
        since_prev = N - 2 - i          # as of the PRIOR session's close
        print(f"     {int(th*100)}% : last breach {last.date()} "
              f"(dd {100*dser.loc[last]:.2f}%)   sessions since, as of "
              f"{C.index[-1].date()} = {since_today}   as of "
              f"{C.index[-2].date()} = {since_prev}   [reported {rep}]")

print("\n  RECONCILIATION of the two borderline April sessions (this basis):")
sub = pd.DataFrame({"close": C.loc["2026-04-01":"2026-04-08"],
                    "run_max": RUNMAX.loc["2026-04-01":"2026-04-08"],
                    "dd_pct": 100 * DD.loc["2026-04-01":"2026-04-08"],
                    "dd252_pct": 100 * DD252.loc["2026-04-01":"2026-04-08"]})
print(sub.round(4).to_string())

RD2 = Path(__file__).resolve().parents[3] / "data" / "rd2_spy_ohlc.parquet"
if RD2.exists():
    r = pd.read_parquet(RD2)["Close"].astype("float64").dropna()
    rdd = 100 * (r / r.rolling(252).max() - 1.0)
    print(f"\n  The production risk page's OWN yfinance cache "
          f"({RD2.name}, last bar {r.index[-1].date()}), rolling-252 basis:")
    print(rdd.loc["2026-04-01":"2026-04-08"].round(4).to_string())
    br5 = rdd[rdd <= -5.0]
    print(f"     -> last 5% breach in THAT vintage: {br5.index[-1].date()} "
          f"(dd {br5.iloc[-1]:.2f}%)")


# ---------------------------------------------------------------------------
# streak machinery
# ---------------------------------------------------------------------------
def breach_mask(th: float) -> np.ndarray:
    return (DD <= -th).values


def streak_counter(mask: np.ndarray) -> np.ndarray:
    """sessions since the last True in mask; NaN before the first True."""
    idx = np.arange(len(mask), dtype=float)
    last = np.where(mask, idx, np.nan)
    last = pd.Series(last).ffill().values
    return idx - last


def completed_streaks(th: float) -> pd.DataFrame:
    """One row per maximal run of non-breach sessions bounded by breach days."""
    mask = breach_mask(th)
    bidx = np.flatnonzero(mask)
    rows = []
    for a, b in zip(bidx[:-1], bidx[1:]):
        length = b - a - 1
        if length < 1:
            continue                      # adjacent breach days: no streak
        rows.append({
            "start": C.index[a + 1],      # first session of the run
            "end": C.index[b - 1],        # last session before the next breach
            "length": int(length),
            "ended_on": C.index[b],       # the breach day that ended it
            "end_i": int(b),
            "start_i": int(a + 1),
        })
    return pd.DataFrame(rows)


def zero_gaps(th: float) -> int:
    bidx = np.flatnonzero(breach_mask(th))
    return int((np.diff(bidx) == 1).sum())


def current_streak(th: float) -> dict:
    mask = breach_mask(th)
    bidx = np.flatnonzero(mask)
    a = int(bidx[-1])
    return {"start": C.index[a + 1], "length": N - 1 - a, "last_breach": C.index[a]}


def episode_of(breach_i: int) -> dict:
    """The drawdown EPISODE containing the breach at position breach_i:
    peak = the session that set the running max in force at the breach;
    trough = min close from the peak until the close makes a NEW all-time high
    (or the end of data, flagged)."""
    peak_val = RUNMAX.iloc[breach_i]
    # last session at/before the breach whose close == the running max
    peak_i = int(np.flatnonzero(C.values[:breach_i + 1] >= peak_val - 1e-12)[-1])
    after = np.flatnonzero(C.values[peak_i + 1:] > peak_val) + peak_i + 1
    if len(after):
        rec_i = int(after[0])
        recovered = True
    else:
        rec_i = len(C) - 1
        recovered = False
    seg = C.values[peak_i:rec_i + 1]
    tr_off = int(np.argmin(seg))
    trough_i = peak_i + tr_off
    return {
        "peak_date": C.index[peak_i], "peak": float(peak_val),
        "trough_date": C.index[trough_i], "trough": float(C.iloc[trough_i]),
        "depth_pct": 100.0 * (float(C.iloc[trough_i]) / float(peak_val) - 1.0),
        "recovered_on": C.index[rec_i] if recovered else None,
        "recovered": recovered,
        "peak_i": peak_i, "trough_i": trough_i, "rec_i": rec_i,
    }


def dist_block(df: pd.DataFrame, th: float, reported: int, label: str) -> None:
    L = df["length"].values.astype(float)
    print(f"\n  completed streaks: {len(L)}   (zero-length gaps between adjacent "
          f"breach days, excluded: {zero_gaps(th)})")
    print(f"  median {np.median(L):.2f}   mean {L.mean():.2f}   sd {L.std(ddof=1):.2f}"
          f"   min {L.min():.2f}   max {L.max():.2f}")
    qs = np.percentile(L, [25, 50, 75, 90, 95, 99])
    print("  quantiles p25/p50/p75/p90/p95/p99 = " + " / ".join(f"{q:.2f}" for q in qs))
    print(f"\n  longest five {label} streaks:")
    top = df.sort_values("length", ascending=False).head(5)
    for _, r in top.iterrows():
        ep = episode_of(int(r["end_i"]))
        print(f"    {int(r['length']):>4} sessions  {r['start'].date()} .. "
              f"{r['end'].date()}   ended by the breach on {r['ended_on'].date()} "
              f"(that episode: peak {ep['peak_date'].date()} -> trough "
              f"{ep['trough_date'].date()}, {ep['depth_pct']:.2f}%)")
    cur = current_streak(th)
    print(f"\n  CURRENT (ongoing, excluded from the distribution above): "
          f"{cur['length']} sessions since the {cur['last_breach'].date()} breach, "
          f"running {cur['start'].date()} .. {C.index[-1].date()}")
    for v, vlab in ((cur["length"], f"{cur['length']} (this basis)"),
                    (reported, f"{reported} (as reported)")):
        below = int((L < v).sum())
        atleast = int((L >= v).sum())
        print(f"    {vlab}: percentile among completed streaks = "
              f"{100.0*below/len(L):.2f}   completed streaks that ever reached "
              f"{v}+ sessions = {atleast} of {len(L)} "
              f"({100.0*atleast/len(L):.2f}%)")


# ---------------------------------------------------------------------------
# 1. 5% streak distribution
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("1. DISTRIBUTION OF COMPLETED 5% STREAKS (literal definition)")
print("=" * 100)
S5 = completed_streaks(0.05)
first5 = C.index[int(np.flatnonzero(breach_mask(0.05))[0])]
print(f"  left-censored leading run excluded: {C.index[0].date()} .. "
      f"{first5.date()} (first 5% breach), "
      f"{int(np.flatnonzero(breach_mask(0.05))[0])} sessions")
dist_block(S5, 0.05, REPORTED_5, "5%")

print("\n  full sorted list of completed 5% streak lengths:")
print("   " + ", ".join(str(int(x)) for x in np.sort(S5["length"].values)[::-1]))

# --- 1b episode view -------------------------------------------------------
print("\n" + "-" * 100)
print("1b. SUPPLEMENTARY: EPISODE-BASED 5% STREAKS")
print("    (a streak starts the day the index makes a NEW all-time high after")
print("     the previous 5% episode, and ends at the next 5% breach)")
print("-" * 100)
mask5 = breach_mask(0.05)
bidx5 = np.flatnonzero(mask5)
eps, seen = [], set()
for b in bidx5:
    ep = episode_of(int(b))
    key = ep["peak_i"]
    if key in seen:
        continue
    seen.add(key)
    ep["first_breach_i"] = int(b)
    eps.append(ep)
erows = []
for prev, nxt in zip(eps[:-1], eps[1:]):
    if not prev["recovered"]:
        continue
    length = nxt["first_breach_i"] - prev["rec_i"]
    if length < 1:
        continue
    erows.append({"start": C.index[prev["rec_i"]], "end": C.index[nxt["first_breach_i"]],
                  "length": int(length), "depth_pct": nxt["depth_pct"]})
E5 = pd.DataFrame(erows)
EL = E5["length"].values.astype(float)
print(f"  episodes found: {len(eps)}   completed episode streaks: {len(EL)}")
print(f"  median {np.median(EL):.2f}   mean {EL.mean():.2f}   max {EL.max():.2f}")
print("  longest five:")
for _, r in E5.sort_values("length", ascending=False).head(5).iterrows():
    print(f"    {int(r['length']):>4} sessions  {r['start'].date()} .. "
          f"{r['end'].date()}   next episode depth {r['depth_pct']:.2f}%")
last_ep = eps[-1]
if last_ep["recovered"]:
    cur_e = N - 1 - last_ep["rec_i"]
    print(f"  CURRENT episode streak: {cur_e} sessions since the new high on "
          f"{C.index[last_ep['rec_i']].date()}   percentile "
          f"{100.0*int((EL < cur_e).sum())/len(EL):.2f}   "
          f"reached {cur_e}+: {int((EL >= cur_e).sum())} of {len(EL)}")


# ---------------------------------------------------------------------------
# 2. streaks that reached 100
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print(f"2. COMPLETED 5% STREAKS THAT REACHED {CROSS} SESSIONS -- WHAT ENDED THEM")
print("=" * 100)
R = S5[S5["length"] >= CROSS].copy().reset_index(drop=True)
rows = []
for _, r in R.iterrows():
    cross_i = int(r["start_i"]) + CROSS - 1   # the session whose counter == CROSS
    ep = episode_of(int(r["end_i"]))
    rows.append({
        "streak_start": str(r["start"].date()),
        "crossed_100_on": str(C.index[cross_i].date()),
        "streak_end": str(r["end"].date()),
        "length": int(r["length"]),
        "further_after_100": int(r["length"] - CROSS),
        "breach_on": str(r["ended_on"].date()),
        "peak_on": str(ep["peak_date"].date()),
        "trough_on": str(ep["trough_date"].date()),
        "ending_pullback_pct": round(ep["depth_pct"], 2),
        "recovered_on": str(ep["recovered_on"].date()) if ep["recovered"] else "NOT YET",
    })
T = pd.DataFrame(rows)
print(T.to_string(index=False))
fd = T["further_after_100"].values.astype(float)
dp = T["ending_pullback_pct"].values.astype(float)
print(f"\n  N = {len(T)} completed streaks reached {CROSS}")
print(f"  further sessions after crossing 100: median {np.median(fd):.2f}   "
      f"mean {fd.mean():.2f}   min {fd.min():.2f}   max {fd.max():.2f}")
print(f"  ending pullback (peak->trough close, the episode that ended the streak): "
      f"median {np.median(dp):.2f}%   mean {dp.mean():.2f}%   "
      f"shallowest {dp.max():.2f}%   deepest {dp.min():.2f}%")
cur5 = current_streak(0.05)
print(f"  (the ONGOING streak reached 100 on "
      f"{C.index[int(np.flatnonzero(breach_mask(0.05))[-1]) + CROSS].date()} and is "
      f"{cur5['length'] - CROSS} sessions past 100 so far -- not in the table)")


# ---------------------------------------------------------------------------
# 3. lag-1 forward-return cell anchored on the first crossing of 100
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print(f"3. LAG-1 FORWARD RETURNS FROM THE FIRST SESSION A STREAK CROSSES {CROSS}")
print("   (enter at the NEXT close, exit h sessions after that; anchors are one")
print("    per streak and >= 100 sessions apart, so h=63 windows cannot overlap)")
print("=" * 100)

mask5c = breach_mask(0.05)
bidx5c = np.flatnonzero(mask5c)
anchor_i = []
for a in bidx5c:                              # every streak, completed or not
    ci = int(a) + CROSS
    if ci >= N:
        continue
    nxt = bidx5c[bidx5c > a]
    if len(nxt) and nxt[0] <= ci:             # streak died before reaching 100
        continue
    anchor_i.append(ci)
anchor_i = sorted(set(anchor_i))
ANCH = C.index[anchor_i]
print(f"  anchors: {len(ANCH)}  ->  " + ", ".join(str(d.date()) for d in ANCH))
gaps = np.diff(anchor_i)
print(f"  min gap between anchors: {gaps.min() if len(gaps) else 'n/a'} sessions")

FWD = {h: fwd_lag(C, h, lag=1) for h in HORIZONS}
tab = []
for d in ANCH:
    row = {"anchor": str(d.date()), "close": round(float(C.loc[d]), 2)}
    for h in HORIZONS:
        v = FWD[h].get(d, np.nan)
        row[f"h{h}_pct"] = round(100 * v, 2) if pd.notna(v) else None
    tab.append(row)
print("\n  per-anchor lag-1 forward returns (percent):")
print(pd.DataFrame(tab).to_string(index=False))

rows = []
for h in HORIZONS:
    s = FWD[h]
    v = s.reindex(ANCH).values.astype(float)
    v = v[~np.isnan(v)]
    ctrl = s.dropna().values.astype(float)
    for lab, arr in ((f"CELL  cross-{CROSS}  h={h:>2}", v),
                     (f"  CTRL all sessions h={h:>2}", ctrl)):
        st = summarize(arr, lab)
        if st["n"]:
            up, dn = int((arr > 0).sum()), int((arr < 0).sum())
            st["W-L"] = f"{up}-{dn}"
            st["sign_p"] = round(sign_test(max(up, dn), len(arr)), 4)
            st.pop("sd_pct", None)
        rows.append(st)
show(rows, "lag-1 forward returns, cell vs unconditional")

# --- drawdown-from-anchor probability --------------------------------------
print("\n  P(>= 5% drawdown FROM THE ANCHOR CLOSE within the next h sessions)")
print("    definition: min(close over t+1..t+h) / close_t - 1 <= -5%")
rows = []
cv = C.values
for h in (21, 63):
    fmin = pd.Series(
        [cv[i + 1:i + 1 + h].min() if i + h < len(cv) else np.nan for i in range(len(cv))],
        index=C.index, dtype="float64")
    ratio = fmin / C - 1.0
    cell = ratio.reindex(ANCH).dropna().values
    ctrl = ratio.dropna().values
    for lab, arr in ((f"CELL  cross-{CROSS} h={h}", cell), (f"  CTRL all days h={h}", ctrl)):
        if len(arr) == 0:
            rows.append({"label": lab, "n": 0})
            continue
        rows.append({
            "label": lab, "n": len(arr),
            "P_dd5_pct": 100 * float((arr <= -0.05).mean()),
            "P_dd10_pct": 100 * float((arr <= -0.10).mean()),
            "mean_min_pct": 100 * float(arr.mean()),
            "median_min_pct": 100 * float(np.median(arr)),
            "worst_min_pct": 100 * float(arr.min()),
        })
show(rows, "probability of a >=5% (and >=10%) drawdown from the anchor close")


# ---------------------------------------------------------------------------
# 4. 10% streaks
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("4. DISTRIBUTION OF COMPLETED 10% STREAKS")
print("=" * 100)
S10 = completed_streaks(0.10)
print(f"  left-censored leading run excluded: {C.index[0].date()} .. "
      f"{C.index[int(np.flatnonzero(breach_mask(0.10))[0])].date()} "
      f"({int(np.flatnonzero(breach_mask(0.10))[0])} sessions)")
dist_block(S10, 0.10, REPORTED_10, "10%")
print("\n  full sorted list of completed 10% streak lengths:")
print("   " + ", ".join(str(int(x)) for x in np.sort(S10["length"].values)[::-1]))

print("\nDONE.")
