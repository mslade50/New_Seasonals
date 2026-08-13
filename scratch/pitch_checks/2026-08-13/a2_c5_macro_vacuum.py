"""C5 round 1 - the macro vacuum.

Trigger: the session of the month's LAST cpi-or-ppi print, gated on the next
{nfp, cpi, ppi, fomc_decision} being >= K trading sessions away. Today
(2026-08-13) is exactly that: PPI prints, next release is NFP 09-04 (+16 td).

Legs tested: long SPY (event-premium-decay pays drift) and long SVXY (it pays
short-vol carry). Both directions of the SVXY leg reported.

Decisive tests, per the registry's two most repeated traps:
  A. GATE ATTRIBUTION - the same anchor with NO gate, and the complement
     (month-last inflation print WITH a release inside the window). If the
     gate does not move the number, nothing may be attributed to the vacuum.
  B. TRADING-DAY-OF-MONTH matched control - the month's last inflation print
     is always mid-month, and the registry already has a tdom cell reading
     +0.215% at tdom 14 with no event anywhere.
  C. What the gate actually selects (it is suspected to be "not an FOMC
     month" wearing a vacuum label).
Both lag=1 (the book convention) and lag=0 (the map's MOC-tonight form, legal
here because the trigger is a pure calendar fact known at 08:30) are reported.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pitch_lab import (  # noqa: E402
    bootstrap_p_le0, cluster_note, declusters, load_events, load_prices, show,
    sign_test, summarize, fwd_lag,
)

RELEASES = ["nfp", "cpi", "ppi", "fomc_decision"]
px = load_prices(["SPY", "SVXY", "^VIX"])
spy = px["SPY"]["Close"].dropna()
svxy = px["SVXY"]["Close"].dropna()
vix = px["^VIX"]["Close"].dropna()
IDX = spy.index

ev = load_events(RELEASES)
infl = ev[ev["event"].isin(["cpi", "ppi"])].copy()
rel = ev["date"].sort_values().reset_index(drop=True)

# --- month-last inflation print sessions -----------------------------------
infl["ym"] = infl["date"].dt.to_period("M")
last_infl = infl.groupby("ym")["date"].max().sort_values()
print(f"month-last cpi-or-ppi prints: {len(last_infl)} months "
      f"{last_infl.iloc[0].date()} .. {last_infl.iloc[-1].date()}")


def snap(d):
    """map an event date onto the trading index (that session, else next)."""
    p = IDX.searchsorted(d)
    if p >= len(IDX):
        return None
    return IDX[p]


pos = pd.Series(range(len(IDX)), index=IDX)


def td_gap_to_next_release(d):
    """trading sessions from d to the next release strictly after d."""
    nxt = rel[rel > d]
    if len(nxt) == 0:
        return None
    n = snap(nxt.iloc[0])
    p0, p1 = pos.get(d), pos.get(n) if n is not None else None
    if p0 is None or p1 is None:
        return None
    return p1 - p0


anchors, gaps, next_kind = [], [], []
for d in last_infl:
    s = snap(d)
    if s is None or s > IDX[-1]:
        continue
    g = td_gap_to_next_release(s)
    if g is None:
        continue
    anchors.append(s)
    gaps.append(g)
    nxt = rel[rel > s]
    next_kind.append(ev.loc[ev["date"] == nxt.iloc[0], "event"].iloc[0])
A = pd.DataFrame({"date": anchors, "gap_td": gaps, "next": next_kind}).drop_duplicates("date")
A["tdom"] = [int(pos[d] - pos[IDX[(IDX.year == d.year) & (IDX.month == d.month)][0]]) + 1
             for d in A["date"]]
A["year"] = A["date"].dt.year
A["month"] = A["date"].dt.month
print(f"anchors on the trading index: {len(A)}   gap_td distribution:")
print(A["gap_td"].describe().round(1).to_string())
print("\nnext-release kind when gap >= 10:  ",
      A.loc[A.gap_td >= 10, "next"].value_counts().to_dict())
print("next-release kind when gap <  10:  ",
      A.loc[A.gap_td < 10, "next"].value_counts().to_dict())
print(f"\nTODAY's analogue: 2026-08-13 gap to NFP 2026-09-04 = "
      f"{td_gap_to_next_release(pd.Timestamp('2026-08-13'))} td "
      f"(2026-08-13 in index: {pd.Timestamp('2026-08-13') in IDX})")

# --- forward returns -------------------------------------------------------
def leg_ret(series, h, lag):
    return fwd_lag(series, h, lag)


def tdom_matched_control(ret, trig, tdom_map, tdoms, tol=1):
    """mean of ret over non-trigger days whose tdom is within tol of each
    trigger's tdom, averaged with the trigger tdom distribution as weights."""
    pool = ret.dropna()
    tset = set(trig)
    vals = []
    for t in tdoms:
        m = [d for d in pool.index
             if d not in tset and abs(tdom_map.get(d, -99) - t) <= tol]
        if m:
            vals.append(pool.loc[m].mean())
    return float(np.mean(vals)) if vals else np.nan


# tdom map for every trading day
tdom_all = {}
for (y, m), grp in pd.Series(IDX, index=IDX).groupby([IDX.year, IDX.month]):
    for i, d in enumerate(grp.index):
        tdom_all[d] = i + 1

for leg_name, series, cost in (("SPY", spy, 2.0), ("SVXY", svxy, 9.0)):
    for lag in (1, 0):
        for h in (5, 10):
            ret = leg_ret(series, h, lag)
            ok = ret.notna()
            print(f"\n\n########## {leg_name} long, h={h}, lag={lag} ##########")
            rows = []
            for K in (8, 10, 11, 12):
                t = pd.DatetimeIndex([d for d in A.loc[A.gap_td >= K, "date"]
                                      if d in ok.index and ok.loc[d]])
                if len(t) == 0:
                    continue
                v = ret.loc[t].values
                r = summarize(v, f"VACUUM gate K>={K} (N={len(t)})")
                r["boot"] = round(bootstrap_p_le0(v), 3)
                rows.append(r)
            # no-gate anchor + complement
            t_all = pd.DatetimeIndex([d for d in A["date"] if d in ok.index and ok.loc[d]])
            rows.append(summarize(ret.loc[t_all].values,
                                  f"NO GATE all month-last prints (N={len(t_all)})"))
            t_cmp = pd.DatetimeIndex([d for d in A.loc[A.gap_td < 10, "date"]
                                      if d in ok.index and ok.loc[d]])
            rows.append(summarize(ret.loc[t_cmp].values,
                                  f"COMPLEMENT gap<10, release IN window (N={len(t_cmp)})"))
            rows.append(summarize(ret[ok].values, "CTRL-b all days own drift"))
            span = (t_all[0], t_all[-1])
            insp = ok & (ret.index >= span[0]) & (ret.index <= span[1])
            rows.append(summarize(ret[insp].values, "CTRL-a own drift same span"))
            show(rows, f"1+A. {leg_name} h={h} lag={lag}: gate ladder + attribution")

            # tdom matched control at K=10
            t10 = pd.DatetimeIndex([d for d in A.loc[A.gap_td >= 10, "date"]
                                    if d in ok.index and ok.loc[d]])
            tdoms = [tdom_all[d] for d in t10]
            ctl_tdom = tdom_matched_control(ret, t10, tdom_all, tdoms)
            v10 = ret.loc[t10].values
            base_hit = float((ret[ok] > 0).mean())
            wins = int((v10 > 0).sum())
            print(f"  B. tdom-matched control (tdom {sorted(set(tdoms))}): "
                  f"{100*ctl_tdom:+.3f}%   vacuum {100*np.mean(v10):+.3f}%   "
                  f"EXCESS {100*(np.mean(v10)-ctl_tdom):+.3f}%")
            print(f"     sign vs own base rate ({100*base_hit:.1f}%): "
                  f"{wins}-{len(v10)-wins}  p={sign_test(wins, len(v10), base_hit):.4f}")
            edge_bps = 100 * (np.mean(v10) - ctl_tdom) * 100
            print(f"     cost: {cost} bps round trip -> excess {edge_bps:.1f} bps = "
                  f"{edge_bps/cost:.1f}x  (kill below 10 bps excess)")

            # era + midterm + concentration on K=10 episodes
            epi = declusters(t10, 21, ret.index[ok.values])
            vE = ret.loc[epi].values
            mid = np.array([y % 4 == 2 for y in pd.DatetimeIndex(epi).year])
            pre = pd.DatetimeIndex(epi) < pd.Timestamp("2018-01-01")
            show([summarize(vE, f"K>=10 episodes (N={len(epi)})"),
                  summarize(vE[pre], "pre-2018"), summarize(vE[~pre], "2018+"),
                  summarize(vE[mid], "midterm years"),
                  summarize(vE[~mid], "non-midterm")],
                 f"2+3. {leg_name} h={h} lag={lag} era / cycle (episodes)")
            print(f"  {cluster_note(epi, vE)}")
            aug = pd.DatetimeIndex(epi).month == 8
            if aug.sum():
                show([summarize(vE[aug], f"August episodes (N={int(aug.sum())})")])

# --- C. mechanism: does implied vol actually fall more in a vacuum? --------
print("\n\n########## C. mechanism check: VIX path in vacuum vs not ##########")
for h in (5, 10):
    rv = fwd_lag(vix, h, 1)
    ok = rv.notna()
    t10 = pd.DatetimeIndex([d for d in A.loc[A.gap_td >= 10, "date"]
                            if d in ok.index and ok.loc[d]])
    tcm = pd.DatetimeIndex([d for d in A.loc[A.gap_td < 10, "date"]
                            if d in ok.index and ok.loc[d]])
    show([summarize(rv.loc[t10].values, f"VIX chg, vacuum K>=10 (N={len(t10)})"),
          summarize(rv.loc[tcm].values, f"VIX chg, release in window (N={len(tcm)})"),
          summarize(rv[ok].values, "VIX chg all days")],
         f"VIX forward change h={h} (premium decay should be MORE negative in a vacuum)")

# --- D. what the gate selects ---------------------------------------------
print("\n\n########## D. is the gate just 'no FOMC in the window'? ##########")
fomc = load_events(["fomc_decision"])["date"]
A2 = A.copy()
A2["fomc_within_10td"] = [
    bool(((fomc > d) & (fomc <= (IDX[min(pos[d] + 11, len(IDX) - 1)]))).any())
    for d in A2["date"]]
print(pd.crosstab(A2["gap_td"] >= 10, A2["fomc_within_10td"]).to_string())
print("\nrows: gate ON (True) vs OFF; cols: FOMC lands inside 10 td")
