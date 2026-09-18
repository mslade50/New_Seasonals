"""A4 ROUND 1 -- SPY forward from a CPI print session with ^TNX at a 252-day high.

Anchor = the CPI release SESSION close (k=0). Gate = ^TNX closing at (or inside a
whisker of) its trailing-252 maximum on that same session.

Direction is NOT pre-specified. This script measures it.

Conventions enforced here:
  - ^TNX is a caret series and carries bars on NYSE closures the equity complex
    does not. Everything is computed on SPY's calendar with ^TNX reindexed and
    forward-filled onto it.
  - TWO entries are reported: lag=1 (the standard tradeable MOC the session
    AFTER the print) and lag=0 (which for TODAY is real, because today IS the
    print session and the MOC is placeable this afternoon).
  - The control is ALL CPI SESSIONS, not all days. An event cell measured
    against an all-days control is the 2026-08-10 registry trap.
  - The RATES PARENT (^TNX at a 252d high, no CPI) is measured separately
    because watchlist 44 reported it at -0.169% at h=3 and the composer asked
    whether that is what was already measured.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

ASOF = pd.Timestamp("2026-09-10")
TK = ["SPY", "^TNX", "TLT", "IWM", "QQQ"]
raw = load_prices(TK)
SP = raw["SPY"].index                      # THE equity calendar
PX = pd.DataFrame({t: raw[t]["Close"].reindex(SP).ffill() for t in TK})
PX = PX.rename(columns={"^TNX": "TNX"})
print(f"SPY calendar: {len(SP)} sessions {SP[0].date()} .. {SP[-1].date()}")

tnx = PX["TNX"]
tnx_max = rolling_on_valid(tnx, lambda x: x.rolling(252).max())
tnx_dist = (tnx / tnx_max - 1.0) * 100      # 0.00 == exactly at the high
tnx_chg252 = (tnx - tnx.shift(252)) * 100

print(f"LIVE 2026-09-10: ^TNX {tnx.loc[ASOF]:.4f}  dist from 252d max "
      f"{tnx_dist.loc[ASOF]:+.4f}%  252-session change {tnx_chg252.loc[ASOF]:+.1f} bp")

cpi = load_events(["cpi"])["date"]
cpi_days = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(cpi).normalize())
                                   & set(SP)))
print(f"CPI sessions on the SPY calendar: {len(cpi_days)} "
      f"({cpi_days[0].date()} .. {cpi_days[-1].date()})")

IS_CPI = pd.Series(False, index=SP)
IS_CPI.loc[cpi_days] = True

GATES = {
    "EXACT  ^TNX == 252d max": (tnx_dist >= -1e-9),
    "WHISKER 0.10%":           (tnx_dist >= -0.10),
    "WHISKER 0.25%":           (tnx_dist >= -0.25),
    "WHISKER 0.50%":           (tnx_dist >= -0.50),
    "WHISKER 1.00%":           (tnx_dist >= -1.00),
}

print("\n" + "=" * 78)
print("1. DIRECTION AND HORIZON -- CPI session x ^TNX at a 252d high")
print("=" * 78)
for lag in (0, 1):
    for gname, g in GATES.items():
        m = (IS_CPI & g.fillna(False))
        dts = SP[m.values]
        if len(dts) == 0:
            print(f"  lag={lag} {gname}: no anchors")
            continue
        row = []
        for h in (1, 2, 3, 5, 10):
            r = fwd_lag(PX["SPY"], h, lag)
            v = r.loc[dts].dropna()
            ctrl = r.loc[cpi_days].dropna()          # ALL CPI sessions
            row.append(f"h{h}: {100*v.mean():+.3f}% "
                       f"(vs CPI-all {100*ctrl.mean():+.3f}, "
                       f"edge {100*(v.mean()-ctrl.mean()):+.3f}, "
                       f"hit {100*(v>0).mean():.0f}%, n={len(v)})")
        print(f"  lag={lag} {gname:<26s} N={len(dts)}")
        for s in row:
            print(f"      {s}")

print("\n" + "=" * 78)
print("2. THE DEFENDED CELL (whisker 0.25%) IN FULL, both lags, h=1..5")
print("=" * 78)
GATE = (tnx_dist >= -0.25).fillna(False)
m = IS_CPI & GATE
anchors = SP[m.values]
print(f"  anchors ({len(anchors)}): {[str(d.date()) for d in anchors]}")
print(f"  years: {sorted(set(anchors.year))}")
for lag in (0, 1):
    for h in (1, 2, 3, 5):
        r = fwd_lag(PX["SPY"], h, lag)
        v = r.loc[anchors].dropna()
        cp = r.loc[cpi_days].dropna()
        al = r.dropna()
        loc = local_control(SP[r.notna().values], anchors)
        w = int((v > 0).sum())
        print(f"\n  --- lag={lag} h={h}  N={len(v)} ---")
        show([summarize(v.values, "COND CPI x TNX-high"),
              summarize(cp.values, "CTRL all CPI sessions"),
              summarize(al.values, "CTRL all days"),
              summarize(r.loc[loc].dropna().values, "CTRL local +/-126td")],
             "")
        print(f"    record {w}-{len(v)-w}, sign p(>=) {sign_test(w, len(v)):.4f}, "
              f"sign p(<=) {sign_test(len(v)-w, len(v)):.4f}, "
              f"bootstrap P(mean<=0) {bootstrap_p_le0(v.values):.3f}")
        print(f"    {cluster_note(v.index, v.values)}")

print("\n" + "=" * 78)
print("3. GATE ATTRIBUTION -- what does the DISCARDED COMPLEMENT pay?")
print("=" * 78)
for lag in (0, 1):
    for h in (1, 3, 5):
        r = fwd_lag(PX["SPY"], h, lag)
        keep = r.loc[anchors].dropna()
        comp_days = SP[(IS_CPI & ~GATE).values]
        comp = r.loc[comp_days].dropna()
        parent_days = SP[GATE.values]
        parent = r.loc[parent_days].dropna()
        # rates parent DECLUSTERED (it is a persistent state, not an event)
        pe = declusters(pd.DatetimeIndex(parent.index), 10, SP)
        print(f"  lag={lag} h={h}: RETAINED (CPI & TNX-high) n={len(keep)} "
              f"{100*keep.mean():+.3f}% | DISCARDED (CPI & not TNX-high) "
              f"n={len(comp)} {100*comp.mean():+.3f}% | RATES PARENT "
              f"(TNX-high any day) n={len(parent)} {100*parent.mean():+.3f}% "
              f"[episodes n={len(pe)} {100*r.loc[pe].dropna().mean():+.3f}%]")

print("\n" + "=" * 78)
print("4. MIDTERM SPLIT (2026 is midterm, year%4==2)")
print("=" * 78)
for lag in (0, 1):
    for h in (1, 3, 5):
        r = fwd_lag(PX["SPY"], h, lag)
        v = r.loc[anchors].dropna()
        mt = pd.DatetimeIndex(v.index).year % 4 == 2
        show([summarize(v[mt].values, f"MIDTERM (N={int(mt.sum())})"),
              summarize(v[~mt].values, f"non-midterm (N={int((~mt).sum())})")],
             f"lag={lag} h={h}")
        cpv = fwd_lag(PX["SPY"], h, lag).loc[cpi_days].dropna()
        cmt = pd.DatetimeIndex(cpv.index).year % 4 == 2
        print(f"    control: all CPI sessions midterm {100*cpv[cmt].mean():+.3f}% "
              f"(n={int(cmt.sum())}) vs non-midterm {100*cpv[~cmt].mean():+.3f}%")

print("\n" + "=" * 78)
print("5. ERA SPLIT on the defended cell")
print("=" * 78)
for lag in (0, 1):
    for h in (1, 3):
        r = fwd_lag(PX["SPY"], h, lag)
        v = r.loc[anchors].dropna()
        show(era_split(pd.DatetimeIndex(v.index), v.values), f"lag={lag} h={h}")

print("\n" + "=" * 78)
print("6. TAIL -- FOMC 2026-09-16 (+3 td) and quad witching 2026-09-18 (+5 td)")
print("=" * 78)
for lag in (0, 1):
    for h in (3, 5):
        r = fwd_lag(PX["SPY"], h, lag)
        v = r.loc[anchors].dropna()
        fo = event_in_window(pd.DatetimeIndex(v.index), SP, h, lag,
                             ("fomc_decision",))
        qw = event_in_window(pd.DatetimeIndex(v.index), SP, h, lag,
                             ("quad_witching",))
        show([summarize(v.values[fo], f"FOMC in hold (N={int(fo.sum())})"),
              summarize(v.values[~fo], f"no FOMC (N={int((~fo).sum())})"),
              summarize(v.values[qw], f"quad witching in hold (N={int(qw.sum())})"),
              summarize(v.values[~qw], f"no QW (N={int((~qw).sum())})")],
             f"lag={lag} h={h}")

print("\n" + "=" * 78)
print("7. COST -- SPY round trip ~1-2 bps")
print("=" * 78)
for lag in (0, 1):
    for h in (1, 3, 5):
        r = fwd_lag(PX["SPY"], h, lag)
        v = r.loc[anchors].dropna()
        cp = r.loc[cpi_days].dropna()
        edge = 100 * 100 * (v.mean() - cp.mean())
        print(f"  lag={lag} h={h}: raw {100*100*v.mean():+.1f} bps, "
              f"edge-over-CPI-control {edge:+.1f} bps -> "
              f"{abs(edge)/2.0:.1f}x a 2 bps round trip")
