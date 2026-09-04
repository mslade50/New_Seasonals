"""C1 kill attempt 3: is it vol at all, or is it SPY beta wearing a vol costume?

Plus the two controls the first script owed:
  C. era-MATCHED tdom control (the A3 table compared post-break anchors to a
     control that still contained -1x days, which is unfair to the candidate;
     fix it and report the honest post-break excess).
  D. the overnight leg vs SVXY's UNCONDITIONAL overnight drift.  B4 found the
     only post-2018 pulse is the close->open gap into the print; short-vol
     ETPs earn a structurally positive overnight, so that leg needs its own
     control before it counts as a CPI effect.
  E. ^VIX / ^VIX3M / term-structure on the same anchor, 2000+ (~300 events).
     If VIX genuinely falls into and on the print, the mechanism is real.
  F. regress the SVXY window return on the SAME-WINDOW SPY return and report
     the residual.  A -0.5x ETP is ~-0.5 beta to SPY, so an equity drift on
     print days manufactures exactly the C1 table.

Run: python scratch/pitch_checks/2026-08-11/a1_svxy_cpi_mechanism.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a1_lab import (  # noqa: E402
    SVXY_LEV_BREAK, anchor_dates, event_sessions, tdom_of,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, declusters, fwd_lag, load_events, load_prices, show,
    sign_test, summarize,
)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)

HORIZONS = (1, 2, 3, 5, 10)
OFFSET = 2

px = close_panel(["SVXY", "SPY", "^VIX", "^VIX3M"])
all_dates = px.index
pos = pd.Series(np.arange(len(all_dates)), index=all_dates)
TDOM = tdom_of(all_dates)
ev = load_events(["cpi"])
anch_all = declusters(anchor_dates(ev, "cpi", OFFSET, all_dates), 5, all_dates)


def era_matched(f: pd.Series, anchors: pd.DatetimeIndex,
                lo: pd.Timestamp, hi: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
    """(conditional values, tdom-matched control values) both restricted to
    the SAME calendar era."""
    fa = f.dropna()
    fa = fa[(fa.index >= lo) & (fa.index < hi)]
    a = anchors[(anchors >= lo) & (anchors < hi)]
    a = a[a.isin(fa.index)]
    ent_tdom = TDOM.reindex(all_dates)[pos[a].values + 1].values
    m = TDOM.reindex(fa.index).isin(set(ent_tdom.tolist())) & ~fa.index.isin(a)
    return fa.reindex(a).dropna().values, fa[m].values


# ---------------------------------------------------------------------------
print("=" * 78)
print("C. era-MATCHED tdom control (control drawn from the same era only)")
print("=" * 78)
svxy = px["SVXY"].dropna()
ERAS = (("-1x era 2011-10..2018-02", pd.Timestamp("2011-10-01"), SVXY_LEV_BREAK),
        ("-0.5x era 2018-02+ (LIVE)", SVXY_LEV_BREAK, pd.Timestamp("2030-01-01")),
        ("-0.5x, 2021+", pd.Timestamp("2021-01-01"), pd.Timestamp("2030-01-01")))
rows = []
for lbl, lo, hi in ERAS:
    for h in HORIZONS:
        f = fwd_lag(svxy, h, lag=1)
        v, ctl = era_matched(f, anch_all, lo, hi)
        if len(v) < 5:
            continue
        st = summarize(v, f"{lbl} h={h}")
        st["ctl_pct"] = 100 * ctl.mean()
        st["ctl_n"] = len(ctl)
        st["excess_pct"] = st["mean_pct"] - st["ctl_pct"]
        st["signp"] = sign_test(int((v > 0).sum()), len(v))
        rows.append(st)
show(rows, "C1. SVXY CPI cell vs an era-matched tdom control")

# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("D. the overnight leg vs SVXY's unconditional overnight drift")
print("=" * 78)
o = load_prices(["SVXY"])["SVXY"]["Open"]
c = load_prices(["SVXY"])["SVXY"]["Close"]
on_all = (o / c.shift(1) - 1.0).dropna()          # every session's close->open
cpi_sess = event_sessions(ev, "cpi", all_dates)
cpi_sess = cpi_sess[cpi_sess.isin(on_all.index)]
rows = []
for lbl, lo, hi in (("full 2011-10+", pd.Timestamp("2000-01-01"), pd.Timestamp("2030-01-01")),
                    ("-1x era", pd.Timestamp("2000-01-01"), SVXY_LEV_BREAK),
                    ("-0.5x era (LIVE)", SVXY_LEV_BREAK, pd.Timestamp("2030-01-01")),
                    ("2021+", pd.Timestamp("2021-01-01"), pd.Timestamp("2030-01-01"))):
    sub = on_all[(on_all.index >= lo) & (on_all.index < hi)]
    ce = sub.reindex(cpi_sess).dropna()
    non = sub[~sub.index.isin(cpi_sess)]
    # tdom-matched non-CPI overnight control
    ct = set(TDOM.reindex(cpi_sess).dropna().astype(int).tolist())
    non_t = non[TDOM.reindex(non.index).isin(ct)]
    rows.append({"era": lbl, "n_cpi": len(ce),
                 "cpi_overnight_bps": 1e4 * ce.mean(),
                 "all_overnight_bps": 1e4 * non.mean(),
                 "tdom_overnight_bps": 1e4 * non_t.mean(),
                 "excess_vs_tdom_bps": 1e4 * (ce.mean() - non_t.mean()),
                 "cpi_t": ce.mean() / (ce.std(ddof=1) / np.sqrt(len(ce))),
                 "cpi_hit": 100 * (ce > 0).mean(),
                 "signp": sign_test(int((ce > 0).sum()), len(ce))})
show(rows, "D1. is the CPI overnight special, or is it just SVXY overnight?")

# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("E. does ^VIX actually fall?  2000+, ~300 CPI events, no ETP involved")
print("=" * 78)
vix, vix3m = px["^VIX"].dropna(), px["^VIX3M"].dropna()
ratio = (px["^VIX"] / px["^VIX3M"]).dropna()
rows = []
for name, s in (("^VIX", vix), ("^VIX3M", vix3m), ("VIX/VIX3M ratio", ratio)):
    a = anch_all[anch_all.isin(s.index)]
    for h in (1, 2, 3):
        f = fwd_lag(s, h, lag=1)
        v = f.reindex(a).dropna()
        fa = f.dropna()
        ent_tdom = TDOM.reindex(all_dates)[pos[a[a.isin(fa.index)]].values + 1].values
        m = TDOM.reindex(fa.index).isin(set(ent_tdom.tolist())) & ~fa.index.isin(a)
        ctl = fa[m]
        st = summarize(v.values, f"{name} h={h}")
        st["ctl_pct"] = 100 * ctl.mean()
        st["excess_pct"] = st["mean_pct"] - st["ctl_pct"]
        st["signp_DOWN"] = sign_test(int((v.values < 0).sum()), len(v))
        rows.append(st)
show(rows, "E1. VIX change entering the session before CPI (negative = crush)")

print("\nE2. same, split at 2018 (the registry's era claim, on VIX itself)")
rows = []
for name, s in (("^VIX", vix), ("VIX/VIX3M", ratio)):
    for lbl, lo, hi in (("2000-2017", pd.Timestamp("2000-01-01"), pd.Timestamp("2018-01-01")),
                        ("2018+", pd.Timestamp("2018-01-01"), pd.Timestamp("2030-01-01"))):
        f = fwd_lag(s, 3, lag=1)
        v, ctl = era_matched(f, anch_all, lo, hi)
        if len(v) < 5:
            continue
        st = summarize(v, f"{name} h=3 {lbl}")
        st["ctl_pct"] = 100 * ctl.mean()
        st["excess_pct"] = st["mean_pct"] - st["ctl_pct"]
        st["signp_DOWN"] = sign_test(int((v < 0).sum()), len(v))
        rows.append(st)
show(rows, "E2. VIX era split")

# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("F. SPY-beta decomposition: regress the SVXY window on the SPY window")
print("=" * 78)
spy = px["SPY"].dropna()
for lbl, lo, hi in ERAS:
    print(f"\n-- {lbl} --")
    for h in (1, 3, 5):
        fs = fwd_lag(svxy, h, lag=1)
        fp = fwd_lag(spy, h, lag=1)
        j = pd.concat([fs.rename("y"), fp.rename("x")], axis=1).dropna()
        j = j[(j.index >= lo) & (j.index < hi)]
        if len(j) < 60:
            continue
        # fit on ALL days in the era (the trade's own days are ~4% of them)
        X = np.column_stack([np.ones(len(j)), j["x"].values])
        beta, *_ = np.linalg.lstsq(X, j["y"].values, rcond=None)
        resid = j["y"].values - X @ beta
        r = pd.Series(resid, index=j.index)
        a = anch_all[anch_all.isin(r.index)]
        rc = r.reindex(a).dropna()
        cond_spy = j["x"].reindex(a).dropna()
        all_spy = j["x"]
        print(f"  h={h:<2} N_cond={len(rc):<4} alpha={1e4*beta[0]:+7.1f}bps "
              f"beta_SPY={beta[1]:+.2f} | RAW cond {100*j['y'].reindex(a).dropna().mean():+.3f}% "
              f"-> RESIDUAL {100*rc.mean():+.3f}% "
              f"t={rc.mean()/(rc.std(ddof=1)/np.sqrt(len(rc))):+.2f} "
              f"hit={100*(rc>0).mean():.1f}% signp={sign_test(int((rc>0).sum()), len(rc)):.4f}")
        print(f"        SPY on the same days {100*cond_spy.mean():+.3f}% vs "
              f"SPY all days {100*all_spy.mean():+.3f}%  "
              f"(equity drift explains {100*beta[1]*(cond_spy.mean()-all_spy.mean())/max(1e-9, j['y'].reindex(a).dropna().mean()-j['y'].mean())*1:.0f}% "
              f"of the SVXY excess)")
