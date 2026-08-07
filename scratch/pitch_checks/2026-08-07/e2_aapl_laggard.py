"""E2: "The index's biggest laggard" -- LONG AAPL outright when AAPL 5d-rank is at the
floor while SPY sits at its 52w high.

The whole test is the CONTROL: AAPL's own unconditional drift is enormous, so a long
looks good against zero. Everything is reported as EXCESS over AAPL's same-window drift.
Real order: trigger day D close -> ENTER MOC D+1 -> EXIT MOC D+1+h.
MSFT is run as a mechanism check (is "biggest mega-cap laggard bounces" general?).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _common import *  # noqa: F401,F403

px = load_prices(["SPY", "AAPL", "MSFT"])
spy = px["SPY"]["Close"]
idx = spy.index
dist_spy = (spy / spy.rolling(252).max() - 1) * 100


def fwd_entry_next(s, h):
    return s.shift(-(h + 1)) / s.shift(-1) - 1.0


def run(name: str, s: pd.Series, rk_thr: float, near: float | None, h: int, verbose=False):
    s = s.reindex(idx).ffill()
    rk = pct_rank(s, 5)
    f = fwd_entry_next(s, h)
    valid = rk.notna() & f.notna() & dist_spy.notna()
    cond = rk <= rk_thr
    if near is not None:
        cond = cond & (dist_spy >= -near)
    d = idx[cond & valid]
    ep = declusters(d, h + 1, idx)
    v = f[ep].dropna().values
    ctrl = f[valid].values
    s_ep = summarize(v, f"{name} trig rk<={rk_thr} near={near} h={h}")
    s_ct = summarize(ctrl, f"{name} uncond same window h={h}")
    s_ep["excess_pct"] = s_ep.get("mean_pct", np.nan) - s_ct["mean_pct"]
    s_ep["ctrl_mean"] = s_ct["mean_pct"]
    s_ep["n_daylevel"] = len(d)
    return s_ep, s_ct, ep, v, ctrl, f, valid


# ---------- N grid required by the brief ----------
print("=== E2 N grid: AAPL rank5 threshold x SPY-near-high condition (h=5) ===")
rows = []
for rt in (3, 5, 10):
    for near in (1.0, 2.0, None):
        se, sc, ep, v, ctrl, _, _ = run("AAPL", px["AAPL"]["Close"], rt, near, 5)
        rows.append(dict(rk5_thr=rt, spy_near=("none" if near is None else f"{near}%"),
                         n_days=se["n_daylevel"], n_epi=se.get("n", 0),
                         mean=round(se.get("mean_pct", np.nan), 3),
                         ctrl=round(sc["mean_pct"], 3),
                         excess=round(se.get("mean_pct", np.nan) - sc["mean_pct"], 3),
                         t=round(se.get("t", np.nan), 2),
                         hit=round(se.get("hit", np.nan), 0)))
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== E2 same grid, h=10 ===")
rows = []
for rt in (3, 5, 10):
    for near in (1.0, 2.0, None):
        se, sc, ep, v, ctrl, _, _ = run("AAPL", px["AAPL"]["Close"], rt, near, 10)
        rows.append(dict(rk5_thr=rt, spy_near=("none" if near is None else f"{near}%"),
                         n_days=se["n_daylevel"], n_epi=se.get("n", 0),
                         mean=round(se.get("mean_pct", np.nan), 3),
                         ctrl=round(sc["mean_pct"], 3),
                         excess=round(se.get("mean_pct", np.nan) - sc["mean_pct"], 3),
                         t=round(se.get("t", np.nan), 2),
                         hit=round(se.get("hit", np.nan), 0)))
print(pd.DataFrame(rows).to_string(index=False))

# ---------- headline cell: rk<=5, SPY within 1%, h=5 and h=10 ----------
for h in (5, 10):
    print(f"\n########## E2 HEADLINE  AAPL rk5<=5 & SPY within 1% of 52wh, h={h} ##########")
    se, sc, ep, v, ctrl, f, valid = run("AAPL", px["AAPL"]["Close"], 5, 1.0, h)
    cond = (pct_rank(px["AAPL"]["Close"].reindex(idx).ffill(), 5) <= 5) & (dist_spy >= -1.0) & valid
    dlev = idx[cond]
    show([summarize(f[dlev].values, "day-level trigger"),
          summarize(v, "episode-level trigger"),
          summarize(ctrl, "ctrl A: AAPL uncond same window"),
          summarize(f[f.notna()].values, "ctrl B: AAPL all-days")],
         f"E2 h={h} vs controls")
    print(f"EXCESS over same-window AAPL drift: {se['mean_pct']-sc['mean_pct']:+.3f}% "
          f"per {h}td trade")
    a, b = v, ctrl[~np.isnan(ctrl)]
    tw = (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1)/len(a) + b.var(ddof=1)/len(b))
    print(f"Welch t vs own drift: {tw:+.2f}")
    print(f"bootstrap P(mean<=0) episodes: {bootstrap_p_le0(v):.4f}")
    print(f"bootstrap P(EXCESS<=0) episodes: {bootstrap_p_le0(v - sc['mean_pct']/100):.4f}")
    if len(v):
        j = int(np.argmax(v)); k = int(np.argmin(v))
        print(f"best {ep[j].date()} {100*v[j]:+.2f}%  worst {ep[k].date()} {100*v[k]:+.2f}%")
        show([summarize(np.delete(v, j), "drop-BEST"), summarize(np.delete(v, k), "drop-WORST")],
             f"E2 h={h} drop-one")
    show(era_split(ep, v, "2018-01-01"), f"E2 h={h} era 2018")
    show(era_split(ep, v, "2013-01-01"), f"E2 h={h} era 2013")
    for lo, hi in [(2000, 2008), (2008, 2013), (2013, 2018), (2018, 2022), (2022, 2027)]:
        m = (ep >= pd.Timestamp(f"{lo}-01-01")) & (ep < pd.Timestamp(f"{hi}-01-01"))
        if m.sum():
            ss = summarize(v[m], f"{lo}-{hi}")
            print(f"  {ss['label']:>10s} n={ss['n']:3d} mean={ss['mean_pct']:+.3f}% "
                  f"hit={ss['hit']:.0f}% worst={ss['worst_pct']:+.2f}%")
    print(f"  episode dates: {[str(d.date()) for d in ep]}")

# ---------- MSFT mechanism check ----------
print("\n########## E2 MECHANISM CHECK: same trigger on MSFT ##########")
rows = []
for h in (5, 10):
    for rt in (5, 10):
        se, sc, ep, v, ctrl, _, _ = run("MSFT", px["MSFT"]["Close"], rt, 1.0, h)
        rows.append(dict(tkr="MSFT", h=h, rk=rt, n=se.get("n", 0),
                         mean=round(se.get("mean_pct", np.nan), 3),
                         ctrl=round(sc["mean_pct"], 3),
                         excess=round(se.get("mean_pct", np.nan) - sc["mean_pct"], 3),
                         t=round(se.get("t", np.nan), 2), hit=round(se.get("hit", np.nan), 0)))
        se, sc, ep, v, ctrl, _, _ = run("AAPL", px["AAPL"]["Close"], rt, 1.0, h)
        rows.append(dict(tkr="AAPL", h=h, rk=rt, n=se.get("n", 0),
                         mean=round(se.get("mean_pct", np.nan), 3),
                         ctrl=round(sc["mean_pct"], 3),
                         excess=round(se.get("mean_pct", np.nan) - sc["mean_pct"], 3),
                         t=round(se.get("t", np.nan), 2), hit=round(se.get("hit", np.nan), 0)))
print(pd.DataFrame(rows).to_string(index=False))

# ---------- CPI-in-window on the headline cell ----------
cpi = pd.DatetimeIndex(load_events(["cpi"])["date"])
ppi = pd.DatetimeIndex(load_events(["ppi"])["date"])
both = pd.DatetimeIndex(sorted(set(cpi) | set(ppi)))
pos = pd.Series(range(len(idx)), index=idx)
for h in (5, 10):
    se, sc, ep, v, ctrl, f, valid = run("AAPL", px["AAPL"]["Close"], 5, 1.0, h)
    mk = []
    for d in ep:
        p = pos[d]
        if p + 1 + h >= len(idx):
            mk.append(False); continue
        lo, hi = idx[p + 1], idx[p + 1 + h]
        mk.append(bool(((both > lo) & (both <= hi)).any()))
    mk = np.array(mk, dtype=bool)
    show([summarize(v[mk], f"h={h} CPI/PPI in hold"), summarize(v[~mk], f"h={h} neither")],
         f"E2 h={h} CPI/PPI split")
