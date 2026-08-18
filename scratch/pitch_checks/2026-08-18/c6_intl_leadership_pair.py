"""C6 -- international leadership into a 52w high, EFA (and EWJ) against SPY.

Falsification order, per the brief:
  1. price EFA and SPY legs SEPARATELY first (the standing pair kill)
  2. beta of the intl leg to SPY on the sample -> beta-NEUTRAL residual
  3. horizon scan 1..10
  4. decluster at episode scale (a 52w-high state persists for weeks)
  5. era split
  6. regime over-selection: fraction of trigger days above SPY's 200d vs base
  7. cost: two liquid-ETF legs ~5-6 bps round trip
  8. EWJ as the alternate leg -> specific or generic?

Entry convention lag=1 throughout (signal close D, entry MOC D+1).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, declusters, era_split, fwd_lag, local_control, show,
    summarize, bootstrap_p_le0, sign_test, cluster_note,
)

TK = ["EFA", "EWJ", "SPY"]
MIN_GAP = 21          # episode scale: a 52w-high state persists for weeks
DIST_MAX = -0.005     # within 0.5% of the 52w high
LEAD_MIN = 0.020      # 63d leadership margin in pp (live EFA +3.50pp)

px = close_panel(TK).dropna()
print(f"panel {px.index[0].date()} .. {px.index[-1].date()}  n={len(px)}")

spy = px["SPY"]
spy200 = spy.rolling(200).mean()
spy_above200 = (spy > spy200)


def state(tkr: str) -> pd.DataFrame:
    s = px[tkr]
    hi52 = s.rolling(252).max()
    dist = s / hi52 - 1.0
    lead = s.pct_change(63) - spy.pct_change(63)
    return pd.DataFrame({"dist": dist, "lead": lead})


def rolling_beta(a: pd.Series, b: pd.Series, win: int = 252) -> pd.Series:
    ra, rb = a.pct_change(), b.pct_change()
    cov = ra.rolling(win).cov(rb)
    var = rb.rolling(win).var()
    return cov / var


def run(tkr: str, dist_max: float = DIST_MAX, lead_min: float = LEAD_MIN,
        label: str = "") -> dict:
    st = state(tkr)
    beta = rolling_beta(px[tkr], spy)          # point-in-time, trailing 252d
    mask = (st["dist"] >= dist_max) & (st["lead"] >= lead_min)
    mask = mask.fillna(False) & beta.notna()
    trig = px.index[mask.values]
    if len(trig) == 0:
        print(f"  {label or tkr}: NO TRIGGERS")
        return {}
    epi = declusters(trig, MIN_GAP, px.index)
    return {"tkr": tkr, "mask": mask, "trig": trig, "epi": epi, "beta": beta,
            "state": st, "label": label or tkr}


def legs_table(r: dict, h: int) -> None:
    """(1) price the legs SEPARATELY, then (2) the beta-neutral residual."""
    tkr, epi, beta = r["tkr"], r["epi"], r["beta"]
    f_i = fwd_lag(px[tkr], h, 1)
    f_s = fwd_lag(spy, h, 1)
    ok = f_i.notna() & f_s.notna()
    e = pd.DatetimeIndex([d for d in epi if ok.get(d, False)])
    if len(e) == 0:
        print("  (no valid episodes at this h)")
        return
    b = beta.loc[e].values
    intl = f_i.loc[e].values
    spyv = f_s.loc[e].values
    eqd = intl - spyv                 # equal-dollar spread
    resid = intl - b * spyv           # beta-neutral residual
    base = ok.values
    rows = [
        summarize(intl, f"LEG {tkr} long, episodes (N={len(e)})"),
        summarize(f_i[base].values, f"  CTRL {tkr} all days"),
        summarize(spyv, f"LEG SPY (same episodes)"),
        summarize(f_s[base].values, "  CTRL SPY all days"),
        summarize(eqd, "SPREAD equal-dollar (intl - SPY)"),
        summarize(resid, f"SPREAD beta-neutral (beta mean {b.mean():.2f})"),
        summarize((f_i - f_s)[base].values, "  CTRL equal-dollar all days"),
    ]
    show(rows, f"{r['label']}  h={h}  legs priced separately then netted")
    wins = int((resid > 0).sum())
    print(f"  resid record {wins}-{len(e)-wins}, sign p="
          f"{sign_test(wins, len(e)):.4f}   bootstrap P(mean<=0)="
          f"{bootstrap_p_le0(resid):.3f}")
    print(f"  concentration (resid): {cluster_note(e, resid)}")
    show(era_split(e, resid), f"  {r['label']} h={h} resid era split")
    show(era_split(e, eqd), f"  {r['label']} h={h} equal-dollar era split")


def horizon(r: dict) -> None:
    rows = []
    for h in range(1, 11):
        f_i = fwd_lag(px[r["tkr"]], h, 1)
        f_s = fwd_lag(spy, h, 1)
        ok = f_i.notna() & f_s.notna()
        e = pd.DatetimeIndex([d for d in r["epi"] if ok.get(d, False)])
        if len(e) == 0:
            continue
        b = r["beta"].loc[e].values
        resid = f_i.loc[e].values - b * f_s.loc[e].values
        eqd = f_i.loc[e].values - f_s.loc[e].values
        ctrl_resid = (f_i - r["beta"] * f_s)[ok].values
        row = summarize(resid, f"h={h} resid")
        row["eqdollar_pct"] = round(100 * eqd.mean(), 3)
        row["ctrl_resid_pct"] = round(100 * np.nanmean(ctrl_resid), 3)
        row["edge_pct"] = round(row["mean_pct"] - row["ctrl_resid_pct"], 3)
        row["long_leg_pct"] = round(100 * f_i.loc[e].values.mean(), 3)
        rows.append(row)
    show(rows, f"3. horizon scan 1..10, {r['label']} (episodes, lag=1)")


def regime(r: dict) -> None:
    trig = r["trig"]
    ok = spy_above200.notna()
    base = 100 * spy_above200[ok].mean()
    sel = 100 * spy_above200.loc[trig].mean()
    epi_sel = 100 * spy_above200.loc[r["epi"]].mean()
    print(f"6. regime over-selection, {r['label']}: SPY>200d on "
          f"{sel:.1f}% of {len(trig)} trigger days / {epi_sel:.1f}% of "
          f"{len(r['epi'])} episodes vs base rate {base:.1f}%")
    yrs = pd.Series(1, index=r["epi"]).groupby(r["epi"].year).size()
    print(f"   episodes by year: {dict(yrs)}")


def sensitivity(tkr: str, h: int) -> None:
    rows = []
    for dmax in (-0.002, -0.005, -0.010, -0.020):
        for lmin in (0.00, 0.02, 0.04):
            r = run(tkr, dmax, lmin, f"dist>={dmax:+.3f} lead>={lmin:+.2f}")
            if not r:
                continue
            f_i, f_s = fwd_lag(px[tkr], h, 1), fwd_lag(spy, h, 1)
            ok = f_i.notna() & f_s.notna()
            e = pd.DatetimeIndex([d for d in r["epi"] if ok.get(d, False)])
            if len(e) == 0:
                continue
            b = r["beta"].loc[e].values
            resid = f_i.loc[e].values - b * f_s.loc[e].values
            row = summarize(resid, r["label"])
            row["n_days"] = len(r["trig"])
            rows.append(row)
    show(rows, f"4. threshold sensitivity, {tkr} h={h} beta-neutral resid")


for tkr in ("EFA", "EWJ"):
    r = run(tkr)
    if not r:
        continue
    print(f"\n{'='*78}\n{tkr}: {len(r['trig'])} trigger days, "
          f"{len(r['epi'])} episodes (gap {MIN_GAP} td), "
          f"{r['epi'][0].date()} .. {r['epi'][-1].date()}")
    print("episodes:", ", ".join(str(d.date()) for d in r["epi"]))
    horizon(r)
    for h in (3, 5, 10):
        legs_table(r, h)
    regime(r)

sensitivity("EFA", 5)
sensitivity("EFA", 10)

print("\n7. cost: 2 liquid-ETF legs, ~5-6 bps round trip TOTAL for the pair.")
print("   an edge needs >= 5x that = ~+0.30% on the residual to clear.")
