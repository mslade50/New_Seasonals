"""08 round 3: develop-or-bury the HYG post-closure MOC h=5 candidate.

D1  horizon table h=1..10 via pitch_lab.horizon_scan, MOO vs MOC as WHOLE
    variants (not marginal fills)
D2  the state-matched control: near-52w-high AND calm is the LIVE bucket, so
    compare it against ordinary weekends IN THE SAME STATE, which is the only
    fair way to ask whether the closure gate still adds anything today
D3  residual against IEF+SPY inside the live bucket and inside the
    both-prints (PPI+CPI) bucket
D4  mechanism: day-by-day decomposition of the hold, plus the SKIPPED first
    session (Close[k0-1] -> Close[k0]). If carry/accrual makes the effect it
    belongs in the first session back.
D5  exit sensitivity h=3..7 in the live bucket
D6  pitch_lab.episode_paths on the losing episodes
D7  SPY head-to-head on the identical anchors, net of cost
D8  the honest live cell: the intersection of D2's dead state bucket
    and D3's one surviving print bucket

Anchoring for pitch_lab: anchor on PREV (the last session before the gap), so
lag=1 enters at Close[k0] = the pitched MOC entry and exits at Close[k0+h].
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import _closure_common as C  # noqa: E402
from pitch_lab import (  # noqa: E402
    episode_paths, horizon_scan, load_events, show, sign_test, summarize,
)

TKR = "HYG"
COST = 6.0


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return (a.mean() - b.mean()) / se if se > 0 else np.nan


def line(r):
    if not r.get("n"):
        return f"{r.get('label','')}: n=0"
    w = int(round(r["hit"] / 100 * r["n"]))
    return (f"{r['label']:<46s} n={r['n']:>4d} mean={r['mean_pct']:+.3f}% "
            f"med={r['median_pct']:+.3f}% hit={r['hit']:4.1f}% ({w}-{r['n']-w}) "
            f"t={r['t']:+.2f} worst={r['worst_pct']:+.2f}%")


def ols_resid(y, X):
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return beta, y - X1 @ beta


def main() -> None:
    px = C.load_panel()
    cal = C.nyse_calendar(px)
    ct = C.closure_table(cal)
    hol = ct[ct["kind"] == "holiday"]
    wknd = ct[ct["kind"] == "weekend"]
    hyg = px[TKR]

    # close panel for pitch_lab (HYG-only legs, HYG's own calendar)
    cp = pd.DataFrame({"HYG": hyg["Close"], "SPY": px["SPY"]["Close"],
                       "IEF": px["IEF"]["Close"]}).dropna()
    prev_hol = pd.DatetimeIndex(hol["prev"]).intersection(cp.index)
    prev_wkd = pd.DatetimeIndex(wknd["prev"]).intersection(cp.index)
    anch_hol = C.anchors_on(hyg, pd.DatetimeIndex(hol["anchor"]))

    print("=" * 104)
    print("08-DEV  HYG post-closure, round 3")
    print(f"anchors: {len(prev_hol)} pre-closure sessions (lag=1 -> entry MOC "
          f"on the first session back)")
    print("=" * 104)

    # ------------------------------------------------------------------ D1
    print("\n" + "-" * 104)
    print("D1  HORIZON TABLE — where does the shelf actually sit?")
    print("-" * 104)
    print("  MOC entry (pitch_lab.horizon_scan, anchor=prev session, lag=1):")
    rows = horizon_scan(cp, prev_hol, [("HYG", 1.0)],
                        hs=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10), lag=1)
    show(rows, "MOC h=1..10, episode level, edge vs HYG all-days")

    moo_rows = []
    for h in range(1, 11):
        ser = C.moo_series(hyg, h)
        v, _ = C.cell(ser, anch_hol)
        base = ser.dropna().values.astype(float)
        r = summarize(v, f"h={h}")
        r["ctl_all_days_pct"] = round(100 * base.mean(), 3)
        r["edge_pct"] = round(r["mean_pct"] - 100 * base.mean(), 3)
        moo_rows.append(r)
    show(moo_rows, "MOO h=1..10 (Open[k0] -> Close[k0+h-1]) — WHOLE variant")
    print("  NOTE the MOO column: the FIRST SESSION BACK is where the cell is")
    print("  weakest, which is why the candidate skips it. See D4.")

    # ------------------------------------------------------------------ D2
    print("\n" + "-" * 104)
    print("D2  STATE-MATCHED CONTROL — the only fair test of the live instance")
    print("-" * 104)
    close = hyg["Close"]
    dist = (close / close.rolling(252, min_periods=100).max() - 1.0) * 100.0
    rvol = close.pct_change().rolling(21).std() * np.sqrt(252) * 100.0
    calm_cut = float(np.nanpercentile(rvol.dropna(), 33))
    live = ((dist > -1.0) & (rvol < calm_cut))
    print(f"  live bucket definition: within 1.0% of the 252d high AND 21d "
          f"realised vol < {calm_cut:.1f}% (its 33rd pctile). "
          f"Today: -0.41%, 2.6%.")

    for h in (5,):
        moc = C.moc_series(hyg, h)
        anch_w = C.anchors_on(hyg, pd.DatetimeIndex(wknd["anchor"]))
        lv = live.reindex(hyg.index, fill_value=False)
        v_h, _ = C.cell(moc, anch_hol[lv.loc[anch_hol].values])
        v_w, _ = C.cell(moc, anch_w[lv.loc[anch_w].values])
        allv = moc[lv.reindex(moc.index, fill_value=False).values].dropna().values
        print(f"\n  h={h}:")
        print("   " + line(summarize(v_h, "HOLIDAY closure, live state")))
        print("   " + line(summarize(v_w, "ordinary WEEKEND, live state")))
        print("   " + line(summarize(allv, "ALL days, live state")))
        print(f"   closure excess IN THE LIVE STATE = "
              f"{100*(v_h.mean()-v_w.mean()):+.3f}%  welch t={welch(v_h, v_w):+.2f}")
        print(f"   vs all-days-in-state           = "
              f"{100*(v_h.mean()-allv.mean()):+.3f}%")
        print(f"   cost multiple of the live-state cell: "
              f"{100*100*v_h.mean()/COST:.1f}x  (bar is 5x)")

        # the contrast bucket, so the reader can see where the effect lives
        v_h2, _ = C.cell(moc, anch_hol[~lv.loc[anch_hol].values])
        v_w2, _ = C.cell(moc, anch_w[~lv.loc[anch_w].values])
        print("   " + line(summarize(v_h2, "HOLIDAY closure, NOT the live state")))
        print("   " + line(summarize(v_w2, "ordinary WEEKEND, NOT live state")))
        print(f"   closure excess OUTSIDE the live state = "
              f"{100*(v_h2.mean()-v_w2.mean()):+.3f}%  welch t={welch(v_h2, v_w2):+.2f}")

    # ------------------------------------------------------------------ D3
    print("\n" + "-" * 104)
    print("D3  RESIDUAL AGAINST IEF + SPY INSIDE THE LIVE BUCKET AND THE PRINT BUCKET")
    print("-" * 104)
    h = 5
    moc = C.moc_series(hyg, h)
    panel = pd.DataFrame({"hyg": moc,
                          "ief": C.moc_series(px["IEF"], h),
                          "spy": C.moc_series(px["SPY"], h)}).dropna()
    beta, _ = ols_resid(panel["hyg"].values, panel[["ief", "spy"]].values)
    print(f"  full-history betas: alpha {100*beta[0]:+.3f}%  IEF {beta[1]:+.3f}  "
          f"SPY {beta[2]:+.3f}")

    def resid_on(dates, label):
        d = pd.DatetimeIndex(dates).intersection(panel.index)
        s = panel.loc[d]
        r = (s["hyg"].values - beta[0] - beta[1] * s["ief"].values
             - beta[2] * s["spy"].values)
        w = int((r > 0).sum())
        print("  " + line(summarize(r, label))
              + f"  signp={sign_test(w, len(r)):.3f}")
        return r

    lv = live.reindex(hyg.index, fill_value=False)
    resid_on(anch_hol, "all 130 anchors")
    resid_on(anch_hol[lv.loc[anch_hol].values], "LIVE bucket (near-high + calm)")
    resid_on(anch_hol[~lv.loc[anch_hol].values], "NOT the live bucket")

    ev = load_events(["cpi", "ppi"])
    pos = pd.Series(range(len(cal)), index=cal)
    both = []
    for d in anch_hol:
        p = pos.get(d)
        if p is None or p + h >= len(cal):
            continue
        lo, hi = cal[p], cal[p + h]
        k = set(ev.loc[(ev["date"] > lo) & (ev["date"] <= hi), "event"])
        if {"cpi", "ppi"} <= k:
            both.append(d)
    resid_on(pd.DatetimeIndex(both), f"PPI+CPI in hold (live config, n={len(both)})")

    # ------------------------------------------------------------------ D4
    print("\n" + "-" * 104)
    print("D4  MECHANISM — where in the window does the money arrive?")
    print("-" * 104)
    print("  The story on offer: a closure costs credit 3 calendar days of")
    print("  accrual, made up in price when desks come back. If true the")
    print("  effect belongs in the FIRST session back.")
    idx = hyg.index
    p = pd.Series(range(len(idx)), index=idx)
    c = hyg["Close"].values
    o = hyg["Open"].values
    legs = {"gap Open[k0]/Close[k0-1]": [], "session k0 O->C": [],
            "session k0 C[k0-1]->C[k0] (SKIPPED)": []}
    daily = {i: [] for i in range(1, 6)}
    for d in anch_hol:
        k = p.get(d)
        if k is None or k + 5 >= len(idx) or k < 1:
            continue
        legs["gap Open[k0]/Close[k0-1]"].append(o[k] / c[k - 1] - 1)
        legs["session k0 O->C"].append(c[k] / o[k] - 1)
        legs["session k0 C[k0-1]->C[k0] (SKIPPED)"].append(c[k] / c[k - 1] - 1)
        for i in range(1, 6):
            daily[i].append(c[k + i] / c[k + i - 1] - 1)
    allday = hyg["Close"].pct_change().dropna().values
    print(f"\n  HYG unconditional single session = {100*allday.mean():+.4f}%")
    for lbl, v in legs.items():
        v = np.asarray(v)
        print("  " + line(summarize(v, lbl))
              + f"  edge={100*(v.mean()-allday.mean()):+.4f}%")
    print("\n  inside the pitched hold, session by session (entry = Close[k0]):")
    for i, v in daily.items():
        v = np.asarray(v)
        print("  " + line(summarize(v, f"  hold day {i} (k0+{i})"))
              + f"  edge={100*(v.mean()-allday.mean()):+.4f}%")
    cum = np.cumsum([np.mean(daily[i]) for i in range(1, 6)])
    print(f"  cumulative mean by hold day: "
          f"{[f'{100*x:+.3f}%' for x in cum]}")

    # ------------------------------------------------------------------ D5
    print("\n" + "-" * 104)
    print("D5  EXIT SENSITIVITY IN THE LIVE BUCKET")
    print("-" * 104)
    for hh in (2, 3, 4, 5, 6, 7, 10):
        s = C.moc_series(hyg, hh)
        v, _ = C.cell(s, anch_hol[lv.loc[anch_hol].values])
        base = s[lv.reindex(s.index, fill_value=False).values].dropna().values
        r = summarize(v, f"h={hh} live bucket")
        print("  " + line(r)
              + f"  vs all-days-in-state {100*base.mean():+.3f}% -> edge "
                f"{r['mean_pct']-100*base.mean():+.3f}%")

    # ------------------------------------------------------------------ D6
    print("\n" + "-" * 104)
    print("D6  LOSING EPISODES — what the failure actually looks like")
    print("-" * 104)
    paths = episode_paths(cp, prev_hol, [("HYG", 1.0)], 5, lag=1)
    fin = paths[5]
    losers = paths.loc[fin < 0].sort_values(5)
    print(f"  {len(losers)} of {len(paths)} windows finished negative "
          f"({100*len(losers)/len(paths):.0f}%)")
    print("  worst 6 paths (cumulative % from the entry close):")
    print((100 * losers.head(6)).round(2).to_string())
    print(f"\n  median trough across ALL windows = "
          f"{100*paths.min(axis=1).median():+.2f}%")
    print(f"  median trough across LOSING windows = "
          f"{100*losers.min(axis=1).median():+.2f}%")
    print(f"  5th pctile trough (all windows)  = "
          f"{100*np.percentile(paths.min(axis=1), 5):+.2f}%")
    print(f"  windows that were red at day 1 and finished green: "
          f"{int(((paths[1] < 0) & (fin > 0)).sum())} of "
          f"{int((paths[1] < 0).sum())} red-at-day-1")

    # ------------------------------------------------------------------ D7
    print("\n" + "-" * 104)
    print("D7  SPY HEAD-TO-HEAD ON THE IDENTICAL ANCHORS, NET OF COST")
    print("-" * 104)
    for tkr, cost in (("HYG", 6.0), ("SPY", 4.0), ("IEF", 6.0), ("LQD", 6.0)):
        s = C.moc_series(px[tkr], 5)
        v, _ = C.cell(s, C.anchors_on(px[tkr], pd.DatetimeIndex(hol["anchor"])))
        base = s.dropna().values.astype(float)
        r = summarize(v, tkr)
        net = r["mean_pct"] * 100 - cost
        print(f"  {tkr:<4s} n={r['n']:>3d} mean={r['mean_pct']:+.3f}% "
              f"t={r['t']:+.2f} hit={r['hit']:.1f}%  all-days "
              f"{100*base.mean():+.3f}%  edge {r['mean_pct']-100*base.mean():+.3f}%"
              f"  cost {cost} bps -> net {net:+.1f} bps "
              f"({r['mean_pct']*100/cost:.1f}x)")

    print("\n" + "=" * 104)
    print("done")




def d8_intersection() -> None:
    """D8: the live instance is BOTH near-high/calm AND carries PPI+CPI in the
    hold. D2 says the first bucket is dead and D3 says the second is the one
    surviving pocket, so the honest live cell is their intersection."""
    px = C.load_panel()
    cal = C.nyse_calendar(px)
    ct = C.closure_table(cal)
    hyg = px[TKR]
    h = 5
    moc = C.moc_series(hyg, h)
    anch = C.anchors_on(hyg, pd.DatetimeIndex(ct.loc[ct["kind"] == "holiday", "anchor"]))
    close = hyg["Close"]
    dist = (close / close.rolling(252, min_periods=100).max() - 1.0) * 100.0
    rvol = close.pct_change().rolling(21).std() * np.sqrt(252) * 100.0
    live = ((dist > -1.0) & (rvol < float(np.nanpercentile(rvol.dropna(), 33)))
            ).reindex(hyg.index, fill_value=False)

    ev = load_events(["cpi", "ppi"])
    pos = pd.Series(range(len(cal)), index=cal)
    both = []
    for d in anch:
        p = pos.get(d)
        if p is None or p + h >= len(cal):
            continue
        k = set(ev.loc[(ev["date"] > cal[p]) & (ev["date"] <= cal[p + h]), "event"])
        if {"cpi", "ppi"} <= k:
            both.append(d)
    both = pd.DatetimeIndex(both)

    print("\n" + "-" * 104)
    print("D8  THE HONEST LIVE CELL: near-52w-high + calm AND PPI+CPI in the hold")
    print("-" * 104)
    print(f"  PPI+CPI-in-hold anchors: {len(both)}  "
          f"{[str(d.date()) for d in both]}")
    print(f"  years: {sorted(both.year.tolist())}")
    v_both, _ = C.cell(moc, both)
    print("  " + line(summarize(v_both, "PPI+CPI in hold (all states)")))
    inter = both[live.loc[both].values]
    print(f"\n  intersection with the live state: n={len(inter)}  "
          f"{[str(d.date()) for d in inter]}")
    if len(inter):
        v_i, _ = C.cell(moc, inter)
        w = int((v_i > 0).sum())
        print("  " + line(summarize(v_i, "LIVE CELL (state + print config)"))
              + f"  signp={sign_test(w, len(v_i)):.3f}")
    v_out = C.cell(moc, both[~live.loc[both].values])[0]
    print("  " + line(summarize(v_out, "PPI+CPI in hold, NOT the live state")))
    pre = both.year < 2018
    print("  " + line(summarize(C.cell(moc, both[pre])[0], "  PPI+CPI pre-2018")))
    print("  " + line(summarize(C.cell(moc, both[~pre])[0], "  PPI+CPI 2018+")))


if __name__ == "__main__":
    main()
    d8_intersection()
