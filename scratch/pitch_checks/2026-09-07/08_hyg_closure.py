"""08 rounds 1-2: adversarial falsification of "long HYG MOC at the first
close back after a long weekend, 5 sessions".

Candidate: HYG, entry MOC 2026-09-08 (k0 = first NYSE session after the
Labor Day closure), exit MOC 2026-09-15. No stop, no target.

Anchor convention is inherited verbatim from _closure_common (k0 = first
session after a >= 4 calendar-day NYSE gap; MOC(h) = Close[k0] -> Close[k0+h]).
Nothing here recomputes the anchor set.

The attacks, in the order the brief ranked them:
  A1  does the closure gate filter anything at all vs an ordinary weekend
      (and vs "any first session of a week")
  A2  residual alpha of the window return against IEF (duration) and SPY
      (equity beta) over the SAME window
  A3  scheduled prints inside the hold; the live instance carries BOTH
      PPI (+2) and CPI (+3)
  A4  is the Labor-Day slice a grid artifact
  A6  concentration / era: 2008-09, 2020, pre/post 2018
  A7  state conditioner: distance from the 252d high, trailing realised vol
  A8  cost
  A9  tail
  A10 book overlap + the live short-SPY event-sleeve leg inside the hold
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
    declusters, load_events, local_control, sign_test, summarize,
)

H = 5
TKR = "HYG"
COST_RT_BPS = 6.0   # _closure_common's own HYG assumption


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return (a.mean() - b.mean()) / se if se > 0 else np.nan


def line(r, extra=""):
    if not r.get("n"):
        return f"{r.get('label','')}: n=0"
    w = int(round(r["hit"] / 100 * r["n"]))
    return (f"{r['label']:<44s} n={r['n']:>4d} mean={r['mean_pct']:+.3f}% "
            f"med={r['median_pct']:+.3f}% hit={r['hit']:4.1f}% "
            f"({w}-{r['n']-w}) t={r['t']:+.2f} worst={r['worst_pct']:+.2f}%{extra}")


def main() -> None:
    px = C.load_panel()
    cal = C.nyse_calendar(px)
    ct = C.closure_table(cal)
    hol = pd.DatetimeIndex(ct.loc[ct["kind"] == "holiday", "anchor"])
    wkd = pd.DatetimeIndex(ct.loc[ct["kind"] == "weekend", "anchor"])

    hyg = px[TKR]
    moc = C.moc_series(hyg, H)
    a_hol = C.anchors_on(hyg, hol)
    a_wkd = C.anchors_on(hyg, wkd)
    v_hol, d_hol = C.cell(moc, a_hol)
    v_wkd, d_wkd = C.cell(moc, a_wkd)
    all_v = moc.dropna().values.astype(float)

    print("=" * 104)
    print(f"08  HYG post-closure MOC h={H}  |  entry Close[k0], exit Close[k0+{H}]")
    print(f"HYG bars {hyg.index[0].date()} .. {hyg.index[-1].date()}  "
          f"(n={len(hyg)} sessions)")
    print("=" * 104)

    # ---------------------------------------------------------------- A1
    print("\n" + "-" * 104)
    print("A1  DOES THE CLOSURE GATE FILTER ANYTHING?")
    print("-" * 104)
    rows = [summarize(v_hol, f"HOLIDAY closure gap>=4 (the cell)"),
            summarize(v_wkd, "CONTROL ordinary 3-day weekend"),
            summarize(all_v, "CONTROL all HYG days")]
    for r in rows:
        print("  " + line(r))
    exc = v_hol.mean() - v_wkd.mean()
    print(f"\n  excess of the extra calendar day = {100*exc:+.3f}%  "
          f"welch t = {welch(v_hol, v_wkd):+.2f}")
    print(f"  the weekend control ALONE is {100*v_wkd.mean():+.3f}% "
          f"({100*v_wkd.mean()/ (100*v_hol.mean()) * 100:.0f}% of the cell) "
          f"at t={summarize(v_wkd)['t']:+.2f} on n={len(v_wkd)}")

    # a strictly harder version of the same question: what does the FIRST
    # SESSION OF ANY WEEK pay, i.e. drop the closure idea entirely?
    prev_pos = pd.Series(range(len(cal)), index=cal)
    firstweek = pd.DatetimeIndex(
        [d for d in cal[1:]
         if d.isocalendar()[1] != cal[prev_pos[d] - 1].isocalendar()[1]])
    v_fw, _ = C.cell(moc, C.anchors_on(hyg, firstweek))
    print("  " + line(summarize(v_fw, "CONTROL first session of ANY week")))
    print(f"  holiday-vs-first-of-week excess = {100*(v_hol.mean()-v_fw.mean()):+.3f}%  "
          f"welch t = {welch(v_hol, v_fw):+.2f}")

    # and the honest reference class: is this just "HYG goes up"? paired
    # sign test of the cell against HYG's OWN all-days up-rate
    base_hit = float((all_v > 0).mean())
    w = int((v_hol > 0).sum())
    print(f"\n  cell record {w}-{len(v_hol)-w}; vs coin p={sign_test(w, len(v_hol)):.4f}; "
          f"vs HYG's OWN all-days up-rate {100*base_hit:.1f}% "
          f"p={sign_test(w, len(v_hol), base_hit):.4f}")
    ww = int((v_wkd > 0).sum())
    print(f"  weekend control record {ww}-{len(v_wkd)-ww} "
          f"(hit {100*ww/len(v_wkd):.1f}%) vs same base p="
          f"{sign_test(ww, len(v_wkd), base_hit):.4f}")

    # ---------------------------------------------------------------- A2
    print("\n" + "-" * 104)
    print("A2  RESIDUAL AGAINST IEF (duration) AND SPY (equity beta), SAME WINDOW")
    print("-" * 104)
    ief = C.moc_series(px["IEF"], H)
    spy = C.moc_series(px["SPY"], H)
    panel = pd.DataFrame({"hyg": moc, "ief": ief, "spy": spy}).dropna()
    dd = pd.DatetimeIndex(d_hol).intersection(panel.index)
    sub = panel.loc[dd]

    def ols(y, X, names):
        X1 = np.column_stack([np.ones(len(X))] + [X[:, i] for i in range(X.shape[1])])
        beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
        resid = y - X1 @ beta
        dof = len(y) - X1.shape[1]
        s2 = resid @ resid / dof
        cov = s2 * np.linalg.pinv(X1.T @ X1)
        se = np.sqrt(np.diag(cov))
        return beta, se, resid, ["alpha"] + names

    # betas fitted on FULL history (not on the 130 anchors) so the residual
    # is a genuine out-of-cell projection, then applied to the cell
    Xf = panel[["ief", "spy"]].values
    bf, sef, _, nm = ols(panel["hyg"].values, Xf, ["ief", "spy"])
    print(f"  full-history HYG = {100*bf[0]:+.3f}% + {bf[1]:+.3f}*IEF "
          f"+ {bf[2]:+.3f}*SPY   (n={len(panel)}, "
          f"SPY beta t={bf[2]/sef[2]:+.1f}, IEF beta t={bf[1]/sef[1]:+.1f})")
    resid_cell = (sub["hyg"].values - bf[0]
                  - bf[1] * sub["ief"].values - bf[2] * sub["spy"].values)
    rw = int((resid_cell > 0).sum())
    print("  " + line(summarize(resid_cell,
                                "RESIDUAL on the 130 anchors (full-hist betas)")))
    print(f"    residual record {rw}-{len(resid_cell)-rw}, sign p="
          f"{sign_test(rw, len(resid_cell)):.4f}")

    # what the raw legs did over the same anchors
    print("  " + line(summarize(sub["hyg"].values, "  raw HYG")))
    print("  " + line(summarize(sub["ief"].values, "  raw IEF same window")))
    print("  " + line(summarize(sub["spy"].values, "  raw SPY same window")))
    print(f"    beta-explained part = {100*(bf[1]*sub['ief'].mean() + bf[2]*sub['spy'].mean()):+.3f}%"
          f"   (IEF {100*bf[1]*sub['ief'].mean():+.3f}%, "
          f"SPY {100*bf[2]*sub['spy'].mean():+.3f}%)")
    print(f"    unconditional intercept alpha carried into the cell = {100*bf[0]:+.3f}%")

    # and against the ordinary-weekend control on the SAME residual basis
    dw = pd.DatetimeIndex(d_wkd).intersection(panel.index)
    subw = panel.loc[dw]
    resid_w = (subw["hyg"].values - bf[0]
               - bf[1] * subw["ief"].values - bf[2] * subw["spy"].values)
    print("  " + line(summarize(resid_w, "RESIDUAL on ordinary weekends")))
    print(f"    holiday-minus-weekend RESIDUAL excess = "
          f"{100*(resid_cell.mean()-resid_w.mean()):+.3f}%  "
          f"welch t = {welch(resid_cell, resid_w):+.2f}")

    # SPY-only, the "levered index long wearing a credit label" charge
    b1, se1, _, _ = ols(panel["hyg"].values, panel[["spy"]].values, ["spy"])
    r1 = sub["hyg"].values - b1[0] - b1[1] * sub["spy"].values
    print(f"  SPY-only beta = {b1[1]:.3f} (t={b1[1]/se1[1]:+.1f}); "
          f"cell residual mean {100*r1.mean():+.3f}% t={summarize(r1)['t']:+.2f}")

    # ---------------------------------------------------------------- A3
    print("\n" + "-" * 104)
    print("A3  SCHEDULED PRINTS INSIDE THE HOLD  (live instance = PPI +2, CPI +3)")
    print("-" * 104)
    ev = load_events(["nfp", "cpi", "ppi", "fomc_decision"])
    pos = pd.Series(range(len(cal)), index=cal)

    def prints_in_hold(anchor):
        p = pos.get(anchor)
        if p is None or p + H >= len(cal):
            return None
        lo, hi = cal[p], cal[p + H]
        m = (ev["date"] > lo) & (ev["date"] <= hi)
        return sorted(ev.loc[m, "event"].unique()), int(m.sum())

    kinds, counts = [], []
    keep = []
    for d in d_hol:
        r = prints_in_hold(d)
        if r is None:
            continue
        keep.append(d)
        kinds.append(set(r[0]))
        counts.append(r[1])
    keep = pd.DatetimeIndex(keep)
    vals = moc.loc[keep].values.astype(float)
    counts = np.asarray(counts)
    has_cpi = np.array(["cpi" in k for k in kinds])
    has_ppi = np.array(["ppi" in k for k in kinds])
    both = has_cpi & has_ppi
    neither = ~has_cpi & ~has_ppi

    for lbl, m in (("0 scheduled prints in hold", counts == 0),
                   ("1 print", counts == 1),
                   ("2 prints", counts == 2),
                   (">=3 prints", counts >= 3)):
        if m.sum():
            print("  " + line(summarize(vals[m], lbl)))
    print()
    print("  " + line(summarize(vals[both], "PPI *and* CPI in hold (LIVE CONFIG)")))
    print("  " + line(summarize(vals[has_cpi], "CPI in hold (any)")))
    print("  " + line(summarize(vals[neither], "neither PPI nor CPI in hold")))
    bw = int((vals[both] > 0).sum())
    print(f"    both-prints record {bw}-{int(both.sum())-bw}, sign p="
          f"{sign_test(bw, int(both.sum())):.4f}; "
          f"excess vs neither = {100*(vals[both].mean()-vals[neither].mean()):+.3f}%  "
          f"welch t={welch(vals[both], vals[neither]):+.2f}")
    print(f"    both-prints vs the ordinary-weekend control: "
          f"{100*(vals[both].mean()-v_wkd.mean()):+.3f}%  "
          f"welch t={welch(vals[both], v_wkd):+.2f}")
    print(f"    both-prints vs HYG all-days: "
          f"{100*(vals[both].mean()-all_v.mean()):+.3f}%")

    # ---------------------------------------------------------------- A4
    print("\n" + "-" * 104)
    print("A4  IS THE LABOR-DAY SLICE A GRID ARTIFACT?")
    print("-" * 104)
    ld_grid = pd.read_csv(HERE / "01_laborday_grid.csv")
    ps = ld_grid["sign_p_up"].dropna().values
    n_cells = len(ld_grid)
    print(f"  Labor-Day grid: {n_cells} cells; sub-0.10 sign_p_up hits = "
          f"{int((ps < 0.10).sum())}, expected under the null ~{0.10*len(ps):.1f}")
    print(f"  sub-0.05 hits = {int((ps < 0.05).sum())}, expected ~{0.05*len(ps):.1f}; "
          f"smallest p = {ps.min():.4f} vs E[min] ~ {1/(len(ps)+1):.4f}")
    ldm = ct["labor_day"].values
    ld_anchors = pd.DatetimeIndex(ct.loc[ct["labor_day"], "anchor"])
    v_ld, d_ld = C.cell(moc, C.anchors_on(hyg, ld_anchors))
    lw = int((v_ld > 0).sum())
    print("  " + line(summarize(v_ld, f"Labor-Day-only slice h={H} MOC")))
    print(f"    record {lw}-{len(v_ld)-lw}, sign p={sign_test(lw, len(v_ld)):.4f}; "
          f"vs base {100*base_hit:.1f}% p={sign_test(lw, len(v_ld), base_hit):.4f}")
    print("    -> read the 130-anchor cell, not this slice.")

    # ---------------------------------------------------------------- A6
    print("\n" + "-" * 104)
    print("A6  CONCENTRATION AND ERA")
    print("-" * 104)
    yr = pd.DatetimeIndex(d_hol).year
    crisis = np.isin(yr, [2008, 2009]) | (yr == 2020)
    print("  " + line(summarize(v_hol, "all anchors")))
    print("  " + line(summarize(v_hol[~crisis], "ex 2008-09 and 2020")))
    print("  " + line(summarize(v_hol[np.isin(yr, [2008, 2009])], "  2008-09 only")))
    print("  " + line(summarize(v_hol[yr == 2020], "  2020 only")))
    pre = yr < 2018
    print("  " + line(summarize(v_hol[pre], "pre-2018")))
    print("  " + line(summarize(v_hol[~pre], "2018+")))
    print("  " + line(summarize(v_hol[yr >= 2021], "2021+")))
    exc_nc = v_hol[~crisis].mean() - v_wkd[~np.isin(pd.DatetimeIndex(d_wkd).year,
                                                    [2008, 2009, 2020])].mean()
    print(f"  ex-crisis excess over ex-crisis weekend control = {100*exc_nc:+.3f}%")
    epi = declusters(pd.DatetimeIndex(d_hol), H, hyg.index)
    ve = moc.loc[epi].values.astype(float)
    print(f"  declustered episodes: n={len(epi)} mean={100*ve.mean():+.3f}%")
    tot = v_hol.sum()
    order = np.argsort(-np.abs(v_hol))[:5]
    print(f"  top-5 |moves| = {100*v_hol[order].sum():+.2f}pp of {100*tot:+.2f}pp "
          f"total ({100*v_hol[order].sum()/tot:.0f}%) on "
          f"{[str(pd.Timestamp(d_hol[i]).date()) for i in order]}")
    byyr = pd.Series(v_hol, index=yr).groupby(level=0).sum().sort_values()
    print(f"  worst years {dict((int(y), round(100*v,2)) for y, v in byyr.head(3).items())}"
          f"  best years {dict((int(y), round(100*v,2)) for y, v in byyr.tail(3).items())}")

    # ---------------------------------------------------------------- A7
    print("\n" + "-" * 104)
    print("A7  STATE CONDITIONER: is the LIVE bucket the strong half?")
    print("-" * 104)
    close = hyg["Close"]
    hi252 = close.rolling(252, min_periods=100).max()
    dist = (close / hi252 - 1.0) * 100.0          # % below the 252d high (<=0)
    rvol = close.pct_change().rolling(21).std() * np.sqrt(252) * 100.0
    st = pd.DataFrame({"dist": dist, "rvol": rvol}).loc[d_hol]
    live_dist, live_rvol = -0.41, 2.6
    print(f"  LIVE state: {live_dist:.2f}% below the 252d high, "
          f"21d realised vol {live_rvol:.1f}% annualised")
    near = (st["dist"] > -1.0).values          # within 1% of the 252d high
    print("  " + line(summarize(v_hol[near], "within 1.0% of the 252d high (LIVE)")))
    print("  " + line(summarize(v_hol[~near], "more than 1.0% below")))
    print(f"    near-vs-far diff = {100*(v_hol[near].mean()-v_hol[~near].mean()):+.3f}%  "
          f"welch t={welch(v_hol[near], v_hol[~near]):+.2f}")
    nw = int((v_hol[near] > 0).sum())
    print(f"    near-high record {nw}-{int(near.sum())-nw}, sign p="
          f"{sign_test(nw, int(near.sum())):.4f}; vs weekend control excess "
          f"{100*(v_hol[near].mean()-v_wkd.mean()):+.3f}% "
          f"welch t={welch(v_hol[near], v_wkd):+.2f}")
    calm = (st["rvol"] < np.nanpercentile(rvol.dropna(), 33)).values
    print("  " + line(summarize(v_hol[calm], "calmest realised-vol tercile (LIVE)")))
    print("  " + line(summarize(v_hol[~calm], "middle+high vol")))
    print(f"    calm-vs-rest diff = {100*(v_hol[calm].mean()-v_hol[~calm].mean()):+.3f}%  "
          f"welch t={welch(v_hol[calm], v_hol[~calm]):+.2f}")
    both_live = near & calm
    if both_live.sum():
        bl = int((v_hol[both_live] > 0).sum())
        print("  " + line(summarize(v_hol[both_live], "near-high AND calm (the LIVE cell)")))
        print(f"    record {bl}-{int(both_live.sum())-bl}, sign p="
              f"{sign_test(bl, int(both_live.sum())):.4f}")
        print(f"    dates: {[str(pd.Timestamp(d).date()) for d in pd.DatetimeIndex(d_hol)[both_live]]}")

    # ---------------------------------------------------------------- A8
    print("\n" + "-" * 104)
    print("A8  COST")
    print("-" * 104)
    edge_bps = 100 * v_hol.mean() * 100
    exc_bps = 100 * exc * 100
    print(f"  assumption: HYG round trip {COST_RT_BPS} bps (1 leg, both sides)")
    print(f"  raw cell edge   {edge_bps:5.1f} bps -> {edge_bps/COST_RT_BPS:.1f}x cost")
    print(f"  EXCESS over the ordinary weekend {exc_bps:5.1f} bps -> "
          f"{exc_bps/COST_RT_BPS:.1f}x cost   (pitch_lab.battery bar is >=5x)")
    resid_bps = 100 * resid_cell.mean() * 100
    print(f"  residual after IEF+SPY {resid_bps:5.1f} bps -> "
          f"{resid_bps/COST_RT_BPS:.1f}x cost")
    # cheaper same-exposure alternative
    print(f"  same exposure via SPY at matched beta {b1[1]:.2f}: SPY's own cell mean "
          f"{100*sub['spy'].mean():+.3f}% x {b1[1]:.2f} = "
          f"{100*b1[1]*sub['spy'].mean():+.3f}% at 4 bps round trip")

    # ---------------------------------------------------------------- A9
    print("\n" + "-" * 104)
    print("A9  TAIL (no stop proposed)")
    print("-" * 104)
    ordr = np.argsort(v_hol)
    print("  worst 5 post-closure windows:")
    for i in ordr[:5]:
        print(f"    {pd.Timestamp(d_hol[i]).date()}  {100*v_hol[i]:+.2f}%")
    post18 = pd.DatetimeIndex(d_hol).year >= 2018
    o18 = np.argsort(v_hol[post18])
    d18 = pd.DatetimeIndex(d_hol)[post18]
    print("  worst 3 since 2018:")
    for i in o18[:3]:
        print(f"    {d18[i].date()}  {100*v_hol[post18][i]:+.2f}%")
    atr_pct = 0.23
    print(f"  HYG Wilder-14 ATR today ~ {atr_pct}% of price; a 1-ATR stop is a "
          f"{atr_pct*100:.0f} bp move")
    print(f"  daily |return| median over full history = "
          f"{100*close.pct_change().abs().median():.3f}%  "
          f"(so 1 ATR is ~{atr_pct/(100*close.pct_change().abs().median()):.1f}x a median day)")
    print(f"  share of the 130 windows whose 5-session return is worse than "
          f"-{atr_pct:.2f}%: {100*float((v_hol < -atr_pct/100).mean()):.1f}%")

    # ---------------------------------------------------------------- A10
    print("\n" + "-" * 104)
    print("A10 BOOK OVERLAP + THE LIVE SHORT-SPY EVENT LEG (T2, MOC 09-10 -> MOO 09-16)")
    print("-" * 104)
    led = ROOT / "data" / "backtest_trades_full.parquet"
    if led.exists():
        tr = pd.read_parquet(led)
        col = "Ticker" if "Ticker" in tr.columns else tr.columns[0]
        hy = tr[tr[col].astype(str).str.upper() == "HYG"]
        print(f"  ledger rows {len(tr)}; HYG rows = {len(hy)}")
        if len(hy):
            print(hy.groupby(["Strategy_Name", "Direction"]).size().to_string()
                  if "Strategy_Name" in hy.columns else hy.head().to_string())
        credit = tr[tr[col].astype(str).str.upper().isin(["LQD", "JNK", "HYG", "TLT", "IEF"])]
        print(f"  ledger rows in credit/duration proxies = {len(credit)}: "
              f"{credit[col].value_counts().to_dict() if len(credit) else '{}'}")
    else:
        print("  (no ledger parquet)")

    # correlation of long-HYG vs short-SPY over the overlapping window
    ov = pd.DataFrame({"hyg": moc, "spy": spy}).dropna()
    sub2 = ov.loc[pd.DatetimeIndex(d_hol).intersection(ov.index)]
    rho = np.corrcoef(sub2["hyg"], sub2["spy"])[0, 1]
    rho_all = np.corrcoef(ov["hyg"], ov["spy"])[0, 1]
    print(f"\n  corr(HYG 5d, SPY 5d) on the 130 anchors = {rho:+.3f}; "
          f"full history = {rho_all:+.3f}")
    print(f"  T2 is SHORT SPY 10% NAV; this idea is LONG HYG with a SPY beta of "
          f"{b1[1]:.2f}. Sign of the pair: "
          f"{'HEDGE (offsetting)' if rho > 0 else 'ADDITIVE'} — a long-credit leg "
          f"and a short-SPY leg move opposite ways when the tape moves, so the "
          f"book's realised sum is close to a wash on the shared factor.")
    print(f"  at 25% NAV HYG (beta {b1[1]:.2f}) the implied long-SPY-equivalent is "
          f"{25*b1[1]:.1f}% NAV vs T2's -10% NAV, i.e. the pair is net "
          f"{25*b1[1]-10:+.1f}% NAV long equity beta over 09-10..09-15.")

    print("\n" + "=" * 104)
    print("done")


if __name__ == "__main__":
    main()
