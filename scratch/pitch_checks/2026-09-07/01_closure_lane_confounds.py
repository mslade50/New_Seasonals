"""The attack on whatever showed a pulse in scripts 1-4.

Five confounds, each of which can manufacture the exact pattern the drift
grid printed:

 A. TURN OF MONTH / DIVIDEND ACCRUAL. master_prices is ADJUSTED, so a window
    containing an ex-div day books the distribution as price return. LQD/HYG
    pay MONTHLY (~0.30%/mo of income). Holiday closures cluster on month
    boundaries (Jan 1, Jul 4, Sep 1, Memorial Day) far more than random
    weekends do. If the bond/credit "post-closure bid" only lives in windows
    that straddle a month turn, it is coupon accrual plus the turn-of-month
    effect, not a closure effect.
 B. OVERLAP. holiday anchors are not independent at h=10 (Dec 26 and Jan 2
    are ~4 sessions apart). Decluster with pitch_lab.declusters and re-read.
 C. SVXY BASIS BREAK. SVXY was -1.0x inverse VIX until 2018-02-05 and -0.5x
    after; it also lost ~90% on 2018-02-05 itself. Pre-2018 SVXY numbers are
    a different instrument and are not poolable with post-2018 ones.
 D. THE VIX OPEN. ^VIX's 9:30 print comes off wide opening SPX option quotes
    and is systematically inflated, which mechanically makes every MOO-entry
    long-vol cell look bad and every MOO-entry short-vol cell look good.
 E. GAP MOMENTUM. If the post-closure drift is just a continuation of the
    overnight gap, it is not a separate edge and a MOO/MOC entry is buying
    after the news.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _closure_common as C
from pitch_lab import declusters, summarize, sign_test


def welch_t(a, b):
    if len(a) < 2 or len(b) < 2:
        return np.nan
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return (a.mean() - b.mean()) / se if se > 0 else np.nan


def month_turn_flags(idx: pd.DatetimeIndex, dates, lo: int, hi: int) -> pd.Series:
    """True when [p+lo, p+hi] contains a session that starts a new month."""
    idx = pd.DatetimeIndex(idx)
    is_first = np.zeros(len(idx), dtype=bool)
    is_first[1:] = idx[1:].month.values != idx[:-1].month.values
    pos = pd.Series(range(len(idx)), index=idx)
    out = {}
    for d in pd.DatetimeIndex(dates):
        p = pos.get(d)
        if p is None:
            continue
        a, b = max(0, p + lo), min(len(idx), p + hi + 1)
        out[d] = bool(is_first[a:b].any())
    return pd.Series(out)


def section_a(px, hol, wkd) -> None:
    print("\n" + "=" * 104)
    print("A. TURN OF MONTH / COUPON ACCRUAL  --  the bond & credit block's headline cells")
    print("split the holiday cell AND the weekend control by whether the HOLD window")
    print("contains a month's first session. MOC(h) window = [k0+1, k0+h].")
    print("=" * 104)
    for tkr, h in (("LQD", 3), ("LQD", 5), ("IEF", 3), ("IEF", 5),
                   ("HYG", 2), ("HYG", 5), ("TLT", 3), ("SPY", 1), ("GLD", 5)):
        df = px[tkr]
        ser = C.moc_series(df, h)
        a_h = C.anchors_on(df, hol)
        a_w = C.anchors_on(df, wkd)
        fh = month_turn_flags(df.index, a_h, 1, h)
        fw = month_turn_flags(df.index, a_w, 1, h)
        print(f"\n  {tkr} MOC h={h}   month-turn rate: holiday "
              f"{100*fh.mean():.1f}%  weekend {100*fw.mean():.1f}%")
        rows = []
        for lbl, flag in (("month-turn IN window", True), ("no month turn", False)):
            dh = pd.DatetimeIndex(fh[fh == flag].index)
            dw = pd.DatetimeIndex(fw[fw == flag].index)
            vh, _ = C.cell(ser, dh)
            vw, _ = C.cell(ser, dw)
            rh, rw = summarize(vh), summarize(vw)
            rows.append({
                "subset": lbl, "n_hol": rh.get("n", 0),
                "hol_mean": rh.get("mean_pct", np.nan),
                "hol_hit": rh.get("hit", np.nan), "hol_t": rh.get("t", np.nan),
                "n_wknd": rw.get("n", 0), "wknd_mean": rw.get("mean_pct", np.nan),
                "excess": rh.get("mean_pct", np.nan) - rw.get("mean_pct", np.nan),
                "t_excess": welch_t(vh, vw),
            })
        d = pd.DataFrame(rows)
        for c in d.columns:
            if d[c].dtype.kind == "f":
                d[c] = d[c].round(3)
        print(d.to_string(index=False))


def section_b(px, hol) -> None:
    print("\n" + "=" * 104)
    print("B. OVERLAP  --  holiday anchors declustered to min_gap = h sessions")
    print("=" * 104)
    gaps = []
    cal = C.nyse_calendar(px)
    pos = pd.Series(range(len(cal)), index=cal)
    p = [pos[d] for d in hol if d in pos.index]
    gaps = np.diff(sorted(p))
    print(f"  consecutive holiday anchors, session gap: min {gaps.min()}, "
          f"5th pct {np.percentile(gaps, 5):.0f}, median {np.median(gaps):.0f}; "
          f"pairs closer than 10 sessions: {(gaps < 10).sum()} of {len(gaps)}")
    rows = []
    for tkr in ("LQD", "IEF", "HYG", "TLT", "GLD", "SPY", "SVXY"):
        df = px[tkr]
        a_h = C.anchors_on(df, hol)
        for h in (3, 5, 10):
            for form, ser in (("MOO", C.moo_series(df, h)),
                              ("MOC", C.moc_series(df, h))):
                v_all, d_all = C.cell(ser, a_h)
                epi = declusters(d_all, h, pd.DatetimeIndex(df.index))
                v_ep, _ = C.cell(ser, epi)
                ra, re = summarize(v_all), summarize(v_ep)
                rows.append({"proxy": tkr, "h": h, "form": form,
                             "n_all": ra.get("n", 0),
                             "mean_all": ra.get("mean_pct", np.nan),
                             "t_all": ra.get("t", np.nan),
                             "n_epi": re.get("n", 0),
                             "mean_epi": re.get("mean_pct", np.nan),
                             "t_epi": re.get("t", np.nan)})
    d = pd.DataFrame(rows)
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].round(3)
    print(d.to_string(index=False))


def section_c(px, hol, wkd) -> None:
    print("\n" + "=" * 104)
    print("C. SVXY BASIS BREAK  --  -1.0x inverse VIX before 2018-02-05, -0.5x after")
    print("=" * 104)
    df = px["SVXY"]
    a_h = C.anchors_on(df, hol)
    a_w = C.anchors_on(df, wkd)
    cut = pd.Timestamp("2018-02-06")
    rows = []
    for h in (2, 3, 5, 10):
        for form, ser in (("MOO", C.moo_series(df, h)),
                          ("MOC", C.moc_series(df, h))):
            v, d = C.cell(ser, a_h)
            vw, dw = C.cell(ser, a_w)
            pre, post = summarize(v[d < cut]), summarize(v[d >= cut])
            wpost = summarize(vw[dw >= cut])
            rows.append({
                "h": h, "form": form,
                "n_pre": pre.get("n", 0), "pre_mean": pre.get("mean_pct", np.nan),
                "pre_hit": pre.get("hit", np.nan),
                "n_post": post.get("n", 0), "post_mean": post.get("mean_pct", np.nan),
                "post_hit": post.get("hit", np.nan), "post_t": post.get("t", np.nan),
                "post_signp": sign_test(int((v[d >= cut] > 0).sum()),
                                        int((d >= cut).sum())),
                "wknd_post_mean": wpost.get("mean_pct", np.nan),
                "post_excess": post.get("mean_pct", np.nan) - wpost.get("mean_pct", np.nan),
            })
    dd = pd.DataFrame(rows)
    for c in dd.columns:
        if dd[c].dtype.kind == "f":
            dd[c] = dd[c].round(3)
    print(dd.to_string(index=False))
    v, d = C.cell(C.moc_series(df, 5), a_h)
    print("\n  SVXY holiday MOC h=5, every observation:")
    print("  " + "  ".join(f"{x.date()}:{100*y:+.1f}" for x, y in zip(d, v)))


def section_d(px, hol, wkd) -> None:
    print("\n" + "=" * 104)
    print("D. THE ^VIX OPEN  --  is the post-closure vol collapse an opening-print artifact?")
    print("=" * 104)
    df = px["^VIX"]
    a_h = C.anchors_on(df, hol)
    a_w = C.anchors_on(df, wkd)
    gap = C.gap_series(df)
    o2c = df["Close"] / df["Open"] - 1.0
    c2c = df["Close"] / df["Close"].shift(1) - 1.0
    rows = []
    for lbl, dts in (("holiday k0", a_h), ("weekend k0", a_w)):
        for nm, s in (("gap C[-1]->O[0]", gap), ("O[0]->C[0]", o2c),
                      ("C[-1]->C[0] (net)", c2c)):
            v, _ = C.cell(s, dts)
            r = summarize(v)
            rows.append({"cell": lbl, "leg": nm, "n": r["n"],
                         "mean_pct": r["mean_pct"], "med_pct": r["median_pct"],
                         "hit": r["hit"], "t": r["t"]})
    d = pd.DataFrame(rows)
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].round(3)
    print(d.to_string(index=False))
    print("\n  If gap and O->C are near mirror images, the VIX 'move' is the opening")
    print("  print, not the tape. Net C->C is the only honest state read.")


def section_e(px, hol, wkd) -> None:
    print("\n" + "=" * 104)
    print("E. GAP MOMENTUM  --  is the drift just the overnight gap continuing?")
    print("=" * 104)
    rows = []
    for tkr, h, form in (("GLD", 5, "MOO"), ("GLD", 3, "MOC"), ("LQD", 3, "MOC"),
                         ("IEF", 5, "MOC"), ("SPY", 1, "MOC"), ("TLT", 3, "MOC")):
        df = px[tkr]
        ser = C.moo_series(df, h) if form == "MOO" else C.moc_series(df, h)
        g = C.gap_series(df)
        a_h = C.anchors_on(df, hol)
        v, d = C.cell(ser, a_h)
        gv = g.reindex(d).values
        ok = ~np.isnan(gv)
        v, d, gv = v[ok], d[ok], gv[ok]
        up, dn = summarize(v[gv > 0]), summarize(v[gv <= 0])
        corr = float(np.corrcoef(gv, v)[0, 1]) if len(v) > 2 else np.nan
        rows.append({"proxy": tkr, "h": h, "form": form, "n": len(v),
                     "corr_gap_vs_fwd": corr,
                     "n_gapup": up.get("n", 0), "gapup_mean": up.get("mean_pct", np.nan),
                     "n_gapdn": dn.get("n", 0), "gapdn_mean": dn.get("mean_pct", np.nan)})
    d = pd.DataFrame(rows)
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].round(3)
    print(d.to_string(index=False))


def section_f(px, ct) -> None:
    print("\n" + "=" * 104)
    print("F. WHICH HOLIDAY?  --  is the post-closure bid one calendar slot wearing a costume?")
    print("=" * 104)
    hol = ct[ct["kind"] == "holiday"].copy()
    hol["slot"] = hol["anchor"].dt.month.map({
        1: "Jan (NewYear/MLK)", 2: "Feb (Presidents)", 3: "Mar", 4: "Apr (GoodFri)",
        5: "May", 6: "Jun (Memorial/Juneteenth)", 7: "Jul (July4)",
        9: "Sep (Labor)", 10: "Oct", 11: "Nov", 12: "Dec (Christmas)"})
    for tkr, h in (("LQD", 3), ("IEF", 5), ("GLD", 5), ("SPY", 1)):
        df = px[tkr]
        ser = C.moo_series(df, h) if tkr == "GLD" else C.moc_series(df, h)
        form = "MOO" if tkr == "GLD" else "MOC"
        rows = []
        for slot, g in hol.groupby("slot"):
            v, _ = C.cell(ser, C.anchors_on(df, pd.DatetimeIndex(g["anchor"])))
            r = summarize(v)
            if r.get("n"):
                rows.append({"slot": slot, "n": r["n"], "mean_pct": r["mean_pct"],
                             "hit": r["hit"], "t": r["t"]})
        d = pd.DataFrame(rows).sort_values("mean_pct", ascending=False)
        for c in d.columns:
            if d[c].dtype.kind == "f":
                d[c] = d[c].round(3)
        print(f"\n  {tkr} {form} h={h} by holiday slot:")
        print(d.to_string(index=False))

    print("\n  era split on the pulse cells (episode-free, day level):")
    rows = []
    a = pd.DatetimeIndex(hol["anchor"])
    for tkr, h, form in (("LQD", 3, "MOC"), ("IEF", 5, "MOC"), ("HYG", 5, "MOC"),
                         ("TLT", 3, "MOC"), ("GLD", 5, "MOO"), ("SPY", 1, "MOC")):
        df = px[tkr]
        ser = C.moo_series(df, h) if form == "MOO" else C.moc_series(df, h)
        v, d = C.cell(ser, C.anchors_on(df, a))
        for lbl, m in (("pre-2013", d < pd.Timestamp("2013-01-01")),
                       ("2013+", d >= pd.Timestamp("2013-01-01"))):
            r = summarize(v[m])
            rows.append({"proxy": tkr, "h": h, "form": form, "era": lbl,
                         "n": r.get("n", 0), "mean_pct": r.get("mean_pct", np.nan),
                         "hit": r.get("hit", np.nan), "t": r.get("t", np.nan)})
    d = pd.DataFrame(rows)
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].round(3)
    print(d.to_string(index=False))


PULSE = [("LQD", 3, "MOC"), ("IEF", 3, "MOC"), ("IEF", 5, "MOC"),
         ("HYG", 5, "MOC"), ("HYG", 2, "MOC"), ("TLT", 3, "MOC"),
         ("GLD", 5, "MOO"), ("SPY", 1, "MOC"), ("SVXY", 3, "MOC")]


def _ser(df, h, form):
    return C.moo_series(df, h) if form == "MOO" else C.moc_series(df, h)


def section_g(px, ct) -> None:
    print("\n" + "=" * 104)
    print("G. HOW MANY INDEPENDENT BETS?  --  the bond/credit block is one duration trade")
    print("=" * 104)
    hol = pd.DatetimeIndex(ct.loc[ct["kind"] == "holiday", "anchor"])
    ld = pd.DatetimeIndex(ct.loc[ct["labor_day"], "anchor"])
    cols = {}
    for tkr, h, form in PULSE:
        df = px[tkr]
        v, d = C.cell(_ser(df, h, form), C.anchors_on(df, hol))
        cols[f"{tkr}{form}{h}"] = pd.Series(v, index=d)
    M = pd.DataFrame(cols)
    print("\n  pairwise correlation of the post-closure returns themselves:")
    print(M.corr().round(2).to_string())
    print(f"\n  overlapping observations: {len(M.dropna())}")

    print("\n  the LIVE slot: each pulse cell restricted to Labor Day anchors")
    rows = []
    for tkr, h, form in PULSE:
        df = px[tkr]
        s = _ser(df, h, form)
        v_all, _ = C.cell(s, C.anchors_on(df, hol))
        v, d = C.cell(s, C.anchors_on(df, ld))
        r_all, r = summarize(v_all), summarize(v)
        wins = int((v > 0).sum())
        cost = C.COST_RT_BPS.get(tkr, np.nan)
        rows.append({"proxy": tkr, "h": h, "form": form,
                     "n_allhol": r_all["n"], "allhol_mean": r_all["mean_pct"],
                     "n_LD": r.get("n", 0), "LD_record": f"{wins}-{r.get('n', 0)-wins}",
                     "LD_mean": r.get("mean_pct", np.nan),
                     "LD_med": r.get("median_pct", np.nan),
                     "LD_hit": r.get("hit", np.nan),
                     "LD_signp_up": sign_test(wins, r.get("n", 0)),
                     "LD_worst": r.get("worst_pct", np.nan),
                     "x_cost_allhol": (abs(r_all["mean_pct"]) * 100 / cost
                                       if cost == cost and cost else np.nan)})
    d = pd.DataFrame(rows)
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].round(3)
    print(d.to_string(index=False))

    print("\n  HYG and TLT by holiday slot (the two not covered in F):")
    holdf = ct[ct["kind"] == "holiday"].copy()
    holdf["slot"] = holdf["anchor"].dt.month
    for tkr, h, form in (("HYG", 5, "MOC"), ("TLT", 3, "MOC"), ("SVXY", 3, "MOC")):
        df = px[tkr]
        s = _ser(df, h, form)
        rows = []
        for slot, g in holdf.groupby("slot"):
            v, _ = C.cell(s, C.anchors_on(df, pd.DatetimeIndex(g["anchor"])))
            r = summarize(v)
            if r.get("n"):
                rows.append({"month": slot, "n": r["n"], "mean_pct": r["mean_pct"],
                             "hit": r["hit"], "t": r["t"]})
        dd = pd.DataFrame(rows).sort_values("mean_pct", ascending=False)
        for c in dd.columns:
            if dd[c].dtype.kind == "f":
                dd[c] = dd[c].round(3)
        print(f"\n  {tkr} {form} h={h} by anchor month (9 = Labor Day):")
        print(dd.to_string(index=False))


def main() -> None:
    px = C.load_panel()
    cal = C.nyse_calendar(px)
    ct = C.closure_table(cal)
    hol = pd.DatetimeIndex(ct.loc[ct["kind"] == "holiday", "anchor"])
    wkd = pd.DatetimeIndex(ct.loc[ct["kind"] == "weekend", "anchor"])
    section_a(px, hol, wkd)
    section_b(px, hol)
    section_c(px, hol, wkd)
    section_d(px, hol, wkd)
    section_e(px, hol, wkd)
    section_f(px, ct)
    section_g(px, ct)


if __name__ == "__main__":
    main()
