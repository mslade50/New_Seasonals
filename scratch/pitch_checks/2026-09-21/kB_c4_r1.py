"""c4 round 1: single-stock window dressing into the quarter-end.

Short every liquid single stock whose close sits within 2% of its 252-session
closing low at the signal close QE-8 (live: 09-18), entry MOC QE-7 (09-21),
exit MOC on the quarter-end close QE (09-30) -> h=7. One EPISODE per
month-end = the cross-sectional mean across qualifying names (date clustering
is exact here: every name shares the calendar).

The decisive control is ORDINARY MONTH-ENDS with the same state and timing
(ME-8 signal, ME-7 entry, ME exit). Also: all-days same-state control, own
drift, era / midterm / SPY-200d, September and December rows, beta-hedged,
print-free (drop names with a print in (entry, exit+1]), FOMC-in-window split,
and the reversal row (QE close -> QE+5, LONG the same names; plus names
re-selected at the QE close). Survivorship: delisted losers are absent, which
understates a short-loser edge.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kB_common import *  # noqa

H = 7
P = build_panel()
C, beta, lodist, r21 = P["C"], P["beta"], P["lodist"], P["r21"]
idx = C.index
spy = C["SPY"].values
E = P["E"]
LIQ_U = [t for t in LIQ if t in C.columns]
per = idx.to_period("M")
me_pos = np.flatnonzero(np.r_[per[1:] != per[:-1], False])  # completed months
fomc = load_events(["fomc_decision"])["date"].values.astype("datetime64[ns]")


def month_end_rows(univ, thr=0.02, sig_off=-8, ent_off=-7, ex_off=0, rev=(0, 5)):
    ci = np.array([C.columns.get_loc(t) for t in univ])
    V = C.values[:, ci]
    LD = lodist.values[:, ci]
    B = beta.values[:, ci]
    pm = print_mask(event_positions(E, idx, univ), idx, univ).values
    rows = []
    for m in me_pos:
        s, e, x = m + sig_off, m + ent_off, m + ex_off
        ra, rb = m + rev[0], m + rev[1]
        if s < 252 or rb >= len(idx):
            continue
        g = LD[s] <= thr
        g &= ~np.isnan(V[e]) & ~np.isnan(V[x])
        if g.sum() == 0:
            continue
        rn = V[x, g] / V[e, g] - 1.0
        rs = spy[x] / spy[e] - 1.0
        bh = rn - B[s, g] * rs
        prt = pm[e + 1:x + 2, :][:, g].any(axis=0)
        rrev = V[rb, g] / V[ra, g] - 1.0
        rsrev = spy[rb] / spy[ra] - 1.0
        g2 = (LD[m] <= thr) & ~np.isnan(V[rb]) & ~np.isnan(V[m])
        rrev2 = (V[rb, g2] / V[m, g2] - 1.0) - B[m, g2] * rsrev if g2.any() else np.array([np.nan])
        d = idx[m]
        rows.append({
            "me": d, "month": d.month, "year": d.year, "qe": d.month in (3, 6, 9, 12),
            "n": int(g.sum()),
            "short": -np.nanmean(rn), "rel": -np.nanmean(rn - rs), "bh": -np.nanmean(bh),
            "bh_noprint": -np.nanmean(bh[~prt]) if (~prt).any() else np.nan,
            "n_print": int(prt.sum()),
            "rev_long_bh": np.nanmean((rrev - B[m, g] * rsrev)),
            "rev_long_raw": np.nanmean(rrev),
            "rev2_long_bh": np.nanmean(rrev2),
            "spy200": bool(P["spy_above200"].values[s]),
            "fomc_in": bool(((fomc > np.datetime64(idx[e])) & (fomc <= np.datetime64(idx[x]))).any()),
            "spy_ret": rs,
        })
    return pd.DataFrame(rows)


def all_days(univ, thr=0.02):
    """Same state any day: cross-sectional mean of the h=7 short, per signal day."""
    Cu = C[univ]
    fwd = Cu.shift(-(1 + H)) / Cu.shift(-1) - 1.0
    spf = C["SPY"].shift(-(1 + H)) / C["SPY"].shift(-1) - 1.0
    bh = -(fwd - beta[univ].mul(spf, axis=0))
    g = lodist[univ] <= thr
    s = bh.where(g).mean(axis=1).dropna()
    s = s[s.index >= idx[252]]
    return s.iloc[::H]  # non-overlapping sample


def st(df, col, label):
    r = stat(df[col].values, label)
    r["names_avg"] = round(df["n"].mean(), 1) if len(df) else np.nan
    return r


for name, univ in [("LIQ", LIQ_U), ("BROAD", [t for t in C.columns if t != "SPY"])]:
    M = month_end_rows(univ)
    Q, O = M[M.qe], M[~M.qe]
    print(f"\n{'=' * 78}\n[{name}] month-ends with >=1 qualifying name: {len(M)} "
          f"(QE {len(Q)}, ordinary {len(O)})\n{'=' * 78}")
    ad = all_days(univ)
    rows = [st(Q, "short", "QE-7 -> QE short [raw]"), st(Q, "rel", "QE [SPY-rel]"),
            st(Q, "bh", "QE [beta-hedged]"),
            st(O, "short", "ORDINARY ME-7 -> ME short [raw]"), st(O, "bh", "ORDINARY ME [bh]"),
            stat(ad.values, "ALL DAYS same state, h=7 [bh], every 7th day")]
    show(rows, f"[{name}] 1. quarter-end vs ordinary month-end vs all days")
    print(f"  QE minus ordinary ME [bh]: {100 * (Q.bh.mean() - O.bh.mean()):+.3f}pp, "
          f"Welch t {welch(Q.bh, O.bh):+.2f};  [raw] {100 * (Q.short.mean() - O.short.mean()):+.3f}pp "
          f"t {welch(Q.short, O.short):+.2f}")
    rows = []
    for lbl, m in [("Sep QE", M.month == 9), ("Dec QE (year-end)", M.month == 12),
                   ("Mar QE", M.month == 3), ("Jun QE", M.month == 6),
                   ("QE pre-2018", M.qe & (M.year < 2018)), ("QE 2018+", M.qe & (M.year >= 2018)),
                   ("ORD pre-2018", ~M.qe & (M.year < 2018)), ("ORD 2018+", ~M.qe & (M.year >= 2018)),
                   ("QE midterm", M.qe & (M.year % 4 == 2)), ("Sep midterm", (M.month == 9) & (M.year % 4 == 2)),
                   ("QE SPY>200d", M.qe & M.spy200), ("QE SPY<200d", M.qe & ~M.spy200),
                   ("ORD SPY>200d", ~M.qe & M.spy200),
                   ("Sep SPY>200d", (M.month == 9) & M.spy200),
                   ("QE, FOMC NOT in window", M.qe & ~M.fomc_in), ("QE, FOMC in window", M.qe & M.fomc_in),
                   ("Sep 2018+", (M.month == 9) & (M.year >= 2018))]:
        rows.append(st(M[m], "bh", f"{lbl} [bh]"))
    show(rows, f"[{name}] 2. splits (beta-hedged short)")
    rows = [st(Q, "bh_noprint", "QE, print-free names only [bh]"),
            st(O, "bh_noprint", "ORD, print-free names only [bh]"),
            st(M[M.month == 9], "bh_noprint", "Sep, print-free [bh]")]
    show(rows, f"[{name}] 3. is it just the c3 print effect? (drop names printing in window)")
    print(f"  QE minus ORD print-free: {100 * (Q.bh_noprint.mean() - O.bh_noprint.mean()):+.3f}pp "
          f"t {welch(Q.bh_noprint, O.bh_noprint):+.2f}")
    rows = [st(Q, "rev_long_bh", "REVERSAL QE -> QE+5 LONG same names [bh]"),
            st(O, "rev_long_bh", "REVERSAL ORD ME -> ME+5 LONG [bh]"),
            st(Q, "rev2_long_bh", "REVERSAL QE -> QE+5, names re-selected at QE [bh]"),
            st(M[M.month == 12], "rev_long_bh", "REVERSAL Dec -> Jan+5 [bh]"),
            st(M[M.month == 9], "rev_long_bh", "REVERSAL Sep -> Oct+5 [bh]")]
    show(rows, f"[{name}] 4. reversal rows")
    print(f"  reversal QE minus ORD [bh]: {100 * (Q.rev_long_bh.mean() - O.rev_long_bh.mean()):+.3f}pp "
          f"t {welch(Q.rev_long_bh, O.rev_long_bh):+.2f}; corr(run-in short, reversal long) QE "
          f"{np.corrcoef(Q.bh.fillna(0), Q.rev_long_bh.fillna(0))[0, 1]:+.2f}")
    rows = []
    for thr in (0.005, 0.01, 0.02, 0.03, 0.05):
        Mt = month_end_rows(univ, thr=thr)
        rows.append({**st(Mt[Mt.qe], "bh", f"QE lodist<={thr:.3f}"),
                     "ord_mean": round(100 * Mt[~Mt.qe].bh.mean(), 3),
                     "qe_minus_ord": round(100 * (Mt[Mt.qe].bh.mean() - Mt[~Mt.qe].bh.mean()), 3),
                     "welch_t": round(welch(Mt[Mt.qe].bh, Mt[~Mt.qe].bh), 2)})
    for so, eo in ((-10, -9), (-6, -5), (-4, -3)):
        Mt = month_end_rows(univ, sig_off=so, ent_off=eo)
        rows.append({**st(Mt[Mt.qe], "bh", f"QE entry ME{eo} (h={-eo})"),
                     "ord_mean": round(100 * Mt[~Mt.qe].bh.mean(), 3),
                     "qe_minus_ord": round(100 * (Mt[Mt.qe].bh.mean() - Mt[~Mt.qe].bh.mean()), 3),
                     "welch_t": round(welch(Mt[Mt.qe].bh, Mt[~Mt.qe].bh), 2)})
    show(rows, f"[{name}] 5. definition neighbours: threshold + entry offset (QE vs ORD)")
    s9 = M[M.month == 9]
    print("  Sep by year [bh %]:", ", ".join(f"{int(y)}:{100 * v:+.2f}(n{n})" for y, v, n in
                                         zip(s9.year, s9.bh, s9.n)))
    print("  worst QE episode [bh]: %.2f%% on %s" % (100 * Q.bh.min(), Q.loc[Q.bh.idxmin(), "me"].date()))
    print("  concentration QE:", cluster_note(pd.DatetimeIndex(Q.me), Q.bh.values))

# live basket
last = idx[-1]
lv = [(t, lodist[t].iloc[-1]) for t in LIQ_U if lodist[t].iloc[-1] <= 0.02]
print(f"\nLIVE basket (LIQ, within 2% of the 252 closing low on {last.date()}):")
print("  " + ", ".join(f"{t} {100 * v:.2f}%" for t, v in sorted(lv, key=lambda z: z[1])))
