"""C7 -- the SEPTEMBER month-of-year on the DOLLAR, entered at the August
month-end close.

Closed on equities 2026-08-27 (c9_month_of_year.py, max-of-12 permutation
P = 0.238). Never run on FX. Vehicles DX-Y.NYB (tradeable as DX futures,
~1.5 bps round trip) and UUP (~6 bps, and the registry already holds a
standing finding that UUP's drag/spread cannot pay ~6 bps of edge).

Entry convention: signal read at the ME-1 close, entry MOC at the LAST close
of the prior month (ME-0), exit h sessions later. Month LABEL is the NEW
month, so "September" = entered at the last August close.

Kill tests:
 1. twelve-month table at h = 1, 3, 5, 10, CHARGED with a max-of-12
    permutation P (shuffle month labels across the same anchor returns)
 2. midterm vs non-midterm, graded with sign_test (six observations)
 3. era split pre/post 2013 AND pre/post 2018
 4. the dollar's own drift control (same span) + local +/-126td ex-trigger
 5. cost at 1.5 bps (DX) and 6.0 bps (UUP), need >= 5x
 6. is this just the September EQUITY cell sign-flipped? correlate the
    September dollar window with the matched SPY window across years
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul",
          "Aug", "Sep", "Oct", "Nov", "Dec"]
VEH = ["DX-Y.NYB", "UUP", "SPY"]
COST = {"DX-Y.NYB": 1.5, "UUP": 6.0, "SPY": 4.0}
HS = (1, 3, 5, 10)
rng = np.random.default_rng(42)

px = load_prices(VEH)
ser = {t: px[t]["Close"].dropna() for t in VEH}


def ltd_positions(idx):
    """Positions of the LAST trading session of each COMPLETE calendar month."""
    per = pd.Series(idx.to_period("M"), index=range(len(idx)))
    out = [int(g.index.max()) for _, g in per.groupby(per.values)]
    return out[:-1]           # drop the INCOMPLETE final month (2026-08)


def cell(t, h):
    """Entry at the last close of month M, exit h sessions later.
    Label = the month ENTERED INTO (M+1)."""
    s = ser[t]
    idx, v = s.index, s.values
    e_, r_, lab_ = [], [], []
    for p in ltd_positions(idx):
        if p + h >= len(idx) or p + 1 >= len(idx):
            continue
        e_.append(idx[p])
        r_.append(v[p + h] / v[p] - 1.0)
        lab_.append(idx[p + 1].month)          # the NEW month
    return pd.DatetimeIndex(e_), np.asarray(r_, float), np.asarray(lab_)


print("=" * 78)
print("1. TWELVE-MONTH TABLE ON THE DOLLAR (entry = prior month's last close)")
print("=" * 78)
for t in ("DX-Y.NYB", "UUP"):
    for h in HS:
        d, r, m = cell(t, h)
        rows = []
        for mm in range(1, 13):
            v = r[m == mm]
            rr = summarize(v, MONTHS[mm - 1])
            rows.append(rr)
        show(rows, "%s  h=%d  (entry at prior month end, N total=%d, span %s..%s)"
             % (t, h, len(r), d[0].date(), d[-1].date()))
        sep = r[m == 9]
        base = summarize(r, "ALL months pooled (own drift, same anchors)")
        print("  ALL-month pooled control: %+.3f%% (N=%d)  |  September %+.3f%% (N=%d)  "
              "-> September EXCESS %+.3fpp"
              % (base["mean_pct"], base["n"], 100 * sep.mean(), len(sep),
                 100 * sep.mean() - base["mean_pct"]))

print()
print("=" * 78)
print("1b. THE TWELVE-MONTH SCAN CHARGE (max-of-12 permutation, 3000 draws)")
print("=" * 78)
for t in ("DX-Y.NYB", "UUP"):
    for h in HS:
        d, r, m = cell(t, h)
        ts, mus = {}, {}
        for mm in range(1, 13):
            v = r[m == mm]
            if len(v) < 3:
                ts[mm], mus[mm] = np.nan, np.nan
                continue
            ts[mm] = v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))
            mus[mm] = v.mean()
        obs_t = ts[9]
        rank_t = 1 + sum(1 for mm in ts if not np.isnan(ts[mm]) and ts[mm] > obs_t)
        rank_mu = 1 + sum(1 for mm in mus if not np.isnan(mus[mm]) and mus[mm] > mus[9])
        mx = []
        for _ in range(3000):
            perm = rng.permutation(m)
            best = -1e9
            for mm in range(1, 13):
                v = r[perm == mm]
                if len(v) < 3:
                    continue
                sd = v.std(ddof=1)
                if sd <= 0:
                    continue
                best = max(best, v.mean() / (sd / np.sqrt(len(v))))
            mx.append(best)
        mx = np.asarray(mx)
        bestm = max((mm for mm in ts if not np.isnan(ts[mm])), key=lambda k: ts[k])
        print("  %-9s h=%2d: Sep t=%+.2f  ranks %d of 12 by t / %d of 12 by mean;  "
              "P(max-of-12 t >= obs) = %.3f;  best month %s at t=%+.2f"
              % (t, h, obs_t, rank_t, rank_mu, (mx >= obs_t).mean(),
                 MONTHS[bestm - 1], ts[bestm]))

print()
print("=" * 78)
print("2. MIDTERM SPLIT (sign_test, not a t-stat)")
print("=" * 78)
for t in ("DX-Y.NYB", "UUP"):
    for h in HS:
        d, r, m = cell(t, h)
        sel = m == 9
        a, dd = r[sel], d[sel]
        mid = np.array([x.year % 4 == 2 for x in dd])
        w = int((a[mid] > 0).sum())
        rows = [summarize(a, "%s Sep h=%d ALL" % (t, h)),
                summarize(a[mid], "MIDTERM (today) N=%d" % int(mid.sum())),
                summarize(a[~mid], "non-midterm")]
        show(rows)
        print("  midterm years: %s"
              % [(x.year, round(100 * y, 2)) for x, y in zip(dd[mid], a[mid])])
        print("  midterm record %d-%d, sign p = %.4f"
              % (w, int(mid.sum()) - w, sign_test(w, int(mid.sum()))))

print()
print("=" * 78)
print("3. ERA SPLIT pre/post 2013 and pre/post 2018 (September cell)")
print("=" * 78)
for t in ("DX-Y.NYB", "UUP"):
    for h in HS:
        d, r, m = cell(t, h)
        sel = m == 9
        a, dd = r[sel], d[sel]
        show(era_split(dd, a, "2013-01-01") + era_split(dd, a, "2018-01-01"),
             "%s September h=%d" % (t, h))

print()
print("=" * 78)
print("4. CONTROLS: own drift same span + local +/-126td ex-trigger")
print("=" * 78)
for t in ("DX-Y.NYB", "UUP"):
    for h in HS:
        d, r, m = cell(t, h)
        sel = m == 9
        a, dd = r[sel], d[sel]
        s = ser[t]
        allf = (s.shift(-h) / s - 1.0)
        span = (allf.index >= dd[0]) & (allf.index <= dd[-1])
        loc = local_control(allf.dropna().index, dd, 126)
        show([summarize(a, "COND September (N=%d)" % len(a)),
              summarize(r, "CTRL month-end anchors, all 12 months"),
              summarize(allf[span].values, "CTRL-a own drift, same span, all days"),
              summarize(allf.dropna().values, "CTRL-b all days full history"),
              summarize(allf.loc[loc].values, "CTRL-c local +/-126td ex-trigger")],
             "%s h=%d" % (t, h))
        print("  concentration: %s" % cluster_note(dd, a, k=2))
        w = int((a > 0).sum())
        print("  record %d-%d, sign p = %.4f, bootstrap P(mean<=0) = %.3f"
              % (w, len(a) - w, sign_test(w, len(a)), bootstrap_p_le0(a)))

print()
print("=" * 78)
print("5. COST (need >= 5x)")
print("=" * 78)
for t in ("DX-Y.NYB", "UUP"):
    for h in HS:
        d, r, m = cell(t, h)
        sel = m == 9
        a = r[sel]
        base = r.mean()
        edge_bps = 100 * 100 * a.mean()
        exc_bps = 100 * 100 * (a.mean() - base)
        print("  %-9s h=%2d: raw %+.2f bps -> %.2fx | excess-over-all-months %+.2f bps"
              " -> %.2fx  (cost %.1f bps rt)"
              % (t, h, edge_bps, edge_bps / COST[t], exc_bps, exc_bps / COST[t],
                 COST[t]))

print()
print("=" * 78)
print("6. IS THIS THE SEPTEMBER EQUITY CELL SIGN-FLIPPED?")
print("=" * 78)
for h in HS:
    dd_, rd, md = cell("DX-Y.NYB", h)
    ds_, rs, ms = cell("SPY", h)
    a = pd.Series(rd[md == 9], index=dd_[md == 9])
    b = pd.Series(rs[ms == 9], index=ds_[ms == 9])
    j = pd.concat([a.rename("dx"), b.rename("spy")], axis=1).dropna()
    if len(j) < 3:
        continue
    c = float(np.corrcoef(j["dx"], j["spy"])[0, 1])
    print("  h=%2d: corr(Sep DX window, Sep SPY window) = %+.3f over %d years; "
          "DX %+.3f%% / SPY %+.3f%%"
          % (h, c, len(j), 100 * j["dx"].mean(), 100 * j["spy"].mean()))
