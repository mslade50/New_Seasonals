"""kR red team on c10: LONG dollar MOC on the FOMC decision close, h=3/h=5.

Alignment reuses kD/kA: returns on the vehicle's OWN index, entry close on the
decision session D (lag=0 on D), exit close D+h. tdom-matched excess = return
minus the mean h-return of same trading-day-of-month sessions with FOMC
windows p-5..p+5 removed.

1. SEP split. SEP released ON the decision day (press conference with
   projections): 2011-04-27, 2011-06-22, 2011-11-02, 2012 Jan/Apr/Jun/Sep/Dec,
   then every Mar/Jun/Sep/Dec meeting 2013+. 2007-11..2011-01 projections came
   out with the MINUTES three weeks later, so those meetings are NOT SEP-on-D.
   Splits: full sample; SEP era (>= 2011-04-27); 2019+ (press conference at
   every meeting, so SEP is the only format difference); pre-SEP-era quarterly
   months (Mar/Jun/Sep/Dec, the quad-witch calendar WITHOUT a SEP); September
   alone; quad witch in hold x SEP. Difference test: Welch t + label
   permutation (20000, seed 11), two-sided, on tdom-matched excess.
2. Tape conditioners on the eve: DX 5d rank >= 75, DX 21d rank, UUP 5d rank,
   ^TNX at trailing-252 max / within 2%.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pxd = load_prices(["DX-Y.NYB", "UUP", "^TNX"])
FOMC = load_events(["fomc_decision"])["date"]
FOMC = pd.DatetimeIndex(FOMC[FOMC <= pd.Timestamp("2026-09-15")])
SEP_EARLY = pd.DatetimeIndex(["2011-04-27", "2011-06-22", "2011-11-02", "2012-01-25",
                              "2012-04-25", "2012-06-20", "2012-09-13", "2012-12-12"])
SEP_START = pd.Timestamp("2011-04-27")


def is_sep(d):
    return (d in SEP_EARLY) or (d.year >= 2013 and d.month in (3, 6, 9, 12))


def tdom_of(ix):
    ym = pd.Series(ix.year * 100 + ix.month, index=ix)
    return ym.groupby(ym.values).cumcount().values + 1


class Veh:
    def __init__(self, tkr):
        s = pxd[tkr]["Close"].dropna()
        self.s, self.idx = s, s.index
        self.td = tdom_of(s.index)
        self.pos = pd.Series(np.arange(len(s)), index=s.index)
        dd = [d for d in FOMC if d in self.pos.index]
        self.dates = pd.DatetimeIndex(dd)
        self.dp = np.array([self.pos[d] for d in dd])
        self.excl = np.zeros(len(s), bool)
        for p in self.dp:
            self.excl[max(0, p - 5):p + 6] = True

    def cell(self, h):
        r = (self.s.shift(-h) / self.s - 1.0).values
        ok = ~np.isnan(r)
        b = {j: np.nanmean(r[(self.td == j) & ~self.excl & ok]) for j in np.unique(self.td)}
        keep = ok[self.dp]
        dp = self.dp[keep]
        x = np.array([r[p] - b[self.td[p]] for p in dp])
        return self.dates[keep], r[dp], x


def tst(x):
    x = x[~np.isnan(x)]
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 else np.nan


def row(lbl, v, x):
    n = len(x)
    if n == 0:
        return {"label": lbl, "n": 0}
    we = int((x > 0).sum())
    return {"label": lbl, "n": n, "raw%": 100 * v.mean(), "tdomX%": 100 * x.mean(), "X_t": tst(x),
            "X_rec": f"{we}-{n - we}", "X_sign_p": sign_test(we, n), "sd%": 100 * v.std(ddof=1)}


def diff(a, b, nperm=20000, seed=11):
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    d = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    rng = np.random.default_rng(seed)
    pool = np.concatenate([a, b])
    na = len(a)
    cnt = 0
    for _ in range(nperm):
        p = rng.permutation(pool)
        cnt += abs(p[:na].mean() - p[na:].mean()) >= abs(d) - 1e-15
    return f"diff {100 * d:+.3f}pp  Welch t {d / se:+.2f}  perm p(2s) {(1 + cnt) / (nperm + 1):.3f}"


def split(title, v, x, m_live, lbl_live, lbl_rest):
    show([row(lbl_live, v[m_live], x[m_live]), row(lbl_rest, v[~m_live], x[~m_live])], title)
    print("   ", diff(x[m_live], x[~m_live]))


VEHS = {"DX-Y.NYB": Veh("DX-Y.NYB"), "UUP": Veh("UUP")}
qw_ev = load_events(["quad_witching"])["date"]

for tkr, V in VEHS.items():
    for h in (3, 5):
        dd, v, x = V.cell(h)
        print(f"\n{'#' * 100}\n# {tkr} LONG, entry D close, h={h}, N={len(v)}  all tdomX {100 * x.mean():+.3f}% "
              f"t {tst(x):.2f}\n{'#' * 100}")
        sep = np.array([is_sep(d) for d in dd])
        era = dd >= SEP_START
        qm = np.isin(dd.month, [3, 6, 9, 12])
        sept = dd.month == 9
        qw = event_in_window(dd, V.idx, h, lag=0, kinds=("quad_witching",))
        split("1a. full sample: SEP-on-D vs not", v, x, sep, "SEP", "non-SEP")
        m = era
        split("1b. SEP era (>=2011-04-27): SEP vs non-SEP", v[m], x[m], sep[m], "SEP (era)", "non-SEP (era)")
        m = dd >= pd.Timestamp("2019-01-01")
        split("1c. 2019+ (presser every meeting): SEP vs non-SEP", v[m], x[m], sep[m], "SEP 2019+", "non-SEP 2019+")
        m = dd < SEP_START
        split("1d. PRE-SEP era: quarterly month (QW calendar, no SEP) vs other", v[m], x[m], qm[m],
              "Mar/Jun/Sep/Dec pre-2011", "other months pre-2011")
        split("1e. September decisions vs rest", v, x, sept, "September", "not September")
        m = era
        split("1f. September vs rest, SEP era", v[m], x[m], sept[m], "Sept (era)", "not Sept (era)")
        show([row("QW in & SEP", v[qw & sep], x[qw & sep]), row("QW in & non-SEP", v[qw & ~sep], x[qw & ~sep]),
              row("QW out & SEP", v[~qw & sep], x[~qw & sep]), row("QW out & non-SEP", v[~qw & ~sep], x[~qw & ~sep])],
             "1g. quad witch in hold x SEP")
        # SEP-side per-era stability
        yrs = dd.year.values
        show([row("SEP 2011-2017", v[sep & (yrs < 2018)], x[sep & (yrs < 2018)]),
              row("SEP 2018+", v[sep & (yrs >= 2018)], x[sep & (yrs >= 2018)]),
              row("nonSEP-era 2011-2017", v[~sep & era & (yrs < 2018)], x[~sep & era & (yrs < 2018)]),
              row("nonSEP 2018+", v[~sep & (yrs >= 2018)], x[~sep & (yrs >= 2018)])], "1h. SEP side by era")
        if tkr == "DX-Y.NYB" and h == 5:
            s_idx = np.where(sep)[0]
            print("   SEP-side h5 episodes (date, raw%, tdomX%):")
            print("   " + "; ".join(f"{dd[i].date()} {100 * v[i]:+.2f}/{100 * x[i]:+.2f}" for i in s_idx))

# ------------------------------------------------------------------ 2. tape conditioners
print(f"\n{'=' * 100}\n2. TAPE CONDITIONERS ON THE EVE\n{'=' * 100}")
dxs = pxd["DX-Y.NYB"]["Close"].dropna()
uus = pxd["UUP"]["Close"].dropna()
tnx = pxd["^TNX"]["Close"].dropna()
dx_r5, dx_r21 = pct_rank(dxs, 5), pct_rank(dxs, 21)
uu_r5 = pct_rank(uus, 5)
tmx = tnx.rolling(252).max()
print(f"today's eve 2026-09-15: DX r5 {dx_r5.iloc[-1]:.1f} r21 {dx_r21.iloc[-1]:.1f}  UUP r5 {uu_r5.iloc[-1]:.1f}  "
      f"TNX {tnx.iloc[-1]:.3f} max252 {tmx.iloc[-1]:.3f}")


def eve_val(ser, d):
    e = ser.index[ser.index < d]
    return ser[e[-1]] if len(e) else np.nan


for tkr, V in VEHS.items():
    for h in (3, 5):
        dd, v, x = V.cell(h)
        r5 = np.array([eve_val(dx_r5, d) for d in dd])
        r21 = np.array([eve_val(dx_r21, d) for d in dd])
        u5 = np.array([eve_val(uu_r5, d) for d in dd])
        tv = np.array([eve_val(tnx, d) for d in dd])
        tm = np.array([eve_val(tmx, d) for d in dd])
        sep = np.array([is_sep(d) for d in dd])
        print(f"\n--- {tkr} h={h} ---")
        g = ~np.isnan(r5)
        hi = g & (r5 >= 75)
        split(f"2a. DX eve 5d rank >= 75 vs < 75 ({tkr} h{h})", v[g], x[g], hi[g], "DX r5>=75", "DX r5<75")
        show([row("DX r5 >= 85", v[g & (r5 >= 85)], x[g & (r5 >= 85)]),
              row("DX r5 >= 75 & SEP", v[hi & sep], x[hi & sep]),
              row("DX r5 >= 75 & nonSEP", v[hi & ~sep], x[hi & ~sep]),
              row("DX r21 40-60", v[g & (r21 >= 40) & (r21 <= 60)], x[g & (r21 >= 40) & (r21 <= 60)]),
              row("DX r5>=75 & r21<=60", v[hi & (r21 <= 60)], x[hi & (r21 <= 60)])], "2b. finer")
        cc = np.corrcoef(r5[g], x[g])[0, 1]
        print(f"    corr(eve DX r5, tdomX h{h}) {cc:+.3f} t {cc * np.sqrt((g.sum() - 2) / (1 - cc ** 2)):+.2f} n {g.sum()}")
        gu = ~np.isnan(u5)
        if gu.sum() > 20:
            split(f"2c. UUP eve 5d rank >= 75 vs < 75 ({tkr} h{h})", v[gu], x[gu], (u5 >= 75)[gu], "UUP r5>=75", "UUP r5<75")
        gt = ~np.isnan(tm)
        atm = gt & (tv >= tm - 1e-9)
        w2 = gt & (tv / tm - 1 >= -0.02)
        show([row("TNX at 252 max", v[atm], x[atm]), row("TNX within 2%", v[w2], x[w2]),
              row("TNX not within 2%", v[gt & ~w2], x[gt & ~w2])], f"2d. ^TNX eve state ({tkr} h{h})")
        if atm.any():
            print("    at-max episodes: " + "; ".join(f"{dd[i].date()} {100 * v[i]:+.2f}"
                                                     for i in np.where(atm)[0]))
