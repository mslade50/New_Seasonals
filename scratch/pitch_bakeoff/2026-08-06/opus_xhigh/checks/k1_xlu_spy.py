"""K1 ADVERSARIAL CHECK -- SHORT XLU / LONG SPY after a 21d defensive washout.

Brief: kill it. Default to dead.

Attacks:
  (0) honest trigger -- does it fire TODAY (2026-08-05 close)? threshold and
      lookback sensitivity; the z10 leg the triage used.
  (a) leg decomposition -- short-XLU vs long-SPY contribution net of each
      instrument's own unconditional forward drift.
  (b) era stability -- 2018 split, decade buckets, drop 2020-2022, drop
      2000-2008.
  (c) rates in disguise -- condition on ^TNX 21d change; report which side
      TODAY sits in.
  (d) worst window / worst calendar year.
  (e) MOO-next-session entry (how it would actually be placed).
  (f) overlap -- corr/beta of the spread vs SPY's own forward return.
  (g) carry -- XLU short dividend accrual + ex-div window timing.
  (h) today-matched cell -- SPY at a 52w high AND rates rising, which is the
      actual state on 2026-08-05.
  robustness -- episode declustering at three gaps, LOYO, bootstrap,
      drop-best-episode.

Run:  python k1_xlu_spy.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _common as C

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 40)
RNG = np.random.default_rng(42)

P = C.load(["XLU", "SPY", "^TNX"])
XLU, SPY, TNX = P["XLU"], P["SPY"], P["^TNX"]
IDX = XLU.index.intersection(SPY.index)
XLU, SPY = XLU.reindex(IDX), SPY.reindex(IDX)
xlu, spy = XLU["Close"], SPY["Close"]
tnx = TNX["Close"].reindex(IDX).ffill()

HOR = (3, 5, 10, 21)


def hdr(t: str) -> None:
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


def sub(t: str) -> None:
    print("\n" + "-" * 100 + f"\n{t}\n" + "-" * 100)


def spread(k: int) -> pd.Series:
    """SHORT XLU / LONG SPY, MOC at signal close -> MOC k sessions later."""
    return C.fwd(spy, k) - C.fwd(xlu, k)


def spread_moo(k: int) -> pd.Series:
    """SHORT XLU / LONG SPY, MOO the session AFTER the signal -> MOC t+k."""
    return C.fwd_from_next_open(SPY, k) - C.fwd_from_next_open(XLU, k)


# ================================================================ 0. TRIGGER
hdr("0. THE TRIGGER -- does an honest version fire TODAY (signal = 2026-08-05 close)?")

rel21 = C.ret(xlu, 21) - C.ret(spy, 21)
z = C.z10(xlu)
spy_52wh = spy >= spy.rolling(252, min_periods=200).max() * 0.999
dtnx = tnx.diff(21)

print(f"last usable bar          : {IDX[-1].date()}")
print(f"XLU 21d ret              : {C.ret(xlu,21).iloc[-1]:+.2f}%")
print(f"SPY 21d ret              : {C.ret(spy,21).iloc[-1]:+.2f}%")
print(f"rel21 (XLU-SPY)          : {rel21.iloc[-1]:+.2f}pp")
print(f"z10(XLU) [_common form]  : {z.iloc[-1]:+.2f}   (triage leg needs < -1.50 -> "
      f"{'FIRES' if z.iloc[-1] < -1.5 else 'DOES NOT FIRE'})")
print(f"SPY within 0.1% of 52w hi: {bool(spy_52wh.iloc[-1])}")
print(f"^TNX 21d change          : {dtnx.iloc[-1]:+.3f}pp  "
      f"({'RISING' if dtnx.iloc[-1] > 0 else 'FALLING'})")
print()

variants = {}
for lb in (504, 756, 1008):
    for q in (0.03, 0.05, 0.10):
        thr = rel21.rolling(lb, min_periods=250).quantile(q)
        cond = (rel21 <= thr) & rel21.notna()
        variants[(lb, q)] = cond
        print(f"  rel21 <= {int(q*100):>2}th pctile of trailing {lb:>4}d "
              f"-> today's thresh {thr.iloc[-1]:+.2f}pp | today rel21 {rel21.iloc[-1]:+.2f}pp "
              f"| FIRES TODAY: {bool(cond.iloc[-1])!s:>5} | N={int(cond.sum())}")

COND_TRIAGE = variants[(756, 0.05)] & (z < -1.5)
COND_5 = variants[(756, 0.05)]            # as-pitched, relative leg alone
COND_10 = variants[(756, 0.10)]           # loosest honest form that FIRES today

print(f"\n  triage trigger (5th pctile + z10 < -1.5) : fires today "
      f"{bool(COND_TRIAGE.iloc[-1])} | N={int(COND_TRIAGE.sum())}")
print(f"  as-pitched relative leg alone (5th pctile): fires today "
      f"{bool(COND_5.iloc[-1])} | N={int(COND_5.sum())}")
print(f"  ONLY form that fires today (10th pctile)  : fires today "
      f"{bool(COND_10.iloc[-1])} | N={int(COND_10.sum())}")
print("\n  >>> NOTE: to make this fire on 2026-08-05 the percentile band has to be")
print("  >>> DOUBLED (5th -> 10th), which roughly doubles the sample. Everything")
print("  >>> below is therefore run on BOTH, with the 10th-pctile form as the one")
print("  >>> a pitch written today would actually have to use.")


# ============================================================ battery pieces
def cell_table(cond: pd.Series, label: str, moo: bool = False, gap: int = 5):
    rows = []
    for k in HOR:
        sp = spread_moo(k) if moo else spread(k)
        m = cond & sp.notna()
        sig = sp[m]
        if len(sig) == 0:
            continue
        ep = C.declusterize(sig.index, gap_td=gap)
        rows.append(C.describe(f"{label} h{k} all-days", sig, baseline=sp.dropna()))
        rows.append(C.describe(f"{label} h{k} episodes", sig[ep], baseline=sp.dropna()))
    print(pd.DataFrame(rows).to_string(index=False))


def decomposition(cond: pd.Series, label: str):
    rows = []
    for k in HOR:
        fx, fs = C.fwd(xlu, k), C.fwd(spy, k)
        m = cond & fx.notna() & fs.notna()
        cx, cs, ux, us = fx[m], fs[m], fx.dropna(), fs.dropna()
        tot = cs.mean() - cx.mean()
        rows.append({
            "h": k, "n": int(m.sum()),
            "XLU_cell": round(cx.mean(), 3), "XLU_uncond": round(ux.mean(), 3),
            "XLU_lift": round(cx.mean() - ux.mean(), 3), "XLU_lift_t": round(C.tstat(cx.values - ux.mean()), 2),
            "SPY_cell": round(cs.mean(), 3), "SPY_uncond": round(us.mean(), 3),
            "SPY_lift": round(cs.mean() - us.mean(), 3), "SPY_lift_t": round(C.tstat(cs.values - us.mean()), 2),
            "spread": round(tot, 3),
            "shortXLU_share_%": round(-cx.mean() / tot * 100, 1) if abs(tot) > 1e-9 else np.nan,
            "longSPY_share_%": round(cs.mean() / tot * 100, 1) if abs(tot) > 1e-9 else np.nan,
        })
    print(f"  [{label}]")
    print(pd.DataFrame(rows).to_string(index=False))


def eras(cond: pd.Series, label: str, ks=(5, 10)):
    for k in ks:
        sp = spread(k)
        sig = sp[cond & sp.notna()]
        ep = sig[C.declusterize(sig.index, gap_td=5)]
        print(f"\n  [{label}] h{k} -- 2018 split (all-days):")
        C.show(C.era_split(sig.index, sig.values, cut="2018-01-01"))
        print(f"  [{label}] h{k} -- 2018 split (episodes, gap 5td):")
        C.show(C.era_split(ep.index, ep.values, cut="2018-01-01"))
        buckets = {"2000-2008": ("2000-01-01", "2009-01-01"),
                   "2009-2019": ("2009-01-01", "2020-01-01"),
                   "2020-2022": ("2020-01-01", "2023-01-01"),
                   "2023-2026": ("2023-01-01", "2027-01-01")}
        rows = []
        for name, (a, b) in buckets.items():
            s = sig[(sig.index >= a) & (sig.index < b)]
            e = ep[(ep.index >= a) & (ep.index < b)]
            rows.append({**C.describe(f"h{k} {name} all-days", s), "eps": len(e),
                         "eps_avg": round(float(e.mean()), 3) if len(e) else np.nan,
                         "eps_t": round(C.tstat(e.values), 2) if len(e) else np.nan})
        print(f"  [{label}] h{k} -- regime buckets:")
        print(pd.DataFrame(rows).to_string(index=False))
        dr = sig[(sig.index < "2020-01-01") | (sig.index >= "2023-01-01")]
        de = sig[sig.index >= "2009-01-01"]
        print(f"    DROP 2020-2022 rate shock : n={len(dr)} avg={dr.mean():+.3f}% t={C.tstat(dr.values):+.2f}")
        print(f"    DROP 2000-2008            : n={len(de)} avg={de.mean():+.3f}% t={C.tstat(de.values):+.2f}")


def rates(cond: pd.Series, label: str, ks=(5, 10)):
    for k in ks:
        sp = spread(k)
        m = cond & sp.notna() & dtnx.notna()
        rows = []
        for name, s2 in (("rates RISING (dTNX21>0)  <-- TODAY", m & (dtnx > 0)),
                         ("rates FALLING (dTNX21<=0)", m & (dtnx <= 0))):
            s = sp[s2]
            e = s[C.declusterize(s.index, gap_td=5)]
            rows.append({**C.describe(f"h{k} {name}", s), "eps": len(e),
                         "eps_avg": round(float(e.mean()), 3) if len(e) else np.nan,
                         "eps_t": round(C.tstat(e.values), 2) if len(e) else np.nan})
        print(f"  [{label}]")
        print(pd.DataFrame(rows).to_string(index=False))
        c = pd.concat([sp[m], dtnx[m]], axis=1).dropna().corr().iloc[0, 1]
        print(f"    h{k} in-cell corr(spread, dTNX21) = {c:+.3f}\n")


def worst(cond: pd.Series, label: str, ks=(5, 10)):
    for k in ks:
        sp = spread(k)
        sig = sp[cond & sp.notna()]
        ep = sig[C.declusterize(sig.index, gap_td=5)]
        print(f"\n  [{label}] h{k}: worst window {sig.min():+.2f}% on {sig.idxmin().date()} | "
              f"best {sig.max():+.2f}% on {sig.idxmax().date()}")
        yr = sig.groupby(sig.index.year).agg(["count", "mean", "min", "max"])
        yr.columns = ["n", "avg", "worst", "best"]
        yr["sum"] = sig.groupby(sig.index.year).sum()
        print(yr.round(2).to_string())
        print(f"    worst calendar year by avg: {yr['avg'].idxmin()} ({yr['avg'].min():+.2f}%)")
        neg = sorted(yr.index[yr['avg'] < 0].tolist())
        print(f"    NEGATIVE-avg years: {neg}  ({len(neg)} of {len(yr)})")
        if len(ep) > 2:
            nb = ep.drop(ep.idxmax())
            print(f"    episodes n={len(ep)} avg={ep.mean():+.3f}% t={C.tstat(ep.values):+.2f} | "
                  f"drop-best avg={nb.mean():+.3f}% t={C.tstat(nb.values):+.2f}")


def robust(cond: pd.Series, label: str, ks=(5, 10)):
    for k in ks:
        sp = spread(k)
        sig = sp[cond & sp.notna()]
        for gap in (5, 10, 21):
            ep = sig[C.declusterize(sig.index, gap_td=gap)]
            print(f"  [{label}] h{k} gap={gap:>2}td: n={len(ep):>3} avg={ep.mean():+.3f}% "
                  f"med={np.median(ep):+.3f}% hit={100*(ep>0).mean():.1f}% "
                  f"t={C.tstat(ep.values):+.2f} worst={ep.min():+.2f}%")
        ep = sig[C.declusterize(sig.index, gap_td=10)]
        yrs = sorted(set(ep.index.year))
        loyo = [(y, round(C.tstat(ep[ep.index.year != y].values), 2)) for y in yrs]
        ts = [v for _, v in loyo if np.isfinite(v)]
        print(f"    h{k} LOYO(gap10): min t={min(ts):+.2f} (drop {min(loyo, key=lambda z: z[1])[0]}), "
              f"max t={max(ts):+.2f}")
        if len(ep) > 3:
            bs = RNG.choice(ep.values, size=(20000, len(ep)), replace=True).mean(axis=1)
            print(f"    h{k} episode bootstrap: P(mean<=0)={(bs<=0).mean():.4f}  "
                  f"5th pct of mean={np.percentile(bs,5):+.3f}%")


# ======================================================== RUN: as-pitched 5th
hdr("1A. AS-PITCHED CELL (rel21 <= 5th pctile 756d) -- DOES NOT FIRE TODAY")
cell_table(COND_5, "p5")
sub("(a) leg decomposition")
decomposition(COND_5, "p5")
sub("(b) era stability")
eras(COND_5, "p5")
sub("(c) rates conditioning")
rates(COND_5, "p5")
sub("(d) worst window / worst year")
worst(COND_5, "p5")
sub("robustness")
robust(COND_5, "p5")

# ================================================== RUN: fires-today 10th
hdr("1B. THE ONLY FORM THAT FIRES TODAY (rel21 <= 10th pctile 756d)")
cell_table(COND_10, "p10")
sub("(a) leg decomposition")
decomposition(COND_10, "p10")
sub("(b) era stability")
eras(COND_10, "p10")
sub("(c) rates conditioning")
rates(COND_10, "p10")
sub("(d) worst window / worst year")
worst(COND_10, "p10")
sub("robustness")
robust(COND_10, "p10")

hdr("1C. TRIAGE TRIGGER WITH THE z10 LEG (does not fire today) -- for reference")
cell_table(COND_TRIAGE, "p5+z10")
sub("(b) era stability")
eras(COND_TRIAGE, "p5+z10")
sub("robustness")
robust(COND_TRIAGE, "p5+z10")


# ------------------------------------------------------------ (e) MOO entry
hdr("(e) ENTRY REALISM -- MOO the session AFTER the signal")
print("  [p10 = the fires-today trigger]")
cell_table(COND_10, "p10 MOO", moo=True)
print("\n  side-by-side means (all-days, p10):")
for k in HOR:
    a = spread(k)[COND_10].dropna()
    b = spread_moo(k)[COND_10].dropna()
    print(f"    h{k:>2}: MOC-at-signal {a.mean():+.3f}% (t {C.tstat(a.values):+.2f})  ->  "
          f"MOO-next-open {b.mean():+.3f}% (t {C.tstat(b.values):+.2f})  "
          f"delta {b.mean()-a.mean():+.3f}pp")
on = ((SPY["Open"].shift(-1) / spy - 1) - (XLU["Open"].shift(-1) / xlu - 1)) * 100
print(f"\n  overnight spread move signal-close -> next open: in-cell {on[COND_10].mean():+.3f}% "
      f"(uncond {on.dropna().mean():+.3f}%)")


# ------------------------------------------------------------- (f) overlap
hdr("(f) BOOK OVERLAP -- the spread vs plain long SPY")
for k in (5, 10):
    sp, fs = spread(k), C.fwd(spy, k)
    m = COND_10 & sp.notna() & fs.notna()
    d = pd.concat([sp[m], fs[m]], axis=1).dropna(); d.columns = ["spread", "spy"]
    du = pd.concat([sp, fs], axis=1).dropna(); du.columns = ["spread", "spy"]
    print(f"  h{k}: in-cell corr(spread, SPY fwd)={d.corr().iloc[0,1]:+.3f} "
          f"beta={np.polyfit(d['spy'], d['spread'],1)[0]:+.3f} | "
          f"uncond corr={du.corr().iloc[0,1]:+.3f} beta={np.polyfit(du['spy'],du['spread'],1)[0]:+.3f}")
    print(f"       in-cell SPY fwd mean {d['spy'].mean():+.3f}% vs uncond {du['spy'].mean():+.3f}%")
print("\n  Live book context on 2026-08-06: dial 54.7 (>=50) with P/C fear OFF ->")
print("  PC_FEAR_BANDS zero the 6 frag-band carriers (FAMILY4 + 3x Bear Fade +")
print("  Monthly Weak Close). 52wh Breakout's dial_filter is 63d-10ma < 30, so it")
print("  is BLOCKED at 54.7. Net: the systematic book's long-SPY-beta sleeves are")
print("  mostly OFF today, so the overlap is smaller than it looks -- but the")
print("  Event Sleeve is not, and this trade is +0.26 beta to SPY regardless.")


# ------------------------------------------------------------- (g) carry
hdr("(g) CARRY / BORROW ON THE SHORT XLU LEG")
print("  master_prices is DIVIDEND-ADJUSTED: the measured spread ALREADY pays XLU's")
print("  dividend on the short and receives SPY's on the long, so the historical")
print("  numbers are on a total-return basis and are NOT flattered by carry.")
print("  XLU yield ~3.0%/yr vs SPY ~1.1%/yr -> ~1.9%/yr net drag on the structure:")
for k in HOR:
    print(f"    h{k:>2}: ~{-(3.0-1.1)/252*k:.3f}% of notional, but only if an ex-div lands in the window")
print("\n  XLU (Select Sector SPDR) ex-div convention: quarterly, ~3rd Friday of")
print("  MAR / JUN / SEP / DEC. Next expected ~2026-09-18. A 10-session hold entered")
print("  2026-08-06 ends ~2026-08-20 -> NO XLU ex-div inside the window.")
print("  NOT verifiable from the adjusted cache -- convention, not measurement.")
fr = []
for k in (5, 10, 21):
    sig = spread(k)[COND_10].dropna()
    hit = [(d.month in (3, 6, 9, 12)) or
           ((d + pd.tseries.offsets.BDay(k)).month in (3, 6, 9, 12)) for d in sig.index]
    fr.append((k, round(float(np.mean(hit)) * 100, 1)))
print(f"  share of historical p10 windows touching a dividend-bearing month: {fr}")
print("  Borrow: XLU is a ~$15bn ETF, general collateral, ~0.3%/yr -- immaterial at h10.")


# ------------------------------------- (i) THE PAIR vs EACH LEG STANDALONE
hdr("(i) DOES THE SHORT-XLU LEG EARN ITS PLACE? pair vs long-SPY-only vs short-XLU-only")
print("  If the pair's return and risk-adjusted return are both WORSE than plain")
print("  long SPY inside the same cell, the pair structure is decoration.")
for name, cond in (("p5", COND_5), ("p10", COND_10)):
    for k in (5, 10):
        fx, fs = C.fwd(xlu, k), C.fwd(spy, k)
        m = cond & fx.notna() & fs.notna()
        legs = {"PAIR (long SPY/short XLU)": (fs[m] - fx[m]).values,
                "long SPY only": fs[m].values,
                "short XLU only": (-fx[m]).values}
        rows = []
        for lab, v in legs.items():
            rows.append({"variant": f"[{name}] h{k} {lab}", "n": len(v),
                         "avg%": round(v.mean(), 3), "sd%": round(v.std(ddof=1), 3),
                         "avg/sd": round(v.mean() / v.std(ddof=1), 3),
                         "hit%": round((v > 0).mean() * 100, 1),
                         "t": round(C.tstat(v), 2), "worst%": round(v.min(), 2)})
        print(pd.DataFrame(rows).to_string(index=False))
        print()


# --------------------------------------------------------- (h) today-matched
hdr("(h) TODAY-MATCHED CELL -- SPY near a 52w high AND rates rising (2026-08-05 state)")
hi252 = spy.rolling(252, min_periods=200).max()
print(f"  SPY is {(spy.iloc[-1]/hi252.iloc[-1]-1)*100:+.2f}% vs its trailing 252d CLOSING high")
print("  -> a 0.1%-of-high definition EXCLUDES today; a 1.0% definition INCLUDES it.")
print("  WARNING: this is a THIRD conditioning layer chosen AFTER seeing the")
print("  base cell. Treat as specification search, not evidence.\n")
for tol, tlab in ((0.999, "within 0.1% of 52w closing high (today: NO)"),
                  (0.99, "within 1.0% of 52w closing high (today: YES)")):
    near_hi = spy >= hi252 * tol
    print(f"  --- {tlab} ---")
    for name, cond in (("p5", COND_5), ("p10", COND_10)):
        for k in (5, 10):
            sp = spread(k)
            m = cond & sp.notna() & near_hi & (dtnx > 0)
            s = sp[m]
            if len(s) == 0:
                print(f"  [{name}] h{k}: EMPTY cell")
                continue
            e = s[C.declusterize(s.index, gap_td=10)]
            print(f"  [{name}] h{k}: n={len(s)} avg={s.mean():+.3f}% med={np.median(s):+.3f}% "
                  f"hit={100*(s>0).mean():.1f}% t={C.tstat(s.values):+.2f} worst={s.min():+.2f}% "
                  f"|| episodes n={len(e)} avg={e.mean():+.3f}% t={C.tstat(e.values):+.2f}")
            if k == 10 and name == "p10":
                print(f"        episode dates: {[str(d.date()) for d in e.index]}")
            # and the same cell measured as plain long SPY
            fs = C.fwd(spy, k)
            print(f"        ...same cell, PLAIN LONG SPY: avg={fs[m].mean():+.3f}% "
                  f"t={C.tstat(fs[m].values):+.2f} (pair keeps "
                  f"{s.mean()/fs[m].mean()*100:.0f}% of it)")
    print()


hdr("PRIOR EPISODES (p10, h10 MOC) -- last 25")
sp10 = spread(10)
sig = sp10[COND_10 & sp10.notna()]
ep = sig[C.declusterize(sig.index, gap_td=10)]
print(pd.DataFrame({"episode_date": [d.date() for d in ep.index],
                    "spread_h10_%": ep.round(2).values}).tail(25).to_string(index=False))

hdr("SANITY -- unconditional XLU vs SPY forward drift, full sample 2000-2026")
for k in HOR:
    print(f"  h{k:>2}: XLU {C.fwd(xlu,k).mean():+.3f}%  SPY {C.fwd(spy,k).mean():+.3f}%  "
          f"SPY-XLU {C.fwd(spy,k).mean()-C.fwd(xlu,k).mean():+.3f}%  "
          f"(the structure has NO unconditional drift to lean on)")
