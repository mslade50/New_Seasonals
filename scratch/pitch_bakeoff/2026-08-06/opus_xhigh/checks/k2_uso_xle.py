"""K2 ADVERSARIAL CHECK -- SHORT USO / LONG XLE after a crude dislocation.

Brief: kill it. Default to dead.

Triage claim: trigger (USO 5d ret - XLE 5d ret) <= -7pp AND USO 5d rank <= 5th
pctile; LONG USO / SHORT XLE returned -4.07% h5 (N=60, t=-2.41), so the
INVERSE (short USO / long XLE) is the candidate.

Attacks:
  (0) does it fire today, and how knife-edge is the -7pp / 5th-pctile pair?
  (a) leg decomposition -- USO short vs XLE long, each against its OWN
      unconditional forward drift.
  (b) USO ROLL DECAY -- USO's unconditional forward 5d/10d mean over the whole
      sample and 2015+; subtract it from the cell. If the cell edge is smaller
      than the unconditional decay, this is a carry claim, not a signal.
  (c) era stability -- 2018 split; DROP 2020 entirely.
  (d) episode clustering at wide gaps.
  (e) overnight gap risk against a short USO -- worst adverse gap inside the
      hold window.
  (f) beta of the spread to SPY inside the cell; borrow costs.
  (g) entry realism -- MOO the next session.
  robustness -- LOYO, bootstrap, drop-best-episode, worst window / worst year.

Run:  python k2_uso_xle.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _common as C

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 40)
RNG = np.random.default_rng(42)

P = C.load(["USO", "XLE", "SPY"])
USO, XLE, SPY = P["USO"], P["XLE"], P["SPY"]
IDX = USO.index.intersection(XLE.index).intersection(SPY.index)
USO, XLE, SPY = USO.reindex(IDX), XLE.reindex(IDX), SPY.reindex(IDX)
uso, xle, spy = USO["Close"], XLE["Close"], SPY["Close"]

HOR = (3, 5, 10, 21)


def hdr(t): print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)
def sub(t): print("\n" + "-" * 100 + f"\n{t}\n" + "-" * 100)


def spread(k: int) -> pd.Series:
    """SHORT USO / LONG XLE, MOC at signal close -> MOC k sessions later."""
    return C.fwd(xle, k) - C.fwd(uso, k)


def spread_moo(k: int) -> pd.Series:
    return C.fwd_from_next_open(XLE, k) - C.fwd_from_next_open(USO, k)


# ================================================================ 0. TRIGGER
hdr("0. THE TRIGGER -- does it fire on the 2026-08-05 close, and how knife-edge?")
d5 = C.ret(uso, 5) - C.ret(xle, 5)
r5u = C.pct_rank(C.ret(uso, 5))
r5u_full = C.pct_rank(C.ret(uso, 5), 252)

print(f"last usable bar     : {IDX[-1].date()}   USO close {uso.iloc[-1]:.2f}  XLE close {xle.iloc[-1]:.2f}")
print(f"USO 5d ret          : {C.ret(uso,5).iloc[-1]:+.2f}%   (252d rank {r5u.iloc[-1]:.1f} pctile)")
print(f"XLE 5d ret          : {C.ret(xle,5).iloc[-1]:+.2f}%")
print(f"d5 (USO - XLE)      : {d5.iloc[-1]:+.2f}pp")
print(f"USO vs 200d SMA     : {(uso.iloc[-1]/uso.rolling(200).mean().iloc[-1]-1)*100:+.2f}%")
print(f"USO 63d ret rank    : {C.pct_rank(C.ret(uso,63)).iloc[-1]:.1f} pctile")
print()

grid = {}
for dthr in (-5.0, -6.0, -7.0, -8.0, -10.0):
    for rthr in (5, 10, 20):
        c = (d5 <= dthr) & (r5u <= rthr)
        grid[(dthr, rthr)] = c
        print(f"  d5 <= {dthr:>6.1f}pp AND USO 5d rank <= {rthr:>2} : "
              f"FIRES TODAY {bool(c.iloc[-1])!s:>5} | N={int(c.sum()):>4} | "
              f"episodes(gap10) {int(C.declusterize(c[c].index, gap_td=10).sum()):>3}")

COND = grid[(-7.0, 5)]
print(f"\n  AS-TRIAGED TRIGGER (d5<=-7pp, rank<=5): fires today "
      f"{bool(COND.iloc[-1])} | N={int(COND.sum())}")


def cell_table(cond, label, moo=False, gap=5):
    rows = []
    for k in HOR:
        sp = spread_moo(k) if moo else spread(k)
        sig = sp[cond & sp.notna()]
        if len(sig) == 0:
            continue
        ep = C.declusterize(sig.index, gap_td=gap)
        rows.append(C.describe(f"{label} h{k} all-days", sig, baseline=sp.dropna()))
        rows.append(C.describe(f"{label} h{k} episodes", sig[ep], baseline=sp.dropna()))
    print(pd.DataFrame(rows).to_string(index=False))


hdr("1. HEADLINE CELL -- SHORT USO / LONG XLE, MOC at the signal close")
cell_table(COND, "cell")
print("\n  (the LONG USO / SHORT XLE direction the triage measured, sign-flipped, "
      "for cross-check)")
for k in HOR:
    s = (-spread(k))[COND].dropna()
    print(f"    long-USO/short-XLE h{k:>2}: n={len(s)} avg={s.mean():+.3f}% t={C.tstat(s.values):+.2f}")


# ------------------------------------------------------ (a) leg decomposition
hdr("(a) LEG DECOMPOSITION -- is the edge the USO short or the XLE long?")
rows = []
for k in HOR:
    fu, fx = C.fwd(uso, k), C.fwd(xle, k)
    m = COND & fu.notna() & fx.notna()
    cu, cx, uu, ux = fu[m], fx[m], fu.dropna(), fx.dropna()
    tot = cx.mean() - cu.mean()
    rows.append({
        "h": k, "n": int(m.sum()),
        "USO_cell": round(cu.mean(), 3), "USO_uncond": round(uu.mean(), 3),
        "USO_lift": round(cu.mean() - uu.mean(), 3),
        "USO_lift_t": round(C.tstat(cu.values - uu.mean()), 2),
        "XLE_cell": round(cx.mean(), 3), "XLE_uncond": round(ux.mean(), 3),
        "XLE_lift": round(cx.mean() - ux.mean(), 3),
        "XLE_lift_t": round(C.tstat(cx.values - ux.mean()), 2),
        "spread": round(tot, 3),
        "shortUSO_%": round(-cu.mean() / tot * 100, 1) if abs(tot) > 1e-9 else np.nan,
        "longXLE_%": round(cx.mean() / tot * 100, 1) if abs(tot) > 1e-9 else np.nan,
    })
print(pd.DataFrame(rows).to_string(index=False))
print("\n  Each leg on its own, in-cell, with its own t (is either leg significant?):")
for k in (5, 10):
    fu, fx = C.fwd(uso, k), C.fwd(xle, k)
    m = COND & fu.notna() & fx.notna()
    print(f"    h{k:>2}: short-USO leg avg {-fu[m].mean():+.3f}% t {C.tstat(-fu[m].values):+.2f} | "
          f"long-XLE leg avg {fx[m].mean():+.3f}% t {C.tstat(fx[m].values):+.2f}")


# -------------------------------------------------------- (b) USO roll decay
hdr("(b) USO ROLL DECAY -- is the 'edge' just the fund's unconditional bleed?")
rows = []
for k in HOR:
    fu = C.fwd(uso, k)
    full = fu.dropna()
    p15 = fu[fu.index >= "2015-01-01"].dropna()
    p15n20 = fu[(fu.index >= "2015-01-01") & ~((fu.index >= "2020-01-01") & (fu.index < "2021-01-01"))].dropna()
    cell = fu[COND & fu.notna()]
    rows.append({
        "h": k,
        "USO_uncond_full": round(full.mean(), 3),
        "USO_uncond_2015+": round(p15.mean(), 3),
        "USO_uncond_2015+_ex2020": round(p15n20.mean(), 3),
        "USO_cell": round(cell.mean(), 3),
        "cell_minus_uncond_full": round(cell.mean() - full.mean(), 3),
        "cell_minus_uncond_2015+": round(cell.mean() - p15.mean(), 3),
    })
print(pd.DataFrame(rows).to_string(index=False))
print("\n  Spread edge NET of USO's unconditional decay and XLE's unconditional drift:")
for k in HOR:
    fu, fx = C.fwd(uso, k), C.fwd(xle, k)
    m = COND & fu.notna() & fx.notna()
    raw = fx[m].mean() - fu[m].mean()
    base = fx.dropna().mean() - fu.dropna().mean()
    print(f"    h{k:>2}: raw spread {raw:+.3f}%  |  unconditional spread drift {base:+.3f}%  "
          f"|  NET LIFT {raw-base:+.3f}%   ({(base/raw*100 if abs(raw)>1e-9 else float('nan')):.1f}% of "
          f"the raw number is just carry/drift)")
print("\n  Annualized: USO unconditional bleed vs XLE unconditional drift")
for lab, a, b in (("full sample", uso.index[0], uso.index[-1]),
                  ("2015+", pd.Timestamp("2015-01-01"), uso.index[-1])):
    su = uso[(uso.index >= a) & (uso.index <= b)]
    sx = xle[(xle.index >= a) & (xle.index <= b)]
    yrs = (b - a).days / 365.25
    print(f"    {lab:>12}: USO {(su.iloc[-1]/su.iloc[0])**(1/yrs)*100-100:+.2f}%/yr  "
          f"XLE {(sx.iloc[-1]/sx.iloc[0])**(1/yrs)*100-100:+.2f}%/yr  "
          f"(structure earns the difference passively)")


# --------------------------------------------------------- (c) era stability
hdr("(c) ERA STABILITY -- 2018 split, and DROP 2020 entirely")
for k in (5, 10):
    sp = spread(k)
    sig = sp[COND & sp.notna()]
    ep = sig[C.declusterize(sig.index, gap_td=5)]
    print(f"\n  h{k} -- 2018 split (all-days):")
    C.show(C.era_split(sig.index, sig.values, cut="2018-01-01"))
    print(f"  h{k} -- 2018 split (episodes gap5):")
    C.show(C.era_split(ep.index, ep.values, cut="2018-01-01"))
    ex20 = sig[~((sig.index >= "2020-01-01") & (sig.index < "2021-01-01"))]
    ex20e = ex20[C.declusterize(ex20.index, gap_td=10)]
    only20 = sig[(sig.index >= "2020-01-01") & (sig.index < "2021-01-01")]
    print(f"  h{k} -- 2020 ONLY        : n={len(only20)} avg={only20.mean():+.3f}% "
          f"t={C.tstat(only20.values):+.2f} sum={only20.sum():+.1f}")
    print(f"  h{k} -- DROP 2020        : n={len(ex20)} avg={ex20.mean():+.3f}% "
          f"t={C.tstat(ex20.values):+.2f}  || episodes(gap10) n={len(ex20e)} "
          f"avg={ex20e.mean():+.3f}% t={C.tstat(ex20e.values):+.2f}")
    buckets = {"2006-2013": ("2006-01-01", "2014-01-01"),
               "2014-2019": ("2014-01-01", "2020-01-01"),
               "2020": ("2020-01-01", "2021-01-01"),
               "2021-2026": ("2021-01-01", "2027-01-01")}
    rows = []
    for name, (a, b) in buckets.items():
        s = sig[(sig.index >= a) & (sig.index < b)]
        e = ep[(ep.index >= a) & (ep.index < b)]
        rows.append({**C.describe(f"h{k} {name}", s), "eps": len(e),
                     "eps_avg": round(float(e.mean()), 3) if len(e) else np.nan})
    print(pd.DataFrame(rows).to_string(index=False))


# ----------------------------------------------------- (d) episode clustering
hdr("(d) EPISODE CLUSTERING at wide gaps")
for k in (5, 10):
    sp = spread(k)
    sig = sp[COND & sp.notna()]
    for gap in (5, 10, 21, 42):
        ep = sig[C.declusterize(sig.index, gap_td=gap)]
        print(f"  h{k} gap={gap:>2}td: n={len(ep):>3} avg={ep.mean():+.3f}% "
              f"med={np.median(ep):+.3f}% hit={100*(ep>0).mean():.1f}% "
              f"t={C.tstat(ep.values):+.2f} worst={ep.min():+.2f}% best={ep.max():+.2f}%")
    ep = sig[C.declusterize(sig.index, gap_td=21)]
    print(f"    h{k} episode dates (gap21): {[str(d.date()) for d in ep.index]}")
    if len(ep) > 2:
        nb = ep.drop(ep.idxmax())
        print(f"    h{k} drop-best-episode: n={len(nb)} avg={nb.mean():+.3f}% t={C.tstat(nb.values):+.2f}")
    if len(ep) > 3:
        bs = RNG.choice(ep.values, size=(20000, len(ep)), replace=True).mean(axis=1)
        print(f"    h{k} episode bootstrap: P(mean<=0)={(bs<=0).mean():.4f} "
              f"5th pct of mean={np.percentile(bs,5):+.3f}%")
        yrs = sorted(set(ep.index.year))
        loyo = [(y, round(C.tstat(ep[ep.index.year != y].values), 2)) for y in yrs]
        ts = [v for _, v in loyo if np.isfinite(v)]
        print(f"    h{k} LOYO(gap21): min t={min(ts):+.2f} (drop {min(loyo,key=lambda z: z[1])[0]}) "
              f"max t={max(ts):+.2f}  full={loyo}")


# ------------------------------------------------------ (e) overnight gap risk
hdr("(e) TAIL RISK -- worst overnight gap AGAINST a short USO inside the hold")
on_uso = (USO["Open"] / uso.shift(1) - 1) * 100      # gap at each session's open
on_xle = (XLE["Open"] / xle.shift(1) - 1) * 100
on_sp = on_xle - on_uso                              # spread's overnight move
for k in (5, 10):
    worst_gap_uso, worst_gap_sp, dates = [], [], []
    idx = list(IDX)
    pos = {d: i for i, d in enumerate(idx)}
    for d in spread(k)[COND & spread(k).notna()].index:
        i = pos[d]
        w = slice(i + 1, min(i + 1 + k, len(idx)))
        gu = on_uso.iloc[w]
        gs = on_sp.iloc[w]
        if len(gu) == 0:
            continue
        worst_gap_uso.append(gu.max())   # biggest UP gap in USO = worst for a short
        worst_gap_sp.append(gs.min())    # worst overnight move for the spread
        dates.append(d)
    gu = pd.Series(worst_gap_uso, index=pd.DatetimeIndex(dates))
    gs = pd.Series(worst_gap_sp, index=pd.DatetimeIndex(dates))
    print(f"\n  h{k}: per-signal WORST single-overnight USO UP-gap inside the hold")
    print(f"    mean {gu.mean():+.2f}%  median {np.median(gu):+.2f}%  "
          f"90th pct {np.percentile(gu,90):+.2f}%  max {gu.max():+.2f}% on "
          f"{gu.idxmax().date()}")
    print(f"    share of signals with a >= +5% overnight USO gap: {(gu>=5).mean()*100:.1f}%  "
          f">= +8%: {(gu>=8).mean()*100:.1f}%")
    print(f"  h{k}: per-signal WORST single-overnight SPREAD move (XLE-USO)")
    print(f"    mean {gs.mean():+.2f}%  median {np.median(gs):+.2f}%  "
          f"10th pct {np.percentile(gs,10):+.2f}%  worst {gs.min():+.2f}% on {gs.idxmin().date()}")
print("\n  Unconditional USO overnight-gap tail, whole sample (a short is naked to this):")
g = on_uso.dropna()
print(f"    max up-gap {g.max():+.2f}%  99th pct {np.percentile(g,99):+.2f}%  "
      f"95th pct {np.percentile(g,95):+.2f}%  std {g.std():.2f}%")


# ------------------------------------------------------------ (f) SPY beta
hdr("(f) MARKET BETA OF THE SPREAD + COSTS")
for k in (5, 10):
    sp, fs = spread(k), C.fwd(spy, k)
    m = COND & sp.notna() & fs.notna()
    d = pd.concat([sp[m], fs[m]], axis=1).dropna(); d.columns = ["spread", "spy"]
    du = pd.concat([sp, fs], axis=1).dropna(); du.columns = ["spread", "spy"]
    print(f"  h{k}: in-cell corr(spread, SPY fwd)={d.corr().iloc[0,1]:+.3f} "
          f"beta={np.polyfit(d['spy'], d['spread'],1)[0]:+.3f} | "
          f"uncond corr={du.corr().iloc[0,1]:+.3f} beta={np.polyfit(du['spy'],du['spread'],1)[0]:+.3f}")
    # beta of each leg
    fu, fx = C.fwd(uso, k), C.fwd(xle, k)
    mm = COND & fu.notna() & fx.notna() & fs.notna()
    bu = np.polyfit(fs[mm], fu[mm], 1)[0]
    bx = np.polyfit(fs[mm], fx[mm], 1)[0]
    print(f"       in-cell leg betas to SPY: USO {bu:+.3f}, XLE {bx:+.3f} "
          f"-> a 1x/1x pair is NET {bx-bu:+.3f} SPY beta (NOT market neutral)")
print("\n  Costs: USO is a ~$1bn commodity pool (K-1 partnership). Short borrow on")
print("  commodity-pool ETFs is routinely 1-3%/yr and can spike on a crude squeeze;")
print("  it is NOT general collateral like an equity sector SPDR. Assume ~2%/yr")
print("  -> ~0.08% over a 10-session hold. XLE long is GC (~0.3%/yr).")


# ----------------------------------------------------------- (g) MOO realism
hdr("(g) ENTRY REALISM -- MOO the session AFTER the signal")
cell_table(COND, "MOO", moo=True)
print("\n  side-by-side (all-days):")
for k in HOR:
    a = spread(k)[COND].dropna()
    b = spread_moo(k)[COND].dropna()
    print(f"    h{k:>2}: MOC-at-signal {a.mean():+.3f}% (t {C.tstat(a.values):+.2f})  ->  "
          f"MOO-next-open {b.mean():+.3f}% (t {C.tstat(b.values):+.2f})  "
          f"delta {b.mean()-a.mean():+.3f}pp")


# --------------------------------------------------- worst window / worst year
hdr("WORST WINDOW AND WORST CALENDAR YEAR")
for k in (5, 10):
    sp = spread(k)
    sig = sp[COND & sp.notna()]
    print(f"\n  h{k}: worst window {sig.min():+.2f}% on {sig.idxmin().date()} | "
          f"best {sig.max():+.2f}% on {sig.idxmax().date()}")
    yr = sig.groupby(sig.index.year).agg(["count", "mean", "min", "max"])
    yr.columns = ["n", "avg", "worst", "best"]
    yr["sum"] = sig.groupby(sig.index.year).sum()
    print(yr.round(2).to_string())
    neg = sorted(yr.index[yr["avg"] < 0].tolist())
    print(f"    worst calendar year by avg: {yr['avg'].idxmin()} ({yr['avg'].min():+.2f}%)")
    print(f"    NEGATIVE-avg years: {neg} ({len(neg)} of {len(yr)})")

# ------------------------------------------- CONTROLS: what is the trigger adding?
hdr("CONTROLS -- does either trigger leg add anything, or is one of them the whole cell?")
ctrl_rank = (r5u <= 5)                      # crude washout alone
ctrl_d5 = (d5 <= -7.0)                      # dislocation vs energy equities alone
for k in (5, 10):
    sp = spread(k)
    rows = []
    for lab, c in (("BOTH (the cell)", COND),
                   ("USO 5d rank<=5 ONLY", ctrl_rank),
                   ("d5<=-7pp ONLY", ctrl_d5),
                   ("rank<=5 but d5>-7pp", ctrl_rank & ~ctrl_d5),
                   ("d5<=-7pp but rank>5", ctrl_d5 & ~ctrl_rank),
                   ("ALL DAYS (baseline)", pd.Series(True, index=sp.index))):
        s = sp[c.reindex(sp.index).fillna(False) & sp.notna()]
        e = s[C.declusterize(s.index, gap_td=10)] if len(s) else s
        rows.append({"control": f"h{k} {lab}", "n": len(s),
                     "avg%": round(float(s.mean()), 3) if len(s) else np.nan,
                     "med%": round(float(np.median(s)), 3) if len(s) else np.nan,
                     "hit%": round(float((s > 0).mean() * 100), 1) if len(s) else np.nan,
                     "t": round(C.tstat(s.values), 2) if len(s) else np.nan,
                     "eps": len(e),
                     "eps_avg%": round(float(e.mean()), 3) if len(e) else np.nan,
                     "eps_t": round(C.tstat(e.values), 2) if len(e) else np.nan})
    print(pd.DataFrame(rows).to_string(index=False))
    print()


hdr("THE PAIR vs EACH LEG STANDALONE vs an SPY-hedged XLE long")
print("  If long XLE alone (or XLE hedged with SPY) beats the pair on risk-adjusted")
print("  terms, the short-USO leg is decoration and the trade is 'buy energy equities'.")
for k in (5, 10):
    fu, fx, fs = C.fwd(uso, k), C.fwd(xle, k), C.fwd(spy, k)
    m = COND & fu.notna() & fx.notna() & fs.notna()
    variants = {"PAIR (long XLE / short USO)": (fx[m] - fu[m]).values,
                "long XLE only": fx[m].values,
                "short USO only": (-fu[m]).values,
                "long XLE / short SPY": (fx[m] - fs[m]).values}
    rows = []
    for lab, v in variants.items():
        rows.append({"variant": f"h{k} {lab}", "n": len(v),
                     "avg%": round(v.mean(), 3), "sd%": round(v.std(ddof=1), 3),
                     "avg/sd": round(v.mean() / v.std(ddof=1), 3),
                     "hit%": round((v > 0).mean() * 100, 1),
                     "t": round(C.tstat(v), 2), "worst%": round(v.min(), 2),
                     "best%": round(v.max(), 2)})
    print(pd.DataFrame(rows).to_string(index=False))
    print()


hdr("MODERN ERA ONLY (2021+) -- the regime the trade would actually run in")
for k in (3, 5, 10, 21):
    sp = spread(k)
    s = sp[COND & sp.notna() & (sp.index >= "2021-01-01")]
    e = s[C.declusterize(s.index, gap_td=10)]
    print(f"  h{k:>2}: n={len(s)} avg={s.mean():+.3f}% med={np.median(s):+.3f}% "
          f"hit={100*(s>0).mean():.1f}% t={C.tstat(s.values):+.2f} "
          f"worst={s.min():+.2f}% best={s.max():+.2f}% || episodes n={len(e)} "
          f"avg={e.mean():+.3f}% t={C.tstat(e.values):+.2f}")
print("\n  Same, per leg (2021+):")
for k in (5, 10):
    fu, fx = C.fwd(uso, k), C.fwd(xle, k)
    m = COND & fu.notna() & fx.notna() & (fu.index >= "2021-01-01")
    print(f"    h{k:>2}: short-USO leg {-fu[m].mean():+.3f}% (t {C.tstat(-fu[m].values):+.2f}) | "
          f"long-XLE leg {fx[m].mean():+.3f}% (t {C.tstat(fx[m].values):+.2f})")


hdr("TREND MISMATCH -- today crude is washing out INSIDE AN UPTREND")
sma200 = uso.rolling(200).mean()
above = uso > sma200
print(f"  2026-08-05: USO is {(uso.iloc[-1]/sma200.iloc[-1]-1)*100:+.2f}% vs its 200d SMA "
      f"-> above200d = {bool(above.iloc[-1])}")
print("  Most of the cell's history (2008, 2014-15, 2020) is crude collapsing inside a")
print("  DOWNTREND, where 'short the loser' is momentum. Today it is a pullback in an")
print("  uptrend, where the same trigger is a counter-trend short.")
for k in (5, 10):
    sp = spread(k)
    for lab, c in (("USO ABOVE 200d SMA  <-- TODAY", COND & above),
                   ("USO BELOW 200d SMA", COND & ~above)):
        s = sp[c & sp.notna()]
        e = s[C.declusterize(s.index, gap_td=10)] if len(s) else s
        print(f"  h{k} {lab:<30}: n={len(s):>3} avg={s.mean():+.3f}% "
              f"med={np.median(s) if len(s) else float('nan'):+.3f}% "
              f"hit={100*(s>0).mean():.1f}% t={C.tstat(s.values):+.2f} "
              f"worst={s.min():+.2f}% || eps n={len(e)} avg={e.mean():+.3f}% "
              f"t={C.tstat(e.values):+.2f}")
        if "ABOVE" in lab:
            print(f"       dates: {[str(d.date()) for d in s.index]}")


hdr("TODAY-MATCHED SUBSET -- crude collapses while ENERGY EQUITIES HOLD UP")
print(f"  2026-08-05 state: USO 5d {C.ret(uso,5).iloc[-1]:+.2f}%, XLE 5d "
      f"{C.ret(xle,5).iloc[-1]:+.2f}% -- XLE is barely down. Most of the cell's")
print("  history has BOTH legs collapsing together (2008, 2014-15, 2020).")
xle_ok = C.ret(xle, 5) > -4.0
print(f"  XLE 5d > -4% today: {bool(xle_ok.iloc[-1])}")
for k in (5, 10):
    sp = spread(k)
    m = COND & sp.notna() & xle_ok
    s = sp[m]
    e = s[C.declusterize(s.index, gap_td=10)] if len(s) else s
    print(f"  h{k}: n={len(s)} avg={s.mean():+.3f}% med={np.median(s):+.3f}% "
          f"hit={100*(s>0).mean():.1f}% t={C.tstat(s.values):+.2f} worst={s.min():+.2f}% "
          f"|| episodes n={len(e)} avg={e.mean():+.3f}% t={C.tstat(e.values):+.2f}")
    print(f"      dates: {[str(d.date()) for d in s.index]}")


hdr("SIGNAL DATE LIST (all firings, with h5 and h10 spread outcomes)")
sig5, sig10 = spread(5), spread(10)
dd = pd.DataFrame({"date": [d.date() for d in COND[COND].index],
                   "h5_%": sig5[COND].round(2).values,
                   "h10_%": sig10[COND].round(2).values})
print(dd.to_string(index=False))
