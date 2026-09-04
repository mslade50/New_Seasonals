"""G1 ADVERSARIAL CHECK -- STANDALONE LONG XLU after a 21d relative washout.

Candidate: buy XLU outright (no SPY short) when
    rel21 = XLU 21d return - SPY 21d return  <=  its trailing-756d q-th pctile

Provenance: k1 killed the LONG XLU / SHORT SPY pair. Its leg decomposition
showed XLU's OWN forward return in the washout cell beats XLU's unconditional
drift at every horizon, lift t=3.13 at h21 (all-days, close-to-close). This
check grades THAT, on the executable MOO-next-open basis, at episode level.

Attacks:
  (a) trigger honesty -- today's trailing-756d pctile of rel21; grid
      3/5/10/15/20 pctile x h 3/5/10/21; which fire TODAY.
  (b) THE LIFT -- cell vs XLU's unconditional MOO->MOC drift, at EPISODE
      level, with Welch + a random-date null + a YEAR-STRATIFIED null.
  (b2) MATCHED-ON-SPY test -- within SPY-21d-return deciles, does the XLU
      washout add anything, or is the cell just "the market has been strong,
      everything drifts up"?
  (b3) does the SPY-relative framing earn its place vs plain "XLU 21d return
      in its own bottom decile"?
  (c) era split at 2018 + regime buckets 2000-08 / 2009-19 / 2020-22 / 2023-26.
  (d) RATES -- dTNX21 rising vs falling (today = RISING); beta to TLT and dTNX.
  (e) 200d SMA -- above vs below (today = BELOW).
  (f) horizon honesty -- 3/5/10/21, which carries the lift.
  (g) worst window, worst year, year-by-year episodes, drop-best-episode, LOYO.
  (h) book overlap + fragility dial (>=50 today, P/C fear OFF -> dip-buy family
      ZEROED). Does the cell behave differently when the dial was >= 50?
  (i) cost / ex-div.

Run:  python g1_xlu_outright.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _common as C

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 300)
RNG = np.random.default_rng(42)

P = C.load(["XLU", "SPY", "TLT", "^TNX"])
XLU, SPY, TLT, TNX = P["XLU"], P["SPY"], P["TLT"], P["^TNX"]
IDX = XLU.index.intersection(SPY.index)
XLU, SPY = XLU.reindex(IDX), SPY.reindex(IDX)
xlu, spy = XLU["Close"], SPY["Close"]
tnx = TNX["Close"].reindex(IDX).ffill()
tlt = TLT["Close"].reindex(IDX)

HOR = (3, 5, 10, 21)
LB = 756

rel21 = C.ret(xlu, 21) - C.ret(spy, 21)
sma200 = xlu.rolling(200, min_periods=150).mean()
dtnx21 = tnx.diff(21)
spy21 = C.ret(spy, 21)
xlu21 = C.ret(xlu, 21)

# executable basis: enter MOO the session after the signal, exit MOC k later
FWD_MOO = {k: C.fwd_from_next_open(XLU, k) for k in HOR}
FWD_MOC = {k: C.fwd(xlu, k) for k in HOR}
FWD_SPY_MOO = {k: C.fwd_from_next_open(SPY, k) for k in HOR}
FWD_TLT_MOO = {k: C.fwd_from_next_open(TLT, k) for k in HOR}


def hdr(t: str) -> None:
    print("\n" + "=" * 108 + f"\n{t}\n" + "=" * 108)


def sub(t: str) -> None:
    print("\n" + "-" * 108 + f"\n{t}\n" + "-" * 108)


def thresh_series(q: float) -> pd.Series:
    return rel21.rolling(LB, min_periods=250).quantile(q)


def cond_at(q: float) -> pd.Series:
    return (rel21 <= thresh_series(q)) & rel21.notna()


# ===================================================================== (a)
hdr("(a) THE TRIGGER -- what is today's reading and which thresholds FIRE?")
today_rel = rel21.iloc[-1]
tr = rel21.rolling(LB, min_periods=250)
today_pctile = float((rel21.iloc[-LB:] <= today_rel).mean() * 100)
print(f"last usable bar                  : {IDX[-1].date()}  (pitch written pre-market 2026-08-06)")
print(f"XLU close                        : {xlu.iloc[-1]:.2f}")
print(f"XLU 21d ret                      : {xlu21.iloc[-1]:+.2f}%")
print(f"SPY 21d ret                      : {spy21.iloc[-1]:+.2f}%")
print(f"rel21 (XLU-SPY)                  : {today_rel:+.2f}pp")
print(f"trailing-{LB}d PERCENTILE of today's rel21 : {today_pctile:.2f}th")
print(f"XLU vs 200d SMA                  : {(xlu.iloc[-1]/sma200.iloc[-1]-1)*100:+.2f}%  "
      f"({'BELOW' if xlu.iloc[-1] < sma200.iloc[-1] else 'ABOVE'})")
print(f"^TNX 21d change                  : {dtnx21.iloc[-1]:+.3f}pp  "
      f"({'RISING' if dtnx21.iloc[-1] > 0 else 'FALLING'})")
print()

GRID_Q = (0.03, 0.05, 0.10, 0.15, 0.20)
CONDS = {q: cond_at(q) for q in GRID_Q}
fires = {}
for q in GRID_Q:
    c = CONDS[q]
    thr = thresh_series(q).iloc[-1]
    fires[q] = bool(c.iloc[-1])
    print(f"  rel21 <= {int(q*100):>2}th pctile(756d) -> today's thresh {thr:+.2f}pp | "
          f"today {today_rel:+.2f}pp | FIRES TODAY: {str(fires[q]):>5} | signal-days N={int(c.sum())}")
firing = [q for q in GRID_Q if fires[q]]
print(f"\n  >>> thresholds that FIRE today: {[f'{int(q*100)}th' for q in firing]}")
print(f"  >>> tightest firing threshold  : {int(min(firing)*100)}th pctile" if firing else "  >>> NOTHING FIRES")
print(f"  >>> knife edge? today {today_rel:+.2f}pp vs 10th-pctile thresh "
      f"{thresh_series(0.10).iloc[-1]:+.2f}pp -> margin "
      f"{today_rel - thresh_series(0.10).iloc[-1]:+.2f}pp "
      f"({'INSIDE' if today_rel <= thresh_series(0.10).iloc[-1] else 'OUTSIDE'})")

sub("(a2) FULL GRID: threshold x horizon, EXECUTABLE MOO basis, all-days + episodes")
rows = []
for q in GRID_Q:
    c = CONDS[q]
    for k in HOR:
        f = FWD_MOO[k]
        m = c & f.notna()
        s = f[m]
        if len(s) == 0:
            continue
        e10 = s[C.declusterize(s.index, gap_td=10)]
        e21 = s[C.declusterize(s.index, gap_td=21)]
        base = f.dropna()
        rows.append({
            "q": f"{int(q*100)}th", "fires": fires[q], "h": k, "n_days": len(s),
            "cell_avg": round(s.mean(), 3), "uncond": round(base.mean(), 3),
            "LIFT": round(s.mean() - base.mean(), 3),
            "t_days": round(C.tstat(s.values), 2),
            "n_ep10": len(e10), "ep10_avg": round(e10.mean(), 3),
            "ep10_LIFT": round(e10.mean() - base.mean(), 3),
            "ep10_t": round(C.tstat(e10.values), 2),
            "n_ep21": len(e21), "ep21_avg": round(e21.mean(), 3),
            "ep21_LIFT": round(e21.mean() - base.mean(), 3),
            "ep21_t": round(C.tstat(e21.values), 2),
        })
print(pd.DataFrame(rows).to_string(index=False))

# the honest trigger for today = loosest/tightest that fires
Q = min(firing) if firing else 0.10
COND = CONDS[Q]
QLAB = f"{int(Q*100)}th"
print(f"\n  >>> EVERYTHING BELOW USES q = {QLAB} pctile (the tightest form that fires today).")


# ===================================================================== (b)
hdr(f"(b) THE LIFT IS THE WHOLE CASE -- XLU cell vs XLU's own unconditional drift [{QLAB}]")


def welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    t = (a.mean() - b.mean()) / np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return float(t), float(df)


print("  NOTE ON THE BASELINE: XLU's unconditional forward return is measured on")
print("  OVERLAPPING daily windows, so its own SE is understated. The Welch t below")
print("  therefore FLATTERS the lift. The random-date and year-stratified nulls that")
print("  follow are the honest versions -- they resample the same overlap structure.\n")

rows = []
for k in HOR:
    f = FWD_MOO[k]
    base = f.dropna()
    # non-overlapping baseline: every k-th valid day
    base_nl = base.iloc[::k]
    s = f[COND & f.notna()]
    for lab, x in (("all-days", s),
                   ("episodes g10", s[C.declusterize(s.index, gap_td=10)]),
                   ("episodes g21", s[C.declusterize(s.index, gap_td=21)])):
        tw, dfw = welch(x.values, base.values)
        tn, _ = welch(x.values, base_nl.values)
        rows.append({"h": k, "cohort": lab, "n": len(x),
                     "cell_avg": round(x.mean(), 3), "uncond": round(base.mean(), 3),
                     "LIFT": round(x.mean() - base.mean(), 3),
                     "welch_t_vs_all": round(tw, 2),
                     "welch_t_vs_nonoverlap_base": round(tn, 2),
                     "hit%": round((x > 0).mean() * 100, 1),
                     "base_hit%": round((base > 0).mean() * 100, 1)})
print(pd.DataFrame(rows).to_string(index=False))

sub("(b) HONEST NULLS -- random-date and YEAR-STRATIFIED resampling (20k draws)")
print("  Random-date null: draw the same number of dates uniformly from the valid")
print("  sample -> distribution of the mean. Year-stratified null: draw the same")
print("  number of dates PER CALENDAR YEAR as the episodes have, which removes the")
print("  cell's year composition (the cell concentrates in strong-SPY periods).\n")
rows = []
for k in HOR:
    f = FWD_MOO[k]
    base = f.dropna()
    s = f[COND & f.notna()]
    for gap in (10, 21):
        ep = s[C.declusterize(s.index, gap_td=gap)]
        n = len(ep)
        obs = ep.mean()
        # uniform null
        draws = RNG.choice(base.values, size=(20000, n), replace=True).mean(axis=1)
        p_uni = float((draws >= obs).mean())
        # year-stratified null
        by_year = {y: base[base.index.year == y].values for y in sorted(set(base.index.year))}
        counts = ep.index.year.value_counts().to_dict()
        acc = np.zeros(20000)
        tot = 0
        for y, cnt in counts.items():
            pool = by_year.get(y)
            if pool is None or len(pool) == 0:
                continue
            acc += RNG.choice(pool, size=(20000, cnt), replace=True).sum(axis=1)
            tot += cnt
        strat = acc / max(tot, 1)
        p_str = float((strat >= obs).mean())
        rows.append({"h": k, "gap": gap, "n_ep": n, "obs_avg": round(obs, 3),
                     "uniform_null_mean": round(draws.mean(), 3),
                     "p_uniform(>=obs)": round(p_uni, 4),
                     "yearstrat_null_mean": round(strat.mean(), 3),
                     "LIFT_vs_yearstrat": round(obs - strat.mean(), 3),
                     "p_yearstrat(>=obs)": round(p_str, 4)})
print(pd.DataFrame(rows).to_string(index=False))


# ==================================================================== (b2)
sub("(b2) MATCHED ON SPY 21d RETURN -- is the cell just 'the market has been strong'?")
print("  rel21 is low mostly because SPY is UP. Within SPY-21d-return DECILES,")
print("  compare XLU forward (MOO) in-cell vs out-of-cell. If the matched lift")
print("  collapses, the 'washout' is a proxy for bull-tape composition.\n")
print(f"  in-cell SPY 21d ret mean: {spy21[COND].mean():+.2f}%  vs uncond {spy21.dropna().mean():+.2f}%")
print(f"  in-cell XLU 21d ret mean: {xlu21[COND].mean():+.2f}%  vs uncond {xlu21.dropna().mean():+.2f}%\n")
for k in HOR:
    f = FWD_MOO[k]
    d = pd.DataFrame({"f": f, "spy21": spy21, "cell": COND}).dropna()
    d["dec"] = pd.qcut(d["spy21"], 10, labels=False, duplicates="drop")
    rows, wnum, wden = [], 0.0, 0
    for dec, g in d.groupby("dec"):
        a, b = g[g["cell"]]["f"], g[~g["cell"]]["f"]
        if len(a) < 5 or len(b) < 5:
            continue
        rows.append({"spy21_decile": int(dec),
                     "spy21_lo": round(g["spy21"].min(), 2), "spy21_hi": round(g["spy21"].max(), 2),
                     "n_cell": len(a), "cell_avg": round(a.mean(), 3),
                     "n_out": len(b), "out_avg": round(b.mean(), 3),
                     "matched_lift": round(a.mean() - b.mean(), 3)})
        wnum += (a.mean() - b.mean()) * len(a)
        wden += len(a)
    print(f"  --- h{k} (MOO basis) ---")
    print(pd.DataFrame(rows).to_string(index=False))
    raw = d[d["cell"]]["f"].mean() - d["f"].mean()
    print(f"    RAW lift = {raw:+.3f}%   |   SPY-21d-MATCHED lift = {wnum/max(wden,1):+.3f}%  "
          f"(retains {100*(wnum/max(wden,1))/raw:.0f}% of raw)" if abs(raw) > 1e-9 else "")
    print()


# ==================================================================== (b3)
sub("(b3) DOES THE SPY-RELATIVE FRAMING EARN ITS PLACE? vs plain XLU-own-washout")
own = (xlu21 <= xlu21.rolling(LB, min_periods=250).quantile(Q)) & xlu21.notna()
print(f"  plain 'XLU 21d ret <= {QLAB} pctile(756d)' fires today: {bool(own.iloc[-1])} | N={int(own.sum())}")
rows = []
for k in HOR:
    f = FWD_MOO[k]
    base = f.dropna()
    for lab, c in (("REL (XLU-SPY)", COND), ("OWN (XLU only)", own), ("BOTH", COND & own)):
        s = f[c & f.notna()]
        if len(s) < 5:
            continue
        e = s[C.declusterize(s.index, gap_td=10)]
        rows.append({"h": k, "trigger": lab, "n_days": len(s),
                     "avg": round(s.mean(), 3), "LIFT": round(s.mean() - base.mean(), 3),
                     "t_days": round(C.tstat(s.values), 2),
                     "n_ep10": len(e), "ep_avg": round(e.mean(), 3),
                     "ep_LIFT": round(e.mean() - base.mean(), 3),
                     "ep_t": round(C.tstat(e.values), 2)})
print(pd.DataFrame(rows).to_string(index=False))


# ===================================================================== (c)
hdr(f"(c) ERA STABILITY [{QLAB}] -- 2018 split + regime buckets (MOO basis)")
BUCKETS = {"2000-2008": ("2000-01-01", "2009-01-01"),
           "2009-2019": ("2009-01-01", "2020-01-01"),
           "2020-2022": ("2020-01-01", "2023-01-01"),
           "2023-2026": ("2023-01-01", "2027-01-01")}
for k in HOR:
    f = FWD_MOO[k]
    base = f.dropna()
    s = f[COND & f.notna()]
    e = s[C.declusterize(s.index, gap_td=10)]
    print(f"\n  --- h{k} MOO ---")
    print("  2018 split (all-days):")
    C.show(C.era_split(s.index, s.values, cut="2018-01-01"))
    print("  2018 split (episodes gap10):")
    C.show(C.era_split(e.index, e.values, cut="2018-01-01"))
    rows = []
    for name, (a, b) in BUCKETS.items():
        ss = s[(s.index >= a) & (s.index < b)]
        ee = e[(e.index >= a) & (e.index < b)]
        bb = base[(base.index >= a) & (base.index < b)]
        rows.append({"bucket": name, "n_days": len(ss),
                     "cell_avg": round(ss.mean(), 3) if len(ss) else np.nan,
                     "era_uncond": round(bb.mean(), 3) if len(bb) else np.nan,
                     "LIFT": round(ss.mean() - bb.mean(), 3) if len(ss) and len(bb) else np.nan,
                     "t_days": round(C.tstat(ss.values), 2) if len(ss) else np.nan,
                     "n_ep": len(ee),
                     "ep_avg": round(ee.mean(), 3) if len(ee) else np.nan,
                     "ep_LIFT": round(ee.mean() - bb.mean(), 3) if len(ee) and len(bb) else np.nan,
                     "ep_t": round(C.tstat(ee.values), 2) if len(ee) > 1 else np.nan})
    print("  regime buckets (LIFT is vs that era's OWN unconditional XLU drift):")
    print(pd.DataFrame(rows).to_string(index=False))
    post = e[e.index >= "2022-01-01"]
    print(f"    POST-2022 only (the rates-regime break): n_ep={len(post)} "
          f"avg={post.mean():+.3f}% t={C.tstat(post.values):+.2f}"
          if len(post) > 1 else "    POST-2022: too few episodes")


# ===================================================================== (d)
hdr(f"(d) RATES -- utilities as a duration proxy [{QLAB}]. TODAY IS THE RISING SUBSET.")
for k in HOR:
    f = FWD_MOO[k]
    base = f.dropna()
    rows = []
    for name, mm in (("dTNX21 > 0 RISING  <-- TODAY", COND & (dtnx21 > 0)),
                     ("dTNX21 <= 0 FALLING", COND & (dtnx21 <= 0))):
        s = f[mm & f.notna()]
        if len(s) < 3:
            continue
        e10 = s[C.declusterize(s.index, gap_td=10)]
        e21 = s[C.declusterize(s.index, gap_td=21)]
        bs = base[base.index.isin(dtnx21[(dtnx21 > 0) if "RISING" in name else (dtnx21 <= 0)].index)]
        rows.append({"h": k, "subset": name, "n_days": len(s),
                     "cell_avg": round(s.mean(), 3),
                     "uncond_same_rate_state": round(bs.mean(), 3),
                     "LIFT": round(s.mean() - bs.mean(), 3),
                     "t_days": round(C.tstat(s.values), 2),
                     "n_ep10": len(e10), "ep10_avg": round(e10.mean(), 3),
                     "ep10_t": round(C.tstat(e10.values), 2),
                     "n_ep21": len(e21), "ep21_avg": round(e21.mean(), 3),
                     "ep21_t": round(C.tstat(e21.values), 2)})
    print(pd.DataFrame(rows).to_string(index=False))
    print()

sub("(d2) BETA of the cell's XLU forward return to TLT and to dTNX (forward change)")
rows = []
for k in HOR:
    f, ft = FWD_MOO[k], FWD_TLT_MOO[k]
    dtnx_fwd = (tnx.shift(-k) - tnx.shift(-1))  # rate change over the hold
    d = pd.DataFrame({"x": f, "tlt": ft, "dtnx": dtnx_fwd, "cell": COND}).dropna()
    dc = d[d["cell"]]
    b_tlt = np.polyfit(dc["tlt"], dc["x"], 1)[0]
    b_tnx = np.polyfit(dc["dtnx"], dc["x"], 1)[0]
    bu_tlt = np.polyfit(d["tlt"], d["x"], 1)[0]
    bu_tnx = np.polyfit(d["dtnx"], d["x"], 1)[0]
    rows.append({"h": k, "n_cell": len(dc),
                 "beta_TLT_incell": round(b_tlt, 3), "beta_TLT_uncond": round(bu_tlt, 3),
                 "corr_TLT_incell": round(dc["x"].corr(dc["tlt"]), 3),
                 "beta_per_1bp_TNX_incell": round(b_tnx / 100, 4),
                 "beta_per_1bp_TNX_uncond": round(bu_tnx / 100, 4),
                 "corr_dTNX_incell": round(dc["x"].corr(dc["dtnx"]), 3)})
print(pd.DataFrame(rows).to_string(index=False))
print("  (beta_per_1bp = % XLU move per 1bp move in the 10y over the hold window)")


# ===================================================================== (e)
hdr(f"(e) 200d SMA -- washout BELOW trend vs pullback IN an uptrend. TODAY IS BELOW.")
above = xlu > sma200
for k in HOR:
    f = FWD_MOO[k]
    rows = []
    for name, mm in (("XLU BELOW 200d  <-- TODAY", COND & ~above & sma200.notna()),
                     ("XLU ABOVE 200d", COND & above & sma200.notna())):
        s = f[mm & f.notna()]
        if len(s) < 3:
            continue
        bs = f[(~above if "BELOW" in name else above) & sma200.notna() & f.notna()]
        e10 = s[C.declusterize(s.index, gap_td=10)]
        e21 = s[C.declusterize(s.index, gap_td=21)]
        rows.append({"h": k, "subset": name, "n_days": len(s),
                     "cell_avg": round(s.mean(), 3),
                     "uncond_same_trend_state": round(bs.mean(), 3),
                     "LIFT": round(s.mean() - bs.mean(), 3),
                     "t_days": round(C.tstat(s.values), 2), "worst": round(s.min(), 2),
                     "n_ep10": len(e10), "ep10_avg": round(e10.mean(), 3),
                     "ep10_t": round(C.tstat(e10.values), 2),
                     "n_ep21": len(e21), "ep21_avg": round(e21.mean(), 3),
                     "ep21_t": round(C.tstat(e21.values), 2)})
    print(pd.DataFrame(rows).to_string(index=False))
    print()

sub("(e2) TODAY-MATCHED TRIPLE CELL: rel21 washout AND below 200d AND rates rising")
print("  WARNING: this is a third/fourth conditioning layer chosen AFTER seeing the")
print("  base cell -- report it as specification search, not as evidence.\n")
trip = COND & ~above & sma200.notna() & (dtnx21 > 0)
for k in HOR:
    f = FWD_MOO[k]
    s = f[trip & f.notna()]
    if len(s) < 3:
        print(f"  h{k}: n={len(s)} -- too thin")
        continue
    e10 = s[C.declusterize(s.index, gap_td=10)]
    e21 = s[C.declusterize(s.index, gap_td=21)]
    base = f.dropna()
    print(f"  h{k}: n_days={len(s)} avg={s.mean():+.3f}% (uncond {base.mean():+.3f}%, "
          f"LIFT {s.mean()-base.mean():+.3f}%) hit={100*(s>0).mean():.1f}% t={C.tstat(s.values):+.2f} "
          f"worst={s.min():+.2f}% || ep10 n={len(e10)} avg={e10.mean():+.3f}% t={C.tstat(e10.values):+.2f} "
          f"|| ep21 n={len(e21)} avg={e21.mean():+.3f}% t={C.tstat(e21.values):+.2f}")
ep_t = None
s21 = FWD_MOO[21][trip & FWD_MOO[21].notna()]
if len(s21):
    e = s21[C.declusterize(s21.index, gap_td=21)]
    print(f"\n  triple-cell h21 episode dates (gap21): {[str(d.date()) for d in e.index]}")
    print(f"  triple-cell h21 episode values: {[round(v,2) for v in e.values]}")


# ===================================================================== (f)
hdr(f"(f) HORIZON HONESTY [{QLAB}] -- MOC-at-signal vs EXECUTABLE MOO-next-open")
rows = []
for k in HOR:
    a, b = FWD_MOC[k], FWD_MOO[k]
    sa, sb = a[COND & a.notna()], b[COND & b.notna()]
    ea = sa[C.declusterize(sa.index, gap_td=21)]
    eb = sb[C.declusterize(sb.index, gap_td=21)]
    rows.append({"h": k,
                 "MOC_days_avg": round(sa.mean(), 3), "MOC_days_LIFT": round(sa.mean() - a.dropna().mean(), 3),
                 "MOC_days_t": round(C.tstat(sa.values), 2),
                 "MOO_days_avg": round(sb.mean(), 3), "MOO_days_LIFT": round(sb.mean() - b.dropna().mean(), 3),
                 "MOO_days_t": round(C.tstat(sb.values), 2),
                 "MOO_ep21_n": len(eb), "MOO_ep21_avg": round(eb.mean(), 3),
                 "MOO_ep21_LIFT": round(eb.mean() - b.dropna().mean(), 3),
                 "MOO_ep21_t": round(C.tstat(eb.values), 2)})
print(pd.DataFrame(rows).to_string(index=False))
on = (XLU["Open"].shift(-1) / xlu - 1) * 100
print(f"\n  overnight signal-close -> next open: in-cell {on[COND].mean():+.3f}% "
      f"(uncond {on.dropna().mean():+.3f}%) -> MOO entry gives up "
      f"{on[COND].mean()-on.dropna().mean():+.3f}pp vs a close entry")


# ===================================================================== (g)
hdr(f"(g) WORST WINDOW / WORST YEAR / YEAR-BY-YEAR EPISODES / DROP-BEST / LOYO [{QLAB}]")
for k in HOR:
    f = FWD_MOO[k]
    s = f[COND & f.notna()]
    print(f"\n  --- h{k} MOO ---")
    print(f"  worst window {s.min():+.2f}% on {s.idxmin().date()} | best {s.max():+.2f}% on {s.idxmax().date()}")
    for gap in (10, 21):
        e = s[C.declusterize(s.index, gap_td=gap)]
        nb = e.drop(e.idxmax())
        print(f"  gap={gap:>2}td: n_ep={len(e):>3} avg={e.mean():+.3f}% med={np.median(e):+.3f}% "
              f"hit={100*(e>0).mean():.1f}% t={C.tstat(e.values):+.2f} worst={e.min():+.2f}% || "
              f"drop-best avg={nb.mean():+.3f}% t={C.tstat(nb.values):+.2f}")
    e = s[C.declusterize(s.index, gap_td=21)]
    yrs = sorted(set(e.index.year))
    loyo = [(y, C.tstat(e[e.index.year != y].values), e[e.index.year != y].mean()) for y in yrs]
    fin = [(y, t, m) for y, t, m in loyo if np.isfinite(t)]
    lo = min(fin, key=lambda z: z[1])
    hi = max(fin, key=lambda z: z[1])
    print(f"  LOYO (episodes gap21, {len(yrs)} years): FLOOR t={lo[1]:+.2f} (drop {lo[0]}, avg {lo[2]:+.3f}%) | "
          f"ceiling t={hi[1]:+.2f} (drop {hi[0]})")
    if len(e) > 3:
        bs = RNG.choice(e.values, size=(20000, len(e)), replace=True).mean(axis=1)
        print(f"  episode bootstrap (gap21): P(mean<=0)={(bs<=0).mean():.4f}  5th pct of mean={np.percentile(bs,5):+.3f}%")

sub("year-by-year EPISODE table (gap 21td), h10 and h21 MOO")
for k in (10, 21):
    f = FWD_MOO[k]
    s = f[COND & f.notna()]
    e = s[C.declusterize(s.index, gap_td=21)]
    t = pd.DataFrame({"ret": e.values}, index=e.index).groupby(lambda d: d.year)["ret"].agg(
        ["count", "mean", "min", "max", "sum"]).round(2)
    t.columns = ["n_ep", "avg", "worst", "best", "sum"]
    print(f"\n  h{k}:")
    print(t.to_string())
    neg = sorted(t.index[t["avg"] < 0].tolist())
    print(f"    NEGATIVE-avg years: {neg}  ({len(neg)} of {len(t)})")
    print(f"    worst year by avg: {t['avg'].idxmin()} ({t['avg'].min():+.2f}%)")

sub("ALL episodes (gap 21td) with h21 MOO returns -- last 30")
f21 = FWD_MOO[21]
s21 = f21[COND & f21.notna()]
e21 = s21[C.declusterize(s21.index, gap_td=21)]
d = pd.DataFrame({"episode": [x.date() for x in e21.index],
                  "h21_MOO_%": e21.round(2).values,
                  "dTNX21": dtnx21.reindex(e21.index).round(3).values,
                  "vs200d_%": ((xlu / sma200 - 1) * 100).reindex(e21.index).round(2).values})
print(d.tail(30).to_string(index=False))


# ===================================================================== (h)
hdr("(h) BOOK OVERLAP + FRAGILITY DIAL")
frag = pd.read_parquet(C.ROOT / "data" / "rd2_fragility.parquet")
frag.index = pd.DatetimeIndex(frag.index).normalize()
frag = frag[frag.index < C.ASOF_EXCL]
ma10 = frag["63d"].rolling(10, min_periods=10).mean()
print(f"  rd2_fragility.parquet: {frag.index[0].date()} -> {frag.index[-1].date()}, rows={len(frag)}")
print(f"  TODAY (2026-08-05): 63d dial {frag['63d'].iloc[-1]:.1f}, 10d-MA(63d) = {ma10.iloc[-1]:.2f} "
      f"-> {'ABOVE' if ma10.iloc[-1] >= 50 else 'below'} the book's 50 threshold")
print("  VINTAGE CAVEAT: rows before 2026-07-02 are a RECOMPUTE vintage (CLAUDE.md:")
print("  drifted up to ~7 pts vs the point-in-time series). Pre-2026-07 dial history")
print("  below is INDICATIVE ONLY -- it cannot be treated as what the book saw.\n")

ma_al = ma10.reindex(IDX)
for k in HOR:
    f = FWD_MOO[k]
    rows = []
    for name, mm in ((">=50 dial  <-- TODAY", COND & (ma_al >= 50)),
                     ("<50 dial", COND & (ma_al < 50))):
        s = f[mm & f.notna()]
        if len(s) < 3:
            rows.append({"h": k, "dial": name, "n_days": len(s)})
            continue
        e = s[C.declusterize(s.index, gap_td=21)]
        bs = f[(ma_al >= 50 if ">=50" in name else ma_al < 50) & f.notna()]
        rows.append({"h": k, "dial": name, "n_days": len(s), "cell_avg": round(s.mean(), 3),
                     "uncond_same_dial": round(bs.mean(), 3),
                     "LIFT": round(s.mean() - bs.mean(), 3),
                     "t_days": round(C.tstat(s.values), 2), "worst": round(s.min(), 2),
                     "n_ep21": len(e), "ep_avg": round(e.mean(), 3),
                     "ep_t": round(C.tstat(e.values), 2)})
    print(pd.DataFrame(rows).to_string(index=False))
    print()

print("  BOOK POLICY THIS MORNING (CLAUDE.md 'P/C Fear-Conditioned Family Bands'):")
print("  dial 10d-MA(63d) = 54.7 >= 50 AND equity P/C fear state OFF -> PC_FEAR_BANDS")
print("  selects the fear-OFF table [[0,50,1.0],[50,999,0.0]], which ZEROES all six")
print("  frag_risk_bands carriers (FAMILY4: Weak Close Decent Sznls, SPY QQQ MonFri")
print("  Reversion, Monday Dip, Indices Oversold Bounce; plus 3x Bear ETF Overbot")
print("  Fade and Monthly Weak Close). Those are the book's DIP-BUY family. A")
print("  discretionary sector dip-buy this morning is the same trade the book is")
print("  deliberately switched off from taking, at the same dial reading, on a")
print("  shipped-live policy McKinley signed off on 2026-08-05.")
print("  Also live and unaffected: OLV / LT Trend ST OS / Sector BO can still fire on")
print("  XLU itself, so a manual XLU long can double up on a scanner leg.")


# ===================================================================== (i)
hdr("(i) COST / DIVIDEND")
print("  XLU: ~$18bn Select Sector SPDR, penny-wide, 1-2 bps round trip -- immaterial")
print("  vs the measured lift only if the lift is >= ~0.10%. Compare below.")
print("  master_prices is DIVIDEND-ADJUSTED -> every number here is TOTAL RETURN,")
print("  so the ~3% yield is already inside the measured drift (it is NOT extra edge).")
print("  XLU ex-div convention: quarterly, ~3rd Friday of MAR/JUN/SEP/DEC. Next")
print("  expected ~2026-09-18. Entry 2026-08-06:")
for k in HOR:
    exit_d = (pd.Timestamp("2026-08-06") + pd.tseries.offsets.BDay(k)).date()
    print(f"    h{k:>2} -> exit ~{exit_d}  -> {'NO ex-div in window' if k <= 21 else 'CHECK'}")
print("  (h21 exit ~2026-09-04 is still before ~2026-09-18, so all four horizons are")
print("   clear of the ex-date. Convention, not measured from the adjusted cache.)")
rows = []
for k in HOR:
    f = FWD_MOO[k]
    s = f[COND & f.notna()]
    e = s[C.declusterize(s.index, gap_td=21)]
    base = f.dropna()
    lift = e.mean() - base.mean()
    rows.append({"h": k, "ep21_LIFT_%": round(lift, 3), "round_trip_cost_%": 0.02,
                 "net_LIFT_%": round(lift - 0.02, 3),
                 "cost_as_%_of_lift": round(0.02 / lift * 100, 1) if lift > 0 else np.nan})
print(pd.DataFrame(rows).to_string(index=False))


hdr("(j) EVEN-HANDED: is there ANY honest threshold that both FIRES today and carries the lift?")
print(f"  Today's rel21 sits at the {today_pctile:.2f}th pctile of its trailing 756d.")
print("  The tight cells (3rd/5th) carry the lift but DO NOT fire. Scan the fine grid")
print("  to find the tightest firing threshold and see whether the lift survives there.\n")
rows = []
for q in (0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12):
    c = cond_at(q)
    f = FWD_MOO[21]
    base = f.dropna()
    s = f[c & f.notna()]
    e10 = s[C.declusterize(s.index, gap_td=10)]
    e21 = s[C.declusterize(s.index, gap_td=21)]
    tw, _ = welch(e21.values, base.values)
    rows.append({"q": f"{q*100:.0f}th", "fires_today": bool(c.iloc[-1]),
                 "thresh_pp": round(thresh_series(q).iloc[-1], 2),
                 "n_days": len(s), "n_ep10": len(e10), "n_ep21": len(e21),
                 "h21_ep21_avg": round(e21.mean(), 3),
                 "h21_uncond": round(base.mean(), 3),
                 "h21_ep21_LIFT": round(e21.mean() - base.mean(), 3),
                 "welch_LIFT_t": round(tw, 2),
                 "ep21_t": round(C.tstat(e21.values), 2)})
print(pd.DataFrame(rows).to_string(index=False))

sub("(j2) is a DEEPER washout better? continuous grading inside the 20th-pctile cell")
c20 = CONDS[0.20]
rk = rel21.rolling(LB, min_periods=250).rank(pct=True) * 100.0
for k in (10, 21):
    f = FWD_MOO[k]
    s = f[c20 & f.notna()]
    e = s[C.declusterize(s.index, gap_td=21)]
    r = rk.reindex(e.index)
    d = pd.DataFrame({"ret": e.values, "pct": r.values}).dropna()
    d["bucket"] = pd.cut(d["pct"], [0, 3, 6, 10, 15, 21],
                         labels=["0-3", "3-6", "6-10", "10-15", "15-21"])
    g = d.groupby("bucket", observed=True)["ret"].agg(["count", "mean", "median"]).round(3)
    g["t"] = d.groupby("bucket", observed=True)["ret"].apply(lambda v: round(C.tstat(v.values), 2))
    print(f"\n  h{k} MOO, episodes gap21, by rel21 trailing-756d percentile bucket "
          f"(TODAY = {today_pctile:.1f}th -> bucket '3-6'):")
    print(g.to_string())
    if len(d) > 3:
        b, a = np.polyfit(d["pct"], d["ret"], 1)
        print(f"    slope of return on percentile: {b:+.4f}%/pctile  "
              f"(corr {d['ret'].corr(d['pct']):+.3f}) -- negative slope = deeper is better")


hdr("(k) GRADE THE TRIPLE CELL PROPERLY -- it is the only sub-cell showing life")
print("  Cell: rel21 <= 10th pctile AND XLU below its 200d AND dTNX21 > 0.")
print("  The fair test is NOT vs XLU's overall drift -- it is vs XLU's drift in the")
print("  SAME state (below 200d AND rates rising), which is a weak-XLU state by")
print("  construction. Anything left after that is the actual claim.\n")
same_state = (~above) & sma200.notna() & (dtnx21 > 0)
rows = []
for k in HOR:
    f = FWD_MOO[k]
    s = f[trip & f.notna()]
    bs = f[same_state & f.notna() & ~trip]
    for gap in (10, 21):
        e = s[C.declusterize(s.index, gap_td=gap)]
        tw, _ = welch(e.values, bs.values)
        nb = e.drop(e.idxmax())
        rows.append({"h": k, "gap": gap, "n_ep": len(e),
                     "ep_avg": round(e.mean(), 3),
                     "same_state_uncond": round(bs.mean(), 3),
                     "LIFT_vs_same_state": round(e.mean() - bs.mean(), 3),
                     "welch_LIFT_t": round(tw, 2),
                     "ep_t_vs_zero": round(C.tstat(e.values), 2),
                     "hit%": round((e > 0).mean() * 100, 1),
                     "worst": round(e.min(), 2),
                     "dropbest_avg": round(nb.mean(), 3),
                     "dropbest_t": round(C.tstat(nb.values), 2)})
print(pd.DataFrame(rows).to_string(index=False))

sub("(k2) triple cell: year-stratified null, LOYO, era, dial")
base21 = FWD_MOO[21].dropna()
for k in (10, 21):
    f = FWD_MOO[k]
    s = f[trip & f.notna()]
    e = s[C.declusterize(s.index, gap_td=21)]
    base = f.dropna()
    by_year = {y: base[base.index.year == y].values for y in sorted(set(base.index.year))}
    counts = e.index.year.value_counts().to_dict()
    acc, tot = np.zeros(20000), 0
    for y, cnt in counts.items():
        pool = by_year.get(y)
        if pool is None or len(pool) == 0:
            continue
        acc += RNG.choice(pool, size=(20000, cnt), replace=True).sum(axis=1)
        tot += cnt
    strat = acc / max(tot, 1)
    yrs = sorted(set(e.index.year))
    loyo = [(y, C.tstat(e[e.index.year != y].values), e[e.index.year != y].mean()) for y in yrs]
    fin = [(y, t, m) for y, t, m in loyo if np.isfinite(t)]
    lo = min(fin, key=lambda z: z[1])
    bs = RNG.choice(e.values, size=(20000, len(e)), replace=True).mean(axis=1)
    print(f"\n  h{k} triple cell, episodes gap21: n={len(e)} avg={e.mean():+.3f}% t={C.tstat(e.values):+.2f}")
    print(f"    year-stratified null mean={strat.mean():+.3f}% -> LIFT {e.mean()-strat.mean():+.3f}% "
          f"p(null>=obs)={float((strat>=e.mean()).mean()):.4f}")
    print(f"    LOYO ({len(yrs)} yrs): FLOOR t={lo[1]:+.2f} (drop {lo[0]}, avg {lo[2]:+.3f}%)")
    print(f"    bootstrap P(mean<=0)={(bs<=0).mean():.4f}  5th pct of mean={np.percentile(bs,5):+.3f}%")
    pre = e[e.index < "2018-01-01"]; post = e[e.index >= "2018-01-01"]
    print(f"    pre-2018 n={len(pre)} avg={pre.mean():+.3f}% t={C.tstat(pre.values):+.2f} | "
          f"2018+ n={len(post)} avg={post.mean():+.3f}% t={C.tstat(post.values):+.2f}")
    p22 = e[e.index >= "2022-01-01"]
    print(f"    2022+ (post rates break) n={len(p22)} avg={p22.mean():+.3f}% t={C.tstat(p22.values):+.2f} "
          f"values={[round(v,2) for v in p22.values]}")
    md = ma_al.reindex(e.index)
    hi = e[md >= 50]
    print(f"    of those, dial 10dMA63 >= 50 (today's state, recompute vintage): n={len(hi)} "
          f"avg={hi.mean():+.3f}%" if len(hi) else "    dial >=50 sub-cell: n=0 -- NO precedent at today's dial")


hdr("SUMMARY NUMBERS FOR THE VERDICT")
for k in HOR:
    f = FWD_MOO[k]
    base = f.dropna()
    s = f[COND & f.notna()]
    e10 = s[C.declusterize(s.index, gap_td=10)]
    e21 = s[C.declusterize(s.index, gap_td=21)]
    tw10, _ = welch(e10.values, base.values)
    tw21, _ = welch(e21.values, base.values)
    print(f"  h{k:>2} MOO: days n={len(s)} avg={s.mean():+.3f}% t={C.tstat(s.values):+.2f} | "
          f"uncond {base.mean():+.3f}% | ep10 n={len(e10)} avg={e10.mean():+.3f}% t={C.tstat(e10.values):+.2f} "
          f"welchLIFT_t={tw10:+.2f} | ep21 n={len(e21)} avg={e21.mean():+.3f}% t={C.tstat(e21.values):+.2f} "
          f"welchLIFT_t={tw21:+.2f}")
