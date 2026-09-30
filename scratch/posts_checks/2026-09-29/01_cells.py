"""Posts check (2026-09-29, Tuesday): stat and idea candidates for tonight's queue.

ASOF = Tuesday 2026-09-29 (freshest bar). Next session Wednesday 2026-09-30 is the
final session of September and of Q3. Thursday 10/1 opens Q4; NFP Friday 10/2.
Every tradeable return is LAG-1 (next open) or a MOC entry whose condition is known
before that close; close-to-close is reported alongside. SPY calendar from the cache
is the master index. September 2026 is incomplete and is EXCLUDED from every
month-end sample (engine month-end defect).

A. TLT / IEF on a month's final session (c2c and o2c), quarter-end split.
B. SPY on a month's final session vs all other sessions.
C. TLT sixth consecutive lower close (one anchor per run).
D. ^TNX level and up-run facts; TLT lowest-since.
E. HYG volume spike vs prior 63-session mean; HYG sixth lower close.
F. IDEA CANDIDATE: short TLT MOC on a bad month's final session, out at the close of
   the next month's second session. Try to kill it.
G. NFP k3 cell and its position / weekday x position controls.
H. Open idea marks at the 9/29 close.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pitch_lab import (  # noqa: E402
    anchor_positions, cluster_note, declusters, load_events, load_prices,
    local_control, sign_test, summarize, wilder_atr,
)

ROOT = Path(__file__).resolve().parents[3]
ASOF = pd.Timestamp("2026-09-29")
ERA = pd.Timestamp("2018-01-01")
CUR = 202609
ETF = ["SPY", "TLT", "IEF", "HYG", "XLE", "XLU"]
TK = ETF + ["^TNX"]
# 9/28 bars as printed by scratch/posts_checks/2026-09-28/out_01.txt (O, H, L, C)
PREV0928 = {"SPY": (768.35, 769.54, 763.715, 765.61), "TLT": (78.645, 78.8696, 78.2698, 78.62),
            "IEF": (89.51, 89.685, 89.29, 89.53), "XLE": (62.77, 62.7865, 61.83, 62.10),
            "XLU": (39.485, 39.545, 39.06, 39.25)}

raw = load_prices(TK)
print("=== data check (cache) ===")
for t in TK:
    if t not in raw:
        print(f"  {t}: MISSING")
        continue
    f = raw[t]
    f = f[f.index <= ASOF]
    bad_open = int(((f["Open"] <= 0) | f["Open"].isna()).sum())
    eq_oc = int((f["Open"] == f["Close"]).sum())
    flat = int(((f["Open"] == f["Close"]) & (f["High"] == f["Low"])).sum())
    lb = f.iloc[-1]
    print(f"  {t}: {f.index[0].date()} .. {f.index[-1].date()} n={len(f)} "
          f"last O/H/L/C {lb['Open']:.4f}/{lb['High']:.4f}/{lb['Low']:.4f}/{lb['Close']:.4f} "
          f"vol {lb['Volume']:.0f} | Open<=0/NaN {bad_open} | Open==Close {eq_oc} "
          f"| flat O=C,H=L {flat}")
print("  9/28 bar now vs the 09-28 run (O/H/L/C), and 9/29 volume roundness (snapshot tell):")
for t in ETF:
    f = raw[t]
    b = f.loc["2026-09-28"]
    now = (float(b["Open"]), float(b["High"]), float(b["Low"]), float(b["Close"]))
    was = PREV0928.get(t)
    chg = "" if was is None else " | changed: " + ", ".join(
        f"{k} {w:.4f}->{n:.4f}" for k, w, n in zip("OHLC", was, now) if abs(w - n) > 5e-4)
    v28, v29 = float(f.loc["2026-09-28", "Volume"]), float(f.loc["2026-09-29", "Volume"])
    print(f"    {t}: 9/28 now {now[0]:.4f}/{now[1]:.4f}/{now[2]:.4f}/{now[3]:.4f}{chg or ' | unchanged'}"
          f" | vol 9/28 {v28:.0f} (x100 {v28 % 100 == 0}) 9/29 {v29:.0f} (x100 {v29 % 100 == 0})")

nyse = raw["SPY"].index
nyse = pd.DatetimeIndex(nyse[nyse <= ASOF])
POS = pd.Series(np.arange(len(nyse)), index=nyse)


def frame(t: str) -> pd.DataFrame:
    f = raw[t]
    f = f[f.index <= ASOF].astype(float)
    miss = nyse[(nyse >= f.index[0])].difference(f.index)
    if len(miss):
        print(f"  NOTE {t}: {len(miss)} SPY-calendar sessions missing after its start "
              f"(first {[str(d.date()) for d in miss[:5]]})")
    return f.reindex(nyse)


F = {t: frame(t) for t in ETF}


def atr_series(t: str) -> pd.Series:
    f = raw[t]
    f = f[f.index <= ASOF].dropna(subset=["High", "Low", "Close"]).astype(float)
    a = wilder_atr(f["High"].to_numpy(), f["Low"].to_numpy(), f["Close"].to_numpy())
    return pd.Series(np.asarray(a, float), index=f.index).reindex(nyse)


ATR = {t: atr_series(t) for t in ("TLT", "HYG", "IEF")}
R1 = {t: F[t]["Close"] / F[t]["Close"].shift(1) - 1.0 for t in ETF}
print("  tonight 1d: " + ", ".join(f"{t} {100*R1[t].iloc[-1]:+.2f}%" for t in ETF))

# month position on the SPY calendar; the current (incomplete) month gets no from-end count
YM = np.asarray(nyse.year * 100 + nyse.month)
PER = pd.Series(YM, index=nyse)
FE = PER.groupby(PER.values).cumcount(ascending=False).astype(float)
FS = PER.groupby(PER.values).cumcount() + 1
FE[PER.values == CUR] = np.nan
try:
    from trading_calendar import TRADING_DAY
    nxt = [ASOF + k * TRADING_DAY for k in range(1, 5)]
    print(f"  next sessions per trading_calendar: {[str(d.date()) for d in nxt]} | Sept "
          f"sessions left after 9/29: {sum(d.month == 9 for d in nxt)} (expect 1)")
except Exception as exc:  # noqa: BLE001
    print(f"  trading_calendar import failed {exc!r}")


# ---------------------------------------------------------------------------
# helpers (template 2026-09-28)
# ---------------------------------------------------------------------------
def o2c_in(t: str, h: int) -> pd.Series:
    """Open[D+1] -> Close[D+h] (entry session counted as session 1)."""
    return F[t]["Close"].shift(-h) / F[t]["Open"].shift(-1) - 1.0


def c2c0(t: str, h: int) -> pd.Series:
    c = F[t]["Close"]
    return c.shift(-h) / c - 1.0


def rec(v) -> tuple[int, int, int]:
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return int((v > 0).sum()), int((v < 0).sum()), len(v)


def welch(a, b) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    return float((a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))


def era_str(v: pd.Series) -> str:
    out = []
    for lab, x in (("pre-2018", v[v.index < ERA]), ("2018+", v[v.index >= ERA])):
        if len(x) == 0:
            out.append(f"{lab} n=0")
            continue
        u, d, n = rec(x.values)
        out.append(f"{lab} n={n} {u}-{d} mean {100*x.mean():+.3f}% med {100*x.median():+.3f}%")
    return " | ".join(out)


def best2(v: pd.Series) -> str:
    s = v.sort_values(ascending=False)
    tot = float(v.sum())
    top = float(s.iloc[:2].sum())
    share = 100 * top / tot if tot != 0 else np.nan
    ex = float(s.iloc[2:].mean()) if len(s) > 2 else np.nan
    return (f"best2 {[str(i.date()) for i in s.index[:2]]} {100*top:+.2f}pp of "
            f"{100*tot:+.2f}pp total ({share:.0f}%) | mean ex-best2 {100*ex:+.3f}% | "
            f"best {v.idxmax().date()} {100*v.max():+.2f}% | "
            f"worst {v.idxmin().date()} {100*v.min():+.2f}%")


def block(label: str, s: pd.Series, trig: pd.DatetimeIndex, start: str,
          show_dates: bool | None = None, ctrl: bool = True,
          mae: pd.Series | None = None) -> pd.Series:
    v = s.reindex(trig).dropna()
    v = v[v.index >= start]
    allv = s[(s.index >= start)].dropna()
    if len(v) == 0:
        print(f"  {label}: n=0")
        return v
    u, d, n = rec(v.values)
    sm = summarize(v.values)
    base_hit = float((allv > 0).mean())
    t = sm["t"]
    tt = f"{t:+.2f}" if t is not None and np.isfinite(t) else "n/a"
    lc = ""
    if ctrl:
        loc = local_control(nyse, pd.DatetimeIndex(v.index), 126)
        locv = s.reindex(loc).dropna()
        locv = locv[locv.index >= start]
        lu, ld, _ = rec(locv.values)
        lc = f" local+/-126 {100*locv.mean():+.3f}% ({lu}-{ld})"
    print(f"  {label}: n={n} {u}-{d} mean {sm['mean_pct']:+.3f}% med "
          f"{sm['median_pct']:+.3f}% t {tt} | ctrl all-days {100*allv.mean():+.3f}% "
          f"(hit {100*base_hit:.1f}%, n {len(allv)}){lc} | sign p(up) "
          f"{sign_test(u, n):.4f} p(down) {sign_test(d, n):.4f} | p(up) vs base-rate "
          f"{sign_test(u, n, base_hit):.4f}")
    print(f"      era: {era_str(v)}")
    print(f"      conc: {best2(v)}")
    print(f"      cluster_note: {cluster_note(v.index, v.values)}")
    if mae is not None:
        x = mae.reindex(v.index).dropna()
        if len(x):
            print(f"      MAE (ATR@D, long): med {x.median():+.2f} p10 {x.quantile(0.1):+.2f} "
                  f"worst {x.min():+.2f} ({x.idxmin().date()}) | share <= -1 ATR "
                  f"{100*(x <= -1).mean():.0f}%")
    if show_dates is None:
        show_dates = n <= 30
    if show_dates:
        print("      dates: " + ", ".join(f"{i.date()} {100*x:+.2f}" for i, x in v.items()))
    return v


def st(label: str, v: pd.Series, unit: str = "%", k: float = 100.0, dates: bool = False) -> None:
    """Compact stat line on an already-selected series (dates as index)."""
    v = v.dropna()
    if len(v) == 0:
        print(f"    {label}: n=0")
        return
    u, d, n = rec(v.values)
    fl = n - u - d
    sd = v.std(ddof=1) if n > 1 else np.nan
    t = v.mean() / (sd / np.sqrt(n)) if n > 1 and sd > 0 else np.nan
    print(f"    {label}: n={n} {u}-{d}{f' (flat {fl})' if fl else ''} mean {k*v.mean():+.3f}{unit} "
          f"med {k*v.median():+.3f}{unit} t {t:+.2f} sign p(up) {sign_test(u, n):.4f} "
          f"p(down) {sign_test(d, n):.4f}")
    if n >= 2:
        eo = []
        for lab, x in (("pre-2018", v[v.index < ERA]), ("2018+", v[v.index >= ERA])):
            xu, xd, xn = rec(x.values)
            eo.append(f"{lab} {xu}-{xd}/{xn} {k*x.mean():+.3f}{unit}" if xn else f"{lab} n=0")
        print(f"        era: {' | '.join(eo)} | worst {v.idxmin().date()} {k*v.min():+.2f}{unit} "
              f"best {v.idxmax().date()} {k*v.max():+.2f}{unit}")
    if dates:
        print("        dates: " + ", ".join(f"{i.date()} {k*x:+.2f}" for i, x in v.items()))


def streak(s: pd.Series, down: bool = True) -> pd.Series:
    s = s.dropna()
    m = (s < s.shift(1)) if down else (s > s.shift(1))
    grp = (~m).cumsum()
    return m.astype(int).groupby(grp).cumsum()


def frozen(t: str) -> tuple[float, float]:
    f = raw[t]
    f = f[f.index <= ASOF].dropna(subset=["High", "Low", "Close"]).astype(float)
    a = float(np.asarray(wilder_atr(f["High"].to_numpy(), f["Low"].to_numpy(),
                                    f["Close"].to_numpy()), float)[-1])
    c = float(f["Close"].iloc[-1])
    print(f"  FROZEN {t}: bar {f.index[-1].date()} close {c:.4f} Wilder-14 ATR {a:.4f} "
          f"({100*a/c:.2f}%)")
    return c, a


# ---------------------------------------------------------------------------
# month table on the SPY calendar (complete months only; Sept 2026 excluded)
# ---------------------------------------------------------------------------
rows = []
for m in pd.unique(YM):
    if m == CUR:
        continue
    ix = np.where(YM == m)[0]
    if ix[0] == 0 or ix[-1] + 3 >= len(nyse):
        continue
    rows.append({"ym": int(m), "mo": int(m % 100), "prev_end": nyse[ix[0] - 1],
                 "d4": nyse[ix[-4]], "d3": nyse[ix[-3]], "d2": nyse[ix[-2]],
                 "last": nyse[ix[-1]], "n1": nyse[ix[-1] + 1], "n2": nyse[ix[-1] + 2],
                 "n3": nyse[ix[-1] + 3]})
MT = pd.DataFrame(rows)
MT.index = pd.DatetimeIndex(MT["last"])
print(f"\nmonth table: {len(MT)} complete months {MT['ym'].iloc[0]}..{MT['ym'].iloc[-1]} | "
      f"last->n2 always 2: {bool(all(POS[MT['n2']].values - POS[MT['last']].values == 2))}")
QEM = [3, 6, 9, 12]


def fin_table(t: str, lo: int, hi: int) -> pd.DataFrame:
    M = MT[(MT["ym"] >= lo) & (MT["ym"] <= hi)].copy()
    C, O = F[t]["Close"], F[t]["Open"]
    M["c2c"] = C.reindex(M["last"]).values / C.reindex(M["d2"]).values - 1
    M["o2c"] = C.reindex(M["last"]).values / O.reindex(M["last"]).values - 1
    M["mtd4"] = C.reindex(M["d4"]).values / C.reindex(M["prev_end"]).values - 1
    M["mtd2"] = C.reindex(M["d2"]).values / C.reindex(M["prev_end"]).values - 1
    return M


def other_sessions(t: str, lo: int, hi: int) -> tuple[pd.Series, pd.Series]:
    C, O = F[t]["Close"], F[t]["Open"]
    m = (YM >= lo) & (YM <= hi) & (FE.values >= 1)
    r = (C / C.shift(1) - 1)[m].dropna()
    oc = (C / O - 1)[m].dropna()
    return r, oc


# ===========================================================================
# A. TLT / IEF final session
# ===========================================================================
print("\n" + "=" * 78)
print("=== A. TLT and IEF ON A MONTH'S FINAL SESSION, complete months 2002-08..2026-08 ===")
tlt_q20_mtd4 = None
for t in ("TLT", "IEF"):
    M = fin_table(t, 200208, 202608)
    print(f"\n  ##### {t}: {len(M)} months {M['ym'].iloc[0]}..{M['ym'].iloc[-1]} | NaN c2c "
          f"{int(M['c2c'].isna().sum())} o2c {int(M['o2c'].isna().sum())} #####")
    qe = M["mo"].isin(QEM)
    if t == "TLT":
        tlt_q20_mtd4 = float(M["mtd4"].quantile(0.2))
        TLT_COND = M["mtd4"] <= tlt_q20_mtd4
    q20_2 = float(M["mtd2"].quantile(0.2))
    own_q20 = float(M["mtd4"].quantile(0.2))
    cond4 = TLT_COND.reindex(M.index).fillna(False)
    cond2 = M["mtd2"] <= q20_2
    print(f"  cuts: TLT MTD-at-4th-last worst quintile {100*tlt_q20_mtd4:.3f}% (full sample) | "
          f"{t} own MTD-at-4th-last quintile {100*own_q20:.3f}% | {t} MTD entering the final "
          f"session (at 2nd-last close) quintile {100*q20_2:.3f}%")
    for form in ("c2c", "o2c"):
        oth_r, oth_oc = other_sessions(t, 200208, 202608)
        base = oth_r if form == "c2c" else oth_oc
        v = M[form]
        print(f"  -- {t} {form} ({'Close[2nd-last] -> Close[final]' if form == 'c2c' else 'Open[final] -> Close[final]'}) --")
        st("all finals", v)
        st(f"all OTHER sessions same months ({form})", base)
        print(f"      finals vs other sessions: Welch t {welch(v.values, base.values):+.2f} | up rate "
              f"finals {100*(v > 0).mean():.1f}% vs others {100*(base > 0).mean():.1f}%")
        st("quarter-end finals (Mar/Jun/Sep/Dec)", v[qe])
        st("other-month finals", v[~qe])
        print(f"      Welch t QE vs other finals: {welch(v[qe].values, v[~qe].values):+.2f}")
        for lab, lo_, hi_ in (("pre-2018", None, ERA), ("2018+", ERA, None)):
            mm = pd.Series(True, index=M.index)
            if lo_ is not None:
                mm &= M.index >= lo_
            if hi_ is not None:
                mm &= M.index < hi_
            a, b = v[qe & mm], v[~qe & mm]
            print(f"      {lab}: QE {int((a > 0).sum())}/{len(a)} {100*a.mean():+.3f}% | other "
                  f"{int((b > 0).sum())}/{len(b)} {100*b.mean():+.3f}% | Welch {welch(a.values, b.values):+.2f}")
        st("September finals", v[M["mo"] == 9], dates=(form == "c2c"))
        st("December finals", v[M["mo"] == 12], dates=(form == "c2c"))
        for mo in (3, 6):
            x = v[M["mo"] == mo]
            print(f"      month {mo} finals: {int((x > 0).sum())}/{len(x)} {100*x.mean():+.3f}%")
        st("TLT-worst-quintile MTD entering final THREE (<= cut)", v[cond4], dates=False)
        st("  ... AND quarter-end", v[cond4 & qe], dates=True)
        st("  ... AND other months", v[cond4 & ~qe])
        st(f"worst-fifth MTD entering the FINAL SESSION ({t} own, brief 03 def)", v[cond2])
        st("  ... AND quarter-end (brief 03 def)", v[cond2 & qe], dates=True)
        trimmed = v[qe].drop(v[qe].abs().nlargest(2).index)
        print(f"      QE finals without two largest |moves|: {100*trimmed.mean():+.3f}% "
              f"({int((trimmed > 0).sum())}/{len(trimmed)})")

# ===========================================================================
# B. SPY final session
# ===========================================================================
print("\n" + "=" * 78)
print("=== B. SPY ON A MONTH'S FINAL SESSION, complete months 2000-02..2026-08 ===")
MB = fin_table("SPY", 200002, 202608)
qeb = MB["mo"].isin(QEM)
print(f"  months {len(MB)} ({MB['ym'].iloc[0]}..{MB['ym'].iloc[-1]}) | QE {int(qeb.sum())} other {int((~qeb).sum())}")
for form in ("c2c", "o2c"):
    oth_r, oth_oc = other_sessions("SPY", 200002, 202608)
    base = oth_r if form == "c2c" else oth_oc
    v = MB[form]
    u = int((v > 0).sum())
    ph = float((base > 0).mean())
    print(f"  -- SPY {form} --")
    st("all finals", v)
    st("all other sessions", base)
    print(f"      final up {u}/{len(v)} = {100*u/len(v):.1f}% vs other sessions {100*ph:.1f}% | "
          f"P(up <= {u} | p=other rate) {sign_test(len(v) - u, len(v), 1 - ph):.4f} | Welch t "
          f"{welch(v.values, base.values):+.2f}")
    for lab, x, b in (("pre-2018", v[v.index < ERA], base[base.index < ERA]),
                      ("2018+", v[v.index >= ERA], base[base.index >= ERA])):
        print(f"      {lab}: finals {int((x > 0).sum())}/{len(x)} ({100*(x > 0).mean():.1f}%) "
              f"{100*x.mean():+.3f}% | others up {100*(b > 0).mean():.1f}% {100*b.mean():+.3f}%")
    st("quarter-end finals", v[qeb])
    st("other-month finals", v[~qeb])
    print(f"      Welch t QE vs other finals {welch(v[qeb].values, v[~qeb].values):+.2f}")
    st("September finals", v[MB["mo"] == 9], dates=(form == "c2c"))

# ===========================================================================
# C. TLT sixth consecutive lower close
# ===========================================================================
print("\n" + "=" * 78)
print("=== C. TLT SIXTH CONSECUTIVE LOWER CLOSE (anchor = run first reaches 6) ===")
tcr = raw["TLT"]["Close"]
tcr = tcr[tcr.index <= ASOF].astype(float).dropna()
STK = streak(tcr).reindex(nyse)
for d in nyse[-9:]:
    print(f"    {d.date()} close {F['TLT']['Close'][d]:.4f} chg {100*R1['TLT'][d]:+.3f}% "
          f"streak {int(STK[d])}")
run_start = nyse[-int(STK.iloc[-1])]
print(f"  tonight streak {int(STK.iloc[-1])}, run started {run_start.date()} | exact close ties in "
      f"TLT history {int((tcr.diff() == 0).sum())}")
five_p = nyse[(STK == 5).values]
five_p = five_p[five_p < run_start]
six_p = nyse[(STK == 6).values]
six_p = six_p[six_p < run_start]
sev_p = nyse[(STK == 7).values]
reach6 = sum(1 for d in five_p if STK.iloc[POS[d] + 1] == 6)
print(f"  earlier runs reaching 5: {len(five_p)} | of those reaching 6: {reach6} | runs reaching 7: "
      f"{len(sev_p)} | brief: 23 of 63")
S0 = "2002-07-31"
block("C1 TLT next session Close D -> Close D+1 (c2c)", c2c0("TLT", 1), six_p, S0, show_dates=True)
block("C2 TLT next session Open D+1 -> Close D+1 (lag1 o2c)", o2c_in("TLT", 1), six_p, S0,
      show_dates=False)
block("C3 TLT Open D+1 -> Close D+2 (lag1 h2)", o2c_in("TLT", 2), six_p, S0, show_dates=False)
st("TLT Close D -> Close D+2 (c2c h2)", c2c0("TLT", 2).reindex(six_p))
print("  sixth-close anchors by month position (FE = sessions left in month after D):")
for d in six_p:
    fe = FE[d]
    tag = ""
    if fe == 1:
        tag = " <-- 6th on 2nd-last session; next = month final" + (" (QUARTER)" if d.month in QEM else "")
    elif fe == 0:
        tag = " <-- 6th ON the final session"
    print(f"    {d.date()} FE {int(fe)} next c2c {100*c2c0('TLT', 1)[d]:+.2f}% next o2c "
          f"{100*o2c_in('TLT', 1)[d]:+.2f}% streak next {int(STK.iloc[POS[d] + 1])}{tag}")
print(f"  tonight FE (by calendar): 9/29 is the 2nd-last September session; quarter-end: True")

# ===========================================================================
# D. ^TNX facts, TLT lowest-since
# ===========================================================================
print("\n" + "=" * 78)
print("=== D. ^TNX LEVEL AND UP-RUN FACTS; TLT LOWEST-SINCE ===")
tnx = raw["^TNX"]["Close"]
tnx = tnx[tnx.index <= ASOF].astype(float).dropna()
tnx = tnx[tnx > 0]
v0 = float(tnx.iloc[-1])
prior = tnx.iloc[:-1]
ge_ = prior[prior >= v0]
gt_ = prior[prior > v0]
print(f"  ^TNX {tnx.index[0].date()}..{tnx.index[-1].date()} n={len(tnx)} | not on SPY cal "
      f"{len(tnx.index.difference(nyse))} | SPY sessions missing {len(nyse[nyse >= tnx.index[0]].difference(tnx.index))}"
      f" | exact ties {int((tnx.diff() == 0).sum())}")
print(f"  tonight close {v0:.3f} ({100*(v0 - prior.iloc[-1]):+.1f}bp) | last close >= tonight: "
      f"{ge_.index[-1].date() if len(ge_) else 'none'} ({ge_.iloc[-1] if len(ge_) else np.nan:.3f}) | "
      f"last close > tonight: {gt_.index[-1].date() if len(gt_) else 'none'}")
SU = streak(tnx, down=False)
print(f"  current run of higher closes: {int(SU.iloc[-1])} (last 8: "
      f"{[(str(d.date()), round(x, 3), int(SU[d])) for d, x in tnx.iloc[-8:].items()]})")
tpos = pd.Series(np.arange(len(tnx)), index=tnx.index)
rs = tnx.index[-int(SU.iloc[-1])]
six_t = tnx.index[(SU == 6).values]
six_t = six_t[six_t < rs]
nxt_bp = (tnx.shift(-1) - tnx) * 100
rows_t = []
for d in six_t:
    p = int(tpos[d])
    L = 6
    while p + (L - 5) < len(tnx) and SU.iloc[p + (L - 5)] == L + 1:
        L += 1
    rows_t.append((d, float(nxt_bp[d]), L, float(tnx[d])))
bpv = pd.Series([r[1] for r in rows_t], index=[r[0] for r in rows_t])
print(f"  earlier runs of 6+ higher closes since {tnx.index[0].year}: {len(six_t)} | next session "
      f"after the 6th: up {int((bpv > 0).sum())} down {int((bpv < 0).sum())} flat {int((bpv == 0).sum())} "
      f"mean {bpv.mean():+.2f}bp median {bpv.median():+.2f}bp | reached 7+: "
      f"{sum(1 for r in rows_t if r[2] >= 7)}")
tlt_nx = c2c0("TLT", 1)
print("    runs: " + ", ".join(f"{d.date()} len{L} lvl {lv:.2f} next {b:+.1f}bp "
                              f"TLT {100*tlt_nx.get(d, np.nan):+.2f}%" for d, b, L, lv in rows_t))
tp = tcr.iloc[:-1]
v1 = float(tcr.iloc[-1])
le_ = tp[tp <= v1]
lt_ = tp[tp < v1]
print(f"  TLT tonight {v1:.4f} (cache, adjusted basis) | last close <= tonight: "
      f"{le_.index[-1].date()} ({le_.iloc[-1]:.4f}) | last close < tonight: {lt_.index[-1].date()} "
      f"({lt_.iloc[-1]:.4f})")

# ===========================================================================
# E. HYG volume spike; HYG sixth lower close
# ===========================================================================
print("\n" + "=" * 78)
print("=== E. HYG VOLUME vs PRIOR 63-SESSION MEAN; HYG SIXTH LOWER CLOSE ===")
hf = raw["HYG"]
hf = hf[hf.index <= ASOF].astype(float).dropna(subset=["Close"])
hv = hf["Volume"]
VR = (hv / hv.shift(1).rolling(63).mean()).reindex(nyse)
vr0 = float(VR.iloc[-1])
gev = VR[VR >= vr0].dropna()
print(f"  tonight vol {hv.iloc[-1]:.0f} / prior-63 mean {hv.iloc[-64:-1].mean():.0f} = {vr0:.3f}x | "
      f"sessions at/above incl tonight: {len(gev)} (first ratio {VR.dropna().index[0].date()}) | "
      f"2007-2009: {int(((gev.index.year >= 2007) & (gev.index.year <= 2009)).sum())} | previous: "
      f"{gev.index[-2].date()}")
print("  by year: " + str(pd.Series(gev.index.year).value_counts().sort_index().to_dict()))
print("  list (date ratio 1d FE FS): " + ", ".join(
    f"{d.date()} {x:.2f}x {100*R1['HYG'][d]:+.2f}% FE{'' if np.isnan(FE[d]) else int(FE[d])} FS{int(FS[d])}"
    for d, x in gev.items()))
S_E = "2007-07-01"
for thr, lab in ((vr0, f">= tonight {vr0:.2f}x"), (3.5, ">= 3.5x")):
    rawE = nyse[(VR >= thr).fillna(False).values]
    trigE = declusters(rawE[rawE < ASOF], 5, nyse)
    for nm, tr in (("all", trigE), ("ex-2008", trigE[trigE.year != 2008]),
                   ("ex-2007-09", trigE[(trigE.year < 2007) | (trigE.year > 2009)])):
        print(f"\n  ##### HYG spike {lab}, declustered 5, {nm}: n {len(tr)} #####")
        sd = len(tr) <= 30 and thr == vr0
        block("E HYG next session Open D+1 -> Close D+1", o2c_in("HYG", 1), tr, S_E, show_dates=sd)
        st("HYG next session Close D -> Close D+1", c2c0("HYG", 1).reindex(tr))
        block("E HYG Open D+1 -> Close D+5 (lag1 next five)", o2c_in("HYG", 5), tr, S_E, show_dates=False)
        st("HYG Close D -> Close D+5", c2c0("HYG", 5).reindex(tr))

hcr = hf["Close"]
SH = streak(hcr).reindex(nyse)
hrs = nyse[-int(SH.iloc[-1])]
six_h = nyse[(SH == 6).values]
six_h = six_h[six_h < hrs]
print(f"\n  HYG streak tonight {int(SH.iloc[-1])} (run from {hrs.date()}); earlier sixth-close anchors "
      f"{len(six_h)} | HYG tonight {hcr.iloc[-1]:.2f}, last close <= tonight "
      f"{hcr.iloc[:-1][hcr.iloc[:-1] <= hcr.iloc[-1]].index[-1].date()}")
st("HYG 6th close, next session Close D -> Close D+1", c2c0("HYG", 1).reindex(six_h))
st("HYG 6th close, next five Close D -> Close D+5", c2c0("HYG", 5).reindex(six_h))
st("HYG 6th close, lag1 Open D+1 -> Close D+1", o2c_in("HYG", 1).reindex(six_h))
st("HYG 6th close, lag1 Open D+1 -> Close D+5", o2c_in("HYG", 5).reindex(six_h))
allh1 = c2c0("HYG", 1).dropna()
allh5 = c2c0("HYG", 5).dropna()
print(f"    all-days HYG h1 up {100*(allh1 > 0).mean():.1f}% mean {100*allh1.mean():+.3f}% | h5 up "
      f"{100*(allh5 > 0).mean():.1f}% mean {100*allh5.mean():+.3f}%")

# ===========================================================================
# F. IDEA: short TLT MOC on a bad month's final session, out next month's 2nd close
# ===========================================================================
print("\n" + "=" * 78)
print("=== F. IDEA CANDIDATE: SHORT TLT at the final-session CLOSE (MOC) of a bad month, "
      "exit Close of next month's 2nd session ===")
tC, tO, tH = F["TLT"]["Close"], F["TLT"]["Open"], F["TLT"]["High"]
MF = MT[(MT["ym"] >= 200208) & (MT["ym"] <= 202608)].copy()


def at(s: pd.Series, col: str) -> np.ndarray:
    return s.reindex(MF[col]).values


MF["mtd4"] = at(tC, "d4") / at(tC, "prev_end") - 1
Q20 = float(MF["mtd4"].quantile(0.2))
MF["cond"] = MF["mtd4"] <= Q20
MF["tlt2"] = at(tC, "n2") / at(tC, "last") - 1
MF["short2"] = -MF["tlt2"]
MF["tlt1"] = at(tC, "n1") / at(tC, "last") - 1
MF["tlt3"] = at(tC, "n3") / at(tC, "last") - 1
MF["tlt_o2"] = at(tC, "n2") / at(tO, "n1") - 1
MF["tlt_f3"] = at(tC, "last") / at(tC, "d4") - 1
hi2 = np.maximum(at(tH, "n1"), at(tH, "n2"))
MF["mae_s"] = -(hi2 - at(tC, "last")) / at(ATR["TLT"], "last")
MF["f3dir"] = at(tC, "last") / at(tO, "d3") - 1
MF["two_fell"] = (at(tC, "d3") < at(tC, "d4")) & (at(tC, "d2") < at(tC, "d3"))
MF["qs"] = MF["mo"].isin(QEM)
nfp_all = load_events(["nfp"])["date"]
nfpv = nfp_all.values.astype("datetime64[ns]")
MF["nfp"] = [bool(((nfpv > np.datetime64(a)) & (nfpv <= np.datetime64(b))).any())
             for a, b in zip(MF["last"], MF["n2"])]
lo252 = tC.rolling(252, min_periods=252).min()
is_lo = (tC <= lo252 + 1e-9) & lo252.notna()
MF["low252"] = is_lo.reindex(MF["d3"]).values | is_lo.reindex(MF["d2"]).values | is_lo.reindex(MF["last"]).values
MF["low252_ok"] = lo252.reindex(MF["d3"]).notna().values
exp24, exp60 = [], []
vals = MF["mtd4"].values
for i in range(len(MF)):
    exp24.append(np.quantile(vals[:i], 0.2) if i >= 24 else np.nan)
    exp60.append(np.quantile(vals[:i], 0.2) if i >= 60 else np.nan)
MF["cut24"] = exp24
MF["cut60"] = exp60
print(f"  months {len(MF)} ({MF['ym'].iloc[0]}..{MF['ym'].iloc[-1]}) | NaN tlt2 {int(MF['tlt2'].isna().sum())} "
      f"| cut = FULL-SAMPLE 20th pct of MTD at the 4th-last close (09-27 def): {100*Q20:.3f}% | n cond "
      f"{int(MF['cond'].sum())}")
print("  SIGN KEY: 'SHORT' rows are the short's P&L (= -TLT). 'TLT' rows are TLT's own return. "
      "Short wins = TLT down.")


def fshow(label: str, mask, dates: bool = False) -> pd.Series:
    x = MF.loc[np.asarray(mask, bool), "short2"].dropna()
    n = len(x)
    if n == 0:
        print(f"  {label}: n=0")
        return x
    w, l, _ = rec(x.values)
    sm = summarize(x.values)
    tl = -x
    mae = MF.loc[x.index, "mae_s"]
    print(f"  {label}: n={n} | SHORT {w}-{l} mean {sm['mean_pct']:+.3f}% med {sm['median_pct']:+.3f}% "
          f"t {sm['t']:+.2f} sign p(short wins) {sign_test(w, n):.4f} | TLT side {l}-{w} (up-down) "
          f"mean {100*tl.mean():+.3f}% med {100*tl.median():+.3f}%")
    eo = []
    for lab, y in (("pre-2018", x[x.index < ERA]), ("2018+", x[x.index >= ERA])):
        yw, yl, yn = rec(y.values)
        eo.append(f"{lab} SHORT {yw}-{yl}/{yn} {100*y.mean():+.3f}%" if yn else f"{lab} n=0")
    s = x.sort_values(ascending=False)
    tot = float(x.sum())
    top = float(s.iloc[:2].sum())
    print(f"      era: {' | '.join(eo)}")
    print(f"      conc (short): top2 wins {[str(i.date()) for i in s.index[:2]]} {100*top:+.2f}pp of "
          f"{100*tot:+.2f}pp ({100*top/tot if tot else np.nan:.0f}%) | ex-top2 mean "
          f"{100*s.iloc[2:].mean() if n > 2 else np.nan:+.3f}% | WORST for short (TLT rally) entry "
          f"{x.idxmin().date()} exit {MF.loc[x.idxmin(), 'n2'].date()} TLT {100*tl.max():+.2f}%")
    print(f"      MAE short (max High D+1..D+2 vs entry close, ATR@entry, neg=adverse): med "
          f"{mae.median():+.2f} p10 {mae.quantile(0.1):+.2f} worst {mae.min():+.2f} "
          f"({mae.idxmin().date()}) | share <= -1 ATR {100*(mae <= -1).mean():.0f}%")
    if dates:
        print("      entries (final session) short%: " + ", ".join(
            f"{i.date()} {100*v:+.2f}" for i, v in x.items()))
    return x


cond = MF["cond"].values
xc = fshow("F1 CONDITIONED (MTD<=cut)", cond, dates=True)
xo = fshow("F1c CONTROL other months (not cond)", ~cond)
xa = fshow("F1a all months", np.ones(len(MF), bool))
print(f"  Welch t conditioned vs other months (SHORT side): {welch(xc.values, xo.values):+.2f} | TLT "
      f"side {welch(-xc.values, -xo.values):+.2f}")
all2 = (tC.shift(-2) / tC - 1).dropna()
all2 = all2[all2.index >= "2002-08-01"]
print(f"  unconditional TLT any 2-session c2c window: mean {100*all2.mean():+.3f}% up {100*(all2 > 0).mean():.1f}% "
      f"n {len(all2)}")
v27 = MF.loc[cond, "tlt2"]
print(f"  RECONCILE 09-27 (TLT 20-38, -0.358%, t -1.96, n 58): TLT {int((v27 > 0).sum())}-"
      f"{int((v27 < 0).sum())} mean {100*v27.mean():+.3f}% t {summarize(v27.values)['t']:+.2f} n {len(v27)}")

print("\n  -- splits within CONDITIONED months --")
f3u = MF["f3dir"].values > 0
fshow("F2 final three UP (Open first-of-three -> final Close)", cond & f3u)
fshow("F2 final three DOWN", cond & ~f3u, dates=True)
tf = MF["two_fell"].values
fshow("F2b first two of final three BOTH fell c2c (this month's state)", cond & tf, dates=True)
fshow("F2b not both fell", cond & ~tf)
fshow("F2c both fell AND final three down", cond & tf & ~f3u, dates=True)
fshow("F2d both fell AND final three up (final session bounced enough)", cond & tf & f3u, dates=True)
qs = MF["qs"].values
fshow("F3 window in quarter-START month (condition month Mar/Jun/Sep/Dec)", cond & qs, dates=True)
fshow("F3 window in other months", cond & ~qs)
nf = MF["nfp"].values
fshow("F4 NFP inside the two-session window", cond & nf, dates=True)
fshow("F4 no NFP in window", cond & ~nf)
fshow("F4b NFP in window AND quarter-start (this month's config)", cond & nf & qs, dates=True)
lw = MF["low252"].values
print(f"  (252d-low flag needs 252 bars: {int((~MF['low252_ok']).sum())} early months cannot qualify)")
fshow("F5 TLT at a 252-session closing low within the final three", cond & lw, dates=True)
fshow("F5 not at 252d low", cond & ~lw)
fshow("F5b 252d low AND both fell", cond & lw & tf, dates=True)
print("  controls without the condition:")
fshow("  F-ctl quarter-start all months", qs)
fshow("  F-ctl NFP-in-window all months", nf)
fshow("  F-ctl 252d low in final three, all months", lw)

print("\n  -- out-of-sample cut (expanding 20th pct of PRIOR months' MTD only) --")
for col, mn in (("cut24", 24), ("cut60", 60)):
    ok = MF[col].notna().values
    ce = ok & (MF["mtd4"].values <= MF[col].values)
    print(f"  expanding cut, min {mn} prior months: first usable {MF.index[ok][0].date()}, cut range "
          f"{100*np.nanmin(MF[col]):.2f}%..{100*np.nanmax(MF[col]):.2f}% | latest {100*MF[col].iloc[-1]:.3f}%")
    fshow(f"F6 expanding({mn}) conditioned", ce)
    fshow(f"F6 expanding({mn}) others (same span)", ok & ~ce)
    fshow(f"F6 full-sample cut on the same span", ok & cond)

print("\n  -- neighbour definitions (fragility) --")
for qq in (0.10, 0.15, 0.25, 0.33, 0.50):
    cq = MF["mtd4"].values <= MF["mtd4"].quantile(qq)
    fshow(f"cut pct {int(100*qq)} ({100*MF['mtd4'].quantile(qq):.2f}%)", cq)
for fx in (-0.02, -0.03, -0.04):
    fshow(f"fixed cut {100*fx:.0f}%", MF["mtd4"].values <= fx)
for col, lab in (("tlt1", "hold 1 session (Close final -> Close n1)"),
                 ("tlt3", "hold 3 sessions (-> Close n3)"),
                 ("tlt_o2", "lag-1 open entry: Open n1 -> Close n2")):
    y = MF.loc[cond, col]
    yo = MF.loc[~cond, col]
    print(f"  {lab}: cond TLT {int((y > 0).sum())}-{int((y < 0).sum())} mean {100*y.mean():+.3f}% t "
          f"{summarize(y.values)['t']:+.2f} | others {100*yo.mean():+.3f}% | Welch {welch(y.values, yo.values):+.2f}")
ic = F["IEF"]["Close"]
ief2 = ic.reindex(MF["n2"]).values / ic.reindex(MF["last"]).values - 1
iefs = pd.Series(ief2, index=MF.index)
st("IEF companion on TLT-cond months (IEF return, NOT short side)", iefs[cond])
st("IEF other months", iefs[~cond])
fb = MF.loc[cond, "tlt_f3"]
print(f"  final-three bid in cond months (Close 4th-last -> final close, TLT): mean {100*fb.mean():+.3f}% | "
      f"corr(final-three, next-two) cond {np.corrcoef(MF.loc[cond, 'tlt_f3'], MF.loc[cond, 'tlt2'])[0, 1]:+.2f}")

print("\n  -- tonight's state --")
aug_end = nyse[nyse < "2026-09-01"][-1]
d4_26 = pd.Timestamp("2026-09-25")
mtd_d4 = tC[d4_26] / tC[aug_end] - 1
print(f"  TLT MTD at 9/25 (4th-last) {100*mtd_d4:+.3f}% vs full cut {100*Q20:.3f}% -> qualifies "
      f"{bool(mtd_d4 <= Q20)} | vs expanding cut (all 289 prior) {100*np.quantile(vals, 0.2):.3f}% | rank "
      f"{int((MF['mtd4'] <= mtd_d4).sum())} of {len(MF)} months at/below")
print(f"  MTD at 9/29 {100*(tC[ASOF]/tC[aug_end]-1):+.3f}% | 9/28 {100*R1['TLT']['2026-09-28']:+.2f}% "
      f"9/29 {100*R1['TLT'][ASOF]:+.2f}% -> first two of final three both fell: "
      f"{bool(tC['2026-09-28'] < tC[d4_26] and tC[ASOF] < tC['2026-09-28'])} | 252d closing low: "
      f"9/28 {bool(is_lo['2026-09-28'])} 9/29 {bool(is_lo[ASOF])} (252d min {lo252[ASOF]:.4f})")
nf26 = nfp_all[(nfp_all > "2026-09-29") & (nfp_all <= "2026-10-10")]
print(f"  next NFP in calendar: {[str(d.date()) for d in nf26]} | window sessions Thu 10/1, Fri 10/2 "
      f"-> NFP inside window: {bool(any(d == pd.Timestamp('2026-10-02') for d in nf26))} | window month "
      f"Oct = quarter start")
frozen("TLT")
frozen("IEF")

# ===========================================================================
# G. NFP k3
# ===========================================================================
print("\n" + "=" * 78)
print("=== G. NFP k3: SPY on the session after the anchor (h1 = 2 td before NFP) ===")
sp = raw["SPY"]
sp = sp[sp.index <= ASOF].astype(float)
c, o = sp["Close"], sp["Open"]
idx = c.index
r = c.pct_change()
roc = c / o - 1
nfp = nfp_all[nfp_all <= ASOF]
pos, kept = anchor_positions(idx, nfp, offset=-2)
sess = idx[pos]
keep_m = sess < idx[-1]
sess = sess[keep_m]
kept = kept[keep_m]
print(f"  NFP events {kept[0].date()}..{kept[-1].date()} ({len(kept)}) | h1 sessions "
      f"{sess[0].date()}..{sess[-1].date()} | anchors (k3) {idx[np.array(pos)[keep_m] - 1][0].date()}.."
      f"{idx[np.array(pos)[keep_m] - 1][-1].date()} | weekdays of h1: "
      f"{pd.Series(sess.dayofweek).value_counts().sort_index().to_dict()}")
per = pd.Series(idx.to_period("M"), index=idx)
fe = per.groupby(per.values).cumcount(ascending=False)
fs = per.groupby(per.values).cumcount() + 1
complete = pd.Series(idx.to_period("M") < pd.Period("2026-09", "M"), index=idx)
is_k3 = pd.Series(idx.isin(sess), index=idx)
wd = pd.Series(idx.dayofweek, index=idx)
cat_pos = pd.Series(np.where(fe == 0, "last", np.where(fe == 1, "2nd-last", np.where(
    fs <= 2, "first12", np.where(fs <= 5, "td35", "mid")))), index=idx)
ok = complete & ~is_k3
for nm, ser in (("c2c Close[anchor] -> Close[h1] (MOC at anchor, calendar known)", r),
                ("lag1 Open[h1] -> Close[h1]", roc)):
    v = ser.reindex(sess).dropna()
    print(f"\n  -- {nm} --")
    st("G1 raw k3 h1", v)
    allv = ser.dropna()
    print(f"      unconditional daily baseline: n {len(allv)} mean {100*allv.mean():+.3f}% up "
          f"{100*(allv > 0).mean():.1f}% | vs base: sign p(up) {sign_test(int((v > 0).sum()), len(v), float((allv > 0).mean())):.4f}"
          f" Welch {welch(v.values, allv.values):+.2f}")
    base_p = ser[ok].groupby(cat_pos[ok]).mean()
    adj_p = (v - cat_pos.reindex(v.index).map(base_p)).dropna()
    st("G2 k3 minus position-only control", adj_p)
    key = cat_pos + "|" + wd.astype(str)
    base_w = ser[ok].groupby(key[ok]).mean()
    adj_w = (v - key.reindex(v.index).map(base_w)).dropna()
    st("G3 k3 minus weekday x position control", adj_w)
    print("      position counts of k3 h1: " + str(cat_pos.reindex(v.index).value_counts().to_dict()))
    print("      position-only control means: " + ", ".join(f"{k} {100*x:+.3f}%" for k, x in base_p.items()))
    lastk = v[cat_pos.reindex(v.index).values == "last"]
    ctl_last = ser[ok & (fe == 0)].dropna()
    ctl_last_wed = ser[ok & (fe == 0) & (wd == 2)].dropna()
    st("G4 k3 h1 that IS the month's last session", lastk, dates=(ser is r))
    st("  control: other last sessions (no NFP 2 td later)", ctl_last)
    st("  control: other last sessions on Wednesday", ctl_last_wed)
    print(f"      Welch k3-last vs other last {welch(lastk.values, ctl_last.values):+.2f}")

# ===========================================================================
# H. Open idea marks
# ===========================================================================
print("\n" + "=" * 78)
print("=== H. OPEN IDEA MARKS at the 2026-09-29 close (cache bars, adjusted basis, read only) ===")
import posts_journal  # noqa: E402
from posts_grammar import derive_order_row  # noqa: E402

drafts = {d["draft_id"]: d for d in posts_journal.fold_drafts(posts_journal.load(posts_journal.JOURNAL_PATH))}
fills = {"x20260924-1": (pd.Timestamp("2026-09-25"), 89.79, "given MOO fill")}
for did in ("x20260921-1", "x20260924-1", "x20260925-1", "x20260927-1"):
    d = drafts[did]
    sp_ = d["idea"]
    t, side, atr = sp_["ticker"], sp_["side"], float(sp_["atr"])
    ex = pd.Timestamp(sp_["execute_on"])
    try:
        row = derive_order_row(d)
        tx = pd.Timestamp(row["Time_Exit_Date"])
    except Exception as exc:  # noqa: BLE001
        print(f"  {did}: derive_order_row failed {exc!r}")
        tx = pd.NaT
    if did in fills:
        ed, px, how = fills[did]
        how += f" (cache open {F[t]['Open'][ed]:.4f})"
    elif sp_["entry"]["type"] == "MOO":
        ed, px, how = ex, float(F[t]["Open"][ex]), "cache open on execute_on"
    else:
        ed, px, how = ex, float(F[t]["Close"][ex]), "cache close on execute_on (MOC)"
    final = pd.notna(tx) and tx <= ASOF
    md = tx if final else ASOF
    mk = float(F[t]["Close"][md])
    sgn = 1.0 if side == "long" else -1.0
    mv = sgn * (mk - px)
    dd = nyse[nyse <= pd.Timestamp(d["date"])][-1]
    refc = float(F[t]["Close"][dd])
    print(f"  {did} {t} {side} {sp_['entry']['type']} exec {ex.date()} time_td {sp_['time_td']} -> "
          f"Time_Exit_Date {tx.date() if pd.notna(tx) else 'n/a'} | stop_atr {sp_.get('stop_atr')} | "
          f"entry {px:.4f} [{how}] | {'FINAL exit' if final else 'mark'} {md.date()} close {mk:.4f} | "
          f"favour {100*mv/px:+.2f}% | R {mv/atr:+.3f} (move/atr, atr {atr})")
    print(f"      ref_close {sp_.get('ref_close')} vs cache close on {dd.date()} {refc:.4f} "
          f"(diff {refc - float(sp_.get('ref_close') or np.nan):+.4f}: nonzero = adjustment since) | "
          f"journal outcome {json.dumps(d.get('outcome'))[:160] if d.get('outcome') else 'none'}")
print("\ndone.")
