"""Posts check (2026-09-25): idea and stat candidates for Friday night's queue.

Run date Friday 2026-09-25 is the freshest bar (signal close). Next session
is Monday 2026-09-28. Every idea return is LAG-1: entry at the D+1 open (MOO)
or the D+1 close (MOC reference), exit at the close h sessions AFTER the entry
session. So "h5 from open" = Open[D+1] -> Close[D+6]. Quarter-end cell A is
the exception by design: its exit is pinned to the quarter's last close
(Open[D+1] -> Close[D+3]).

A. Quarter-end rebalance: anchor D = 3 sessions before the quarter's last
   session (SPY calendar). TLT, IEF, SPY, TLT-SPY. Condition SPY QTD - TLT QTD
   >= 8pp / >= 10pp. Controls: all quarters 2002+, September quarters, and the
   SPY 1993+ calendar stat (yfinance supplement, the cache starts 2000).
B. IWM 63d return at its trailing-252 low (rank <= 2) with SPY within 1% of
   its 252d closing high. Declustered 21. Relaxed rank <= 5 / 2%. ^RUT vs
   ^GSPC 1988+ via yfinance supplement.
C. ^MOVE -8% day from a prior close at/above its trailing-252 90th pct.
   Declustered 5, 2002+. SPY, TLT, IEF. VIX < 20 subset.
D. XLU 21d <= -8% with SPY 21d >= 0. Declustered 21. Near-52w-low subset.
E. ^TNX 252d closing high three sessions running (D = third). Declustered 10.
   1990+ (yfinance supplement) and 2003+ (cache).
F. USO September Fridays through 2026-09-25.
H. Frozen levels at the 2026-09-25 close (Wilder-14 ATR, adjusted bars).
I. Marks for the open ideas (approximate; the grader books them separately).
G (breadth divergence) is in 02_cells.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pitch_lab import (  # noqa: E402
    cluster_note, declusters, load_prices, local_control, sign_test,
    summarize, wilder_atr,
)

ASOF = pd.Timestamp("2026-09-25")
ERA = pd.Timestamp("2018-01-01")
ETF = ["SPY", "TLT", "IEF", "IWM", "QQQ", "XLU", "XLE", "USO"]
IDX = ["^TNX", "^MOVE", "^VIX", "^RUT", "^GSPC"]
TK = ETF + IDX

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
    print(f"  {t}: {f.index[0].date()} .. {f.index[-1].date()} n={len(f)} "
          f"last close {f['Close'].iloc[-1]:.4f} | Open<=0/NaN rows {bad_open} "
          f"| Open==Close rows {eq_oc}")

nyse = raw["SPY"].index
nyse = pd.DatetimeIndex(nyse[nyse <= ASOF])
POS = pd.Series(np.arange(len(nyse)), index=nyse)


def frame(t: str) -> pd.DataFrame:
    f = raw[t]
    f = f[f.index <= ASOF].astype(float)
    return f.reindex(nyse)


F = {t: frame(t) for t in ETF + ["^RUT", "^GSPC", "^VIX"]}


def atr_series(t: str) -> pd.Series:
    f = raw[t]
    f = f[f.index <= ASOF].dropna(subset=["High", "Low", "Close"]).astype(float)
    a = wilder_atr(f["High"].to_numpy(), f["Low"].to_numpy(), f["Close"].to_numpy())
    return pd.Series(np.asarray(a, float), index=f.index).reindex(nyse)


ATR = {t: atr_series(t) for t in ETF}

# ---------------------------------------------------------------------------
# yfinance supplement for pre-2000 history (cache starts 2000-01-03)
# ---------------------------------------------------------------------------
YF: dict[str, pd.DataFrame] = {}
try:
    import yfinance as yf
    ydl = yf.download(["SPY", "^GSPC", "^RUT", "^TNX"], start="1985-01-01",
                      end="2026-09-26", auto_adjust=True, progress=False)
    for t in ["SPY", "^GSPC", "^RUT", "^TNX"]:
        df = ydl.xs(t, level="Ticker", axis=1) if isinstance(ydl.columns, pd.MultiIndex) else ydl
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.capitalize() for c in df.columns]
        df = df[df.index <= ASOF].dropna(subset=["Close"]).astype(float)
        YF[t] = df
    print("\n=== data check (yfinance supplement, auto_adjust=True) ===")
    for t, f in YF.items():
        bad = int(((f["Open"] <= 0) | f["Open"].isna()).sum())
        eq = int((f["Open"] == f["Close"]).sum())
        pre = f[f.index < "2000-01-01"]
        eqp = int((pre["Open"] == pre["Close"]).sum())
        print(f"  {t}: {f.index[0].date()} .. {f.index[-1].date()} n={len(f)} | "
              f"Open<=0/NaN {bad} | Open==Close {eq} (pre-2000 {eqp} of {len(pre)})")
except Exception as exc:  # noqa: BLE001
    print(f"\nyfinance supplement FAILED: {exc!r}")


# ---------------------------------------------------------------------------
# forward returns on the SPY calendar, aligned to the signal date D
# ---------------------------------------------------------------------------
def g_o2c(f: pd.DataFrame, h: int) -> pd.Series:
    return f["Close"].shift(-(1 + h)) / f["Open"].shift(-1) - 1.0


def g_o2c_in(f: pd.DataFrame, h: int) -> pd.Series:
    return f["Close"].shift(-h) / f["Open"].shift(-1) - 1.0


def g_c2c(f: pd.DataFrame, h: int) -> pd.Series:
    c = f["Close"]
    return c.shift(-(1 + h)) / c.shift(-1) - 1.0


def g_c2c0(f: pd.DataFrame, h: int) -> pd.Series:
    c = f["Close"]
    return c.shift(-h) / c - 1.0


def o2c(t: str, h: int) -> pd.Series:
    """Open[D+1] -> Close[D+1+h]."""
    return g_o2c(F[t], h)


def o2c_in(t: str, h: int) -> pd.Series:
    """Open[D+1] -> Close[D+h] (entry session counted as session 1)."""
    return g_o2c_in(F[t], h)


def c2c(t: str, h: int) -> pd.Series:
    """Close[D+1] -> Close[D+1+h] (fwd_lag lag=1)."""
    return g_c2c(F[t], h)


def c2c0(t: str, h: int) -> pd.Series:
    return g_c2c0(F[t], h)


def rec(v) -> tuple[int, int, int]:
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return int((v > 0).sum()), int((v < 0).sum()), len(v)


def era_str(v: pd.Series) -> str:
    out = []
    for lab, x in (("pre-2018", v[v.index < ERA]), ("2018+", v[v.index >= ERA])):
        if len(x) == 0:
            out.append(f"{lab} n=0")
            continue
        u, d, n = rec(x.values)
        out.append(f"{lab} n={n} {u}-{d} mean {100*x.mean():+.2f}%")
    return " | ".join(out)


def best2(v: pd.Series) -> str:
    s = v.sort_values(ascending=False)
    tot = float(v.sum())
    top = float(s.iloc[:2].sum())
    share = 100 * top / tot if tot != 0 else np.nan
    ex = float(s.iloc[2:].mean()) if len(s) > 2 else np.nan
    return (f"best2 {[str(i.date()) for i in s.index[:2]]} {100*top:+.2f}pp of "
            f"{100*tot:+.2f}pp total ({share:.0f}%) | mean ex-best2 {100*ex:+.3f}% | "
            f"worst {v.idxmin().date()} {100*v.min():+.2f}%")


def block(label: str, s: pd.Series, trig: pd.DatetimeIndex, start: str,
          show_dates: bool | None = None, cal: pd.DatetimeIndex | None = None,
          ctrl: bool = True) -> pd.Series:
    cal = nyse if cal is None else cal
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
        loc = local_control(cal, pd.DatetimeIndex(v.index), 126)
        locv = s.reindex(loc).dropna()
        locv = locv[locv.index >= start]
        lu, ld, _ = rec(locv.values)
        lc = f" local+/-126 {100*locv.mean():+.3f}% ({lu}-{ld})"
    print(f"  {label}: n={n} {u}-{d} mean {sm['mean_pct']:+.3f}% med "
          f"{sm['median_pct']:+.3f}% t {tt} | ctrl all {100*allv.mean():+.3f}% "
          f"(hit {100*base_hit:.1f}%, n {len(allv)}){lc} | sign p(up) "
          f"{sign_test(u, n):.4f} vs base-rate {sign_test(u, n, base_hit):.4f}")
    print(f"      era: {era_str(v)}")
    print(f"      conc: {best2(v)}")
    print(f"      cluster_note: {cluster_note(v.index, v.values)}")
    if show_dates is None:
        show_dates = n <= 30
    if show_dates:
        print("      dates: " + ", ".join(f"{i.date()} {100*x:+.2f}" for i, x in v.items()))
    return v


def line(label: str, s: pd.Series, trig: pd.DatetimeIndex, start: str) -> None:
    v = s.reindex(trig).dropna()
    v = v[v.index >= start]
    if len(v) == 0:
        print(f"    {label}: n=0")
        return
    u, d, n = rec(v.values)
    print(f"    {label}: n={n} {u}-{d} mean {100*v.mean():+.3f}% med "
          f"{100*v.median():+.3f}% sign p(up) {sign_test(u, n):.4f} | {era_str(v)}")


def on_cal(mask: pd.Series, cal: pd.DatetimeIndex, gap: int, start: str) -> pd.DatetimeIndex:
    m = mask.reindex(cal).eq(True)
    d = cal[m.to_numpy()]
    d = d[(d >= start) & (d <= ASOF)]
    epi = declusters(d, gap, cal)
    last_raw = [str(x.date()) for x in d[-4:]]
    print(f"  [on_cal] latest declustered starts {[str(x.date()) for x in epi[-3:]]} | "
          f"latest raw qualifying {last_raw} | tonight qualifies {bool(m.iloc[-1])}, "
          f"tonight starts a new episode {ASOF in epi}")
    return epi


def roll_rank(c: pd.Series, n: int) -> pd.Series:
    c = c.dropna()
    r = c / c.shift(n) - 1.0
    return r.rolling(252, min_periods=200).rank(pct=True) * 100.0


def dist_hi(c: pd.Series) -> pd.Series:
    c = c.dropna()
    return c / c.rolling(252, min_periods=200).max() - 1.0


def dist_lo(c: pd.Series) -> pd.Series:
    c = c.dropna()
    return c / c.rolling(252, min_periods=200).min() - 1.0


def cl(t: str) -> pd.Series:
    f = raw[t]
    return f[f.index <= ASOF]["Close"].astype(float).dropna()


# ===========================================================================
# A. Quarter-end rebalance
# ===========================================================================
print("\n" + "=" * 78)
print("=== A. QUARTER-END: anchor D = 3 sessions before the quarter's last "
      "session; Open[D+1] -> Close[D+3] ===")


def quarter_table(cal: pd.DatetimeIndex, closes: dict[str, pd.Series]) -> pd.DataFrame:
    pos = pd.Series(np.arange(len(cal)), index=cal)
    q = cal.to_period("Q")
    cur = ASOF.to_period("Q")
    rows = []
    for qq in pd.unique(q):
        if qq == cur:
            continue
        g = cal[q == qq]
        last, first = g.max(), g.min()
        p = int(pos[last])
        pf = int(pos[first])
        prev = cal[pf - 1] if pf > 0 else pd.NaT
        row = {"q": str(qq), "anchor": cal[p - 3], "last": last, "prev_end": prev,
               "month": last.month}
        for t, c in closes.items():
            if pd.isna(prev):
                row[f"qtd_{t}"] = np.nan
            else:
                row[f"qtd_{t}"] = c.get(cal[p - 3], np.nan) / c.get(prev, np.nan) - 1.0
        rows.append(row)
    return pd.DataFrame(rows).set_index("anchor")


qt = quarter_table(nyse, {"SPY": F["SPY"]["Close"], "TLT": F["TLT"]["Close"]})
qt["spread"] = qt["qtd_SPY"] - qt["qtd_TLT"]
prev_end = nyse[nyse < "2026-07-01"][-1]
qs = F["SPY"]["Close"][ASOF] / F["SPY"]["Close"][prev_end] - 1
qtl = F["TLT"]["Close"][ASOF] / F["TLT"]["Close"][prev_end] - 1
print(f"  TONIGHT {ASOF.date()} (3 sessions left: 9/28, 9/29, 9/30): Q2 end "
      f"{prev_end.date()} | SPY QTD {100*qs:+.2f}% TLT QTD {100*qtl:+.2f}% | "
      f"SPY-TLT {100*(qs-qtl):+.2f}pp")
print(f"  sanity: anchor->last gap always 3 sessions: "
      f"{bool(all(POS[qt['last']].values - POS[qt.index].values == 3))} | quarters "
      f"with spread defined {int(qt['spread'].notna().sum())} "
      f"(first {qt['spread'].dropna().index[0].date()})")

sp = {}
for h, nm in ((3, "D+3"), (2, "D+2")):
    sp[nm] = {
        "TLT": o2c_in("TLT", h), "IEF": o2c_in("IEF", h), "SPY": o2c_in("SPY", h),
        "TLT-SPY": o2c_in("TLT", h) - o2c_in("SPY", h),
        "IEF-SPY": o2c_in("IEF", h) - o2c_in("SPY", h),
    }
lag0 = {"TLT": c2c0("TLT", 3), "SPY": c2c0("SPY", 3),
        "TLT-SPY": c2c0("TLT", 3) - c2c0("SPY", 3)}
lag1c = {"TLT": c2c("TLT", 2), "SPY": c2c("SPY", 2),
         "TLT-SPY": c2c("TLT", 2) - c2c("SPY", 2)}

allq = qt.index[qt.index >= "2002-01-01"]
subsets = {
    "SPY-TLT QTD >= 10pp": qt.index[qt["spread"] >= 0.10],
    "SPY-TLT QTD >= 8pp": qt.index[qt["spread"] >= 0.08],
    "UNCONDITIONAL all quarters 2002+": allq,
    "UNCONDITIONAL September quarters 2002+": allq[qt.loc[allq, "month"] == 9],
    "September quarters AND spread >= 8pp": qt.index[(qt["spread"] >= 0.08) & (qt["month"] == 9)],
    "MIRROR: TLT beat SPY by >= 8pp (spread <= -8pp)": qt.index[qt["spread"] <= -0.08],
}
for nm, trig in subsets.items():
    trig = pd.DatetimeIndex(trig)
    print(f"\n  ##### {nm}: n anchors {len(trig)} #####")
    if len(trig) <= 30:
        print("      anchors (spread pp): " + ", ".join(
            f"{d.date()} {100*qt.loc[d, 'spread']:+.1f}" for d in trig))
    for inst in ("TLT", "IEF", "SPY", "TLT-SPY"):
        block(f"A {inst} OPEN D+1 -> CLOSE D+3 (qtr last)", sp["D+3"][inst], trig,
              "2002-01-01", ctrl=False)
    for inst in ("TLT", "SPY", "TLT-SPY", "IEF-SPY"):
        line(f"{inst} OPEN D+1 -> CLOSE D+2 (skip last session)", sp["D+2"][inst], trig, "2002-01-01")
    for inst in ("TLT", "SPY", "TLT-SPY"):
        line(f"{inst} CLOSE D+1 -> CLOSE D+3 (lag1 MOC)", lag1c[inst], trig, "2002-01-01")
        line(f"{inst} CLOSE D -> CLOSE D+3 (lag0 ref)", lag0[inst], trig, "2002-01-01")

# all-quarter conditional vs unconditional for the spread, by regime of the spread
print("\n  spread-bucket table (TLT-SPY, Open D+1 -> Close D+3, 2002+):")
tab = pd.DataFrame({"spread": qt["spread"], "ret": sp["D+3"]["TLT-SPY"].reindex(qt.index)}).dropna()
tab["bucket"] = pd.cut(tab["spread"], [-1, -0.08, -0.03, 0.03, 0.08, 0.10, 1])
for b, g in tab.groupby("bucket", observed=True):
    u, d, n = rec(g["ret"].values)
    print(f"    {str(b):<16} n={n} {u}-{d} mean {100*g['ret'].mean():+.3f}% med "
          f"{100*g['ret'].median():+.3f}%")

if "SPY" in YF:
    ys = YF["SPY"]
    ycal = pd.DatetimeIndex(ys.index)
    yqt = quarter_table(ycal, {"SPY": ys["Close"]})
    print("\n  ---- SPY 1993+ calendar stat (yfinance SPY, adjusted) ----")
    q3 = yqt.index[yqt["month"] == 9]
    q3 = q3[q3 >= "1993-01-01"]
    ally = yqt.index[yqt.index >= "1993-01-01"]
    for nm, trig in (("Q3 final 3 sessions", q3), ("all quarters final 3 sessions", ally)):
        block(f"SPY {nm} OPEN D+1 -> CLOSE D+3", g_o2c_in(ys, 3), trig, "1993-01-01",
              cal=ycal, ctrl=False, show_dates=(nm.startswith("Q3")))
        line(f"SPY {nm} OPEN D+1 -> CLOSE D+2", g_o2c_in(ys, 2), trig, "1993-01-01")
        line(f"SPY {nm} CLOSE D -> CLOSE D+3 (lag0)", g_c2c0(ys, 3), trig, "1993-01-01")
        line(f"SPY {nm} CLOSE D+2 -> CLOSE D+3 (last session alone)",
             ys["Close"].shift(-3) / ys["Close"].shift(-2) - 1, trig, "1993-01-01")
    oth = ally.difference(q3)
    line("SPY non-Q3 quarters OPEN D+1 -> CLOSE D+3", g_o2c_in(ys, 3), oth, "1993-01-01")
    line("SPY all sessions OPEN D+1 -> CLOSE D+3 (day control)",
         g_o2c_in(ys, 3), ycal[(ycal >= "1993-01-01") & (ycal < ASOF - pd.Timedelta(days=7))],
         "1993-01-01")

# ===========================================================================
# B. IWM worst 63d while SPY near its high
# ===========================================================================
print("\n" + "=" * 78)
print("=== B. IWM 63d rank <= 2 (trailing 252) AND SPY within 1% of 252d closing "
      "high -> declustered 21 ===")
iwm_rk = roll_rank(cl("IWM"), 63).reindex(nyse)
spy_dh = dist_hi(cl("SPY")).reindex(nyse)
print(f"  TONIGHT: IWM 63d ret {100*(cl('IWM').iloc[-1]/cl('IWM').iloc[-64]-1):+.2f}% "
      f"rank {iwm_rk.iloc[-1]:.2f} | SPY vs 252d high {100*spy_dh.iloc[-1]:+.2f}%")
for nm, rk_cut, dh_cut in (("strict rank<=2, SPY within 1%", 2, -0.01),
                           ("relaxed rank<=5, SPY within 2%", 5, -0.02)):
    mask = (iwm_rk <= rk_cut) & (spy_dh >= dh_cut)
    epi = on_cal(mask, nyse, 21, "2000-01-01")
    epi = epi[epi < ASOF]
    raw_n = int(mask.fillna(False).sum())
    print(f"\n  ##### {nm}: raw qualifying sessions {raw_n} -> declustered(21) "
          f"{len(epi)} (ex tonight) #####")
    for h in (5, 10, 21):
        block(f"B IWM h{h} OPEN D+1", o2c("IWM", h), epi, "2000-01-01")
    for h in (5, 10, 21):
        block(f"B IWM-SPY h{h} OPEN D+1", o2c("IWM", h) - o2c("SPY", h), epi, "2000-01-01")
    block("B SPY h21 OPEN D+1", o2c("SPY", 21), epi, "2000-01-01")
    line("IWM h21 CLOSE D+1 (lag1 MOC)", c2c("IWM", 21), epi, "2000-01-01")
    line("IWM h21 CLOSE D (lag0 ref)", c2c0("IWM", 21), epi, "2000-01-01")
    line("IWM-SPY h21 CLOSE D (lag0 ref)", c2c0("IWM", 21) - c2c0("SPY", 21), epi, "2000-01-01")

if "^RUT" in YF and "^GSPC" in YF:
    gs = YF["^GSPC"]
    gcal = pd.DatetimeIndex(gs.index)
    rt = YF["^RUT"].reindex(gcal)
    print(f"\n  ---- ^RUT vs ^GSPC 1988+ (yfinance) | RUT NaN rows on GSPC calendar "
          f"{int(rt['Close'].isna().sum())} ----")
    rrk = roll_rank(rt["Close"], 63).reindex(gcal)
    gdh = dist_hi(gs["Close"]).reindex(gcal)
    print(f"  TONIGHT: RUT rank {rrk.iloc[-1]:.2f} GSPC vs high {100*gdh.iloc[-1]:+.2f}%")
    for nm, rk_cut, dh_cut in (("strict", 2, -0.01), ("relaxed", 5, -0.02)):
        epi = on_cal((rrk <= rk_cut) & (gdh >= dh_cut), gcal, 21, "1988-01-01")
        epi = epi[epi < ASOF]
        print(f"\n  ##### RUT/GSPC {nm}: declustered(21) {len(epi)} "
              f"(pre-2000 {int((epi < '2000-01-01').sum())}) #####")
        for h in (5, 10, 21):
            block(f"RUT h{h} OPEN D+1 ({nm})", g_o2c(rt, h), epi, "1988-01-01", cal=gcal)
        for h in (5, 10, 21):
            block(f"RUT-GSPC h{h} OPEN D+1 ({nm})", g_o2c(rt, h) - g_o2c(gs, h), epi,
                  "1988-01-01", cal=gcal)
        line(f"RUT h21 CLOSE D+1 lag1 ({nm})", g_c2c(rt, 21), epi, "1988-01-01")
        line(f"RUT-GSPC h21 CLOSE D+1 lag1 ({nm})", g_c2c(rt, 21) - g_c2c(gs, 21), epi, "1988-01-01")
        pre = epi[epi < "2000-01-01"]
        line(f"RUT h21 OPEN D+1 pre-2000 only ({nm})", g_o2c(rt, 21), pre, "1988-01-01")
        line(f"RUT-GSPC h21 OPEN D+1 pre-2000 only ({nm})", g_o2c(rt, 21) - g_o2c(gs, 21), pre, "1988-01-01")

# ===========================================================================
# C. MOVE drop from a high level
# ===========================================================================
print("\n" + "=" * 78)
print("=== C. ^MOVE 1d <= -8% with prior close >= 90th pct of trailing 252 -> "
      "declustered 5, 2002+ ===")
mv = cl("^MOVE")
mchg = mv.pct_change(fill_method=None)
mrk = (mv.rolling(252, min_periods=200).rank(pct=True) * 100).shift(1)
mmask = (mchg <= -0.08) & (mrk >= 90)
print(f"  TONIGHT: MOVE {mv.iloc[-1]:.2f} prev {mv.iloc[-2]:.2f} chg {100*mchg.iloc[-1]:+.2f}% "
      f"prior-close rank {mrk.iloc[-1]:.1f} trigger {bool(mmask.iloc[-1])} | MOVE rows "
      f"not on SPY calendar {len(mv.index.difference(nyse))}, SPY sessions missing MOVE "
      f"{int(mv.reindex(nyse)[nyse >= mv.index[0]].isna().sum())}")
vix = F["^VIX"]["Close"]
print(f"  TONIGHT VIX {vix.iloc[-1]:.2f}")
for nm, mk in (("all", mmask.reindex(nyse)),
               ("VIX < 20 at D", mmask.reindex(nyse) & (vix < 20))):
    epi = on_cal(mk, nyse, 5, "2002-01-01")
    epi = epi[epi < ASOF]
    print(f"\n  ##### MOVE drop [{nm}]: declustered(5) {len(epi)} #####")
    for tk in ("SPY", "TLT", "IEF"):
        for h in (5, 10):
            block(f"C {tk} h{h} OPEN D+1 [{nm}]", o2c(tk, h), epi, "2002-01-01",
                  show_dates=(h == 5 and len(epi) <= 30))
        line(f"{tk} h5 CLOSE D (lag0 ref)", c2c0(tk, 5), epi, "2002-01-01")

# ===========================================================================
# D. Utilities washout with SPY flat-to-up
# ===========================================================================
print("\n" + "=" * 78)
print("=== D. XLU 21d <= -8% AND SPY 21d >= 0 -> declustered 21, 1999+ ===")
x = cl("XLU")
x21 = (x / x.shift(21) - 1).reindex(nyse)
s21 = (cl("SPY") / cl("SPY").shift(21) - 1).reindex(nyse)
xlo = dist_lo(x).reindex(nyse)
print(f"  TONIGHT: XLU 21d {100*x21.iloc[-1]:+.2f}% SPY 21d {100*s21.iloc[-1]:+.2f}% "
      f"XLU vs 252d low {100*xlo.iloc[-1]:+.2f}%")
for nm, mk in (("base", (x21 <= -0.08) & (s21 >= 0)),
               ("AND XLU within 1% of 252d low", (x21 <= -0.08) & (s21 >= 0) & (xlo <= 0.01)),
               ("reference: XLU 21d <= -8% any SPY", (x21 <= -0.08))):
    epi = on_cal(mk, nyse, 21, "1999-01-01")
    epi = epi[epi < ASOF]
    print(f"\n  ##### XLU washout [{nm}]: declustered(21) {len(epi)} #####")
    for h in (5, 10, 21):
        block(f"D XLU h{h} OPEN D+1 [{nm}]", o2c("XLU", h), epi, "1999-01-01")
    block(f"D XLU-SPY h21 OPEN D+1 [{nm}]", o2c("XLU", 21) - o2c("SPY", 21), epi, "1999-01-01")
    lag2 = lambda h: F["XLU"]["Close"].shift(-(2 + h)) / F["XLU"]["Open"].shift(-2) - 1  # noqa: E731
    for h in (5, 10):
        line(f"XLU h{h} OPEN D+2 (lag2: live run began 9/24, Monday = D+2)", lag2(h), epi, "1999-01-01")
    line("XLU h10 CLOSE D+1 (lag1 MOC)", c2c("XLU", 10), epi, "1999-01-01")
    line("XLU h10 CLOSE D (lag0 ref)", c2c0("XLU", 10), epi, "1999-01-01")

# ===========================================================================
# E. Ten-year streak of 52w closing highs
# ===========================================================================
print("\n" + "=" * 78)
print("=== E. ^TNX 252d closing high 3 sessions running (D = third) -> "
      "declustered 10 on the TNX calendar ===")


def tnx_trig(tn: pd.Series) -> tuple[pd.Series, pd.Series]:
    hi = (tn >= tn.rolling(252, min_periods=200).max() - 1e-9)
    tri = hi & hi.shift(1, fill_value=False) & hi.shift(2, fill_value=False)
    first = tri & ~tri.shift(1, fill_value=False)
    return hi, tri


tn_c = cl("^TNX")
hi_c, tri_c = tnx_trig(tn_c)
print("  cache TNX last 6: " + ", ".join(
    f"{d.date()} {v:.3f}{'H' if hi_c[d] else ''}{'*' if tri_c[d] else ''}"
    for d, v in tn_c.tail(6).items()) + "  (H = 252d high, * = 3-run)")
bp5 = lambda tn: (tn.shift(-6) - tn.shift(-1)) * 100  # noqa: E731


def bp_line(label: str, tn: pd.Series, epi: pd.DatetimeIndex) -> None:
    v = bp5(tn).reindex(epi).dropna()
    u, d, n = rec(v.values)
    ctrl = bp5(tn)[tn.index >= epi[0]].dropna()
    print(f"    {label}: n={n} up {u} down {d} mean {v.mean():+.1f}bp med "
          f"{v.median():+.1f}bp | ctrl all {ctrl.mean():+.2f}bp | "
          f"pre-2018 {v[v.index < ERA].mean():+.1f}bp (n {int((v.index < ERA).sum())}) "
          f"2018+ {v[v.index >= ERA].mean():+.1f}bp (n {int((v.index >= ERA).sum())})")


epiE_c = declusters(tn_c.index[tri_c.values], 10, tn_c.index)
epiE_c = epiE_c[(epiE_c >= "2003-01-01") & (epiE_c < ASOF)]
print(f"  tonight is a 3-run day {bool(tri_c.iloc[-1])}; is it the FIRST of its run "
      f"{bool(tri_c.iloc[-1] and not tri_c.iloc[-2])}")
off = [d for d in epiE_c if d not in POS.index]
epiE_c = pd.DatetimeIndex([d for d in epiE_c if d in POS.index])
print(f"\n  ##### 2003+ (cache): declustered(10) {len(epiE_c)} | off SPY calendar {off} #####")
for tk in ("SPY", "TLT", "IWM"):
    for h in (5, 10):
        block(f"E {tk} h{h} OPEN D+1 [2003+]", o2c(tk, h), epiE_c, "2003-01-01",
              show_dates=(h == 5 and len(epiE_c) <= 30))
    line(f"{tk} h5 CLOSE D (lag0 ref)", c2c0(tk, 5), epiE_c, "2003-01-01")
bp_line("TNX bp change Close D+1 -> Close D+6 (lag1, h5)", tn_c.reindex(nyse), epiE_c)

if "^TNX" in YF:
    tn_y = YF["^TNX"]["Close"]
    hi_y, tri_y = tnx_trig(tn_y)
    epiE_y = declusters(tn_y.index[tri_y.values], 10, tn_y.index)
    epiE_y = epiE_y[(epiE_y >= "1990-01-01") & (epiE_y < ASOF)]
    print(f"\n  ##### 1990+ (yfinance TNX): declustered(10) {len(epiE_y)} "
          f"(pre-2003 {int((epiE_y < '2003-01-01').sum())}) #####")
    gs = YF["^GSPC"]
    ys = YF["SPY"]
    gcal = pd.DatetimeIndex(gs.index)
    e_g = pd.DatetimeIndex([d for d in epiE_y if d in gcal])
    for h in (5, 10):
        block(f"E ^GSPC h{h} CLOSE D+1 lag1 [1990+]", g_c2c(gs, h), e_g, "1990-01-01",
              cal=gcal, show_dates=False)
        block(f"E ^GSPC h{h} OPEN D+1 [1990+]", g_o2c(gs, h), e_g, "1990-01-01",
              cal=gcal, show_dates=False)
        block(f"E SPY(yf) h{h} OPEN D+1 [1993+]", g_o2c(ys, h), e_g, "1993-01-01",
              cal=pd.DatetimeIndex(ys.index), show_dates=False)
    e_n = pd.DatetimeIndex([d for d in epiE_y if d in POS.index])
    for tk in ("TLT", "IWM"):
        for h in (5, 10):
            block(f"E {tk} h{h} OPEN D+1 [1990+ triggers, data-limited]", o2c(tk, h),
                  e_n, "2000-01-01", show_dates=False)
    bp_line("TNX bp change lag1 h5 [1990+]", tn_y, epiE_y)
    print("    pre-2000 triggers: " + ", ".join(str(d.date()) for d in epiE_y[epiE_y < "2000-01-01"]))

# ===========================================================================
# F. USO September Fridays
# ===========================================================================
print("\n" + "=" * 78)
print("=== F. USO Fridays, 2006+ through 2026-09-25 (Sep vs other Fridays) ===")
u = raw["USO"]
u = u[(u.index <= ASOF) & (u.index >= "2006-01-01")].astype(float)
oc = (u["Close"] / u["Open"] - 1.0)
cc = u["Close"].pct_change(fill_method=None)
fri = u.index.weekday == 4
for lab, s in (("close->close", cc), ("open->close", oc)):
    s = s[fri].dropna()
    sep, oth = s[s.index.month == 9], s[s.index.month != 9]
    for nm, v in (("Sep Fridays", sep), ("other Fridays", oth)):
        up, dn, n = rec(v.values)
        sm = summarize(v.values)
        print(f"  {lab:<14} {nm:<14} n={n} {up}-{dn} ({n-up-dn} flat) mean "
              f"{sm['mean_pct']:+.3f}% med {sm['median_pct']:+.3f}% t {sm['t']:+.2f} "
              f"sign p(down) {sign_test(dn, up + dn):.4f}")
        print(f"      era: {era_str(v)}")
    wt = (sep.mean() - oth.mean()) / np.sqrt(sep.var() / len(sep) + oth.var() / len(oth))
    print(f"      Welch t Sep vs other Fridays: {wt:+.2f}")
    s26 = sep[sep.index.year == 2026]
    print(f"      Sep 2026 Fridays: " + ", ".join(f"{d.date()} {100*x:+.2f}%" for d, x in s26.items())
          + f" -> {int((s26 < 0).sum())} down of {len(s26)}")

# ===========================================================================
# H. Frozen levels
# ===========================================================================
print("\n" + "=" * 78)
print("=== H. Frozen levels at the 2026-09-25 close (adjusted bars, Wilder-14) ===")
for t in ("TLT", "IEF", "IWM", "SPY", "XLU", "QQQ"):
    f = raw[t]
    f = f[f.index <= ASOF].dropna(subset=["High", "Low", "Close"]).astype(float)
    a = float(np.asarray(wilder_atr(f["High"].to_numpy(), f["Low"].to_numpy(),
                                    f["Close"].to_numpy()), float)[-1])
    c = float(f["Close"].iloc[-1])
    print(f"  {t}: bar {f.index[-1].date()} close {c:.4f} ATR {a:.4f} ({100*a/c:.2f}%) "
          f"open {f['Open'].iloc[-1]:.4f}")

# ===========================================================================
# I. Open-idea marks
# ===========================================================================
print("\n" + "=" * 78)
print("=== I. Open idea marks (adjusted bars, approximate) ===")
iw = raw["IWM"]
e, xcl = float(iw.loc["2026-09-18", "Close"]), float(iw.loc["2026-09-25", "Close"])
print(f"  x20260917-1 SHORT IWM MOC 9/18 -> 9/25 close: entry {e:.4f} (ref 285.43) exit "
      f"{xcl:.4f} | short ret {100*-(xcl/e-1):+.3f}% | R "
      f"{(e-xcl)/3.5221:+.3f} (vs ref_close: {(285.43-xcl)/3.5221:+.3f})")
xe = raw["XLE"]
e, m = float(xe.loc["2026-09-22", "Close"]), float(xe.loc["2026-09-25", "Close"])
print(f"  x20260921-1 SHORT XLE MOC 9/22, out 9/29: entry {e:.4f} (ref 62.46) mark 9/25 "
      f"{m:.4f} | short ret {100*-(m/e-1):+.3f}% | R {(e-m)/1.3245:+.3f} (vs ref: "
      f"{(62.46-m)/1.3245:+.3f})")
ie = raw["IEF"]
o, m = float(ie.loc["2026-09-25", "Open"]), float(ie.loc["2026-09-25", "Close"])
print(f"  x20260924-1 LONG IEF MOO 9/25, out 9/30: open fill {o:.4f} mark 9/25 {m:.4f} | "
      f"ret {100*(m/o-1):+.3f}% | R {(m-o)/0.4516:+.3f}")
print("\ndone.")
