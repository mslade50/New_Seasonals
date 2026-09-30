"""Posts check (2026-09-27, Sunday): idea and stat candidates for tonight's queue.

ASOF = Friday 2026-09-25 (freshest bar). Next session Monday 2026-09-28.
9/28, 9/29, 9/30 are September's (and Q3's) final three sessions; 10/1 and
10/2 are October's first two. Every idea return is LAG-1 (entry at the D+1
open, exit at a later close). SPY calendar from the cache is the master index.
September 2026 is incomplete and is EXCLUDED from every month-end sample.

A. TLT month-end after a bad month (idea candidate). D = 4th-last session of
   the month. Condition TLT MTD at D <= -2.93% (brief's worst-quintile cut).
B. Midterm Q4 turn: reproduce brief cell 03 (^GSPC 2000-2025), extend 1950+.
C. SPY Sep->Oct turn window: reproduce brief cell 08/09 (Close D -> Close D+5).
D. Sector breadth near a high: reproduce brief cell 04.
E. Scoreboard detail from data/posts_journal.jsonl (read only).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pitch_lab import (  # noqa: E402
    cluster_note, declusters, era_split, fwd_ret, load_prices, local_control,
    sign_test, summarize, wilder_atr,
)

ROOT = Path(__file__).resolve().parents[3]
ASOF = pd.Timestamp("2026-09-25")
ERA = pd.Timestamp("2018-01-01")
SECT9 = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
ETF = ["SPY", "TLT", "IEF"]
TK = ETF + ["^GSPC"] + SECT9

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


F = {t: frame(t) for t in ETF}


def atr_series(t: str) -> pd.Series:
    f = raw[t]
    f = f[f.index <= ASOF].dropna(subset=["High", "Low", "Close"]).astype(float)
    a = wilder_atr(f["High"].to_numpy(), f["Low"].to_numpy(), f["Close"].to_numpy())
    return pd.Series(np.asarray(a, float), index=f.index).reindex(nyse)


ATR = {t: atr_series(t) for t in ETF}

# ---------------------------------------------------------------------------
# yfinance supplement (^GSPC 1950+)
# ---------------------------------------------------------------------------
YF: dict[str, pd.DataFrame] = {}
try:
    import yfinance as yf
    ydl = yf.download(["^GSPC"], start="1950-01-01", end="2026-09-26",
                      auto_adjust=True, progress=False)
    for t in ["^GSPC"]:
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
        print(f"  {t}: {f.index[0].date()} .. {f.index[-1].date()} n={len(f)} | "
              f"Open<=0/NaN {bad} | Open==Close {eq} (closes only used below)")
except Exception as exc:  # noqa: BLE001
    print(f"\nyfinance supplement FAILED: {exc!r}")


# ---------------------------------------------------------------------------
# forward returns on the SPY calendar, aligned to the signal date D
# ---------------------------------------------------------------------------
def g_o2c_in(f: pd.DataFrame, h: int) -> pd.Series:
    return f["Close"].shift(-h) / f["Open"].shift(-1) - 1.0


def g_c2c0(f: pd.DataFrame, h: int) -> pd.Series:
    c = f["Close"]
    return c.shift(-h) / c - 1.0


def o2c_in(t: str, h: int) -> pd.Series:
    """Open[D+1] -> Close[D+h] (entry session counted as session 1)."""
    return g_o2c_in(F[t], h)


def o2c(t: str, h: int) -> pd.Series:
    """Open[D+1] -> Close[D+1+h]."""
    return F[t]["Close"].shift(-(1 + h)) / F[t]["Open"].shift(-1) - 1.0


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
        out.append(f"{lab} n={n} {u}-{d} mean {100*x.mean():+.3f}%")
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
          f"{sm['median_pct']:+.3f}% t {tt} | ctrl all-days {100*allv.mean():+.3f}% "
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
    sm = summarize(v.values)
    print(f"    {label}: n={n} {u}-{d} mean {100*v.mean():+.3f}% med "
          f"{100*v.median():+.3f}% t {sm['t']:+.2f} sign p(up) {sign_test(u, n):.4f} "
          f"p(down) {sign_test(d, n):.4f} | {era_str(v)}")


# ---------------------------------------------------------------------------
# month anchors on the SPY calendar: D = 4th-last session, complete months only
# ---------------------------------------------------------------------------
def month_anchors(cal: pd.DatetimeIndex, need_next: int = 2) -> pd.DataFrame:
    ym = np.asarray(cal.year * 100 + cal.month)
    cur = ASOF.year * 100 + ASOF.month
    rows = []
    for m in pd.unique(ym):
        if m == cur:
            continue
        ix = np.where(ym == m)[0]
        if len(ix) < 5 or ix[-1] + need_next >= len(cal):
            continue
        rows.append({"ym": int(m), "mo": int(m % 100), "anchor": cal[ix[-4]],
                     "last": cal[ix[-1]], "prev_end": cal[ix[0] - 1] if ix[0] > 0 else pd.NaT,
                     "n2": cal[ix[-1] + 2]})
    return pd.DataFrame(rows).set_index("anchor")


MA = month_anchors(nyse)
print(f"\nmonth anchors (SPY cal): {len(MA)} months {MA['ym'].iloc[0]}..{MA['ym'].iloc[-1]} | "
      f"anchor->last always 3: {bool(all(POS[MA['last']].values - POS[MA.index].values == 3))} "
      f"| anchor->n2 always 5: {bool(all(POS[MA['n2']].values - POS[MA.index].values == 5))}")

# ===========================================================================
# A. TLT month-end after a bad month
# ===========================================================================
print("\n" + "=" * 78)
print("=== A. TLT MONTH-END AFTER A BAD MONTH: D = 4th-last session; TLT MTD at D "
      "<= -2.93% ===")
tc = F["TLT"]["Close"]
A = MA.copy()
A["mtd"] = tc.reindex(A.index).values / tc.reindex(A["prev_end"]).values - 1.0
A = A[tc.reindex(A.index).notna().values]
A = A[(A["ym"] >= 200208) & (A["ym"] <= 202608)]
qcut = float(A["mtd"].quantile(0.2))
print(f"  months with TLT at D: {len(A)} ({A['ym'].iloc[0]}..{A['ym'].iloc[-1]}), "
      f"mtd NaN {int(A['mtd'].isna().sum())} | my worst-quintile cut {100*qcut:.3f}% "
      f"(brief -2.93%)")
bad_q = pd.DatetimeIndex(A.index[A["mtd"] <= qcut])
bad = pd.DatetimeIndex(A.index[A["mtd"] <= -0.0293])
allm = pd.DatetimeIndex(A.index)
print(f"  n bad (<= my cut) {len(bad_q)} | n bad (<= -2.93% rounded) {len(bad)} | identical sets "
      f"{bool(bad.equals(bad_q))} | in quintile but not in rounded cut: "
      f"{[(str(x.date()), round(100*A.loc[x, 'mtd'], 4)) for x in bad_q.difference(bad)]}")
bad_round = bad
bad = bad_q  # the brief's set: mtd <= its computed 20th percentile

# September 2026 state
sep26 = nyse[(nyse.year == 2026) & (nyse.month == 9)]
aug_end = nyse[nyse < "2026-09-01"][-1]
try:
    from trading_calendar import TRADING_DAY
    nxt5 = [ASOF + k * TRADING_DAY for k in range(1, 6)]
    print(f"  next sessions per trading_calendar: {[str(d.date()) for d in nxt5]}")
    rem = [d for d in nxt5 if d.month == 9]
    print(f"  Sept 2026 sessions in cache {len(sep26)} (last {sep26[-1].date()}); remaining "
          f"Sept sessions {len(rem)} -> total {len(sep26) + len(rem)}; 2026-09-25 is the "
          f"{len(rem) + 1}th-last September session: {len(rem) == 3}")
except Exception as exc:  # noqa: BLE001
    print(f"  trading_calendar import failed {exc!r}")
mtd26 = tc[ASOF] / tc[aug_end] - 1
print(f"  TONIGHT: TLT close {tc[ASOF]:.4f} vs Aug end {aug_end.date()} {tc[aug_end]:.4f} "
      f"-> MTD {100*mtd26:+.3f}% | qualifies (<= -2.93%) {bool(mtd26 <= -0.0293)} | "
      f"rank of this MTD among the {len(A)} anchors: "
      f"{int((A['mtd'] <= mtd26).sum())} months at or below")

tlt_f3 = c2c0("TLT", 3)
tlt_o3 = o2c_in("TLT", 3)
tlt_o5 = o2c_in("TLT", 5)
tlt_n2 = F["TLT"]["Close"].shift(-5) / F["TLT"]["Close"].shift(-3) - 1.0
tlt_h5 = c2c0("TLT", 5)
S0 = "2002-08-01"

for nm, trig in ((f"BAD MONTHS (MTD <= quintile {100*qcut:.3f}%)", bad),
                 ("BAD MONTHS rounded cut (MTD <= -2.9300%)", bad_round),
                 ("ALL MONTHS (control)", allm)):
    sd = False
    print(f"\n  ##### {nm}: n anchors {len(trig)} #####")
    block("A1 brief form TLT CLOSE D -> CLOSE D+3 (lag0)", tlt_f3, trig, S0, show_dates=sd)
    block("A2 IDEA TLT OPEN D+1 -> CLOSE D+3 (Mon MOO, out Wed 9/30 close)", tlt_o3, trig, S0,
          show_dates=sd)
    block("A3 TLT OPEN D+1 -> CLOSE D+5 (hold through give-back)", tlt_o5, trig, S0,
          show_dates=sd)
    block("A4 TLT next2 CLOSE D+3 -> CLOSE D+5", tlt_n2, trig, S0, show_dates=sd)
    line("TLT CLOSE D -> CLOSE D+5 (lag0 h5)", tlt_h5, trig, S0)

# explicit brief reconciliation line
v = tlt_f3.reindex(bad).dropna()
u, d, n = rec(v.values)
print(f"\n  RECONCILE brief (n=58, 37-21, +0.625%, t 2.78, sign p 0.024): n={n} {u}-{d} "
      f"mean {100*v.mean():+.3f}% t {summarize(v.values)['t']:.2f} sign p {sign_test(u, n):.4f}")
v = tlt_n2.reindex(bad).dropna()
u, d, n = rec(v.values)
print(f"  RECONCILE next2 (brief 20/58 up, -0.358%, t -1.96): n={n} up {u} mean "
      f"{100*v.mean():+.3f}% t {summarize(v.values)['t']:.2f}")

# MAE of the idea form, in ATR at D
L, O = F["TLT"]["Low"], F["TLT"]["Open"]
mn3 = pd.concat([L.shift(-k) for k in (1, 2, 3)], axis=1).min(axis=1)
mn5 = pd.concat([L.shift(-k) for k in (1, 2, 3, 4, 5)], axis=1).min(axis=1)
mae3 = (mn3 - O.shift(-1)) / ATR["TLT"]
mae5 = (mn5 - O.shift(-1)) / ATR["TLT"]
for nm, trig in (("bad", bad), ("all", allm)):
    for lab, m in (("D+1..D+3", mae3), ("D+1..D+5", mae5)):
        x = m.reindex(trig).dropna()
        print(f"  MAE [{nm}] {lab} (worst low vs Open D+1, ATR@D): n {len(x)} mean "
              f"{x.mean():+.2f} med {x.median():+.2f} p10 {x.quantile(0.1):+.2f} worst "
              f"{x.min():+.2f} ({x.idxmin().date()}) | share <= -1 ATR "
              f"{100*(x <= -1).mean():.0f}% <= -1.5 ATR {100*(x <= -1.5).mean():.0f}%")
atr_pct = (ATR["TLT"] / F["TLT"]["Close"]).reindex(bad)
print(f"  TLT ATR/close at bad anchors: median {100*atr_pct.median():.2f}%")

print("\n  bad-month table (anchor, MTD, brief f3, idea o3, o5, next2, MAE3 ATR):")
for dd in bad:
    print(f"    {dd.date()} mtd {100*A.loc[dd, 'mtd']:+.2f} f3 {100*tlt_f3[dd]:+.2f} "
          f"o3 {100*tlt_o3[dd]:+.2f} o5 {100*tlt_o5[dd]:+.2f} n2 {100*tlt_n2[dd]:+.2f} "
          f"mae3 {mae3[dd]:+.2f}")

# IEF side check
print("\n  ---- IEF side check ----")
ief_f3 = c2c0("IEF", 3)
ief_o3 = o2c_in("IEF", 3)
ic = F["IEF"]["Close"]
A["ief_mtd"] = ic.reindex(A.index).values / ic.reindex(A["prev_end"]).values - 1.0
iq = float(A["ief_mtd"].quantile(0.2))
ibad = pd.DatetimeIndex(A.index[A["ief_mtd"] <= iq])
line("IEF brief form CLOSE D -> CLOSE D+3 on TLT-bad months", ief_f3, bad, S0)
line("IEF IDEA form OPEN D+1 -> CLOSE D+3 on TLT-bad months", ief_o3, bad, S0)
line(f"IEF brief form on IEF's own worst quintile (<= {100*iq:.2f}%)", ief_f3, ibad, S0)
line("IEF brief form all months", ief_f3, allm, S0)
print(f"  IEF MTD tonight {100*(ic[ASOF]/ic[aug_end]-1):+.2f}%")

# Frozen levels
print("\n  ---- frozen levels at 2026-09-25 (cache bars, pitch_lab.wilder_atr) ----")
for t in ("TLT", "IEF"):
    f = raw[t]
    f = f[f.index <= ASOF].dropna(subset=["High", "Low", "Close"]).astype(float)
    a = float(np.asarray(wilder_atr(f["High"].to_numpy(), f["Low"].to_numpy(),
                                    f["Close"].to_numpy()), float)[-1])
    c = float(f["Close"].iloc[-1])
    print(f"  {t}: bar {f.index[-1].date()} close {c:.4f} ATR14 {a:.4f} ({100*a/c:.2f}%) "
          f"| {'posts_state TLT atr 0.7425 match ' + str(abs(a - 0.7425) < 5e-4) if t == 'TLT' else ''}")

# ===========================================================================
# B. Midterm Q4
# ===========================================================================
print("\n" + "=" * 78)
print("=== B. MIDTERM Q4: anchor = 4th-last Sept session -> year end ===")


def midterm_table(g: pd.Series, years: range, mid: set[int]) -> pd.DataFrame:
    idx = g.index
    rows = []
    for y in years:
        sep = idx[(idx.year == y) & (idx.month == 9)]
        oct_ = idx[(idx.year == y) & (idx.month == 10)]
        dec = idx[(idx.year == y) & (idx.month == 12)]
        if len(sep) < 4 or len(oct_) < 2 or len(dec) == 0:
            continue
        anc = sep[-4]
        a = g.loc[anc]
        path = g.loc[anc:oct_[-1]]
        rows.append({"year": y, "mid": y in mid, "anchor": anc,
                     "to_oct_end": 100 * (g.loc[oct_[-1]] / a - 1),
                     "to_dec_end": 100 * (g.loc[dec[-1]] / a - 1),
                     "q4": 100 * (g.loc[dec[-1]] / g.loc[sep[-1]] - 1),
                     "dip_by_oct_end": 100 * (path.min() / a - 1)})
    return pd.DataFrame(rows)


def midterm_print(df: pd.DataFrame, tag: str) -> None:
    for lab, m in (("midterm", df["mid"]), ("other", ~df["mid"])):
        d = df[m]
        print(f"  [{tag}] {lab} n {len(d)}:")
        for c in ("to_dec_end", "q4", "to_oct_end", "dip_by_oct_end"):
            v = d[c]
            print(f"      {c:15s} {int((v > 0).sum())}-{int((v < 0).sum())} mean {v.mean():+.2f}% "
                  f"median {v.median():+.2f}%")
    d = df[df["mid"]]
    up = int((d["to_dec_end"] > 0).sum())
    print(f"  [{tag}] midterm to_dec_end sign p {sign_test(up, len(d)):.4f}; vs other-year "
          f"up-rate {sign_test(up, len(d), float((df[~df['mid']]['to_dec_end'] > 0).mean())):.4f}")


gc = raw["^GSPC"]
gc = gc[gc.index <= ASOF]["Close"].astype(float).dropna()
MID6 = {2002, 2006, 2010, 2014, 2018, 2022}
b1 = midterm_table(gc, range(2000, 2026), MID6)
print("  -- cache ^GSPC 2000-2025 (brief: mid 5/6 +2.83%, other 16/20 +4.33%; Q4 mid 5/6 "
      "+3.63% vs 16/20 +4.28%; median dip -3.59% vs -2.02%) --")
midterm_print(b1, "2000-2025")
print("  midterm rows: " + ", ".join(
    f"{int(r.year)} to_dec {r.to_dec_end:+.2f} q4 {r.q4:+.2f} oct {r.to_oct_end:+.2f}"
    for r in b1[b1["mid"]].itertuples()))

if "^GSPC" in YF:
    gy = YF["^GSPC"]["Close"]
    yrs = range(1950, 2026)
    b2 = midterm_table(gy, yrs, {y for y in yrs if y % 4 == 2})
    print(f"\n  -- yfinance ^GSPC 1950-2025 (n years {len(b2)}) --")
    midterm_print(b2, "1950-2025")
    print("  midterm list (anchor -> Dec 31 | Q4 from Sep end | anchor -> Oct end | dip):")
    for r in b2[b2["mid"]].itertuples():
        print(f"    {r.year} anchor {r.anchor.date()} to_dec {r.to_dec_end:+.2f}% q4 {r.q4:+.2f}% "
              f"to_oct {r.to_oct_end:+.2f}% dip {r.dip_by_oct_end:+.2f}%")
    cmp = b2[(b2["year"] >= 2000)].set_index("year")["to_dec_end"] - b1.set_index("year")["to_dec_end"]
    print(f"  yf vs cache 2000-2025 to_dec_end max abs diff {cmp.abs().max():.3f}pp")
    for lab, lo, hi in (("1950-1999", 1950, 1999), ("2000-2025", 2000, 2025)):
        d = b2[(b2["year"] >= lo) & (b2["year"] <= hi)]
        for ml, m in (("mid", d["mid"]), ("oth", ~d["mid"])):
            v = d[m]["to_dec_end"]
            print(f"    {lab} {ml}: n {len(v)} {int((v > 0).sum())}-{int((v < 0).sum())} "
                  f"mean {v.mean():+.2f}%")

# ===========================================================================
# C. SPY turn window Close D -> Close D+5
# ===========================================================================
print("\n" + "=" * 78)
print("=== C. SPY TURN WINDOW Close[D] -> Close[D+5] (D = 4th-last session), one per month ===")
sc = F["SPY"]["Close"]
C = MA[(MA["ym"] >= 200001) & (MA["ym"] <= 202608)].copy()
C["h5"] = sc.reindex(C["n2"]).values / sc.reindex(C.index).values - 1
C["o5"] = o2c_in("SPY", 5).reindex(C.index).values
C["f3"] = sc.reindex(C["last"]).values / sc.reindex(C.index).values - 1
for lab, d in (("all months", C), ("2018+ all months", C[C.index >= ERA]),
               ("September", C[C["mo"] == 9]), ("September 2018+", C[(C["mo"] == 9) & (C.index >= ERA)]),
               ("non-September", C[C["mo"] != 9])):
    v = d["h5"]
    u, dn, n = rec(v.values)
    sm = summarize(v.values)
    w = d["o5"]
    wu, wd, wn = rec(w.values)
    print(f"  {lab:18s} n {n} {u}-{dn} mean {100*v.mean():+.3f}% med {100*v.median():+.3f}% "
          f"t {sm['t']:+.2f} | lag1 Open D+1->Close D+5 {wu}-{wd} mean {100*w.mean():+.3f}%")
sep = C[C["mo"] == 9]["h5"]
sep.index = sep.index.year
print("  September by year: " + ", ".join(f"{y} {100*x:+.2f}" for y, x in sep.items()))
w2 = sep.nsmallest(2)
print(f"  two worst: {', '.join(f'{y} {100*x:+.2f}%' for y, x in w2.items())} | Sept mean ex "
      f"those two {100*sep.drop(w2.index).mean():+.3f}% ({int((sep.drop(w2.index) > 0).sum())}-"
      f"{int((sep.drop(w2.index) < 0).sum())})")
r5 = (sc.shift(-5) / sc - 1).dropna()
print(f"  control every 5-session window {100*r5.mean():+.3f}% up {100*(r5 > 0).mean():.1f}%")

# ===========================================================================
# D. Sector breadth near a high
# ===========================================================================
print("\n" + "=" * 78)
print("=== D. SPY within 1% of 252d closing high x count of 9 sector SPDRs above 200d ===")
px = pd.DataFrame({t: raw[t]["Close"] for t in ["SPY"] + SECT9})
px = px[px.index <= ASOF].reindex(nyse).astype(float)
spy = px["SPY"]
above = pd.DataFrame({t: px[t] > px[t].rolling(200, min_periods=200).mean() for t in SECT9})
valid = pd.DataFrame({t: px[t].rolling(200, min_periods=200).mean().notna() for t in SECT9}).all(axis=1)
count = above.sum(axis=1).where(valid)
dist = spy / spy.rolling(252, min_periods=200).max() - 1
near = (dist >= -0.01) & valid
print(f"  today {nyse[-1].date()}: SPY vs 252d high {100*dist.iloc[-1]:+.2f}% | count "
      f"{int(count.iloc[-1])} of 9 | above: {[t for t in SECT9 if above[t].iloc[-1]]} | below: "
      f"{[t for t in SECT9 if not above[t].iloc[-1]]}")
print(f"  count 21 sessions ago ({nyse[-22].date()}): {int(count.iloc[-22])} of 9, above then: "
      f"{[t for t in SECT9 if above[t].iloc[-22]]} | 63 ago: {int(count.iloc[-64])}")
print(f"  near-high sessions (all 9 SMAs valid): {int(near.sum())} (first {nyse[near.values][0].date()})"
      f" | brief 2,291")
print("  count distribution on near-high days: " + ", ".join(
    f"{int(k)}:{int(v)}" for k, v in count[near].value_counts().sort_index().items()))
lo4 = nyse[(near & (count <= 4)).values]
print(f"  near-high with <= 4 of 9: {len(lo4)} -> {[str(d.date()) for d in lo4]}")


def fwd_block(mask: pd.Series, label: str, gap: int = 21) -> pd.DatetimeIndex:
    trig = nyse[mask.fillna(False).values]
    trig = trig[trig < nyse[-1]]
    epi = declusters(trig, gap, nyse)
    print(f"\n  ##### {label}: raw {len(trig)}, declustered({gap}) {len(epi)}")
    if len(epi) <= 30:
        print("     dates: " + ", ".join(str(d.date()) for d in epi))
    for h in (21, 63):
        r = fwd_ret(spy, h)
        v = r.reindex(epi).dropna()
        sm = summarize(v.values)
        u, dn, n = rec(v.values)
        loc = local_control(nyse, epi, 126)
        print(f"     SPY h{h} lag0 Close D: n {n} {u}-{dn} mean {sm['mean_pct']:+.3f}% med "
              f"{sm['median_pct']:+.3f}% t {sm['t']:+.2f} | all-days {100*r.mean():+.3f}% | "
              f"local {100*r.reindex(loc).mean():+.3f}% | era "
              f"{[(e['label'], e['n'], round(e.get('mean_pct', np.nan), 2)) for e in era_split(v.index, v.values)]}")
        w = o2c("SPY", h).reindex(epi).dropna()
        wu, wd, wn = rec(w.values)
        print(f"     SPY h{h} lag1 Open D+1: n {wn} {wu}-{wd} mean {100*w.mean():+.3f}% "
              f"t {summarize(w.values)['t']:+.2f}")
    return epi


fwd_block(near & (count <= 4), "near high, <= 4 of 9")
fwd_block(near & (count >= 5) & (count <= 6), "near high, 5-6 of 9 (brief 16 episodes, h21 "
          "+1.09% 10-6, h63 +1.41% 10-6 t 0.77)")
fwd_block(near & (count >= 7), "near high, >= 7 of 9 (brief 158 episodes, h63 +2.11%)")
print("  last 30 sessions count: " + str(count.tail(30).astype(int).tolist()))

# ===========================================================================
# E. Scoreboard detail
# ===========================================================================
print("\n" + "=" * 78)
print("=== E. SCOREBOARD (data/posts_scoreboard.json + folded posts_journal.jsonl) ===")
sb = json.loads((ROOT / "data" / "posts_scoreboard.json").read_text())
lt = sb["lifetime"]
print(f"  scoreboard asof {sb['asof']} lifetime: n {lt['n']} graded {lt['graded']} avg "
      f"{lt['avg_r']} med {lt['median_r']} hit {lt['hit_rate']}% total {lt['total_r']} | "
      f"posted n {lt['posted']['n']}")
import posts_journal  # noqa: E402
from trading_calendar import TRADING_DAY  # noqa: E402

recs = [json.loads(x) for x in (ROOT / "data" / "posts_journal.jsonl").read_text().splitlines()
        if x.strip()]
ideas = [d for d in posts_journal.fold_drafts(recs) if d.get("type") == "idea"]
graded, opn = [], []
for d in ideas:
    i, o = d["idea"], d.get("outcome") or {}
    if o.get("r_multiple") is not None:
        graded.append((d["draft_id"], i["ticker"], i["side"], float(o["r_multiple"]),
                       bool(d.get("posted")), o.get("entry_date"), o.get("exit_date"),
                       i["entry"]["type"], i.get("time_td")))
    else:
        ts = (pd.Timestamp(i["execute_on"]) + int(i["time_td"]) * TRADING_DAY).normalize()
        opn.append((d["draft_id"], i["ticker"], i["side"], i["entry"]["type"],
                    i["execute_on"], i["time_td"], str(ts.date()), bool(d.get("posted"))))
print(f"  graded {len(graded)}:")
for g in graded:
    print(f"    {g[0]} {g[1]} {g[2]} {g[7]} t{g[8]} R {g[3]:+.3f} posted {g[4]} "
          f"({g[5]} -> {g[6]})")
rs = np.array([g[3] for g in graded])
print(f"  W-L {int((rs > 0).sum())}-{int((rs < 0).sum())} | total R {rs.sum():+.3f} | mean "
      f"{rs.mean():+.3f} median {np.median(rs):+.3f}")
bi, wi = int(rs.argmax()), int(rs.argmin())
print(f"  best {graded[bi][0]} {graded[bi][1]} {graded[bi][2]} R {rs[bi]:+.3f} "
      f"({graded[bi][5]} -> {graded[bi][6]}) | worst {graded[wi][0]} {graded[wi][1]} "
      f"{graded[wi][2]} R {rs[wi]:+.3f} ({graded[wi][5]} -> {graded[wi][6]})")
print(f"  open {len(opn)}:")
for o in opn:
    print(f"    {o[0]} {o[1]} {o[2]} {o[3]} execute_on {o[4]} time_td {o[5]} -> time stop "
          f"{o[6]} posted {o[7]}")
print("\ndone.")
