"""Posts check (2026-09-24): idea candidates for Thursday night's queue.

Run date Thursday 2026-09-24 is the freshest bar (signal close). Next session
is Friday 2026-09-25. Every idea return is LAG-1: entry at the D+1 open (MOO)
or the D+1 close (MOC reference), exit at the close h sessions AFTER the entry
session (matches the idea grammar: exit MOC time_td sessions after
execute_on). So "h5 from open" = Open[D+1] -> Close[D+6].

A. LONG IEF / TLT after ^TNX +15bp over two sessions with a 252-session
   closing high on the second (min_periods 200, as context 08/09). Trigger on
   the ^TNX calendar, declustered 5 on that calendar (first qualifying close
   of a run), 2003+. Subset: the SECOND consecutive qualifying day.
B. LONG XLU after XLU z10 <= -2.0 with ^TNX at (or within 0.5% of) its 252d
   closing high. z10 = build_pitch_state convention (10d return / (21d daily
   sd * sqrt(10))), which is what posts_state prints (-2.46 tonight). The
   literal (close - 10d mean) / 10d sd form is run as an alternate.
   Declustered 10. Plus LIMIT variant, unconditional z10 reference, and a
   52w-low-conditioned version.
C. USO September Fridays open-to-close, 2006+ vs other Fridays.
D. Frozen levels at the 2026-09-24 close (Wilder-14 ATR, adjusted bars).
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

ASOF = pd.Timestamp("2026-09-24")
ERA = pd.Timestamp("2018-01-01")
TK = ["SPY", "TLT", "IEF", "XLU", "USO", "^TNX"]

raw = load_prices(TK)
print("=== data check ===")
for t in TK:
    f = raw[t]
    f = f[f.index <= ASOF]
    bad_open = int(((f["Open"] <= 0) | f["Open"].isna()).sum())
    print(f"  {t}: {f.index[0].date()} .. {f.index[-1].date()} n={len(f)} "
          f"last close {f['Close'].iloc[-1]:.4f} | Open<=0/NaN rows {bad_open}"
          f" | cols {list(f.columns)}")

nyse = raw["SPY"].index
nyse = pd.DatetimeIndex(nyse[nyse <= ASOF])
POS = pd.Series(np.arange(len(nyse)), index=nyse)


def frame(t: str) -> pd.DataFrame:
    f = raw[t]
    f = f[f.index <= ASOF].astype(float)
    return f.reindex(nyse)


F = {t: frame(t) for t in ["SPY", "TLT", "IEF", "XLU", "USO"]}


def atr_series(t: str) -> pd.Series:
    f = raw[t]
    f = f[f.index <= ASOF].dropna(subset=["High", "Low", "Close"]).astype(float)
    a = wilder_atr(f["High"].to_numpy(), f["Low"].to_numpy(), f["Close"].to_numpy())
    return pd.Series(np.asarray(a, float), index=f.index).reindex(nyse)


ATR = {t: atr_series(t) for t in F}


# ---------------------------------------------------------------------------
# forward returns on the SPY calendar, aligned to the signal date D
# ---------------------------------------------------------------------------
def o2c(t: str, h: int) -> pd.Series:
    """Open[D+1] -> Close[D+1+h]."""
    return F[t]["Close"].shift(-(1 + h)) / F[t]["Open"].shift(-1) - 1.0


def o2c_in(t: str, h: int) -> pd.Series:
    """Open[D+1] -> Close[D+h] (entry session counted as session 1)."""
    return F[t]["Close"].shift(-h) / F[t]["Open"].shift(-1) - 1.0


def c2c(t: str, h: int) -> pd.Series:
    """Close[D+1] -> Close[D+1+h] (fwd_lag lag=1)."""
    c = F[t]["Close"]
    return c.shift(-(1 + h)) / c.shift(-1) - 1.0


def c2c0(t: str, h: int) -> pd.Series:
    c = F[t]["Close"]
    return c.shift(-h) / c - 1.0


def mae_atr(t: str, h: int) -> pd.Series:
    """Long MAE from the D+1 open, lows of sessions D+1..D+1+h, in ATR(D)."""
    lo = F[t]["Low"]
    mins = pd.concat([lo.shift(-k) for k in range(1, h + 2)], axis=1).min(axis=1,
                                                                          skipna=False)
    return (mins - F[t]["Open"].shift(-1)) / ATR[t]


def limit_ret(t: str, h: int, k_atr: float = 0.25) -> tuple[pd.Series, pd.Series]:
    """Buy limit at Open[D+1] - k*ATR(D); filled if Low[D+1] touches it.
    Exit Close[D+1+h]. Returns (return if filled else NaN, filled flag)."""
    lim = F[t]["Open"].shift(-1) - k_atr * ATR[t]
    filled = F[t]["Low"].shift(-1) <= lim
    r = F[t]["Close"].shift(-(1 + h)) / lim - 1.0
    return r.where(filled), filled.where(F[t]["Low"].shift(-1).notna())


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


def block(label: str, s: pd.Series, trig: pd.DatetimeIndex, start: str,
          show_dates: bool = False, mae: pd.Series | None = None) -> pd.Series:
    v = s.reindex(trig).dropna()
    allv = s[(s.index >= start)].dropna()
    loc = local_control(nyse, trig, 126)
    locv = s.reindex(loc).dropna()
    locv = locv[locv.index >= start]
    if len(v) == 0:
        print(f"  {label}: n=0")
        return v
    u, d, n = rec(v.values)
    sm = summarize(v.values)
    base_hit = float((allv > 0).mean())
    t = sm["t"]
    tt = f"{t:+.2f}" if t is not None and np.isfinite(t) else "n/a"
    print(f"  {label}: n={n} {u}-{d} mean {sm['mean_pct']:+.3f}% med "
          f"{sm['median_pct']:+.3f}% t {tt} | ctrl all {100*allv.mean():+.3f}% "
          f"(hit {100*base_hit:.1f}%, n {len(allv)}) local+/-126 "
          f"{100*locv.mean():+.3f}% | sign p(up) {sign_test(u, n):.4f} "
          f"vs base-rate {sign_test(u, n, base_hit):.4f} | worst "
          f"{sm['worst_pct']:+.2f}% best {sm['best_pct']:+.2f}%")
    print(f"      era: {era_str(v)}")
    print(f"      conc: {cluster_note(v.index, v.values)}")
    if mae is not None:
        m = mae.reindex(v.index).dropna()
        if len(m):
            print(f"      MAE (ATR, from entry open, lows D+1..exit): median "
                  f"{m.median():+.2f} worst {m.min():+.2f} ({m.idxmin().date()}) "
                  f"| share <= -1.0 ATR {100*(m <= -1.0).mean():.0f}%")
    if show_dates:
        print("      dates: " + ", ".join(f"{i.date()} {100*x:+.2f}" for i, x in v.items()))
    return v


# ===========================================================================
# A. ^TNX two-session +15bp with a 252d closing high -> LONG IEF / TLT
# ===========================================================================
print("\n" + "=" * 78)
print("=== A. ^TNX +15bp over 2 sessions, 252d closing high on the 2nd -> "
      "LONG IEF / TLT, MOO D+1 ===")
tnx = raw["^TNX"]["Close"].astype(float)
tnx = tnx[tnx.index <= ASOF].dropna()
tidx = tnx.index
bp2 = tnx.diff(2) * 100
bp1 = tnx.diff() * 100
hi = tnx >= tnx.rolling(252, min_periods=200).max() - 1e-9
trig_mask = (hi & (bp2 >= 15)).fillna(False)
print(f"  TONIGHT {tidx[-1].date()}: TNX {tnx.iloc[-1]:.3f} bp1 {bp1.iloc[-1]:+.1f} "
      f"bp2 {bp2.iloc[-1]:+.1f} 52w-high {bool(hi.iloc[-1])} trigger "
      f"{bool(trig_mask.iloc[-1])} | yesterday trigger {bool(trig_mask.iloc[-2])}")
trig = tidx[trig_mask.values]
trig = trig[trig < ASOF]
epiA_all = declusters(trig, 5, tidx)
epiA = epiA_all[epiA_all >= "2003-01-01"]
off = [d for d in epiA if d not in POS.index]
print(f"  raw qualifying closes (ex tonight) {len(trig)} -> declustered(5) "
      f"{len(epiA_all)} all-years, {len(epiA)} 2003+ | not on SPY calendar: {off}")
epiA = pd.DatetimeIndex([d for d in epiA if d in POS.index])
day2 = tidx[(trig_mask & trig_mask.shift(1, fill_value=False)).values]
day2 = day2[(day2 < ASOF) & (day2 >= "2003-01-01")]
epiA2 = pd.DatetimeIndex([d for d in declusters(day2, 5, tidx) if d in POS.index])
print(f"  second-consecutive-day subset: raw {len(day2)} -> declustered(5) "
      f"{len(epiA2)}: " + ", ".join(str(d.date()) for d in epiA2))

for tk in ("IEF", "TLT"):
    print(f"\n  ---------- {tk} (ctrl = all sessions 2003+ with the same entry/exit) ----------")
    for h in (3, 5, 10):
        block(f"A {tk} h{h} OPEN D+1 -> close +{h}", o2c(tk, h), epiA, "2003-01-01",
              show_dates=(h == 5), mae=mae_atr(tk, h))
    block(f"A {tk} h5 OPEN D+1 -> close D+5 (entry day counted)", o2c_in(tk, 5),
          epiA, "2003-01-01")
    for h in (3, 5, 10):
        block(f"A {tk} h{h} CLOSE D+1 (lag1) ref", c2c(tk, h), epiA, "2003-01-01")
    block(f"A {tk} h5 lag0 close D (brief's number, not tradeable)", c2c0(tk, 5),
          epiA, "2003-01-01")
    print(f"\n  ---- {tk} SECOND-consecutive-day subset ----")
    for h in (5, 10):
        block(f"A2 {tk} h{h} OPEN D+1", o2c(tk, h), epiA2, "2003-01-01",
              show_dates=(h == 5), mae=mae_atr(tk, h))
    block(f"A2 {tk} h5 lag0 close D (brief's N=7 cut)", c2c0(tk, 5), epiA2, "2003-01-01")

# ===========================================================================
# B. XLU washout with the 10y at a 52w high -> LONG XLU
# ===========================================================================
print("\n" + "=" * 78)
print("=== B. XLU z10 <= -2.0 with ^TNX at/within 0.5% of 252d closing high -> "
      "LONG XLU, MOO D+1, declustered 10 ===")
xc = raw["XLU"]["Close"].astype(float)
xc = xc[xc.index <= ASOF].dropna()
z_ps = (xc.pct_change(10) / (xc.pct_change().rolling(21).std() * np.sqrt(10))).reindex(nyse)
z_lit = ((xc - xc.rolling(10).mean()) / xc.rolling(10).std()).reindex(nyse)
xlo = (xc <= xc.rolling(252).min() + 1e-9).reindex(nyse).fillna(False)
tnx_max = tnx.rolling(252, min_periods=200).max()
tnx_hi = (tnx >= tnx_max - 1e-9).reindex(nyse).fillna(False)
tnx_near = (tnx >= 0.995 * tnx_max).reindex(nyse).fillna(False)
print(f"  TONIGHT: XLU z10 (posts_state conv) {z_ps.iloc[-1]:+.2f} | literal "
      f"(close-10d mean)/10d sd {z_lit.iloc[-1]:+.2f} | XLU at 252d closing low "
      f"{bool(xlo.iloc[-1])} | TNX at high {bool(tnx_hi.iloc[-1])} near(0.5%) "
      f"{bool(tnx_near.iloc[-1])}")


def run_b(label: str, mask: pd.Series, start: str = "1999-01-01") -> None:
    m = mask.fillna(False)
    d = nyse[m.to_numpy(dtype=bool)]
    d = d[(d < ASOF) & (d >= start)]
    epi = declusters(d, 10, nyse)
    print(f"\n  ##### {label}: raw {len(d)} -> declustered(10) {len(epi)} "
          f"(first {epi[0].date() if len(epi) else None}) #####")
    for h in (5, 10):
        block(f"B XLU h{h} OPEN D+1 -> close +{h}", o2c("XLU", h), epi, start,
              show_dates=True, mae=mae_atr("XLU", h))
    for h in (5, 10):
        block(f"B XLU h{h} CLOSE D+1 (lag1) ref", c2c("XLU", h), epi, start)
    lr, fl = limit_ret("XLU", 5)
    f = fl.reindex(epi).dropna()
    print(f"  LIMIT open-0.25ATR: filled {int(f.sum())}/{len(f)}")
    block("B XLU h5 LIMIT (filled only)", lr, epi, start, show_dates=True)
    unf = pd.DatetimeIndex([x for x, ok in f.items() if not bool(ok)])
    if len(unf):
        u = o2c("XLU", 5).reindex(unf).dropna()
        print(f"      unfilled episodes' MOO h5 (what was missed): n={len(u)} mean "
              f"{100*u.mean():+.2f}% " + ", ".join(f"{i.date()} {100*x:+.2f}" for i, x in u.items()))


for zname, z in (("posts_state z10", z_ps), ("literal 10d z", z_lit)):
    print(f"\n  ================ z definition: {zname} ================")
    run_b(f"[{zname}] z<=-2 AND TNX 252d high or within 0.5%",
          (z <= -2.0) & tnx_near)
    run_b(f"[{zname}] z<=-2 AND TNX exactly at 252d closing high",
          (z <= -2.0) & tnx_hi)
    run_b(f"[{zname}] z<=-2 unconditional (reference class)", (z <= -2.0))
    run_b(f"[{zname}] z<=-2 AND XLU at 252d closing low", (z <= -2.0) & xlo)

# ===========================================================================
# C. USO September Fridays, open-to-close
# ===========================================================================
print("\n" + "=" * 78)
print("=== C. USO Friday OPEN->CLOSE, 2006+ (Sep vs other Fridays) ===")
u = raw["USO"]
u = u[(u.index <= ASOF) & (u.index >= "2006-01-01")].astype(float)
oc = (u["Close"] / u["Open"] - 1.0)
cc = u["Close"].pct_change()
on = u["Open"] / u["Close"].shift(1) - 1.0
fri = u.index.weekday == 4
for lab, s in (("open->close", oc), ("close->close (brief)", cc), ("overnight Thu close->Fri open", on)):
    s = s[fri].dropna()
    sep, oth = s[s.index.month == 9], s[s.index.month != 9]
    for nm, v in (("Sep Fridays", sep), ("other Fridays", oth)):
        up, dn, n = rec(v.values)
        sm = summarize(v.values)
        print(f"  {lab:<30} {nm:<14} n={n} {up}-{dn} ({n-up-dn} flat) mean "
              f"{sm['mean_pct']:+.3f}% med {sm['median_pct']:+.3f}% t {sm['t']:+.2f} "
              f"sign p(down) {sign_test(dn, up + dn):.4f}")
        print(f"      era: {era_str(v)}")
    wt = (sep.mean() - oth.mean()) / np.sqrt(sep.var() / len(sep) + oth.var() / len(oth))
    print(f"      Welch t Sep vs other Fridays: {wt:+.2f}")
    if lab == "open->close":
        print(f"      Sep Fridays conc (short side = negate): "
              f"{cluster_note(sep.index, -sep.values)}")
        by = sep.groupby(sep.index.year).agg(lambda x: f"{100*x.sum():+.2f}({int((x>0).sum())}-{int((x<0).sum())})")
        print("      by year: " + ", ".join(f"{y} {s_}" for y, s_ in by.items()))

# ===========================================================================
# D. Frozen levels
# ===========================================================================
print("\n" + "=" * 78)
print("=== D. Frozen levels at the 2026-09-24 close (adjusted bars, Wilder-14) ===")
for t in ("IEF", "TLT", "XLU", "SPY"):
    f = raw[t]
    f = f[f.index <= ASOF].dropna(subset=["High", "Low", "Close"]).astype(float)
    a = float(np.asarray(wilder_atr(f["High"].to_numpy(), f["Low"].to_numpy(),
                                    f["Close"].to_numpy()), float)[-1])
    c = float(f["Close"].iloc[-1])
    lo = float(f["Close"].rolling(252).min().iloc[-1])
    hi_ = float(f["Close"].rolling(252).max().iloc[-1])
    print(f"  {t}: bar {f.index[-1].date()} close {c:.4f} ATR {a:.4f} "
          f"({100*a/c:.2f}%) | vs 252d closing low {100*(c/lo-1):+.2f}% "
          f"(low {lo:.4f}) vs high {100*(c/hi_-1):+.2f}%")
print(f"  XLU z10 posts_state {z_ps.iloc[-1]:+.3f} | literal {z_lit.iloc[-1]:+.3f}")
print("\ndone.")
