"""Posts check (2026-09-28, Monday): idea and stat candidates for tonight's queue.

ASOF = Monday 2026-09-28 (freshest bar). Next session Tuesday 2026-09-29.
9/29 and 9/30 are September's (and Q3's) final two sessions; 10/1 and 10/2 are
October's first two (NFP Fri 10/2). Every idea return is LAG-1 (entry at the
D+1 open, exit at a later close); close-to-close is reported alongside. SPY
calendar from the cache is the master index. September 2026 is incomplete and
is EXCLUDED from every month-end sample.

A. EEM quarter-end, last two sessions (idea candidate). D = session with two
   left in the month. Open D+1 -> Close D+2. vs SPY, EEM-SPY, give-back.
B. TLT fifth consecutive lower close (companion stat). Streak verification.
C. SPY, TLT and GLD all down on one session (SPY<=-0.5, TLT<=-0.5, GLD<=-2).
D. GLD down 3.5%+ in a session; subset 15%+ below its 252d closing high.
E. GDX down 5%+ with GLD down 2%+.
F. Open idea marks at the 9/28 close (read only).
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
ASOF = pd.Timestamp("2026-09-28")
ERA = pd.Timestamp("2018-01-01")
TK = ["SPY", "TLT", "IEF", "EEM", "GLD", "GDX", "SLV", "XLE", "XLU"]

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


F = {t: frame(t) for t in TK}


def atr_series(t: str) -> pd.Series:
    f = raw[t]
    f = f[f.index <= ASOF].dropna(subset=["High", "Low", "Close"]).astype(float)
    a = wilder_atr(f["High"].to_numpy(), f["Low"].to_numpy(), f["Close"].to_numpy())
    return pd.Series(np.asarray(a, float), index=f.index).reindex(nyse)


ATR = {t: atr_series(t) for t in TK}
R1 = {t: F[t]["Close"] / F[t]["Close"].shift(1) - 1.0 for t in TK}
print("  tonight 1d: " + ", ".join(f"{t} {100*R1[t].iloc[-1]:+.2f}%" for t in TK))


# ---------------------------------------------------------------------------
# forward returns on the SPY calendar, aligned to the signal date D
# ---------------------------------------------------------------------------
def o2c_in(t: str, h: int) -> pd.Series:
    """Open[D+1] -> Close[D+h] (entry session counted as session 1)."""
    return F[t]["Close"].shift(-h) / F[t]["Open"].shift(-1) - 1.0


def c2c0(t: str, h: int) -> pd.Series:
    c = F[t]["Close"]
    return c.shift(-h) / c - 1.0


def c2c_span(t: str, a: int, b: int) -> pd.Series:
    """Close[D+a] -> Close[D+b]."""
    c = F[t]["Close"]
    return c.shift(-b) / c.shift(-a) - 1.0


def mae_atr(t: str, k: int) -> pd.Series:
    """Worst low over D+1..D+k vs Open D+1, in ATR at D (long side)."""
    lo = pd.concat([F[t]["Low"].shift(-j) for j in range(1, k + 1)], axis=1).min(axis=1)
    return (lo - F[t]["Open"].shift(-1)) / ATR[t]


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


def line(label: str, s: pd.Series, trig: pd.DatetimeIndex, start: str) -> None:
    v = s.reindex(trig).dropna()
    v = v[v.index >= start]
    if len(v) == 0:
        print(f"    {label}: n=0")
        return
    u, d, n = rec(v.values)
    sm = summarize(v.values)
    tt = f"{sm['t']:+.2f}" if np.isfinite(sm["t"]) else "n/a"
    print(f"    {label}: n={n} {u}-{d} mean {100*v.mean():+.3f}% med "
          f"{100*v.median():+.3f}% t {tt} sign p(up) {sign_test(u, n):.4f} "
          f"p(down) {sign_test(d, n):.4f} | {era_str(v)} | worst {v.idxmin().date()} "
          f"{100*v.min():+.2f}% best {v.idxmax().date()} {100*v.max():+.2f}%")


def frozen(t: str) -> None:
    f = raw[t]
    f = f[f.index <= ASOF].dropna(subset=["High", "Low", "Close"]).astype(float)
    a = float(np.asarray(wilder_atr(f["High"].to_numpy(), f["Low"].to_numpy(),
                                    f["Close"].to_numpy()), float)[-1])
    c = float(f["Close"].iloc[-1])
    print(f"  FROZEN {t}: bar {f.index[-1].date()} close {c:.4f} Wilder-14 ATR {a:.4f} "
          f"({100*a/c:.2f}%)")


# ===========================================================================
# A. EEM quarter-end, last two sessions
# ===========================================================================
print("\n" + "=" * 78)
print("=== A. EEM LAST TWO SESSIONS OF THE MONTH: D = session with two left; "
      "Open D+1 -> Close D+2 ===")
try:
    from trading_calendar import TRADING_DAY
    nxt = [ASOF + k * TRADING_DAY for k in range(1, 5)]
    print(f"  next sessions per trading_calendar: {[str(d.date()) for d in nxt]} | Sept "
          f"sessions left after 9/28: {sum(d.month == 9 for d in nxt)} (need 2)")
except Exception as exc:  # noqa: BLE001
    print(f"  trading_calendar import failed {exc!r}")

ym = np.asarray(nyse.year * 100 + nyse.month)
cur = ASOF.year * 100 + ASOF.month
rows = []
for m in pd.unique(ym):
    if m == cur:
        continue
    ix = np.where(ym == m)[0]
    if len(ix) < 5 or ix[-1] + 2 >= len(nyse):
        continue
    rows.append({"ym": int(m), "mo": int(m % 100), "anchor": nyse[ix[-3]],
                 "last": nyse[ix[-1]]})
MA = pd.DataFrame(rows).set_index("anchor")
print(f"  anchor->last always 2 sessions: {bool(all(POS[MA['last']].values - POS[MA.index].values == 2))}")
MA = MA[MA.index >= "2003-04-15"]
MA = MA[F["EEM"]["Close"].reindex(MA.index).notna().values]
QE = pd.DatetimeIndex(MA.index[MA["mo"].isin([3, 6, 9, 12])])
OTH = pd.DatetimeIndex(MA.index[~MA["mo"].isin([3, 6, 9, 12])])
SEP = pd.DatetimeIndex(MA.index[MA["mo"] == 9])
ALLM = pd.DatetimeIndex(MA.index)
print(f"  EEM months {len(ALLM)} ({MA['ym'].iloc[0]}..{MA['ym'].iloc[-1]}) | QE {len(QE)} "
      f"| other {len(OTH)} | Sept {len(SEP)}")
S_A = "2003-04-15"
eem_o2 = o2c_in("EEM", 2)
eem_c2 = c2c0("EEM", 2)
spy_o2 = o2c_in("SPY", 2)
spy_c2 = c2c0("SPY", 2)
diff_o2 = eem_o2 - spy_o2
eem_gb = c2c_span("EEM", 2, 4)
spy_gb = c2c_span("SPY", 2, 4)
eem_o4 = o2c_in("EEM", 4)
mae_eem2 = mae_atr("EEM", 2)

for nm, trig in (("QUARTER-END MONTHS (Mar/Jun/Sep/Dec)", QE), ("OTHER MONTHS", OTH),
                 ("SEPTEMBER ONLY", SEP), ("ALL MONTHS", ALLM)):
    print(f"\n  ##### {nm}: n anchors {len(trig)} #####")
    block("A1 IDEA EEM Open D+1 -> Close D+2 (lag1)", eem_o2, trig, S_A, mae=mae_eem2)
    block("A2 EEM Close D -> Close D+2 (lag0)", eem_c2, trig, S_A, show_dates=False)
    block("A3 SPY Open D+1 -> Close D+2 (comparison)", spy_o2, trig, S_A, show_dates=False)
    line("SPY Close D -> Close D+2", spy_c2, trig, S_A)
    block("A4 EEM minus SPY Open D+1 -> Close D+2", diff_o2, trig, S_A, show_dates=False)
    block("A5 EEM give-back Close D+2 -> Close D+4 (first two of next month)", eem_gb, trig,
          S_A, show_dates=False)
    line("SPY give-back Close D+2 -> Close D+4", spy_gb, trig, S_A)
    line("EEM hold-through Open D+1 -> Close D+4", eem_o4, trig, S_A)

print("\n  QE table 2018+ (anchor | EEM o2 | SPY o2 | diff | EEM gb | EEM c2):")
for d in QE[QE >= "2018-01-01"]:
    print(f"    {d.date()} EEM {100*eem_o2[d]:+.2f} SPY {100*spy_o2[d]:+.2f} diff "
          f"{100*diff_o2[d]:+.2f} gb {100*eem_gb[d]:+.2f} c2 {100*eem_c2[d]:+.2f}")
print("  September table (all years):")
for d in SEP:
    print(f"    {d.date()} EEM {100*eem_o2[d]:+.2f} SPY {100*spy_o2[d]:+.2f} diff "
          f"{100*diff_o2[d]:+.2f} gb {100*eem_gb[d]:+.2f}")
ec = F["EEM"]["Close"]
aug_end = nyse[nyse < "2026-09-01"][-1]
print(f"  EEM MTD at 9/28 {100*(ec[ASOF]/ec[aug_end]-1):+.2f}% | 5d {100*(ec[ASOF]/ec.iloc[-6]-1):+.2f}% "
      f"| 1d {100*R1['EEM'].iloc[-1]:+.2f}%")
frozen("EEM")

# ===========================================================================
# B. TLT fifth consecutive lower close
# ===========================================================================
print("\n" + "=" * 78)
print("=== B. TLT FIFTH CONSECUTIVE LOWER CLOSE (anchor = run first reaches 5) ===")


def streak(t: str) -> pd.Series:
    c = raw[t]["Close"]
    c = c[c.index <= ASOF].astype(float).dropna()
    dn = (c < c.shift(1)).astype(int)
    grp = (dn == 0).cumsum()
    return dn.groupby(grp).cumsum().reindex(nyse)


STK = {t: streak(t) for t in ("TLT", "IEF")}
tc = F["TLT"]["Close"]
print("  TLT last 8 closes (streak of consecutive lower closes):")
for d in nyse[-8:]:
    print(f"    {d.date()} close {tc[d]:.4f} chg {100*R1['TLT'][d]:+.3f}% streak {int(STK['TLT'][d])}")
print(f"  IEF streak tonight {int(STK['IEF'].iloc[-1])} | TLT streak tonight "
      f"{int(STK['TLT'].iloc[-1])} -> tonight is a first-reach-5 anchor: "
      f"{bool(STK['TLT'].iloc[-1] == 5)}")
ties = int((raw["TLT"]["Close"].diff() == 0).sum())
print(f"  TLT exact close ties (break a run here): {ties}")

for t, S0 in (("TLT", "2002-07-31"), ("IEF", "2002-07-31")):
    trig = nyse[(STK[t] == 5).values]
    trig = trig[trig < ASOF]
    ge5 = int((STK[t] >= 5).sum())
    print(f"\n  ##### {t}: {len(trig)} runs reaching 5 (excl tonight) | days at >=5 {ge5} | "
          f"runs reaching 6+: {int((STK[t] == 6).sum())}, 7+: {int((STK[t] == 7).sum())} #####")
    for h in (1, 2, 3, 5):
        block(f"B {t} lag1 Open D+1 -> Close D+{h}", o2c_in(t, h), trig, S0,
              show_dates=False, mae=mae_atr(t, h) if h in (3, 5) else None)
        line(f"{t} lag0 Close D -> Close D+{h}", c2c0(t, h), trig, S0)
    if t == "TLT":
        line("IEF on TLT anchors, lag1 Open D+1 -> Close D+2 (= 9/29 open -> 9/30 close)",
             o2c_in("IEF", 2), trig, S0)
        line("TLT on TLT anchors, Open D -> Close D+2 (the open idea: bought D open)",
             F["TLT"]["Close"].shift(-2) / F["TLT"]["Open"] - 1.0, trig, S0)
        print("  TLT 5-streak anchors 2018+ (h3 lag1):")
        v = o2c_in("TLT", 3).reindex(trig[trig >= ERA]).dropna()
        print("    " + ", ".join(f"{i.date()} {100*x:+.2f}" for i, x in v.items()))

# ===========================================================================
# C. Stocks, bonds and gold all down
# ===========================================================================
print("\n" + "=" * 78)
print("=== C. SPY <= -0.5%, TLT <= -0.5%, GLD <= -2.0% same session, declustered 5 ===")
mC = (R1["SPY"] <= -0.005) & (R1["TLT"] <= -0.005) & (R1["GLD"] <= -0.02)
rawC = nyse[mC.fillna(False).values]
print(f"  tonight qualifies: {bool(mC.iloc[-1])} | raw days {len(rawC)}")
trigC = declusters(rawC[rawC < ASOF], 5, nyse)
print(f"  declustered(5) priors {len(trigC)} {'(SMALL N: treat as texture, not evidence)' if len(trigC) <= 25 else ''}")
S_C = "2004-11-19"
print("  priors (date | SPY 1d TLT 1d GLD 1d | SPY h21 lag1 | SPY h21 lag0 | GLD h21 lag1 | TLT h21 lag1):")
for d in trigC:
    print(f"    {d.date()} {100*R1['SPY'][d]:+.2f} {100*R1['TLT'][d]:+.2f} {100*R1['GLD'][d]:+.2f} | "
          f"SPY21L1 {100*o2c_in('SPY', 21)[d]:+.2f} SPY21L0 {100*c2c0('SPY', 21)[d]:+.2f} | "
          f"GLD21L1 {100*o2c_in('GLD', 21)[d]:+.2f} TLT21L1 {100*o2c_in('TLT', 21)[d]:+.2f}")
for h in (5, 21):
    block(f"C SPY lag1 Open D+1 -> Close D+{h}", o2c_in("SPY", h), trigC, S_C,
          mae=mae_atr("SPY", h) if h == 5 else None)
    line(f"SPY lag0 Close D -> Close D+{h}", c2c0("SPY", h), trigC, S_C)
for t in ("GLD", "TLT"):
    for h in (5, 21):
        block(f"C {t} lag1 Open D+1 -> Close D+{h}", o2c_in(t, h), trigC, S_C, show_dates=False)
        line(f"{t} lag0 Close D -> Close D+{h}", c2c0(t, h), trigC, S_C)

# ===========================================================================
# D. GLD down 3.5%+
# ===========================================================================
print("\n" + "=" * 78)
print("=== D. GLD DOWN 3.5%+ IN A SESSION, declustered 5 ===")
gc = F["GLD"]["Close"]
dd252 = gc / gc.rolling(252, min_periods=200).max() - 1.0
print(f"  tonight GLD 1d {100*R1['GLD'].iloc[-1]:+.2f}% | vs 252d closing high "
      f"{100*dd252.iloc[-1]:+.2f}% (high {gc.rolling(252, min_periods=200).max().iloc[-1]:.2f})")
mD = R1["GLD"] <= -0.035
rawD = nyse[mD.fillna(False).values]
trigD = declusters(rawD[rawD < ASOF], 5, nyse)
y26 = rawD[rawD.year == 2026]
print(f"  raw days {len(rawD)} (incl tonight) | declustered priors {len(trigD)} | 2026 raw days "
      f"{len(y26)}: {[(str(d.date()), round(100*R1['GLD'][d], 2)) for d in y26]} | 2026 "
      f"declustered {len(declusters(y26, 5, nyse))}")
print("  raw days per year: " + str(pd.Series(rawD.year).value_counts().sort_index().to_dict()))
S_D = "2004-11-19"
for h in (1, 5, 21):
    block(f"D GLD lag1 Open D+1 -> Close D+{h}", o2c_in("GLD", h), trigD, S_D,
          mae=mae_atr("GLD", h) if h == 5 else None)
    line(f"GLD lag0 Close D -> Close D+{h}", c2c0("GLD", h), trigD, S_D)
mD2 = mD & (dd252 <= -0.15)
rawD2 = nyse[mD2.fillna(False).values]
trigD2 = declusters(rawD2[rawD2 < ASOF], 5, nyse)
print(f"\n  ##### subset: GLD 15%+ below 252d closing high on the anchor: raw {len(rawD2)}, "
      f"declustered priors {len(trigD2)} #####")
print("    priors: " + ", ".join(f"{d.date()} (1d {100*R1['GLD'][d]:+.2f}, dd {100*dd252[d]:+.1f})"
                                 for d in trigD2))
for h in (1, 5, 21):
    block(f"D2 GLD lag1 Open D+1 -> Close D+{h} [dd<=-15%]", o2c_in("GLD", h), trigD2, S_D)
    line(f"GLD lag0 Close D -> Close D+{h} [dd<=-15%]", c2c0("GLD", h), trigD2, S_D)

# ===========================================================================
# E. GDX down 5%+ with GLD down 2%+
# ===========================================================================
print("\n" + "=" * 78)
print("=== E. GDX <= -5% AND GLD <= -2% same session, declustered 5, 2006+ ===")
mE = (R1["GDX"] <= -0.05) & (R1["GLD"] <= -0.02)
rawE = nyse[mE.fillna(False).values]
trigE = declusters(rawE[rawE < ASOF], 5, nyse)
print(f"  tonight qualifies {bool(mE.iloc[-1])} | raw {len(rawE)} | declustered priors {len(trigE)}")
S_E = "2006-05-23"
block("E GDX lag1 Open D+1 -> Close D+5", o2c_in("GDX", 5), trigE, S_E, mae=mae_atr("GDX", 5))
line("GDX lag0 Close D -> Close D+5", c2c0("GDX", 5), trigE, S_E)
line("GDX lag1 Open D+1 -> Close D+1", o2c_in("GDX", 1), trigE, S_E)

# ===========================================================================
# F. Open idea marks
# ===========================================================================
print("\n" + "=" * 78)
print("=== F. OPEN IDEA MARKS at the 2026-09-28 close (cache bars, read only) ===")
specs = {}
for qd, did in (("2026-09-21", "x20260921-1"), ("2026-09-24", "x20260924-1"),
                ("2026-09-25", "x20260925-1"), ("2026-09-27", "x20260927-1")):
    q = json.loads((ROOT / "content" / "queue" / f"{qd}.json").read_text())
    for x in q["drafts"]:
        if x.get("id") == did:
            specs[did] = x["idea"]
fills = {"x20260924-1": (pd.Timestamp("2026-09-25"), 89.79, "given MOO fill")}
for did, sp in specs.items():
    t, side, atr = sp["ticker"], sp["side"], float(sp["atr"])
    ex = pd.Timestamp(sp["execute_on"])
    if did in fills:
        ed, px, how = fills[did]
        how += f" (cache open {F[t]['Open'][ed]:.4f})"
    elif sp["entry"]["type"] == "MOO":
        ed, px, how = ex, float(F[t]["Open"][ex]), "cache open on execute_on"
    else:
        ed, px, how = ex, float(F[t]["Close"][ex]), "cache close on execute_on (MOC)"
    mk = float(F[t]["Close"][ASOF])
    sgn = 1.0 if side == "long" else -1.0
    mv = sgn * (mk - px)
    print(f"  {did} {t} {side} {sp['entry']['type']} exec {ex.date()} time_td {sp['time_td']} "
          f"| entry {px:.4f} [{how}] | mark {mk:.4f} | favour {100*mv/px:+.2f}% | "
          f"R {mv/atr:+.3f} (atr {atr})")
print("\ndone.")
