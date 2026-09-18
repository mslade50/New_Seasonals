"""ROUND 1 kill battery: long SVXY MOC 2026-09-08 -> MOC 2026-09-15, the
five sessions ending the close before the 2026-09-16 September VIX settle.

The cell as screened (02_event_class_grid.py, k=6 "pre" mode on vix_expiry,
h=5, September only): 13-1 over 14 instances 2012-2025, mean +4.05%,
p_coin 0.0009.

Ten attacks, ordered cheapest-first. Each prints the number that decides it.
Nothing here widens a definition to reach a sample: every section measures
the SAME 14 anchors the screen produced, or a control built from days the
candidate does not own.

SEARCH CHARGE, stated once and carried into every headline below: the screen
that produced this cell ran 192 event x class cells plus 78 cycle sub-cells
= 270 measurements. Sidak on the reported p_coin 0.0009 is
1-(1-0.0009)^270 = 0.214. That charge is not itself the kill; the kill has to
be substantive. It is stated so no number below is read as a discovery.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa: F401,F403
from pitch_lab import (
    anchor_positions, bootstrap_p_le0, cluster_note, fwd_ret, load_events,
    load_prices, rolling_on_valid, sign_test,
)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

BREAK = pd.Timestamp("2018-02-28")   # SVXY -1.0x -> -0.5x
K, H = 6, 5                          # entry = expiry_pos - 6, exit = pos - 1
COST_BPS_RT = 20.0                   # SVXY round trip, see section J

px = load_prices(["SVXY", "UVXY", "^VIX", "^VIX3M", "SPY"])
sv = px["SVXY"]["Close"].dropna()
vix = px["^VIX"]["Close"].dropna()
spy = px["SPY"]["Close"].dropna()
cal = sv.index
pos = pd.Series(range(len(cal)), index=cal)

ev = load_events()
VXP = pd.DatetimeIndex(sorted(ev.loc[ev.event == "vix_expiry", "date"]))
FOMC = set(pd.DatetimeIndex(ev.loc[ev.event == "fomc_decision", "date"]))

print("=" * 78)
print("SETUP")
print("=" * 78)
print("SVXY bars {} .. {}  n={}".format(cal[0].date(), cal[-1].date(), len(cal)))
r1 = sv.pct_change()
print("leverage break: worst SVXY day {:.2f}% on {}".format(
    100 * r1.min(), r1.idxmin().date()))
print("  daily sd pre-{} {:.2f}%   post {:.2f}%".format(
    BREAK.date(), 100 * r1[cal < BREAK].std(),
    100 * r1[cal > pd.Timestamp("2018-06-01")].std()))
print("LIVE: 2026-09-16 is a VIX settle AND an FOMC decision -> coincident = {}"
      .format(pd.Timestamp("2026-09-16") in FOMC))


def windows(expiries, k=K, h=H):
    """Entry/exit pairs exactly as the screen builds them: entry at
    (expiry position - k), exit at (expiry position - k + h) = pos - 1."""
    p, kept = anchor_positions(cal, expiries, offset=0)
    recs = []
    for pp, d in zip(p, kept):
        a, b = pp - k, pp - k + h
        if a < 0 or b >= len(cal):
            continue
        recs.append({"entry": cal[a], "expiry": d, "exit": cal[b],
                     "ret": float(sv.iloc[b] / sv.iloc[a] - 1.0)})
    if not recs:
        return pd.DataFrame()
    return pd.DataFrame(recs).set_index("entry").sort_index()


def line(label, v, extra=""):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        print("  {:<42s}  n=0".format(label))
        return
    w = int((v > 0).sum())
    print("  {:<42s}  n={:>3d}  mean={:+7.3f}%  med={:+7.3f}%  {}-{}  "
          "sign_p={:.4f}  worst={:+7.2f}%  {}".format(
              label, len(v), 100 * v.mean(), 100 * np.median(v), w,
              len(v) - w, sign_test(w, len(v)), 100 * v.min(), extra))


# ===========================================================================
print("\n" + "=" * 78)
print("A. REPRODUCE THE CELL")
print("=" * 78)
sep = VXP[VXP.month == 9]
A = windows(sep)
A["coincident_fomc"] = [d in FOMC for d in A["expiry"]]
A["midterm"] = A.index.year % 4 == 2
A["era"] = np.where(A.index < BREAK, "-1.0x", "-0.5x")
print(A.assign(ret_pct=(100 * A["ret"]).round(2)).drop(columns="ret").to_string())
line("SEPTEMBER expiry, all instances", A["ret"].values)
print("  " + cluster_note(A.index, A["ret"].values, k=2))

# ===========================================================================
print("\n" + "=" * 78)
print("B. SAME-OBJECT TEST vs the 2026-08-07 corpse")
print("   registry: 'pre-expiry short-vol carry (long SVXY into VIX expiry)'")
print("   corpse def: entry MOC k td before the last close prior to expiry,")
print("               exit that pre-expiry close.  corpse k=5 == this k=6.")
print("=" * 78)
ALL = windows(VXP)                       # every month, same construction
ALL["coincident_fomc"] = [d in FOMC for d in ALL["expiry"]]
ALL["month"] = ALL.index.month
sep_days, corpse_days = set(), set()
for df, s in ((A, sep_days), (ALL, corpse_days)):
    for e, row in df.iterrows():
        s.update(cal[pos[e]:pos[row["exit"]] + 1])
inter = len(sep_days & corpse_days)
print("  corpse anchors (all months, same k)  n={}".format(len(ALL)))
print("  candidate anchors (September only)   n={}".format(len(A)))
print("  candidate anchors that ARE corpse anchors: {} / {}".format(
    sum(1 for e in A.index if e in set(ALL.index)), len(A)))
print("  P(inside corpse day-mask | candidate day-mask) = {:.4f}   "
      "(same-object line ~0.91)".format(inter / len(sep_days)))
line("corpse, ALL months", ALL["ret"].values)
line("corpse, ALL months, -0.5x era", ALL.loc[ALL.index >= BREAK, "ret"].values)
line("corpse, ex-September, -0.5x era",
     ALL.loc[(ALL.index >= BREAK) & (ALL["month"] != 9), "ret"].values)
mm = ALL.groupby("month")["ret"].agg(["count", "mean"])
mm["mean_pct"] = (100 * mm["mean"]).round(3)
print("\n  parent's month means (the twelve-cell scan this carve-out came from):")
print(mm[["count", "mean_pct"]].sort_values("mean_pct", ascending=False).to_string())
print("  September's rank among 12 months: {} of 12".format(
    int(mm["mean_pct"].rank(ascending=False).loc[9])))

# ===========================================================================
print("\n" + "=" * 78)
print("C. THE COINCIDENCE SPLIT  (2026-09-16 is BOTH settle and FOMC)")
print("   registry 2026-09-01: 'SVXY -0.121pp coincident vs +1.904pp otherwise'")
print("=" * 78)
for nm, m in (("Sep expiry, FOMC-coincident <- LIVE", A["coincident_fomc"].values),
              ("Sep expiry, NOT coincident", ~A["coincident_fomc"].values)):
    line(nm, A["ret"].values[m])
print()
for nm, m in (("ALL-month expiry, coincident", ALL["coincident_fomc"].values),
              ("ALL-month expiry, not coincident", ~ALL["coincident_fomc"].values)):
    line(nm, ALL["ret"].values[m])
print()
sub = ALL[(ALL.index >= BREAK)]
for nm, m in (("ALL-month, -0.5x era, coincident", sub["coincident_fomc"].values),
              ("ALL-month, -0.5x era, not coinc.", ~sub["coincident_fomc"].values)):
    line(nm, sub["ret"].values[m])
print("\n  coincident Sep instances:", ", ".join(
    "{}:{:+.2f}%".format(d.date(), 100 * v) for d, v in
    zip(A.index[A["coincident_fomc"].values], A["ret"].values[A["coincident_fomc"].values])))

# ===========================================================================
print("\n" + "=" * 78)
print("D. MIDTERM  (repo prior: T1 long-SPY pre-FOMC is NON-midterm only;")
print("   T2 goes SHORT in midterms. Registry: UVXY pre-FOMC crush killed")
print("   in midterms 2026-08-06.)")
print("=" * 78)
for nm, m in (("Sep expiry, MIDTERM <- LIVE (2026)", A["midterm"].values),
              ("Sep expiry, non-midterm", ~A["midterm"].values)):
    line(nm, A["ret"].values[m])
print("  midterm instances:", ", ".join(
    "{}:{:+.2f}%".format(d.date(), 100 * v) for d, v in
    zip(A.index[A["midterm"]], A["ret"].values[A["midterm"].values])))
FD = pd.DatetimeIndex(sorted(ev.loc[ev.event == "fomc_decision", "date"]))
F = windows(FD)
F["midterm"] = F.index.year % 4 == 2
print()
line("pre-FOMC SVXY k=6 h=5, midterm", F["ret"].values[F["midterm"].values])
line("pre-FOMC SVXY k=6 h=5, non-midterm", F["ret"].values[~F["midterm"].values])
lm = A["midterm"].values & A["coincident_fomc"].values
line("Sep expiry, MIDTERM x COINCIDENT = LIVE", A["ret"].values[lm])

# ===========================================================================
print("\n" + "=" * 78)
print("E. TRADING-DAY-OF-MONTH CONTROL")
print("   registry killed VIX-expiry-week drift as 'mid-month position'.")
print("=" * 78)
tvals = pd.Series(cal, index=cal).groupby([cal.year, cal.month]).cumcount() + 1
tdom = pd.Series(tvals.values, index=cal)
month_s = pd.Series(cal.month, index=cal)
A["tdom_in"] = tdom.reindex(A.index).values
A["tdom_out"] = tdom.reindex(pd.DatetimeIndex(A["exit"])).values
print("  entry tdom: {}   exit tdom: {}".format(
    sorted(set(A["tdom_in"])), sorted(set(A["tdom_out"]))))
f5 = fwd_ret(sv, H)
tin_set = set(A["tdom_in"])
in_anchor = pd.Series(cal.isin(A.index), index=cal)
mask_all = tdom.isin(tin_set) & f5.notna() & (~in_anchor)
mask_notsep = mask_all & (month_s != 9)
mask_sep_other = mask_all & (month_s == 9)
line("candidate (Sep expiry anchors)", A["ret"].values)
line("CTRL tdom-matched, ALL months ex-anchor", f5[mask_all.values].values)
line("CTRL tdom-matched, non-September", f5[mask_notsep.values].values)
line("CTRL tdom-matched, September ex-anchor", f5[mask_sep_other.values].values)
line("CTRL SVXY all days h=5 (own drift)", f5.dropna().values)
line("CTRL SVXY all days h=5, -0.5x era", f5[cal >= BREAK].dropna().values)
paired = []
for e, row in A.iterrows():
    m = (cal.year == e.year) & (cal.month == e.month) & f5.notna().values
    o = f5[m & (cal != e)]
    if len(o):
        paired.append(row["ret"] - float(o.mean()))
line("within-month PAIRED excess", np.array(paired))
era_mask = np.asarray(A.index >= BREAK)
line("within-month PAIRED excess, -0.5x era",
     np.array(paired)[era_mask[:len(paired)]])
# the control that matters: is the ANCHOR anything beyond generic September
# mid-month SVXY, in the tradeable era?
e5 = (cal >= BREAK)
line("CANDIDATE, -0.5x era", A["ret"].values[era_mask])
line("CTRL Sep tdom-matched ex-anchor, -0.5x era",
     f5[(mask_sep_other.values) & e5].values)
line("CTRL Sep ALL days h=5, -0.5x era",
     f5[(month_s == 9).values & e5 & f5.notna().values].values)
line("CTRL Sep ALL days h=5, all eras",
     f5[(month_s == 9).values & f5.notna().values].values)

# ===========================================================================
print("\n" + "=" * 78)
print("F. THE 2018-02-28 LEVERAGE BREAK  (never quote the pooled +4.05%)")
print("=" * 78)
for nm, m in (("-1.0x era (2012-2017)", np.asarray(A.index < BREAK)),
              ("-0.5x era (2018-2025) <- tradeable", np.asarray(A.index >= BREAK))):
    line(nm, A["ret"].values[m])
rs = sv.pct_change(H)
rp = spy.pct_change(H).reindex(sv.index)
coef = {}
for nm, m in (("-1.0x", cal < BREAK), ("-0.5x", cal >= BREAK)):
    d = pd.DataFrame({"s": rs[m], "p": rp[m]}).dropna().iloc[::H]   # non-overlap
    b, a = np.polyfit(d["p"].values, d["s"].values, 1)
    coef[nm] = (a, b)
    print("  {} era: SPY beta on non-overlapping 5d windows = {:+.3f} "
          "(alpha {:+.3f}%, n={})".format(nm, b, 100 * a, len(d)))
spy5 = pd.Series(index=A.index, dtype=float)
for e, row in A.iterrows():
    spy5[e] = float(spy.loc[row["exit"]] / spy.loc[e] - 1.0)
alph = []
for e, row in A.iterrows():
    a, b = coef["-1.0x" if e < BREAK else "-0.5x"]
    alph.append(row["ret"] - (a + b * spy5[e]))
A["alpha"] = alph
m05 = np.asarray(A.index >= BREAK)
line("beta-adjusted alpha, ALL", A["alpha"].values)
line("beta-adjusted alpha, -0.5x era", A["alpha"].values[m05])
print("  -0.5x era alphas by year:", ", ".join(
    "{}:{:+.2f}%".format(d.year, 100 * v) for d, v in
    zip(A.index[m05], A["alpha"].values[m05])))
line("SPY over the same 14 windows", spy5.values)
vixw = np.array([float(vix.loc[r["exit"]] / vix.loc[e] - 1.0)
                 for e, r in A.iterrows()
                 if e in vix.index and r["exit"] in vix.index])
line("^VIX over the same windows (mechanism)", vixw)

# ===========================================================================
print("\n" + "=" * 78)
print("G. PASS-THROUGH RATIO  (registry 2026-08-27: run before any statistics)")
print("   implied front future = SVXY return de-levered by the era factor")
print("=" * 78)
lev = pd.Series(np.where(cal < BREAK, -1.0, -0.5), index=cal)
fut_d = r1 / lev
vix_d = vix.pct_change().reindex(cal)
dn = (vix_d < 0) & fut_d.notna()
print("  BASELINE, all down-VIX sessions (n={}): spot {:+.3f}%  future {:+.3f}%"
      "  ratio {:.2f}x".format(int(dn.sum()), 100 * vix_d[dn].mean(),
                               100 * fut_d[dn].mean(),
                               fut_d[dn].mean() / vix_d[dn].mean()))
dn5 = dn & (cal >= BREAK)
print("  BASELINE, -0.5x era only    (n={}): spot {:+.3f}%  future {:+.3f}%"
      "  ratio {:.2f}x".format(int(dn5.sum()), 100 * vix_d[dn5].mean(),
                               100 * fut_d[dn5].mean(),
                               fut_d[dn5].mean() / vix_d[dn5].mean()))
rows = []
for e, row in A.iterrows():
    a, b = pos[e], pos[row["exit"]]
    seg = fut_d.iloc[a + 1:b + 1]
    fut_w = float(np.prod(1 + seg.values) - 1)
    vx_w = float(vix.loc[row["exit"]] / vix.loc[e] - 1.0)
    rows.append({"entry": e.date(), "era": row["era"], "vix_w": vx_w,
                 "fut_w": fut_w, "svxy": row["ret"]})
P = pd.DataFrame(rows)
for nm, m in (("ALL 14 windows", np.ones(len(P), bool)),
              ("-0.5x era only", (P["era"] == "-0.5x").values)):
    s, f = P["vix_w"][m].mean(), P["fut_w"][m].mean()
    print("  {:<22s} spot {:+.3f}%  implied future {:+.3f}%  pass-through {:.2f}x"
          .format(nm, 100 * s, 100 * f, f / s))
# 5-session-window baseline, matched horizon, down-VIX windows only
vix5 = vix.pct_change(H).reindex(cal)
fut5 = (1 + fut_d).rolling(H).apply(np.prod, raw=True) - 1
w_dn = (vix5 < 0) & fut5.notna() & (cal >= BREAK)
print("  BASELINE, -0.5x era 5-SESSION down-VIX windows (n={}): spot {:+.3f}%  "
      "future {:+.3f}%  ratio {:.2f}x".format(
          int(w_dn.sum()), 100 * vix5[w_dn].mean(), 100 * fut5[w_dn].mean(),
          fut5[w_dn].mean() / vix5[w_dn].mean()))
print(P.assign(vix_w=(100 * P.vix_w).round(2), fut_w=(100 * P.fut_w).round(2),
               svxy=(100 * P.svxy).round(2)).to_string(index=False))

# ===========================================================================
print("\n" + "=" * 78)
print("H. VIX 21-DAY RELATIVE-RANGE PERCENTILE AT ENTRY")
print("   watchlist 33: (0,5] pays -0.096% / 13-11; (5,10] +1.465%; (10,15] +2.034%")
print("=" * 78)
mx = rolling_on_valid(vix, lambda x: x.rolling(21).max())
mn = rolling_on_valid(vix, lambda x: x.rolling(21).min())
mu = rolling_on_valid(vix, lambda x: x.rolling(21).mean())
rel = (mx - mn) / mu
relp = rolling_on_valid(rel, lambda x: x.rolling(252).rank(pct=True) * 100)
live_rel = float(rel.dropna().iloc[-1])
live_relp = float(relp.dropna().iloc[-1])
print("  LIVE 2026-09-04: rel-range {:.4f}  percentile {:.2f}   ^VIX {:.2f}".format(
    live_rel, live_relp, float(vix.iloc[-1])))
A["relp"] = relp.reindex(A.index).values
for lo, hi in ((0, 5), (5, 15), (15, 40), (40, 101)):
    m = ((A["relp"] >= lo) & (A["relp"] < hi)).values
    line("Sep expiry, rel-range pctile [{},{})".format(lo, hi), A["ret"].values[m])
print("  per-instance:", ", ".join(
    "{}:{:.0f}p/{:+.1f}%".format(d.date(), r, 100 * v)
    for d, r, v in zip(A.index, A["relp"], A["ret"])))
sma20 = rolling_on_valid(vix, lambda x: x.rolling(20).mean())
v_now, s_now = float(vix.iloc[-1]), float(sma20.dropna().iloc[-1])
print("  production VIX Range Compression needs ^VIX above its 20d SMA: "
      "VIX {:.2f} vs SMA20 {:.2f} -> {}".format(
          v_now, s_now, "ON" if v_now > s_now else "OFF"))

# ===========================================================================
print("\n" + "=" * 78)
print("I. FRAGILITY DIAL, gradient form (2026-08-18 method)")
print("=" * 78)
frag = pd.read_parquet(Path(__file__).resolve().parents[3] / "data" /
                       "rd2_fragility.parquet")
ma = frag["63d"].rolling(10).mean().dropna()
live_dial = float(ma.iloc[-1])
print("  LIVE ma10(63d) = {:.1f}   percentile of own series {:.1f}   "
      "days >=85 ever = {}".format(live_dial, 100 * (ma <= live_dial).mean(),
                                   int((ma >= 85).sum())))
A["dial"] = ma.reindex(A.index).values
have = A["dial"].notna().values
print("  dial at each instance:", ", ".join(
    "{}:{:.0f}".format(d.date(), v) for d, v in zip(A.index[have], A["dial"][have])))
print("  max dial ever at a Sep-expiry anchor: {:.1f}  (live {:.1f} -> outside "
      "support: {})".format(A["dial"].max(), live_dial,
                            live_dial > A["dial"].max()))
if have.sum() >= 3:
    b, a = np.polyfit(A["dial"][have].values, A["ret"].values[have], 1)
    print("  within-trigger gradient: ret = {:+.3f}% {:+.4f}%/dial-pt   "
          "fitted at dial {:.0f} -> {:+.3f}%".format(
              100 * a, 100 * b, live_dial, 100 * (a + b * live_dial)))
ALL["dial"] = ma.reindex(ALL.index).values
h2 = np.asarray(ALL["dial"].notna().values & np.asarray(ALL.index >= BREAK))
b2, a2 = np.polyfit(ALL["dial"].values[h2], ALL["ret"].values[h2], 1)
print("  parent (-0.5x era, n={}) gradient: {:+.3f}% {:+.4f}%/pt -> fitted at "
      "{:.0f} = {:+.3f}%".format(int(h2.sum()), 100 * a2, 100 * b2, live_dial,
                                 100 * (a2 + b2 * live_dial)))
for lo, hi in ((0, 40), (40, 70), (70, 200)):
    m = h2 & (ALL["dial"] >= lo).values & (ALL["dial"] < hi).values
    line("parent -0.5x, dial [{},{})".format(lo, hi), ALL["ret"].values[m])

# ===========================================================================
print("\n" + "=" * 78)
print("J. COST AND TAIL")
print("=" * 78)
print("  assumed SVXY round trip {:.0f} bps (4x an index ETF's 5 bps: ~10-15 bps"
      .format(COST_BPS_RT))
print("    quoted spread on a ~$40 ETP plus roll drag inside a 5-session hold).")
w5 = sv.pct_change(H)
print("  worst realised 5-session SVXY window, -0.5x era: {:+.2f}% on {}".format(
    100 * w5[cal >= BREAK].min(), w5[cal >= BREAK].idxmin().date()))
print("  worst realised 5-session SVXY window, all eras : {:+.2f}% on {}".format(
    100 * w5.min(), w5.idxmin().date()))
p05 = np.nanpercentile(w5[cal >= BREAK].dropna().values, 5)
print("  5th pctile of -0.5x era 5d windows: {:+.2f}%".format(100 * p05))
e05 = A["ret"].values[m05]
al05 = A["alpha"].values[m05]
print("  -0.5x era raw   mean {:+.3f}% -> {:.1f}x cost".format(
    100 * e05.mean(), 100 * e05.mean() / (COST_BPS_RT / 100)))
print("  -0.5x era alpha mean {:+.3f}% -> {:.1f}x cost".format(
    100 * al05.mean(), 100 * al05.mean() / (COST_BPS_RT / 100)))
print("  bootstrap P(mean<=0), -0.5x era raw  : {:.4f}".format(bootstrap_p_le0(e05)))
print("  bootstrap P(mean<=0), -0.5x era alpha: {:.4f}".format(bootstrap_p_le0(al05)))

# ===========================================================================
print("\n" + "=" * 78)
print("K. SEARCH CHARGE ON THE HEADLINE")
print("=" * 78)
print("  screened p_coin 0.0009 (pooled 14, BOTH securities): Sidak over 270 "
      "= {:.4f}".format(1 - (1 - 0.0009) ** 270))
w = int((e05 > 0).sum())
pw = sign_test(w, len(e05))
print("  -0.5x era raw record {}-{}, sign p {:.4f} -> Sidak {:.4f}".format(
    w, len(e05) - w, pw, 1 - (1 - pw) ** 270))
wa = int((al05 > 0).sum())
pa = sign_test(wa, len(al05))
print("  -0.5x era alpha record {}-{}, sign p {:.4f} -> Sidak {:.4f}".format(
    wa, len(al05) - wa, pa, 1 - (1 - pa) ** 270))
print("\nDONE")
