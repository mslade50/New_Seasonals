"""D1 round 1 -- watchlist #33, long SVXY into a scheduled print out of a
21-day VIX range in the (5,15] band with >= 3 td of clear calendar.

The entry armed for the first time on the 2026-09-09 bar. This script tries to
kill it. Sections:

 0. THE GATE'S OWN NUMBER, three conventions, and which one entry #33 uses.
 1. REPRODUCE the parked arm verbatim (n=31, 25-6, +1.722%, t 4.943,
    sign p 0.0004, boot 0.0000; short ^VIX +4.181%, t 4.552).
 2. THE FEB-2018 RE-LEVER. -0.5x era ALONE with its own N and record. This is
    the whole verdict if the number lives in the -1.0x product.
 3. RUNWAY DEFINITION. Narrow {nfp,cpi,ppi,fomc} vs wide (+vix_expiry, opex,
    quad_witching). Live anchor under both, and the whole historical cell
    rebuilt under the stricter definition.
 4. DOSE RESPONSE around 9.92 -- fine ladder, knife-edge test.
 5. CONCENTRATION -- cluster_note, drop-best-2, drop-best-year, dates.
 6. SPY RESIDUAL (mandatory rule 8) -- alpha and beta, per era.
 7. MIDTERM split.
 8. TAIL -- worst episode, adverse-move distribution, CPI-in-hold.
 9. PLACEBO LADDER k=-5..+5 in the -0.5x era specifically.
10. DIAL -- did any armed episode ever carry ma10(63d) near 87.66?
11. CHARGED PERMUTATION against the DEFENDED cell's own statistic.
12. COST.
13. HORIZON SCAN h=1..10 (so the pitched h comes from a table).
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa: F401,F403,E402
from pitch_lab import (close_panel, fwd_lag, summarize, sign_test, load_events,
                       rolling_on_valid, show, anchor_positions, bootstrap_p_le0,
                       cluster_note, horizon_scan, declusters)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 260)

RELEVER = pd.Timestamp("2018-02-28")   # SVXY -1.0x -> -0.5x
px = close_panel(["SVXY", "UVXY", "^VIX", "^VIX3M", "SPY"])
cal = px["SPY"].dropna().index
vix = px["^VIX"]


def relrange_pctile(v, rank_kw=None):
    rng = (rolling_on_valid(v, lambda x: x.rolling(21).max())
           - rolling_on_valid(v, lambda x: x.rolling(21).min()))
    rel = rng / rolling_on_valid(v, lambda x: x.rolling(21).mean())
    return rolling_on_valid(rel, lambda x: x.rolling(252).rank(pct=True) * 100)


REL = relrange_pctile(vix)                       # entry #33's OWN convention
REL_SPYCAL = relrange_pctile(px["^VIX"].reindex(cal))
VLP = rolling_on_valid(vix, lambda x: x.rolling(252).rank(pct=True) * 100)
TS = px["^VIX3M"] / px["^VIX"] - 1.0

print("=" * 120)
print("0. THE GATE NUMBER -- three conventions for one concept")
print("=" * 120)
absrng = (rolling_on_valid(vix, lambda x: x.rolling(21).max())
          - rolling_on_valid(vix, lambda x: x.rolling(21).min()))
prod = rolling_on_valid(absrng, lambda x: x.rolling(504).apply(
    lambda w: 100.0 * (w[:-1] < w[-1]).mean(), raw=True))
print(f"  (a) entry #33 as implemented in 2026-09-03/a9_c1_live_rung_verdict.py:")
print(f"      rel-range = (21d max - 21d min) / 21d mean, trailing-252 rank")
print(f"      rolled on ^VIX's OWN valid bars (close_panel + rolling_on_valid)")
print(f"      2026-09-09 -> {float(REL.dropna().iloc[-1]):.3f}")
print(f"  (b) same statistic, ^VIX first reindexed to SPY's calendar:")
print(f"      2026-09-09 -> {float(REL_SPYCAL.dropna().iloc[-1]):.3f}")
print(f"  (c) production risk dashboard compute_vix_range_compression:")
print(f"      ABSOLUTE 21d range (no /mean), 504d lookback, strict-less rank")
print(f"      2026-09-09 -> {float(prod.dropna().iloc[-1]):.3f}")
print("  raw rel-range on both calendars:")
vr = vix.dropna()
vc = px["^VIX"].reindex(cal).dropna()
print(f"      own  max {vr.iloc[-21:].max():.2f} min {vr.iloc[-21:].min():.2f} "
      f"mean {vr.iloc[-21:].mean():.4f} -> {(vr.iloc[-21:].max()-vr.iloc[-21:].min())/vr.iloc[-21:].mean():.4f}")
print(f"      SPY  max {vc.iloc[-21:].max():.2f} min {vc.iloc[-21:].min():.2f} "
      f"mean {vc.iloc[-21:].mean():.4f} -> {(vc.iloc[-21:].max()-vc.iloc[-21:].min())/vc.iloc[-21:].mean():.4f}")
print("  => the RAW statistic is identical to 4 dp; only the trailing-252 RANK")
print("     POPULATION differs (^VIX carries 4 extra NYSE-closure bars). The band")
print("     is (5,15] and both land inside it, so the ARM is NOT calendar-fragile.")
print("     Entry #33's number is (a) = 9.921. 9.127 is convention (b).")

# ---------------------------------------------------------------------------
NARROW = ("nfp", "cpi", "ppi", "fomc_decision")
WIDE = NARROW + ("vix_expiry", "opex", "quad_witching")


def build_frame(kinds, prints_kinds, rel_series):
    EV = {k: load_events([k])["date"] for k in kinds}
    ALL = pd.DatetimeIndex(sorted(pd.concat(
        [load_events([k])["date"] for k in prints_kinds]).unique()))
    pos = pd.Series(range(len(cal)), index=cal)
    rows = []
    for kind in kinds:
        p, kept = anchor_positions(cal, EV[kind], -2)
        for i, ap in enumerate(p):
            d0 = kept[i]
            nxt = ALL[ALL > d0]
            rw = 99 if len(nxt) == 0 else int(
                pos.get(nxt[0], int(cal.searchsorted(nxt[0])))
                - pos.get(d0, int(cal.searchsorted(d0))))
            rows.append({"anchor": cal[ap], "kind": kind, "runway_td": rw})
    F = pd.DataFrame(rows).set_index("anchor").sort_index()
    g = F.groupby(level=0)
    F = F[~F.index.duplicated(keep="first")].assign(
        runway_td=g["runway_td"].min(),
        kind=g["kind"].apply(lambda x: "+".join(sorted(set(x)))))
    for c, s in (("rel", rel_series), ("vlp", VLP), ("ts", TS), ("vix", vix)):
        F[c] = s.reindex(F.index).values
    for h in range(1, 11):
        F[f"svxy{h}"] = fwd_lag(px["SVXY"].dropna(), h, lag=1).reindex(F.index).values
        F[f"spy{h}"] = fwd_lag(px["SPY"].dropna(), h, lag=1).reindex(F.index).values
    F["svxy"] = F["svxy1"]
    F["spy"] = F["spy1"]
    F["nvix"] = (-fwd_lag(vix.dropna(), 1, lag=1)).reindex(F.index).values
    return F


F = build_frame(NARROW, NARROW, REL)
CL = F[F["runway_td"] >= 3].copy()
MATCH = CL[CL["svxy"].notna()]
DEF = MATCH[(MATCH["rel"] > 5) & (MATCH["rel"] <= 15)]        # THE DEFENDED CELL


def cc(v, label):
    v = pd.Series(v).dropna()
    st = summarize(v.values, label)
    if st["n"]:
        st["signp"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
        st["rec"] = f"{int((v>0).sum())}-{int((v<0).sum())}"
    return st


print("\n" + "=" * 120)
print("1. REPRODUCE THE PARKED ARM  (claim: SVXY n=31, 25-6, +1.722%, t 4.943,")
print("   sign p 0.0004, boot 0.0000; short ^VIX +4.181%, t 4.552)")
print("=" * 120)
print(f"  all k=-2 anchors {len(F)}; clear-calendar (runway>=3) {len(CL)}; "
      f"SVXY-covered {len(MATCH)}; in band (5,15] {len(DEF)}")
nv_band = CL[(CL["rel"] > 5) & (CL["rel"] <= 15)]["nvix"].dropna()
show([cc(DEF["svxy"].values, "LONG SVXY, (5,15] band, runway>=3, h=1"),
      cc(nv_band.values, "SHORT ^VIX, same gate, FULL history")],
     "the defended cell as parked")
print(f"  SVXY boot P(mean<=0) = {bootstrap_p_le0(DEF['svxy'].dropna().values):.4f}")
print("\n  episode dates + values (band, matched):")
D = DEF.assign(svxy_pct=(100 * DEF["svxy"]).round(2), spy_pct=(100 * DEF["spy"]).round(2),
               rel=DEF["rel"].round(2))
print(D[["kind", "runway_td", "rel", "vix", "svxy_pct", "spy_pct"]].to_string())

# ---------------------------------------------------------------------------
print("\n" + "=" * 120)
print("2. THE FEB-2018 RE-LEVER -- the -0.5x era ALONE")
print("=" * 120)
pre = DEF[DEF.index < RELEVER]
post = DEF[DEF.index >= RELEVER]
show([cc(pre["svxy"].values, f"SVXY -1.0x era (<{RELEVER.date()})"),
      cc(post["svxy"].values, f"SVXY -0.5x era (>={RELEVER.date()}) -- THE ONLY TRADEABLE ONE")],
     "defended cell, split on the re-lever")
if len(post):
    print(f"  -0.5x era boot P(mean<=0) = {bootstrap_p_le0(post['svxy'].dropna().values):.4f}")
    print("  -0.5x era episode list:")
    print(post.assign(svxy_pct=(100 * post["svxy"]).round(2),
                      spy_pct=(100 * post["spy"]).round(2),
                      rel=post["rel"].round(2))[
        ["kind", "runway_td", "rel", "vix", "svxy_pct", "spy_pct"]].to_string())
# unconditional benchmark in each era
for lab, lo, hi in (("-1.0x", pd.Timestamp("2000-01-01"), RELEVER),
                    ("-0.5x", RELEVER, pd.Timestamp("2099-01-01"))):
    s = fwd_lag(px["SVXY"].dropna(), 1, lag=1)
    s = s[(s.index >= lo) & (s.index < hi)].dropna()
    print(f"  UNCONDITIONAL 1-session SVXY, {lab} era: n={len(s)} "
          f"mean {100*s.mean():+.4f}% hit {100*(s>0).mean():.1f}%")
print("  and the same-era cell EXCESS over that unconditional:")
for lab, sub, lo, hi in (("-1.0x", pre, pd.Timestamp("2000-01-01"), RELEVER),
                         ("-0.5x", post, RELEVER, pd.Timestamp("2099-01-01"))):
    s = fwd_lag(px["SVXY"].dropna(), 1, lag=1)
    s = s[(s.index >= lo) & (s.index < hi)].dropna()
    if len(sub):
        print(f"    {lab}: cell {100*sub['svxy'].mean():+.3f}%  uncond "
              f"{100*s.mean():+.3f}%  edge {100*(sub['svxy'].mean()-s.mean()):+.3f}pp")

# ---------------------------------------------------------------------------
print("\n" + "=" * 120)
print("3. RUNWAY DEFINITION -- which events count as 'the next scheduled print'?")
print("=" * 120)
ev = load_events(None)
print("  LIVE anchor, CPI 2026-09-11 (k=-2 anchor 2026-09-09, entry MOC 2026-09-10):")
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
fwd = pd.DatetimeIndex(pd.date_range(pd.Timestamp("2026-09-01"), periods=140, freq=bd))
for name, kinds in (("NARROW", NARROW), ("WIDE(+vixexp,opex,quad)", WIDE)):
    dates = pd.DatetimeIndex(sorted(ev[ev["event"].isin(kinds)]["date"].unique()))
    d0 = pd.Timestamp("2026-09-11")
    nxt = dates[dates > d0]
    rw = int(fwd.searchsorted(nxt[0]) - fwd.searchsorted(d0))
    print(f"    {name:24s} next {nxt[0].date()} "
          f"{sorted(ev[ev['date']==nxt[0]]['event'].tolist())} runway {rw} td -> "
          f"{'QUALIFIES' if rw >= 3 else 'DISQUALIFIED'}")
print("\n  HISTORICAL cell rebuilt with the WIDE 'next print' set (anchors still")
print("  k=-2 of {nfp,cpi,ppi,fomc}, but runway measured to the next event of ANY")
print("  of the 7 kinds):")
FW = build_frame(NARROW, WIDE, REL)
CLW = FW[FW["runway_td"] >= 3]
MW = CLW[CLW["svxy"].notna()]
DEFW = MW[(MW["rel"] > 5) & (MW["rel"] <= 15)]
show([cc(DEF["svxy"].values, "NARROW runway (as parked)"),
      cc(DEFW["svxy"].values, "WIDE runway"),
      cc(DEFW[DEFW.index >= RELEVER]["svxy"].values, "WIDE runway, -0.5x era only")],
     "gate identical, only the definition of 'clear calendar' changes")
lost = DEF.index.difference(DEFW.index)
print(f"  anchors DISCARDED by the wider definition: {len(lost)} -> "
      f"{[str(d.date()) for d in lost]}")
if len(lost):
    print(f"  the DISCARDED COMPLEMENT pays "
          f"{100*DEF.loc[lost,'svxy'].mean():+.3f}% on n={len(lost)}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 120)
print("4. DOSE RESPONSE AROUND 9.92 -- is the live reading on a plateau or an edge?")
print("=" * 120)
rows = []
for lo, hi in ((0, 2.5), (2.5, 5), (5, 7.5), (7.5, 10), (10, 12.5), (12.5, 15),
               (15, 20), (20, 30), (30, 101)):
    m = (MATCH["rel"] > lo) & (MATCH["rel"] <= hi)
    rows.append(cc(MATCH.loc[m, "svxy"].values, f"SVXY | rel ({lo},{hi}]"))
show(rows, "fine ladder, matched (SVXY-covered), runway>=3")
print("  same ladder, -0.5x ERA ONLY:")
M5 = MATCH[MATCH.index >= RELEVER]
rows = []
for lo, hi in ((0, 5), (5, 10), (10, 15), (15, 30), (30, 101)):
    m = (M5["rel"] > lo) & (M5["rel"] <= hi)
    rows.append(cc(M5.loc[m, "svxy"].values, f"SVXY -0.5x | rel ({lo},{hi}]"))
show(rows, "the bimodality the entry rests on, in the tradeable era")
print("  moving-window ladder centred on the LIVE 9.92 (+/- width), matched:")
for w in (2, 3, 5):
    m = (MATCH["rel"] > 9.92 - w) & (MATCH["rel"] <= 9.92 + w)
    m5 = (M5["rel"] > 9.92 - w) & (M5["rel"] <= 9.92 + w)
    a, b = cc(MATCH.loc[m, "svxy"].values, ""), cc(M5.loc[m5, "svxy"].values, "")
    print(f"    +/-{w}: all-era n={a.get('n',0)} {a.get('mean_pct',float('nan')):+.3f}% "
          f"| -0.5x n={b.get('n',0)} {b.get('mean_pct',float('nan')):+.3f}%")

# ---------------------------------------------------------------------------
print("\n" + "=" * 120)
print("5. CONCENTRATION")
print("=" * 120)
for lab, sub in (("all-era band cell", DEF), ("-0.5x era band cell", post)):
    if not len(sub):
        continue
    v = sub["svxy"].dropna()
    print(f"  {lab}: n={len(v)}")
    print("    " + cluster_note(v.index, v.values, k=2))
    order = np.argsort(-np.abs(v.values))[:2]
    keep = np.ones(len(v), bool)
    keep[order] = False
    print(f"    drop-best-2 (by |value|): {100*v.values[keep].mean():+.3f}% on n={keep.sum()}")
    byyr = pd.Series(v.values).groupby(v.index.year.values).sum()
    worst_yr = byyr.idxmax()
    kv = v[v.index.year != worst_yr]
    print(f"    drop-best-year ({worst_yr}): {100*kv.mean():+.3f}% on n={len(kv)} "
          f"record {int((kv>0).sum())}-{int((kv<0).sum())}")
    print(f"    by year: {dict((y, round(100*r,2)) for y, r in byyr.items())}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 120)
print("6. SPY RESIDUAL  (mandatory rule 8)")
print("=" * 120)
for lab, sub in (("all-era band cell", DEF), ("-0.5x era band cell", post)):
    s = sub.dropna(subset=["svxy", "spy"])
    if len(s) < 4:
        continue
    b, a = np.polyfit(s["spy"].values, s["svxy"].values, 1)
    resid = s["svxy"].values - (a + b * s["spy"].values)
    se = resid.std(ddof=2) / np.sqrt(len(s))
    print(f"  {lab}: n={len(s)}  SVXY = {100*a:+.3f}% + {b:.2f} * SPY   "
          f"alpha t {a/se:+.2f}  R2 {np.corrcoef(s['spy'],s['svxy'])[0,1]**2:.3f}")
    print(f"     corr(SPY,SVXY) {np.corrcoef(s['spy'],s['svxy'])[0,1]:+.3f}   "
          f"SPY leg alone {100*s['spy'].mean():+.3f}% "
          f"({int((s['spy']>0).sum())}-{int((s['spy']<0).sum())})")
    # trigger-set beta alpha: subtract beta * unconditional SPY over same era
    lo = RELEVER if "0.5" in lab else pd.Timestamp("2000-01-01")
    hi = pd.Timestamp("2099-01-01") if "0.5" in lab else RELEVER
    sp = fwd_lag(px["SPY"].dropna(), 1, lag=1)
    sp = sp[(sp.index >= lo) & (sp.index < hi)].dropna()
    sv = fwd_lag(px["SVXY"].dropna(), 1, lag=1)
    sv = sv[(sv.index >= lo) & (sv.index < hi)].dropna()
    j = pd.concat([sp.rename("spy"), sv.rename("svxy")], axis=1).dropna()
    bb, aa = np.polyfit(j["spy"].values, j["svxy"].values, 1)
    exp = aa + bb * s["spy"].values
    print(f"     ERA-WIDE beta {bb:.2f}; cell alpha vs era-wide line = "
          f"{100*(s['svxy'].values - exp).mean():+.3f}pp")

# ---------------------------------------------------------------------------
print("\n" + "=" * 120)
print("7. MIDTERM SPLIT (2026 is midterm)")
print("=" * 120)
for lab, sub in (("all-era", DEF), ("-0.5x era", post)):
    if not len(sub):
        continue
    mt = sub[sub.index.year % 4 == 2]
    nm = sub[sub.index.year % 4 != 2]
    show([cc(mt["svxy"].values, f"{lab} MIDTERM years"),
          cc(nm["svxy"].values, f"{lab} non-midterm")], f"{lab} cycle split")

# ---------------------------------------------------------------------------
print("\n" + "=" * 120)
print("8. TAIL -- the hold contains a CPI print BY CONSTRUCTION (k=-2, h=1 ->")
print("   entry MOC on k=-1, exit at the print close)")
print("=" * 120)
for lab, sub in (("all-era", DEF), ("-0.5x era", post)):
    if not len(sub):
        continue
    v = sub["svxy"].dropna().sort_values()
    print(f"  {lab}: worst 5 -> " +
          ", ".join(f"{d.date()} {100*x:+.2f}%" for d, x in v.head(5).items()))
    print(f"     losers {int((v<0).sum())}/{len(v)} at mean {100*v[v<0].mean():+.3f}%, "
          f"5th pctile {100*np.percentile(v,5):+.2f}%, worst {100*v.min():+.2f}%")
# what did ^VIX do on the losing sessions
lv = DEF[DEF["svxy"] < 0]
vch = (fwd_lag(vix.dropna(), 1, lag=1)).reindex(lv.index)
print(f"  ^VIX move on the {len(lv)} losing anchors: mean {100*vch.mean():+.2f}%, "
      f"max {100*vch.max():+.2f}% ({vch.idxmax().date() if len(vch.dropna()) else 'n/a'})")

# ---------------------------------------------------------------------------
print("\n" + "=" * 120)
print("9. PLACEBO LADDER k=-5..+5, -0.5x ERA SPECIFICALLY")
print("=" * 120)
EVn = {k: load_events([k])["date"] for k in NARROW}
ALLn = pd.DatetimeIndex(sorted(pd.concat(list(EVn.values())).unique()))
posn = pd.Series(range(len(cal)), index=cal)
lad = []
for k in range(-5, 6):
    rows = []
    for kind in NARROW:
        p, kept = anchor_positions(cal, EVn[kind], k)
        for i, ap in enumerate(p):
            d0 = kept[i]
            nxt = ALLn[ALLn > d0]
            rw = 99 if len(nxt) == 0 else int(posn.get(nxt[0], 0) - posn.get(d0, 0))
            rows.append({"anchor": cal[ap], "runway_td": rw})
    G = pd.DataFrame(rows).set_index("anchor").sort_index()
    G = G.groupby(level=0).min()
    G["rel"] = REL.reindex(G.index).values
    G["svxy"] = fwd_lag(px["SVXY"].dropna(), 1, lag=1).reindex(G.index).values
    sub = G[(G["runway_td"] >= 3) & (G["rel"] > 5) & (G["rel"] <= 15)]
    for era_lab, ss in (("all", sub), ("-0.5x", sub[sub.index >= RELEVER])):
        v = ss["svxy"].dropna()
        lad.append({"k": k, "era": era_lab, "n": len(v),
                    "mean_pct": round(100 * v.mean(), 3) if len(v) else np.nan,
                    "hit": round(100 * (v > 0).mean(), 1) if len(v) else np.nan})
L = pd.DataFrame(lad)
print(L.pivot(index="k", columns="era", values=["n", "mean_pct", "hit"]).to_string())
for era_lab in ("all", "-0.5x"):
    sl = L[L["era"] == era_lab].dropna(subset=["mean_pct"]).sort_values("mean_pct", ascending=False)
    rank = list(sl["k"]).index(-2) + 1 if -2 in list(sl["k"]) else None
    print(f"  TRUE anchor k=-2 ranks {rank} of {len(sl)} by episode mean in the "
          f"{era_lab} sample")

# ---------------------------------------------------------------------------
print("\n" + "=" * 120)
print("10. THE FRAGILITY DIAL -- today's ma10(63d) is 87.66")
print("=" * 120)
frag = pd.read_parquet(Path(__file__).resolve().parents[3] / "data" / "rd2_fragility.parquet")
ma = frag["63d"].rolling(10).mean()
d_at = ma.reindex(DEF.index)
print(f"  dial coverage starts {frag.index[0].date()}; band anchors with a dial "
      f"reading: {int(d_at.notna().sum())} of {len(DEF)}")
print(f"  live ma10(63d) 2026-09-09 = {float(ma.dropna().iloc[-1]):.2f} "
      f"({100*(ma.dropna() <= ma.dropna().iloc[-1]).mean():.2f}th pctile of its own series)")
if d_at.notna().any():
    print(f"  armed-episode dial: min {d_at.min():.1f} median {d_at.median():.1f} "
          f"max {d_at.max():.1f}")
    print("  " + ", ".join(f"{d.date()}:{v:.0f}" for d, v in d_at.dropna().items()))
    print(f"  episodes at dial >= 70: {int((d_at >= 70).sum())};  >= 80: "
          f"{int((d_at >= 80).sum())}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 120)
print("11. CHARGED PERMUTATION")
print("=" * 120)
print("  DEFENDED statistic: MEAN 1-session lag-1 long-SVXY return over anchors")
print("  with rel in (5,15], runway>=3, k=-2, SVXY-covered  (all-era, as parked).")
GRID_BANDS = [(0, 5), (5, 10), (10, 15), (5, 15), (5, 20), (5, 25), (10, 20),
              (15, 30), (0, 15), (0, 101)]
GRID_RW = [1, 2, 3, 4]
GRID_H = list(range(1, 11))
print(f"  disclosed walk = {len(GRID_BANDS)} bands x {len(GRID_RW)} runway rungs x "
      f"{len(GRID_H)} horizons x 2 vehicles = "
      f"{len(GRID_BANDS)*len(GRID_RW)*len(GRID_H)*2} cells (floor)")
obs = float(DEF["svxy"].mean())
svxy_ret = {h: fwd_lag(px["SVXY"].dropna(), h, lag=1) for h in GRID_H}
nvix_ret = {h: -fwd_lag(vix.dropna(), h, lag=1) for h in GRID_H}
base = F.copy()
rng = np.random.default_rng(7)
NB = 2000
svc = px["SVXY"].dropna().index
uncharged_hits = 0
charged_hits = 0
n_all = len(cal)
for b in range(NB):
    sh = int(rng.integers(21, n_all - 21))
    shifted = pd.Series(cal[(np.arange(n_all) + sh) % n_all], index=cal)
    # map each anchor to a pseudo-date, read the RETURNS there (gate stays put)
    pseudo = shifted.reindex(base.index)
    best = -9.9
    for h in GRID_H:
        for vname, rmap in (("svxy", svc), ("nvix", None)):
            r = svxy_ret[h] if vname == "svxy" else nvix_ret[h]
            vals = r.reindex(pseudo.values)
            tmp = base.assign(_r=vals.values)
            if vname == "svxy":
                tmp = tmp[pd.Index(pseudo.values).isin(svc)]
            for rw in GRID_RW:
                t2 = tmp[tmp["runway_td"] >= rw]
                for lo, hi in GRID_BANDS:
                    m = (t2["rel"] > lo) & (t2["rel"] <= hi)
                    v = t2.loc[m, "_r"].dropna()
                    if len(v) >= 10:
                        best = max(best, float(v.mean()))
                    if (vname == "svxy" and h == 1 and rw == 3 and (lo, hi) == (5, 15)
                            and len(v) >= 10):
                        if float(v.mean()) >= obs:
                            uncharged_hits += 1
    if best >= obs:
        charged_hits += 1
print(f"  observed defended-cell mean = {100*obs:+.3f}%  (n={len(DEF)})")
print(f"  UNCHARGED p (same cell under the null)  = {uncharged_hits/NB:.4f}  ({NB} draws)")
print(f"  CHARGED   p (grid max >= defended stat) = {charged_hits/NB:.4f}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 120)
print("12. COST")
print("=" * 120)
for lab, sub in (("all-era", DEF), ("-0.5x era", post)):
    if len(sub):
        print(f"  {lab}: {100*sub['svxy'].mean():+.3f}% per episode vs ~5 bp round "
              f"trip -> {100*100*sub['svxy'].mean()/5:.1f}x")

print("\n" + "=" * 120)
print("13. HORIZON SCAN h=1..10 (episode level, min_gap=h)")
print("=" * 120)
for lab, idxs in (("all-era band cell", DEF.index), ("-0.5x era band cell", post.index)):
    if not len(idxs):
        continue
    print(f"  {lab}")
    show(horizon_scan(px, idxs, [("SVXY", 1.0)], tuple(range(1, 11))), lab)
