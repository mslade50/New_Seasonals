"""C1 ROUND 3 (_dev) -- develop the survivor: horizon, entry form, exits,
and the one open problem round 2 raised.

THE OPEN PROBLEM, taken first because it can still kill the idea:
round 2's threshold ladder is NOT monotone at the tight end. On the pooled
clear-calendar object, long SVXY reads -0.096% (52.0% hit, n=25) at rel-range
pctile <= 5, against +0.536 / +0.910 / +0.664 at <= 10 / <= 15 / <= 20.
**Today's live reading is 3.57**, i.e. inside the one rung where the pitched
vehicle has no edge. Short ^VIX at the same rung is +1.178% (69.4%, n=49,
sign p 0.0047), so the two legs disagree exactly where it matters. Section 0
decides whether that dip is a real dose-response inversion (a filter that does
not filter -> KILL) or a coverage artifact of SVXY's 2011 inception.

Then: horizon_scan 1..10 to CHOOSE h (h=1 vs h=2 settled on evidence), entry
form as WHOLE variants (MOC vs a close-anchored LIMIT at k ATR, with fill
rates and per-SIGNAL expectancy so no marginal-fill decomposition happens),
exits (target/stop sensitivity on the session's own High/Low, pessimistic
grader convention: a bar touching both books the stop), and episode_paths on
the losers so what_kills_it can quote a number.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa: F401,F403,E402
from pitch_lab import (close_panel, load_prices, fwd_lag, summarize, sign_test,
                       load_events, rolling_on_valid, show, anchor_positions,
                       horizon_scan, episode_paths, bootstrap_p_le0, declusters,
                       wilder_atr)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)

RAW = load_prices(["SVXY", "UVXY", "^VIX", "SPY", "^VIX3M"])
px = close_panel(["SVXY", "UVXY", "^VIX", "SPY", "^VIX3M"])
cal = px["SPY"].dropna().index
vix = px["^VIX"]
rng21 = (rolling_on_valid(vix, lambda x: x.rolling(21).max())
         - rolling_on_valid(vix, lambda x: x.rolling(21).min()))
REL = rolling_on_valid(rng21 / rolling_on_valid(vix, lambda x: x.rolling(21).mean()),
                       lambda x: x.rolling(252).rank(pct=True) * 100)

KINDS = ("nfp", "cpi", "ppi", "fomc_decision")
EV = {k: load_events([k])["date"] for k in KINDS}
ALL_PRINTS = pd.DatetimeIndex(sorted(pd.concat(list(EV.values())).unique()))
pos = pd.Series(range(len(cal)), index=cal)
rows = []
for kind in KINDS:
    p, kept = anchor_positions(cal, EV[kind], -2)
    for i, ap in enumerate(p):
        d0 = kept[i]
        nxt = ALL_PRINTS[ALL_PRINTS > d0]
        rw = 99 if len(nxt) == 0 else int(
            pos.get(nxt[0], int(cal.searchsorted(nxt[0])))
            - pos.get(d0, int(cal.searchsorted(d0))))
        rows.append({"anchor": cal[ap], "kind": kind, "runway_td": rw})
F = pd.DataFrame(rows).set_index("anchor").sort_index()
g = F.groupby(level=0)
F = F[~F.index.duplicated(keep="first")].assign(
    runway_td=g["runway_td"].min(),
    kind=g["kind"].apply(lambda x: "+".join(sorted(set(x)))))
F["rel"] = REL.reindex(F.index).values
CLEAR = F["runway_td"] >= 3
POOL = F[(F["rel"] <= 15) & CLEAR].index
NFPA = F[(F["rel"] <= 15) & CLEAR & F["kind"].str.contains("nfp")].index
sv_h1 = fwd_lag(px["SVXY"].dropna(), 1, lag=1)
vx_h1 = -fwd_lag(px["^VIX"].dropna(), 1, lag=1)


def cell(v, label):
    v = pd.Series(v).dropna()
    st = summarize(v.values, label)
    if st["n"]:
        st["signp"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
        st["rec"] = f"{int((v>0).sum())}-{int((v<0).sum())}"
    return st


print("=" * 118)
print("0. THE TIGHT-RUNG PROBLEM -- is the <=5 dip a dose inversion or coverage?")
print(f"   LIVE rel-range pctile = {REL.iloc[-1]:.2f}  (inside the <=5 rung)")
print("=" * 118)
print("\n0a. THE SAME RUNGS ON MATCHED DATES. SVXY starts 2011-10-04 and ^VIX")
print("    starts 2000, so the two legs' <=5 samples are NOT the same days.")
print("    Restrict ^VIX to SVXY-covered anchors and re-read the ladder.")
sv_dates = sv_h1.dropna().index
tbl = []
for lo, hi in ((0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 50), (50, 101)):
    a = F[(F["rel"] > lo if lo else F["rel"] >= 0) & (F["rel"] <= hi) & CLEAR].index
    a_m = a.intersection(sv_dates)
    s = cell(sv_h1.reindex(a_m).values, "")
    v_full = cell(vx_h1.reindex(a).values, "")
    v_m = cell(vx_h1.reindex(a_m).values, "")
    tbl.append({"rung": f"({lo},{hi}]", "n_anch": len(a), "n_matched": len(a_m),
                "SVXY_mean": round(s.get("mean_pct", np.nan), 3),
                "SVXY_hit": round(s.get("hit", np.nan), 1),
                "VIX_full_n": v_full.get("n"),
                "VIX_full_mean": round(v_full.get("mean_pct", np.nan), 3),
                "VIX_matched_mean": round(v_m.get("mean_pct", np.nan), 3),
                "VIX_matched_hit": round(v_m.get("hit", np.nan), 1)})
print(pd.DataFrame(tbl).to_string(index=False))
print("\n    reading: if VIX_matched_mean also dips in (0,5] then the inversion is")
print("    REAL and shared; if only SVXY dips it is a 25-observation vehicle")
print("    artifact and the ^VIX column is the honest dose response.")

print("\n0b. the (0,5] anchors themselves, SVXY-covered, with outcomes")
a5 = F[(F["rel"] <= 5) & CLEAR].index.intersection(sv_dates)
det = pd.DataFrame({"rel": REL.reindex(a5).round(2),
                    "kind": F.loc[a5, "kind"],
                    "svxy_h1": (100 * sv_h1.reindex(a5)).round(2),
                    "nvix_h1": (100 * vx_h1.reindex(a5)).round(2)})
print(det.to_string())
print(f"   SVXY (0,5]: {int((det['svxy_h1']>0).sum())}-"
      f"{int((det['svxy_h1']<0).sum())}, mean {det['svxy_h1'].mean():+.3f}%; "
      f"short ^VIX on the SAME days: {int((det['nvix_h1']>0).sum())}-"
      f"{int((det['nvix_h1']<0).sum())}, mean {det['nvix_h1'].mean():+.3f}%")
print(f"   the two legs disagree on {int((np.sign(det['svxy_h1'])!=np.sign(det['nvix_h1'])).sum())} "
      f"of {len(det)} days -- SVXY is a TERM-STRUCTURE product, ^VIX is spot, so")
print("   a day where spot falls and the front future does not is a real wedge.")

print("\n0c. era-controlled: is the (0,5] dip just recent years?")
for lbl, m in (("(0,5] pre-2018", (F["rel"] <= 5) & CLEAR & (F.index < "2018-02-28")),
               ("(0,5] post-2018", (F["rel"] <= 5) & CLEAR & (F.index >= "2018-02-28")),
               ("(5,15] pre-2018", (F["rel"] > 5) & (F["rel"] <= 15) & CLEAR & (F.index < "2018-02-28")),
               ("(5,15] post-2018", (F["rel"] > 5) & (F["rel"] <= 15) & CLEAR & (F.index >= "2018-02-28"))):
    a = F[m].index
    show([cell(sv_h1.reindex(a.intersection(sv_dates)).values, f"SVXY {lbl}"),
          cell(vx_h1.reindex(a).values, f"-^VIX {lbl}")], lbl)

print("\n0d. LIVE-STATE-MATCHED cell: rel <= 5 AND clear calendar AND the")
print("    production legs (VIX>13, VIX>20d SMA) -- today's exact configuration")
sma20 = rolling_on_valid(vix, lambda x: x.rolling(20, min_periods=16).mean())
live = F[(F["rel"] <= 5) & CLEAR
         & (vix.reindex(F.index) > 13).values
         & (vix.reindex(F.index) > sma20.reindex(F.index)).values].index
print(f"    n_anchors={len(live)}: {', '.join(str(d.date()) for d in live)}")
show([cell(sv_h1.reindex(live).values, "SVXY, live-matched"),
      cell(vx_h1.reindex(live).values, "-^VIX, live-matched")], "today's exact rung")

# ===========================================================================
print("\n" + "=" * 118)
print("1. HORIZON SCAN 1..10 -- CHOOSE h, do not assume it")
print("=" * 118)
for nm, A in (("POOLED clear-calendar", POOL), ("NFP-only", NFPA)):
    show(horizon_scan(px, A, [("SVXY", 1.0)], hs=tuple(range(1, 11)), lag=1, min_gap=5),
         f"{nm}: long SVXY, episode level")
    show(horizon_scan(px, A, [("^VIX", -1.0)], hs=tuple(range(1, 11)), lag=1, min_gap=5),
         f"{nm}: short ^VIX, episode level")

print("\n1b. h=1 vs h=2 SETTLED -- day-level, paired on the same anchors.")
print("    h=1 exits at the PRINT close. h=2 holds one session PAST the print,")
print("    i.e. it is no longer an event trade for that extra session.")
for nm, A in (("POOLED", POOL), ("NFP-only", NFPA)):
    r1 = fwd_lag(px["SVXY"].dropna(), 1, lag=1).reindex(A).dropna()
    r2 = fwd_lag(px["SVXY"].dropna(), 2, lag=1).reindex(A).dropna()
    both = r1.index.intersection(r2.index)
    r1, r2 = r1.reindex(both), r2.reindex(both)
    extra = (1 + r2) / (1 + r1) - 1.0        # the marginal session AFTER the print
    show([cell(r1.values, "h=1 (exit at print close)"),
          cell(r2.values, "h=2 (one session past)"),
          cell(extra.values, "the EXTRA session alone")], f"{nm}: h=1 vs h=2")
    print(f"    sd h=1 {100*r1.std():.3f}pp -> h=2 {100*r2.std():.3f}pp "
          f"({r2.std()/r1.std():.2f}x); mean/sd {r1.mean()/r1.std():+.4f} -> "
          f"{r2.mean()/r2.std():+.4f}")
    print(f"    the extra session on its own: mean {100*extra.mean():+.3f}%, "
          f"hit {100*(extra>0).mean():.1f}%, sign p "
          f"{sign_test(int((extra>0).sum()), len(extra)):.4f}, worst "
          f"{100*extra.min():+.2f}%")

# ===========================================================================
print("\n" + "=" * 118)
print("2. ENTRY FORM -- WHOLE VARIANTS, per-SIGNAL expectancy (unfilled = 0)")
print("   MOC at the close of the session before the print, versus a")
print("   close-anchored LIMIT at close(anchor) - k*ATR worked during that")
print("   session. Fill = the session's LOW touches the limit; fill price is")
print("   min(limit, that session's OPEN). Exit is MOC at the print close in")
print("   BOTH variants, so the comparison is entry form only.")
print("=" * 118)
sv = RAW["SVXY"]
atr = pd.Series(np.asarray(wilder_atr(sv["High"], sv["Low"], sv["Close"], 14)).ravel(),
                index=sv.index)
svi = sv.index
spos = pd.Series(range(len(svi)), index=svi)
print(f"   live SVXY close {sv['Close'].iloc[-1]:.2f}, Wilder-14 ATR "
      f"{atr.iloc[-1]:.3f} ({100*atr.iloc[-1]/sv['Close'].iloc[-1]:.2f}% of price)")
for nm, A in (("POOLED", POOL), ("NFP-only", NFPA)):
    out = []
    for k in (0.0, 0.10, 0.25, 0.40, 0.60):
        fills, rets = 0, []
        per_signal = []
        n_sig = 0
        for d in pd.DatetimeIndex(A):
            p = spos.get(d)
            if p is None or p + 2 >= len(svi):
                continue
            anc_close = sv["Close"].iloc[p]
            a = atr.iloc[p]
            if not np.isfinite(a) or not np.isfinite(anc_close):
                continue
            n_sig += 1
            e_open = sv["Open"].iloc[p + 1]
            e_low = sv["Low"].iloc[p + 1]
            x_close = sv["Close"].iloc[p + 2]
            if k == 0.0:
                entry = sv["Close"].iloc[p + 1]      # MOC
                filled = True
            else:
                lim = anc_close - k * a
                filled = bool(e_low <= lim)
                entry = min(lim, e_open) if filled else np.nan
            if filled and np.isfinite(entry) and entry > 0:
                fills += 1
                r = x_close / entry - 1.0
                rets.append(r)
                per_signal.append(r)
            else:
                per_signal.append(0.0)
        lbl = "MOC (k=0)" if k == 0 else f"LIMIT close-{k:.2f}ATR"
        st = summarize(np.array(per_signal), f"{lbl} per-SIGNAL")
        st["fill_rate"] = round(100 * fills / max(1, n_sig), 1)
        st["n_sig"] = n_sig
        if rets:
            st["on_fills_pct"] = round(100 * np.mean(rets), 3)
            st["fills_hit"] = round(100 * np.mean(np.array(rets) > 0), 1)
        out.append(st)
    show(out, f"{nm}: entry-form whole variants (mean_pct IS per-signal expectancy)")

# ===========================================================================
print("\n" + "=" * 118)
print("3. EXITS -- target / stop sensitivity inside the single held session.")
print("   Grader convention (pessimistic): a bar touching BOTH books the STOP.")
print("=" * 118)
for nm, A in (("POOLED", POOL), ("NFP-only", NFPA)):
    out = []
    for tgt, stp in ((None, None), (1.0, None), (2.0, None), (None, 1.0),
                     (None, 1.5), (1.0, 1.0), (2.0, 1.5)):
        vals = []
        for d in pd.DatetimeIndex(A):
            p = spos.get(d)
            if p is None or p + 2 >= len(svi):
                continue
            entry = sv["Close"].iloc[p + 1]
            a = atr.iloc[p + 1]
            if not np.isfinite(a) or not np.isfinite(entry):
                continue
            hi, lo, cl = (sv["High"].iloc[p + 2], sv["Low"].iloc[p + 2],
                          sv["Close"].iloc[p + 2])
            hit_s = stp is not None and lo <= entry - stp * a
            hit_t = tgt is not None and hi >= entry + tgt * a
            if hit_s:
                vals.append((entry - stp * a) / entry - 1.0)
            elif hit_t:
                vals.append((entry + tgt * a) / entry - 1.0)
            else:
                vals.append(cl / entry - 1.0)
        lbl = (f"tgt {tgt if tgt else '-'} / stp {stp if stp else '-'} ATR")
        out.append(cell(np.array(vals), lbl))
    show(out, f"{nm}: exit variants on the single held session")
print("   time-only is the null; anything that does not beat it materially is")
print("   machinery for its own sake on a ONE-SESSION hold.")

# ===========================================================================
print("\n" + "=" * 118)
print("4. LOSER PATHS -- what_kills_it needs a number, not a generic risk")
print("=" * 118)
for nm, A in (("POOLED", POOL), ("NFP-only", NFPA)):
    paths = episode_paths(px, pd.DatetimeIndex(A), [("SVXY", 1.0)], h=3, lag=1)
    lose = paths[paths[1] < 0]
    print(f"\n   {nm}: {len(lose)} of {len(paths)} anchors lose on the print "
          f"session; mean loss d1 {100*lose[1].mean():+.3f}%, "
          f"worst d1 {100*lose[1].min():+.2f}% on {lose[1].idxmin().date()}")
    print(f"     of those losers, {int((lose[3] < lose[1]).sum())} kept falling "
          f"through d3 (mean d3 {100*lose[3].mean():+.2f}%, worst "
          f"{100*lose[3].min():+.2f}%)")
    print("     worst five print sessions:")
    w = paths[1].nsmallest(5)
    for d, r in w.items():
        print(f"       {d.date()} {100*r:+.2f}%  (d3 {100*paths.loc[d,3]:+.2f}%)  "
              f"^VIX that session {100*(-vx_h1.get(d, np.nan)):+.2f}%")
atr_pct = float(atr.iloc[-1] / sv["Close"].iloc[-1])
print(f"\n   TRANSLATION at today's ATR% ({100*atr_pct:.2f}%): the pooled cell's")
worst = min(episode_paths(px, pd.DatetimeIndex(POOL), [("SVXY", 1.0)], h=1, lag=1)[1])
print(f"   worst print session ({100*worst:.2f}%) is {abs(worst)/atr_pct:.2f} ATR; "
      f"at 45 bps risk that is -${750_000*0.0045*abs(worst)/atr_pct:,.0f} "
      f"(-{100*0.0045*abs(worst)/atr_pct:.2f}% NAV).")
