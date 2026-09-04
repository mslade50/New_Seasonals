"""C1 ROUND 2 -- on the REFRAMED object.

a3 changed what C1 is. The coherent cell is not "payrolls": it is THE LAST
SCHEDULED PRINT BEFORE A CLEAR CALENDAR, out of a dead 21-day VIX range,
entered MOC the session before and exited at the print close. NFP is today's
instance and is the kind that most reliably HAS a clear runway (87.5% of NFP
prints have >= 3 sessions to the next print; PPI only 37.6%).

So round 2 is re-run on the pooled clear-calendar object, with the NFP-only
sub-cell reported beside it every time:
  1. decluster + concentration (drop-top-k, per-year, top episodes)
  2. definition neighbours (gate threshold, range definition, runway threshold,
     anchor offset, and the production signal's extra legs)
  3. era / regime split (pre-post 2018 done in a4; here: thirds, LOYO)
  4. gate attribution on the pooled object
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa: F401,F403,E402
from pitch_lab import (close_panel, fwd_lag, summarize, sign_test, load_events,
                       rolling_on_valid, show, anchor_positions, declusters,
                       bootstrap_p_le0, cluster_note, local_control)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)

px = close_panel(["^VIX", "^VIX3M", "SVXY", "UVXY", "SPY"])
cal = px["SPY"].dropna().index
vix = px["^VIX"]
rng21 = (rolling_on_valid(vix, lambda x: x.rolling(21).max())
         - rolling_on_valid(vix, lambda x: x.rolling(21).min()))
REL = rolling_on_valid(rng21 / rolling_on_valid(vix, lambda x: x.rolling(21).mean()),
                       lambda x: x.rolling(252).rank(pct=True) * 100)
ABS = rolling_on_valid(rng21, lambda x: x.rolling(252).rank(pct=True) * 100)
rv21 = rolling_on_valid(px["SPY"].pct_change(),
                        lambda x: x.rolling(21).std() * np.sqrt(252) * 100)
RV = rolling_on_valid(rv21, lambda x: x.rolling(252).rank(pct=True) * 100)
sma20 = rolling_on_valid(vix, lambda x: x.rolling(20, min_periods=16).mean())

KINDS = ("nfp", "cpi", "ppi", "fomc_decision")
EV = {k: load_events([k])["date"] for k in KINDS}
ALL_PRINTS = pd.DatetimeIndex(sorted(pd.concat(list(EV.values())).unique()))
pos = pd.Series(range(len(cal)), index=cal)
svxy_h1 = fwd_lag(px["SVXY"].dropna(), 1, lag=1)
vix_h1 = -fwd_lag(px["^VIX"].dropna(), 1, lag=1)


def frame(k=-2):
    out = []
    for kind in KINDS:
        p, kept = anchor_positions(cal, EV[kind], k)
        for i, ap in enumerate(p):
            d0 = kept[i]
            nxt = ALL_PRINTS[ALL_PRINTS > d0]
            if len(nxt) == 0:
                rw = 99
            else:
                rw = int(pos.get(nxt[0], int(cal.searchsorted(nxt[0])))
                         - pos.get(d0, int(cal.searchsorted(d0))))
            out.append({"anchor": cal[ap], "kind": kind, "runway_td": rw})
    df = pd.DataFrame(out).set_index("anchor").sort_index()
    df["rel"] = REL.reindex(df.index).values
    df["abs"] = ABS.reindex(df.index).values
    df["rv"] = RV.reindex(df.index).values
    df["vix"] = vix.reindex(df.index).values
    df["sma"] = sma20.reindex(df.index).values
    df["svxy"] = svxy_h1.reindex(df.index).values
    df["nvix"] = vix_h1.reindex(df.index).values
    return df


def dedupe(df):
    """One ROW PER ANCHOR DATE. Two print kinds can map to the same k=-2 anchor
    (CPI and PPI a day apart, or two prints on one date), which would book the
    same one-session trade twice and inflate N. Keep the MOST CONSERVATIVE
    runway (min) for that date -- if any print at this anchor is crowded from
    behind, the calendar is not clear.
    Found 2026-09-03 when a3's pooled frame raised a duplicate-label reindex;
    a3's pooled numbers below are RESTATED on the deduped basis."""
    g = df.groupby(level=0)
    out = df[~df.index.duplicated(keep="first")].copy()
    out["runway_td"] = g["runway_td"].min()
    out["kind"] = g["kind"].apply(lambda x: "+".join(sorted(set(x))))
    return out.sort_index()


F_RAW = frame(-2)
print("=" * 118)
print("0. DUPLICATE-ANCHOR CORRECTION (a3's pooled frame double-counted days)")
dup = F_RAW.index.duplicated(keep=False)
print(f"   raw anchor rows {len(F_RAW)}, distinct anchor dates "
      f"{F_RAW.index.nunique()}, rows sharing a date {int(dup.sum())}")
_p_raw = F_RAW[(F_RAW["rel"] <= 15) & (F_RAW["runway_td"] >= 3)]
F = dedupe(F_RAW)
_p_ded = F[(F["rel"] <= 15) & (F["runway_td"] >= 3)]
for side, col in (("SVXY", "svxy"), ("-^VIX", "nvix")):
    a = _p_raw[col].dropna(); b = _p_ded[col].dropna()
    print(f"   {side} pooled clear-calendar: RAW n={len(a)} mean {100*a.mean():+.3f}% "
          f"hit {100*(a>0).mean():.1f}%  ->  DEDUPED n={len(b)} mean "
          f"{100*b.mean():+.3f}% hit {100*(b>0).mean():.1f}%")
print("   every number below this line is on the DEDUPED basis.")

POOL = F[(F["rel"] <= 15) & (F["runway_td"] >= 3)]
NFPC = POOL[POOL["kind"].str.contains("nfp")]


def cell(v, label):
    v = pd.Series(v).dropna()
    st = summarize(v.values, label)
    if st["n"]:
        st["signp"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
        st["rec"] = f"{int((v>0).sum())}-{int((v<0).sum())}"
    return st


print("=" * 118)
print(f"OBJECT: gate rel-range<=15 AND runway>=3 td, anchor k=-2, entry lag=1, h=1")
print(f"  POOLED n_anchors={len(POOL)} (SVXY-covered {int(POOL['svxy'].notna().sum())}), "
      f"span {POOL.index[0].date()}..{POOL.index[-1].date()}")
print(f"  NFP sub-cell n_anchors={len(NFPC)} (SVXY-covered "
      f"{int(NFPC['svxy'].notna().sum())})")
print(f"  LIVE: rel {REL.iloc[-1]:.2f}, next print after 2026-09-04 is 2026-09-10 "
      f"-> runway 4 td, VIX {vix.iloc[-1]:.2f} vs 20d SMA {sma20.iloc[-1]:.2f}")
print("=" * 118)

# ===========================================================================
print("\n1. DECLUSTER + CONCENTRATION")
for nm, S in (("POOLED", POOL), ("NFP-only", NFPC)):
    for side, col in (("SVXY", "svxy"), ("-^VIX", "nvix")):
        v = S[col].dropna()
        epi = declusters(v.index, 5, cal)
        e10 = declusters(v.index, 21, cal)
        print(f"\n  {nm} {side}: n={len(v)}; declustered 5td -> {len(epi)}, "
              f"21td -> {len(e10)}")
        rows = [cell(v.values, "day-level"),
                cell(v.reindex(epi).values, "declustered 5td"),
                cell(v.reindex(e10).values, "declustered 21td")]
        show(rows, f"{nm} {side} declustering")
        print("   " + cluster_note(v.index, v.values, k=3))
        o = np.argsort(-v.values)
        for k in (1, 3, 5):
            keep = np.delete(v.values, o[:k])
            st = summarize(keep)
            print(f"   drop top {k}: n={st['n']} mean {st['mean_pct']:+.3f}% "
                  f"hit {st['hit']:.1f}% sign p "
                  f"{sign_test(int((keep>0).sum()), len(keep)):.4f}")
        by = pd.Series(v.values).groupby(v.index.year.values)
        yr = pd.DataFrame({"n": by.size(), "mean_pct": (100*by.mean()).round(2),
                           "wins": by.apply(lambda x: int((x > 0).sum()))})
        print("   per year:\n" + yr.T.to_string())

# ===========================================================================
print("\n" + "=" * 118)
print("2. DEFINITION NEIGHBOURS")
print("=" * 118)
print("\n2a. gate threshold ladder (rel-range pctile), runway>=3 held")
rows = []
for thr in (5, 10, 15, 20, 25, 30, 40, 50, 100):
    S = F[(F["rel"] <= thr) & (F["runway_td"] >= 3)]
    s1 = cell(S["svxy"].values, f"SVXY thr<={thr}")
    s2 = cell(S["nvix"].values, f"-^VIX thr<={thr}")
    rows.append({"thr": thr, "n_anch": len(S),
                 "svxy_n": s1.get("n"), "svxy_mean": round(s1.get("mean_pct", np.nan), 3),
                 "svxy_hit": round(s1.get("hit", np.nan), 1), "svxy_p": s1.get("signp"),
                 "vix_n": s2.get("n"), "vix_mean": round(s2.get("mean_pct", np.nan), 3),
                 "vix_hit": round(s2.get("hit", np.nan), 1), "vix_p": s2.get("signp")})
print(pd.DataFrame(rows).to_string(index=False))

print("\n2b. range DEFINITION (rel / abs / SPY realised vol), thr<=15, runway>=3")
rows = []
for lbl, m in (("rel range (pitched)", F["rel"] <= 15), ("abs range", F["abs"] <= 15),
               ("SPY realised vol", F["rv"] <= 15),
               ("rel<=15 AND abs<=15", (F["rel"] <= 15) & (F["abs"] <= 15))):
    S = F[m & (F["runway_td"] >= 3)]
    rows.append(cell(S["svxy"].values, f"SVXY | {lbl} (n_anch={len(S)})"))
    rows.append(cell(S["nvix"].values, f"-^VIX | {lbl}"))
show(rows, "range definition robustness")

print("\n2c. runway threshold ladder (gate held at rel<=15)")
rows = []
for rw in (0, 1, 2, 3, 4, 5, 7):
    S = F[(F["rel"] <= 15) & (F["runway_td"] >= rw)]
    rows.append(cell(S["svxy"].values, f"SVXY runway>={rw} (n_anch={len(S)})"))
show(rows, "runway ladder, SVXY")
rows = []
for rw in (0, 1, 2, 3, 4, 5, 7):
    S = F[(F["rel"] <= 15) & (F["runway_td"] >= rw)]
    rows.append(cell(S["nvix"].values, f"-^VIX runway>={rw}"))
show(rows, "runway ladder, short ^VIX")

print("\n2d. anchor offset ladder k=-6..+2 (gate + runway held, h=1)")
rows = []
for k in range(-6, 3):
    Fk = dedupe(frame(k))
    S = Fk[(Fk["rel"] <= 15) & (Fk["runway_td"] >= 3)]
    r1 = cell(S["svxy"].values, f"SVXY k={k:+d}")
    r2 = cell(S["nvix"].values, f"-^VIX k={k:+d}")
    rows.append({"k": k, "n_anch": len(S),
                 "svxy_n": r1.get("n"), "svxy_mean": round(r1.get("mean_pct", np.nan), 3),
                 "svxy_hit": round(r1.get("hit", np.nan), 1),
                 "vix_n": r2.get("n"), "vix_mean": round(r2.get("mean_pct", np.nan), 3),
                 "vix_hit": round(r2.get("hit", np.nan), 1)})
print(pd.DataFrame(rows).to_string(index=False))
print("   k=-2 is the pitched anchor; h=1 then exits at the print close.")

print("\n2e. the PRODUCTION signal's extra legs (VIX>13 and VIX>20d SMA).")
print("    today VIX 15.20 > 13 and > its 20d SMA, so both are ON. Do they help?")
rows = []
base = (F["rel"] <= 15) & (F["runway_td"] >= 3)
for lbl, m in (("pitch gate only", base),
               ("+ VIX > 13", base & (F["vix"] > 13)),
               ("+ VIX > 20d SMA", base & (F["vix"] > F["sma"])),
               ("+ both (= production VRC legs)",
                base & (F["vix"] > 13) & (F["vix"] > F["sma"]))):
    S = F[m]
    rows.append(cell(S["svxy"].values, f"SVXY | {lbl} (n_anch={len(S)})"))
    rows.append(cell(S["nvix"].values, f"-^VIX | {lbl}"))
show(rows, "production-signal legs added to the pitch gate")

# ===========================================================================
print("\n" + "=" * 118)
print("3. ERA / REGIME")
print("=" * 118)
for nm, S in (("POOLED", POOL), ("NFP-only", NFPC)):
    for side, col in (("SVXY", "svxy"), ("-^VIX", "nvix")):
        v = S[col].dropna()
        if len(v) < 9:
            continue
        thirds = np.array_split(np.arange(len(v)), 3)
        rows = [cell(v.values[t], f"{side} third {i+1} "
                     f"({v.index[t[0]].date()}..{v.index[t[-1]].date()})")
                for i, t in enumerate(thirds)]
        show(rows, f"{nm} {side}: sample thirds")
        print("   leave-one-YEAR-out means:")
        outs = []
        for y in sorted(set(v.index.year)):
            k = v[v.index.year != y]
            outs.append((y, 100 * k.mean(), int((k > 0).sum()), len(k)))
        print("   " + " | ".join(f"{y}:{m:+.2f}%" for y, m, _, _ in outs))
        print(f"   LOYO min {min(o[1] for o in outs):+.3f}%  "
              f"max {max(o[1] for o in outs):+.3f}%  (full {100*v.mean():+.3f}%)")
        print(f"   sign of every LOYO mean positive: "
              f"{all(o[1] > 0 for o in outs)}")

print("\n3b. midterm / September / month splits on the POOLED cell (today is a")
print("    September print in a midterm year)")
v = POOL["svxy"].dropna()
rows = []
for lbl, m in (("midterm years (y%4==2)", v.index.year % 4 == 2),
               ("non-midterm", v.index.year % 4 != 2),
               ("September prints", v.index.month == 9),
               ("non-September", v.index.month != 9)):
    rows.append(cell(v.values[m], f"SVXY | {lbl}"))
show(rows, "POOLED cell calendar splits")
sept_mid = v[(v.index.year % 4 == 2) & (v.index.month == 9)]
print(f"   TODAY'S EXACT CELL (September, midterm, gated, clear calendar): "
      f"n={len(sept_mid)}"
      + (f" -> {', '.join(f'{d.date()}:{100*r:+.2f}%' for d, r in sept_mid.items())}"
         if len(sept_mid) else " -- EMPTY, as the surface map said"))

# ===========================================================================
print("\n" + "=" * 118)
print("4. GATE ATTRIBUTION ON THE POOLED OBJECT")
print("=" * 118)
rows = []
S_on = F[(F["rel"] <= 15) & (F["runway_td"] >= 3)]
S_off = F[(F["rel"] > 15) & (F["runway_td"] >= 3)]
allday_sv = svxy_h1.dropna()
allday_vx = vix_h1.dropna()
gate_days = cal[(REL.reindex(cal) <= 15).fillna(False).values]
nonanchor_gate = gate_days.difference(F.index)
for side, col, base in (("SVXY", "svxy", allday_sv), ("-^VIX", "nvix", allday_vx)):
    rows = [cell(S_on[col].values, f"{side} | clear-calendar anchor + gate ON"),
            cell(S_off[col].values, f"{side} | clear-calendar anchor, gate OFF"),
            cell(base.reindex(nonanchor_gate).values,
                 f"{side} | gate ON, NO print anchor (gate alone)"),
            cell(base.values, f"{side} | all days, full history")]
    show(rows, f"{side} h=1 gate attribution")
    on = rows[0]["mean_pct"]
    off = rows[1]["mean_pct"]
    alone = rows[2]["mean_pct"]
    allday = rows[3]["mean_pct"]
    print(f"   decomposition: all days {allday:+.3f}%  -> gate alone {alone:+.3f}%  "
          f"-> anchor without gate {off:+.3f}%  -> BOTH {on:+.3f}%")
    print(f"   gate's marginal contribution at the anchor = {on-off:+.3f}pp; "
          f"anchor's marginal contribution at the gate = {on-alone:+.3f}pp")

print("\n4b. LOCAL control (+/-126td around the pooled anchors, anchors removed)")
valid = svxy_h1.dropna().index
loc = local_control(valid, POOL.index.intersection(valid), 126)
show([cell(POOL["svxy"].values, "POOLED anchors"),
      cell(svxy_h1.reindex(loc).values, "local +/-126td ex-anchor"),
      cell(allday_sv.values, "all days")], "SVXY local control")
loc2 = local_control(vix_h1.dropna().index,
                     POOL.index.intersection(vix_h1.dropna().index), 126)
show([cell(POOL["nvix"].values, "POOLED anchors"),
      cell(vix_h1.reindex(loc2).values, "local +/-126td ex-anchor"),
      cell(allday_vx.values, "all days")], "short ^VIX local control")
