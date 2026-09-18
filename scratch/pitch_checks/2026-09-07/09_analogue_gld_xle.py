"""ADVERSARIAL ROUND 1 -- kill the two historical-analogue survivors.

Candidate 1: short GLD, h=5.   Candidate 2: short XLE, h=10.
Both came out of 05_analogue_{knn,conjunction,agreement}.py on the strength of
"negative EDGE against the instrument's own drift in all 6 neighbour
constructions, hit rate below base in all 6".

This script does not re-derive the neighbour sets. It imports them and then
attacks the inference. Sections:

  S1  RAW vs EDGE. A short is paid in raw return, not in relative drift.
  S2  Discriminating power of the headline statistic: how many of the 44
      (instrument x horizon) cells in the SAME construction are also
      negative-edge? If nearly all of them are, "negative in all 6" says
      nothing about GLD or XLE.
  S3  Honest N. Pairwise overlap of the 6 sets, union episodes at 21td.
  S4  Mechanism. Neither state vector contains a gold or an energy feature.
      Split the neighbours by the instrument's OWN state and see whether the
      stated mechanism ("gold fails to earn its drift", "fade the energy
      leadership") is what the cell keys on.
  S5  Support. Where does TODAY sit inside the neighbour support on the axes
      the trigger does not mention (2026-08-21 method trap)?
  S6  Placebo anchor ladder on Method A.
  S7  Era: drop 2018 from A, drop 2013 from B.
  S8  XLE reference class across the 9 SPDR sectors + permutation P.
  S9  XLE crude beta, both registry specifications, and the residual.
  S10 XLE corpse-mask overlap + book overlap.
  S11 Cost.

Run: python scratch/pitch_checks/2026-09-07/09_analogue_gld_xle.py
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))

import numpy as np
import pandas as pd

from _analogue_common import (  # noqa: E402
    ASOF, EXCLUDE_TD, FEATURES, INSTRUMENTS, build_features, eligible, load_all,
)
from pitch_lab import (  # noqa: E402
    load_prices, declusters, fwd_lag, sign_test, summarize, pct_rank,
    bootstrap_p_le0, cluster_note,
)

knn_mod = importlib.import_module("05_analogue_knn")
conj_mod = importlib.import_module("05_analogue_conjunction")

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 400)

SPDRS = ["XLE", "XLB", "XLI", "XLF", "XLK", "XLP", "XLU", "XLV", "XLY"]
EXTRA = SPDRS + ["XOP", "OIH", "CL=F", "SLV", "NEM", "DBC", "QQQ"]
GLD_COST_BPS = 4.0   # brief says 3-5
XLE_COST_BPS = 5.0   # brief says 4-6
RNG = np.random.default_rng(20260907)


# ---------------------------------------------------------------------------
def build_sets(px, f, idx):
    pool = eligible(f, FEATURES)
    dist = knn_mod.knn(f, FEATURES, pool, ASOF)
    sets = {
        "A_k20_dec": declusters(pd.DatetimeIndex(sorted(dist.index[:20])), 21, idx),
        "A_k20_greedy": knn_mod.greedy_episodes(dist, 20, idx, 21),
        "A_k50_dec": declusters(pd.DatetimeIndex(sorted(dist.index[:50])), 21, idx),
        "A_k50_greedy": knn_mod.greedy_episodes(dist, 50, idx, 21),
    }
    pool_b = eligible(f, conj_mod.COND_COLS)
    trig = conj_mod.screen(f, pool_b)
    sets["B_episodes"] = declusters(trig, 21, idx)
    sets["B_daylevel"] = trig
    return sets, dist, pool


def short_stats(s: pd.Series, dates, h: int, cost_bps: float) -> dict:
    """Short leg: return = -1 * long forward return, lag=1 MOC entry."""
    r = -fwd_lag(s, h, lag=1)
    v = r.reindex(pd.DatetimeIndex(dates)).dropna()
    if len(v) == 0:
        return {"n": 0}
    base = -fwd_lag(s, h, lag=1).dropna()
    base = base[base.index <= ASOF]
    w = int((v.values > 0).sum())
    mean_bps = 100 * 100 * float(v.mean())
    return {
        "n": len(v),
        "raw_short_pct": round(100 * float(v.mean()), 3),
        "med_pct": round(100 * float(np.median(v.values)), 3),
        "rec": f"{w}-{len(v) - w}",
        "hit": round(100 * w / len(v), 1),
        "base_hit": round(100 * float((base.values > 0).mean()), 1),
        "sign_p": round(sign_test(w, len(v), float((base.values > 0).mean())), 4),
        "worst_pct": round(100 * float(v.min()), 2),
        "cost_x": round(mean_bps / cost_bps, 1) if mean_bps > 0 else round(mean_bps / cost_bps, 1),
        "boot_p_le0": round(bootstrap_p_le0(v.values), 3) if len(v) >= 3 else np.nan,
    }


def edge_grid(px_all: dict, sets: dict, tickers: list[str],
              hs=(1, 3, 5, 10)) -> pd.DataFrame:
    """EDGE (conditional mean minus own all-days drift) for every
    (ticker, horizon, set). This is the statistic the candidates rest on."""
    rows = []
    for tkr in tickers:
        s = px_all[tkr]
        for h in hs:
            r = fwd_lag(s, h, lag=1)
            base = r.dropna()
            base = base[base.index <= ASOF]
            if len(base) < 200:
                continue
            drift = float(base.mean())
            row = {"ticker": tkr, "h": h, "drift_pct": round(100 * drift, 3)}
            for name, dates in sets.items():
                v = r.reindex(pd.DatetimeIndex(dates)).dropna()
                row[name] = (round(100 * float(v.mean()) - 100 * drift, 3)
                             if len(v) else np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    px, breadth_names = load_all()
    f = build_features(px, breadth_names)
    idx = f.index
    sets, dist, pool = build_sets(px, f, idx)

    extra_px = load_prices([t for t in EXTRA if t not in px])
    px_all = {t: px[t]["Close"] for t in px}
    for t, d in extra_px.items():
        px_all[t] = d["Close"]
    # own-calendar series reindexed onto the SPY calendar, no lookahead
    for t in list(px_all):
        px_all[t] = px_all[t].reindex(idx.union(px_all[t].index)).ffill().reindex(idx)

    print("=" * 110)
    print("NEIGHBOUR SETS (imported, not re-derived)")
    print("=" * 110)
    for k, v in sets.items():
        print(f"  {k:<14s} N={len(v):>3d}   {v[0].date()} .. {v[-1].date()}")

    # =====================================================================
    print("\n" + "=" * 110)
    print("S1. RAW vs EDGE -- what a SHORT actually collects")
    print("=" * 110)
    print("A short is paid the negative of the raw forward return. 'Negative")
    print("edge against the instrument's own drift' is a relative-value claim;")
    print("it pays only if the raw side is also negative.\n")
    for tkr, hs, cost in (("GLD", (3, 5), GLD_COST_BPS),
                          ("XLE", (1, 5, 10), XLE_COST_BPS)):
        for h in hs:
            rows = []
            for name, dates in sets.items():
                d = short_stats(px_all[tkr], dates, h, cost)
                d = {"set": name, **d}
                rows.append(d)
            df = pd.DataFrame(rows)
            print(f"\n--- SHORT {tkr}, h={h}, cost {cost} bps round trip "
                  f"(raw_short_pct > 0 = the short makes money) ---")
            print(df.to_string(index=False))
            pos = int((df["raw_short_pct"] > 0).sum())
            print(f"    raw short POSITIVE in {pos} of {len(df)} sets; "
                  f"mean across sets {df['raw_short_pct'].mean():+.3f}%; "
                  f"the two largest-N sets "
                  f"({df.iloc[3]['set']} N={df.iloc[3]['n']}, "
                  f"{df.iloc[5]['set']} N={df.iloc[5]['n']}) give "
                  f"{df.iloc[3]['raw_short_pct']:+.3f}% and "
                  f"{df.iloc[5]['raw_short_pct']:+.3f}%")

    # =====================================================================
    print("\n" + "=" * 110)
    print("S2. DISCRIMINATING POWER of 'negative edge in all 6 constructions'")
    print("=" * 110)
    print("If the neighbour construction produces negative edge for almost")
    print("every instrument, the headline statistic carries no information")
    print("about GLD or XLE specifically. Reference class = the 11 lane")
    print("instruments plus 9 SPDR sectors, XOP, OIH, SLV, NEM, DBC, QQQ.\n")
    ref = sorted(set(INSTRUMENTS) | set(EXTRA) - {"CL=F"})
    grid = edge_grid(px_all, sets, ref)
    setcols = list(sets)
    print("Fraction of (ticker x horizon) cells with NEGATIVE edge, per set:")
    for c in setcols:
        v = grid[c].dropna()
        print(f"  {c:<14s} {100*float((v < 0).mean()):5.1f}% negative "
              f"of {len(v)} cells   median edge {v.median():+.3f}pp   "
              f"mean {v.mean():+.3f}pp")
    grid["n_neg"] = (grid[setcols] < 0).sum(axis=1)
    grid["avg_edge"] = grid[setcols].mean(axis=1)
    unan = grid[grid["n_neg"] == 6]
    print(f"\nCells unanimous-negative across all 6 sets: {len(unan)} of "
          f"{len(grid)} ({100*len(unan)/len(grid):.0f}%)")
    print(unan.sort_values("avg_edge")[["ticker", "h", "drift_pct", "avg_edge"]]
          .to_string(index=False))
    for tkr, h in (("GLD", 5), ("XLE", 10), ("XLE", 5)):
        sub = grid[(grid.ticker == tkr) & (grid.h == h)]
        if len(sub) == 0:
            continue
        val = float(sub["avg_edge"].iloc[0])
        rank = int((grid["avg_edge"] < val).sum()) + 1
        print(f"  {tkr} h={h}: avg edge {val:+.3f}pp ranks {rank} of "
              f"{len(grid)} cells (1 = most negative)")
    print("\nSame question restricted to h=5 and h=10 across the reference class:")
    for h in (5, 10):
        sub = grid[grid.h == h].sort_values("avg_edge")
        print(f"\n  h={h}, all {len(sub)} instruments by avg edge across the 6 sets:")
        print("   " + sub[["ticker", "drift_pct", "avg_edge", "n_neg"]]
              .to_string(index=False).replace("\n", "\n   "))

    # =====================================================================
    print("\n" + "=" * 110)
    print("S3. HONEST N -- the 6 sets are nested views of 2 selections")
    print("=" * 110)
    names = list(sets)
    ov = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            A, B = set(sets[a]), set(sets[b])
            ov.loc[a, b] = len(A & B) / max(1, len(A))
    print("P(date in COLUMN set | date in ROW set):")
    print(ov.round(2).to_string())
    union_all = pd.DatetimeIndex(sorted(set().union(*[set(v) for v in sets.values()])))
    uni_epi = declusters(union_all, 21, idx)
    print(f"\nunion of all 6 sets: {len(union_all)} distinct dates -> "
          f"{len(uni_epi)} independent episodes at a 21td gap")
    a_union = pd.DatetimeIndex(sorted(set(sets['A_k20_dec']) | set(sets['A_k20_greedy'])
                                      | set(sets['A_k50_dec']) | set(sets['A_k50_greedy'])))
    b_union = pd.DatetimeIndex(sorted(set(sets['B_episodes']) | set(sets['B_daylevel'])))
    print(f"  Method A union {len(a_union)} dates -> "
          f"{len(declusters(a_union, 21, idx))} episodes")
    print(f"  Method B union {len(b_union)} dates -> "
          f"{len(declusters(b_union, 21, idx))} episodes")
    print(f"  A-union INTERSECT B-union: {len(set(a_union) & set(b_union))} dates")
    print("\nShort performance on the honest union of independent episodes:")
    for tkr, h, cost in (("GLD", 5, GLD_COST_BPS), ("XLE", 10, XLE_COST_BPS)):
        d = short_stats(px_all[tkr], uni_epi, h, cost)
        print(f"  SHORT {tkr} h={h} on the {len(uni_epi)}-episode union: "
              f"{d['raw_short_pct']:+.3f}% raw, record {d['rec']}, "
              f"hit {d['hit']}% vs base {100-d['base_hit']:.1f}% (short base), "
              f"{d['cost_x']}x cost, bootstrap P(mean<=0) {d['boot_p_le0']}")

    # =====================================================================
    print("\n" + "=" * 110)
    print("S4. MECHANISM -- does the trigger key on what the pitch claims?")
    print("=" * 110)
    gld = px_all["GLD"]
    xle = px_all["XLE"]
    gld_d252h = (gld / gld.rolling(252).max() - 1.0) * 100.0
    gld_dsma200 = (gld / gld.rolling(200).mean() - 1.0) * 100.0
    xle_d252h = (xle / xle.rolling(252).max() - 1.0) * 100.0
    xle_r21 = pct_rank(xle, 21, 252)
    xle_dsma200 = (xle / xle.rolling(200).mean() - 1.0) * 100.0

    print(f"\nTODAY {ASOF.date()}: GLD d252h {gld_d252h.loc[ASOF]:+.2f}%, "
          f"GLD vs 200d {gld_dsma200.loc[ASOF]:+.2f}% | "
          f"XLE d252h {xle_d252h.loc[ASOF]:+.2f}%, XLE r21 "
          f"{xle_r21.loc[ASOF]:.1f}, XLE vs 200d {xle_dsma200.loc[ASOF]:+.2f}%")
    print("\nNEITHER state vector contains a gold or an energy feature. The")
    print("7 Method A features are SPY x4, breadth, ^TNX, TLT-HYG. Method B's")
    print("4 conditions are SPY x2, TLT, HYG. So the selection cannot have")
    print("keyed on 'gold is lagging' or 'energy is leading'. Split and see.\n")

    for label, dates in (("A_k50_greedy", sets["A_k50_greedy"]),
                         ("B_daylevel", sets["B_daylevel"]),
                         ("union_episodes", uni_epi)):
        print(f"\n--- {label} (N={len(dates)}) ---")
        # GLD: is the negative edge in the drawdown half (today's state)?
        d = gld_d252h.reindex(pd.DatetimeIndex(dates)).dropna()
        r5 = -fwd_lag(gld, 5, 1).reindex(d.index)
        deep = d <= -10.0
        print(f"  GLD state at the neighbours: median d252h {d.median():+.2f}%, "
              f"min {d.min():+.2f}%, "
              f"{int(deep.sum())} of {len(d)} more than 10% below the high "
              f"(TODAY is {gld_d252h.loc[ASOF]:+.2f}%)")
        for tag, m in (("GLD >10% below high (today's state)", deep),
                       ("GLD within 10% of high", ~deep)):
            v = r5[m].dropna()
            if len(v) == 0:
                print(f"    {tag:<38s} n=0")
                continue
            w = int((v > 0).sum())
            print(f"    {tag:<38s} n={len(v):>3d} short mean "
                  f"{100*float(v.mean()):+.3f}%  rec {w}-{len(v)-w}")
        # XLE: is the negative edge in the leadership half (today's state)?
        rk = xle_r21.reindex(pd.DatetimeIndex(dates)).dropna()
        r10 = -fwd_lag(xle, 10, 1).reindex(rk.index)
        lead = rk >= 80.0
        print(f"  XLE state at the neighbours: median r21 {rk.median():.1f}, "
              f"{int(lead.sum())} of {len(rk)} with r21>=80 "
              f"(TODAY is {xle_r21.loc[ASOF]:.1f})")
        for tag, m in (("XLE r21>=80 (today's state)", lead),
                       ("XLE r21<80", ~lead)):
            v = r10[m].dropna()
            if len(v) == 0:
                print(f"    {tag:<38s} n=0")
                continue
            w = int((v > 0).sum())
            print(f"    {tag:<38s} n={len(v):>3d} short mean "
                  f"{100*float(v.mean()):+.3f}%  rec {w}-{len(v)-w}")

    # =====================================================================
    print("\n" + "=" * 110)
    print("S5. SUPPORT -- where does today sit inside the neighbour support?")
    print("=" * 110)
    tab = pd.DataFrame({
        "GLD_d252h": gld_d252h, "GLD_dsma200": gld_dsma200,
        "XLE_d252h": xle_d252h, "XLE_r21": xle_r21,
    })
    for label in ("A_k20_dec", "A_k50_greedy", "B_episodes"):
        sub = tab.reindex(pd.DatetimeIndex(sets[label])).dropna(how="all")
        print(f"\n{label} (N={len(sub)}):")
        print(sub.round(2).to_string())
    today = tab.loc[ASOF]
    print(f"\nTODAY: " + "  ".join(f"{c} {today[c]:+.2f}" for c in tab.columns))
    for c in tab.columns:
        for label in ("A_k50_greedy", "B_daylevel"):
            sub = tab[c].reindex(pd.DatetimeIndex(sets[label])).dropna()
            pctl = 100 * float((sub < today[c]).mean())
            print(f"  {c:<12s} today sits at the {pctl:5.1f}th percentile of "
                  f"{label} (n={len(sub)})")

    # =====================================================================
    print("\n" + "=" * 110)
    print("S6. PLACEBO ANCHOR LADDER (Method A only -- Method B is anchor-free)")
    print("=" * 110)
    print("Method B is a static 4-condition screen. It does not reference the")
    print("2026-09-04 tape at all beyond the fact that today passes it, so it")
    print("has no anchor to placebo. Method A does. Rebuild the k=50 greedy")
    print("set with the as-of date shifted back 5/10/21/42/63 sessions and")
    print("re-read the two verdicts. If they survive an anchor from a quarter")
    print("ago, 'today's analogue' is doing no work.\n")
    all_idx = f.index
    rows = []
    for shift in (0, -5, -10, -21, -42, -63, -126):
        pos = int(all_idx.searchsorted(ASOF)) + shift
        anchor = all_idx[pos]
        ok = f[FEATURES].dropna().index
        cut = all_idx[max(0, pos - EXCLUDE_TD)]
        pl = ok[ok < cut]
        if len(pl) < 500:
            continue
        d = knn_mod.knn(f, FEATURES, pl, anchor)
        g = knn_mod.greedy_episodes(d, 50, all_idx, 21)
        gs = short_stats(px_all["GLD"], g, 5, GLD_COST_BPS)
        xs = short_stats(px_all["XLE"], g, 10, XLE_COST_BPS)
        rows.append({
            "shift_td": shift, "anchor": str(anchor.date()), "N": len(g),
            "overlap_with_live": len(set(g) & set(sets["A_k50_greedy"])),
            "GLDshort_h5_pct": gs["raw_short_pct"], "GLD_rec": gs["rec"],
            "XLEshort_h10_pct": xs["raw_short_pct"], "XLE_rec": xs["rec"],
        })
    print(pd.DataFrame(rows).to_string(index=False))

    # =====================================================================
    print("\n" + "=" * 110)
    print("S7. ERA -- drop the disclosed concentrations")
    print("=" * 110)
    for label, drop_year, dates in (
            ("A_k20_greedy ex-2018", 2018, sets["A_k20_greedy"]),
            ("A_k50_greedy ex-2018", 2018, sets["A_k50_greedy"]),
            ("A_k50_dec ex-2018", 2018, sets["A_k50_dec"]),
            ("B_episodes ex-2013", 2013, sets["B_episodes"]),
            ("B_daylevel ex-2013", 2013, sets["B_daylevel"]),
            ("B_daylevel ex-2016", 2016, sets["B_daylevel"])):
        keep = pd.DatetimeIndex([d for d in dates if d.year != drop_year])
        g = short_stats(px_all["GLD"], keep, 5, GLD_COST_BPS)
        x = short_stats(px_all["XLE"], keep, 10, XLE_COST_BPS)
        print(f"  {label:<24s} N {len(dates)}->{len(keep)}   "
              f"GLD short h5 {g['raw_short_pct']:+.3f}% ({g['rec']})   "
              f"XLE short h10 {x['raw_short_pct']:+.3f}% ({x['rec']})")
    print("\nDrop the top-2 episodes by |contribution| (the lane's own ask):")
    for tkr, h, cost, label in (("GLD", 5, GLD_COST_BPS, "A_k50_greedy"),
                                ("GLD", 5, GLD_COST_BPS, "B_episodes"),
                                ("XLE", 10, XLE_COST_BPS, "A_k50_greedy"),
                                ("XLE", 10, XLE_COST_BPS, "B_episodes")):
        r = -fwd_lag(px_all[tkr], h, 1).reindex(pd.DatetimeIndex(sets[label])).dropna()
        order = np.argsort(-np.abs(r.values))
        for k in (0, 1, 2, 3):
            keep = r.drop(r.index[order[:k]]) if k else r
            w = int((keep > 0).sum())
            print(f"  SHORT {tkr} h={h} {label:<13s} drop-{k}: "
                  f"{100*float(keep.mean()):+.3f}%  rec {w}-{len(keep)-w}"
                  + ("" if k else
                     f"   | {cluster_note(r.index, r.values, k=2)}"))

    # =====================================================================
    print("\n" + "=" * 110)
    print("S8. XLE REFERENCE CLASS -- 9 SPDR sectors on the identical dates")
    print("=" * 110)
    for label in ("A_k50_greedy", "B_episodes", "B_daylevel"):
        dates = pd.DatetimeIndex(sets[label])
        obs = {}
        for s in SPDRS:
            r = fwd_lag(px_all[s], 10, 1)
            base = r.dropna()
            base = base[base.index <= ASOF]
            v = r.reindex(dates).dropna()
            obs[s] = 100 * float(v.mean()) - 100 * float(base.mean())
        order = sorted(obs, key=lambda s: obs[s])
        rank = order.index("XLE") + 1
        print(f"\n{label} (N={len(dates)}), h=10 EDGE by sector "
              f"(most negative first):")
        print("  " + "  ".join(f"{s}:{obs[s]:+.2f}" for s in order))
        print(f"  XLE ranks {rank} of 9 (1 = most negative)")
        # permutation: random date sets of the same size, min-of-9 edge
        mins = []
        pool_dates = pd.DatetimeIndex(
            [d for d in idx if d >= pd.Timestamp("2008-04-09") and d <= ASOF])
        rmat = {}
        for s in SPDRS:
            r = fwd_lag(px_all[s], 10, 1)
            rmat[s] = r.reindex(pool_dates)
        base_means = {s: float(rmat[s].dropna().mean()) for s in SPDRS}
        valid = pool_dates[~np.isnan(np.column_stack(
            [rmat[s].values for s in SPDRS])).any(axis=1)]
        for _ in range(3000):
            pick = RNG.choice(len(valid), size=min(len(dates), len(valid)),
                              replace=False)
            dsel = valid[pick]
            e = [100 * float(rmat[s].reindex(dsel).mean()) - 100 * base_means[s]
                 for s in SPDRS]
            mins.append(min(e))
        p = float(np.mean(np.array(mins) <= obs["XLE"]))
        print(f"  permutation P(min-of-9 sector edge <= XLE's {obs['XLE']:+.3f}pp "
              f"on a random date set of the same size) = {p:.3f}  "
              f"[3000 draws]")

    # =====================================================================
    print("\n" + "=" * 110)
    print("S9. XLE CRUDE BETA -- reconciling the registry's 0.479 and 0.112")
    print("=" * 110)
    rx = px_all["XLE"].pct_change()
    for proxy in ("CL=F", "USO"):
        rp = px_all[proxy].pct_change()
        j = pd.concat([rx, rp], axis=1).dropna()
        j.columns = ["xle", "crude"]
        for lab, sub in (("full history", j),
                         ("2008+", j[j.index >= "2008-01-01"]),
                         ("2015+", j[j.index >= "2015-01-01"]),
                         ("last 252d", j.tail(252))):
            b = float(np.polyfit(sub["crude"], sub["xle"], 1)[0])
            corr = float(sub["xle"].corr(sub["crude"]))
            print(f"  XLE daily beta on {proxy:<5s} {lab:<12s} "
                  f"{b:.3f}  (corr {corr:.3f}, n={len(sub)})")
    # h=10 residual on the neighbour episodes
    print("\n  h=10 XLE residual net of contemporaneous crude, per set:")
    for proxy in ("CL=F", "USO"):
        rxh = fwd_lag(px_all["XLE"], 10, 1)
        rph = fwd_lag(px_all[proxy], 10, 1)
        j = pd.concat([rxh, rph], axis=1).dropna()
        j.columns = ["xle", "crude"]
        j = j[j.index <= ASOF]
        b = float(np.polyfit(j["crude"], j["xle"], 1)[0])
        resid = j["xle"] - b * j["crude"]
        base_res = float(resid.mean())
        for label in ("A_k50_greedy", "B_episodes", "B_daylevel"):
            v = resid.reindex(pd.DatetimeIndex(sets[label])).dropna()
            if len(v) == 0:
                continue
            w = int((-v > 0).sum())
            print(f"    {proxy:<5s} h10 beta {b:.3f}  {label:<13s} "
                  f"SHORT-residual mean {-100*float(v.mean()):+.3f}% "
                  f"(base {-100*base_res:+.3f}%)  rec {w}-{len(v)-w}")

    # =====================================================================
    print("\n" + "=" * 110)
    print("S10. XLE CORPSE-MASK OVERLAP + BOOK OVERLAP")
    print("=" * 110)
    uso = px_all["USO"]
    xle_r5 = pct_rank(xle, 5, 252)
    corpses = {
        "2026-08-17 5d energy thrust into 52w high":
            (xle_d252h >= -2.0) & (xle_r5 >= 90),
        "2026-08-28 energy pullback inside thrust near high":
            (xle_d252h >= -2.0) & (xle_r21 >= 65) & (xle.pct_change(3) < 0),
        "2026-08-11 long XLE on a crude 1d thrust":
            (uso.pct_change() >= 0.05),
        "2026-08-24 energy z10 cluster (XLE z10 >= 2 proxy)":
            ((xle.pct_change(10) - xle.pct_change(10).rolling(252).mean())
             / xle.pct_change(10).rolling(252).std() >= 2.0),
    }
    anymask = None
    for m in corpses.values():
        anymask = m.fillna(False) if anymask is None else (anymask | m.fillna(False))
    for label in ("A_k50_greedy", "B_daylevel", "union_episodes"):
        dates = uni_epi if label == "union_episodes" else pd.DatetimeIndex(sets[label])
        print(f"\n{label} (N={len(dates)}):")
        for name, m in corpses.items():
            mm = m.fillna(False)
            inside = int(mm.reindex(dates).fillna(False).sum())
            base = 100 * float(mm.reindex(idx).fillna(False).mean())
            print(f"  P(inside | this mask) = {inside}/{len(dates)} = "
                  f"{100*inside/len(dates):5.1f}%   base rate {base:5.1f}%   "
                  f"{name}")
        inside = int(anymask.reindex(dates).fillna(False).sum())
        base = 100 * float(anymask.reindex(idx).fillna(False).mean())
        print(f"  P(inside ANY corpse | this mask) = {inside}/{len(dates)} = "
              f"{100*inside/len(dates):5.1f}%   base rate {base:5.1f}%")

    led = pd.read_parquet(HERE.parents[2] / "data" / "backtest_trades_full.parquet")
    led["Signal Date"] = pd.to_datetime(led["Signal Date"])
    energy = {"XLE", "XOP", "OIH", "USO", "XOM", "CVX", "COP", "SLB", "EOG",
              "OXY", "PSX", "VLO", "MPC", "HAL", "DVN", "FANG", "HES", "MRO",
              "PXD", "APA", "BKR", "KMI", "WMB", "OKE", "ERX", "ERY", "GUSH",
              "DRIP", "UCO", "SCO", "BNO", "UNG", "XLE"}
    print("\nBOOK OVERLAP: systematic-ledger energy signals inside a +/-5td "
          "window around the union episodes")
    win = set()
    posn = pd.Series(range(len(idx)), index=idx)
    for d in uni_epi:
        p = posn.get(d)
        if p is None:
            continue
        for q in range(max(0, p - 5), min(len(idx), p + 6)):
            win.add(idx[q])
    sub = led[led["Ticker"].isin(energy) & led["Signal Date"].isin(win)]
    if len(sub):
        print(f"  {len(sub)} energy signals; direction mix "
              f"{dict(sub['Direction'].value_counts())}; "
              f"avgR {sub['R_Multiple'].mean():+.3f}")
        print("  by strategy: " + ", ".join(
            f"{k} n={v}" for k, v in sub["Strategy"].value_counts().items()))
        for dirn, g in sub.groupby("Direction"):
            print(f"    {dirn}: n={len(g)} avgR {g['R_Multiple'].mean():+.3f}")
    else:
        print("  none")

    # =====================================================================
    print("\n" + "=" * 110)
    print("S11. COST against the RAW short mean")
    print("=" * 110)
    for tkr, h, cost in (("GLD", 5, GLD_COST_BPS), ("XLE", 10, XLE_COST_BPS)):
        print(f"\n  SHORT {tkr} h={h}, {cost} bps round trip:")
        for label in list(sets) + ["union_episodes"]:
            dates = uni_epi if label == "union_episodes" else sets[label]
            d = short_stats(px_all[tkr], dates, h, cost)
            if not d.get("n"):
                continue
            print(f"    {label:<15s} N={d['n']:>3d}  raw "
                  f"{d['raw_short_pct']:+.3f}% = "
                  f"{100*d['raw_short_pct']:+.1f} bps -> {d['cost_x']:>6.1f}x "
                  f"cost   (need >= 5x)")


def round2() -> None:
    """ROUND 2 -- close the two doors round 1 left open.

    GLD: the union split showed the short's whole content sitting in the
    22 episodes where GLD was already >10% below its 252d high, which is
    today's state. Is the analogue adding anything to that plain conditioner?
    XLE: the only honest-ish positive cell was B_daylevel (+0.464%), 57
    overlapping days inside 11 episodes. Test both for definition fragility.
    """
    px, breadth_names = load_all()
    f = build_features(px, breadth_names)
    idx = f.index
    sets, dist, pool = build_sets(px, f, idx)
    extra_px = load_prices([t for t in EXTRA if t not in px])
    px_all = {t: px[t]["Close"] for t in px}
    for t, d in extra_px.items():
        px_all[t] = d["Close"]
    for t in list(px_all):
        px_all[t] = px_all[t].reindex(idx.union(px_all[t].index)).ffill().reindex(idx)
    union_all = pd.DatetimeIndex(sorted(set().union(*[set(v) for v in sets.values()])))
    uni_epi = declusters(union_all, 21, idx)

    gld = px_all["GLD"]
    xle = px_all["XLE"]
    gdx = px_all["GDX"]
    gld_d252h = (gld / gld.rolling(252).max() - 1.0) * 100.0
    gdx_r21 = pct_rank(gdx, 21, 252)
    xle_r21 = pct_rank(xle, 21, 252)
    valid = fwd_lag(gld, 5, 1).dropna().index
    valid = valid[valid <= ASOF]

    print("\n\n" + "#" * 110)
    print("ROUND 2")
    print("#" * 110)

    # -----------------------------------------------------------------
    print("\n" + "=" * 110)
    print("S12. GLD -- does the ANALOGUE add anything to the plain drawdown cell?")
    print("=" * 110)
    print("Round 1 S4: on the 53-episode union the GLD short pays +0.360% in")
    print("the 22 episodes with GLD >10% below its high and -0.242% in the 31")
    print("that are not. If the plain drawdown conditioner pays the same on its")
    print("own, the neighbour construction is decoration.\n")
    deep_all = (gld_d252h <= -10.0).reindex(valid).fillna(False)
    deep_days = valid[deep_all.values]
    rows = []
    for lbl, dates in (
            ("ALL days GLD >10% below 252d high", deep_days),
            ("  same, 2018+", deep_days[deep_days >= "2018-01-01"]),
            ("  same, episodes @21td", declusters(deep_days, 21, idx)),
            ("  same, episodes @21td 2018+",
             declusters(deep_days[deep_days >= "2018-01-01"], 21, idx)),
            ("ANALOGUE union episodes, all", uni_epi),
            ("ANALOGUE union INTERSECT drawdown",
             pd.DatetimeIndex([d for d in uni_epi if d in set(deep_days)])),
            ("ANALOGUE union, NOT in drawdown",
             pd.DatetimeIndex([d for d in uni_epi if d not in set(deep_days)])),
            ("ALL drawdown days NOT in the analogue union",
             pd.DatetimeIndex([d for d in declusters(deep_days, 21, idx)
                               if d not in set(uni_epi)]))):
        d = short_stats(gld, dates, 5, GLD_COST_BPS)
        rows.append({"cell": lbl, **{k: d.get(k) for k in
                                     ("n", "raw_short_pct", "rec", "hit",
                                      "cost_x", "boot_p_le0")}})
    print(pd.DataFrame(rows).to_string(index=False))

    # -----------------------------------------------------------------
    print("\n" + "=" * 110)
    print("S13. GLD -- dose response on the drawdown axis, and the miners "
          "divergence the pitch actually describes")
    print("=" * 110)
    print("Registry 2026-08-10: 'distance-from-high is a U-shaped noise carve,")
    print("not a conditioner'. Test the gradient, episode level, h=5 short.\n")
    bands = [(-100, -20), (-20, -15), (-15, -10), (-10, -5), (-5, -2), (-2, 0.1)]
    for lo, hi in bands:
        m = ((gld_d252h > lo) & (gld_d252h <= hi)).reindex(valid).fillna(False)
        dts = declusters(valid[m.values], 21, idx)
        d = short_stats(gld, dts, 5, GLD_COST_BPS)
        tag = " <-- TODAY -17.97%" if lo == -20 and hi == -15 else ""
        print(f"  GLD d252h ({lo:>4.0f}, {hi:>4.1f}]  n={d.get('n', 0):>3d}  "
              f"short {d.get('raw_short_pct', float('nan')):+.3f}%  "
              f"rec {d.get('rec', '-')}{tag}")
    print("\nThe pitch's actual story is 'miners leading hard, the metal is not")
    print("confirming' (GDX +18.28% 21d, NEM r21 90.5, GLD 18% below its high).")
    print("That conjunction is testable directly and is NOT what either")
    print("neighbour method selected on:\n")
    for gdx_cut in (80, 90, 95):
        m = ((gdx_r21 >= gdx_cut) & (gld_d252h <= -10.0)).reindex(valid).fillna(False)
        dts = declusters(valid[m.values], 21, idx)
        d = short_stats(gld, dts, 5, GLD_COST_BPS)
        if not d.get("n"):
            print(f"  GDX r21>={gdx_cut} AND GLD >10% below high: n=0")
            continue
        print(f"  GDX r21>={gdx_cut} AND GLD >10% below high: n={d['n']:>3d} "
              f"episodes, short h=5 {d['raw_short_pct']:+.3f}%, rec {d['rec']}, "
              f"{d['cost_x']}x cost, bootstrap P(mean<=0) {d['boot_p_le0']}")
    print("  overlap of that conjunction with the analogue union episodes: "
          f"{len(set(valid[((gdx_r21 >= 90) & (gld_d252h <= -10.0)).reindex(valid).fillna(False).values]) & set(uni_epi))} dates")

    # -----------------------------------------------------------------
    print("\n" + "=" * 110)
    print("S14. DEFINITION FRAGILITY -- sign across the declustering gap")
    print("=" * 110)
    for tkr, h, cost, base_set in (("GLD", 5, GLD_COST_BPS, "B_daylevel"),
                                   ("XLE", 10, XLE_COST_BPS, "B_daylevel")):
        print(f"\n  SHORT {tkr} h={h}, Method B day-level regrouped at "
              f"different gaps (57 days, 11 contiguous runs):")
        raw = pd.DatetimeIndex(sets[base_set])
        for gap in (1, 5, 10, 21, 42, 63):
            dts = raw if gap == 1 else declusters(raw, gap, idx)
            d = short_stats(px_all[tkr], dts, h, cost)
            print(f"    gap {gap:>2d}td  n={d['n']:>3d}  "
                  f"{d['raw_short_pct']:+.3f}%  rec {d['rec']}  "
                  f"{d['cost_x']}x cost")
        print(f"  SHORT {tkr} h={h}, Method A k=50 greedy at different gaps:")
        pool2 = eligible(f, FEATURES)
        dist2 = knn_mod.knn(f, FEATURES, pool2, ASOF)
        for gap in (5, 10, 21, 42, 63):
            g = knn_mod.greedy_episodes(dist2, 50, idx, gap)
            d = short_stats(px_all[tkr], g, h, cost)
            print(f"    gap {gap:>2d}td  n={d['n']:>3d}  "
                  f"{d['raw_short_pct']:+.3f}%  rec {d['rec']}  "
                  f"{d['cost_x']}x cost")
        print(f"  SHORT {tkr} h={h}, Method A greedy at different k (gap 21):")
        for k in (10, 20, 30, 50, 75, 100):
            g = knn_mod.greedy_episodes(dist2, k, idx, 21)
            d = short_stats(px_all[tkr], g, h, cost)
            print(f"    k={k:>3d}  n={d['n']:>3d}  "
                  f"{d['raw_short_pct']:+.3f}%  rec {d['rec']}  "
                  f"{d['cost_x']}x cost")

    # -----------------------------------------------------------------
    print("\n" + "=" * 110)
    print("S15. LOCAL CONTROL -- is the cell just the local regime?")
    print("=" * 110)
    from pitch_lab import local_control
    for tkr, h, cost in (("GLD", 5, GLD_COST_BPS), ("XLE", 10, XLE_COST_BPS)):
        r = -fwd_lag(px_all[tkr], h, 1)
        vd = r.dropna().index
        vd = vd[vd <= ASOF]
        for label in ("A_k50_greedy", "B_daylevel"):
            trig = pd.DatetimeIndex(sets[label]).intersection(vd)
            loc = local_control(vd, trig, 126)
            a = r.reindex(trig).dropna()
            b = r.reindex(loc).dropna()
            print(f"  SHORT {tkr} h={h} {label:<13s} trigger "
                  f"{100*float(a.mean()):+.3f}% (n={len(a)})  vs  local "
                  f"+/-126td ex-trigger {100*float(b.mean()):+.3f}% "
                  f"(n={len(b)})  ->  local excess "
                  f"{100*float(a.mean()-b.mean()):+.3f}pp")

    # -----------------------------------------------------------------
    print("\n" + "=" * 110)
    print("S16. XLE -- the day-level positive is two contiguous runs")
    print("=" * 110)
    r = -fwd_lag(xle, 10, 1)
    v = r.reindex(pd.DatetimeIndex(sets["B_daylevel"])).dropna()
    by_run = v.groupby([d.to_period("M") for d in v.index]).agg(["count", "mean"])
    by_run["contrib_pp"] = 100 * by_run["count"] * by_run["mean"] / len(v)
    by_run["mean"] = (100 * by_run["mean"]).round(3)
    print(by_run.sort_values("contrib_pp").to_string())
    print(f"\n  total {100*float(v.mean()):+.3f}% over {len(v)} overlapping days "
          f"in {len(declusters(v.index, 21, idx))} episodes")
    top = by_run["contrib_pp"].abs().sort_values(ascending=False)
    print(f"  top-2 calendar months = "
          f"{by_run.loc[top.index[:2], 'contrib_pp'].sum():+.3f}pp of "
          f"{100*float(v.mean()):+.3f}pp "
          f"({100*by_run.loc[top.index[:2], 'contrib_pp'].sum()/(100*float(v.mean())):.0f}%)"
          f"  [{', '.join(str(p) for p in top.index[:2])}]")


if __name__ == "__main__":
    main()
    round2()
