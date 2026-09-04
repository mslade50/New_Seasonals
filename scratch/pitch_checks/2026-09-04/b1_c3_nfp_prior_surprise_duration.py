"""C3 blocker: post-NFP duration conditioned on the PRIOR print's surprise.

Thesis under test: after a large downside payroll miss the policy path is
re-priced and the term premium carries INTO the following month's print, so
the session after the NEXT print behaves differently in duration.

Live conditioner today (2026-09-04): the prior print was 2026-08-07, actual
-23k against consensus +80k, surprise -103k.

CONVENTION. The pitch wants entry at TODAY's close, which is the print
session's close. In lab terms the SIGNAL is the session BEFORE the print
(the conditioner, the prior month's surprise, is known then) and lag=1 puts
the order in MOC on the print session. That is a real lag=1 construction,
not a lag=0 dressed up.

Blockers run: staleness, joint-state count BEFORE design, gate attribution
with the discarded complement, midterm 2x2, placebo anchor ladder k=-8..+8,
definition fragility on the surprise cut, decluster + era split + local
control + cost (all inside battery()).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

REL = Path(__file__).resolve().parents[3] / "data" / "macro_release_history.parquet"
COST = {"TLT": 3.0, "IEF": 2.0}
TODAY = pd.Timestamp("2026-09-04")
LIVE_PRIOR_SURPRISE = -103.0


def nfp_table() -> pd.DataFrame:
    df = pd.read_parquet(REL)
    n = df[df["event_name"] == "Non Farm Payrolls"].copy()
    n = n.dropna(subset=["surprise"]).sort_values("release_date")
    n = n.drop_duplicates(subset=["release_date"], keep="last")
    n = n[["release_date", "actual", "consensus", "surprise"]].reset_index(drop=True)
    n["prior_surprise"] = n["surprise"].shift(1)
    n["prior_date"] = n["release_date"].shift(1)
    n["midterm"] = (n["release_date"].dt.year % 4 == 2)
    return n


def anchor_dates(idx: pd.DatetimeIndex, prints: pd.DatetimeIndex,
                 k: int = 0) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Signal dates = (print position - 1 + k). lag=1 then enters at the
    print session's close when k=0. Returns (dates, keep-mask into prints)."""
    pos, kept = anchor_positions(idx, prints, offset=-1 + k)
    keep = np.isin(np.asarray(prints), np.asarray(kept))
    return pd.DatetimeIndex(idx[pos]), keep


def cell(px, sig_dates, legs, h, lag=1):
    ret = vehicle_ret(px, legs, h, lag)
    v = ret.reindex(sig_dates).dropna()
    return v


def main() -> None:
    print("=" * 78)
    print("C3  post-NFP duration | conditioner = PRIOR print's surprise")
    print("=" * 78)

    n = nfp_table()

    # ---------------- 0. STALENESS ----------------
    print("\n--- 0. STALENESS OF THE LIVE CONDITIONER ---")
    df_all = pd.read_parquet(REL)
    print(f"macro_release_history newest row overall : "
          f"{df_all['release_date'].max().date()}")
    print(f"newest NFP print with a surprise         : "
          f"{n['release_date'].max().date()}  "
          f"actual {n['actual'].iloc[-1]:.0f} vs cons {n['consensus'].iloc[-1]:.0f} "
          f"surprise {n['surprise'].iloc[-1]:+.0f}")
    print(f"today                                    : {TODAY.date()}")
    print("VERDICT: the conditioner for TODAY is the PRIOR print, 2026-08-07,")
    print("         which IS in the file. The file being 4 weeks stale does NOT")
    print("         block C3 -- NFP is monthly and the last one predates the")
    print("         cutoff. C3 passes the staleness blocker. Live value "
          f"{LIVE_PRIOR_SURPRISE:+.0f}k.")

    # ---------------- 1. JOINT STATE COUNT, BEFORE ANY EDGE ----------------
    print("\n--- 1. JOINT-STATE OCCURRENCE COUNT (before looking at any edge) ---")
    have = n.dropna(subset=["prior_surprise"])
    print(f"prints with a prior surprise available   : {len(have)}  "
          f"({have['release_date'].min().date()} .. "
          f"{have['release_date'].max().date()})")
    for thr in (-25, -40, -50, -60, -75, -100, -103):
        sub = have[have["prior_surprise"] <= thr]
        mt = sub[sub["midterm"]]
        print(f"  prior surprise <= {thr:>5}k : N={len(sub):>3}   "
              f"of which midterm N={len(mt):>2}  "
              f"years {sorted(set(sub['release_date'].dt.year))}"
              if len(sub) <= 40 else
              f"  prior surprise <= {thr:>5}k : N={len(sub):>3}   midterm N={len(mt):>2}")
    print("\n  midterm x large-miss joint cell, the LIVE state:")
    for thr in (-50, -75, -100):
        sub = have[(have["prior_surprise"] <= thr) & have["midterm"]]
        print(f"    <= {thr}k AND midterm : N={len(sub)}  "
              f"{[str(d.date()) for d in sub['release_date']]}")

    px = close_panel(["TLT", "IEF", "SPY"]).dropna()
    prints = pd.DatetimeIndex(have["release_date"])

    sig, keep = anchor_dates(px.index, prints, k=0)
    sub = have[keep].reset_index(drop=True)
    sub["signal"] = sig
    print(f"\n  anchors that survive the price index      : {len(sub)} of {len(have)}")

    THR = -50.0
    gated = pd.DatetimeIndex(sub.loc[sub["prior_surprise"] <= THR, "signal"])
    compl = pd.DatetimeIndex(sub.loc[sub["prior_surprise"] > THR, "signal"])
    print(f"  gated cell at <= {THR:.0f}k : N={len(gated)}   complement N={len(compl)}")

    # ---------------- 2. GATE ATTRIBUTION ----------------
    print("\n--- 2. GATE ATTRIBUTION: parent / gated / DISCARDED complement ---")
    print("    (mean % of the h-session hold entering MOC on the print session)")
    hdr = f"{'vehicle':>10} {'h':>2} | {'PARENT all prints':>22} | " \
          f"{'GATED <=-50k':>20} | {'COMPLEMENT >-50k':>20} | {'gate lift':>9}"
    print(hdr)
    print("-" * len(hdr))
    allsig = pd.DatetimeIndex(sub["signal"])
    best = []
    for tkr in ("TLT", "IEF"):
        for h in (1, 2, 3, 5):
            p = cell(px, allsig, [(tkr, 1.0)], h)
            g = cell(px, gated, [(tkr, 1.0)], h)
            c = cell(px, compl, [(tkr, 1.0)], h)
            lift = 100 * (g.mean() - p.mean())
            print(f"{tkr:>10} {h:>2} | {100*p.mean():>+8.3f}% n={len(p):<3} "
                  f"hit {100*(p>0).mean():>4.1f}% | "
                  f"{100*g.mean():>+8.3f}% n={len(g):<3} hit {100*(g>0).mean():>4.1f}% | "
                  f"{100*c.mean():>+8.3f}% n={len(c):<3} hit {100*(c>0).mean():>4.1f}% | "
                  f"{lift:>+8.3f}pp")
            best.append((abs(lift), tkr, h, 100 * g.mean(), len(g)))
    print("\n  NOTE the search size: 2 vehicles x 4 horizons x 2 directions = 16 cells.")
    print("  A direction is not specified by the thesis ('should behave")
    print("  differently'), which is itself a mechanism weakness.")

    # ---------------- 3. MIDTERM 2x2 ----------------
    print("\n--- 3. MIDTERM x SURPRISE 2x2 (the documented killer) ---")
    print("    registry: post-NFP rates is midterm-DEAD, +0.071% N=12 hit 58%")
    for tkr in ("TLT", "IEF"):
        for h in (3,):
            print(f"\n  {tkr}  h={h}")
            for mt in (False, True):
                for lo, hi, lbl in ((-1e9, THR, "surprise<=-50k"),
                                    (THR, 1e9, "surprise >-50k")):
                    m = ((sub["prior_surprise"] > lo) & (sub["prior_surprise"] <= hi)
                         & (sub["midterm"] == mt))
                    d = pd.DatetimeIndex(sub.loc[m, "signal"])
                    v = cell(px, d, [(tkr, 1.0)], h)
                    tag = "MIDTERM " if mt else "non-mid "
                    if len(v):
                        print(f"    {tag}{lbl}: N={len(v):>3}  "
                              f"mean {100*v.mean():>+7.3f}%  hit {100*(v>0).mean():>5.1f}%  "
                              f"sign p={sign_test(int((v>0).sum()), len(v)):.4f}")
                    else:
                        print(f"    {tag}{lbl}: N=0")

    # ---------------- 4. PLACEBO ANCHOR LADDER ----------------
    print("\n--- 4. PLACEBO ANCHOR LADDER k=-8..+8 (five-for-five in this repo) ---")
    print("    k=0 is the live rung: enter MOC on the print session.")
    for tkr in ("TLT", "IEF"):
        for h in (3,):
            rows = []
            for k in range(-8, 9):
                s_k, keep_k = anchor_dates(px.index, prints, k=k)
                hk = have[keep_k].reset_index(drop=True)
                g_k = pd.DatetimeIndex(s_k[(hk["prior_surprise"] <= THR).values])
                v = cell(px, g_k, [(tkr, 1.0)], h)
                rows.append((k, 100 * v.mean() if len(v) else np.nan, len(v)))
            order = sorted([r for r in rows if not np.isnan(r[1])],
                           key=lambda r: -r[1])
            rank = [r[0] for r in order].index(0) + 1
            print(f"\n  {tkr} h={h} gated<= {THR:.0f}k   LIVE k=0 ranks "
                  f"{rank} of {len(order)} (long side)")
            rank_s = [r[0] for r in sorted(order, key=lambda r: r[1])].index(0) + 1
            print(f"  {tkr} h={h}                        LIVE k=0 ranks "
                  f"{rank_s} of {len(order)} (short side)")
            print("   " + "  ".join(f"k={k:+d}:{m:+.2f}" for k, m, _ in rows))

    # ---------------- 5. DEFINITION FRAGILITY ----------------
    print("\n--- 5. DEFINITION FRAGILITY: nudge the surprise cut ---")
    for tkr in ("TLT", "IEF"):
        print(f"\n  {tkr}, h=3, long, episode-level (prints are ~21td apart)")
        for thr in (-25, -40, -50, -60, -75, -100, -120):
            d = pd.DatetimeIndex(sub.loc[sub["prior_surprise"] <= thr, "signal"])
            v = cell(px, d, [(tkr, 1.0)], 3)
            if len(v) < 2:
                print(f"    <= {thr:>5}k : N={len(v)}  (too few to state)")
                continue
            print(f"    <= {thr:>5}k : N={len(v):>3}  mean {100*v.mean():>+7.3f}%  "
                  f"med {100*v.median():>+7.3f}%  hit {100*(v>0).mean():>5.1f}%  "
                  f"sign p={sign_test(int((v>0).sum()), len(v)):.4f}")
        print(f"    percentile form (prior surprise in bottom X of its own history):")
        pr = sub["prior_surprise"]
        for q in (0.10, 0.15, 0.20, 0.25, 0.33):
            cut = pr.quantile(q)
            d = pd.DatetimeIndex(sub.loc[pr <= cut, "signal"])
            v = cell(px, d, [(tkr, 1.0)], 3)
            print(f"      q{q:.2f} (cut {cut:>7.1f}k): N={len(v):>3}  "
                  f"mean {100*v.mean():>+7.3f}%  hit {100*(v>0).mean():>5.1f}%")

    # ---------------- 6. FULL BATTERY on the least-bad cell ----------------
    print("\n--- 6. FULL BATTERY (decluster / controls / era / cost) ---")
    for tkr, side in (("TLT", 1.0), ("IEF", 1.0)):
        mask = pd.Series(False, index=px.index)
        mask.loc[gated] = True
        battery(px, mask, [(tkr, side)], h=3,
                title=f"C3 long {tkr} MOC on the print, prior surprise <= {THR:.0f}k",
                cost_bps=COST[tkr], min_gap=5, event_kinds=("cpi",))

    # ---------------- 7. THE HONEST ALTERNATIVE ----------------
    print("\n--- 7. THE HONEST ALTERNATIVE: condition on TODAY'S surprise ---")
    print("    (NOT knowable at 05:00 when the pitch is composed; measured only")
    print("     to see whether the whole effect lives in the unknowable version)")
    for tkr in ("TLT", "IEF"):
        for thr in (-50,):
            d_own = pd.DatetimeIndex(sub.loc[sub["surprise"] <= thr, "signal"])
            v = cell(px, d_own, [(tkr, 1.0)], 3)
            d_own_hi = pd.DatetimeIndex(sub.loc[sub["surprise"] > thr, "signal"])
            vh = cell(px, d_own_hi, [(tkr, 1.0)], 3)
            print(f"  {tkr} h=3 TODAY'S surprise <= {thr}k : N={len(v):>3} "
                  f"mean {100*v.mean():>+7.3f}% hit {100*(v>0).mean():>5.1f}%   "
                  f"| complement N={len(vh):>3} mean {100*vh.mean():>+7.3f}%")

    # ---------------- 8. ERA SPLIT on the gated cell ----------------
    print("\n--- 8. ERA SPLIT + EPISODE YEAR HISTOGRAM ---")
    yrs = pd.DatetimeIndex(gated).year
    print("  gated-cell episodes by year:",
          dict(pd.Series(yrs).value_counts().sort_index()))
    for tkr in ("TLT", "IEF"):
        v = cell(px, gated, [(tkr, 1.0)], 3)
        show(era_split(v.index, v.values), f"  {tkr} h=3 long, era split")

    # ---------------- 9. COST ----------------
    print("\n--- 9. COST ---")
    for tkr in ("TLT", "IEF"):
        v = cell(px, gated, [(tkr, 1.0)], 3)
        bps = 100 * v.mean() * 100
        print(f"  {tkr}: edge {bps:+.1f} bps vs {COST[tkr]:.0f} bps round trip "
              f"= {bps/COST[tkr]:+.1f}x  (need >= 5x)")


if __name__ == "__main__":
    main()
