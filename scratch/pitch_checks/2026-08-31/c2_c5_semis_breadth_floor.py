"""C5 -- equipment/analog semis at simultaneous 63d rank floors while NVDA runs.

Trigger: >=k of {AMAT, ADI, TXN, MU, QCOM, INTC} at 63d return rank <= 10
(trailing 252d) on the same session, AND NVDA 21d rank >= 85.
Trade: LONG the equal-weight laggard basket (outright) and, separately,
LONG basket / SHORT NVDA beta-matched. MOC next session (lag=1), h 1..10.

The burden set by the brief: this only lives if the MEMBER-LEVEL BREADTH
version does something the ETF-level version (watchlist 30 / 33, both dead)
does not, AND the NVDA leg earns its place.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

BASKET = ["AMAT", "ADI", "TXN", "MU", "QCOM", "INTC"]
LEADER = "NVDA"
H_MAIN = 10
FLOOR = 10.0
LEAD = 85.0
KMIN = 4


def build(tickers):
    px = load_prices(tickers)
    return px


def equal_weight_index(px, names):
    """Daily-rebalanced equal-weight price index over the names that have a
    bar that day AND the day before. Starts when all names have data."""
    rets = {}
    for t in names:
        c = px[t]["Close"].dropna()
        rets[t] = c / c.shift(1) - 1.0
    R = pd.DataFrame(rets)
    R = R.loc[R.notna().all(axis=1)]
    idx = (1.0 + R.mean(axis=1)).cumprod()
    return idx


def main():
    all_t = BASKET + [LEADER, "SMH", "SPY", "QQQ"]
    px_raw = build(all_t)
    basket = equal_weight_index(px_raw, BASKET)

    panel = pd.DataFrame({t: px_raw[t]["Close"] for t in all_t})
    panel["BASKET"] = basket
    panel = panel.loc[panel["BASKET"].notna()]
    panel = panel.dropna(how="all")

    print("=" * 78)
    print("C5  semis breadth floor + NVDA leader clause")
    print("=" * 78)
    print(f"basket span {panel.index[0].date()} .. {panel.index[-1].date()}  "
          f"({len(panel)} sessions)")

    r63 = pd.DataFrame({t: pct_rank(panel[t], 63) for t in BASKET})
    nfloor = (r63 <= FLOOR).sum(axis=1)
    lead21 = pct_rank(panel[LEADER], 21)
    smh63 = pct_rank(panel["SMH"], 63)

    last = panel.index[-1]
    print(f"\nlive read {last.date()}: "
          + ", ".join(f"{t} r63 {r63[t].iloc[-1]:.1f}" for t in BASKET))
    print(f"  members at floor = {int(nfloor.iloc[-1])}, NVDA r21 = "
          f"{lead21.iloc[-1]:.1f}, SMH r63 = {smh63.iloc[-1]:.1f}")

    ret_b = fwd_lag(panel["BASKET"], H_MAIN, lag=1)
    valid = ret_b.notna()
    base_b = 100 * ret_b[valid].mean()

    # ---------------- 1. BREADTH GATE ATTRIBUTION ---------------------------
    print("\n" + "-" * 78)
    print(f"1. BREADTH GATE ATTRIBUTION (outright basket, h={H_MAIN}, episodes)")
    print("-" * 78)
    rows = []
    for k in range(1, 7):
        m = (nfloor >= k) & (lead21 >= LEAD)
        d = panel.index[m.values & valid.values]
        if len(d) < 3:
            rows.append({"label": f"k>={k} + NVDA", "n": len(d)})
            continue
        epi = declusters(d, H_MAIN, panel.index)
        s = summarize(ret_b.loc[epi].values, f"k>={k} + NVDA")
        s["n_days"] = len(d)
        s["excess_pp"] = s["mean_pct"] - base_b
        rows.append(s)
    rows.append({**summarize(ret_b[valid].values, "CTRL basket all days"),
                 "n_days": int(valid.sum()), "excess_pp": 0.0})
    show(rows, "breadth ladder (NVDA clause ON) -- monotone?")

    # ---------------- 2. NVDA CLAUSE ATTRIBUTION ----------------------------
    print("\n" + "-" * 78)
    print("2. NVDA CLAUSE ATTRIBUTION -- with and without")
    print("-" * 78)
    rows = []
    for lbl, m in [
        (f"breadth k>={KMIN} ALONE (no NVDA)", (nfloor >= KMIN)),
        (f"NVDA r21>={LEAD} ALONE (no breadth)", (lead21 >= LEAD)),
        (f"JOINT k>={KMIN} & NVDA r21>={LEAD}", (nfloor >= KMIN) & (lead21 >= LEAD)),
        (f"k>={KMIN} & NVDA r21<{LEAD} (anti)", (nfloor >= KMIN) & (lead21 < LEAD)),
    ]:
        d = panel.index[m.values & valid.values]
        epi = declusters(d, H_MAIN, panel.index)
        s = summarize(ret_b.loc[epi].values, lbl)
        s["n_days"] = len(d)
        s["excess_pp"] = s.get("mean_pct", np.nan) - base_b
        rows.append(s)
    rows.append({**summarize(ret_b[valid].values, "CTRL basket all days"),
                 "n_days": int(valid.sum()), "excess_pp": 0.0})
    show(rows, "NVDA clause attribution (outright basket)")
    jb = rows[2]["mean_pct"]; bb = rows[0]["mean_pct"]
    print(f"\n  NVDA clause is worth {jb-bb:+.3f}pp over bare breadth "
          f"(threshold for 'not decoration' is ~+0.2pp)")

    mask = (nfloor >= KMIN) & (lead21 >= LEAD)

    # ---------------- 3. LEG ATTRIBUTION ON THE PAIR ------------------------
    print("\n" + "-" * 78)
    print("3. LEG ATTRIBUTION -- what does each side of the pair contribute?")
    print("-" * 78)
    rb = panel["BASKET"] / panel["BASKET"].shift(1) - 1.0
    rn = panel[LEADER] / panel[LEADER].shift(1) - 1.0
    ok = rb.notna() & rn.notna()
    beta = float(np.polyfit(rn[ok].values, rb[ok].values, 1)[0])
    print(f"  full-sample beta of basket on NVDA daily returns = {beta:.3f}")

    ret_n = fwd_lag(panel[LEADER], H_MAIN, lag=1)
    d = panel.index[mask.values & valid.values & ret_n.notna().values]
    epi = declusters(d, H_MAIN, panel.index)
    legs_rows = [
        summarize(ret_b.loc[epi].values, "LONG basket only"),
        summarize(-ret_n.loc[epi].values, "SHORT NVDA only"),
        summarize((ret_b - ret_n).loc[epi].values, "pair, equal dollar"),
        summarize((ret_b - beta * ret_n).loc[epi].values,
                  f"pair, beta-matched ({beta:.2f})"),
    ]
    show(legs_rows, f"leg attribution (episodes N={len(epi)}, h={H_MAIN})")
    print(f"\n  SHORT NVDA leg contributes {legs_rows[1]['mean_pct']:+.3f}pp; "
          f"pair {legs_rows[2]['mean_pct']:+.3f}% vs outright "
          f"{legs_rows[0]['mean_pct']:+.3f}%")

    # ---------------- 4. FULL BATTERY, OUTRIGHT -----------------------------
    print("\n" + "-" * 78)
    print("4. FULL BATTERY -- outright basket")
    print("-" * 78)
    print("  COST ARITHMETIC, stated explicitly: six legs each at 1/6 weight,")
    print("  10 bps single-name round trip each => 10 bps of PORTFOLIO notional,")
    print("  not 60. The pair adds a second unit of notional => 20 bps.")
    variants = {
        "floor <= 5": (( pd.DataFrame({t: pct_rank(panel[t], 63) for t in BASKET}) <= 5).sum(axis=1) >= KMIN) & (lead21 >= LEAD),
        "floor <= 20": ((r63 <= 20).sum(axis=1) >= KMIN) & (lead21 >= LEAD),
        "NVDA r21>=75": mask.copy() if False else ((nfloor >= KMIN) & (lead21 >= 75)),
        "NVDA r21>=95": (nfloor >= KMIN) & (lead21 >= 95),
        "k>=3": (nfloor >= 3) & (lead21 >= LEAD),
        "k>=5": (nfloor >= 5) & (lead21 >= LEAD),
        "no NVDA clause": (nfloor >= KMIN),
    }
    battery(panel, mask, [("BASKET", 1.0)], H_MAIN,
            f"C5 outright: >={KMIN}/6 semis at r63<=10 & NVDA r21>=85  LONG basket",
            cost_bps=10.0, variants=variants, min_gap=H_MAIN)

    print("\n" + "-" * 78)
    print("4b. FULL BATTERY -- the pair (beta-matched)")
    print("-" * 78)
    battery(panel, mask, [("BASKET", 1.0), (LEADER, -beta)], H_MAIN,
            f"C5 pair: LONG basket / SHORT {beta:.2f}x NVDA",
            cost_bps=10.0, variants=None, min_gap=H_MAIN)

    # ---------------- 5. SMH EXPRESSION COLLISION ---------------------------
    print("\n" + "-" * 78)
    print("5. IS THE TRADEABLE EXPRESSION JUST SMH? (watchlist 30 collision)")
    print("-" * 78)
    ret_s = fwd_lag(panel["SMH"], H_MAIN, lag=1)
    vs = ret_s.notna()
    ds = panel.index[mask.values & vs.values]
    es = declusters(ds, H_MAIN, panel.index)
    show([summarize(ret_b.loc[epi].values, "basket vehicle"),
          summarize(ret_s.loc[es].values, "SMH vehicle, SAME trigger"),
          summarize(ret_s[vs].values, "SMH all days")],
         "vehicle swap")
    corr = pd.concat([rb, panel["SMH"] / panel["SMH"].shift(1) - 1.0],
                     axis=1).dropna().corr().iloc[0, 1]
    print(f"  daily-return correlation basket vs SMH = {corr:.3f}")
    print(f"  trigger days where SMH r63 <= 10 too: "
          f"{int((smh63.loc[panel.index[mask.values]] <= 10).sum())} of "
          f"{int(mask.sum())}")

    # ---------------- 6. ERA / MIDTERM / DIAL -------------------------------
    print("\n" + "-" * 78)
    print("6. ERA, MIDTERM, CONCENTRATION, DECLUSTER, DIAL")
    print("-" * 78)
    ev = ret_b.loc[epi].values
    mid = np.array([d.year % 4 == 2 for d in epi])
    show([summarize(ev[mid], f"midterm (N={int(mid.sum())})"),
          summarize(ev[~mid], f"non-midterm (N={int((~mid).sum())})")],
         "midterm split, outright")
    print("\n  decluster ladder (outright):")
    dd = panel.index[mask.values & valid.values]
    for g in (5, 10, 21):
        e2 = declusters(dd, g, panel.index)
        s = summarize(ret_b.loc[e2].values, "")
        wn = int((ret_b.loc[e2].values > 0).sum())
        print(f"   min_gap {g:2d}: N={s['n']:3d} mean {s['mean_pct']:+.3f}% "
              f"hit {s['hit']:.1f}% sign p {sign_test(wn, s['n']):.4f}")

    print("\n  horizon scan h=1..10, outright:")
    show(horizon_scan(panel, dd, [("BASKET", 1.0)], hs=tuple(range(1, 11))),
         "outright horizon scan")
    print("\n  horizon scan h=1..10, pair:")
    show(horizon_scan(panel, dd, [("BASKET", 1.0), (LEADER, -beta)],
                      hs=tuple(range(1, 11))), "pair horizon scan")

    frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
    ma = frag["63d"].rolling(10).mean().reindex(panel.index).ffill()
    tv = ma.loc[epi].dropna()
    print(f"\n  DIAL of trigger population (today 87.6): N={len(tv)} of "
          f"{len(epi)} episodes; min {tv.min() if len(tv) else np.nan:.1f} "
          f"median {tv.median() if len(tv) else np.nan:.1f} "
          f"MAX {tv.max() if len(tv) else np.nan:.1f}; "
          f">=85: {int((tv>=85).sum()) if len(tv) else 0}")
    print("  episode dates:", ", ".join(str(d.date()) for d in epi))


if __name__ == "__main__":
    main()
