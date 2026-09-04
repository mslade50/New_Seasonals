"""K7 — Nearest-neighbour historical analogue to today's tape.

ADVERSARIAL CHECK. Feature vector per date (SPY, 2000-2026):
  5d return, 21d return, distance below trailing 252d high, 21d realised vol,
  VIX level, VIX 5d change, breadth = fraction of the sector-ETF proxy above
  its own 200d SMA.
Standardise over the full sample, take the k nearest neighbours to the
2026-08-05 vector by Euclidean distance (excluding the last 21 sessions to
avoid self-matching), report forward 5 / 10 session SPY returns vs the
unconditional baseline.

Breadth caveat baked in: XLC and XLRE only start 2018 / 2015, so early breadth
is computed over fewer members. That is printed per era.

All data truncated to bars strictly before 2026-08-06.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _common as C

SECTORS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
VIX_CANDIDATES = ["^VIX", "VIX"]
EXCLUDE_LAST_TD = 21
KS = [10, 25, 50]
HORIZONS = [5, 10]
FEATURES = ["r5", "r21", "dist_hi", "rv21", "vix", "vix_chg5", "breadth"]


def episodes(dates: pd.DatetimeIndex, gap_td: int = 21) -> list[list[pd.Timestamp]]:
    if len(dates) == 0:
        return []
    dates = pd.DatetimeIndex(sorted(dates))
    eps, cur = [], [dates[0]]
    for prev, d in zip(dates[:-1], dates[1:]):
        if (d - prev).days > gap_td * 1.6:
            eps.append(cur)
            cur = [d]
        else:
            cur.append(d)
    eps.append(cur)
    return eps


def main() -> None:
    pd.set_option("display.width", 220)

    px = C.load(["SPY"] + SECTORS + VIX_CANDIDATES)
    spy = px["SPY"]
    c = spy["Close"]

    vix_key = next((k for k in VIX_CANDIDATES if k in px), None)
    if vix_key is None:
        raise SystemExit("no VIX series in master_prices — cannot build the feature vector")
    vix = px[vix_key]["Close"].reindex(c.index).ffill(limit=3)
    print(f"  VIX series used: {vix_key}  ({vix.dropna().index.min().date()} .. "
          f"{vix.dropna().index.max().date()})")

    # ---- breadth proxy ----
    have = [s for s in SECTORS if s in px]
    print(f"  sector proxy members available: {len(have)} -> {', '.join(have)}")
    above, present = None, None
    for s in have:
        cs = px[s]["Close"].reindex(c.index)
        sma = cs.rolling(200, min_periods=200).mean()
        a = (cs > sma).astype(float).where(sma.notna())
        above = a if above is None else above.add(a, fill_value=0.0)
        p = sma.notna().astype(float)
        present = p if present is None else present.add(p, fill_value=0.0)
    breadth = (above / present.replace(0, np.nan)) * 100.0
    for s in have:
        cs = px[s]["Close"].reindex(c.index).dropna()
        print(f"    {s}: first bar {cs.index.min().date()}")

    feat = pd.DataFrame({
        "r5": C.ret(c, 5),
        "r21": C.ret(c, 21),
        "dist_hi": (c / c.rolling(252, min_periods=126).max() - 1.0) * 100.0,
        "rv21": c.pct_change().rolling(21).std() * np.sqrt(252) * 100.0,
        "vix": vix,
        "vix_chg5": (vix / vix.shift(5) - 1.0) * 100.0,
        "breadth": breadth,
    })
    feat["n_breadth_members"] = present
    for h in HORIZONS + [3, 21]:
        feat[f"f{h}"] = C.fwd(c, h)
        feat[f"o{h}"] = C.fwd_from_next_open(spy, h)

    tab = feat.dropna(subset=FEATURES).copy()
    tab = tab[tab.index >= "2000-01-01"]
    print(f"\n  feature panel: {tab.index.min().date()} .. {tab.index.max().date()}  n={len(tab)}")
    print("  breadth member count by era:")
    print(tab.groupby(tab.index.year)["n_breadth_members"].agg(["min", "max"]).T.to_string())

    today = tab.index.max()
    tv = tab.loc[today, FEATURES].astype(float)
    print(f"\n  TODAY vector ({today.date()}):")
    print("   " + "  ".join(f"{k}={tv[k]:.2f}" for k in FEATURES))

    # ---- standardise over the FULL sample (as specified) ----
    mu, sd = tab[FEATURES].mean(), tab[FEATURES].std(ddof=0)
    Z = (tab[FEATURES] - mu) / sd
    zt = (tv - mu) / sd
    print("\n  z-scored today vector (vs 2000-2026 full-sample moments):")
    print("   " + "  ".join(f"{k}={zt[k]:+.2f}" for k in FEATURES))

    cutoff = tab.index[-(EXCLUDE_LAST_TD + 1)]
    pool = Z[Z.index <= cutoff]
    print(f"\n  self-match exclusion: neighbours must be <= {cutoff.date()} "
          f"(last {EXCLUDE_LAST_TD} sessions dropped); pool n={len(pool)}")

    dist = np.sqrt(((pool - zt) ** 2).sum(axis=1)).sort_values()

    base = {h: tab[f"f{h}"].dropna() for h in HORIZONS}
    base_o = {h: tab[f"o{h}"].dropna() for h in HORIZONS}

    for k in KS:
        nn = dist.index[:k]
        print("\n" + "=" * 78)
        print(f"k = {k} NEAREST NEIGHBOURS")
        print("=" * 78)
        eps = episodes(nn, gap_td=21)
        print(f"  distance range {dist.iloc[0]:.3f} .. {dist.iloc[k-1]:.3f}")
        print(f"  {k} neighbour days -> {len(eps)} distinct episodes (>21-session gap)")
        for i, e in enumerate(eps, 1):
            sub = tab.loc[e]
            print(f"   ep{i:2d}  {e[0].date()} .. {e[-1].date()}  n={len(e):2d}  "
                  f"f5 {sub['f5'].mean():+.2f}  f10 {sub['f10'].mean():+.2f}")
        rows = []
        for h in HORIZONS:
            rows.append({"h": h, **C.describe(f"NN{k} all days", tab.loc[nn, f"f{h}"], base[h])})
            rows.append({"h": h, **C.describe(f"NN{k} MOO-next", tab.loc[nn, f"o{h}"], base_o[h])})
            first = pd.DatetimeIndex([e[0] for e in eps])
            rows.append({"h": h, **C.describe(f"NN{k} episode-first", tab.loc[first, f"f{h}"], base[h])})
            epm = [tab.loc[e, f"f{h}"].mean() for e in eps]
            rows.append({"h": h, **C.describe(f"NN{k} episode-mean", epm, base[h])})
            rows.append({"h": h, **C.describe("BASELINE uncond", base[h])})
        C.show(rows)

    # ---- the 25-neighbour dates, printed in full ----
    k = 25
    nn = dist.index[:k]
    print("\n" + "=" * 78)
    print("THE 25 NEIGHBOUR DATES (with distance and outcomes)")
    print("=" * 78)
    det = pd.DataFrame({
        "dist": dist.iloc[:k].round(3),
        **{f: tab.loc[nn, f].round(2) for f in FEATURES},
        "f5": tab.loc[nn, "f5"].round(2),
        "f10": tab.loc[nn, "f10"].round(2),
        "f21": tab.loc[nn, "f21"].round(2),
    })
    det.index = det.index.date
    print(det.to_string())
    print(f"\n  neighbour year histogram: {dict(pd.Series(nn.year).value_counts().sort_index())}")

    # ---- sign stability in k ----
    print("\n" + "=" * 78)
    print("SIGN STABILITY IN k")
    print("=" * 78)
    rows = []
    for k in [5, 10, 15, 25, 40, 50, 75, 100, 200]:
        nn = dist.index[:k]
        e = episodes(nn, gap_td=21)
        r = {"k": k, "n_eps": len(e), "max_dist": round(float(dist.iloc[k - 1]), 3)}
        for h in HORIZONS:
            x = tab.loc[nn, f"f{h}"].dropna()
            r[f"f{h}_avg"] = round(float(x.mean()), 3)
            r[f"f{h}_med"] = round(float(x.median()), 3)
            r[f"f{h}_hit"] = round(float((x > 0).mean() * 100), 1)
            r[f"f{h}_t"] = round(C.tstat(x), 2)
        rows.append(r)
    C.show(rows)
    print("\n  unconditional baseline for reference:")
    C.show([C.describe(f"uncond f{h}", base[h]) for h in HORIZONS])

    # ---- feature-drop sensitivity: is the match driven by one feature? ----
    print("\n" + "=" * 78)
    print("FEATURE-DROP SENSITIVITY (k=25, drop one feature at a time)")
    print("=" * 78)
    rows = []
    for drop in [None] + FEATURES:
        cols = [f for f in FEATURES if f != drop]
        d2 = np.sqrt(((pool[cols] - zt[cols]) ** 2).sum(axis=1)).sort_values()
        nn2 = d2.index[:25]
        e = episodes(nn2, gap_td=21)
        r = {"dropped": drop or "(none)", "n_eps": len(e),
             "overlap_w_full": len(set(nn2) & set(dist.index[:25]))}
        for h in HORIZONS:
            x = tab.loc[nn2, f"f{h}"].dropna()
            r[f"f{h}_avg"] = round(float(x.mean()), 3)
            r[f"f{h}_t"] = round(C.tstat(x), 2)
        rows.append(r)
    C.show(rows)

    # ---- era check on the neighbour set ----
    print("\n" + "=" * 78)
    print("ERA CHECK — restrict the neighbour POOL to pre-2015 / 2015+ separately")
    print("=" * 78)
    rows = []
    for lo, hi, tag in [("2000-01-01", "2015-01-01", "pool pre-2015"),
                        ("2015-01-01", "2027-01-01", "pool 2015+")]:
        p2 = pool[(pool.index >= lo) & (pool.index < hi)]
        d2 = np.sqrt(((p2 - zt) ** 2).sum(axis=1)).sort_values()
        nn2 = d2.index[:25]
        e = episodes(nn2, gap_td=21)
        r = {"cohort": tag, "n_eps": len(e), "max_dist": round(float(d2.iloc[24]), 3)}
        for h in HORIZONS:
            x = tab.loc[nn2, f"f{h}"].dropna()
            r[f"f{h}_avg"] = round(float(x.mean()), 3)
            r[f"f{h}_t"] = round(C.tstat(x), 2)
            r[f"f{h}_worst"] = round(float(x.min()), 2)
        rows.append(r)
    C.show(rows)


if __name__ == "__main__":
    main()
