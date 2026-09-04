"""C4 blocker: CFTC speculative net positioning in the metals complex as a
conditioner on forward GLD / SLV / GDX.

Thesis under test: extreme speculative length is a forced-unwind setup, so
the percentile of COT spec net positions should condition forward metals
returns at h=1..10.

KNOWABILITY. The COT report covers TUESDAY's positions and is released the
following FRIDAY at 15:30 ET. Two separate lags therefore exist and both are
reported here:
  (a) report-date -> release-date, 3 calendar days, unavoidable and part of
      the instrument;
  (b) release-date -> first tradeable close. The release lands 30 minutes
      before the bell, so lag=1 (the NEXT close) is used throughout. Nothing
      here is measured at the release-day close.

Blockers: staleness priced in sessions AND in percentile drift, live-trigger
existence, joint-state count, gate attribution with the discarded
complement, placebo anchor ladder, definition fragility on the percentile
cut, decluster, era split inside a single macro regime, cost.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

REL = Path(__file__).resolve().parents[3] / "data" / "macro_release_history.parquet"
TODAY = pd.Timestamp("2026-09-04")
COST = {"GLD": 3.0, "SLV": 4.0, "GDX": 4.0}
SERIES = {"gold": "CFTC Gold Speculative net positions",
          "silver": "CFTC Silver Speculative net positions",
          "copper": "CFTC Copper Speculative net positions"}
VEH = {"gold": "GLD", "silver": "SLV", "copper": "GLD"}


def cot(name: str) -> pd.DataFrame:
    df = pd.read_parquet(REL)
    c = df[df["event_name"] == SERIES[name]].copy()
    c = c.dropna(subset=["actual"]).sort_values("release_date")
    c = c.drop_duplicates(subset=["release_date"], keep="last")
    c = c[["release_date", "actual"]].reset_index(drop=True)
    # percentile of the reading within its own trailing history
    c["pct104"] = c["actual"].rolling(104, min_periods=52).apply(
        lambda w: 100.0 * (w.iloc[-1] >= w).mean(), raw=False)
    c["pct52"] = c["actual"].rolling(52, min_periods=26).apply(
        lambda w: 100.0 * (w.iloc[-1] >= w).mean(), raw=False)
    return c


def main() -> None:
    print("=" * 78)
    print("C4  CFTC metals speculative net positioning as a live conditioner")
    print("=" * 78)

    # ---------------- 0. STALENESS, PRICED ----------------
    print("\n--- 0. STALENESS OF THE LIVE CONDITIONER (the first question) ---")
    px = close_panel(["GLD", "SLV", "GDX", "SPY"]).dropna()
    for nm in SERIES:
        c = cot(nm)
        last = c["release_date"].iloc[-1]
        n_sessions = int(((px.index > last) & (px.index <= TODAY)).sum())
        # how many weekly releases are missing between last and today
        missed = pd.bdate_range(last + pd.Timedelta(days=1), TODAY,
                                freq="W-FRI")
        print(f"  {SERIES[nm]:<44} newest actual {last.date()}  "
              f"({n_sessions} sessions ago)  missing Friday releases: "
              f"{[str(d.date()) for d in missed]}")
    c = cot("gold")
    print(f"\n  Gold latest reading {c['actual'].iloc[-1]:.1f}K at "
          f"{c['pct104'].iloc[-1]:.1f} pctile of its trailing 104 weeks "
          f"({c['pct52'].iloc[-1]:.1f} of 52).")
    print("  A COT release lands TODAY at 15:30 ET, after the pitch is composed")
    print("  and after any MOO entry. It is not in the file either.")

    print("\n  HOW MUCH does the percentile move over the missing window?")
    for nm in SERIES:
        c = cot(nm)
        d5 = (c["pct104"] - c["pct104"].shift(5)).dropna()
        cross_up = ((c["pct104"] >= 90) & (c["pct104"].shift(5) < 90)).sum()
        cross_dn = ((c["pct104"] < 90) & (c["pct104"].shift(5) >= 90)).sum()
        base_hi = int((c["pct104"] >= 90).sum())
        print(f"    {nm:<7} |5-release pctile change| median "
              f"{d5.abs().median():5.1f} pts, p90 {d5.abs().quantile(0.9):5.1f} pts; "
              f">=90 state changed across a 5-release gap "
              f"{cross_up + cross_dn} times against {base_hi} hi readings")

    # ---------------- 1. LIVE TRIGGER EXISTENCE ----------------
    print("\n--- 1. IS THE TRIGGER EVEN ON AT THE LAST KNOWN READING? ---")
    for nm in SERIES:
        c = cot(nm)
        p = c["pct104"].iloc[-1]
        print(f"  {nm:<7} last known pctile {p:5.1f}  -> "
              f">=90 extreme long? {'YES' if p >= 90 else 'NO'}   "
              f"<=10 extreme short? {'YES' if p <= 10 else 'NO'}")

    # ---------------- 2. JOINT-STATE COUNT ----------------
    print("\n--- 2. OCCURRENCE COUNT, before any edge is looked at ---")
    for nm in SERIES:
        c = cot(nm)
        v = c.dropna(subset=["pct104"])
        print(f"  {nm:<7} usable releases with a 104w pctile: {len(v)}  "
              f"({v['release_date'].min().date()} .. "
              f"{v['release_date'].max().date()})")
        for thr in (95, 90, 85, 80):
            n_hi = int((v["pct104"] >= thr).sum())
            d = pd.DatetimeIndex(v.loc[v["pct104"] >= thr, "release_date"])
            d = d[d.isin(px.index) | True]
            pos, kept = anchor_positions(px.index, d, offset=0)
            epi = declusters(pd.DatetimeIndex(px.index[pos]), 5, px.index)
            print(f"      pctile >= {thr}: {n_hi} releases -> "
                  f"{len(epi)} declustered episodes  "
                  f"years {sorted(set(pd.DatetimeIndex(kept).year))}")

    # ---------------- 3. GATE ATTRIBUTION ----------------
    print("\n--- 3. GATE ATTRIBUTION: parent / gated / DISCARDED complement ---")
    print("    (long the vehicle from the close AFTER the release; the")
    print("     unwind thesis wants the gated cell to be WORSE than parent)")
    hdr = (f"{'series':>7} {'veh':>4} {'h':>3} | {'PARENT all rel.':>21} | "
           f"{'GATED pct>=90':>21} | {'COMPLEMENT <90':>21}")
    print(hdr)
    print("-" * len(hdr))
    for nm, veh in (("gold", "GLD"), ("silver", "SLV"), ("gold", "GDX"),
                    ("silver", "GDX"), ("copper", "GLD")):
        c = cot(nm).dropna(subset=["pct104"])
        allp, _ = anchor_positions(px.index, pd.DatetimeIndex(c["release_date"]))
        alld = pd.DatetimeIndex(px.index[allp])
        hip, hik = anchor_positions(
            px.index, pd.DatetimeIndex(c.loc[c["pct104"] >= 90, "release_date"]))
        hid = pd.DatetimeIndex(px.index[hip])
        lop, _ = anchor_positions(
            px.index, pd.DatetimeIndex(c.loc[c["pct104"] < 90, "release_date"]))
        lod = pd.DatetimeIndex(px.index[lop])
        for h in (1, 3, 5, 10):
            ret = vehicle_ret(px, [(veh, 1.0)], h, 1)
            p = ret.reindex(alld).dropna()
            g = ret.reindex(declusters(hid, 5, px.index)).dropna()
            cm = ret.reindex(declusters(lod, 5, px.index)).dropna()
            print(f"{nm:>7} {veh:>4} {h:>3} | "
                  f"{100*p.mean():>+7.3f}% n={len(p):<3} hit {100*(p>0).mean():>4.1f}% | "
                  f"{100*g.mean():>+7.3f}% n={len(g):<3} hit {100*(g>0).mean():>4.1f}% | "
                  f"{100*cm.mean():>+7.3f}% n={len(cm):<3} hit {100*(cm>0).mean():>4.1f}%")

    # ---------------- 4. PLACEBO ANCHOR LADDER ----------------
    print("\n--- 4. PLACEBO ANCHOR LADDER k=-8..+8 (gold pct>=90, h=5) ---")
    c = cot("gold").dropna(subset=["pct104"])
    hi_dates = pd.DatetimeIndex(c.loc[c["pct104"] >= 90, "release_date"])
    for veh in ("GLD", "GDX"):
        rows = []
        for k in range(-8, 9):
            pos, _ = anchor_positions(px.index, hi_dates, offset=k)
            d = declusters(pd.DatetimeIndex(px.index[pos]), 5, px.index)
            v = vehicle_ret(px, [(veh, 1.0)], 5, 1).reindex(d).dropna()
            rows.append((k, 100 * v.mean() if len(v) else np.nan, len(v)))
        ok = [r for r in rows if not np.isnan(r[1])]
        up = sorted(ok, key=lambda r: -r[1])
        dn = sorted(ok, key=lambda r: r[1])
        print(f"\n  {veh} h=5: LIVE k=0 ranks {[r[0] for r in up].index(0)+1} "
              f"of {len(ok)} long / {[r[0] for r in dn].index(0)+1} of {len(ok)} short")
        print("   " + "  ".join(f"k={k:+d}:{m:+.2f}" for k, m, _ in rows))

    # ---------------- 5. DEFINITION FRAGILITY ----------------
    print("\n--- 5. DEFINITION FRAGILITY: nudge the percentile cut ---")
    for nm, veh in (("gold", "GLD"), ("gold", "GDX"), ("silver", "SLV")):
        c = cot(nm).dropna(subset=["pct104"])
        print(f"\n  {nm} -> {veh}, h=5, long, declustered")
        for thr in (95, 92, 90, 88, 85, 80, 75):
            pos, _ = anchor_positions(
                px.index, pd.DatetimeIndex(c.loc[c["pct104"] >= thr,
                                                 "release_date"]))
            d = declusters(pd.DatetimeIndex(px.index[pos]), 5, px.index)
            v = vehicle_ret(px, [(veh, 1.0)], 5, 1).reindex(d).dropna()
            if len(v) < 2:
                print(f"    pct >= {thr}: N={len(v)}")
                continue
            w = int((v > 0).sum())
            print(f"    pct >= {thr}: N={len(v):>3}  mean {100*v.mean():>+7.3f}%  "
                  f"med {100*v.median():>+7.3f}%  record {w}-{len(v)-w}  "
                  f"sign p={sign_test(w, len(v)):.4f}")
        # short-side extreme for symmetry
        print("    (low end, the crowded-short mirror)")
        for thr in (5, 10, 15):
            pos, _ = anchor_positions(
                px.index, pd.DatetimeIndex(c.loc[c["pct104"] <= thr,
                                                 "release_date"]))
            d = declusters(pd.DatetimeIndex(px.index[pos]), 5, px.index)
            v = vehicle_ret(px, [(veh, 1.0)], 5, 1).reindex(d).dropna()
            if len(v) < 2:
                print(f"    pct <= {thr}: N={len(v)}")
                continue
            print(f"    pct <= {thr}: N={len(v):>3}  mean {100*v.mean():>+7.3f}%  "
                  f"hit {100*(v>0).mean():>5.1f}%")

    # ---------------- 6. FULL BATTERY ----------------
    print("\n--- 6. FULL BATTERY on the loudest gated cell ---")
    c = cot("gold").dropna(subset=["pct104"])
    pos, _ = anchor_positions(px.index,
                              pd.DatetimeIndex(c.loc[c["pct104"] >= 90,
                                                     "release_date"]))
    mask = pd.Series(False, index=px.index)
    mask.loc[pd.DatetimeIndex(px.index[pos])] = True
    for veh in ("GLD", "GDX"):
        battery(px, mask, [(veh, 1.0)], h=5,
                title=f"C4 long {veh}, gold COT 104w pctile >= 90, entry next close",
                cost_bps=COST[veh], min_gap=5)

    # ---------------- 7. ONE-REGIME COST ----------------
    print("\n--- 7. WHAT THE 2020+ WINDOW COSTS THE CLAIM ---")
    c = cot("gold").dropna(subset=["pct104"])
    hi = c[c["pct104"] >= 90]
    print("  gold pctile >= 90 releases by year:",
          dict(pd.Series(pd.DatetimeIndex(hi["release_date"]).year)
               .value_counts().sort_index()))
    pos, _ = anchor_positions(px.index, pd.DatetimeIndex(hi["release_date"]))
    d = declusters(pd.DatetimeIndex(px.index[pos]), 5, px.index)
    for veh in ("GLD", "GDX"):
        v = vehicle_ret(px, [(veh, 1.0)], 5, 1).reindex(d).dropna()
        yrs = pd.DatetimeIndex(v.index).year
        by = pd.Series(100 * v.values).groupby(yrs.values).agg(["count", "mean", "sum"])
        print(f"\n  {veh} h=5 gated cell by year (pp):")
        print(by.round(3).to_string())
        for y in sorted(set(yrs)):
            vv = v[yrs != y]
            flag = "  <-- SIGN FLIPS" if (vv.mean() * v.mean()) < 0 else ""
            print(f"    drop {y}: N={len(vv):>3} mean {100*vv.mean():>+7.3f}%{flag}")


if __name__ == "__main__":
    main()
