"""Diff the old-engine vs new-engine sig_df dumps from ab_engine_check.py.

Expected diff surface (2026-08-12 fixes):
  - ONLY the overlap pair (Indices Oversold Bounce, SPY QQQ MonFri Reversion)
    may change, and only on staged-collision days (clamp: 20-effective
    proportional on filled pairs -> 30-effective absolute on staged pairs).
  - Everything else — OLV overflow (overflow_active now live), earnings
    override carriers (OLV, St OS Sznl), the rest of the book — must be
    byte-identical on the compared columns.
"""
import sys

import pandas as pd

PAIR = {"Indices Oversold Bounce", "SPY QQQ MonFri Reversion"}
# Tranche + per-key sequence keep stacked positions and OVS near/far
# tranches from exploding the merge (1,211 duplicate 3-col keys).
KEY = ["Strategy", "Ticker", "Date", "Tranche", "_seq"]
CMP = ["Shares", "Risk $", "PnL", "Entry Date", "Exit Date", "Exit Type"]


def main(old_path: str, new_path: str) -> int:
    old = pd.read_parquet(old_path)
    new = pd.read_parquet(new_path)
    print(f"old: {len(old)} trades   new: {len(new)} trades")

    for df in (old, new):
        df["Date"] = pd.to_datetime(df["Date"]).astype(str)
        df["Entry Date"] = pd.to_datetime(df["Entry Date"]).astype(str)
        df["Exit Date"] = pd.to_datetime(df["Exit Date"]).astype(str)

    for df in (old, new):
        df["Tranche"] = df.get("Tranche", "").fillna("")
        base = ["Strategy", "Ticker", "Date", "Tranche", "Entry Date"]
        df.sort_values(base, inplace=True, kind="mergesort")
        df["_seq"] = df.groupby(base[:4]).cumcount()

    cols = KEY + [c for c in CMP if c in old.columns and c in new.columns]
    o = old[cols].sort_values(cols).reset_index(drop=True)
    n = new[cols].sort_values(cols).reset_index(drop=True)

    merged = o.merge(n, on=KEY, how="outer", suffixes=("_old", "_new"),
                     indicator=True)
    only_old = merged[merged["_merge"] == "left_only"]
    only_new = merged[merged["_merge"] == "right_only"]
    both = merged[merged["_merge"] == "both"]

    diff_mask = pd.Series(False, index=both.index)
    for c in CMP:
        co, cn = f"{c}_old", f"{c}_new"
        if co in both.columns:
            a, b = both[co], both[cn]
            if a.dtype.kind in "fc":
                neq = ~((a - b).abs() < 1e-6) & ~(a.isna() & b.isna())
            else:
                neq = a.astype(str) != b.astype(str)
            diff_mask |= neq
    changed = both[diff_mask]

    def bucket(df, label):
        in_pair = df[df["Strategy"].isin(PAIR)]
        outside = df[~df["Strategy"].isin(PAIR)]
        print(f"{label}: {len(df)} total — pair {len(in_pair)}, "
              f"OUTSIDE PAIR {len(outside)}")
        if len(outside):
            print(outside.head(20).to_string())
        return len(outside)

    bad = 0
    bad += bucket(only_old, "rows only in OLD")
    bad += bucket(only_new, "rows only in NEW")
    bad += bucket(changed, "rows changed")

    if len(changed):
        pair_changed = changed[changed["Strategy"].isin(PAIR)]
        print("\nchanged pair rows (expected: staged-collision days only):")
        show = [c for c in ("Strategy", "Ticker", "Date", "Risk $_old",
                            "Risk $_new", "Shares_old", "Shares_new") if c in pair_changed.columns]
        print(pair_changed[show].to_string())

    if bad:
        print(f"\nFAIL: {bad} row(s) changed OUTSIDE the intended surface")
        return 1
    print("\nPASS: all differences confined to the overlap pair")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
