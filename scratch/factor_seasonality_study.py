"""Factor seasonality — first-pass study (2026-07-17).

Question (McKinley): does trading a factor only in certain calendar months
carry edge — e.g. momentum only in its good months? Data: Fama-French
long-short factors, monthly (data/factor_returns_monthly.parquet via
scripts/fetch_factor_returns.py): Mom 1927+, HML/SMB/MktRF 1926+, RMW/CMA
1963+. Percent units.

Multiple-testing frame: 7 factors x 12 months = 84 cells; at |t|>2 expect
~4 false positives. A cell is INTERESTING only if (a) |t| >= 2.5 full
sample, (b) sign agrees across all three eras, and (c) the
leave-one-decade-out t floor stays >= 1.5. Literature-prior cells
(momentum's January crash, sell-in-May on the market, the January
small/value effect) get called out separately since they were hypothesized
before this data was cut.

Output: per-factor month table + gauntlet survivors + classic composites.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FACTORS = ["Mom", "HML", "SMB", "RMW", "CMA", "MktRF"]
ERA_EDGES = [1926, 1960, 1995, 2027]   # three eras per factor (clipped to coverage)


def month_stats(s: pd.Series) -> pd.DataFrame:
    rows = []
    for m in range(1, 13):
        x = s[s.index.month == m].dropna()
        n = len(x)
        mean = x.mean()
        t = mean / (x.std(ddof=1) / np.sqrt(n)) if n > 2 and x.std(ddof=1) > 0 else np.nan
        rows.append({"month": m, "n": n, "mean": mean, "t": t,
                     "hit": (x > 0).mean() * 100})
    return pd.DataFrame(rows).set_index("month")


def era_signs(s: pd.Series, m: int) -> list:
    signs = []
    for lo, hi in zip(ERA_EDGES[:-1], ERA_EDGES[1:]):
        x = s[(s.index.month == m) & (s.index.year >= lo) & (s.index.year < hi)].dropna()
        if len(x) >= 8:
            signs.append(np.sign(x.mean()))
    return signs


def lodo_floor(s: pd.Series, m: int) -> float:
    """Leave-one-decade-out |t| floor for month m."""
    x = s[s.index.month == m].dropna()
    decades = sorted({y // 10 for y in x.index.year})
    ts = []
    for d in decades:
        keep = x[(x.index.year // 10) != d]
        n = len(keep)
        if n > 10 and keep.std(ddof=1) > 0:
            ts.append(abs(keep.mean() / (keep.std(ddof=1) / np.sqrt(n))))
    return min(ts) if ts else np.nan


def main():
    df = pd.read_parquet(os.path.join(ROOT, "data", "factor_returns_monthly.parquet"))

    survivors = []
    for f in FACTORS:
        s = df[f].dropna()
        st = month_stats(s)
        print(f"\n=== {f} ({s.index.min().year}-{s.index.max().year}, "
              f"ann mean {s.mean()*12:.1f}%) ===")
        print(st.round(2).to_string())
        for m in range(1, 13):
            t = st.loc[m, "t"]
            if pd.notna(t) and abs(t) >= 2.5:
                signs = era_signs(s, m)
                consistent = len(signs) >= 2 and len(set(signs)) == 1
                floor = lodo_floor(s, m)
                verdict = "SURVIVES" if (consistent and floor >= 1.5) else "fails gauntlet"
                survivors.append((f, m, st.loc[m, "mean"], t, consistent, floor, verdict))

    print("\n" + "=" * 78)
    print(f"GAUNTLET (|t|>=2.5 of 84 cells; era-sign agreement; LODO floor >= 1.5)")
    print("=" * 78)
    if survivors:
        out = pd.DataFrame(survivors, columns=[
            "factor", "month", "mean%", "t", "era_consistent", "lodo_floor", "verdict"])
        print(out.round(2).to_string(index=False))
    else:
        print("  none reached |t| >= 2.5")

    # ---- literature-prior composites (hypothesized before looking) ----
    print("\n" + "=" * 78)
    print("LITERATURE-PRIOR COMPOSITES")
    print("=" * 78)

    def comp(name, s, months):
        x_in = s[s.index.month.isin(months)].dropna()
        x_out = s[~s.index.month.isin(months)].dropna()
        d_mean = x_in.mean() - x_out.mean()
        # two-sample t (unequal n, pooled-ish)
        se = np.sqrt(x_in.var(ddof=1) / len(x_in) + x_out.var(ddof=1) / len(x_out))
        print(f"  {name:<38} in {x_in.mean():+.2f}%/mo (n={len(x_in)}) vs "
              f"out {x_out.mean():+.2f}%/mo  diff t={d_mean/se:+.2f}")

    comp("Momentum January crash (Jan vs rest)", df["Mom"].dropna(), [1])
    comp("Momentum ex-January (Feb-Dec vs Jan)", df["Mom"].dropna(),
         [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    comp("Sell-in-May on MktRF (Nov-Apr vs May-Oct)", df["MktRF"].dropna(),
         [11, 12, 1, 2, 3, 4])
    comp("January small-cap effect (SMB Jan)", df["SMB"].dropna(), [1])
    comp("Value in January (HML Jan)", df["HML"].dropna(), [1])

    # post-2000 check on the composites that matter for trading today
    recent = df[df.index.year >= 2000]
    print("\n  post-2000 only:")
    comp("Momentum January crash (post-2000)", recent["Mom"].dropna(), [1])
    comp("Sell-in-May on MktRF (post-2000)", recent["MktRF"].dropna(),
         [11, 12, 1, 2, 3, 4])
    return 0


if __name__ == "__main__":
    sys.exit(main())
