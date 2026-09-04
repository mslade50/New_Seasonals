"""a0: recon the CBOE put/call surface before C1/C2 get any battery time.

Questions this answers, in order:
  1. Coverage. index/etp only exist from a date; with a 10d MA and a 252d
     trailing rank the first usable trigger day is much later still.
  2. Today's readings reproduce the surface map (index 7.1, equity 51.2,
     etp 9.9).
  3. Are the index and etp masks the SAME OBJECT? If they fire together,
     C2 is C1 in a costume.
  4. How many declustered episodes exist at all, and in which years.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import declusters, load_prices  # noqa: E402

PC = Path(__file__).resolve().parents[3] / "data" / "cboe_putcall.parquet"


def pit_pctile(s: pd.Series, ma: int = 10, window: int = 252) -> pd.Series:
    """pc_fear's statistic, generalised to any column: trailing-`window`
    inclusive percentile rank of the `ma`-day MA."""
    m = s.dropna().rolling(ma, min_periods=ma).mean()
    return m.rolling(window, min_periods=window).apply(
        lambda w: (w <= w[-1]).mean() * 100.0, raw=True)


def main() -> None:
    pc = pd.read_parquet(PC)
    print("=== 1. raw coverage ===")
    for c in pc.columns:
        s = pc[c].dropna()
        print(f"  {c:7s} n={len(s):5d}  {s.index.min().date()} .. "
              f"{s.index.max().date()}  mean={s.mean():.3f}")

    p = {c: pit_pctile(pc[c]) for c in ("index", "equity", "etp", "total")}
    print("\n=== 2. first usable PIT percentile day / today's reading ===")
    for c, s in p.items():
        v = s.dropna()
        print(f"  {c:7s} first={v.index.min().date()}  n_days={len(v)}  "
              f"last({v.index[-1].date()})={v.iloc[-1]:.1f}")

    idx = p["index"].dropna()
    eq = p["equity"].reindex(idx.index)
    etp = p["etp"].reindex(idx.index)

    print("\n=== 3. masks ===")
    m_c1 = (idx <= 10) & (eq >= 25) & (eq <= 75)
    m_c1_bare = idx <= 10
    m_c2 = etp <= 10
    for lbl, m in [("C1 index<=10 & eq 25-75", m_c1),
                   ("C1-bare index<=10", m_c1_bare),
                   ("C2 etp<=10", m_c2)]:
        d = idx.index[m.fillna(False).values]
        ep = declusters(d, 10, idx.index)
        yrs = pd.Series(1, index=d).groupby(d.year).sum().to_dict()
        print(f"  {lbl:28s} days={len(d):4d} episodes(gap10)={len(ep):3d}  {yrs}")
        print(f"      episodes: {', '.join(str(x.date()) for x in ep)}")

    print("\n=== 3b. is C2 a costume of C1? ===")
    a = m_c1_bare.fillna(False).values
    b = m_c2.fillna(False).values
    both = int((a & b).sum())
    print(f"  index<=10 days {a.sum()}, etp<=10 days {b.sum()}, both {both}")
    print(f"  P(etp<=10 | index<=10) = {both / max(a.sum(),1):.3f}   "
          f"P(index<=10 | etp<=10) = {both / max(b.sum(),1):.3f}   "
          f"jaccard = {both / max((a | b).sum(),1):.3f}")
    print("  pctile-level corr (pearson) index vs etp = "
          f"{idx.corr(etp):.3f}; index vs equity = {idx.corr(eq):.3f}; "
          f"etp vs equity = {etp.corr(eq):.3f}")
    raw = pc[["index", "etp", "equity"]].dropna()
    print("  RAW ratio corr:\n", raw.corr().round(3).to_string())
    ma10 = raw.rolling(10).mean().dropna()
    print("  MA10 corr:\n", ma10.corr().round(3).to_string())

    print("\n=== 4. today's raw levels ===")
    print(pc.tail(3).to_string())
    print("\n  regime check: index P/C ma10 by year (secular downtrend?)")
    m = pc["index"].dropna().rolling(10).mean()
    print(m.groupby(m.index.year).mean().round(3).to_string())

    px = load_prices(["SPY", "IWM"])
    print("\n  SPY bars available from", px["SPY"].index.min().date(),
          "to", px["SPY"].index.max().date())


if __name__ == "__main__":
    main()
