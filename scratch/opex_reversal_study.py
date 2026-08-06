"""Does opex mark a turning point after outsized moves into expiry?

For every opex 2000+: pre-window return into the opex close (td -5..0 and
td -10..0), normalized by trailing 21d realized vol into a z-score, then
forward returns (+1..+5, +1..+10, opex -> month-end) bucketed by that z.
Splits: all opex / quad only / non-quad. SPY primary, QQQ + IWM confirm.
Also rank-IC of pre vs forward, and per-episode lists for the tail buckets.

Run: python scratch/opex_reversal_study.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from macro_calendar import event_dates  # noqa: E402


def load(tkr: str) -> pd.DataFrame:
    mp = pd.read_parquet(ROOT / "data" / "master_prices.parquet",
                         columns=["ticker", "date", "Close"])
    df = mp[mp["ticker"] == tkr].set_index("date").sort_index()[["Close"]]
    df.index = pd.to_datetime(df.index).normalize()
    df = df[~df.index.duplicated(keep="last")]
    df = df[df.index >= "1999-06-01"]
    df["ret"] = df["Close"].pct_change()
    df["vol21"] = df["ret"].rolling(21).std()
    return df


def stats(x: pd.Series) -> dict:
    x = x.dropna()
    if len(x) < 3:
        return {"mean_bps": np.nan, "t": np.nan, "N": len(x), "hit": np.nan}
    t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    return {"mean_bps": round(x.mean() * 1e4, 1), "t": round(t, 2),
            "N": len(x), "hit": round((x > 0).mean(), 2)}


def build_table(tkr: str) -> pd.DataFrame:
    df = load(tkr)
    idx = df.index
    rows = []
    for d in event_dates("opex"):
        p = idx.searchsorted(d)
        if p < 30 or p >= len(idx) - 11 or idx[p] > pd.Timestamp("2026-08-01"):
            continue
        c = df["Close"]
        vol = df["vol21"].iloc[p - 5]  # known before the pre-window resolves
        me = idx.searchsorted(pd.Timestamp(d.year, d.month, 28)
                              + pd.Timedelta(days=4), side="left") - 1
        rows.append({
            "date": idx[p], "month": d.month,
            "quad": d.month in (3, 6, 9, 12),
            "pre5": float(c.iloc[p] / c.iloc[p - 5] - 1),
            "pre10": float(c.iloc[p] / c.iloc[p - 10] - 1),
            "z5": float((c.iloc[p] / c.iloc[p - 5] - 1)
                        / (vol * np.sqrt(5))) if vol > 0 else np.nan,
            "z10": float((c.iloc[p] / c.iloc[p - 10] - 1)
                         / (vol * np.sqrt(10))) if vol > 0 else np.nan,
            "fwd5": float(c.iloc[p + 5] / c.iloc[p] - 1),
            "fwd10": float(c.iloc[p + 10] / c.iloc[p] - 1),
            "fwd_me": float(c.iloc[me] / c.iloc[p] - 1) if me > p else np.nan,
        })
    return pd.DataFrame(rows).set_index("date")


def bucket_report(w: pd.DataFrame, zcol: str, title: str) -> None:
    print(f"\n--- {title} ---")
    buckets = [("big DOWN (z<-1)", w[zcol] < -1),
               ("mild (-1..+1)", w[zcol].abs() <= 1),
               ("big UP (z>+1)", w[zcol] > 1),
               ("extreme DOWN (z<-1.5)", w[zcol] < -1.5),
               ("extreme UP (z>+1.5)", w[zcol] > 1.5)]
    hdr = f"{'bucket':24s}"
    for f in ("fwd5", "fwd10", "fwd_me"):
        hdr += f" | {f}: mean_bps t N hit"
    print(hdr)
    for name, m in buckets:
        line = f"{name:24s}"
        for f in ("fwd5", "fwd10", "fwd_me"):
            s = stats(w.loc[m, f])
            line += (f" | {s['mean_bps']:+8.1f} {s['t']:+5.2f} "
                     f"{s['N']:3d} {s['hit']}")
        print(line)


def main() -> None:
    for tkr in ("SPY", "QQQ", "IWM"):
        w = build_table(tkr)
        print("=" * 100)
        print(f"{tkr}  ({w.index.min():%Y-%m} .. {w.index.max():%Y-%m}, "
              f"{len(w)} opex)")
        print("=" * 100)
        for zcol, pre in (("z5", "pre5"), ("z10", "pre10")):
            ic5 = w[pre].corr(w["fwd5"], method="spearman")
            ic10 = w[pre].corr(w["fwd10"], method="spearman")
            print(f"\n[{zcol}] rank-IC pre vs fwd5 {ic5:+.3f}, "
                  f"vs fwd10 {ic10:+.3f}")
            bucket_report(w, zcol, f"{zcol} ALL opex")
            bucket_report(w[w.quad], zcol, f"{zcol} QUAD only")
            bucket_report(w[~w.quad], zcol, f"{zcol} monthly (non-quad)")

    # episode lists for the interesting tails, SPY
    w = build_table("SPY")
    for name, m in (("SPY big DOWN into opex (z10<-1)", w.z10 < -1),
                    ("SPY big UP into opex (z10>+1)", w.z10 > 1)):
        print(f"\n{name}: {int(m.sum())} episodes")
        sub = w.loc[m, ["pre10", "fwd5", "fwd10", "fwd_me", "quad"]]
        for d, r in sub.iterrows():
            print(f"  {d:%Y-%m-%d} {'Q' if r.quad else ' '} "
                  f"pre10 {r.pre10*1e4:+7.0f}  fwd5 {r.fwd5*1e4:+7.0f}  "
                  f"fwd10 {r.fwd10*1e4:+7.0f}  to_me {r.fwd_me*1e4:+7.0f}")


if __name__ == "__main__":
    main()
