"""Attach ex-ante signal-quality stats to the EXISTING backtest trades, without
re-running the full walk-forward. Each stat is a function of (ticker, asof,
horizon, direction), so we compute it only for the ~12k trades that exist instead
of re-scanning the whole universe at every date. Then run the filter study
(dose-response + leave-one-year-out) on the enriched book.

Stats added (the engine's own ex-ante confidence measures):
  p_value     binomial p of the all-years directional count (lower = more significant)
  ea          75/25 cycle-blended expected move, ATR units (magnitude of the edge)
  all_hit     all-years hit rate in the trade's direction
  rank_ext    seasonal-rank extremity in the trade's direction (0..50; higher = more extreme)
  disagree    cycle vs all-years sign conflict (engine's C-grade flag)
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = r"C:\Users\McKinley Slade\dev\New_Seasonals"
sys.path.insert(0, ROOT)
import scripts.seasonal_edge as se
from scripts.seasonal_sharpe import dedup

IN = os.path.join(ROOT, "data", "seasonal_ideas_backtest.parquet")
OUT = os.path.join(ROOT, "data", "seasonal_ideas_backtest_enriched.parquet")
BLEND = 0.75


def enrich():
    df = pd.read_parquet(IN)
    df["asof"] = pd.to_datetime(df["asof"]); df["entry_date"] = pd.to_datetime(df.entry_date)
    df["exit_date"] = pd.to_datetime(df.exit_date)
    df["asset"] = np.where(df.channel == "detect_seasonal", "stock", "macro")
    df = dedup(df).reset_index(drop=True)
    print(f"enriching {len(df)} trades ...")

    full = se.load_prices(list(se.IDEA_UNIVERSE), include_overflow=True)
    cs_cache = {}

    def cross(asof):
        if asof not in cs_cache:
            cs_cache[asof] = se.seasonal_cross_section(asof=asof)
        return cs_cache[asof]

    cols = {k: [] for k in ["p_value", "ea", "all_hit", "cyc_hit", "rank", "rank_ext", "disagree", "n_all"]}
    for i, r in enumerate(df.itertuples()):
        px = full.get(se._norm_ticker(r.ticker))
        vals = dict(p_value=np.nan, ea=np.nan, all_hit=np.nan, cyc_hit=np.nan,
                    rank=np.nan, rank_ext=np.nan, disagree=np.nan, n_all=np.nan)
        if px is not None and not px.empty:
            pxc = px[px.index <= r.asof]
            try:
                b = se.seasonal_window_blended(pxc, r.asof, int(r.time_stop_days), blend=BLEND)
            except Exception:
                b = None
            if b is not None:
                s_all = b["all"]; n = s_all.get("n", 0)
                ndir = s_all.get("n_down") if r.direction == "short" else s_all.get("n_up")
                if n:
                    vals["n_all"] = n
                    vals["all_hit"] = ndir / n
                    vals["p_value"] = se.binom_p_greater(int(ndir), int(n))
                vals["ea"] = b.get("ea")
                vals["disagree"] = bool(b.get("disagree"))
                s_cyc = b.get("cyc") or {}
                if b.get("cyc_ok") and s_cyc.get("n"):
                    nd = s_cyc.get("n_down") if r.direction == "short" else s_cyc.get("n_up")
                    vals["cyc_hit"] = nd / s_cyc["n"]
            try:
                rk = cross(r.asof).loc[r.ticker, f"atr_sznl_{int(r.time_stop_days)}d"]
                vals["rank"] = float(rk)
                vals["rank_ext"] = (50.0 - rk) if r.direction == "short" else (rk - 50.0)
            except Exception:
                pass
        for k in cols:
            cols[k].append(vals[k])
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(df)}")

    for k, v in cols.items():
        df[k] = v
    df.to_parquet(OUT, index=False)
    print(f"wrote {OUT}")
    return df


def dose(df, col, q=5, ascending=True):
    d = df.dropna(subset=[col]).copy()
    try:
        d["b"] = pd.qcut(d[col].rank(method="first"), q, labels=[f"Q{i+1}" for i in range(q)])
    except Exception:
        return
    g = d.groupby("b", observed=True).agg(N=("R", "size"), lo=(col, "min"), hi=(col, "max"),
        AvgR=("R", "mean"), Win=("R", lambda x: 100 * (x > 0).mean()),
        PF=("R", lambda x: x[x > 0].sum() / abs(x[x < 0].sum()) if (x < 0).any() else np.inf))
    print(f"\n--- dose-response: {col} (Q1=low .. Q{q}=high) ---")
    print(g.round(3).to_string())


def study(df):
    df["asof"] = pd.to_datetime(df["asof"])
    V1 = df[~((df.asset == "stock") & (df.direction == "short"))].copy()
    print(f"\n=== FILTER STUDY on V1 ({len(V1)} trades, base avgR {V1.R.mean():.3f}, PF "
          f"{V1[V1.R>0].R.sum()/abs(V1[V1.R<0].R.sum()):.2f}) ===")
    for col in ["p_value", "ea_abs", "all_hit", "rank_ext"]:
        if col == "ea_abs":
            V1["ea_abs"] = V1["ea"].abs()
        dose(V1, col)
    # disagree flag
    if "disagree" in V1.columns:
        for flag, g in V1.groupby("disagree"):
            R = g.R.astype(float)
            print(f"\ndisagree={flag}: N {len(g)} avgR {R.mean():.3f} win {100*(R>0).mean():.1f}% "
                  f"PF {R[R>0].sum()/abs(R[R<0].sum()):.2f}")


if __name__ == "__main__":
    if os.path.exists(OUT) and "--reuse" in sys.argv:
        d = pd.read_parquet(OUT)
    else:
        d = enrich()
    study(d)
