"""Walk-forward backtest of the seasonal IDEAS engine.

Replays daily_seasonal_ideas' candidate pipeline at historical as-of dates and
realizes every emitted ticket forward, producing a track record of the engine as
configured. The detectors are already asof-parameterized; the only leak is that
build_trade_ticket anchors entry to the LAST price bar, so we monkeypatch
se.load_prices to return frames capped to <= asof. The simulator separately reads
the FULL (uncapped) series to see the forward outcome.

Basis: adjusted prices (se.load_prices) for both ticket-minting and forward
simulation — internally scale-invariant (the strat_backtester basis), so no
frozen-level dividend phantom. The forward LIVE scorer uses raw bars instead
(verify_fills basis); both share scripts/seasonal_ticket_sim.simulate_ticket.

KNOWN CAVEATS (by request):
  - Meta in-sample: detector rules/thresholds/FDR alpha were designed knowing
    history. This validates "do flagged ideas realize their edge," not OOS rule
    discovery.
  - Survivorship: IDEA_UNIVERSE / sznl_ranks reflect current membership.
  - Negative filter vs the live book is SKIPPED by default (replaying the full
    STRATEGY_BOOK scan at every asof is expensive); --negative-filter enables it.
  - Regime context is neutralized (no historical rd2_environment snapshots), so
    the regime_sleeve channel is effectively off in replay.
  - Entry = T+1 open by default; the seasonal stat is measured close-to-close, a
    minor offset (see --entry-mode asof_close for the faithful-to-stat variant).
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import scripts.seasonal_edge as se
from scripts.seasonal_ticket_sim import parse_ticket, simulate_ticket
import daily_seasonal_ideas as dsi

OUT = os.path.join(_ROOT, "data", "seasonal_ideas_backtest.parquet")
CAND_OUT = os.path.join(_ROOT, "data", "seasonal_ideas_candidates.parquet")

# ---- point-in-time price patch -------------------------------------------------
_FULL: dict[str, pd.DataFrame] = {}
_CAP = {"asof": None}
_orig_load_prices = se.load_prices


def _capped_load_prices(tickers, include_overflow: bool = True):
    a = _CAP["asof"]
    out = {}
    for t in tickers:
        k = se._norm_ticker(t)
        df = _FULL.get(k)
        if df is None or df.empty:
            continue
        out[k] = df if a is None else df[df.index <= a]
    return out


def _install_speedups():
    """Memoize seasonal_cross_section by asof (both channels call it per asof) and
    pre-filter the 7M-row ranks frame to the idea universe. Both preserve scan
    output (scan only reads universe tickers); verified before the daily run."""
    full_ranks = se.load_seasonal_ranks()
    uni = set(t.upper() for t in se.IDEA_UNIVERSE) | {se._norm_ticker(t).upper() for t in se.IDEA_UNIVERSE}
    uni_ranks = full_ranks[full_ranks["ticker"].isin(uni)].sort_values("Date").reset_index(drop=True)
    se.load_seasonal_ranks = lambda path=None: uni_ranks
    _orig_cs = se.seasonal_cross_section
    cache = {}

    def _memo_cs(asof=None, ranks=None):
        if ranks is not None:
            return _orig_cs(asof=asof, ranks=ranks)
        key = pd.Timestamp(asof).normalize() if asof is not None else None
        if key not in cache:
            cache[key] = _orig_cs(asof=asof, ranks=uni_ranks)
        return cache[key]

    se.seasonal_cross_section = _memo_cs
    print(f"  speedups: ranks {len(full_ranks)} -> {len(uni_ranks)} rows; cross-section memoized")


def dedup(d: pd.DataFrame) -> pd.DataFrame:
    """One open position per (ticker, direction): skip any flag whose entry is on/
    before the last kept trade's exit. 'If it's already on, it's done.'"""
    d = d.sort_values(["ticker", "direction", "entry_date"])
    keep, last_exit = [], {}
    for r in d.itertuples():
        k = (r.ticker, r.direction)
        le = last_exit.get(k)
        if le is None or r.entry_date > le:
            keep.append(r.Index)
            last_exit[k] = r.exit_date
    return d.loc[keep]


def _trading_days(start, end, cadence: str):
    """As-of dates to replay, taken from the SPY/index calendar in the cache."""
    spx = _FULL.get("^GSPC")
    if spx is None or spx.empty:
        spx = _FULL.get("SPY")
    if spx is None or spx.empty:
        raise SystemExit("no ^GSPC/SPY in cache for the trading calendar")
    idx = spx.index
    idx = idx[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))]
    if cadence == "daily":
        return list(idx)
    if cadence == "weekly":
        s = pd.Series(idx, index=idx)
        return list(s.groupby(s.dt.isocalendar().week.astype(int) + 100 * s.dt.year).first())
    if cadence == "monthly":
        s = pd.Series(idx, index=idx)
        return list(s.groupby([s.dt.year, s.dt.month]).first())
    raise SystemExit(f"bad cadence {cadence}")


def _run_detectors(asof, ctx, channels):
    """Like dsi.run_detectors but restricted to the requested channels."""
    import scripts.seasonal_detectors as sd
    out = []
    for key, attr, _title in dsi.CHANNELS:
        if channels and key not in channels:
            continue
        fn = getattr(sd, attr, None)
        if fn is None:
            continue
        try:
            out.extend(fn(asof, ctx) or [])
        except Exception:
            pass
    return out


def generate_candidates(asof, grades, negative_filter: bool, channels=None):
    """build()'s pipeline minus current-regime contamination."""
    ctx = {"asof": asof, "regime": {}, "min_rr": 2.0, "universe": list(se.IDEA_UNIVERSE)}
    cands = _run_detectors(asof, ctx, channels)
    cands = dsi.universe_filter(cands)
    if negative_filter:
        cands = dsi.negative_filter(cands, asof, ctx)
    cands = dsi.apply_fdr(cands)
    if grades:
        cands = [c for c in cands if c.get("conviction") in grades]
    cands = dsi.cap_per_channel(cands)
    return cands


def run(start, end, cadence, grades, entry_mode, negative_filter, channels=None,
        verbose=False, do_dedup=True, speedups=True, entry_atr_mult=0.25):
    global _FULL
    print(f"Loading full price history for {len(se.IDEA_UNIVERSE)} idea-universe tickers ...")
    _FULL = _orig_load_prices(list(se.IDEA_UNIVERSE), include_overflow=True)
    print(f"  loaded {len(_FULL)} tickers")
    if speedups:
        _install_speedups()
    se.load_prices = _capped_load_prices  # patch for all detectors

    asofs = _trading_days(start, end, cadence)
    print(f"Replaying {len(asofs)} as-of dates ({cadence}) {start} -> {end} ...")

    trades, candidates, nofill = [], [], 0
    try:
        for i, asof in enumerate(asofs):
            asof = pd.Timestamp(asof).normalize()
            _CAP["asof"] = asof
            try:
                cands = generate_candidates(asof, grades, negative_filter, channels)
            except Exception as e:
                if verbose:
                    print(f"  [{asof.date()}] detector error: {e}")
                continue
            for c in cands:
                tk = parse_ticket(c)
                if tk is None:
                    continue
                full = _FULL.get(se._norm_ticker(tk["ticker"]))
                if full is None:
                    continue
                rec = {
                    "asof": asof, "ticker": tk["ticker"], "channel": tk["channel"],
                    "direction": tk["direction"], "horizon": tk["horizon"],
                    "time_stop_days": tk["time_stop_days"], "conviction": tk["conviction"],
                    "rr": tk["rr"], "cycle": int(asof.year % 4),
                    "t_entry": tk["entry"], "t_stop": tk["stop"], "t_target": tk["target"],
                }
                candidates.append(rec)  # every flagged ticket, for fast entry re-sims
                out = simulate_ticket(tk, full, asof, entry_mode=entry_mode,
                                      entry_atr_mult=entry_atr_mult)
                if out is None:
                    continue  # not matured (too recent) — drop from the backtest
                if not out.get("filled", True):
                    nofill += 1
                    continue  # limit not touched — missed fill, no trade
                trades.append({**rec, **out})
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(asofs)} asofs, {len(trades)} fills ({nofill} missed) so far")
    finally:
        se.load_prices = _orig_load_prices  # always restore

    if candidates:
        pd.DataFrame(candidates).to_parquet(CAND_OUT, index=False)
        print(f"Wrote {len(candidates)} candidate tickets -> {CAND_OUT} (for fast entry re-sims)")
    attempted = len(trades) + nofill
    if attempted:
        print(f"Entry '{entry_mode}' (mult {entry_atr_mult}): {len(trades)} fills / "
              f"{attempted} attempted = {100*len(trades)/attempted:.0f}% fill rate")
    df = pd.DataFrame(trades)
    if df.empty:
        print("No trades generated.")
        return df
    if do_dedup:
        n0 = len(df)
        df = dedup(df).reset_index(drop=True)
        print(f"\nDedup: {n0} -> {len(df)} trades (one open per ticker+direction)")
    df.to_parquet(OUT, index=False)
    print(f"Wrote {len(df)} simulated trades -> {OUT}")
    _report(df)
    return df


def _agg(g: pd.DataFrame) -> dict:
    R = g["R"].astype(float)
    wins, losses = R[R > 0], R[R < 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() else np.inf
    return {
        "N": len(g), "Win%": round(100 * (R > 0).mean(), 1),
        "AvgR": round(R.mean(), 3), "MedR": round(R.median(), 3),
        "TotR": round(R.sum(), 1), "AvgR/Std": round(R.mean() / R.std(), 3) if R.std() else np.nan,
        "PF": round(pf, 2) if np.isfinite(pf) else np.inf,
        "%Tgt": round(100 * (g["exit_type"] == "Target").mean(), 1),
        "%Stop": round(100 * (g["exit_type"] == "Stop").mean(), 1),
        "%Time": round(100 * (g["exit_type"] == "Time").mean(), 1),
    }


def _report(df: pd.DataFrame):
    pd.set_option("display.width", 220)
    print("\n=== OVERALL ===")
    print(pd.DataFrame([_agg(df)]).to_string(index=False))
    for col in ["channel", "conviction", "horizon", "direction", "cycle"]:
        if col in df.columns:
            rows = {k: _agg(g) for k, g in df.groupby(col)}
            print(f"\n=== by {col} ===")
            print(pd.DataFrame(rows).T.to_string())
    # equity curve by exit date
    d = df.sort_values("exit_date")
    eq = d["R"].cumsum()
    dd = float((eq - eq.cummax()).min())
    print(f"\nCumulative R: {eq.iloc[-1]:.1f} | max drawdown: {dd:.1f} R | "
          f"trades/yr: {len(df) / max(1, (df['asof'].max() - df['asof'].min()).days / 365):.0f}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 8))
        a1.plot(pd.to_datetime(d["exit_date"]).values, eq.values, color="#222", lw=1.6, label="all")
        for ch, g in df.groupby("channel"):
            gg = g.sort_values("exit_date")
            a1.plot(pd.to_datetime(gg["exit_date"]).values, gg["R"].cumsum().values, lw=1.2, label=ch)
        a1.set_title("Seasonal ideas backtest — cumulative R by exit date")
        a1.legend(); a1.grid(alpha=.3); a1.set_ylabel("cum R")
        a2.plot(pd.to_datetime(d["exit_date"]).values, (eq - eq.cummax()).values, color="#c0392b", lw=1)
        a2.set_title("Underwater (R below peak)"); a2.grid(alpha=.3); a2.set_ylabel("R")
        fig.tight_layout()
        png = os.path.join(_ROOT, "scratch", "seasonal_ideas_backtest.png")
        os.makedirs(os.path.dirname(png), exist_ok=True)
        fig.savefig(png, dpi=110)
        print(f"Saved curve -> {png}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2012-01-01")
    ap.add_argument("--end", default=(datetime.date.today() - datetime.timedelta(days=120)).isoformat())
    ap.add_argument("--cadence", default="weekly", choices=["daily", "weekly", "monthly"])
    ap.add_argument("--grades", default="A", help="conviction grades, e.g. A / AB / ABC; 'all' = no filter")
    ap.add_argument("--entry-mode", default="t1_open", choices=["t1_open", "asof_close", "limit"])
    ap.add_argument("--entry-atr-mult", type=float, default=0.25, help="limit offset from T+1 open in ATR")
    ap.add_argument("--negative-filter", action="store_true", help="replay live-book overlap removal (slow)")
    ap.add_argument("--channels", default="seasonal,cross_asset",
                    help="comma list of detector channels to replay; 'all' for every channel")
    ap.add_argument("--no-dedup", action="store_true", help="keep overlapping re-emissions")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    grades = None if a.grades.lower() == "all" else tuple(a.grades.upper())
    channels = None if a.channels.lower() == "all" else set(a.channels.split(","))
    run(a.start, a.end, a.cadence, grades, a.entry_mode, a.negative_filter, channels,
        a.verbose, do_dedup=not a.no_dedup, entry_atr_mult=a.entry_atr_mult)
