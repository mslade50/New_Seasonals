"""scratch/seasonal_path_entry.py — entry-timing on the EXPECTED seasonal path.

Two tradeable (ex-ante) entry ideas vs the baseline T+1 market-on-open:

  delayed (path nadir/peak): for each signal, build the EXPECTED day-by-day
    seasonal path from prior years (same pick logic as seasonal_window_returns),
    find the day the path bottoms (long) / peaks (short), and enter at that day's
    open instead of T+1. Known at signal time -> tradeable, not look-ahead.

  limit_persistent 0.75 ATR: GTC limit at open -/+ 0.75 ATR that lives the whole
    trade window (vs the plain T+1-only limit). Plus the plain 0.75 limit for ref.

Reports avgR / PF / TotR / Sharpe (deduped, V1 = excl stock shorts) + fill rate,
the nadir-day distribution, and the uplift on the signals the path rule moves.
"""
import os
import sys

ROOT = r"C:\Users\McKinley Slade\dev\New_Seasonals"
sys.path.insert(0, ROOT)
import numpy as np
import pandas as pd
import scripts.seasonal_edge as se
from scripts.seasonal_ticket_sim import simulate_ticket
from scripts.seasonal_sharpe import dedup
from scripts.resim_seasonal_entry import report

CAND = os.path.join(ROOT, "data", "seasonal_ideas_candidates.parquet")


def expected_path(close, asof, N, doy_tol=2, min_years=3):
    """Avg per-day cumulative return path (len N) over prior years' same window."""
    close = close.dropna().sort_index()
    if close.empty:
        return None
    asof = pd.Timestamp(asof).normalize()
    doy = se._trading_doy(close.index).values
    years = close.index.year.values.astype(np.int64)
    le = close.index.values <= np.datetime64(asof)
    if not le.any():
        return None
    target = int(doy[le][-1])
    picks = se._window_pick_positions(doy, years, target, asof.year, None, doy_tol, True)
    if picks.size < min_years:
        return None
    cv = close.values.astype(np.float64)
    paths = [cv[p + 1:p + N + 1] / cv[p] - 1.0 for p in picks if p + N < cv.size]
    if len(paths) < min_years:
        return None
    return np.nanmean(np.vstack(paths), axis=0)


def run(full, cand, entry_mode, mult=0.75):
    trades, nofill, nadir_days = [], 0, []
    for r in cand.itertuples():
        px = full.get(se._norm_ticker(r.ticker))
        if px is None or px.empty:
            continue
        ew = None
        if entry_mode in ("delayed", "delayed_limit"):
            pth = se.expected_seasonal_path(px, r.asof, int(r.time_stop_days))  # ATR-normalized core fn
            if pth is None:
                continue
            ew = int(np.argmin(pth)) if r.direction == "long" else int(np.argmax(pth))
            nadir_days.append(ew + 1)
        tk = {"ticker": r.ticker, "direction": r.direction, "entry": float(r.t_entry),
              "stop": float(r.t_stop), "target": float(r.t_target),
              "time_stop_days": int(r.time_stop_days)}
        # limit fills are intrabar-ambiguous -> forbid same-bar target (arms T+1)
        edt = entry_mode not in ("limit", "limit_persistent", "delayed_limit")
        out = simulate_ticket(tk, px, r.asof, entry_mode=entry_mode,
                              entry_atr_mult=mult, entry_window=ew, reanchor=True,
                              entry_day_target=edt)
        if out is None:
            continue
        if not out.get("filled", True):
            nofill += 1
            continue
        trades.append({"asof": r.asof, "ticker": r.ticker, "channel": r.channel,
                       "direction": r.direction, "horizon": r.horizon,
                       "time_stop_days": r.time_stop_days, "cycle": r.cycle, **out})
    df = pd.DataFrame(trades)
    attempted = len(df) + nofill
    if not df.empty:
        df["asset"] = np.where(df["channel"] == "detect_seasonal", "stock", "macro")
        df = dedup(df).reset_index(drop=True)
    return df, attempted, nofill, nadir_days


def uplift(base, alt, only_delayed_idx=None):
    key = ["asof", "ticker", "direction"]
    b = base[~((base.asset == "stock") & (base.direction == "short"))]
    m = b[key + ["R", "entry_date"]].merge(alt[key + ["R", "entry_date"]], on=key, suffixes=("_b", "_a"))
    if m.empty:
        return "  (no overlap)"
    moved = (pd.to_datetime(m["entry_date_a"]) - pd.to_datetime(m["entry_date_b"])).dt.days > 0
    dR = m["R_a"] - m["R_b"]
    s = (f"  all: avgR {m['R_b'].mean():+.3f} -> {m['R_a'].mean():+.3f} (Δ{dR.mean():+.3f}R), "
         f"{100 * (dR > 1e-9).mean():.0f}% improved")
    if moved.any():
        mm = m[moved]
        s += (f"\n  moved only ({moved.sum()} of {len(m)}): "
              f"avgR {mm['R_b'].mean():+.3f} -> {mm['R_a'].mean():+.3f} (Δ{(mm['R_a']-mm['R_b']).mean():+.3f}R)")
    return s


def main():
    cand = pd.read_parquet(CAND)
    cand["asof"] = pd.to_datetime(cand["asof"])
    print(f"candidates: {len(cand)} | universe: {len(se.IDEA_UNIVERSE)} tickers")
    full = se.load_prices(list(se.IDEA_UNIVERSE), include_overflow=True)

    base, att, nf, _ = run(full, cand, "t1_open")
    report(base, "BASELINE  t1_open")
    print(f"  fills {len(base)}/{att}")

    for mode, mult, label in [("delayed", 0.75, "DELAYED to expected seasonal-path nadir/peak"),
                              ("limit_persistent", 0.75, "LIMIT 0.75 ATR (persistent, life of trade)"),
                              ("delayed_limit", 0.75, "COMBINED: path-nadir delay + persistent 0.75 ATR limit")]:
        df, att, nf, nd = run(full, cand, mode, mult)
        report(df, label)
        print(f"  fills {len(df)}/{att} = {100*len(df)/max(1,att):.0f}%" + (f" (nofill {nf})" if nf else ""))
        if nd:
            nd = np.array(nd)
            print(f"  nadir-day: day1 {100*(nd==1).mean():.0f}%  day2 {100*(nd==2).mean():.0f}%  "
                  f"day3+ {100*(nd>=3).mean():.0f}%  (mean {nd.mean():.1f})")
        print(uplift(base, df))


if __name__ == "__main__":
    main()
