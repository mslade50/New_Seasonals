"""Long-only seasonal ideas: no-stop sqrt-time sizing vs stop/target systems.

Question (McKinley 2026-08-05): take ONLY the long tickets, drop stops entirely,
size on a sqrt-of-time risk unit (5d hold = 1 ATR, h-day hold = ATR*sqrt(h/5)),
hold to the seasonal time-stop. How does that compare to any stop/target system
on the SAME entries? (Thesis: oversold into seasonal buys — stops sell max fear.)

Design:
  - Entries identical across variants: T+1 open, long candidates only, both
    channels. Deduped ONCE on full-window occupancy (one open per ticker,
    window = asof..asof+h), so every variant trades the exact same list.
  - Adjusted bars (backtest basis, scale-invariant — relative levels only).
  - Stop fills use the book gap-through convention (min(stop, open) + 3bps,
    +10bps gapped). Targets/time exits: no slippage.
  - Dollar comparison at $1k risk per trade: shares = 1000 / risk_unit_$.
      stop-sized variants: risk = ticket entry-stop distance (planned risk)
      sqrt-sized variants: risk = ATR(asof) * sqrt(h/5)
  - Ticket levels FROZEN (reanchor=False), matching the incumbent backtest.

Variants:
  nostop_sqrt  time exit only                       sqrt-time sizing
  sqrt_stop    stop at entry - ATR*sqrt(h/5), time  sqrt-time sizing
  ticket       ticket stop + target + time          stop-distance sizing
  stop_only    ticket stop + time                   stop-distance sizing
  target_only  ticket target + time                 sqrt-time sizing
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = r"C:\Users\McKinley Slade\dev\New_Seasonals"
sys.path.insert(0, ROOT)
import scripts.seasonal_edge as se
from scripts.seasonal_sharpe import ratios

CAND = os.path.join(ROOT, "data", "seasonal_ideas_candidates.parquet")
OUT = os.path.join(ROOT, "scratch", "seasonal_nostop_trades.parquet")

RISK_DOLLARS = 1000.0
STOP_SLIP_BPS, STOP_GAP_SLIP_BPS = 3.0, 10.0
VARIANTS = ["nostop_sqrt", "sqrt_stop", "ticket", "stop_only", "target_only"]


def atr_series(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False).mean()


def stop_fill(stop_price: float, day_open: float) -> float:
    gapped = day_open < stop_price
    fill = min(stop_price, day_open)
    bps = STOP_SLIP_BPS + (STOP_GAP_SLIP_BPS if gapped else 0.0)
    return fill * (1.0 - bps / 1e4)


def first_hit(arr: np.ndarray, level: float, below: bool) -> int:
    idx = np.flatnonzero(arr <= level if below else arr >= level)
    return int(idx[0]) if idx.size else -1


def sim_all(cand: pd.DataFrame) -> pd.DataFrame:
    full = se.load_prices(list(se.IDEA_UNIVERSE), include_overflow=True)
    atr_cache: dict[str, pd.Series] = {}
    rows = []
    skipped = {"no_px": 0, "not_matured": 0, "no_atr": 0}
    for r in cand.itertuples():
        px = full.get(se._norm_ticker(r.ticker))
        if px is None or px.empty:
            skipped["no_px"] += 1
            continue
        df = px[~px.index.duplicated(keep="last")].sort_index()
        asof = pd.Timestamp(r.asof).normalize()
        fwd = df[df.index > asof]
        n = int(r.time_stop_days)
        if len(fwd) < n:
            skipped["not_matured"] += 1
            continue
        t = se._norm_ticker(r.ticker)
        if t not in atr_cache:
            atr_cache[t] = atr_series(df)
        past = atr_cache[t][atr_cache[t].index <= asof]
        atr = float(past.iloc[-1]) if len(past) >= 15 else np.nan
        if not np.isfinite(atr) or atr <= 0:
            skipped["no_atr"] += 1
            continue

        win = fwd.iloc[:n]
        op = win["Open"].values.astype(float)
        hi = win["High"].values.astype(float)
        lo = win["Low"].values.astype(float)
        cl = win["Close"].values.astype(float)
        entry = op[0]

        risk_stop = float(r.t_entry) - float(r.t_stop)          # planned risk
        risk_sqrt = atr * np.sqrt(n / 5.0)
        if risk_stop <= 0:
            continue
        sqrt_stop_lvl = entry - risk_sqrt
        tk_stop, tk_tgt = float(r.t_stop), float(r.t_target)

        i_tkstop = first_hit(lo, tk_stop, below=True)
        i_tktgt = first_hit(hi, tk_tgt, below=False)
        i_sqstop = first_hit(lo, sqrt_stop_lvl, below=True)
        time_exit = cl[-1]

        base = dict(asof=asof, ticker=r.ticker, channel=r.channel,
                    horizon=r.horizon, h=n, cycle=int(r.cycle),
                    entry_date=win.index[0], entry=entry, atr=atr,
                    risk_stop=risk_stop, risk_sqrt=risk_sqrt,
                    mae_atr=(lo.min() - entry) / atr,
                    held_R_sqrt=(time_exit - entry) / risk_sqrt)

        for v in VARIANTS:
            if v == "nostop_sqrt":
                i_s, i_t, s_lvl, tgt, risk = -1, -1, np.nan, np.nan, risk_sqrt
            elif v == "sqrt_stop":
                i_s, i_t, s_lvl, tgt, risk = i_sqstop, -1, sqrt_stop_lvl, np.nan, risk_sqrt
            elif v == "ticket":
                i_s, i_t, s_lvl, tgt, risk = i_tkstop, i_tktgt, tk_stop, tk_tgt, risk_stop
            elif v == "stop_only":
                i_s, i_t, s_lvl, tgt, risk = i_tkstop, -1, tk_stop, np.nan, risk_stop
            else:  # target_only
                i_s, i_t, s_lvl, tgt, risk = -1, i_tktgt, np.nan, tk_tgt, risk_sqrt

            # earliest event wins; same-bar tie -> stop first (book convention)
            if i_s >= 0 and (i_t < 0 or i_s <= i_t):
                exit_px = stop_fill(s_lvl, op[i_s])
                i_x, etype = i_s, "Stop"
            elif i_t >= 0:
                exit_px, i_x, etype = tgt, i_t, "Target"
            else:
                exit_px, i_x, etype = time_exit, n - 1, "Time"

            pnl_r = (exit_px - entry) / risk
            sh = RISK_DOLLARS / risk
            rows.append({**base, "variant": v, "exit_date": win.index[i_x],
                         "exit_type": etype, "R": pnl_r,
                         "dollars": sh * (exit_px - entry)})
    print(f"skipped: {skipped}")
    return pd.DataFrame(rows)


def dedup_window(cand: pd.DataFrame) -> pd.DataFrame:
    """One open per ticker; occupancy = the FULL seasonal window (asof..asof+h),
    calendar-approximated via busday offsets, identical for every variant."""
    cand = cand.sort_values(["ticker", "asof"])
    keep, busy = [], {}
    for r in cand.itertuples():
        end = np.busday_offset(np.datetime64(pd.Timestamp(r.asof).date()),
                               int(r.time_stop_days), roll="forward")
        if busy.get(r.ticker) is None or np.datetime64(pd.Timestamp(r.asof).date()) > busy[r.ticker]:
            keep.append(r.Index)
            busy[r.ticker] = end
    return cand.loc[keep]


def summarize(df: pd.DataFrame, label: str):
    print(f"\n================ {label} ================")
    full = pd.date_range(df["exit_date"].min().normalize(),
                         df["exit_date"].max().normalize(), freq="B")
    hdr = (f"{'variant':12s} {'N':>5s} {'win%':>5s} {'avgR':>6s} {'PF':>5s} "
           f"{'$/trade':>8s} {'tot$k':>7s} {'Sharpe':>6s} {'Sortino':>7s} "
           f"{'maxDD$k':>7s} {'worstTrade':>10s} {'worstMo$k':>9s} {'%stop':>5s} {'%tgt':>5s}")
    print(hdr)
    for v in VARIANTS:
        b = df[df.variant == v]
        if b.empty:
            continue
        d = b["dollars"].astype(float)
        R = b["R"].astype(float)
        pf = d[d > 0].sum() / abs(d[d < 0].sum()) if (d < 0).any() else np.inf
        daily = b.groupby(b["exit_date"].dt.normalize())["dollars"].sum().reindex(full, fill_value=0.0)
        monthly = daily.resample("ME").sum()
        sh, so = ratios(monthly, 12)
        eq = daily.cumsum()
        maxdd = float((eq - eq.cummax()).min())
        pstop = 100 * (b.exit_type == "Stop").mean()
        ptgt = 100 * (b.exit_type == "Target").mean()
        print(f"{v:12s} {len(b):5d} {100*(d>0).mean():5.1f} {R.mean():6.3f} {pf:5.2f} "
              f"{d.mean():8.1f} {d.sum()/1e3:7.0f} {sh:6.2f} {so:7.2f} "
              f"{maxdd/1e3:7.1f} {d.min():10.0f} {monthly.min()/1e3:9.1f} {pstop:5.1f} {ptgt:5.1f}")


def main():
    cand = pd.read_parquet(CAND)
    cand["asof"] = pd.to_datetime(cand["asof"])
    cand = cand[cand.direction == "long"].copy()
    print(f"long candidates: {len(cand)}")
    cand = dedup_window(cand)
    print(f"after window-occupancy dedup (one open per ticker): {len(cand)}")

    df = sim_all(cand)
    df.to_parquet(OUT)
    print(f"simmed {df.variant.value_counts().iloc[0]} trades/variant -> {OUT}")

    summarize(df, "ALL LONGS (stocks + macro)")
    df["asset"] = np.where(df["channel"] == "detect_seasonal", "stock", "macro")
    for a in ["stock", "macro"]:
        summarize(df[df.asset == a], f"{a.upper()} longs")
    for h in [5, 10, 21]:
        summarize(df[df.h == h], f"{h}d horizon")
    summarize(df[df.cycle != 2], "EX-MIDTERM")

    # What did the stop cost? Ticket-stopped trades: stop outcome vs held-to-time
    tk = df[df.variant == "ticket"]
    stopped = tk[tk.exit_type == "Stop"]
    if len(stopped):
        held = stopped["held_R_sqrt"] * stopped["risk_sqrt"] / stopped["risk_stop"]
        print(f"\n=== Stop decomposition (ticket variant, N={len(stopped)} stopped, "
              f"{100*len(stopped)/len(tk):.1f}% of trades) ===")
        print(f"realized at stop:   avgR {stopped['R'].mean():+.3f} (in stop-R units)")
        print(f"if held to time:    avgR {held.mean():+.3f} (same units) | "
              f"{100*(held > stopped['R']).mean():.0f}% would have exited better | "
              f"{100*(held > 0).mean():.0f}% would have finished positive")
    # Tail exposure of the no-stop book
    ns = df[df.variant == "nostop_sqrt"]
    mae_r = ns["mae_atr"] / np.sqrt(ns["h"] / 5.0)
    print(f"\n=== No-stop tail exposure (MAE in sqrt-time R units) ===")
    print(f"median {mae_r.median():.2f} | p90 {mae_r.quantile(0.10):.2f} | "
          f"p99 {mae_r.quantile(0.01):.2f} | worst {mae_r.min():.2f}")
    print(f"worst single trades ($): {ns['dollars'].nsmallest(5).round(0).tolist()}")


if __name__ == "__main__":
    main()
