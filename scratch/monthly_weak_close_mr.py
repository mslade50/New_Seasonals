"""Monthly weak-close mean reversion — SPY/QQQ/TLT.

Signal: month closes in the lower X% of its monthly high-low range
(fires on the month's last trading day). Buy, hold N trading days,
optional target at K*ATR above entry. No stop.

Entry variants:
  close   — buy the signal-day close
  t1open  — buy the next session's open
  limit   — limit at signal close - 0.25*ATR, GTC 2 sessions (T+1..T+2)

Conventions match the book engine where sensible:
  - Wilder-14 ATR at the signal day
  - entry-day targets never credited (checked from entry_idx+1)
  - limit fill = min(Open, limit) on the first bar whose Low touches it
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TICKERS = ["SPY", "QQQ", "TLT"]
PARQUET = "data/master_prices.parquet"


def wilder_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def load_data() -> dict[str, pd.DataFrame]:
    raw = pd.read_parquet(PARQUET)
    out = {}
    for t in TICKERS:
        df = (raw[raw.ticker == t]
              .set_index("date")[["Open", "High", "Low", "Close"]]
              .sort_index())
        df["ATR"] = wilder_atr(df)
        out[t] = df
    return out


def month_signals(df: pd.DataFrame, thresh: float) -> pd.DataFrame:
    g = df.groupby(df.index.to_period("M"))
    m = pd.DataFrame({
        "hi": g["High"].max(),
        "lo": g["Low"].min(),
        "close": g["Close"].last(),
        "last_day": g.apply(lambda x: x.index[-1]),
    })
    m["pos"] = (m["close"] - m["lo"]) / (m["hi"] - m["lo"])
    m = m.iloc[:-1] if _month_incomplete(df) else m
    return m[m["pos"] <= thresh]


def _month_incomplete(df: pd.DataFrame) -> bool:
    last = df.index[-1]
    return last != (last + pd.offsets.BMonthEnd(0))


def run_trades(df: pd.DataFrame, sigs: pd.DataFrame, entry_mode: str,
               hold: int, tgt_atr: float | None,
               limit_atr: float = 0.25, fill_window: int = 2) -> list[dict]:
    idx = df.index
    trades = []
    for _, s in sigs.iterrows():
        sig_day = s["last_day"]
        i = idx.get_loc(sig_day)
        atr = df["ATR"].iloc[i]
        if np.isnan(atr) or atr <= 0:
            continue

        if entry_mode == "close":
            e_i, e_px = i, df["Close"].iloc[i]
        elif entry_mode == "t1open":
            if i + 1 >= len(df):
                continue
            e_i, e_px = i + 1, df["Open"].iloc[i + 1]
        elif entry_mode == "limit":
            lim = df["Close"].iloc[i] - limit_atr * atr
            e_i = e_px = None
            for j in range(i + 1, min(i + 1 + fill_window, len(df))):
                if df["Low"].iloc[j] <= lim:
                    e_i, e_px = j, min(df["Open"].iloc[j], lim)
                    break
            if e_i is None:
                continue
        else:
            raise ValueError(entry_mode)

        x_i = min(e_i + hold, len(df) - 1)
        exit_px, exit_kind = df["Close"].iloc[x_i], "time"
        if tgt_atr is not None:
            tgt = e_px + tgt_atr * atr
            for j in range(e_i + 1, x_i + 1):
                if df["High"].iloc[j] >= tgt:
                    exit_px, exit_kind, x_i = tgt, "target", j
                    break
        ret = exit_px / e_px - 1
        trades.append({
            "sig_day": sig_day, "entry_day": idx[e_i], "exit_day": idx[x_i],
            "entry": e_px, "exit": exit_px, "ret": ret,
            "r_atr": (exit_px - e_px) / atr, "exit_kind": exit_kind,
            "pos": s["pos"],
        })
    return trades


def summarize(trades: list[dict], label: str, n_sigs: int) -> dict:
    if not trades:
        return {"variant": label, "signals": n_sigs, "fills": 0}
    t = pd.DataFrame(trades)
    wins = (t["ret"] > 0).mean()
    gross_up = t.loc[t.ret > 0, "ret"].sum()
    gross_dn = -t.loc[t.ret < 0, "ret"].sum()
    return {
        "variant": label, "signals": n_sigs, "fills": len(t),
        "win%": round(100 * wins, 1),
        "avg_ret%": round(100 * t["ret"].mean(), 3),
        "med_ret%": round(100 * t["ret"].median(), 3),
        "tot_ret%": round(100 * t["ret"].sum(), 1),
        "avg_R": round(t["r_atr"].mean(), 3),
        "PF": round(gross_up / gross_dn, 2) if gross_dn > 0 else np.inf,
        "worst%": round(100 * t["ret"].min(), 2),
        "tgt_hit%": round(100 * (t["exit_kind"] == "target").mean(), 1),
    }


def baseline_5d(data: dict[str, pd.DataFrame], hold: int) -> float:
    rets = []
    for df in data.values():
        r = df["Close"].shift(-hold) / df["Close"] - 1
        rets.append(r.dropna())
    return float(pd.concat(rets).mean())


def main() -> None:
    data = load_data()
    thresh, hold, tgt = 0.15, 5, 2.0

    print(f"=== Base config: thresh={thresh}, hold={hold}d, target={tgt} ATR ===")
    all_rows = []
    per_ticker_trades: dict[str, dict[str, list]] = {}
    for mode in ["close", "t1open", "limit"]:
        agg, n_sigs = [], 0
        for tk, df in data.items():
            sigs = month_signals(df, thresh)
            n_sigs += len(sigs)
            tr = run_trades(df, sigs, mode, hold, tgt)
            per_ticker_trades.setdefault(mode, {})[tk] = tr
            agg += tr
        all_rows.append(summarize(agg, mode, n_sigs))
    print(pd.DataFrame(all_rows).to_string(index=False))
    print(f"\nBaseline unconditional {hold}d fwd return (all 3 tickers): "
          f"{100 * baseline_5d(data, hold):.3f}%")

    print("\n=== Per-ticker (t1open, base config) ===")
    rows = []
    for tk, tr in per_ticker_trades["t1open"].items():
        n_sigs = len(month_signals(data[tk], thresh))
        rows.append(summarize(tr, tk, n_sigs))
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== Per-year avg ret% (t1open, all tickers pooled) ===")
    t = pd.DataFrame([x for tr in per_ticker_trades["t1open"].values() for x in tr])
    t["yr"] = pd.to_datetime(t["sig_day"]).dt.year
    yr = t.groupby("yr")["ret"].agg(["count", "mean", "sum"])
    yr[["mean", "sum"]] = (100 * yr[["mean", "sum"]]).round(2)
    print(yr.to_string())

    print("\n=== Sensitivity: threshold x hold (t1open, target=2 ATR) — avg_ret% / N ===")
    grid = {}
    for th in [0.10, 0.15, 0.20, 0.25]:
        row = {}
        for hd in [3, 5, 10, 21]:
            agg = []
            for tk, df in data.items():
                agg += run_trades(df, month_signals(df, th), "t1open", hd, tgt)
            tt = pd.DataFrame(agg)
            row[f"h{hd}"] = f"{100 * tt.ret.mean():+.2f} ({len(tt)})" if len(tt) else "-"
        grid[f"pos<={th:.2f}"] = row
    print(pd.DataFrame(grid).T.to_string())

    print("\n=== Target sensitivity (t1open, thresh=0.15, hold=5) ===")
    rows = []
    for tg in [None, 1.0, 1.5, 2.0, 3.0]:
        agg, n_sigs = [], 0
        for tk, df in data.items():
            sigs = month_signals(df, thresh)
            n_sigs += len(sigs)
            agg += run_trades(df, sigs, "t1open", hold, tg)
        rows.append(summarize(agg, f"tgt={tg}", n_sigs))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
