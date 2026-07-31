"""In-market Sharpe: h21/no-target vs h5/2ATR-target weak-close variants.

Daily MTM per trade (close-to-close; exit day marks to the exit price),
equal-weight average across concurrent positions. In-market Sharpe uses
only days with >=1 open position; calendar Sharpe spreads the same PnL
over all trading days.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from monthly_weak_close_mr import load_data, month_signals, run_trades
from monthly_weak_close_mr_regime import sma10_map


def trade_daily_rets(df: pd.DataFrame, tr: dict) -> pd.Series:
    idx = df.index
    e_i, x_i = idx.get_loc(tr["entry_day"]), idx.get_loc(tr["exit_day"])
    if x_i <= e_i:
        return pd.Series(dtype=float)
    marks = [tr["entry"]] + df["Close"].iloc[e_i + 1:x_i].tolist() + [tr["exit"]]
    marks = pd.Series(marks, index=idx[e_i:x_i + 1])
    return (marks / marks.shift() - 1).dropna()


def variant_series(data: dict, hold: int, tgt: float | None,
                   ma_filter: bool) -> pd.Series:
    daily: dict[pd.Timestamp, list[float]] = {}
    for tk in ["SPY", "QQQ"]:
        df = data[tk]
        above = sma10_map(df)
        for tr in run_trades(df, month_signals(df, 0.15), "close", hold, tgt):
            per = pd.Timestamp(tr["sig_day"]).to_period("M")
            if ma_filter and not bool(above.get(per, False)):
                continue
            for d, r in trade_daily_rets(df, tr).items():
                daily.setdefault(d, []).append(r)
    return pd.Series({d: np.mean(v) for d, v in daily.items()}).sort_index()


def main() -> None:
    data = load_data()
    n_days = len(data["SPY"].index)
    rows = []
    for label, hold, tgt in [("h21 no tgt", 21, None), ("h5 tgt2ATR", 5, 2.0)]:
        for ma in [False, True]:
            s = variant_series(data, hold, tgt, ma)
            sharpe_im = s.mean() / s.std(ddof=1) * np.sqrt(252)
            cal = s.reindex(data["SPY"].index).fillna(0.0)
            sharpe_cal = cal.mean() / cal.std(ddof=1) * np.sqrt(252)
            eq = (1 + s).cumprod()
            dd = (eq / eq.cummax() - 1).min()
            rows.append({
                "variant": label + (" +MA" if ma else ""),
                "in_mkt_days": len(s),
                "time_in_mkt%": round(100 * len(s) / n_days, 1),
                "Sharpe_inmkt": round(float(sharpe_im), 2),
                "Sharpe_cal": round(float(sharpe_cal), 2),
                "ann_ret_inmkt%": round(100 * s.mean() * 252, 1),
                "maxDD_inmkt%": round(100 * float(dd), 2),
            })
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
