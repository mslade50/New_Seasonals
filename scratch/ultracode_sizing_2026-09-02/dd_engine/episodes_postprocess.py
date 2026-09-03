"""Fold drawdown-episode tables (top 6 per window per scenario, with
per-strategy attribution) into engine_partial_replay.json. Read-only on the
trade parquets the main runner wrote."""
import json
import os
import sys

import numpy as np
import pandas as pd

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

OUT = os.path.join(_HERE, "engine_partial_replay.json")
ACCOUNT = 750000.0


def per_strategy_daily(code, md, index):
    import pages.strat_backtester as sb
    import build_trade_ledger as btl
    tr = pd.read_parquet(os.path.join(_HERE, "trades", f"{code}_main.parquet"))
    out = {}
    for strat, g in tr.groupby("Strategy"):
        out[strat] = sb.get_daily_mtm_series(g, md, start_date=btl.BT_START).reindex(index).fillna(0.0)
    return out


def episodes(book, per, start, n=6):
    s = book[book.index >= pd.Timestamp(start)]
    cum = s.cumsum()
    dd = cum - cum.cummax()
    work = dd.copy()
    eps = []
    for _ in range(n):
        t = work.idxmin()
        if work[t] >= 0:
            break
        pk = cum.loc[:t].idxmax()
        contrib = {k: float(v.loc[pk:t].sum()) for k, v in per.items()}
        top = sorted(contrib.items(), key=lambda kv: kv[1])[:3]
        eps.append({"peak": str(pk.date()), "trough": str(t.date()),
                    "dd_pct": float(work[t] / ACCOUNT * 100),
                    "top_contrib": [(k, round(v)) for k, v in top]})
        work.loc[pk:t] = 0
    return eps


def main():
    import build_trade_ledger as btl
    r = json.load(open(OUT))
    codes = list(r["scenarios"].keys())
    tickers = set()
    for c in codes:
        tickers.update(pd.read_parquet(os.path.join(_HERE, "trades", f"{c}_main.parquet"))["Ticker"].unique())
    md = btl.load_data(tickers)
    index = pd.date_range(btl.BT_START, pd.Timestamp.today().normalize(), freq="B")
    for c in codes:
        per = per_strategy_daily(c, md, index)
        book = sum(per.values())
        for wname, wstart in r["meta"]["windows"].items():
            r["scenarios"][c]["windows"][wname]["dd_episodes_top6"] = episodes(book, per, wstart)
        # the Jun-Jul 2026 span specifically
        w = book[(book.index >= "2026-06-01") & (book.index <= "2026-07-31")]
        r["scenarios"][c]["junjul_2026"] = {
            "book_pnl": float(w.sum()),
            "worst_21d_pct": float(w.rolling(21).sum().min() / ACCOUNT * 100),
            "olv_pnl": float(per.get("Oversold Low Volume", pd.Series(dtype=float)).loc["2026-06-01":"2026-07-31"].sum()),
            "olv_worst_21d_pct": float(per["Oversold Low Volume"].loc["2026-06-01":"2026-07-31"].rolling(21).sum().min() / ACCOUNT * 100)
            if "Oversold Low Volume" in per else None,
        }
        print(c, "2016-07+ episodes:")
        for e in r["scenarios"][c]["windows"]["2016-07+"]["dd_episodes_top6"]:
            print(f"   {e['peak']}..{e['trough']} {e['dd_pct']:6.2f}%  {e['top_contrib']}")
        print("   junjul_2026:", r["scenarios"][c]["junjul_2026"])
    json.dump(r, open(OUT, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("updated", OUT)


if __name__ == "__main__":
    main()
