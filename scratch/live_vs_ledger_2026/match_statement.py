"""Match the primary account's IBKR activity statement to the backtest ledger.

Closes the measurement half of gap 4 in the 2026-09-02 sizing due diligence:
"no live R series exists". The broker ring only reaches back 14 days, so the
statement is the only record of Jan-Aug executions. It carries no order
reference (IBKR offers that field on single-day flex queries only), so the
match is KEYLESS on symbol + session + side.

Two aggregations make the sides comparable:

* The ledger books OVS scale-outs as TWO tranche rows per fill (near 40% at
  1 ATR, far 60% at 2 ATR), which exit on different sessions. A live fill is
  one order. So the ledger is collapsed to POSITION level -- one row per
  (strategy, ticker, entry date, direction) -- and the live exit is the
  share-weighted average across whichever sessions the tranches exited on.
* The statement is order-level, but a position can fill through several
  orders in a session (and a hand trim adds more), so live fills collapse to
  one weighted-average price per (symbol, session, side).

Reported deltas isolate EXECUTION, not edge: live prices are scored against
the ledger's own risk unit (ATR x stop_atr), so a difference is slippage or a
discretionary override, never a different model.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
STATEMENT = Path.home() / "Downloads" / "U16584234_20260101_20260902.csv"
LEDGER = ROOT / "data" / "backtest_trades_full.parquet"
CUTOFF = "2026-09-02"


def _num(x) -> float:
    try:
        return float(str(x).replace(",", "").strip())
    except ValueError:
        return float("nan")


def load_statement(path: Path) -> pd.DataFrame:
    rows = list(csv.reader(path.open(encoding="utf-8-sig")))
    header = None
    out = []
    for r in rows:
        if not r or r[0] != "Trades" or len(r) < 3:
            continue
        if r[1] == "Header":
            header = r[2:]
            continue
        if r[1] != "Data" or header is None or r[2] != "Order":
            continue
        rec = dict(zip(header, r[2:]))
        if rec.get("Asset Category") != "Stocks":
            continue
        out.append({
            "symbol": rec.get("Symbol", "").strip(),
            "session": rec.get("Date/Time", "")[:10],
            "qty": _num(rec.get("Quantity")),
            "price": _num(rec.get("T. Price")),
            "comm": _num(rec.get("Comm/Fee")),
            "code": rec.get("Code", ""),
        })
    df = pd.DataFrame(out)
    df["side"] = np.where(df["qty"] > 0, "BUY", "SELL")
    df["abs_qty"] = df["qty"].abs()
    return df


def session_fills(stmt: pd.DataFrame) -> pd.DataFrame:
    def agg(d):
        return pd.Series({
            "qty": d["abs_qty"].sum(),
            "vwap": np.average(d["price"], weights=d["abs_qty"]),
            "orders": len(d),
        })
    return stmt.groupby(["symbol", "session", "side"]).apply(agg, include_groups=False)


def collapse_ledger(y: pd.DataFrame) -> pd.DataFrame:
    """One row per position, carrying each tranche's exit session and weight."""
    keys = ["Strategy", "Ticker", "Entry Date", "Direction"]
    rows = []
    for key, g in y.groupby(keys, sort=False):
        sh = g["Shares_flat"].astype(float)
        w = sh / sh.sum() if sh.sum() else pd.Series(1.0 / len(g), index=g.index)
        rows.append({
            "Strategy": key[0], "Ticker": key[1], "Entry Date": key[2], "Direction": key[3],
            "tranches": len(g),
            "shares": float(sh.sum()),
            "ledger_entry_px": float(np.average(g["Entry Price"], weights=w)),
            "ledger_exit_px": float(np.average(g["Exit Price"], weights=w)),
            "ledger_R": float(np.average(g["R_Multiple"], weights=w)),
            "ATR": float(g["ATR"].iloc[0]),
            "stop_atr": float(g["stop_atr"].iloc[0]),
            "exit_legs": tuple(zip(g["Exit Date"].dt.strftime("%Y-%m-%d"), w.to_numpy())),
            "trade_ids": tuple(g["trade_id"]),
        })
    return pd.DataFrame(rows)


def main() -> int:
    stmt = load_statement(STATEMENT)
    print(f"statement: {len(stmt)} stock orders, {stmt['session'].min()} -> "
          f"{stmt['session'].max()}, {stmt['symbol'].nunique()} symbols")
    sess = session_fills(stmt)

    led = pd.read_parquet(LEDGER)
    led["Entry Date"] = pd.to_datetime(led["Entry Date"])
    led["Exit Date"] = pd.to_datetime(led["Exit Date"])
    y = led[(led["Entry Date"] >= "2026-01-01") & (led["Exit Date"] <= CUTOFF)].copy()
    pos = collapse_ledger(y)
    print(f"ledger: {len(y)} trade rows -> {len(pos)} positions entered 2026 "
          f"and closed by {CUTOFF}")

    long_ = pos["Direction"].str.lower().eq("long")
    pos["entry_side"] = np.where(long_, "BUY", "SELL")
    pos["exit_side"] = np.where(long_, "SELL", "BUY")
    pos["dir"] = np.where(long_, 1.0, -1.0)
    pos["entry_session"] = pos["Entry Date"].dt.strftime("%Y-%m-%d")

    vwap, qty = sess["vwap"], sess["qty"]
    pos["live_entry_px"] = [vwap.get((t, s, sd), np.nan) for t, s, sd
                            in zip(pos["Ticker"], pos["entry_session"], pos["entry_side"])]
    pos["live_entry_qty"] = [qty.get((t, s, sd), np.nan) for t, s, sd
                             in zip(pos["Ticker"], pos["entry_session"], pos["entry_side"])]

    # Live exit: weight each tranche's exit session by that tranche's shares.
    exits = []
    for t, sd, legs in zip(pos["Ticker"], pos["exit_side"], pos["exit_legs"]):
        px = [(vwap.get((t, s, sd), np.nan), w) for s, w in legs]
        good = [(p, w) for p, w in px if p == p]
        exits.append(np.average([p for p, _ in good], weights=[w for _, w in good])
                     if good and sum(w for _, w in good) > 0 else np.nan)
    pos["live_exit_px"] = exits

    # Two positions sharing a ticker, session and side cannot be told apart
    # without an order reference: same strategy stacking, or two strategies
    # firing the same name (what the cross-strategy clamp exists for).
    pos["ambiguous"] = pos.groupby(["Ticker", "entry_session", "entry_side"])[
        "Strategy"].transform("size") > 1

    m = pos[pos["live_entry_px"].notna() & pos["live_exit_px"].notna()].copy()
    print(f"matched both legs: {len(m)} of {len(pos)} ({100*len(m)/len(pos):.0f}%)"
          f"; ambiguous {int(m['ambiguous'].sum())}")

    rps = (m["ATR"] * m["stop_atr"]).replace(0, np.nan)
    m["live_R"] = m["dir"] * (m["live_exit_px"] - m["live_entry_px"]) / rps
    m["entry_slip_bps"] = 1e4 * m["dir"] * (m["live_entry_px"] - m["ledger_entry_px"]) / m["ledger_entry_px"]
    m["exit_slip_bps"] = 1e4 * m["dir"] * (m["ledger_exit_px"] - m["live_exit_px"]) / m["ledger_exit_px"]
    m["qty_ratio"] = m["live_entry_qty"] / m["shares"].replace(0, np.nan)
    m["diff"] = m["live_R"] - m["ledger_R"]

    clean = m[~m["ambiguous"]].copy()
    rng = np.random.default_rng(42)
    for label, s in (("all matches", m), ("unambiguous only", clean)):
        d = s["diff"].dropna().to_numpy()
        if not len(d):
            continue
        boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print()
        print(f"=== {label} (N={len(s)}) ===")
        print(f"  ledger avgR {s['ledger_R'].mean():+.3f}   live avgR {s['live_R'].mean():+.3f}"
              f"   ratio {s['live_R'].mean()/s['ledger_R'].mean():.3f}")
        print(f"  paired diff mean {d.mean():+.3f}  median {np.median(d):+.3f}"
              f"  CI95 [{lo:+.3f}, {hi:+.3f}]")
        print(f"  entry slip bps  mean {s['entry_slip_bps'].mean():+.1f}"
              f"  median {s['entry_slip_bps'].median():+.1f}   (positive = live worse)")
        print(f"  exit  slip bps  mean {s['exit_slip_bps'].mean():+.1f}"
              f"  median {s['exit_slip_bps'].median():+.1f}")
        print(f"  live/ledger shares  median {s['qty_ratio'].median():.3f}")

    print()
    print("=== by strategy (all matches) ===")
    by = m.groupby("Strategy").agg(
        n=("live_R", "size"), ledger=("ledger_R", "mean"), live=("live_R", "mean"),
        diff=("diff", "mean"), entry_bps=("entry_slip_bps", "median"),
        qty=("qty_ratio", "median"),
    ).sort_values("n", ascending=False)
    print(by.round(3).to_string())

    un = pos[pos["live_entry_px"].isna()]
    print()
    print(f"=== {len(un)} positions with no live entry fill ===")
    print(un.groupby("Strategy").size().sort_values(ascending=False).to_string())

    m.drop(columns=["exit_legs", "trade_ids"]).to_csv(HERE / "matched_positions_2026.csv", index=False)
    print()
    print(f"wrote {HERE / 'matched_positions_2026.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
