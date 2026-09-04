"""ENTERTAIN-THE-IDEA probe (tiny sample, exploratory — NOT a rule proposal).

Question (McKinley, 2026-07-22): for OLV (Oversold Low Volume) signals, when a
ticker fires >= 3 signals inside a 2-week window, buy the 3rd firing and hold
6 months with NO stop and an 8-ATR target. What does the return profile look
like?

Interpretation: "3 instances in a 2-week span" = 3 OLV signals on the SAME
ticker within a trailing 14 calendar days (a name in a persistent quiet bleed
that keeps re-triggering). "Buy the 3rd" = enter on the signal at which the
trailing-14d count first reaches 3, then reset (need 3 fresh signals to fire
again). Book-wide clustering is a different question, noted but not run here.

Signal source: OLV Signal Dates from data/backtest_trades_full.parquet (the
prod ledger). These are FILLED signals, so clusters are counted on fills, not
raw signals — a minor undercount (a signal that fired but never filled is
invisible). Fine for an exploratory read; a raw-signal regeneration off
filters.py is the natural next step if the profile is interesting.

Hold sim is FRESH (does not reuse ledger exits): next-open entry, target =
entry + 8*ATR (14d TR mean at signal date), no stop, time exit at 126 td.
Basis = master_prices adjusted OHLCV (total-return; target is a relative level
recomputed from the same series, so scale-invariant per CLAUDE.md).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STRAT = "Oversold Low Volume"
WINDOW_CAL_DAYS = 14      # "2 week" span
HOLD_TD = 126            # ~6 months
TGT_ATR = 8.0
STOP_ATR_UNIT = 1.25     # OLV sizing risk unit — for expressing R only

# ---- OLV signals from the prod ledger ----
led = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
olv = led[led["Strategy"] == STRAT].copy()
olv["Signal Date"] = pd.to_datetime(olv["Signal Date"]).dt.normalize()
sig = (olv[["Ticker", "Signal Date"]]
       .drop_duplicates()
       .sort_values(["Ticker", "Signal Date"])
       .reset_index(drop=True))
print(f"OLV unique (ticker, signal-date) firings in ledger: {len(sig)}  "
      f"tickers: {sig['Ticker'].nunique()}  "
      f"span: {sig['Signal Date'].min().date()}..{sig['Signal Date'].max().date()}")

# ---- price frames + ATR ----
tickers = sorted(set(sig["Ticker"]) | {"SPY"})
px = pd.read_parquet(ROOT / "data" / "master_prices.parquet",
                     filters=[("ticker", "in", tickers)])
px["date"] = pd.to_datetime(px["date"]).dt.normalize()
frames: dict[str, pd.DataFrame] = {}
for tkr, g in px.groupby("ticker"):
    g = g.sort_values("date").drop_duplicates("date").set_index("date")
    tr = pd.concat([g["High"] - g["Low"],
                    (g["High"] - g["Close"].shift(1)).abs(),
                    (g["Low"] - g["Close"].shift(1)).abs()], axis=1).max(axis=1)
    g["ATR"] = tr.rolling(14).mean()
    frames[tkr] = g
spy = frames.get("SPY")


def spy_fwd(entry_date, exit_date):
    if spy is None:
        return np.nan
    idx = spy.index
    try:
        e = idx[idx.get_indexer([entry_date], method="bfill")[0]]
        x = idx[idx.get_indexer([exit_date], method="ffill")[0]]
    except Exception:
        return np.nan
    if pd.isna(e) or pd.isna(x):
        return np.nan
    return spy.loc[x, "Close"] / spy.loc[e, "Close"] - 1.0


def sim(tkr, signal_date):
    """Next-open entry, 8-ATR target, no stop, 126-td time exit."""
    df = frames.get(tkr)
    if df is None or signal_date not in df.index:
        return None
    sidx = df.index.get_loc(signal_date)
    if sidx + 1 >= len(df):
        return None
    atr = df.iloc[sidx]["ATR"]
    if pd.isna(atr) or atr <= 0:
        return None
    entry_idx = sidx + 1
    entry = df.iloc[entry_idx]["Open"]
    if pd.isna(entry) or entry <= 0:
        return None
    tgt = entry + TGT_ATR * atr
    last = min(entry_idx + HOLD_TD, len(df) - 1)
    exit_idx, exit_px, hit = last, df.iloc[last]["Close"], False
    lowest = entry
    for ci in range(entry_idx + 1, last + 1):
        r = df.iloc[ci]
        lowest = min(lowest, r["Low"])
        if r["High"] >= tgt:
            exit_idx, exit_px, hit = ci, tgt, True
            break
    ed, xd = df.index[entry_idx], df.index[exit_idx]
    return {
        "Ticker": tkr, "Signal Date": signal_date, "Entry Date": ed,
        "Exit Date": xd, "entry": entry, "exit": exit_px, "atr": atr,
        "ret_pct": (exit_px / entry - 1.0) * 100,
        "move_atr": (exit_px - entry) / atr,
        "R_1.25atr": (exit_px - entry) / (STOP_ATR_UNIT * atr),
        "hold_td": exit_idx - entry_idx,
        "hit_tgt": hit,
        "mae_atr": (entry - lowest) / atr,        # max adverse excursion
        "spy_ret_pct": spy_fwd(ed, xd) * 100,
    }


# ---- per-signal ±window cluster count (partition base rates) ----
cnt = np.zeros(len(sig), dtype=int)
dvals = sig["Signal Date"].values.astype("datetime64[D]")
tvals = sig["Ticker"].values
for i in range(len(sig)):
    m = (tvals == tvals[i]) & (np.abs((dvals - dvals[i]).astype(int)) <= WINDOW_CAL_DAYS)
    cnt[i] = int(m.sum())
sig["win_cnt"] = cnt

# ---- reset-based "3rd of a fresh burst" entries (+ 1st of that burst) ----
third_entries, first_of_cluster = [], []
for tkr, g in sig.groupby("Ticker"):
    recent: list[pd.Timestamp] = []
    for d in g["Signal Date"]:
        recent = [x for x in recent if (d - x).days <= WINDOW_CAL_DAYS]
        recent.append(d)
        if len(recent) >= 3:
            third_entries.append((tkr, d))
            first_of_cluster.append((tkr, recent[0]))
            recent = []
third_set = set(third_entries)
print(f"'3rd-of-burst' entries: {len(third_entries)}  "
      f"(tickers: {len(set(t for t, _ in third_entries))})")


def run(pairs):
    rows = [sim(t, d) for t, d in pairs]
    return pd.DataFrame([r for r in rows if r is not None])


def profile(df, label):
    if len(df) == 0:
        print(f"  {label:28s} N=  0"); return
    r = df["ret_pct"]
    print(f"  {label:28s} N={len(df):4d}  win={(r>0).mean():5.1%}  "
          f"avgRet={r.mean():+6.1f}%  medRet={r.median():+6.1f}%  "
          f"totRet={r.sum():+7.0f}%  hitTgt={df['hit_tgt'].mean():4.0%}  "
          f"avgHold={df['hold_td'].mean():5.0f}td  avgMAE={df['mae_atr'].mean():4.1f}ATR  "
          f"avgR={df['R_1.25atr'].mean():+5.2f}  vsSPY={ (r - df['spy_ret_pct']).mean():+6.1f}%")


def matured(df):
    # a trade is matured only if it hit target or ran the full 126 td;
    # otherwise it was force-closed at the data edge (2026-07-20) — not a
    # real 6-month outcome.
    return df[df["hit_tgt"] | (df["hold_td"] >= HOLD_TD)]


all_pairs = list(sig[["Ticker", "Signal Date"]].itertuples(index=False, name=None))
third_df = run(third_entries)
first_df = run(first_of_cluster)
clus_pairs = list(sig.loc[sig["win_cnt"] >= 3, ["Ticker", "Signal Date"]].itertuples(index=False, name=None))
iso_pairs = list(sig.loc[sig["win_cnt"] < 3, ["Ticker", "Signal Date"]].itertuples(index=False, name=None))

print("\n6-month hold / 8-ATR target / NO stop / next-open entry")
print("=" * 118)
profile(run(all_pairs),  "ALL OLV signals")
profile(run(iso_pairs),  "isolated (<3 in 2wk)")
profile(run(clus_pairs), "any clustered sig (>=3)")
profile(first_df,        "1st of each burst")
profile(third_df,        ">>> 3rd of each burst <<<")
print("-- matured only (hit target OR full 126td; drops data-edge truncations) --")
profile(matured(run(all_pairs)), "ALL OLV (matured)")
profile(matured(third_df),       ">>> 3rd of burst (matured) <<<")

# ---- the headline set, itemized ----
if len(third_df):
    print("\n3rd-of-burst entries, itemized:")
    show = third_df.sort_values("Signal Date")[
        ["Ticker", "Signal Date", "Entry Date", "Exit Date", "ret_pct",
         "move_atr", "hit_tgt", "hold_td", "mae_atr", "spy_ret_pct"]].copy()
    show["Signal Date"] = show["Signal Date"].dt.date
    show["Entry Date"] = show["Entry Date"].dt.date
    show["Exit Date"] = show["Exit Date"].dt.date
    for c in ["ret_pct", "move_atr", "mae_atr", "spy_ret_pct"]:
        show[c] = show[c].round(1)
    print(show.to_string(index=False))
    print("\nby year (3rd-of-burst):")
    yr = third_df.groupby(third_df["Signal Date"].dt.year).agg(
        n=("ret_pct", "size"), avgRet=("ret_pct", "mean"),
        win=("ret_pct", lambda s: (s > 0).mean()), hitTgt=("hit_tgt", "mean"))
    print(yr.round(2).to_string())
