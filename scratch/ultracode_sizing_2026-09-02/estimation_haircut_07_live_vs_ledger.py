"""Live staged-signal record vs the ledger.

Sources (read-only snapshots pulled by estimation_haircut_02_sheets_pull.py):
  sheets_Signals.csv      Trade_Signals_Log sheet1: every signal STAGED live since
                          2026-03-26, with Fill_Status (verify_fills.py marks it from
                          yfinance daily bars, so it is a modeled fill, not a broker fill)
  sheets_Trade_Journal.csv 27 rows of ACTUAL IBKR fills from Feb 2026 (pre-current book)
  sheets_Manual_Journal.csv hand-graded outcomes, Feb 2026
What can be measured honestly:
  (a) parity: does a live-staged (Strategy, Ticker, Signal Date) appear in the ledger
      as a filled trade, and does the Sheets Fill_Status agree with the ledger;
  (b) live staged fill rate by strategy (FILLED / (FILLED+EXPIRED));
  (c) modeled ledger R on the signals that were actually staged live (the realized
      2026 sample, restricted to what the live pipeline actually put in front of IBKR);
  (d) broker fill evidence from the Feb-2026 journal: limit orders fill at the limit
      (0 bps slippage) or miss; miss rate.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:/Users/McKinley Slade/dev/New_Seasonals")
OUT = ROOT / "scratch/ultracode_sizing_2026-09-02"
res: dict = {}

sig = pd.read_csv(OUT / "sheets_Signals.csv")
sig["Date"] = pd.to_datetime(sig["Date"], errors="coerce")
sig = sig[sig["Date"].notna()].copy()
sig["Ticker"] = sig["Ticker"].astype(str).str.upper().str.strip()
led = pd.read_parquet(ROOT / "data/backtest_trades_full.parquet")
led = led[led["PnL_flat_750k"].notna()].copy()
led["Signal Date"] = pd.to_datetime(led["Signal Date"])
alias = {"^GSPC": "SPY", "^NDX": "QQQ"}
led["tk"] = led["Ticker"].map(lambda t: alias.get(t, t))
led_keys = led.groupby(["Strategy", "tk", "Signal Date"]).agg(R=("R_Multiple", "mean"), n=("R_Multiple", "size"), entry=("Entry Price", "first"), exit_type=("Exit Type", "first")).reset_index()

live_names = set(led["Strategy"])
sig["in_book"] = sig["Strategy_Name"].isin(live_names)
res["staged_rows"] = int(len(sig))
res["staged_date_range"] = [str(sig["Date"].min().date()), str(sig["Date"].max().date())]
res["staged_rows_current_book_strats"] = int(sig["in_book"].sum())
res["staged_rows_retired_strats"] = int((~sig["in_book"]).sum())
s = sig[sig["in_book"]].copy()
s["status"] = s["Fill_Status"].fillna("").astype(str).str.upper().str.strip()
m = s.merge(led_keys, left_on=["Strategy_Name", "Ticker", "Date"], right_on=["Strategy", "tk", "Signal Date"], how="left")
m["in_ledger"] = m["R"].notna()

# (a) parity
by_status = m.groupby("status").agg(n=("in_ledger", "size"), in_ledger=("in_ledger", "mean"))
print("staged rows by Sheets fill status vs ledger presence:\n", by_status)
res["parity_by_status"] = {k: {"n": int(v["n"]), "share_in_ledger": float(v["in_ledger"])} for k, v in by_status.iterrows()}
# status blank = verify_fills never graded the row (mostly the Overflow tier / early rows)
graded = m[m["status"].isin(["FILLED", "EXPIRED"])]
tab = pd.crosstab(graded["status"], graded["in_ledger"])
print("\ncrosstab (graded rows):\n", tab)
res["crosstab_graded"] = {str(i): {str(c): int(tab.loc[i, c]) for c in tab.columns} for i in tab.index}
disagree = graded[(graded["status"] == "FILLED") != graded["in_ledger"]]
res["disagreements"] = disagree[["Date", "Strategy_Name", "Ticker", "status", "in_ledger", "Scan_Source"]].astype(str).to_dict("records")
print("\ndisagreements:\n", disagree[["Date", "Strategy_Name", "Ticker", "status", "in_ledger", "Scan_Source", "Limit_Price", "Fill_Price"]].to_string())

# (b) live staged fill rate by strategy (graded rows only)
fr = graded.groupby("Strategy_Name")["status"].apply(lambda x: (x == "FILLED").mean())
fn = graded.groupby("Strategy_Name").size()
res["live_fill_rate_by_strategy"] = {k: {"fill_rate": float(fr[k]), "n_graded": int(fn[k])} for k in fr.index}
print("\nlive staged fill rate (graded rows):\n", pd.DataFrame({"fill_rate": fr, "n": fn}))
# all staged rows (incl. ungraded) -> ledger presence as the fill proxy
fr2 = m.groupby("Strategy_Name")["in_ledger"].agg(["mean", "size"])
res["ledger_presence_by_strategy_all_staged"] = {k: {"share_in_ledger": float(v["mean"]), "n_staged": int(v["size"])} for k, v in fr2.iterrows()}
print("\nledger presence for ALL staged rows (fill proxy):\n", fr2)

# (c) ledger R on live-staged signals
lm = m[m["in_ledger"]]
res["ledger_R_on_live_staged"] = {"N": int(len(lm)), "avgR": float(lm["R"].mean()), "sumR": float(lm["R"].sum()),
                                  "by_strategy": {k: {"N": int(len(g)), "avgR": float(g["R"].mean())} for k, g in lm.groupby("Strategy_Name")}}
print("\nledger R on live-staged signals:", res["ledger_R_on_live_staged"]["N"], round(lm["R"].mean(), 3))
print(pd.DataFrame(res["ledger_R_on_live_staged"]["by_strategy"]).T)
# same window, everything in the ledger (were there ledger trades the live pipeline never staged?)
w0, w1 = sig["Date"].min(), sig["Date"].max()
lw = led[(led["Signal Date"] >= w0) & (led["Signal Date"] <= w1)]
lk = lw.groupby(["Strategy", "tk", "Signal Date"]).size().reset_index()
staged_keys = set(zip(m["Strategy_Name"], m["Ticker"], m["Date"]))
lk["staged"] = [(a, b, c) in staged_keys for a, b, c in zip(lk["Strategy"], lk["tk"], lk["Signal Date"])]
res["ledger_trades_in_window"] = {"N_keys": int(len(lk)), "share_staged_live": float(lk["staged"].mean()),
                                  "unstaged_by_strategy": lk[~lk["staged"]].groupby("Strategy").size().to_dict()}
print("\nledger signal-keys in the live window:", len(lk), "share staged live:", round(lk["staged"].mean(), 3))
print("unstaged (ledger has it, live never staged):", res["ledger_trades_in_window"]["unstaged_by_strategy"])
# fill price agreement where both exist
both = m[m["in_ledger"] & (m["status"] == "FILLED")].copy()
both["Fill_Price"] = pd.to_numeric(both["Fill_Price"], errors="coerce")
both["px_diff_bps"] = (both["Fill_Price"] / both["entry"] - 1) * 1e4
res["fill_price_vs_ledger_entry_bps"] = {"N": int(both["px_diff_bps"].notna().sum()), "mean": float(both["px_diff_bps"].mean()), "median": float(both["px_diff_bps"].median()), "abs_gt_20bps": int((both["px_diff_bps"].abs() > 20).sum())}
print("\nSheets fill price vs ledger entry (bps):", res["fill_price_vs_ledger_entry_bps"])

# (d) broker journal (Feb 2026)
tj = pd.read_csv(OUT / "sheets_Trade_Journal.csv")
tj["fill_status"] = tj["fill_status"].astype(str).str.lower()
byt = tj.groupby("expected_order_type")["fill_status"].apply(lambda x: (x == "filled").mean())
res["broker_journal_feb2026"] = {"rows": int(len(tj)), "filled": int((tj["fill_status"] == "filled").sum()),
                                 "fill_rate_by_type": byt.to_dict(),
                                 "slippage_bps_filled": tj.loc[tj["fill_status"] == "filled", "slippage_bps"].astype(float).describe().to_dict(),
                                 "commission_per_share_filled": float((tj.loc[(tj["fill_status"] == "filled") & (tj["actual_shares"] > 0), "commission"] / tj.loc[(tj["fill_status"] == "filled") & (tj["actual_shares"] > 0), "actual_shares"]).mean())}
print("\nbroker journal:", res["broker_journal_feb2026"])
mj = pd.read_csv(OUT / "sheets_Manual_Journal.csv")
res["manual_journal_results"] = mj["Result"].fillna("").value_counts().to_dict()
print("manual journal results:", res["manual_journal_results"])
(OUT / "estimation_haircut_live_vs_ledger.json").write_text(json.dumps(res, indent=1, default=str))
