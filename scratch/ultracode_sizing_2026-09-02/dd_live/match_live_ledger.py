"""Matched set: broker fills (DO ring, Primary account) vs the ledger.

Entry legs are keyed by orderRef `SYM|BUY|Strategy|StagedDate` (staged date =
signal date + 1 bday). Exit fills are allocated to legs: a tagged exit goes to
its own leg, an untagged sell is FIFO across that symbol's open legs (these are
the manual trims). Unsold remainder is marked at the ledger's own last bar
(2026-09-01 close), which is exactly how the ledger books a still-open trade.
Live R uses the LEDGER risk-per-share (stop_atr x ATR) so the ratio isolates
price and exit differences, not the risk unit.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = Path(r"C:/Users/McKinley Slade/dev/New_Seasonals")
RNG = np.random.default_rng(11)
pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 40)

fills = pd.DataFrame(json.load(open(HERE / "do_fills.json"))["fills"])
fills["day"] = fills["time"].str[:10]
fills = fills[fills["sec_type"] == "STK"].copy()
led = pd.read_parquet(ROOT / "data/backtest_trades_full.parquet")
led["Signal Date"] = pd.to_datetime(led["Signal Date"])
led["Exit Date"] = pd.to_datetime(led["Exit Date"])
LAST_BAR = led["Exit Date"].max()
led["rps"] = led["Risk_flat_750k"] / led["Shares_flat"]
led["open_at_last_bar"] = led["Exit Date"].eq(LAST_BAR) & (led["Exit Type"] == "Time")
mp = pd.read_parquet(ROOT / "data/master_prices.parquet", columns=["ticker", "date", "Close"])
mp["date"] = pd.to_datetime(mp["date"])
sig = pd.read_csv(ROOT / "scratch/ultracode_sizing_2026-09-02/sheets_Signals.csv")
sig["Date"] = pd.to_datetime(sig["Date"], errors="coerce")
sig["Shares"] = pd.to_numeric(sig["Shares"], errors="coerce").fillna(0)
sig["Fill_Price"] = pd.to_numeric(sig["Fill_Price"], errors="coerce")

BOOK = set(led["Strategy"])


def parse_ref(r: str):
    if not isinstance(r, str) or r.count("|") < 3:
        return None
    p = r.split("|")
    sym, act, strat, d = p[0], p[1], p[2], p[3]
    if strat not in BOOK:
        return None
    try:
        ref_date = pd.Timestamp(d)
    except Exception:
        return None
    return {"sym": sym, "act": act, "strat": strat, "ref_date": ref_date,
            "signal_date": ref_date - pd.offsets.BDay(1), "extra": p[4:]}


def close_on(tk: str, d: pd.Timestamp) -> float:
    s = mp[(mp["ticker"] == tk) & (mp["date"] == d)]
    return float(s["Close"].iloc[0]) if len(s) else np.nan


out = {}
for acct_label in ("Primary (TWS)", "PA (Gateway)"):
    f = fills[fills["account_label"] == acct_label].copy()
    f["ref"] = f["order_ref"].map(parse_ref)
    # ---- entry legs from tagged BOT/SLD fills whose ref action matches the side
    legs = []
    entry_ids = set()
    for ref_str, g in f[f["ref"].notna()].groupby("order_ref"):
        r = g["ref"].iloc[0]
        if len(r["extra"]):          # olv_exit_moo refs carry extra fields: exits only
            continue
        want = "BOT" if r["act"] == "BUY" else "SLD"
        g = g[g["side"] == want]     # exit legs share the parent's orderRef; keep the entry side only
        if not len(g):
            continue
        entry_ids |= set(g["exec_id"])
        qty = g["qty"].sum()
        legs.append({"acct": acct_label, "sym": r["sym"], "strat": r["strat"], "signal_date": r["signal_date"], "ref": ref_str,
                     "entry_day": g["day"].min(), "entry_qty": float(qty), "entry_vwap": float((g["price"] * g["qty"]).sum() / qty),
                     "entry_comm": float(g["commission"].fillna(0).sum()), "n_entry_fills": int(len(g)), "source": "broker"})
    legs = pd.DataFrame(legs)
    # pre-ring entries (before 2026-08-20) that have exits inside the ring: Sheets limit as proxy, flagged
    pre = sig[(sig["Date"] >= "2026-08-14") & (sig["Date"] < "2026-08-19") & (sig["Fill_Status"] == "FILLED") & (sig["Shares"] > 0)]
    if acct_label == "Primary (TWS)":
        for _, s in pre.iterrows():
            legs = pd.concat([legs, pd.DataFrame([{"acct": acct_label, "sym": s["Ticker"], "strat": s["Strategy_Name"], "signal_date": s["Date"], "ref": f"{s['Ticker']}|BUY|{s['Strategy_Name']}|sheets",
                                                    "entry_day": s["Fill_Date"], "entry_qty": float(s["Shares"]), "entry_vwap": float(s["Fill_Price"]), "entry_comm": np.nan, "n_entry_fills": 0, "source": "sheets_limit_proxy"}])], ignore_index=True)
    legs = legs.sort_values(["sym", "entry_day"]).reset_index(drop=True)
    legs["exit_qty"] = 0.0
    legs["exit_dollars"] = 0.0
    legs["exit_days"] = [[] for _ in range(len(legs))]
    legs["exit_kinds"] = [[] for _ in range(len(legs))]
    # ---- exits: sells of those symbols, entry fills excluded
    ex = f[(~f["exec_id"].isin(entry_ids)) & f["symbol"].isin(set(legs["sym"]))].sort_values("time")
    lost = []
    for _, x in ex.iterrows():
        if x["side"] != "SLD":
            continue
        r = x["ref"]
        open_legs = legs[(legs["sym"] == x["symbol"]) & (legs["exit_qty"] < legs["entry_qty"] - 1e-9) & (legs["entry_day"] <= x["day"])]
        cands = open_legs
        kind = "manual/untagged"
        if r is not None:
            if r["act"] == "SELL" and r["extra"]:
                kind = "olv_vol_confirm_exit"
            elif r["act"] == "BUY":
                kind = "tagged_time_or_target_leg"
                tagged = open_legs[open_legs["signal_date"] == r["signal_date"]]
                if len(tagged):   # own leg first, then spill FIFO into the symbol's other open legs
                    cands = pd.concat([tagged, open_legs.drop(tagged.index)])
        qty = float(x["qty"])
        for i in cands.index:
            room = legs.at[i, "entry_qty"] - legs.at[i, "exit_qty"]
            take = min(room, qty)
            if take <= 0:
                continue
            legs.at[i, "exit_qty"] += take
            legs.at[i, "exit_dollars"] += take * float(x["price"])
            legs.at[i, "exit_days"].append(x["day"])
            legs.at[i, "exit_kinds"].append(kind)
            qty -= take
            if qty <= 1e-9:
                break
        if qty > 1e-9:
            lost.append({"sym": x["symbol"], "day": x["day"], "qty": qty, "ref": x["order_ref"]})
    if lost:
        print("sells with no open tagged leg (pre-ring entries or non-book):", lost)
    # ---- mark remainder at the ledger's last bar close
    legs["remainder"] = legs["entry_qty"] - legs["exit_qty"]
    legs["mark_px"] = [close_on(s, LAST_BAR) for s in legs["sym"]]
    legs["exit_vwap_realized"] = np.where(legs["exit_qty"] > 0, legs["exit_dollars"] / legs["exit_qty"].replace(0, np.nan), np.nan)
    legs["exit_vwap_blended"] = (legs["exit_dollars"] + legs["remainder"] * legs["mark_px"]) / legs["entry_qty"]
    legs["fully_closed"] = legs["remainder"] <= 1e-9
    legs["live_exit_kind"] = legs["exit_kinds"].map(lambda k: "+".join(sorted(set(k))) if k else "open@mark")
    legs["last_exit_day"] = legs["exit_days"].map(lambda d: max(d) if d else None)
    # ---- join the ledger
    lk = led.groupby(["Strategy", "Ticker", "Signal Date"]).agg(ledger_R=("R_Multiple", "mean"), ledger_entry=("Entry Price", "first"), ledger_exit=("Exit Price", "first"),
                                                                ledger_exit_type=("Exit Type", "first"), ledger_exit_date=("Exit Date", "first"), rps=("rps", "first"),
                                                                ledger_shares=("Shares_flat", "sum"), ledger_open=("open_at_last_bar", "first"), size_mult=("Size_Mult", "first")).reset_index()
    m = legs.merge(lk, left_on=["strat", "sym", "signal_date"], right_on=["Strategy", "Ticker", "Signal Date"], how="left")
    m["in_ledger"] = m["ledger_R"].notna()
    m["live_R"] = (m["exit_vwap_blended"] - m["entry_vwap"]) / m["rps"]
    m["live_R_realized_only"] = (m["exit_vwap_realized"] - m["entry_vwap"]) / m["rps"]
    m["entry_slip_bps"] = (m["entry_vwap"] / m["ledger_entry"] - 1) * 1e4
    m["exit_diff_bps"] = (m["exit_vwap_blended"] / m["ledger_exit"] - 1) * 1e4
    m["shares_ratio"] = m["entry_qty"] / m["ledger_shares"]
    m["comm_R"] = m["entry_comm"] / (m["rps"] * m["entry_qty"])
    cols = ["sym", "strat", "signal_date", "source", "entry_day", "entry_qty", "entry_vwap", "ledger_entry", "entry_slip_bps", "ledger_shares", "shares_ratio",
            "exit_qty", "remainder", "exit_vwap_blended", "ledger_exit", "ledger_exit_type", "ledger_exit_date", "live_exit_kind", "last_exit_day", "live_R", "ledger_R", "ledger_open", "size_mult"]
    print(f"\n===== {acct_label}: {len(m)} entry legs, {int(m['in_ledger'].sum())} matched to the ledger =====")
    print(m[cols].round(3).to_string())
    unmatched = m[~m["in_ledger"]]
    mm = m[m["in_ledger"]].copy()
    stats = {"n_legs": int(len(m)), "n_matched": int(len(mm)), "n_broker_entry": int((mm["source"] == "broker").sum()),
             "unmatched": unmatched[["sym", "strat", "signal_date"]].astype(str).to_dict("records"),
             "n_fully_closed_live": int(mm["fully_closed"].sum()), "n_ledger_open_at_last_bar": int(mm["ledger_open"].sum()),
             "live_avgR": float(mm["live_R"].mean()), "ledger_avgR": float(mm["ledger_R"].mean()),
             "live_sumR": float(mm["live_R"].sum()), "ledger_sumR": float(mm["ledger_R"].sum()),
             "paired_diff_mean": float((mm["live_R"] - mm["ledger_R"]).mean()), "paired_diff_median": float((mm["live_R"] - mm["ledger_R"]).median()),
             "entry_slip_bps_mean": float(mm["entry_slip_bps"].mean()), "entry_slip_bps_median": float(mm["entry_slip_bps"].median()),
             "entry_slip_bps_broker_only_mean": float(mm.loc[mm["source"] == "broker", "entry_slip_bps"].mean()),
             "shares_ratio_mean": float(mm["shares_ratio"].mean()), "commission_R_mean": float(mm["comm_R"].mean()),
             "exit_type_table": pd.crosstab(mm["ledger_exit_type"], mm["live_exit_kind"]).to_dict(),
             "by_strategy": {s: {"N": int(len(g)), "live_avgR": float(g["live_R"].mean()), "ledger_avgR": float(g["ledger_R"].mean())} for s, g in mm.groupby("strat")}}
    # bootstrap the ratio and the paired difference (leg-level; also by symbol-cluster because stacked legs share the path)
    lv, lg = mm["live_R"].to_numpy(), mm["ledger_R"].to_numpy()
    syms = mm["sym"].to_numpy()
    ratios, diffs = [], []
    groups = {s: np.where(syms == s)[0] for s in set(syms)}
    keys = list(groups)
    for _ in range(4000):
        pick = RNG.choice(len(keys), len(keys))
        idx = np.concatenate([groups[keys[i]] for i in pick])
        a, b = lv[idx].mean(), lg[idx].mean()
        ratios.append(a / b if abs(b) > 1e-9 else np.nan)
        diffs.append((lv[idx] - lg[idx]).mean())
    ratios, diffs = np.array(ratios), np.array(diffs)
    stats["ratio_point"] = float(lv.mean() / lg.mean()) if abs(lg.mean()) > 1e-9 else None
    stats["ratio_ci95_symcluster"] = [float(np.nanpercentile(ratios, 2.5)), float(np.nanpercentile(ratios, 97.5))]
    stats["paired_diff_ci95_symcluster"] = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
    # decomposition: entry-only effect (live entry, ledger exit) and exit-only
    mm["R_live_entry_ledger_exit"] = (mm["ledger_exit"] - mm["entry_vwap"]) / mm["rps"]
    stats["decomp"] = {"ledger": float(mm["ledger_R"].mean()), "live_entry_ledger_exit": float(mm["R_live_entry_ledger_exit"].mean()), "live_entry_live_exit": float(mm["live_R"].mean()),
                       "note": "entry effect = second minus first; exit/discretion effect = third minus second"}
    # closed-both subset: live fully closed AND ledger not open at last bar
    cb = mm[mm["fully_closed"] & ~mm["ledger_open"]]
    stats["closed_both"] = {"N": int(len(cb)), "live_avgR": float(cb["live_R"].mean()) if len(cb) else None, "ledger_avgR": float(cb["ledger_R"].mean()) if len(cb) else None}
    print(json.dumps({k: v for k, v in stats.items() if k not in ("exit_type_table", "by_strategy", "unmatched")}, indent=1))
    print("by strategy:", stats["by_strategy"])
    print("exit types (ledger rows x live cols):\n", pd.crosstab(mm["ledger_exit_type"], mm["live_exit_kind"]))
    print("unmatched legs:", stats["unmatched"])
    out[acct_label] = stats
    m.to_csv(HERE / f"matched_legs_{'primary' if 'Primary' in acct_label else 'pa'}.csv", index=False)

# ---- fill rate of staged limits inside the ring window, three graders
w = sig[(sig["Date"] >= "2026-08-19") & (sig["Date"] <= "2026-08-28") & (sig["Strategy_Name"].isin(BOOK))].copy()
w["staged_shares_gt0"] = pd.to_numeric(w["Shares"], errors="coerce").fillna(0) > 0
prim = fills[fills["account_label"] == "Primary (TWS)"]
prim_refs = {parse_ref(r)["sym"] + "|" + str(parse_ref(r)["signal_date"].date()) + "|" + parse_ref(r)["strat"] for r in prim["order_ref"].dropna() if parse_ref(r) and not parse_ref(r)["extra"]}
w["broker_filled"] = [f"{t}|{d.date()}|{s}" in prim_refs for t, d, s in zip(w["Ticker"], w["Date"], w["Strategy_Name"])]
lkeys = set(zip(led["Strategy"], led["Ticker"], led["Signal Date"]))
w["ledger_filled"] = [(s, t, d) in lkeys for s, t, d in zip(w["Strategy_Name"], w["Ticker"], w["Date"])]
w["sheets_filled"] = w["Fill_Status"].eq("FILLED")
fr = w.groupby("Strategy_Name").agg(staged=("Ticker", "size"), staged_gt0=("staged_shares_gt0", "sum"), sheets=("sheets_filled", "sum"), ledger=("ledger_filled", "sum"), broker=("broker_filled", "sum"))
print("\n===== staged rows 2026-08-19..08-28 (signals whose fill window sits inside the DO ring): three graders =====")
print(fr.to_string())
tab = pd.crosstab([w["Strategy_Name"], w["sheets_filled"]], [w["ledger_filled"], w["broker_filled"]])
print(tab.to_string())
out["fill_rate_window"] = {"per_strategy": fr.astype(int).to_dict("index"), "totals": fr.sum().astype(int).to_dict(),
                           "agreement": {"sheets_vs_broker": float((w["sheets_filled"] == w["broker_filled"]).mean()), "ledger_vs_broker": float((w["ledger_filled"] == w["broker_filled"]).mean()),
                                         "sheets_FILLED_but_no_broker_fill": int((w["sheets_filled"] & ~w["broker_filled"]).sum()), "ledger_fill_but_no_broker_fill": int((w["ledger_filled"] & ~w["broker_filled"]).sum()),
                                         "broker_fill_not_in_ledger": int((w["broker_filled"] & ~w["ledger_filled"]).sum())}}
w.to_csv(HERE / "staged_window_three_graders.csv", index=False)
(HERE / "matched_stats.json").write_text(json.dumps(out, indent=1, default=str))
