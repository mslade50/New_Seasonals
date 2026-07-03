"""Trend-following ballast sleeve — monthly rebalance pilot (2026-07-02).

Spec (validated: scratch/ultracode_research/trend-following.md + adversarial
verification + trend_prework_gates.md + scratch/tf_universe_study.py):
  - Universe: 13 liquid ETFs — equities/RE/intl core + GLD/SLV/DBC/UUP +
    TLT/LQD, no USO (see TREND_UNIVERSE note below for the full evidence).
  - Signal (month-end adjusted closes): combo = 12-1 momentum > 0 AND close >
    10-month SMA. Long/flat only — the short leg is dead (Sharpe 0.32).
  - Weights: inverse-vol slots over ALL eligible assets (>= 13 monthly closes),
    capped at 20%; OFF slots sit in cash.
  - Size: TREND_NAV_FRACTION x ACCOUNT_VALUE deployed (0.5x pilot per the
    roadmap; scale to 1.0x after 2 clean quarters).
  - Execution: signals computed on the LAST trading day's close, staged as MOO
    (TIF=OPG) for the next session's open. Next-open slippage vs same-close
    was measured at ~0.06 Sharpe on the 12-ETF variant (gate A, PASS);
    expect roughly Sharpe ~0.8, CAGR ~5.5%, maxDD ~-6.5%, corr to book ~+0.12
    for this universe on the next-open basis. This sleeve is BALLAST — it
    loses ~-0.4%/mo in high-fragility months; the frag hole is handled by
    frag_risk_bands, not this.

Runs weekdays post-close via .github/workflows/trend_sleeve.yml; exits
immediately unless today is the month's last trading day (--force overrides).
Rebalance orders are written to the 'Trend' Sheets tab; state (held shares)
persists in trend_sleeve_state.json on R2. --dry-run prints without writing.

NOTE: order_staging.py does not read the Trend tab yet — same MOO handling the
Seasonal pipeline needs (docs/seasonal_order_staging_spec.md). Until that
lands, the tab is a staged order list for manual/scripted entry.
"""
import argparse
import datetime
import json
import os
import sys

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from strategy_config import ACCOUNT_VALUE
from cache_io import download_to_local, upload_from_local

# 13-ETF universe (2026-07-02, McKinley's spec, confirmed by
# scratch/tf_universe_study.py): USO dropped (roll-decay vehicle; its removal
# costs some 2021-22 oil-trend CAGR but improves Sharpe and halves maxDD),
# TLT+LQD added (trend-filtered duration: +5.8% in 2008 vs +2.1% without,
# still +0.5% in 2022 because the trend filter sidesteps the bond bear;
# IEF/HYG add nothing over these two). Same-close study basis: Sharpe 0.87,
# maxDD -6.4%, corr to book +0.12. Tested and REJECTED: 11 SPDR sectors and
# intl singles (Sharpe 0.74-0.78, 2008/2022 flip negative — correlated equity
# slots crowd out the real diversifiers); an exhaustion scale-down overlay
# (252d & 21d pctile > 95) — Sharpe flat, trends persist past the 95th pctile.
TREND_UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "FXI", "VNQ",
                  "GLD", "SLV", "DBC", "UUP", "TLT", "LQD"]
TREND_NAV_FRACTION = 0.5     # pilot size; 1.0x after 2 clean quarters
WEIGHT_CAP = 0.20
MIN_MONTHLY_CLOSES = 13      # eligibility (needs a valid 12-month lookback)
VOL_FLOOR = 0.04             # ann. vol floor avoids inf inverse-vol weights
REBALANCE_BAND = 0.01        # skip deltas under 1% of sleeve NAV unless a flip

MASTER_PRICES = os.path.join(current_dir, "data", "master_prices.parquet")
STATE_LOCAL = os.path.join(current_dir, "data", "trend_sleeve_state.json")
STATE_R2_KEY = "trend_sleeve_state.json"
SHEET_NAME = "Trade_Signals_Log"
TAB_NAME = "Trend"


def is_last_trading_day(today=None) -> bool:
    """True when the next business day (US federal holidays) starts a new
    month. NYSE-only closures (e.g. Good Friday) can slip the rebalance one
    session — a 1-day slip is verified immaterial vs the 0.55-Sharpe
    full-month-delay bound."""
    cbd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    today = pd.Timestamp(today or pd.Timestamp.today()).normalize()
    return (today + cbd).month != today.month


def load_closes() -> pd.DataFrame:
    mp = pd.read_parquet(MASTER_PRICES)
    tcol = "ticker" if "ticker" in mp.columns else "Ticker"
    dcol = "date" if "date" in mp.columns else "Date"
    sub = mp[mp[tcol].str.upper().isin(TREND_UNIVERSE)]
    closes = sub.assign(_d=pd.to_datetime(sub[dcol]).values).pivot_table(
        index="_d", columns=tcol, values="Close")
    closes.columns = [str(c).upper() for c in closes.columns]
    return closes.sort_index()


def compute_targets(closes: pd.DataFrame) -> pd.DataFrame:
    """One row per universe ticker: signal state, weight, close. The last
    daily bar is treated as the current month-end close (the workflow only
    runs this on the month's last trading day)."""
    monthly = closes.resample("ME").last()
    # the in-progress month's row is today's close — exactly the signal basis
    m = monthly
    mom = m.shift(1) / m.shift(12) - 1.0
    ma10 = m - m.rolling(10).mean()
    eligible = m.notna().rolling(MIN_MONTHLY_CLOSES).count() >= MIN_MONTHLY_CLOSES
    combo = (mom > 0) & (ma10 > 0) & eligible

    dret = closes.pct_change()
    vol63 = (dret.rolling(63).std() * np.sqrt(252)).iloc[-1].clip(lower=VOL_FLOOR)

    last = m.index[-1]
    elig_now = eligible.loc[last]
    inv = (1.0 / vol63).where(elig_now, 0.0)
    slots = inv / inv.sum() if inv.sum() > 0 else inv
    slots = slots.clip(upper=WEIGHT_CAP)

    rows = []
    for t in TREND_UNIVERSE:
        on = bool(combo.loc[last].get(t, False))
        rows.append({
            "Ticker": t,
            "Eligible": bool(elig_now.get(t, False)),
            "Signal": "ON" if on else "OFF",
            "Weight": round(float(slots.get(t, 0.0)) if on else 0.0, 4),
            "Close": round(float(closes[t].dropna().iloc[-1]), 4) if t in closes else np.nan,
        })
    df = pd.DataFrame(rows)
    # label with the actual signal bar, not the calendar month-end label
    # (resample('ME') stamps the in-progress month as the 31st)
    df["Asof"] = str(closes.index.max().date())
    return df


def load_state() -> dict:
    download_to_local(STATE_R2_KEY, STATE_LOCAL)  # no-op without R2 creds
    if os.path.exists(STATE_LOCAL):
        try:
            with open(STATE_LOCAL, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"WARNING: state unreadable ({e}) — assuming flat book")
    return {"positions": {}}


def build_orders(targets: pd.DataFrame, state: dict) -> pd.DataFrame:
    nav = TREND_NAV_FRACTION * ACCOUNT_VALUE
    held = {k.upper(): int(v.get("shares", 0))
            for k, v in state.get("positions", {}).items()}
    orders = []
    for _, r in targets.iterrows():
        if not r.Eligible or pd.isna(r.Close) or r.Close <= 0:
            continue
        tgt_shares = int(nav * r.Weight / r.Close)
        cur = held.get(r.Ticker, 0)
        delta = tgt_shares - cur
        flip = (tgt_shares == 0) != (cur == 0)
        if delta == 0 or (not flip and abs(delta) * r.Close < REBALANCE_BAND * nav):
            continue
        orders.append({
            "Ticker": r.Ticker,
            "Action": "BUY" if delta > 0 else "SELL",
            "Quantity": abs(delta),
            "Order_Type": "MOO",
            "TIF": "OPG",
            "Target_Weight_Pct": round(r.Weight * 100, 2),
            "Target_Shares": tgt_shares,
            "Held_Shares": cur,
            "Signal": r.Signal,
            "Ref_Close": r.Close,
            "Scan_Source": "Trend",
            "Asof": r.Asof,
        })
    return pd.DataFrame(orders)


def save_state(targets: pd.DataFrame, dry_run: bool):
    nav = TREND_NAV_FRACTION * ACCOUNT_VALUE
    positions = {}
    for _, r in targets.iterrows():
        if r.Eligible and not pd.isna(r.Close) and r.Close > 0:
            shares = int(nav * r.Weight / r.Close)
            if shares > 0:
                positions[r.Ticker] = {"shares": shares, "weight": r.Weight,
                                       "ref_close": r.Close}
    state = {
        "asof": targets["Asof"].iloc[0],
        "generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "nav_fraction": TREND_NAV_FRACTION,
        "universe": TREND_UNIVERSE,
        "positions": positions,
    }
    if dry_run:
        print("\n[dry-run] state not saved")
        return
    with open(STATE_LOCAL, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    upload_from_local(STATE_LOCAL, STATE_R2_KEY)
    print(f"State saved ({len(positions)} positions) -> {STATE_LOCAL} + R2")


def write_sheet(orders: pd.DataFrame, dry_run: bool):
    if dry_run:
        print("\n[dry-run] Trend tab not written")
        return
    try:
        import gspread
        if "GCP_JSON" in os.environ:
            gc = gspread.service_account_from_dict(json.loads(os.environ["GCP_JSON"]))
        else:
            gc = gspread.service_account(filename=os.path.join(current_dir, "credentials.json"))
        sh = gc.open(SHEET_NAME)
        try:
            ws = sh.worksheet(TAB_NAME)
        except gspread.exceptions.WorksheetNotFound:
            ws = sh.add_worksheet(title=TAB_NAME, rows=50, cols=14)
        ws.clear()
        if orders.empty:
            ws.update([["No rebalance orders", datetime.datetime.now().strftime("%Y-%m-%d %H:%M")]])
        else:
            ws.update([orders.columns.tolist()] + orders.astype(str).values.tolist())
        print(f"Wrote {len(orders)} orders -> '{TAB_NAME}' tab")
    except Exception as e:
        print(f"ERROR: Sheets write failed: {e}")
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="run even if today is not the month's last trading day")
    ap.add_argument("--dry-run", action="store_true",
                    help="compute and print; no Sheets write, no state save")
    args = ap.parse_args()

    if not args.force and not is_last_trading_day():
        print("Not the last trading day of the month — nothing to do.")
        return

    closes = load_closes()
    missing = [t for t in TREND_UNIVERSE if t not in closes.columns]
    if missing:
        print(f"ERROR: universe tickers missing from master_prices: {missing}")
        sys.exit(1)
    stale_days = (pd.Timestamp.today().normalize() - closes.index.max()).days
    if stale_days > 5:
        print(f"ERROR: master_prices stale ({closes.index.max().date()}) — refusing to rebalance")
        sys.exit(1)
    if not args.force and stale_days > 0:
        # scheduled month-end run: the signal basis IS today's close; if the
        # PM price update hasn't landed yet, staging off yesterday's bar would
        # silently violate the backtested convention. Fail loudly instead.
        print(f"ERROR: last bar is {closes.index.max().date()} — today's close not in "
              "master_prices yet (runs after update_master_prices PM). Aborting.")
        sys.exit(1)

    targets = compute_targets(closes)
    print(f"Trend sleeve targets (asof {targets['Asof'].iloc[0]}, "
          f"{TREND_NAV_FRACTION:.1f}x of ${ACCOUNT_VALUE:,.0f}):")
    print(targets.to_string(index=False))
    on = targets[targets.Weight > 0]
    print(f"\ndeployed: {on.Weight.sum()*100:.1f}% of sleeve NAV across {len(on)} ETFs")

    state = load_state()
    orders = build_orders(targets, state)
    print(f"\nRebalance orders ({len(orders)}):")
    if not orders.empty:
        print(orders.to_string(index=False))

    write_sheet(orders, args.dry_run)
    save_state(targets, args.dry_run)


if __name__ == "__main__":
    main()
