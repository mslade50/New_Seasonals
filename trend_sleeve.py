"""Trend-following ballast sleeve — monthly rebalance pilot (2026-07-02).

Spec (validated: scratch/ultracode_research/trend-following.md + adversarial
verification + trend_prework_gates.md + scratch/tf_universe_study.py):
  - Universe: 12 liquid ETFs — equities/RE/intl core + GLD/SLV/DBC +
    TLT/LQD, no USO/UUP (see TREND_UNIVERSE note below for the evidence).
  - Signal (month-end adjusted closes): combo = 12-1 momentum > 0 AND close >
    10-month SMA. Long/flat only — the short leg is dead (Sharpe 0.32).
  - Weights: inverse-vol slots over ALL eligible assets (>= 13 monthly closes),
    capped at 20%; OFF slots sit in cash.
  - Size: TREND_NAV_FRACTION x ACCOUNT_VALUE deployed (cut 0.6 -> 0.3 on
    2026-07-17, McKinley: caps concurrent index overlap with the dial-gated
    SPY sleeve at ~43% NAV worst case and trims the sleeve's NAV vol
    contribution to ~2.4%/yr; revisit scale-up after 2 clean quarters).
  - Execution: signals computed on the LAST trading day's close, staged as MOO
    (TIF=OPG) for the next session's open. Next-open slippage vs same-close
    was measured at ~0.06 Sharpe on the 12-ETF variant (gate A, PASS);
    expect roughly Sharpe ~0.8, CAGR ~5.5%, maxDD ~-6.5%, corr to book ~+0.12
    for this universe on the next-open basis.
  - Month-entry fragility gate: hold the entire sleeve in cash when the latest
    valid month-end 10d MA of the 63d risk dial is > 50. Missing/stale dial
    data also blocks exposure (fail-closed). Production 12-ETF next-open test:
    -0.50%/mo above 50 vs +0.71% otherwise, 2016-08..2026-06 (N=19/98).

Runs weekdays post-close via .github/workflows/trend_sleeve.yml; exits
immediately unless today is the month's last trading day (--force overrides).
Rebalance orders are written to the 'Trend' Sheets tab; state (held shares)
persists in trend_sleeve_state.json on R2. --dry-run prints without writing.

When activated, execution is handled by the local pre-market ``trend_moo.py``
runner at 09:12 ET. It reads only rows whose Execute_On is today and places
true MKT+OPG orders before the opening auction. An atomic enable marker makes
the 09:31 order_staging chain ignore Trend rows, preventing MKT/DAY duplicates;
until activation, the legacy path remains intact.
"""
import argparse
import datetime
import json
import os
import sys

import numpy as np
import pandas as pd
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from strategy_config import ACCOUNT_VALUE
from cache_io import download_to_local, head, upload_from_local
from trading_calendar import TRADING_DAY

# 12-ETF universe (2026-07-02, McKinley's spec, confirmed by
# scratch/tf_universe_study.py): USO dropped (roll-decay vehicle; its removal
# costs some 2021-22 oil-trend CAGR but improves Sharpe and halves maxDD),
# TLT+LQD added (trend-filtered duration: +5.8% in 2008, +0.5% in 2022 because
# the trend filter sidesteps the bond bear; IEF/HYG add nothing over these two),
# UUP dropped (capital inefficiency: 20% of sleeve parked in a ~7%-vol asset
# for +0.00%/mo average contribution, plus a K-1; costs Sharpe 0.87 -> 0.86
# and gives back 2022 (+0.5% -> -1.9%, the dollar was the only long trend in
# the everything-bear) — accepted trade-off, McKinley 2026-07-02). Tested and
# REJECTED: 11 SPDR sectors and intl singles (Sharpe 0.74-0.78, 2008/2022 flip
# negative — correlated equity slots crowd out the real diversifiers); an
# exhaustion scale-down overlay (252d & 21d pctile > 95) — Sharpe flat.
TREND_UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "FXI", "VNQ",
                  "GLD", "SLV", "DBC", "TLT", "LQD"]
TREND_NAV_FRACTION = 0.3     # 30% of NAV (cut from 0.6 on 2026-07-17 — see above)
WEIGHT_CAP = 0.20
MIN_MONTHLY_CLOSES = 13      # eligibility (needs a valid 12-month lookback)
VOL_FLOOR = 0.04             # ann. vol floor avoids inf inverse-vol weights
REBALANCE_BAND = 0.01        # skip deltas under 1% of sleeve NAV unless a flip

MASTER_PRICES = os.path.join(current_dir, "data", "master_prices.parquet")
FRAGILITY_PATH = os.path.join(current_dir, "data", "rd2_fragility.parquet")
STATE_LOCAL = os.path.join(current_dir, "data", "trend_sleeve_state.json")
STATE_R2_KEY = "trend_sleeve_state.json"
SHEET_NAME = "Trade_Signals_Log"
TAB_NAME = "Trend"
TREND_FRAG_THRESHOLD = 50.0
TREND_FRAG_STALE_TD = 3


def _today_et() -> pd.Timestamp:
    """Trading-calendar 'today' in ET — GHA runs on UTC, where a late-evening
    dispatch has already rolled to tomorrow's date and would mis-stamp
    Execute_On by one session."""
    return pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()


def is_last_trading_day(today=None) -> bool:
    """True when the next NYSE session starts a new month. (2026-07-16:
    switched from the federal calendar, which also mis-stamped Execute_On to
    a Good Friday — order_staging never runs on a closed day, so those rows
    would have been ignored forever and the rebalance silently missed.)"""
    today = pd.Timestamp(today) if today is not None else _today_et()
    return (today.normalize() + TRADING_DAY).month != today.normalize().month


def load_closes() -> pd.DataFrame:
    mp = pd.read_parquet(MASTER_PRICES)
    tcol = "ticker" if "ticker" in mp.columns else "Ticker"
    dcol = "date" if "date" in mp.columns else "Date"
    sub = mp[mp[tcol].str.upper().isin(TREND_UNIVERSE)]
    closes = sub.assign(_d=pd.to_datetime(sub[dcol]).values).pivot_table(
        index="_d", columns=tcol, values="Close")
    closes.columns = [str(c).upper() for c in closes.columns]
    return closes.sort_index()


def read_trend_fragility_gate(fragility_path=FRAGILITY_PATH, asof=None) -> dict:
    """Return the month-entry Trend exposure gate.

    The sizing basis is the book's live series:
    ``rd2_fragility.parquet['63d'].rolling(10).mean()``. The latest reading
    on or before *asof* is used. Missing, malformed, or stale data fails closed
    to cash because enabling a new risk sleeve from an unknown regime is the
    less reversible failure mode.
    """
    asof_ts = pd.Timestamp(asof if asof is not None else _today_et()).normalize()
    try:
        asof_ts = asof_ts.tz_localize(None)
    except (TypeError, AttributeError):
        pass

    def blocked(reason, score=None, score_asof=None):
        return {
            "active": True,
            "valid": False,
            "score": score,
            "asof": str(score_asof.date()) if score_asof is not None else None,
            "threshold": TREND_FRAG_THRESHOLD,
            "reason": reason,
        }

    if not os.path.exists(fragility_path):
        return blocked(f"fragility file missing: {fragility_path}")
    try:
        frag = pd.read_parquet(fragility_path)
    except Exception as exc:
        return blocked(f"fragility file unreadable: {exc}")
    if frag is None or frag.empty or "63d" not in frag.columns:
        return blocked("fragility file empty or missing 63d column")

    series = frag["63d"].dropna().copy()
    series.index = pd.to_datetime(series.index).normalize()
    try:
        series.index = series.index.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    series = series[series.index <= asof_ts].sort_index()
    if series.empty:
        return blocked(f"no fragility reading on or before {asof_ts.date()}")

    score_asof = pd.Timestamp(series.index[-1]).normalize()
    age_td = int(np.busday_count(score_asof.date(), asof_ts.date()))
    if age_td > TREND_FRAG_STALE_TD:
        return blocked(
            f"fragility reading stale: {score_asof.date()} is {age_td} trading "
            f"days old (max {TREND_FRAG_STALE_TD})",
            score_asof=score_asof,
        )

    score = float(series.rolling(10, min_periods=1).mean().iloc[-1])
    active = score > TREND_FRAG_THRESHOLD
    return {
        "active": active,
        "valid": True,
        "score": score,
        "asof": str(score_asof.date()),
        "threshold": TREND_FRAG_THRESHOLD,
        "reason": (
            f"10d-MA 63d fragility {score:.2f} > {TREND_FRAG_THRESHOLD:.0f}"
            if active else
            f"10d-MA 63d fragility {score:.2f} <= {TREND_FRAG_THRESHOLD:.0f}"
        ),
    }


def apply_trend_fragility_gate(targets: pd.DataFrame, gate: dict) -> pd.DataFrame:
    """Stamp the gate and zero all target weights while it is active."""
    out = targets.copy()
    score = gate.get("score")
    out["Fragility_MA10_63d"] = round(float(score), 4) if score is not None else np.nan
    out["Fragility_Gate"] = "CASH" if gate.get("active", True) else "OPEN"
    out["Fragility_Gate_AsOf"] = gate.get("asof") or "UNAVAILABLE"
    out["Fragility_Gate_Reason"] = str(gate.get("reason", "unknown"))
    if gate.get("active", True):
        out["Weight"] = 0.0
    return out


def compute_targets(closes: pd.DataFrame, fragility_path=FRAGILITY_PATH,
                    use_fragility_gate: bool = True) -> pd.DataFrame:
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
    if use_fragility_gate:
        gate = read_trend_fragility_gate(
            fragility_path=fragility_path, asof=closes.index.max())
        df = apply_trend_fragility_gate(df, gate)
    return df


def load_state() -> dict:
    download_to_local(STATE_R2_KEY, STATE_LOCAL)  # no-op without R2 creds
    if os.path.exists(STATE_LOCAL):
        try:
            with open(STATE_LOCAL, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"WARNING: state unreadable ({e}) - assuming flat book")
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
    out = pd.DataFrame(orders)
    if not out.empty:
        # trend_moo.py only submits Trend rows on their execution day: the next
        # session after this post-close run. On the normal month-end schedule
        # run-date == signal date, so this is the first session of the new
        # month; an off-schedule --force run stages for the next open. Stale
        # rows are ignored forever after.
        exec_on = _today_et() + TRADING_DAY
        out["Execute_On"] = str(exec_on.date())
    return out


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
    if "Fragility_Gate" in targets.columns:
        first = targets.iloc[0]
        score = first.get("Fragility_MA10_63d")
        state["fragility_gate"] = {
            "state": first.get("Fragility_Gate"),
            "score": None if pd.isna(score) else float(score),
            "asof": first.get("Fragility_Gate_AsOf"),
            "reason": first.get("Fragility_Gate_Reason"),
            "threshold": TREND_FRAG_THRESHOLD,
        }
    if dry_run:
        print("\n[dry-run] state not saved")
        return
    with open(STATE_LOCAL, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    upload_ok = bool(upload_from_local(STATE_LOCAL, STATE_R2_KEY))
    if os.environ.get("LOCAL_AUTOMATION_STRICT", "").strip() == "1":
        meta = head(STATE_R2_KEY)
        actual = int((meta or {}).get("ContentLength") or -1)
        expected = os.path.getsize(STATE_LOCAL)
        if not upload_ok or actual != expected:
            raise RuntimeError(
                f"Trend state R2 verification failed: uploaded={upload_ok}, "
                f"size={actual}, expected={expected}"
            )
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
            expected = [["No rebalance orders", datetime.datetime.now().strftime("%Y-%m-%d %H:%M")]]
            ws.update(expected)
        else:
            expected = [orders.columns.tolist()] + orders.astype(str).values.tolist()
            ws.update(expected)
        if os.environ.get("LOCAL_AUTOMATION_STRICT", "").strip() == "1":
            actual = ws.get_all_values()
            if actual != expected:
                raise RuntimeError(
                    f"Trend tab readback mismatch: wrote {len(expected)} rows, "
                    f"read {len(actual)}"
                )
        print(f"Wrote {len(orders)} orders -> '{TAB_NAME}' tab")
    except Exception as e:
        print(f"ERROR: Sheets write failed: {e}")
        sys.exit(1)


def reset_state():
    """Reset the sleeve to a FLAT book: empty positions in the R2 state file
    and a cleared Trend tab. Used when a staged basket was never executed
    (inception 2026-07-06: McKinley chose to wait for the natural month-end
    deployment) — without this, the next rebalance computes deltas against
    phantom holdings and stages the wrong orders."""
    state = {
        "asof": _today_et().strftime("%Y-%m-%d"),
        "generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "nav_fraction": TREND_NAV_FRACTION,
        "universe": TREND_UNIVERSE,
        "positions": {},
        "note": "state reset to flat (staged basket not executed)",
    }
    with open(STATE_LOCAL, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    upload_from_local(STATE_LOCAL, STATE_R2_KEY)
    print("State reset to FLAT (0 positions) -> local + R2")
    try:
        import gspread
        if "GCP_JSON" in os.environ:
            gc = gspread.service_account_from_dict(json.loads(os.environ["GCP_JSON"]))
        else:
            gc = gspread.service_account(filename=os.path.join(current_dir, "credentials.json"))
        ws = gc.open(SHEET_NAME).worksheet(TAB_NAME)
        ws.clear()
        ws.update([["Sleeve flat - awaiting next month-end rebalance",
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M")]])
        print(f"Cleared '{TAB_NAME}' tab")
    except Exception as e:
        print(f"WARNING: Trend tab clear failed ({e}) - stale rows are harmless "
              "(Execute_On gate ignores them) but untidy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="run even if today is not the month's last trading day")
    ap.add_argument("--dry-run", action="store_true",
                    help="compute and print; no Sheets write, no state save")
    ap.add_argument("--reset-state", action="store_true",
                    help="reset the sleeve to a flat book (unexecuted staged basket)")
    args = ap.parse_args()

    if args.reset_state:
        reset_state()
        return

    if not args.force and not is_last_trading_day():
        print("Not the last trading day of the month - nothing to do.")
        return

    closes = load_closes()
    missing = [t for t in TREND_UNIVERSE if t not in closes.columns]
    if missing:
        print(f"ERROR: universe tickers missing from master_prices: {missing}")
        sys.exit(1)
    stale_days = (_today_et() - closes.index.max()).days
    if stale_days > 5:
        print(f"ERROR: master_prices stale ({closes.index.max().date()}) - refusing to rebalance")
        sys.exit(1)
    if not args.force and stale_days > 0:
        # scheduled month-end run: the signal basis IS today's close; if the
        # PM price update hasn't landed yet, staging off yesterday's bar would
        # silently violate the backtested convention. Fail loudly instead.
        print(f"ERROR: last bar is {closes.index.max().date()} - today's close not in "
              "master_prices yet (runs after update_master_prices PM). Aborting.")
        sys.exit(1)

    targets = compute_targets(closes)
    if "Fragility_Gate" in targets.columns:
        first = targets.iloc[0]
        print(f"Trend fragility gate: {first['Fragility_Gate']} - "
              f"{first['Fragility_Gate_Reason']} "
              f"(as of {first['Fragility_Gate_AsOf']})")
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
