"""seasonal_order_staging.py — stage seasonal-idea tickets to Google Sheets tabs
for order_staging.py (IBKR) to read, mirroring daily_scan.save_staging_orders.

Reads data/daily_seasonal_ideas.json (the emitted tickets), turns each tradeable
A-grade ticket into an order_staging row, and writes two tabs in the
`Trade_Signals_Log` workbook:
  - `Seasonal`      — tradeable longs + tradeable non-equity shorts (order_staging
                      reads + submits these)
  - `sznl_nostage`  — NOT auto-staged: single-stock equity shorts + non-tradeable
                      signals (futures/index/FX/crypto needing a proxy ETF). A
                      reference/manual-review tab; order_staging does not read it.

Entry type per instrument (validated 2026-06-26 geography analysis):
  - US single stocks + US-session equity ETFs  -> REL_OPEN limit, 0.25 ATR, DAY
  - everything that gaps overnight (intl ETFs, commodity/bond/FX ETFs, GLD/TLT)
    -> MOO (market-on-open, TIF=OPG)
Raw index/future/FX/crypto symbols (^GSPC, CL=F, EURUSD=X, BTC-USD) are NOT
directly tradeable at IBKR and are DEFERRED — they need the proxy-ETF promotion
(seasonal handoff open item #4). They are reported, never staged.

Sizing: SEASONAL_RISK_BPS of ACCOUNT_VALUE per trade (SEASONAL_MIDTERM_RISK_BPS
in midterm years, year % 4 == 2), shares = risk$ / |entry - stop|, with a
SEASONAL_DAILY_CAP_PCT of ACCOUNT_VALUE aggregate cap on the Seasonal tab
(pro-rata scale-down). sznl_nostage rows are sized the same way (equity shorts)
or carry Quantity 0 (non-tradeable, need a proxy) and are NOT counted against the
Seasonal cap.

Columns match daily_scan.save_staging_orders so order_staging.py consumes them
unchanged EXCEPT the new MOO entry instruction (Order_Type='MOO', TIF='OPG') —
see the integration spec in docs/seasonal_order_staging_spec.md.

Default run is a DRY RUN (prints the rows). Pass --write to touch Sheets.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys

import pandas as pd

try:
    import gspread
except Exception:  # gspread only needed for --write
    gspread = None

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

try:
    from strategy_config import ACCOUNT_VALUE
except Exception:
    ACCOUNT_VALUE = 750_000.0

JSON_PATH = os.path.join(_HERE, "data", "daily_seasonal_ideas.json")

# ---- sizing knobs ------------------------------------------------------------
SEASONAL_RISK_BPS = 20.0          # risk per trade, bps of ACCOUNT_VALUE
SEASONAL_MIDTERM_RISK_BPS = 13.0  # midterm-year downsize (year % 4 == 2).
#   NOTE: read from your "downsize midterm to 0.13 bps" as 13 bps (= 0.13%),
#   paralleling the 20 bps (0.20%) base and the OVS 0.75x midterm precedent.
#   If you meant a 0.13x MULTIPLIER instead, set this to 20 * 0.13 = 2.6.
SEASONAL_DAILY_CAP_PCT = 1.0      # Seasonal-tab aggregate risk cap, % of ACCOUNT_VALUE
ENTRY_ATR_OFFSET = 0.25           # REL_OPEN limit offset for US-session names

SEASONAL_TAB = "Seasonal"
NOSTAGE_TAB = "sznl_nostage"
SCAN_SOURCE = "Seasonal"
NOSTAGE_SOURCE = "Seasonal_NoStage"

# US-session equity ETFs that take a limit entry even though they're in the
# macro/cross-asset channel (they trade during the US cash session). Seeded from
# the validated analysis (scratch/complete_book.py US_IDX_ETF) + SPDR sectors.
US_SESSION_ETF = {
    "SPY", "QQQ", "DIA", "IWM", "IJH", "MDY", "ONEQ", "SOXX", "IYT",
    "XLK", "XLF", "XLE", "XLI", "XLV", "XLP", "XLU", "XLY", "XLB", "XLRE", "XLC",
    "SMH", "IBB", "KRE", "XBI",
}

# Output column contract — identical to daily_scan.save_staging_orders so
# order_staging.py reads Seasonal/eq_shorts exactly like Order_Staging/Overflow.
COLUMNS = [
    "Scan_Date", "Symbol", "SecType", "Exchange", "Action", "Quantity",
    "Order_Type", "Limit_Price", "Manual_Limit", "Offset_ATR_Mult", "TIF",
    "Frozen_ATR", "Signal_Close", "Time_Exit_Date", "Strategy_Ref",
    "Tgt_ATR_Mult", "Stop_ATR_Mult", "Use_Target", "Use_Stop", "Hold_Days",
    "Trade_Direction", "Rank_252D", "Risk_Amt", "Risk_Bps", "Scan_Source",
    "Entry_Offset_Days", "Entry_Activate_Date",
]

# BUY ~332.23 | stop 323.91 (1.2 ATR) | target 348.86 | time-stop 21td | R/R 2.0
_TICKET_RE = re.compile(
    r"(BUY|SELL)\s+~?(-?[\d.]+).*?stop\s+(-?[\d.]+)\s*\(([\d.]+)\s*ATR\).*?"
    r"target\s+(-?[\d.]+).*?time-stop\s+(\d+)\s*td.*?R/R\s+(-?[\d.]+)",
    re.IGNORECASE,
)

_STK_RE = re.compile(r"[A-Z]{1,5}")  # plain US equity / ETF symbol


def is_tradeable_stk(ticker: str) -> bool:
    """True only for plain US equity/ETF symbols. Indices (^X), futures (X=F),
    FX (X=X), crypto (X-USD), dollar index (DX-Y.NYB) are NOT directly tradeable."""
    return bool(_STK_RE.fullmatch(str(ticker).strip().upper()))


def sectype_of(ticker: str) -> str:
    """IBKR-ish SecType inferred from the symbol shape (for the nostage tab)."""
    t = str(ticker).strip().upper()
    if t.endswith("=F"):
        return "FUT"
    if "=X" in t:
        return "CASH"          # FX pair
    if t.endswith("-USD"):
        return "CRYPTO"
    if t.startswith("^") or t.endswith(".NYB"):
        return "IND"           # index
    return "STK"


def is_macro_channel(channel: str) -> bool:
    c = str(channel).lower()
    return "macro" in c or "cross-asset" in c or "cross asset" in c


def classify_entry(ticker: str, channel: str) -> str:
    """Return 'REL_OPEN' (US-session limit) or 'MOO' (gaps overnight)."""
    t = str(ticker).strip().upper()
    if is_macro_channel(channel):
        return "REL_OPEN" if t in US_SESSION_ETF else "MOO"
    return "REL_OPEN"  # equity single stock


def parse_seasonal_ticket(cand: dict) -> dict | None:
    """Lift a candidate's evidence.TICKET into structured numeric fields, including
    the stop's ATR multiple (which the simulator's parser drops). None if no
    tradeable TICKET string."""
    ev = cand.get("evidence") or {}
    s = ev.get("TICKET")
    if not s:
        return None
    m = _TICKET_RE.search(str(s))
    if not m:
        return None
    verb, entry, stop, stop_atr, target, tsd, rr = m.groups()
    entry, stop, stop_atr, target, rr = (float(entry), float(stop),
                                         float(stop_atr), float(target), float(rr))
    tsd = int(tsd)
    direction = "Long" if verb.upper() == "BUY" else "Short"
    risk_ps = abs(entry - stop)
    if risk_ps <= 0 or stop_atr <= 0 or tsd <= 0:
        return None
    atr = risk_ps / stop_atr                 # back out the ATR from stop distance
    tgt_atr = abs(target - entry) / atr if atr > 0 else 0.0
    return {
        "entry": entry, "stop": stop, "target": target, "rr": rr,
        "direction": direction, "time_stop_days": tsd,
        "risk_ps": risk_ps, "atr": atr,
        "stop_atr": stop_atr, "tgt_atr": tgt_atr,
    }


def _row(cand: dict, tk: dict, risk_bps: float, account_value: float, asof: str,
         scan_source: str = SCAN_SOURCE, note: str = "") -> dict:
    ticker = str(cand["ticker"]).strip().upper()
    channel = cand.get("channel", "")
    entry_instr = classify_entry(ticker, channel)
    direction = tk["direction"]
    action = "BUY" if direction == "Long" else "SELL"

    risk_amt = account_value * risk_bps / 10000.0
    try:
        shares = int(risk_amt / tk["risk_ps"])
    except (ValueError, ZeroDivisionError, OverflowError):
        shares = 0

    if entry_instr == "MOO":
        order_type, offset, tif, limit_price = "MOO", 0.0, "OPG", 0.0
    else:
        order_type, offset, tif, limit_price = "REL_OPEN", ENTRY_ATR_OFFSET, "DAY", 0.0

    time_exit = (pd.Timestamp(asof) + pd.tseries.offsets.BDay(tk["time_stop_days"])).strftime("%Y-%m-%d")
    horizon = cand.get("horizon", "")
    conv = cand.get("conviction", "")
    # Expected seasonal-path entry day: 0 = T+1 (next session), k = T+(k+1). The
    # order rests with goodAfterTime = Entry_Activate_Date so it activates that
    # session instead of T+1. order_staging reads this; default 0 keeps T+1.
    entry_off = int(cand.get("entry_offset_days", 0) or 0)
    activate = (pd.Timestamp(asof) + pd.tseries.offsets.BDay(entry_off + 1)).strftime("%Y-%m-%d")

    return {
        "Scan_Date": asof,
        "Symbol": ticker,
        "SecType": "STK",
        "Exchange": "SMART",
        "Action": action,
        "Quantity": shares,
        "Order_Type": order_type,
        "Limit_Price": round(limit_price, 2),
        "Manual_Limit": "",
        "Offset_ATR_Mult": offset,
        "TIF": tif,
        "Frozen_ATR": round(tk["atr"], 4),
        "Signal_Close": round(tk["entry"], 2),
        "Time_Exit_Date": time_exit,
        "Strategy_Ref": (f"Seasonal/{conv}/{horizon}".rstrip("/") + (f" {note}" if note else "")),
        "Tgt_ATR_Mult": round(tk["tgt_atr"], 3),
        "Stop_ATR_Mult": round(tk["stop_atr"], 3),
        "Use_Target": True,
        "Use_Stop": True,
        "Hold_Days": tk["time_stop_days"],
        "Trade_Direction": direction,
        "Rank_252D": "",
        "Risk_Amt": round(risk_amt, 2),
        "Risk_Bps": risk_bps,
        "Scan_Source": scan_source,
        "Entry_Offset_Days": entry_off,
        "Entry_Activate_Date": activate,
    }


def _deferred_row(cand: dict, tk: dict, asof: str) -> dict:
    """A non-tradeable signal (future/index/FX/crypto) for the nostage tab. Carries
    the ticket's reference levels but Quantity 0 — it needs a proxy ETF to trade."""
    ticker = str(cand["ticker"]).strip().upper()
    direction = tk["direction"]
    time_exit = (pd.Timestamp(asof) + pd.tseries.offsets.BDay(tk["time_stop_days"])).strftime("%Y-%m-%d")
    return {
        "Scan_Date": asof, "Symbol": ticker, "SecType": sectype_of(ticker),
        "Exchange": "", "Action": "BUY" if direction == "Long" else "SELL",
        "Quantity": 0, "Order_Type": "NONE", "Limit_Price": round(tk["entry"], 2),
        "Manual_Limit": "", "Offset_ATR_Mult": 0.0, "TIF": "",
        "Frozen_ATR": round(tk["atr"], 4), "Signal_Close": round(tk["entry"], 2),
        "Time_Exit_Date": time_exit,
        "Strategy_Ref": f"Seasonal/{cand.get('conviction','')}/{cand.get('horizon','')}".rstrip("/") + " [need-proxy]",
        "Tgt_ATR_Mult": round(tk["tgt_atr"], 3), "Stop_ATR_Mult": round(tk["stop_atr"], 3),
        "Use_Target": True, "Use_Stop": True, "Hold_Days": tk["time_stop_days"],
        "Trade_Direction": direction, "Rank_252D": "", "Risk_Amt": 0.0, "Risk_Bps": 0,
        "Scan_Source": NOSTAGE_SOURCE,
    }


def _apply_daily_cap(rows: list, account_value: float, cap_pct: float) -> list:
    """Pro-rata scale Quantity/Risk_Amt so total Risk_Amt <= cap_pct% of account."""
    cap = account_value * cap_pct / 100.0
    total = sum(r["Risk_Amt"] for r in rows)
    if total <= cap or total <= 0:
        return rows
    scale = cap / total
    for r in rows:
        r["Quantity"] = int(r["Quantity"] * scale)
        r["Risk_Amt"] = round(r["Risk_Amt"] * scale, 2)
        r["Risk_Bps"] = round(r["Risk_Bps"] * scale, 3)
    return rows


def build_seasonal_rows(payload: dict, account_value: float = ACCOUNT_VALUE):
    """Return (seasonal_rows, nostage_rows). Tradeable longs + non-equity shorts go
    to `seasonal`; single-stock equity shorts and non-tradeable signals (which need
    a proxy ETF) go to `nostage`. Only tickets with a parseable TICKET are staged."""
    asof = (payload.get("meta") or {}).get("asof") or datetime.date.today().strftime("%Y-%m-%d")
    year = pd.Timestamp(asof).year
    risk_bps = SEASONAL_MIDTERM_RISK_BPS if (year % 4 == 2) else SEASONAL_RISK_BPS

    seasonal, nostage = [], []
    for cand in payload.get("candidates", []):
        tk = parse_seasonal_ticket(cand)
        if tk is None:
            continue  # context/regime/tilt rows with no order
        ticker = str(cand.get("ticker", "")).strip().upper()
        channel = cand.get("channel", "")
        if not is_tradeable_stk(ticker):
            nostage.append(_deferred_row(cand, tk, asof))      # needs a proxy ETF
            continue
        is_equity = not is_macro_channel(channel)
        if is_equity and tk["direction"] == "Short":
            nostage.append(_row(cand, tk, risk_bps, account_value, asof,
                                scan_source=NOSTAGE_SOURCE, note="[eq-short]"))
        else:
            seasonal.append(_row(cand, tk, risk_bps, account_value, asof))

    seasonal = _apply_daily_cap(seasonal, account_value, SEASONAL_DAILY_CAP_PCT)
    return seasonal, nostage


# ---- Google Sheets I/O -------------------------------------------------------

def get_google_client():
    """gspread client from GCP_JSON env (GHA) or credentials.json (local)."""
    if gspread is None:
        print("gspread not importable — cannot write Sheets")
        return None
    try:
        if "GCP_JSON" in os.environ:
            return gspread.service_account_from_dict(json.loads(os.environ["GCP_JSON"]))
        if os.path.exists(os.path.join(_HERE, "credentials.json")):
            return gspread.service_account(filename=os.path.join(_HERE, "credentials.json"))
        print("No credentials (GCP_JSON env or credentials.json).")
        return None
    except Exception as e:
        print(f"Google auth failed: {e}")
        return None


def _write_tab(sh, tab_name: str, rows: list):
    try:
        ws = sh.worksheet(tab_name)
    except Exception:
        ws = sh.add_worksheet(title=tab_name, rows=200, cols=len(COLUMNS) + 2)
    ws.clear()
    if not rows:
        ws.update(values=[COLUMNS])  # headers only, so stale rows never linger
        print(f"  [{tab_name}] cleared (0 rows)")
        return
    data = [COLUMNS] + [[r.get(c, "") for c in COLUMNS] for r in rows]
    ws.update(values=data)
    print(f"  [{tab_name}] wrote {len(rows)} row(s)")


def save_seasonal_tabs(seasonal_rows: list, nostage_rows: list):
    gc = get_google_client()
    if not gc:
        return False
    try:
        sh = gc.open("Trade_Signals_Log")
    except Exception as e:
        print(f"Could not open Trade_Signals_Log: {e}")
        return False
    _write_tab(sh, SEASONAL_TAB, seasonal_rows)
    _write_tab(sh, NOSTAGE_TAB, nostage_rows)
    return True


def _print_rows(title, rows):
    print(f"\n=== {title} ({len(rows)}) ===")
    if not rows:
        print("  (none)")
        return
    df = pd.DataFrame(rows)[
        ["Symbol", "SecType", "Action", "Order_Type", "TIF", "Quantity",
         "Frozen_ATR", "Stop_ATR_Mult", "Tgt_ATR_Mult", "Hold_Days",
         "Time_Exit_Date", "Risk_Amt", "Risk_Bps", "Strategy_Ref"]
    ]
    print(df.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="write the Seasonal + eq_shorts tabs (default is a dry run)")
    ap.add_argument("--json", default=JSON_PATH, help="path to daily_seasonal_ideas.json")
    args = ap.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    seasonal, nostage = build_seasonal_rows(payload)
    asof = (payload.get("meta") or {}).get("asof", "?")
    year = pd.Timestamp(asof).year if asof != "?" else 0
    bps = SEASONAL_MIDTERM_RISK_BPS if (year % 4 == 2) else SEASONAL_RISK_BPS
    print(f"asof {asof} | account ${ACCOUNT_VALUE:,.0f} | risk {bps} bps/trade"
          f"{' (MIDTERM)' if year % 4 == 2 else ''} | daily cap {SEASONAL_DAILY_CAP_PCT}%")

    _print_rows(f"{SEASONAL_TAB} tab (order_staging executes these)", seasonal)
    _print_rows(f"{NOSTAGE_TAB} tab (eq shorts + need-proxy, NOT executed)", nostage)

    cap = ACCOUNT_VALUE * SEASONAL_DAILY_CAP_PCT / 100.0
    print(f"\nSeasonal tab total risk: ${sum(r['Risk_Amt'] for r in seasonal):,.0f} "
          f"(cap ${cap:,.0f})")

    if args.write:
        print("\n[--write] pushing tabs to Trade_Signals_Log ...")
        save_seasonal_tabs(seasonal, nostage)
    else:
        print("\n(dry run — pass --write to push to Sheets)")


if __name__ == "__main__":
    main()
