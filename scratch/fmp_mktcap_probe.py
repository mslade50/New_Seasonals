"""Probe FMP historical market cap endpoint: coverage depth + row shape."""
import os
import sys

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.build_earnings_calendar import load_env

KEY = load_env()
BASE = "https://financialmodelingprep.com/stable/historical-market-capitalization"

for symbol in ["AAPL", "GME", "SMCI"]:
    r = requests.get(BASE, params={
        "symbol": symbol, "from": "2000-01-01", "to": "2026-07-15",
        "limit": 10000, "apikey": KEY}, timeout=30)
    print(symbol, "status:", r.status_code)
    if r.ok:
        rows = r.json()
        print("  rows:", len(rows))
        if rows:
            print("  first:", rows[-1])
            print("  last:", rows[0])
