"""Integration check: build_trade_console end-to-end against real histories
(on_* columns of rtc_config_history.parquet ARE the signal histories)."""
import json
import os
import sys

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from scripts.build_risk_json import build_trade_console
from scripts.build_trade_console_stats import ABBR

frame = pd.read_parquet(os.path.join(_ROOT, "scratch", "rtc_config_history.parquet"))
signals_ordered = {name: {"signal_history": frame[f"on_{abbr}"]}
                   for name, abbr in ABBR.items()}
computed = {"signals_ordered": signals_ordered,
            "spy_close": frame["spy_close"]}

tc = build_trade_console(computed)
print(json.dumps(tc, indent=1))
