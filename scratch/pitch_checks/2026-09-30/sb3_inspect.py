from pathlib import Path

import pandas as pd

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
d = pd.read_parquet(Path(__file__).parent / "sb3_ledger_r2.parquet")
d["asof"] = pd.to_datetime(d["asof"])
print(d["asof"].value_counts().sort_index())
print(d["logged_at"].astype(str).str[:10].value_counts().sort_index())
print(d.groupby(["channel", "conviction", "direction"]).size())
cols = ["asof", "ticker", "channel", "conviction", "direction", "entry", "stop", "target",
        "time_stop_days", "entry_offset_days", "logged_at"]
print(d[cols].to_string())
