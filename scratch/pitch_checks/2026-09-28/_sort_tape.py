import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
t = json.loads((ROOT / "data/pitch_tape.json").read_text(encoding="utf-8"))
df = pd.DataFrame(t["tickers"]).T
cols = ["ret_1d", "ret_5d", "ret_21d", "ret_63d", "rank_5d", "rank_21d", "rank_63d", "z10",
        "dist_52w_high_pct", "dist_52w_low_pct", "dist_sma200_pct", "atr_pct", "vol_vs_63d"]
df = df[cols].apply(pd.to_numeric, errors="coerce")
df.columns = ["r1", "r5", "r21", "r63", "k5", "k21", "k63", "z10", "d52h", "d52l", "d200", "atr%", "vr"]
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 300)
CLASSES = {
    "us_large": ["SPY", "QQQ", "^GSPC", "^NDX", "DIA"],
    "small": ["IWM"],
    "rates": ["TLT", "IEF", "^TNX", "SHY"],
    "credit": ["HYG", "LQD"],
    "gold": ["GLD", "GDX"],
    "metals": ["SLV", "CPER", "COPX"],
    "energy": ["USO", "UNG", "DBC", "XLE", "XOP"],
    "fx": ["UUP", "DX-Y.NYB", "FXE", "FXY"],
    "intl": ["EFA", "EEM", "FXI", "EWJ", "EWZ", "INDA"],
    "vol": ["^VIX", "^VIX3M", "^MOVE", "SVXY", "UVXY", "VXX"],
}
for c, tk in CLASSES.items():
    have = [x for x in tk if x in df.index]
    print(f"== {c}")
    print(df.loc[have].round(2).to_string())
print("\n== ALL sorted by r21")
print(df.sort_values("r21").round(2).to_string())
