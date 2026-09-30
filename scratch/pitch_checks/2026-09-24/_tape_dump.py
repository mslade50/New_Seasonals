import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
t = json.loads((ROOT / "data" / "pitch_tape.json").read_text())["tickers"]

CLASSES = {
    "us_large": ["SPY", "QQQ", "^GSPC", "^NDX", "XLK", "SMH"],
    "us_small": ["IWM"],
    "rates": ["TLT", "IEF", "^TNX", "SHY"],
    "credit": ["HYG", "LQD"],
    "gold_miners": ["GLD", "GDX"],
    "other_metals": ["SLV", "CPER", "COPX"],
    "energy": ["USO", "UNG", "DBC", "XLE", "XOP"],
    "dollar_fx": ["UUP", "DX-Y.NYB", "FXE", "FXY"],
    "intl": ["EFA", "EEM", "FXI", "EWJ", "EWZ"],
    "vol": ["^VIX", "^VIX3M", "^MOVE", "SVXY", "UVXY"],
}
cols = ["ret_1d", "ret_5d", "ret_21d", "ret_63d", "rank_5d", "rank_21d", "rank_63d", "z10",
        "atr_pct", "dist_52w_high_pct", "dist_52w_low_pct", "dist_sma200_pct", "vol_vs_63d"]


def row(tk: str) -> str:
    d = t.get(tk)
    if d is None:
        return f"{tk:10s} MISSING"
    return f"{tk:10s} " + " ".join(f"{c.replace('dist_','').replace('_pct','')}={d.get(c)}" for c in cols)


for cls, tks in CLASSES.items():
    print(f"== {cls}")
    for tk in tks:
        print(row(tk))

print("\n== all tickers: sectors/other (ret_5d, ret_21d, z10, d200, d52h)")
listed = {x for v in CLASSES.values() for x in v}
others = sorted((k for k in t if k not in listed), key=lambda k: t[k].get("z10") or 0)
for k in others:
    d = t[k]
    print(f"{k:6s} r5={d.get('ret_5d')} r21={d.get('ret_21d')} r63={d.get('ret_63d')} z10={d.get('z10')} d200={d.get('dist_sma200_pct')} d52h={d.get('dist_52w_high_pct')} vol={d.get('vol_vs_63d')}")
