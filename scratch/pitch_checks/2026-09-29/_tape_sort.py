import json
from pathlib import Path

root = Path(__file__).resolve().parents[3]
tape = json.loads((root / "data" / "pitch_tape.json").read_text(encoding="utf-8"))["tickers"]

CLASSES = {
    "us_large": ["SPY", "QQQ", "^GSPC", "^NDX", "DIA", "RSP"],
    "us_small": ["IWM"],
    "rates": ["TLT", "IEF", "^TNX", "SHY", "TIP"],
    "credit": ["HYG", "LQD"],
    "gold_miners": ["GLD", "GDX", "GDXJ"],
    "metals": ["SLV", "COPX", "XME", "CPER"],
    "energy": ["USO", "UNG", "DBC", "XLE", "XOP"],
    "dollar_fx": ["UUP", "DX-Y.NYB", "FXE", "FXY", "FXB", "FXA", "FXC"],
    "intl": ["EFA", "EEM", "FXI", "EWJ", "EWZ", "INDA", "KWEB"],
    "vol": ["^VIX", "^VIX3M", "^MOVE", "SVXY", "UVXY", "VXX"],
}
cols = ["close", "ret_1d", "ret_5d", "ret_21d", "ret_63d", "rank_5d", "rank_21d", "rank_63d", "z10",
        "atr_pct", "dist_52w_high_pct", "dist_52w_low_pct", "dist_sma200_pct", "vol_vs_63d"]
short = ["cl", "r1", "r5", "r21", "r63", "k5", "k21", "k63", "z10", "atr%", "d52h", "d52l", "d200", "vv"]


def row(t: str, d: dict) -> str:
    vals = []
    for c in cols:
        v = d.get(c)
        vals.append(f"{v:>7.1f}" if isinstance(v, (int, float)) else f"{'na':>7}")
    return f"{t:<9}" + "".join(vals)


hdr = f"{'tkr':<9}" + "".join(f"{s:>7}" for s in short)
print(hdr)
seen = set()
for cls, names in CLASSES.items():
    print(f"-- {cls}")
    for n in names:
        if n in tape:
            print(row(n, tape[n]))
            seen.add(n)
print("-- rest sorted by rank_21d")
rest = sorted((t for t in tape if t not in seen), key=lambda t: tape[t].get("rank_21d") or 0)
for t in rest:
    print(row(t, tape[t]))
