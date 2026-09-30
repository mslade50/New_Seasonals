import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
t = json.loads((ROOT / "data" / "pitch_tape.json").read_text())
tk = t["tickers"]
first = next(iter(tk.values()))
print("fields:", list(first.keys()))

CLASSES = {
    "us_large": ["SPY", "QQQ", "^GSPC", "^NDX", "DIA"],
    "us_small": ["IWM"],
    "rates": ["TLT", "IEF", "SHY", "^TNX", "^IRX", "TMF", "TMV"],
    "credit": ["HYG", "LQD", "JNK"],
    "gold_miners": ["GLD", "GDX", "GDXJ", "NUGT", "DUST", "JNUG", "JDST"],
    "other_metals": ["SLV", "CPER", "PPLT"],
    "energy": ["USO", "UNG", "DBC", "XLE", "XOP", "BNO", "UCO", "SCO", "VLO"],
    "dollar_fx": ["UUP", "DX-Y.NYB", "FXE", "FXY", "FXB", "FXA", "FXC", "FXF", "EURUSD=X", "JPY=X"],
    "intl": ["EFA", "EEM", "FXI", "EWJ", "EWZ", "EWG", "EWW", "INDA", "KWEB", "EWY", "EWT"],
    "vol": ["^VIX", "^VIX3M", "^MOVE", "SVXY", "UVXY", "^VVIX", "^SKEW"],
}
keys = ["ret_1d", "ret_5d", "ret_21d", "ret_63d", "rank_5d", "rank_21d", "rank_63d", "z10",
        "dist_52w_high_pct", "dist_52w_low_pct", "dist_200d_pct", "atr_pct"]


def row(sym):
    d = tk.get(sym)
    if d is None:
        return f"{sym:10s} MISSING"
    vals = []
    for k in keys:
        if k in d:
            v = d[k]
            vals.append(f"{k}={v:.2f}" if isinstance(v, (int, float)) and v is not None else f"{k}={v}")
    return f"{sym:10s} " + " ".join(vals)


for c, syms in CLASSES.items():
    print(f"\n== {c}")
    for s in syms:
        if s in tk:
            print(row(s))

listed = {s for v in CLASSES.values() for s in v}
print("\n== other tickers (sector ETFs etc)")
for s in sorted(tk):
    if s not in listed and (s.startswith("X") or s.startswith("I") or len(s) <= 3 and s.isupper() and s in
                            ["VNQ", "SMH", "KRE", "KBE", "ITB", "XHB", "IYR", "IBB"]):
        print(row(s))
print("\nall tickers:", " ".join(sorted(tk)))
