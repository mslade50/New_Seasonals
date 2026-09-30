"""B1 survey helper: print the whole tape, cross-asset proxies first, then sorted extremes."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
tape = json.loads((ROOT / "data" / "pitch_tape.json").read_text())["tickers"]

CLASSES = {
    "us_large": ["SPY", "QQQ", "^GSPC", "^NDX", "RSP"],
    "us_small": ["IWM"],
    "rates": ["TLT", "IEF", "SHY", "^TNX", "^IRX", "TMF", "TMV"],
    "credit": ["HYG", "LQD", "JNK"],
    "gold": ["GLD", "GDX", "GDXJ", "NUGT"],
    "metals": ["SLV", "CPER", "COPX"],
    "energy": ["USO", "UNG", "DBC", "XLE", "XOP", "BNO"],
    "fx": ["UUP", "DX-Y.NYB", "FXE", "FXY"],
    "intl": ["EFA", "EEM", "FXI", "EWJ", "EWZ", "INDA", "EWG", "KWEB"],
    "vol": ["^VIX", "^VIX3M", "^MOVE", "SVXY", "UVXY", "VXX", "^SKEW"],
    "sectors": ["XLK", "XLF", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC", "SMH", "KRE", "XBI", "IBB", "ITB", "XHB"],
}
F = ["ret_1d", "ret_5d", "ret_21d", "ret_63d", "rank_5d", "rank_21d", "rank_63d", "z10", "atr_pct",
     "rvol21_ann", "dist_52w_high_pct", "dist_52w_low_pct", "dist_sma200_pct", "vol_vs_63d"]


def row(t: str) -> str:
    d = tape.get(t)
    if d is None:
        return f"{t:10s} --"
    return f"{t:10s} " + " ".join(f"{d.get(k) if d.get(k) is not None else 'na':>7}" for k in F) + f" {d.get('date')}"


print(" " * 11 + " ".join(f"{k[:7]:>7}" for k in F))
for c, ts in CLASSES.items():
    print(f"--- {c}")
    for t in ts:
        print(row(t))

print("\nall tickers:", len(tape))
print(sorted(tape))
for k, n in [("dist_sma200_pct", 10), ("rank_63d", 10), ("z10", 10), ("ret_5d", 10), ("vol_vs_63d", 8)]:
    vals = sorted(((d[k], t) for t, d in tape.items() if d.get(k) is not None))
    print(f"\n{k} low:", vals[:n])
    print(f"{k} high:", vals[-n:])
