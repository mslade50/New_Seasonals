"""Stage B1 enumeration 2: sort the WHOLE tape, report extremes by class.

Not a check. This is the survey instrument for the surface map. It exists so
the map is built from a sort of all 217 names rather than from whatever
tickers came to mind, which is exactly how 2026-08-07's first run missed
NFP x rates.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TAPE = json.loads((ROOT / "data" / "pitch_tape.json").read_text(encoding="utf-8"))
T = TAPE["tickers"]

CLASSES = {
    "us_large": ["SPY", "QQQ", "^GSPC", "^NDX"],
    "us_small": ["IWM"],
    "rates": ["TLT", "IEF", "^TNX"],
    "credit": ["HYG", "LQD"],
    "gold_miners": ["GLD", "GDX"],
    "other_metals": ["SLV"],
    "energy": ["USO", "UNG", "DBC"],
    "dollar_fx": ["UUP", "DX-Y.NYB"],
    "international": ["EFA", "EEM", "FXI"],
    "volatility": ["^VIX", "^VIX3M", "^MOVE", "SVXY"],
}


def row(t):
    d = T[t]
    return (f"{t:<10} px{d['close']:>9.2f}  r5{d['ret_5d']:>7.2f}  "
            f"r21{d['ret_21d']:>7.2f}  r63{d['ret_63d']:>7.2f}  "
            f"rk5{d['rank_5d']:>6.1f} rk21{d['rank_21d']:>6.1f} "
            f"rk63{d['rank_63d']:>6.1f}  z10{d['z10']:>6.2f}  "
            f"52wh{d['dist_52w_high_pct']:>7.2f} 52wl{d['dist_52w_low_pct']:>8.2f}  "
            f"200d{d['dist_sma200_pct']:>7.2f}  atr%{d['atr_pct']:>5.2f}")


print("=" * 118)
print("PART 1 - THE NAMED CROSS-ASSET GRID (every class in the B1 table)")
print("=" * 118)
for cls, tickers in CLASSES.items():
    print(f"\n[{cls}]")
    for t in tickers:
        print("  " + row(t) if t in T else f"  {t:<10} NOT IN TAPE")

print("\n" + "=" * 118)
print("PART 2 - EXTREMES ACROSS ALL 217 NAMES (sorted, not sampled)")
print("=" * 118)


def top(field, n=12, reverse=True, label=""):
    ok = [t for t in T if T[t].get(field) is not None]
    ranked = sorted(ok, key=lambda t: T[t][field], reverse=reverse)[:n]
    print(f"\n--- {label or field} ---")
    for t in ranked:
        print("  " + row(t))


top("dist_52w_high_pct", reverse=True, label="CLOSEST TO 52w HIGH")
top("dist_52w_low_pct", reverse=False, label="CLOSEST TO 52w LOW")
top("rank_5d", reverse=True, label="HOTTEST 5d RANK")
top("rank_5d", reverse=False, label="COLDEST 5d RANK")
top("rank_63d", reverse=True, label="HOTTEST 63d RANK")
top("rank_63d", reverse=False, label="COLDEST 63d RANK")
top("z10", reverse=True, label="HIGHEST z10")
top("z10", reverse=False, label="LOWEST z10")
top("dist_sma200_pct", reverse=True, label="MOST EXTENDED vs 200d")
top("dist_sma200_pct", reverse=False, label="MOST BELOW 200d")
top("vol_vs_63d", reverse=True, label="BIGGEST VOLUME SURGE vs 63d")

print("\n" + "=" * 118)
print("PART 3 - NON-EQUITY NAMES ONLY, sorted by 5d rank")
print("=" * 118)
NON_EQ = set(sum(CLASSES.values(), [])) - {"SPY", "QQQ", "^GSPC", "^NDX", "IWM"}
extra = ["GDX", "SLV", "GLD", "USO", "UNG", "DBC", "UUP", "DX-Y.NYB", "TLT",
         "IEF", "HYG", "LQD", "EFA", "EEM", "FXI", "^TNX", "^VIX", "^VIX3M",
         "^MOVE", "SVXY", "VNQ", "XLE", "XLU", "XLP", "XLV", "XLF", "XLK",
         "XLI", "XLB", "XLY", "XLRE", "XLC", "SMH", "IBB", "KRE", "XBI"]
pool = [t for t in dict.fromkeys(list(NON_EQ) + extra) if t in T]
for t in sorted(pool, key=lambda x: T[x]["rank_5d"], reverse=True):
    print("  " + row(t))
