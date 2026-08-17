"""Stage B1 tape survey: sort the whole 218-name tape by class and by extreme.

Prints the raw material for 00_surface_map.md. No thesis here, just the sort.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

TAPE = json.load(open(Path(__file__).resolve().parents[3] / "data" / "pitch_tape.json"))
T = TAPE["tickers"]

CLASSES = {
    "us_large": ["SPY", "QQQ", "^GSPC", "^NDX", "DIA", "VOO", "IVV"],
    "us_small_breadth": ["IWM", "MDY", "RSP", "IJR"],
    "rates": ["TLT", "IEF", "SHY", "^TNX", "TMF", "TMV", "TBT", "AGG", "BND", "TIP"],
    "credit": ["HYG", "LQD", "JNK", "EMB"],
    "gold_miners": ["GLD", "GDX", "GDXJ", "IAU", "NEM", "GOLD", "AEM", "NUGT", "DUST"],
    "other_metals": ["SLV", "SIL", "PPLT", "PALL", "COPX", "FCX", "JJC"],
    "energy": ["USO", "UNG", "DBC", "XLE", "XOP", "OIH", "BNO", "UCO", "ERX", "ERY"],
    "dollar_fx": ["UUP", "DX-Y.NYB", "FXE", "FXY", "FXB", "UDN"],
    "international": ["EFA", "EEM", "FXI", "EWZ", "EWJ", "EWG", "INDA", "EWY", "EWT", "VGK", "ASHR"],
    "volatility": ["^VIX", "^VIX3M", "^VVIX", "^MOVE", "SVXY", "VXX", "UVXY", "^SKEW"],
    "sectors": ["XLK", "XLF", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC",
                "SMH", "SOXX", "IHI", "IBB", "XBI", "KRE", "KBE", "ITB", "XHB", "JETS",
                "XRT", "IYT", "VNQ", "TAN", "ICLN", "ARKK", "HACK", "XME", "PAVE"],
    "crypto_adjacent": ["BTC-USD", "ETH-USD", "COIN", "MSTR", "MARA", "RIOT", "IBIT"],
}


def has(t):
    return t in T


def row(t):
    d = T[t]
    return (f"{t:<10} {d['close']:>10.2f} 1d{d['ret_1d']:>7.2f} 5d{d['ret_5d']:>7.2f} "
            f"21d{d['ret_21d']:>7.2f} 63d{d['ret_63d']:>7.2f} 252d{d['ret_252d']:>8.2f} | "
            f"r5{d['rank_5d']:>6.1f} r21{d['rank_21d']:>6.1f} r63{d['rank_63d']:>6.1f} | "
            f"z10{d['z10']:>6.2f} atr%{d['atr_pct']:>5.2f} | "
            f"52wh{d['dist_52w_high_pct']:>7.2f} 52wl{d['dist_52w_low_pct']:>8.2f} "
            f"200d{d['dist_sma200_pct']:>7.2f}")


print("=" * 150)
print(f"TAPE {TAPE['asof']} freshest bar {TAPE['freshest_bar']}  n={len(T)}")
print("=" * 150)

covered = set()
for cls, names in CLASSES.items():
    live = [n for n in names if has(n)]
    covered |= set(live)
    if not live:
        continue
    print(f"\n### {cls}  ({len(live)} in tape)")
    for n in sorted(live, key=lambda x: T[x]["rank_5d"]):
        print("  " + row(n))

rest = sorted(set(T) - covered)
print(f"\n### unclassified ({len(rest)})")
print("  " + " ".join(rest))

print("\n" + "=" * 150)
print("CROSS-SECTION EXTREMES (whole tape)")
print("=" * 150)


def top(key, n=12, rev=True, label=""):
    s = sorted(T, key=lambda x: T[x][key], reverse=rev)[:n]
    print(f"\n-- {label or key} ({'high' if rev else 'low'}) --")
    for t in s:
        print("  " + row(t))


top("dist_52w_high_pct", 14, True, "closest to / at 52w high")
top("dist_52w_low_pct", 14, False, "closest to 52w low")
top("rank_5d", 14, True, "5d rank")
top("rank_5d", 14, False, "5d rank")
top("rank_21d", 14, True, "21d rank")
top("rank_21d", 14, False, "21d rank")
top("rank_63d", 14, True, "63d rank")
top("rank_63d", 14, False, "63d rank")
top("z10", 14, True, "z10")
top("z10", 14, False, "z10")
top("dist_sma200_pct", 14, True, "extension above 200d")
top("dist_sma200_pct", 14, False, "extension below 200d")
top("ret_1d", 12, True, "1d move")
top("ret_1d", 12, False, "1d move")
top("vol_vs_63d", 12, True, "volume vs 63d")
