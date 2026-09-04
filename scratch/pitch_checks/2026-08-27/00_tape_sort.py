"""Stage B1 helper: sort the whole 218-name tape every way that matters."""
import json
from pathlib import Path

T = json.load(open(Path(__file__).resolve().parents[3] / "data/pitch_tape.json"))
tk = T["tickers"]
print(f"asof {T['asof']}  freshest {T['freshest_bar']}  n={len(tk)}")

CLASSES = {
    "us_large": ["SPY", "QQQ", "^GSPC", "^NDX", "DIA", "VOO", "IVV", "RSP", "MDY"],
    "us_small": ["IWM", "IJR", "IWN", "IWO"],
    "rates": ["TLT", "IEF", "SHY", "AGG", "^TNX", "TMF", "TMV", "BND", "TLH"],
    "credit": ["HYG", "LQD", "JNK", "EMB"],
    "gold_miners": ["GLD", "GDX", "GDXJ", "IAU", "NEM", "AEM", "GOLD", "SIL"],
    "other_metals": ["SLV", "PPLT", "PALL", "COPX", "CPER", "FCX"],
    "energy": ["USO", "UNG", "DBC", "XLE", "XOP", "OIH", "BNO", "CL=F", "NG=F"],
    "dollar_fx": ["UUP", "DX-Y.NYB", "FXE", "FXY", "DX=F", "6E=F"],
    "international": ["EFA", "EEM", "FXI", "EWJ", "EWZ", "EWG", "INDA", "VGK"],
    "volatility": ["^VIX", "^VIX3M", "^MOVE", "SVXY", "VXX", "UVXY", "^SKEW"],
    "sectors": ["XLK", "XLV", "XLF", "XLE", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC", "KRE", "SMH", "IHI", "IBB", "ITB", "XRT", "KBE"],
}

def row(t):
    d = tk.get(t)
    if not d:
        return f"  {t:<10} MISSING"
    return (f"  {t:<10} px {d['close']:>9.2f}  1d {d['ret_1d']:>6.2f}  5d {d['ret_5d']:>7.2f}  "
            f"21d {d['ret_21d']:>7.2f}  63d {d['ret_63d']:>8.2f}  252d {d['ret_252d']:>8.2f} | "
            f"r5 {d['rank_5d']:>5.1f} r21 {d['rank_21d']:>5.1f} r63 {d['rank_63d']:>5.1f} | "
            f"z10 {d['z10']:>6.2f} atr% {d['atr_pct']:>5.2f} | "
            f"52wH {d['dist_52w_high_pct']:>7.2f} 52wL {d['dist_52w_low_pct']:>8.2f} 200d {d['dist_sma200_pct']:>7.2f}")

print("\n########## BY CLASS ##########")
for c, names in CLASSES.items():
    print(f"\n=== {c} ===")
    for t in names:
        print(row(t))

def top(key, n=14, rev=True, label=""):
    xs = [(v[key], k) for k, v in tk.items() if v.get(key) is not None]
    xs.sort(reverse=rev)
    print(f"\n--- {label or key} {'high' if rev else 'low'} ---")
    for val, k in xs[:n]:
        print(f"  {k:<12} {val:>9.2f}")

print("\n########## EXTREMES (whole tape) ##########")
for k in ["rank_5d", "rank_21d", "rank_63d", "z10", "dist_52w_high_pct", "dist_52w_low_pct", "dist_sma200_pct", "ret_1d", "ret_5d", "ret_21d", "atr_pct", "vol_vs_63d"]:
    top(k, rev=True)
    top(k, rev=False)
