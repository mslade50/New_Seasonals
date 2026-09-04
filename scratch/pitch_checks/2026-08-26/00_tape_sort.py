"""Sort the whole tape on every axis. Survey input for 00_surface_map.md."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
t = json.load(open(ROOT / "data/pitch_tape.json"))
tk = t["tickers"]
print("freshest bar:", t["freshest_bar"], "| n =", len(tk))

CLASSES = {
    "us_large": ["SPY", "QQQ", "^GSPC", "^NDX", "DIA"],
    "us_small": ["IWM", "MDY"],
    "rates": ["TLT", "IEF", "SHY", "^TNX", "TMF", "TMV", "AGG"],
    "credit": ["HYG", "LQD", "JNK"],
    "gold_miners": ["GLD", "GDX", "GDXJ", "NEM", "NUGT", "DUST"],
    "metals": ["SLV", "PPLT", "PALL", "HG=F", "SI=F", "COPX", "FCX"],
    "energy": ["USO", "UNG", "DBC", "XLE", "XOP", "OIH", "CL=F", "NG=F"],
    "dollar_fx": ["UUP", "DX-Y.NYB", "FXE", "FXY", "6E=F", "6J=F"],
    "intl": ["EFA", "EEM", "FXI", "EWJ", "EWZ", "EWG", "INDA"],
    "vol": ["^VIX", "^VIX3M", "^MOVE", "SVXY", "UVXY", "VXX", "^SKEW"],
    "sectors": ["XLK","XLV","XLF","XLI","XLY","XLP","XLE","XLU","XLB","XLRE","XLC","SMH","IHI","KRE","XBI","IYT","ITB"],
}

def _f(v, w, p=2):
    return ("{:>%d.%df}" % (w, p)).format(v) if isinstance(v, (int, float)) else "{:>{w}}".format("na", w=w)


def row(k, d):
    d = {kk: (vv if isinstance(vv, (int, float)) else None) for kk, vv in d.items()}
    return (f"{k:<10} c={_f(d['close'],9)} r1={_f(d['ret_1d'],6)} r5={_f(d['ret_5d'],7)} "
            f"r21={_f(d['ret_21d'],7)} r63={_f(d['ret_63d'],7)} r252={_f(d['ret_252d'],8)} "
            f"| rk5={_f(d['rank_5d'],5,1)} rk21={_f(d['rank_21d'],5,1)} rk63={_f(d['rank_63d'],5,1)} "
            f"z10={_f(d['z10'],6)} | 52wh={_f(d['dist_52w_high_pct'],7)} 52wl={_f(d['dist_52w_low_pct'],8)} "
            f"200d={_f(d['dist_sma200_pct'],7)} atr%={_f(d['atr_pct'],5)} vol/63={_f(d['vol_vs_63d'],5)}")

print("\n===== BY CLASS =====")
for cls, names in CLASSES.items():
    print(f"\n-- {cls}")
    for n in names:
        if n in tk:
            print("  " + row(n, tk[n]))

def top(key, n=14, rev=True, label=""):
    items = sorted(tk.items(), key=lambda kv: kv[1].get(key, 0) or 0, reverse=rev)[:n]
    print(f"\n-- {label or key} ({'high' if rev else 'low'})")
    for k, d in items:
        print("  " + row(k, d))

print("\n===== EXTREMES (whole tape) =====")
for key in ["ret_1d", "ret_5d", "ret_21d", "ret_63d", "z10", "dist_52w_high_pct",
            "dist_sma200_pct", "vol_vs_63d", "atr_pct"]:
    top(key, 12, True)
    top(key, 12, False)
