"""Sort the whole tape by every extreme axis, and print the class table."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
t = json.load(open(ROOT / "data/pitch_tape.json"))
tk = t["tickers"]

CLASSES = {
    "us_large": ["SPY", "QQQ", "^GSPC", "^NDX", "DIA"],
    "us_small": ["IWM", "MDY"],
    "rates": ["TLT", "IEF", "^TNX", "SHY", "TIP", "AGG", "BND", "TBT", "TMF", "TMV"],
    "credit": ["HYG", "LQD", "JNK", "EMB"],
    "gold_miners": ["GLD", "GDX", "GDXJ", "IAU", "NEM"],
    "other_metals": ["SLV", "PPLT", "PALL", "COPX", "CPER", "SIL"],
    "energy": ["USO", "UNG", "DBC", "XLE", "XOP", "OIH", "BNO", "UGA"],
    "dollar_fx": ["UUP", "DX-Y.NYB", "FXE", "FXY", "FXB", "FXF", "UDN"],
    "international": ["EFA", "EEM", "FXI", "EWZ", "EWJ", "EWG", "EWW", "INDA", "VGK", "ASHR"],
    "volatility": ["^VIX", "^VIX3M", "^MOVE", "SVXY", "VXX", "UVXY", "^SKEW"],
    "sectors": ["XLK", "XLV", "XLF", "XLE", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",
                "SMH", "IBB", "IHI", "KRE", "ITB", "XRT", "XME", "KBE"],
}

print("=== CLASS TABLE (present in tape) ===")
for cls, names in CLASSES.items():
    rows = []
    for n in names:
        d = tk.get(n)
        if not d:
            continue
        rows.append((n, d["ret_5d"], d["ret_21d"], d["rank_5d"], d["rank_21d"], d["rank_63d"],
                     d["z10"], d["dist_52w_high_pct"], d["dist_52w_low_pct"], d["dist_sma200_pct"], d["atr_pct"]))
    print(f"\n--- {cls} ---")
    print(f"{'tkr':<10}{'r5':>7}{'r21':>8}{'rk5':>7}{'rk21':>7}{'rk63':>7}{'z10':>7}{'52wh':>8}{'52wl':>8}{'200d':>8}{'atr%':>7}")
    for r in rows:
        print(f"{r[0]:<10}{r[1]:>7.2f}{r[2]:>8.2f}{r[3]:>7.1f}{r[4]:>7.1f}{r[5]:>7.1f}{r[6]:>7.2f}{r[7]:>8.2f}{r[8]:>8.2f}{r[9]:>8.2f}{r[10]:>7.2f}")

def top(key, n=14, rev=True):
    xs = sorted(tk.items(), key=lambda kv: kv[1].get(key, 0), reverse=rev)[:n]
    return [(k, round(v.get(key, 0), 2)) for k, v in xs]

print("\n\n=== WHOLE-TAPE EXTREMES (218 names) ===")
for key in ["rank_5d", "rank_21d", "rank_63d", "z10", "dist_52w_high_pct", "dist_52w_low_pct", "dist_sma200_pct", "ret_1d", "ret_5d", "vol_vs_63d", "atr_pct"]:
    print(f"\n{key} HIGH: {top(key)}")
    print(f"{key} LOW : {top(key, rev=False)}")
