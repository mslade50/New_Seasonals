"""Stage B1 helper: sort the whole tape and dump the extremes by class."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
tape = json.loads((ROOT / "data" / "pitch_tape.json").read_text())["tickers"]

CLASSES = {
    "us_large": ["SPY", "QQQ", "^GSPC", "^NDX", "DIA", "RSP"],
    "us_small_breadth": ["IWM", "MDY", "IJR"],
    "rates": ["TLT", "IEF", "^TNX", "SHY", "TIP", "AGG", "BND"],
    "credit": ["HYG", "LQD", "JNK", "EMB"],
    "gold_miners": ["GLD", "GDX", "GDXJ", "IAU", "NEM", "AEM"],
    "other_metals": ["SLV", "PPLT", "PALL", "COPX", "CPER", "HG=F"],
    "energy": ["USO", "UNG", "DBC", "XLE", "XOP", "OIH", "CL=F", "NG=F"],
    "dollar_fx": ["UUP", "DX-Y.NYB", "FXE", "FXY", "FXB", "6E=F"],
    "international": ["EFA", "EEM", "FXI", "EWZ", "EWJ", "EWG", "INDA", "EWY", "EWT"],
    "volatility": ["^VIX", "^VIX3M", "^MOVE", "SVXY", "VXX", "UVXY", "^SKEW"],
    "sectors": ["XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",
                "SMH", "XBI", "IBB", "KRE", "ITB", "XRT", "IYT", "XME", "IHI", "KIE", "PAVE"],
}

def row(t):
    d = tape.get(t)
    if not d:
        return None
    return d

print("=== available tickers:", len(tape))
missing = []
for cls, names in CLASSES.items():
    have = [n for n in names if n in tape]
    miss = [n for n in names if n not in tape]
    missing += [(cls, n) for n in miss]
    print(f"\n### {cls}  ({len(have)}/{len(names)} in tape)")
    print(f"{'tkr':<10}{'1d':>7}{'5d':>8}{'21d':>8}{'63d':>8}{'252d':>9}{'r5':>6}{'r21':>6}{'r63':>6}{'z10':>7}{'52wH':>8}{'52wL':>8}{'200d':>8}{'atr%':>7}")
    for n in have:
        d = tape[n]
        print(f"{n:<10}{d['ret_1d']:>7.2f}{d['ret_5d']:>8.2f}{d['ret_21d']:>8.2f}{d['ret_63d']:>8.2f}{d['ret_252d']:>9.2f}"
              f"{d['rank_5d']:>6.1f}{d['rank_21d']:>6.1f}{d['rank_63d']:>6.1f}{d['z10']:>7.2f}"
              f"{d['dist_52w_high_pct']:>8.2f}{d['dist_52w_low_pct']:>8.2f}{d['dist_sma200_pct']:>8.2f}{d['atr_pct']:>7.2f}")
    if miss:
        print("  not in tape:", ", ".join(miss))

print("\n\n=== GLOBAL EXTREMES (all 218) ===")
def top(key, n=12, rev=True, label=""):
    rows = [(k, v[key]) for k, v in tape.items() if v.get(key) is not None]
    rows.sort(key=lambda x: x[1], reverse=rev)
    print(f"\n{label or key} {'high' if rev else 'low'}:")
    print("  " + ", ".join(f"{k} {v:.2f}" for k, v in rows[:n]))

for key in ["rank_5d", "rank_21d", "rank_63d", "z10", "dist_52w_high_pct", "dist_52w_low_pct",
            "dist_sma200_pct", "ret_1d", "ret_5d", "ret_21d", "atr_pct", "vol_vs_63d"]:
    top(key, rev=True)
    top(key, rev=False)
