"""Sort the whole tape by every dimension. Survey input for the B1 surface map."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
t = json.load(open(ROOT / "data/pitch_tape.json"))
ts = t["tickers"]

CLASSES = {
    "us_large": ["SPY", "QQQ", "^GSPC", "^NDX", "DIA", "RSP"],
    "us_small": ["IWM", "IJR", "MDY"],
    "rates": ["TLT", "IEF", "^TNX", "SHY", "TIP", "^TYX", "^FVX"],
    "credit": ["HYG", "LQD", "JNK", "EMB"],
    "gold_miners": ["GLD", "GDX", "GDXJ", "IAU", "SIL"],
    "metals": ["SLV", "PPLT", "COPX", "CPER", "HG=F", "SI=F", "GC=F"],
    "energy": ["USO", "UNG", "DBC", "XLE", "XOP", "OIH", "CL=F", "NG=F", "BNO"],
    "dollar_fx": ["UUP", "DX-Y.NYB", "FXE", "FXY", "FXB", "DX=F", "6E=F", "6J=F"],
    "intl": ["EFA", "EEM", "FXI", "EWJ", "EWZ", "EWG", "EWU", "INDA", "ASHR", "VGK"],
    "vol": ["^VIX", "^VIX3M", "^MOVE", "SVXY", "VXX", "UVXY", "^SKEW", "^VVIX"],
    "sectors": ["XLK", "XLV", "XLF", "XLE", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",
                "SMH", "IHI", "XBI", "IBB", "KRE", "ITB", "XRT", "XME", "JETS"],
}

print("freshest bar:", t["freshest_bar"], " asof:", t["asof"], " n:", len(ts))
present = set(ts)
mapped = set()
for c, names in CLASSES.items():
    have = [n for n in names if n in present]
    mapped |= set(have)
    print(f"\n### {c}  ({len(have)}/{len(names)} in tape)")
    hdr = f"{'tkr':<10}{'1d':>7}{'5d':>8}{'21d':>8}{'63d':>8}{'252d':>9}{'r5':>6}{'r21':>6}{'r63':>6}{'z10':>7}{'52wH':>8}{'52wL':>8}{'200d':>8}{'atr%':>7}"
    print(hdr)
    for n in have:
        d = ts[n]
        print(f"{n:<10}{d['ret_1d']:>7.2f}{d['ret_5d']:>8.2f}{d['ret_21d']:>8.2f}{d['ret_63d']:>8.2f}"
              f"{d['ret_252d']:>9.2f}{d['rank_5d']:>6.1f}{d['rank_21d']:>6.1f}{d['rank_63d']:>6.1f}"
              f"{d['z10']:>7.2f}{d['dist_52w_high_pct']:>8.2f}{d['dist_52w_low_pct']:>8.2f}"
              f"{d['dist_sma200_pct']:>8.2f}{d['atr_pct']:>7.2f}")

print("\n\n### UNMAPPED tickers in tape (single names + others)")
un = sorted(present - mapped)
print(len(un), un)

def top(key, n=12, rev=True, pool=None):
    pool = pool or present
    rows = sorted(((ts[k][key], k) for k in pool if ts[k].get(key) is not None), reverse=rev)
    return [(k, round(v, 2)) for v, k in rows[:n]]

print("\n\n=== EXTREMES, whole tape ===")
for key in ["dist_52w_high_pct", "dist_52w_low_pct", "dist_sma200_pct", "z10",
            "ret_1d", "ret_5d", "ret_21d", "ret_63d", "rank_5d", "rank_21d", "rank_63d", "atr_pct", "vol_vs_63d"]:
    print(f"\n-- {key} HIGH:", top(key, 12, True))
    print(f"-- {key} LOW :", top(key, 12, False))
