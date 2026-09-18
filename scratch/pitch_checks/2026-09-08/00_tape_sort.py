"""Sort the whole 218-name tape into the B1 class buckets and print extremes."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
t = json.load(open(ROOT / "data/pitch_tape.json"))
T = t["tickers"]

CLASSES = {
    "us_large": ["SPY", "QQQ", "^GSPC", "^NDX", "DIA", "VOO"],
    "us_small_breadth": ["IWM", "MDY", "IJR"],
    "rates": ["TLT", "IEF", "^TNX", "SHY", "TBT", "TMF", "TMV", "AGG", "BND"],
    "credit": ["HYG", "LQD", "JNK", "EMB"],
    "gold_miners": ["GLD", "GDX", "GDXJ", "NEM", "AEM", "NUGT", "DUST"],
    "other_metals": ["SLV", "PPLT", "PALL", "COPX", "FCX", "AGQ", "USLV"],
    "energy": ["USO", "UNG", "DBC", "XLE", "XOP", "OIH", "CL=F", "NG=F", "BNO"],
    "dollar_fx": ["UUP", "DX-Y.NYB", "FXE", "FXY", "FXB", "UDN"],
    "international": ["EFA", "EEM", "FXI", "EWZ", "EWJ", "EWG", "INDA", "VGK"],
    "volatility": ["^VIX", "^VIX3M", "^MOVE", "SVXY", "UVXY", "VXX", "^SKEW", "^VVIX"],
    "sectors": ["XLK", "XLV", "XLF", "XLI", "XLY", "XLP", "XLE", "XLU", "XLB",
                "XLRE", "XLC", "SMH", "IHI", "ITA", "KRE", "XBI", "IBB", "XRT", "XHB"],
}

present = {k: [x for x in v if x in T] for k, v in CLASSES.items()}
missing = {k: [x for x in v if x not in T] for k, v in CLASSES.items()}

def row(tk):
    d = T[tk]
    return (f"{tk:>10} d={d.get('date')} c={d.get('close')} "
            f"r1={d.get('ret_1d'):>7} r5={d.get('ret_5d'):>7} r21={d.get('ret_21d'):>7} "
            f"r63={d.get('ret_63d'):>8} r252={d.get('ret_252d'):>8} | "
            f"rk5={d.get('rank_5d'):>5} rk21={d.get('rank_21d'):>5} rk63={d.get('rank_63d'):>5} "
            f"z10={d.get('z10'):>6} atr%={d.get('atr_pct'):>5} rv21={d.get('rvol21_ann'):>5} "
            f"52h={d.get('dist_52w_high_pct'):>7} 52l={d.get('dist_52w_low_pct'):>8} "
            f"s200={d.get('dist_sma200_pct'):>7}")

print("=" * 150)
print("PER-CLASS TAPE")
for cls, tks in present.items():
    print(f"\n--- {cls} --- (missing from tape: {missing[cls]})")
    for tk in tks:
        print(row(tk))

print("\n" + "=" * 150)
print("WHOLE-TAPE EXTREMES (218 names)")
def top(field, n=12, rev=True, fmt="{:.2f}"):
    vals = [(k, v.get(field)) for k, v in T.items() if v.get(field) is not None]
    vals.sort(key=lambda x: x[1], reverse=rev)
    return ", ".join(f"{k} {fmt.format(v)}" for k, v in vals[:n])

for f in ["rank_5d", "rank_21d", "rank_63d", "z10", "dist_52w_high_pct",
          "dist_sma200_pct", "ret_1d", "ret_5d", "atr_pct", "rvol21_ann"]:
    print(f"\nTOP  {f}: {top(f, rev=True)}")
    print(f"BOT  {f}: {top(f, rev=False)}")

# joint extremes
print("\n" + "=" * 150)
print("JOINT: r21>=90 & r63<=10 (watchlist 28):",
      [k for k, v in T.items() if (v.get('rank_21d') or 0) >= 90 and (v.get('rank_63d') or 100) <= 10])
print("JOINT: r5<=5 & within 5% of 52w high:",
      [k for k, v in T.items() if (v.get('rank_5d') or 100) <= 5 and (v.get('dist_52w_high_pct') or -99) >= -5])
print("JOINT: r5,r21,r63 all <=10:",
      [k for k, v in T.items() if (v.get('rank_5d') or 100) <= 10 and (v.get('rank_21d') or 100) <= 10 and (v.get('rank_63d') or 100) <= 10])
print("AT 52w HIGH (within 0.5%):",
      [k for k, v in T.items() if (v.get('dist_52w_high_pct') or -99) >= -0.5])
print("AT 52w LOW (within 2%):",
      [k for k, v in T.items() if (v.get('dist_52w_low_pct') or 99) <= 2.0])
