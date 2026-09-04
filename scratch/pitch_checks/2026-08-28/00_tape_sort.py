"""Sort the whole 218-name tape for the 2026-08-28 surface map."""
import json
from pathlib import Path

T = json.load(open(Path(__file__).resolve().parents[3] / "data/pitch_tape.json"))
tk = T["tickers"]
print(f"asof {T['asof']} freshest {T['freshest_bar']} n={len(tk)}")

def show(title, key, rev=True, n=14, fmt="{:.2f}"):
    rows = [(k, v.get(key)) for k, v in tk.items() if v.get(key) is not None]
    rows.sort(key=lambda x: x[1], reverse=rev)
    print(f"\n--- {title} ---")
    print("  " + " | ".join(f"{k}:{fmt.format(v)}" for k, v in rows[:n]))

show("TOP ret_5d", "ret_5d")
show("BOT ret_5d", "ret_5d", rev=False)
show("TOP ret_21d", "ret_21d")
show("BOT ret_21d", "ret_21d", rev=False)
show("TOP ret_63d", "ret_63d")
show("BOT ret_63d", "ret_63d", rev=False)
show("TOP ret_252d", "ret_252d")
show("BOT ret_252d", "ret_252d", rev=False)
show("TOP z10", "z10")
show("BOT z10", "z10", rev=False)
show("NEAREST 52w high", "dist_52w_high_pct")
show("FURTHEST from 52w high", "dist_52w_high_pct", rev=False, n=18)
show("NEAREST 52w low", "dist_52w_low_pct", rev=False)
show("MOST above 200d", "dist_sma200_pct")
show("MOST below 200d", "dist_sma200_pct", rev=False)
show("TOP rank_5d", "rank_5d")
show("BOT rank_5d", "rank_5d", rev=False)
show("TOP rank_21d", "rank_21d")
show("BOT rank_21d", "rank_21d", rev=False)
show("TOP rank_63d", "rank_63d")
show("BOT rank_63d", "rank_63d", rev=False)
show("HIGHEST vol_vs_63d", "vol_vs_63d")
show("LOWEST vol_vs_63d", "vol_vs_63d", rev=False)
show("HIGH rvol21", "rvol21_ann")

CLASSES = {
    "us_large": ["SPY", "QQQ", "^GSPC", "^NDX", "DIA", "RSP"],
    "us_small": ["IWM", "IJR", "MDY"],
    "rates": ["TLT", "IEF", "^TNX", "SHY", "TIP", "AGG", "BND", "TMF", "TMV"],
    "credit": ["HYG", "LQD", "JNK", "EMB"],
    "gold_miners": ["GLD", "GDX", "GDXJ", "IAU", "NEM", "AEM", "GOLD"],
    "other_metals": ["SLV", "PPLT", "PALL", "COPX", "FCX", "XME"],
    "energy": ["USO", "UNG", "DBC", "XLE", "XOP", "OIH", "COP", "CVX", "SLB", "EOG", "HAL", "OXY", "VLO", "WMB"],
    "dollar_fx": ["UUP", "DX-Y.NYB", "FXE", "FXY", "UDN"],
    "international": ["EFA", "EEM", "FXI", "EWJ", "EWZ", "EWG", "INDA", "VGK"],
    "volatility": ["^VIX", "^VIX3M", "^MOVE", "SVXY", "UVXY", "VXX", "^SKEW"],
    "sectors": ["XLK", "XLV", "XLF", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC", "SMH", "KRE", "IHI", "ITB", "IYT"],
}
print("\n\n===== BY CLASS =====")
for cls, names in CLASSES.items():
    print(f"\n[{cls}]")
    for nm in names:
        v = tk.get(nm)
        if not v:
            print(f"  {nm:10s} ABSENT")
            continue
        print(f"  {nm:10s} c={v['close']:9.2f} 1d={v['ret_1d']:+6.2f} 5d={v['ret_5d']:+7.2f} 21d={v['ret_21d']:+7.2f} "
              f"63d={v['ret_63d']:+7.2f} 252d={v['ret_252d']:+8.2f} | rk5={v['rank_5d']:5.1f} rk21={v['rank_21d']:5.1f} "
              f"rk63={v['rank_63d']:5.1f} z10={v['z10']:+5.2f} | 52wH={v['dist_52w_high_pct']:+7.2f} 52wL={v['dist_52w_low_pct']:+8.2f} "
              f"200d={v['dist_sma200_pct']:+7.2f} atr%={v['atr_pct']:5.2f} rv21={v['rvol21_ann']:5.1f}")

present = set(tk)
listed = {n for v in CLASSES.values() for n in v}
print("\nUNCLASSED (%d):" % len(present - listed))
print("  ", " ".join(sorted(present - listed)))
