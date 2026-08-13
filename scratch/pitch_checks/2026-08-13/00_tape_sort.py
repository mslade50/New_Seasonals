"""Sort the whole tape for the B1 surface map. Not a check; a survey tool."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
t = json.load(open(ROOT / "data/pitch_tape.json"))
rows = t["tickers"]
recs = [dict(ticker=k, **v) for k, v in rows.items()]
print("asof", t["asof"], "freshest", t["freshest_bar"], "n", len(recs))


def show(key, n=14, rev=True, fmt="{:.2f}"):
    ok = [r for r in recs if r.get(key) is not None]
    ok.sort(key=lambda r: r[key], reverse=rev)
    lo = " ".join(f"{r['ticker']}:{fmt.format(r[key])}" for r in ok[:n])
    print(f"  {'TOP ' if rev else 'BOT '}{key:22s} {lo}")


for k in ["rank_5d", "rank_21d", "rank_63d", "z10", "dist_52w_high_pct",
          "dist_52w_low_pct", "dist_sma200_pct", "atr_pct", "vol_vs_63d",
          "ret_5d", "ret_21d", "ret_63d", "ret_252d"]:
    show(k, rev=True)
    show(k, rev=False)
    print()

# class buckets
CLASSES = {
    "us_large": ["SPY", "QQQ", "^GSPC", "^NDX", "DIA"],
    "us_small": ["IWM"],
    "rates": ["TLT", "IEF", "^TNX"],
    "credit": ["HYG", "LQD"],
    "gold_miners": ["GLD", "GDX", "NEM", "CEF"],
    "other_metals": ["SLV", "XME", "FCX"],
    "energy": ["USO", "UNG", "DBC", "XLE", "XOP"],
    "dollar_fx": ["UUP", "DX-Y.NYB"],
    "international": ["EFA", "EEM", "FXI", "EWZ", "EWJ"],
    "volatility": ["^VIX", "^VIX3M", "^MOVE", "^SKEW", "SVXY", "UVXY"],
    "sectors": ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE",
                "XLU", "XLV", "XLY", "SMH", "XBI", "KRE", "IHI", "VNQ"],
}
print("\nBY CLASS")
for cls, tks in CLASSES.items():
    print(f" {cls}")
    for tk in tks:
        r = rows.get(tk)
        if r is None:
            print(f"   {tk:10s} MISSING")
            continue
        print(f"   {tk:10s} c={r['close']:>9.2f} 5d={r['ret_5d']:>6.2f}/{r['rank_5d']:>5.1f} "
              f"21d={r['ret_21d']:>6.2f}/{r['rank_21d']:>5.1f} 63d={r['ret_63d']:>7.2f}/{r['rank_63d']:>5.1f} "
              f"z10={r['z10']:>5.2f} 52wh={r['dist_52w_high_pct']:>7.2f} 52wl={r['dist_52w_low_pct']:>7.2f} "
              f"200d={r['dist_sma200_pct']:>7.2f} atr%={r['atr_pct']:>5.2f}")
