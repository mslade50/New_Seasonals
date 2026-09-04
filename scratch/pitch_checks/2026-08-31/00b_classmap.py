"""Class-by-class tape enumeration for the surface map."""
import json
from pathlib import Path
T = json.load(open(Path(__file__).resolve().parents[3] / "data" / "pitch_tape.json"))["tickers"]

CLASSES = {
 "us_large": ["SPY","QQQ","^GSPC","^NDX","DIA","RSP","MDY"],
 "us_small": ["IWM","IJR"],
 "rates": ["TLT","IEF","SHY","^TNX","AGG","TMF","TMV","^IRX"],
 "credit": ["HYG","LQD","JNK"],
 "gold_miners": ["GLD","GDX","GDXJ","NEM","NUGT","DUST","IAU"],
 "other_metals": ["SLV","PPLT","PALL","COPX","FCX","AGQ"],
 "energy": ["USO","UNG","DBC","XLE","XOP","OIH","BNO","CL=F","NG=F","BOIL","ERX"],
 "dollar_fx": ["UUP","DX-Y.NYB","FXE","FXY","UDN","6E=F","6J=F"],
 "international": ["EFA","EEM","FXI","EWJ","EWZ","EWG","INDA","VGK","ACWX"],
 "volatility": ["^VIX","^VIX3M","^MOVE","SVXY","UVXY","VXX","^SKEW","^VVIX"],
 "sectors": ["XLK","XLV","XLF","XLE","XLI","XLY","XLP","XLU","XLB","XLRE","XLC","SMH","IHI","KRE","ITB","XBI","IYT"],
 "crypto": ["BTC-USD","ETH-USD","IBIT","MSTR","COIN","GBTC"],
 "reits": ["VNQ","IYR"],
}
present = set(T)
for cls, tks in CLASSES.items():
    print(f"\n##### {cls}")
    for tk in tks:
        r = T.get(tk)
        if not r:
            print(f"  {tk:<11} -- not in tape")
            continue
        print(f"  {tk:<11} px={r['close']:>10.2f} 1d={r['ret_1d']:>6.2f} 5d={r['ret_5d']:>7.2f} 21d={r['ret_21d']:>7.2f} 63d={r['ret_63d']:>7.2f} 252d={r['ret_252d']:>8.2f} | r5={r['rank_5d']:>5.1f} r21={r['rank_21d']:>5.1f} r63={r['rank_63d']:>5.1f} z10={r['z10']:>6.2f} | 52wh={r['dist_52w_high_pct']:>7.2f} 52wl={r['dist_52w_low_pct']:>8.2f} 200d={r['dist_sma200_pct']:>7.2f} atr%={r['atr_pct']:>5.2f}")
missing = sorted(present - {t for v in CLASSES.values() for t in v})
print("\n##### unclassified names in tape:", len(missing))
print(", ".join(missing))
