"""Crisis-alpha: check ticker availability in long-format master_prices."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
mp = pd.read_parquet(ROOT / "data/master_prices.parquet", columns=["ticker", "date"])
tickers = set(mp["ticker"].unique())
print("n tickers:", len(tickers))

candidates = ["VIXY", "VXX", "UVXY", "SVXY", "VXZ", "VIXM", "GLD", "GDX", "TLT", "IEF",
              "SHY", "DBMF", "KMLM", "CTA", "PDBC", "DBC", "UUP", "SH", "SPY", "QQQ",
              "^VIX", "^VIX3M", "^VVIX", "BTAL", "TAIL", "IVOL", "GLDM", "SGOL",
              "ZROZ", "EDV", "TMF", "USO", "SLV", "^IRX", "^SKEW", "^MOVE", "IAU",
              "PFIX", "CAOS", "GBTC", "FXY", "FXF", "TBT", "VGLT", "SPTL", "BIL"]
avail = []
for c in candidates:
    ok = c in tickers
    if ok:
        rng = mp.loc[mp.ticker == c, "date"]
        print(f"  {c}: YES  {rng.min().date()} -> {rng.max().date()}  ({len(rng)} bars)")
        avail.append(c)
    else:
        print(f"  {c}: no")

# any vol-ish tickers?
volish = [t for t in tickers if any(k in t.upper() for k in ["VIX", "VXX", "TAIL", "VOL"])]
print("\nvol-ish tickers present:", sorted(volish)[:30])
mfut = [t for t in tickers if t in ("DBMF","KMLM","CTA","WTMF","FMF","MFUT")]
print("managed futures:", mfut)
