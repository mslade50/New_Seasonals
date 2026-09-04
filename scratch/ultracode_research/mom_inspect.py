from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
mp = pd.read_parquet(ROOT / "data" / "master_prices.parquet", columns=["ticker", "date", "Close"])
tickers = set(mp["ticker"].unique())
print("n tickers:", len(tickers))

sectors = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
countries = ["EWA", "EWC", "EWG", "EWH", "EWJ", "EWL", "EWQ", "EWS", "EWU", "EWW", "EWZ",
             "EWT", "EWY", "EWM", "EWP", "EWI", "EWN", "EWD", "EWK", "EWO", "FXI", "MCHI",
             "INDA", "EPI", "EEM", "EFA", "VEA", "VWO", "PIN", "RSX", "TUR", "EZA", "THD",
             "EIDO", "EPHE", "GXG", "ECH", "EPU", "ARGT", "ENOR", "EDEN", "EFNL", "EIRL",
             "GREK", "EIS", "KSA", "UAE", "QAT", "EGPT", "NGE", "VNM"]
present_s = [t for t in sectors if t in tickers]
present_c = [t for t in countries if t in tickers]
print("sectors present:", present_s)
print("countries present:", present_c)

# coverage start dates for present sector/country ETFs + SPY
sub = mp[mp["ticker"].isin(present_s + present_c + ["SPY", "BIL", "SHY", "IEF"])]
g = sub.groupby("ticker")["date"].agg(["min", "max", "count"])
print(g.to_string())
