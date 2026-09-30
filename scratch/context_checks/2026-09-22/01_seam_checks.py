"""Tonight's big futures prints against their ETF twins: real move or roll seam?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

PAIRS = [("CL=F", "USO"), ("HE=F", None), ("NG=F", "UNG"), ("HG=F", "CPER"),
         ("SI=F", "SLV"), ("GC=F", "GLD"), ("PL=F", "PPLT"), ("NQ=F", "QQQ"),
         ("ES=F", "SPY"), ("ZC=F", "CORN")]
tick = sorted({t for p in PAIRS for t in p if t})
tick += ["XLE", "XLK", "XLF", "XLV", "XLP", "XLI", "XLY", "XLU", "XLB", "XLRE", "XLC",
         "^GSPC", "^NDX", "^RUT", "RSP", "^VIX"]
px = load_prices(tick)

for fut, etf in PAIRS:
    if fut not in px:
        continue
    f = px[fut].tail(7).copy()
    f["ret"] = 100 * f["Close"].pct_change()
    f["gap"] = 100 * (f["Open"] / f["Close"].shift(1) - 1)
    f["intra"] = 100 * (f["Close"] / f["Open"] - 1)
    out = f[["Open", "High", "Low", "Close", "Volume", "ret", "gap", "intra"]].round(3)
    if etf and etf in px:
        e = px[etf]["Close"].pct_change().mul(100).round(2)
        out[f"{etf}_ret"] = e.reindex(out.index)
    print(f"\n--- {fut} vs {etf}")
    print(out.tail(6).to_string())

print("\n--- sectors + indices, last 2 sessions ret %")
cp = pd.DataFrame({t: px[t]["Close"] for t in px if t.startswith("XL") or t in
                   ("^GSPC", "^NDX", "^RUT", "RSP", "SPY", "QQQ", "^VIX")})
print(cp.pct_change().mul(100).round(2).tail(3).T.to_string())
