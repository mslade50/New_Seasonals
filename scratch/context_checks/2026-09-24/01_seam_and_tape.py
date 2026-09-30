"""Tonight's tape: which moves are real, which are roll seams, and how last night's items resolved."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

PAIRS = [("CL=F", "USO"), ("HE=F", None), ("NG=F", "UNG"), ("HG=F", "CPER"),
         ("SI=F", "SLV"), ("GC=F", "GLD"), ("PL=F", "PPLT"), ("NQ=F", "QQQ"),
         ("ES=F", "SPY"), ("SB=F", None), ("KC=F", None), ("CT=F", None), ("CC=F", None)]
CORE = ["SPY", "QQQ", "IWM", "^GSPC", "^NDX", "^RUT", "RSP", "TLT", "IEF", "LQD", "HYG",
        "^TNX", "^FVX", "^IRX", "^VIX", "^VIX3M", "^VVIX", "^MOVE", "DX-Y.NYB", "UUP",
        "EURUSD=X", "JPY=X", "GC=F", "GLD", "EEM", "EWZ", "EWJ", "EWW", "BTC-USD"]
tick = sorted({t for p in PAIRS for t in p if t} | set(CORE))
tick += ["XLE", "XLK", "XLF", "XLV", "XLP", "XLI", "XLY", "XLU", "XLB", "XLRE", "XLC", "KRE", "ITB"]
px = load_prices(tick)
print("missing:", [t for t in tick if t not in px])

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
    print(out.tail(4).to_string())

cp = pd.DataFrame({t: px[t]["Close"] for t in px})
print("\n--- core + sectors, last 4 sessions ret % (yields in % change of level)")
print(cp[[c for c in CORE + [t for t in px if t.startswith("XL") or t in ("KRE", "ITB")] if c in cp]]
      .pct_change().mul(100).round(2).tail(4).T.to_string())
print("\n--- levels, last 5")
print(cp[["^TNX", "^FVX", "^IRX", "^VIX", "^VIX3M", "^MOVE", "DX-Y.NYB", "UUP", "TLT"]].tail(5).round(3).to_string())
for t in ["TLT", "IEF", "LQD", "HYG", "SPY", "IWM", "QQQ", "^TNX", "^FVX", "^MOVE", "UUP", "GLD", "EEM"]:
    s = cp[t].dropna()
    print(t, "5d", round(100 * (s.iloc[-1] / s.iloc[-6] - 1), 2), "21d", round(100 * (s.iloc[-1] / s.iloc[-22] - 1), 2),
          "vs 52w hi", round(100 * (s.iloc[-1] / s.tail(252).max() - 1), 2), "vs 52w lo", round(100 * (s.iloc[-1] / s.tail(252).min() - 1), 2),
          "vs 200d", round(100 * (s.iloc[-1] / s.tail(200).mean() - 1), 2))

print("\n--- since the Sep 18 close (post-expiry week)")
for t in ["SPY", "IWM", "QQQ", "^GSPC"]:
    s = cp[t].dropna()
    print(t, round(100 * (s.iloc[-1] / s.loc["2026-09-18"] - 1), 2))

tnx = cp["^TNX"].dropna()
print("\n10y bp last 3:", (tnx.diff() * 100).tail(3).round(1).to_dict())
print("10y 21d change bp:", round(100 * (tnx.iloc[-1] - tnx.iloc[-22]), 1), "63d:", round(100 * (tnx.iloc[-1] - tnx.iloc[-64]), 1))
prev = tnx[tnx >= tnx.iloc[-1]]
print("prior closes >= today's 10y:", [str(d.date()) for d in prev.index[-4:]])
mv = cp["^MOVE"].dropna()
print("MOVE 2-session change %:", round(100 * (mv.iloc[-1] / mv.iloc[-3] - 1), 2), "level", round(mv.iloc[-1], 2),
      "prior close >= today:", str(mv[mv >= mv.iloc[-1]].index[-2].date()) if (mv >= mv.iloc[-1]).sum() > 1 else None)
uup = cp["UUP"].dropna()
print("UUP prior closes >= today:", [str(d.date()) for d in uup[uup >= uup.iloc[-1] - 1e-9].index[-4:]])
