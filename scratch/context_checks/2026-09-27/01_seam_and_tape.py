"""Friday's tape: which moves are real, which are roll seams, and how Thursday's items resolved."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

PAIRS = [("CL=F", "USO"), ("HE=F", None), ("NG=F", "UNG"), ("HG=F", "CPER"),
         ("SI=F", "SLV"), ("GC=F", "GLD"), ("PL=F", "PPLT"), ("NQ=F", "QQQ"),
         ("ES=F", "SPY"), ("SB=F", None), ("KC=F", None), ("CT=F", None), ("CC=F", None)]
CORE = ["SPY", "QQQ", "IWM", "^GSPC", "^NDX", "^RUT", "RSP", "TLT", "IEF", "LQD", "HYG",
        "^TNX", "^FVX", "^IRX", "^TYX", "^VIX", "^VIX3M", "^VVIX", "^MOVE", "DX-Y.NYB", "UUP",
        "GLD", "SLV", "EEM", "EFA", "EWJ", "EWZ", "EWW", "FXY", "BTC-USD", "ETH-USD"]
tick = sorted({t for p in PAIRS for t in p if t} | set(CORE))
tick += ["XLE", "XLK", "XLF", "XLV", "XLP", "XLI", "XLY", "XLU", "XLB", "XLRE", "XLC", "KRE", "ITB", "SMH"]
px = load_prices(tick)
print("missing:", [t for t in tick if t not in px])
print("history starts:", {t: str(px[t].index[0].date()) for t in ["^GSPC", "SPY", "TLT", "IEF", "HYG", "^TNX", "GLD", "FXY"] if t in px})

for fut, etf in PAIRS:
    if fut not in px:
        continue
    f = px[fut].tail(7).copy()
    f["ret"] = 100 * f["Close"].pct_change()
    f["gap"] = 100 * (f["Open"] / f["Close"].shift(1) - 1)
    f["intra"] = 100 * (f["Close"] / f["Open"] - 1)
    out = f[["Open", "Close", "Volume", "ret", "gap", "intra"]].round(3)
    if etf and etf in px:
        e = px[etf]["Close"].pct_change().mul(100).round(2)
        out[f"{etf}_ret"] = e.reindex(out.index)
    print(f"\n--- {fut} vs {etf}")
    print(out.tail(3).to_string())

cp = pd.DataFrame({t: px[t]["Close"] for t in px})
cols = [c for c in CORE + [t for t in px if t.startswith("XL") or t in ("KRE", "ITB", "SMH")] if c in cp]
print("\n--- last 5 sessions ret %")
print(cp[cols].pct_change().mul(100).round(2).tail(5).T.to_string())
print("\n--- levels, last 6")
print(cp[["^TNX", "^FVX", "^IRX", "^VIX", "^VIX3M", "^MOVE", "DX-Y.NYB", "TLT", "GLD", "SLV"]].tail(6).round(3).to_string())
for t in ["SPY", "QQQ", "IWM", "RSP", "TLT", "IEF", "LQD", "HYG", "^TNX", "^MOVE", "UUP", "GLD", "SLV", "EEM", "EWJ", "FXY", "^VIX"]:
    if t not in cp:
        continue
    s = cp[t].dropna()
    print(f"{t:8s} 5d {100 * (s.iloc[-1] / s.iloc[-6] - 1):6.2f} 21d {100 * (s.iloc[-1] / s.iloc[-22] - 1):6.2f} "
          f"63d {100 * (s.iloc[-1] / s.iloc[-64] - 1):6.2f} vs52wH {100 * (s.iloc[-1] / s.tail(252).max() - 1):6.2f} "
          f"vs52wL {100 * (s.iloc[-1] / s.tail(252).min() - 1):6.2f} vs200d {100 * (s.iloc[-1] / s.tail(200).mean() - 1):6.2f} "
          f"52wH date {s.tail(252).idxmax().date()}")

tnx, fvx = cp["^TNX"].dropna(), cp["^FVX"].dropna()
print("\n10y bp last 5:", (tnx.diff() * 100).tail(5).round(1).to_dict())
print("5y bp last 5:", (fvx.diff() * 100).tail(5).round(1).to_dict())
curve = (tnx - fvx).dropna() * 100
print("10y-5y bp last 5:", curve.tail(5).round(1).to_dict())
print("TLT prior closes <= today:", [str(d.date()) for d in cp["TLT"].dropna()[cp["TLT"].dropna() <= cp["TLT"].dropna().iloc[-1] + 1e-9].index[-4:]])

# month / quarter to date
for t in ["SPY", "QQQ", "IWM", "TLT", "IEF", "HYG", "GLD", "UUP", "EEM"]:
    s = cp[t].dropna()
    print(f"{t} MTD {100 * (s.iloc[-1] / s.loc[:'2026-08-31'].iloc[-1] - 1):6.2f}  QTD {100 * (s.iloc[-1] / s.loc[:'2026-06-30'].iloc[-1] - 1):6.2f}")

sect = [t for t in cp if t.startswith("XL")]
above = {t: bool(cp[t].iloc[-1] > cp[t].tail(200).mean()) for t in sect}
print("\nsectors above 200d:", sum(above.values()), "of", len(sect), above)

# Thursday's items resolved
print("\nIEF Fri:", round(100 * cp["IEF"].pct_change().iloc[-1], 2), "USO Fri:", round(100 * cp["USO"].pct_change().iloc[-1], 2),
      "HYG Fri:", round(100 * cp["HYG"].pct_change().iloc[-1], 2), "UUP Fri:", round(100 * cp["UUP"].pct_change().iloc[-1], 2))
print("since Sep 18 close:", {t: round(100 * (cp[t].dropna().iloc[-1] / cp[t].loc["2026-09-18"] - 1), 2) for t in ["SPY", "IWM", "QQQ", "^GSPC"]})
