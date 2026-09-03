"""GAP 3 / WP2 prep: reconstruct the ledger's open book on the four peak
requirement days plus the current date, and write Risk Navigator what-if
files at 1.0x / 1.25x / 1.5x current size.

Book on date d = every ledger trade with Entry Date <= d <= Exit Date.
Shares on the FLAT $750k basis (Shares_flat), notional = Shares_flat x Entry
Price (the convention of unconstrained_growth_02*.py and
growthmax_1_margin_tiered.py, so the requirement tables in those studies and
the ones in requirement_recompute.py are on the same book).

Caret spot tickers are converted to their tradeable ETF at the SAME notional
(^GSPC -> SPY, ^NDX -> QQQ, the SPOT_TO_TRADEABLE aliasing), using the raw
(unadjusted) ETF close on the date, pulled once from yfinance and pinned in
CARET_ETF_CLOSE below so the files are reproducible offline.

Outputs (whatif_books/):
  <date>_detail.csv         one row per open trade (strategy, tier, side, ...)
  <date>_positions_x1.00.csv  Symbol, Position (signed shares) -- paste form
  <date>_positions_x1.25.csv / _x1.50.csv
  <date>_rn_import_x1.00.csv  Risk Navigator Portfolio > Import format
                              (Action, Quantity, Symbol, SecType, Exchange, Currency)
  books.json                  per-date book summaries (feeds requirement_recompute.py)
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
ROOT = Path(r"C:/Users/McKinley Slade/dev/New_Seasonals")
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "whatif_books"
OUT_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))
import strategy_config as sc  # noqa: E402

NAV = 750_000.0
DATES = ["2013-11-04", "2016-06-14", "2019-06-26", "2023-02-03", "2026-09-01"]
MULTS = [1.0, 1.25, 1.5]
ALIAS = {"^GSPC": "SPY", "^NDX": "QQQ"}
# raw closes (auto_adjust=False) from yfinance, pulled 2026-09-02:
CARET_ETF_CLOSE = {
    ("2013-11-04", "SPY"): 176.830002,
    ("2013-11-04", "QQQ"): 82.930000,
    ("2023-02-03", "SPY"): 412.350006,
    ("2023-02-03", "QQQ"): 306.179993,
}

LEV3X = set(sc.LEV3X_ALL)
BEAR_EQ = set(sc.LEV3X_BEAR_EQ)
BULL_EQ = {"SPXL", "TQQQ", "UDOW", "TNA", "MIDU", "SOXL", "TECL", "FAS", "LABU", "WEBL", "CURE", "RETL", "NAIL", "DPST", "DFEN", "EDC", "YINN", "BRZU", "MEXX", "DRN"}
BROAD = {"SPY", "QQQ", "DIA", "^GSPC", "^NDX", "VOO", "IVV", "RSP", "EFA", "EEM", "VEA", "VWO"}
SMALLCAP_IDX = {"IWM", "MDY", "IJR", "IJH"}
SECTOR = {"XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC", "GLD", "SLV", "TLT", "IEF", "LQD", "HYG", "USO", "UNG", "GDX", "GDXJ", "XOP", "XBI", "SMH", "KRE", "XME", "XRT", "XHB", "ITB", "IYR", "VNQ", "FXI", "EWZ", "EWJ", "EWY", "EWT", "EWG", "EWU", "EWC", "EWW", "EWA", "EWH", "INDA", "KWEB", "ARKK", "IBB", "OIH", "TAN", "URA", "LIT", "DBC", "DBA", "UUP", "FXE", "FXY", "SLX", "COPX", "PPLT", "PALL", "CORN", "WEAT", "SOYB", "WOOD", "REMX", "SIL", "JETS", "PHO", "MOO", "BOTZ", "SOXX", "HACK", "IGV", "VGT", "XLG", "SPHB", "SPLV", "VXX", "UVXY", "SVXY", "TLH", "TBT", "BND", "AGG", "CEF", "IHI", "ITA"}


def klass(t: str) -> str:
    if t in LEV3X:
        return "lev3x"
    if t in BROAD:
        return "broad_idx"
    if t in SMALLCAP_IDX:
        return "smallcap_idx"
    if t in SECTOR:
        return "sector_etf"
    return "single"


def main() -> None:
    led = pd.read_parquet(ROOT / "data/backtest_trades_full.parquet")
    led = led[led["PnL_flat_750k"].notna()].copy()
    led["notional"] = led["Entry Price"] * led["Shares_flat"]
    books: dict = {"basis": "flat $750k; Shares_flat x Entry Price; Entry<=d<=Exit", "ledger_rows": int(len(led)),
                   "ledger_last_exit": str(led["Exit Date"].max().date()), "dates": {}}
    for ds in DATES:
        d = pd.Timestamp(ds)
        o = led[(led["Entry Date"] <= d) & (led["Exit Date"] >= d)].copy()
        o["sign"] = np.where(o["Direction"].eq("Short"), -1, 1)
        o["symbol"] = o["Ticker"].map(lambda t: ALIAS.get(t, t))
        # caret -> ETF at the same notional
        conv = []
        for i, r in o.iterrows():
            if r["Ticker"] in ALIAS:
                px = CARET_ETF_CLOSE.get((ds, r["symbol"]))
                if px is None:
                    raise SystemExit(f"no pinned ETF close for {ds} {r['symbol']}")
                sh = int(round(r["notional"] / px))
                conv.append((i, sh, px))
        o["etf_shares"] = o["Shares_flat"].astype(int)
        o["etf_price"] = o["Entry Price"]
        for i, sh, px in conv:
            o.loc[i, "etf_shares"] = sh
            o.loc[i, "etf_price"] = px
        o["klass"] = o["symbol"].map(klass)
        o["signed_shares"] = o["sign"] * o["etf_shares"]
        det = o[["Strategy", "Tier", "Ticker", "symbol", "Direction", "sign", "klass", "Entry Date", "Exit Date", "Exit Type",
                 "Entry Price", "etf_price", "Shares_flat", "etf_shares", "signed_shares", "notional", "Risk_flat_750k", "Tranche"]].copy()
        det["Entry Date"] = det["Entry Date"].dt.date
        det["Exit Date"] = det["Exit Date"].dt.date
        det = det.sort_values(["symbol", "Direction", "Entry Date"])
        det.to_csv(OUT_DIR / f"{ds}_detail.csv", index=False, float_format="%.4f")
        # per-symbol net position (a symbol long and short at once nets; both legs listed in the detail file)
        pos = det.groupby("symbol").agg(position=("signed_shares", "sum"), gross_shares=("etf_shares", "sum"),
                                        notional=("notional", "sum"), price=("etf_price", "mean"), klass=("klass", "first"),
                                        strategies=("Strategy", lambda s: "; ".join(sorted(set(s)))),
                                        tiers=("Tier", lambda s: "; ".join(sorted(set(s))))).reset_index()
        pos["long_notional"] = det[det["sign"] > 0].groupby("symbol")["notional"].sum().reindex(pos["symbol"]).fillna(0).values
        pos["short_notional"] = det[det["sign"] < 0].groupby("symbol")["notional"].sum().reindex(pos["symbol"]).fillna(0).values
        pos["pct_nav"] = pos["notional"] / NAV
        pos = pos.sort_values("notional", ascending=False)
        for m in MULTS:
            tag = f"x{m:.2f}"
            p = pos[["symbol", "position"]].copy()
            p["position"] = (p["position"] * m).round().astype(int)
            p = p[p["position"] != 0]
            p.rename(columns={"symbol": "Symbol", "position": "Position"}).to_csv(OUT_DIR / f"{ds}_positions_{tag}.csv", index=False)
            rn = pd.DataFrame({"Action": np.where(p["position"] > 0, "Buy", "Sell"), "Quantity": p["position"].abs(),
                               "Symbol": p["symbol"], "SecType": "STK", "Exchange": "SMART", "Currency": "USD"})
            rn.to_csv(OUT_DIR / f"{ds}_rn_import_{tag}.csv", index=False)
        pos.to_csv(OUT_DIR / f"{ds}_by_symbol.csv", index=False, float_format="%.4f")
        gross = float(det["notional"].sum()); lng = float(det.loc[det["sign"] > 0, "notional"].sum()); sht = gross - lng
        by_cls = det.groupby("klass")["notional"].sum()
        top = pos.iloc[0]
        books["dates"][ds] = dict(
            n_trades=int(len(det)), n_symbols=int(len(pos)), gross=gross, long=lng, short=sht,
            gross_pct_nav=gross / NAV, long_pct_nav=lng / NAV, short_pct_nav=sht / NAV,
            class_notional={k: float(v) for k, v in by_cls.items()},
            top_symbol=str(top["symbol"]), top_symbol_notional=float(top["notional"]), top_symbol_pct_nav=float(top["pct_nav"]),
            top5={r["symbol"]: round(float(r["pct_nav"]), 3) for _, r in pos.head(5).iterrows()},
            strategies={k: float(v) for k, v in det.groupby("Strategy")["notional"].sum().sort_values(ascending=False).items()},
            carets_converted=[(str(r["Ticker"]), str(r["symbol"]), int(r["etf_shares"])) for _, r in det[det["Ticker"].isin(ALIAS)].iterrows()],
            positions=[dict(symbol=str(r["symbol"]), position=int(r["position"]), notional=float(r["notional"]), long_notional=float(r["long_notional"]),
                            short_notional=float(r["short_notional"]), price=float(r["price"]), klass=str(r["klass"])) for _, r in pos.iterrows()],
        )
        print(f"{ds}: {len(det)} trades / {len(pos)} symbols  gross {gross/NAV:.0%} NAV (L {lng/NAV:.0%} / S {sht/NAV:.0%})  "
              f"top {top['symbol']} {top['pct_nav']:.0%}  classes {{{', '.join(f'{k} {v/NAV:.0%}' for k, v in by_cls.items())}}}")
        if books["dates"][ds]["carets_converted"]:
            print("   caret conversions:", books["dates"][ds]["carets_converted"])
    (HERE / "books.json").write_text(json.dumps(books, indent=1, default=float), encoding="utf-8")
    print("wrote", HERE / "books.json", "and", OUT_DIR)


if __name__ == "__main__":
    main()
