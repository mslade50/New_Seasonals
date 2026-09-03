"""GAP 3: recompute the margin requirement / NLV for the reconstructed books
(books.json from reconstruct_books.py) under the VERIFIED public IBKR
Portfolio Margin parameters, with the two unknowns the what-if must settle
run as sensitivities:

  (i)   plain PM (TIMS worst-case class stress, no cross-class offset for
        stocks/ETFs): stock / sector ETF 15%; broad index ETF +6/-8 (long 8%,
        short 6%); small-cap index +-10%; leveraged ETF = leverage x the
        underlying's stress (3x broad long 24% / short 18%, 3x small-cap 30%,
        3x sector or commodity 45%); Reg-T low-price short minimums carried
        into PM (short stock < $16.67: max(15%, $5.00/sh); < $5: 100%).
  (ii)  rules-based leveraged ETF rates instead of TIMS (3x long 75%,
        3x short 90%) -- the "higher of" reading.
  (iii) the 30% concentration minimum applied to the LARGEST position
        (+ a variant applying it to every position > 25% NAV).
Initial = 1.10 x maintenance (verified).  Multiples m = 1.0 / 1.25 / 1.5 of
current sizing (Shares_flat is at GRM 1.5), against the $750k base, the plan's
$632k note and the live primary NLV from the broker snapshot.

Also prices the ACTUAL live primary book (live_book_<date>.json) at PM rates
so the current account can be placed against the same 70 / 85 / 100% lines.

Writes requirement_table.csv, requirement_results.json and prints the tables.
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
sys.path.insert(0, str(ROOT))
import strategy_config as sc  # noqa: E402

BASE = 750_000.0
PLAN_LIVE = 632_000.0
MULTS = [1.0, 1.25, 1.5]
LINES = [0.70, 0.85, 1.00]
INITIAL_FACTOR = 1.10

LEV3X = set(sc.LEV3X_ALL)
BEAR_EQ = set(sc.LEV3X_BEAR_EQ)
# leveraged ETF -> (underlying stress class, leverage, inverse?)
LEV_BROAD_BULL = {"SPXL", "TQQQ", "UPRO", "UDOW"}
LEV_BROAD_BEAR = {"SPXS", "SQQQ", "SPXU", "SDOW"}
LEV_SMALL_BULL = {"TNA", "MIDU"}
LEV_SMALL_BEAR = {"TZA"}
LEV_2X_BULL = {"SSO", "QLD"}
LEV_2X_BEAR = {"SDS", "QID"}


def tims_rate(symbol: str, klass: str, side: str) -> float:
    """Worst-case TIMS class stress as a fraction of notional for a single-position class."""
    if klass == "broad_idx":
        return 0.08 if side == "long" else 0.06
    if klass == "smallcap_idx":
        return 0.10
    if klass in ("single", "sector_etf"):
        return 0.15
    # leveraged
    if symbol in LEV_BROAD_BULL:
        return 3 * (0.08 if side == "long" else 0.06)
    if symbol in LEV_BROAD_BEAR:                       # inverse: long loses on +6, short loses on -8
        return 3 * (0.06 if side == "long" else 0.08)
    if symbol in LEV_2X_BULL:
        return 2 * (0.08 if side == "long" else 0.06)
    if symbol in LEV_2X_BEAR:
        return 2 * (0.06 if side == "long" else 0.08)
    if symbol in LEV_SMALL_BULL or symbol in LEV_SMALL_BEAR:
        return 0.30
    return 0.45                                          # 3x sector / commodity / treasury


def rules_rate(symbol: str, klass: str, side: str) -> float | None:
    """Reg-T style leveraged-ETF maintenance (leverage x 25% long / 30% short, capped 100%)."""
    if klass != "lev3x":
        return None
    lev = 2 if symbol in LEV_2X_BULL | LEV_2X_BEAR else 3
    return min(1.0, lev * (0.25 if side == "long" else 0.30))


def short_min(price: float, shares: int, req: float, klass: str) -> float:
    if klass != "single":
        return req
    if price < 5.0:
        return max(req, price * shares)
    if price < 16.67:
        return max(req, 5.0 * shares)
    return req


def price_book(positions: list[dict], scenario: str, m: float = 1.0) -> tuple[float, list[dict]]:
    """Maintenance requirement in $ for a list of {symbol, position, price, klass}; scenario in
    {'pm', 'pm_rules3x', 'pm_conc_top', 'pm_conc_25', 'pm_rules3x_conc_top', 'flat45'}."""
    rows = []
    tot_not = sum(abs(p["position"]) * p["price"] for p in positions) * m
    biggest = max(positions, key=lambda p: abs(p["position"]) * p["price"])["symbol"] if positions else None
    for p in positions:
        sh = int(round(p["position"] * m))
        if sh == 0:
            continue
        side = "long" if sh > 0 else "short"
        notional = abs(sh) * p["price"]
        k = p["klass"]
        rate = tims_rate(p["symbol"], k, side)
        if scenario == "flat45" and k == "lev3x":
            rate = 0.45
        if scenario.startswith("pm_rules3x"):
            rr = rules_rate(p["symbol"], k, side)
            if rr is not None:
                rate = max(rate, rr)
        if scenario.endswith("conc_top") and p["symbol"] == biggest:
            rate = max(rate, 0.30)
        if scenario.endswith("conc_25") and notional > 0.25 * BASE:
            rate = max(rate, 0.30)
        req = rate * notional
        req = short_min(p["price"], abs(sh), req, k) if side == "short" else req
        rows.append(dict(symbol=p["symbol"], side=side, shares=sh, notional=notional, klass=k, rate=req / notional if notional else 0.0, req=req))
    return float(sum(r["req"] for r in rows)), rows


def main() -> None:
    books = json.loads((HERE / "books.json").read_text(encoding="utf-8"))
    live_files = sorted(HERE.glob("live_book_*.json"))
    live_nlv = None; live_positions = []; live_stamp = None
    if live_files:
        lb = json.loads(live_files[-1].read_text(encoding="utf-8"))
        accts = lb["book"]["accounts"]
        prim = next(a for a in accts if a.get("key") == "primary")
        live_nlv = float(prim["nlv"]); live_stamp = live_files[-1].stem.replace("live_book_", "")
        live_positions = prim["positions"]
    nlv_bases = {"base_750k": BASE, "plan_live_632k": PLAN_LIVE}
    if live_nlv:
        nlv_bases[f"live_nlv_{live_stamp}"] = live_nlv
    scenarios = ["pm", "flat45", "pm_rules3x", "pm_conc_top", "pm_conc_25", "pm_rules3x_conc_top"]
    out_rows = []
    results = {"nlv_bases": nlv_bases, "initial_factor": INITIAL_FACTOR, "books": {}}
    print(f"NLV bases: {nlv_bases}")
    for ds, b in books["dates"].items():
        pos = b["positions"]
        results["books"][ds] = {"gross_pct_base": b["gross_pct_nav"], "scenarios": {}}
        print(f"\n=== {ds}: {b['n_trades']} trades, {b['n_symbols']} symbols, gross {b['gross_pct_nav']:.0%} of $750k (L {b['long_pct_nav']:.0%} / S {b['short_pct_nav']:.0%}); top {b['top_symbol']} {b['top_symbol_pct_nav']:.0%} ===")
        hdr = f"{'scenario':22s} {'maint@1x':>10s} " + " ".join(f"{'m'+str(m):>7s}/{k[:9]:9s}" for k in nlv_bases for m in MULTS)
        print(hdr)
        for sc_name in scenarios:
            req1, rows = price_book(pos, sc_name, 1.0)
            cells = {}
            line = f"{sc_name:22s} {req1/1e3:9.0f}k "
            for k, nlv in nlv_bases.items():
                for m in MULTS:
                    req_m, _ = price_book(pos, sc_name, m)
                    frac = req_m / nlv
                    cells[f"{k}|{m}"] = frac
                    flag = "" if frac < LINES[0] else ("*" if frac < LINES[1] else ("**" if frac < LINES[2] else "!!!"))
                    line += f" {frac:6.0%}{flag:3s}     "
                    out_rows.append(dict(date=ds, scenario=sc_name, nlv_basis=k, nlv=nlv, m=m, maint_req=req_m, initial_req=req_m * INITIAL_FACTOR,
                                         maint_pct_nlv=frac, initial_pct_nlv=frac * INITIAL_FACTOR,
                                         past_70=frac >= 0.70, past_85=frac >= 0.85, past_100=frac >= 1.0, initial_past_100=frac * INITIAL_FACTOR >= 1.0))
            print(line)
            results["books"][ds]["scenarios"][sc_name] = dict(maint_at_1x=req1, cells=cells, blended_rate=req1 / (b["gross"]) if b["gross"] else None,
                                                             top_contributors=sorted(rows, key=lambda r: -r["req"])[:6])
        # m at which maintenance == NLV, per scenario and basis (linear in m)
        walls = {}
        for sc_name in scenarios:
            req1, _ = price_book(pos, sc_name, 1.0)
            walls[sc_name] = {k: dict(m_maint_eq_nlv=nlv / req1, m_initial_eq_nlv=nlv / (req1 * INITIAL_FACTOR), m_maint_eq_70pct=0.70 * nlv / req1) for k, nlv in nlv_bases.items()}
        results["books"][ds]["walls"] = walls
        print("   m at which maintenance = 100% NLV:  " + "  ".join(f"{s}: " + "/".join(f"{walls[s][k]['m_maint_eq_nlv']:.2f}" for k in nlv_bases) for s in ["pm", "pm_rules3x", "pm_conc_top"]) + f"   (order: {', '.join(nlv_bases)})")
    # ------------------------------------------------------------------ the actual live book
    if live_positions:
        print(f"\n=== ACTUAL live primary book ({live_stamp}, NLV ${live_nlv:,.0f}) priced at PM rates (stocks/ETFs; futures + options listed separately) ===")
        from reconstruct_books import klass as _klass  # noqa: E402
        stk = [dict(symbol=p["symbol"], position=int(p["position"]), price=float(p["market_price"]), klass=_klass(p["symbol"])) for p in live_positions if p.get("sec_type") == "STK"]
        others = [p for p in live_positions if p.get("sec_type") != "STK"]
        live_res = {}
        for sc_name in ["pm", "pm_rules3x", "pm_conc_top"]:
            req, rows = price_book(stk, sc_name, 1.0)
            live_res[sc_name] = dict(maint=req, pct_nlv=req / live_nlv, rows=rows)
            print(f"  {sc_name:14s} stock-leg maintenance ${req:,.0f} = {req/live_nlv:.1%} of live NLV (initial x1.10 = {req*INITIAL_FACTOR/live_nlv:.1%})")
        gross_stk = sum(abs(p["position"]) * p["price"] for p in stk)
        print(f"  stock gross ${gross_stk:,.0f} = {gross_stk/live_nlv:.0%} of NLV; top: " + ", ".join(f"{r['symbol']} {r['notional']/live_nlv:.0%}" for r in sorted(live_res['pm']['rows'], key=lambda r: -r['notional'])[:5]))
        print("  non-stock legs (NOT priced here; SPAN/TIMS from the broker):")
        for p in others:
            print(f"    {p['symbol']:5s} {p['sec_type']:3s} pos {p['position']:>5} mkt_value {p.get('market_value', 0):>12,.0f}")
        results["live_book"] = dict(stamp=live_stamp, nlv=live_nlv, stock_gross=gross_stk, scenarios={k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in live_res.items()},
                                    rows=live_res["pm"]["rows"], non_stock=[{k: p.get(k) for k in ("symbol", "sec_type", "position", "market_value")} for p in others])
    pd.DataFrame(out_rows).to_csv(HERE / "requirement_table.csv", index=False, float_format="%.4f")
    (HERE / "requirement_results.json").write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
    print("\nwrote requirement_table.csv, requirement_results.json")


if __name__ == "__main__":
    main()
