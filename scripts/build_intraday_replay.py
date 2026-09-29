"""Build site/research/intraday_replay.json, the intraday book's research replay series.

Research only, run by hand. The JSON is committed; the cloud site build never runs this script because the
source ledgers live in local, untracked artifacts/. It fails loudly when those artifacts are missing.

Open Breakout: frozen candidate ledgers (micro contracts, base costs), the amended prior-range skip
(filter_vs_skip row (v): ratio >= 1.25 AND the prior session's own summed net R >= +2), then LIVE sizing per
trade (open_breakout/strategy.py size_order + service.py daily/open caps) on the flat $750k basis.

Legend EMA: Databento futures signals executed in SPY/QQQ (primary variant), longs only, sized at the live rule
(40% of $750k SPY, 30% QQQ), returns net of the backtest's 2 bps round trip.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "site" / "research" / "intraday_replay.json"
BASIS = 750_000.0

OB = ROOT / "artifacts" / "research" / "qqq_open_breakout_20260923"
OB_CC = OB / "current_candidate"
OB_FVS = OB / "autocorr" / "lag1_rule" / "composite" / "overlap_walkforward" / "filter_vs_skip"
OB_AUDIT = OB / "breakeven"
OB_FEATS = OB / "range_filter_rty"  # ow_common.MIRROR_FEATS
OB_SPAN = ("2018-01-02", "2026-08-28")
OB_MARKETS = {
    # budget bps, micro multiplier
    "NQ": {"bps": 15.0, "mult": 2.0},
    "ES": {"bps": 10.0, "mult": 5.0},
}
OB_TICK = 0.25
OB_STOP_FRAC = 0.25
OB_FEE_SIDE = 0.85
OB_EXIT_RESERVE_TICKS = 4
# Live fat-finger ceiling (open_breakout.config.LIVE_HARD_MAX_CONTRACTS); no per-market cap since the owner
# decision of 2026-09-28 late. It binds only on 12 MNQ trades in 2018-2020 (NQ prior TR 25-29 at a third
# of today's index level); capped_at_max and budget_above_max report it.
OB_MAX_CONTRACTS = 60
OB_DAILY_BPS = 75.0
OB_OPEN_BPS = 25.0

LG = ROOT / "artifacts" / "legend-etf-execution-backtest-20260901"
LG_TRADES = LG / "trades_and_skips.csv"
LG_VARIANT = "etf_geometry_dynamic_ema_ex_exdiv"
LG_SPAN = ("2016-01-04", "2026-08-05")  # TRUSTED_START 2016-01-01 .. summary.json as_of
LG_WEIGHT = {"SPY": 0.40, "QQQ": 0.30}


def require(paths: list[Path]) -> None:
    missing = [p for p in paths if not p.exists()]
    if missing:
        lines = "\n  ".join(str(p) for p in missing)
        raise SystemExit(
            "build_intraday_replay: local research artifacts missing, cannot rebuild "
            f"{OUT.relative_to(ROOT)}.\n  {lines}\nThe committed JSON stays as is; run this only on the "
            "research machine that holds artifacts/."
        )


def sessions(span: tuple[str, str]) -> pd.DatetimeIndex:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import exchange_calendars as xcals

        cal = xcals.get_calendar("XNYS")
        s = cal.sessions_in_range(span[0], span[1])
    return pd.DatetimeIndex(s).tz_localize(None).normalize()


def ceil_tick(x: float, tick: float) -> float:
    return math.ceil(round(x / tick, 9)) * tick


def ob_skip_days() -> dict[str, set[pd.Timestamp]]:
    sys.path.insert(0, str(OB_FVS))
    import fvs_common as fvs  # noqa: E402

    df = fvs.ow.load("base", False)
    a, F = fvs.features(df)
    out: dict[str, set[pd.Timestamp]] = {}
    for m in OB_MARKETS:
        sL, sS = fvs.skips("v", F[m])
        assert np.array_equal(sL, sS), "(v) is a whole-day skip"
        mask = np.asarray(sL, bool) & np.asarray(a[f"{m}_elig"], bool)
        out[m] = set(pd.DatetimeIndex(df.index[mask]))
    return out


def open_breakout() -> tuple[dict, dict]:
    require([OB_CC / f"{m}_base_trades.csv" for m in OB_MARKETS]
            + [OB_AUDIT / f"{m}_session_audit.csv" for m in OB_MARKETS]
            + [OB_FEATS / f"{m}_features.csv" for m in OB_MARKETS]
            + [OB_FVS / "fvs_common.py"])
    skip = ob_skip_days()
    frames = []
    for m, spec in OB_MARKETS.items():
        t = pd.read_csv(OB_CC / f"{m}_base_trades.csv", parse_dates=["date", "entry_time", "exit_time"])
        audit = pd.read_csv(OB_AUDIT / f"{m}_session_audit.csv", parse_dates=["date"]).set_index("date")
        t["market"] = m
        t["prior_tr"] = t.date.map(audit.prior_tr)
        if t.prior_tr.isna().any():
            raise SystemExit(f"{m}: traded day without prior_tr in {OB_AUDIT}")
        t["skipped"] = t.date.isin(skip[m])
        t["usd_per_micro"] = t.pnl / t.qty  # ledger is micro contracts, base costs
        stop_pts = t.prior_tr.map(lambda x: ceil_tick(OB_STOP_FRAC * x, OB_TICK))
        t["per_contract"] = (stop_pts * spec["mult"] + 2 * OB_FEE_SIDE
                             + OB_EXIT_RESERVE_TICKS * OB_TICK * spec["mult"])
        t["planned"] = np.minimum(OB_MAX_CONTRACTS,
                                  np.floor(BASIS * spec["bps"] / 1e4 / t.per_contract)).astype(int)
        frames.append(t)
    all_t = pd.concat(frames, ignore_index=True)
    raw_sum_r = float(all_t.net_r.sum())
    raw_n = len(all_t)
    t = all_t[~all_t.skipped].sort_values(["entry_time", "market"]).reset_index(drop=True)

    daily_cap, open_cap = BASIS * OB_DAILY_BPS / 1e4, BASIS * OB_OPEN_BPS / 1e4
    qty = np.zeros(len(t), int)
    risk = np.zeros(len(t))
    for _, day in t.groupby("date", sort=True):
        reserved = 0.0
        for i in day.index:
            live = day.index[(day.index < i) & (day.exit_time > t.at[i, "entry_time"]).values]
            opened = float(risk[live].sum())
            room = min(daily_cap - reserved, open_cap - opened)
            pc = t.at[i, "per_contract"]
            fit = math.floor(room / pc + 1e-9) if room > 0 else 0
            q = min(int(t.at[i, "planned"]), fit)
            if q < 1:
                continue
            qty[i], risk[i] = q, q * pc
            reserved += risk[i]
    t["contracts"] = qty
    t["usd"] = t.contracts * t.usd_per_micro

    idx = sessions(OB_SPAN)
    bad = sorted(set(t.date) - set(idx))
    if bad:
        raise SystemExit(f"open_breakout: trade dates not XNYS sessions: {bad[:5]}")
    by_m = {m: t[t.market == m].groupby("date").usd.sum().reindex(idx, fill_value=0.0).round().astype(int)
            for m in OB_MARKETS}
    daily = sum(by_m.values())
    traded = int((t.contracts > 0).sum())
    meta = {
        "raw_trades": raw_n, "raw_sum_r": raw_sum_r,
        "kept_trades": len(t), "kept_sum_r": float(t.net_r.sum()),
        "one_micro_usd": float(t.usd_per_micro.sum()),
        "sized_trades": traded, "zero_size_trades": int(len(t) - traded),
        "capped_at_max": int((t.contracts == OB_MAX_CONTRACTS).sum()),
        "budget_above_max": int((np.floor(BASIS * t.market.map(lambda m: OB_MARKETS[m]["bps"]) / 1e4
                                          / t.per_contract) > OB_MAX_CONTRACTS).sum()),
        "cap_cut": int((t.contracts < t.planned).sum()),
        "max_contracts": int(t.contracts.max()),
        "avg_contracts": {m: float(t.loc[(t.market == m) & (t.contracts > 0), "contracts"].mean()) for m in OB_MARKETS},
        "size_dist": {m: {q: float(t.loc[(t.market == m) & (t.contracts > 0), "contracts"].quantile(p))
                          for q, p in (("p50", .5), ("p90", .9), ("max", 1.0))} for m in OB_MARKETS},
        "min_prior_tr": {m: float(t.loc[t.market == m, "prior_tr"].min()) for m in OB_MARKETS},
        "max_open_risk_ok": bool(risk.max() <= open_cap + 1e-6),
    }
    entry = {
        "id": "open_breakout", "name": "Open Breakout", "book": "intraday",
        "instruments": "MNQ/MES on NQ/ES signals", "span": list(OB_SPAN),
        "sizing": ("15 bps NQ / 10 bps ES of $750k, whole micro contracts, 25 bps open / 75 bps daily risk caps, "
                   "no per-market contract cap (fat-finger ceiling 60); amended prior-range "
                   "skip (ratio >= 1.25 and prior session >= +2R own R); shorts gated on legacy score >= 20"),
        "costs": "base: ledger fees and slippage",
        "_series": (daily, by_m, len(t)),
        "notes": "research replay of the frozen candidate; not a fill record",
    }
    return entry, meta


def legend_ema() -> tuple[dict, dict]:
    require([LG_TRADES])
    x = pd.read_csv(LG_TRADES, parse_dates=["entry_date"])
    x = x[(x.variant == LG_VARIANT) & x.traded.astype(str).eq("True") & (x.side == "long")
          & x.etf.isin(list(LG_WEIGHT))].copy()
    if x.net_return_bps_2bp.isna().any():
        raise SystemExit("legend_ema: traded row without a return")
    x["usd"] = x.etf.map(LG_WEIGHT) * BASIS * x.net_return_bps_2bp / 1e4
    idx = sessions(LG_SPAN)
    bad = sorted(set(x.entry_date) - set(idx))
    if bad:
        raise SystemExit(f"legend_ema: trade dates not XNYS sessions: {bad[:5]}")
    by_m = {e: x[x.etf == e].groupby("entry_date").usd.sum().reindex(idx, fill_value=0.0).round().astype(int)
            for e in LG_WEIGHT}
    daily = sum(by_m.values())
    meta = {"trades": len(x), "by_etf": x.etf.value_counts().to_dict(),
            "avg_bps": float(x.net_return_bps_2bp.mean()), "gross_avg_bps": float(x.gross_return_bps.mean())}
    entry = {
        "id": "legend_ema", "name": "Legend EMA", "book": "intraday",
        "instruments": "SPY/QQQ (ETF leg; futures leg MES/MNQ from 2026-09-29)", "span": list(LG_SPAN),
        "sizing": "40 percent of $750k in SPY / 30 percent in QQQ per the live rule, longs only",
        "costs": "net of 2 bps round trip (the backtest subtracts it from gross ETF returns)",
        "_series": (daily, by_m, len(x)),
        "notes": ("research replay: prior-session Databento futures setups executed in SPY/QQQ 1m IBKR bars, "
                  "enter 09:31 toward the ETF's RTH EMA20, exit at the target or the 10:30 open, ex-dividend "
                  "entry dates excluded; variant etf_geometry_dynamic_ema_ex_exdiv; not a fill record"),
    }
    return entry, meta


def pairs(s: pd.Series) -> list[list]:
    return [[d.strftime("%Y-%m-%d"), int(v)] for d, v in s.items()]


def stats(daily: pd.Series, n_trades: int, span: tuple[str, str]) -> dict:
    years = (pd.Timestamp(span[1]) - pd.Timestamp(span[0])).days / 365.25
    v = daily.astype(float)
    sd = float(v.std(ddof=1))
    eq = v.cumsum()
    dd = float((eq - np.maximum(eq.cummax(), 0.0)).min())
    return {"trades_per_year": round(n_trades / years, 1), "sum_usd": int(daily.sum()),
            "sharpe_daily": round(float(v.mean()) / sd * math.sqrt(252), 2) if sd > 0 else 0.0,
            "max_dd_usd": int(round(dd)), "years": round(years, 2)}


def finish(entry: dict) -> dict:
    daily, by_m, n = entry.pop("_series")
    return {**{k: entry[k] for k in ["id", "name", "book", "instruments", "span", "sizing", "costs"]},
            "daily": pairs(daily), "by_market": {m: pairs(s) for m, s in by_m.items()},
            "stats": stats(daily, n, tuple(entry["span"])), "notes": entry["notes"]}


def validate(doc: dict) -> None:
    text = json.dumps(doc)
    assert "NaN" not in text and "Infinity" not in text
    for s in doc["strategies"]:
        idx = set(d.strftime("%Y-%m-%d") for d in sessions(tuple(s["span"])))
        days = [d for d, _ in s["daily"]]
        assert days == sorted(idx), f"{s['id']}: daily is not every XNYS session in span"
        for m, ser in s["by_market"].items():
            assert [d for d, _ in ser] == days, f"{s['id']}/{m}: calendar mismatch"
        tot = [sum(v[i][1] for v in s["by_market"].values()) for i in range(len(days))]
        assert tot == [v for _, v in s["daily"]], f"{s['id']}: by_market does not sum to daily"
        assert s["stats"]["sum_usd"] == sum(v for _, v in s["daily"]), f"{s['id']}: sum_usd mismatch"


def summary_row(s: dict) -> str:
    d = pd.Series([v for _, v in s["daily"]], index=pd.to_datetime([k for k, _ in s["daily"]]))
    last12 = int(d[d.index > d.index[-1] - pd.DateOffset(years=1)].sum())
    st = s["stats"]
    return (f"{s['id']:<14}{st['sum_usd']:>12,}{st['sharpe_daily']:>8.2f}{st['max_dd_usd']:>12,}"
            f"{st['trades_per_year']:>9.1f}{last12:>12,}")


def main() -> None:
    ob, ob_meta = open_breakout()
    lg, lg_meta = legend_ema()
    doc = {"schema": 1, "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "basis_usd": int(BASIS), "strategies": [finish(ob), finish(lg)]}
    validate(doc)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")
    json.loads(OUT.read_text(encoding="utf-8"))
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024:.0f} KB)")
    print("open_breakout:", json.dumps(ob_meta, default=float))
    print("legend_ema:", json.dumps(lg_meta, default=float))
    print(f"{'strategy':<14}{'sum_usd':>12}{'sharpe':>8}{'max_dd':>12}{'tr/yr':>9}{'last12m':>12}")
    for s in doc["strategies"]:
        print(summary_row(s))


if __name__ == "__main__":
    main()
