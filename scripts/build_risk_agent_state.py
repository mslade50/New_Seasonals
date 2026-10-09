"""Stage A of the Risk Agent: deterministic state assembly.

    python scripts/build_risk_agent_state.py [--asof YYYY-MM-DD]
        [--out data/risk_agent_state.json] [--chains-out data/risk_agent_chains.json]
        [--no-sync]

Reads ONLY the allowlisted market objects in data/risk_agent/cache/ (see
risk_agent_data.py / risk_agent_universe.r2_key_allowed), the agent's own
journal and its own scoreboard. It never touches positions, fills, orders,
sizing state, exposure state or other agents' output.

Every block except the price cache is best effort: a missing input adds a
line to `warnings` and leaves the block partial. A missing master_prices is
fatal (nothing to reason about).

Units: tape r1/r5/r21/r63, dist_*, atr_pct, rv21 are PERCENT. stress values
are FRACTIONS (the validator's stress_up_move wants fractions).
"""
from __future__ import annotations

import argparse
import calendar
import datetime as dt
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import risk_agent_data as rad  # noqa: E402
from risk_agent_grammar import SCHEMA_VERSION as _GRAMMAR_SCHEMA, chain_quote_key  # noqa: E402,F401
from risk_agent_universe import (CONTEXT_SERIES, ETFS, FUTURES, OPTIONABLE,  # noqa: E402
                                 SLEEVE_CAPITAL)
from pitch_grammar import wilder_atr  # noqa: E402
from trading_calendar import TRADING_DAY, NYSE_HOLIDAYS  # noqa: E402
import options_surface as osf  # noqa: E402

STATE_SCHEMA = "risk_agent_state.v2"
STALE_BARS_TD = 5
STRESS_HORIZONS = (5, 10, 21, 42, 63, 126)
STRESS_Q = 0.999
CHAIN_DTE_MIN, CHAIN_DTE_MAX = 0, 200  # 0-1 DTE weeklies are in scope (owner, 2026-10-09)
MEGACAPS = ("AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "JPM")
CB_PAT = re.compile(r"\b(fed|fomc|ecb|boj|boe|snb|rba|boc)\b", re.I)
VOL_SERIES = ("^VIX", "^VIX3M", "^VVIX", "^SKEW", "^MOVE")
RATE_SERIES = ("^TNX", "^FVX", "^IRX")
FX_SERIES = tuple(s for s in CONTEXT_SERIES if s.endswith("=X"))
DEFAULT_JOURNAL = ROOT / "data" / "risk_agent_journal.jsonl"
DEFAULT_SCOREBOARD = ROOT / "data" / "risk_agent_scoreboard.json"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _clean(o, nd: int = 4):
    """JSON-safe: numpy -> python, NaN/inf -> None, floats rounded."""
    if isinstance(o, dict):
        return {str(k): _clean(v, nd) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v, nd) for v in o]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return None if not math.isfinite(f) else round(f, nd)
    if isinstance(o, (pd.Timestamp, dt.datetime, dt.date)):
        return str(o)[:10]
    if o is pd.NaT:
        return None
    return o


def _r(x, nd: int = 2):
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return round(f, nd) if math.isfinite(f) else None


def _pct_rank(series: np.ndarray, current: float, min_obs: int = 20):
    s = np.asarray(series, dtype=float)
    s = s[np.isfinite(s)]
    if len(s) < min_obs or not math.isfinite(current):
        return None
    return round(100.0 * float((s < current).sum()) / len(s), 1)


def _read_parquet(cache_dir: Path, key: str, warnings: list, **kw):
    p = rad.local_path(key, cache_dir)
    if not p.exists():
        warnings.append(f"missing cache object {key}")
        return None
    try:
        return pd.read_parquet(p, **kw)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"unreadable cache object {key}: {type(exc).__name__}: {exc}")
        return None


def _td_add(d: pd.Timestamp, n: int) -> pd.Timestamp:
    return pd.Timestamp(d) + n * TRADING_DAY


def _is_session(d: pd.Timestamp) -> bool:
    return d.weekday() < 5 and d.normalize() not in set(NYSE_HOLIDAYS)


def _sessions_between(a: pd.Timestamp, b: pd.Timestamp) -> int:
    """Count NYSE sessions in (a, b]."""
    if b <= a:
        return 0
    return len(pd.date_range(a + pd.Timedelta(days=1), b, freq=TRADING_DAY))


def monthly_opex(year: int, month: int) -> pd.Timestamp:
    """Third Friday; the prior Thursday if the Friday is an exchange holiday."""
    cal = calendar.Calendar()
    fridays = [d for d in cal.itermonthdates(year, month)
               if d.month == month and d.weekday() == 4]
    d = pd.Timestamp(fridays[2])
    if not _is_session(d):
        d = d - pd.Timedelta(days=1)
    return d


# ---------------------------------------------------------------------------
# price panel
# ---------------------------------------------------------------------------
def load_prices(cache_dir: Path, tickers: list[str], asof: str | None) -> pd.DataFrame:
    p = rad.local_path("master_prices.parquet", cache_dir)
    if not p.exists():
        raise SystemExit(f"FATAL: {p} missing; run risk_agent_data.sync()")
    df = pd.read_parquet(p, filters=[("ticker", "in", list(tickers))])
    df["date"] = pd.to_datetime(df["date"])
    if asof:
        df = df[df["date"] <= pd.Timestamp(asof)]
    for c in ("Open", "High", "Low", "Close"):
        df[c] = df[c].astype("float64")
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def _symbol_metrics(g: pd.DataFrame) -> dict:
    close = g["Close"].to_numpy(float)
    n = len(close)
    last = close[-1]
    atr = wilder_atr(g["High"].to_numpy(float), g["Low"].to_numpy(float), close)
    atr_last = float(atr[-1]) if n > 14 and math.isfinite(atr[-1]) else None

    def ret(k):
        return 100.0 * (last / close[-1 - k] - 1.0) if n > k and close[-1 - k] > 0 else None

    out = {"close": _r(last, 4), "atr": _r(atr_last, 4),
           "atr_pct": _r(100.0 * atr_last / last, 2) if atr_last and last > 0 else None,
           "r1": _r(ret(1)), "r5": _r(ret(5)), "r21": _r(ret(21)), "r63": _r(ret(63))}
    if n >= 200:
        sma = close[-200:].mean()
        out["dist_200d_pct"] = _r(100.0 * (last / sma - 1.0)) if sma > 0 else None
    else:
        out["dist_200d_pct"] = None
    if n >= 20:
        hi = np.nanmax(close[-252:])
        out["dist_52wh_pct"] = _r(100.0 * (last / hi - 1.0)) if hi > 0 else None
    else:
        out["dist_52wh_pct"] = None
    if n > 22:
        lr = np.diff(np.log(np.where(close[-22:] > 0, close[-22:], np.nan)))
        out["rv21"] = _r(100.0 * np.nanstd(lr, ddof=1) * math.sqrt(252)) if np.isfinite(lr).sum() > 15 else None
    else:
        out["rv21"] = None
    out["last_bar"] = str(g["date"].iloc[-1])[:10]
    return out


def build_tape(px: pd.DataFrame, asof: pd.Timestamp, warnings: list,
               iv_hist: pd.DataFrame | None) -> tuple[dict, dict]:
    groups = {t: g for t, g in px.groupby("ticker", sort=False)}
    series_roots: dict[str, list[str]] = {}
    for root, f in FUTURES.items():
        series_roots.setdefault(f.series, []).append(root)

    plan: list[tuple[str, str]] = [(s, "etf") for s in ETFS]
    plan += [(s, "future") for s in series_roots]
    plan += [(s, "context") for s in CONTEXT_SERIES if s not in ETFS]

    tape: dict[str, dict] = {}
    cutoff = asof - STALE_BARS_TD * TRADING_DAY
    for sym, kind in plan:
        g = groups.get(sym)
        if g is None or g.empty:
            if kind != "context":
                warnings.append(f"{kind} {sym} absent from master_prices: dropped")
            else:
                warnings.append(f"context {sym} absent from master_prices")
            continue
        last_bar = g["date"].iloc[-1]
        if kind != "context" and last_bar < cutoff:
            warnings.append(f"{kind} {sym} last bar {str(last_bar)[:10]} older than "
                            f"{STALE_BARS_TD} sessions: dropped")
            continue
        m = _symbol_metrics(g)
        m["kind"] = kind
        if kind == "future":
            m["roots"] = series_roots[sym]
        tape[sym] = m

    # cross-sectional percentile ranks within the tradeable set
    tradeable = [s for s, m in tape.items() if m["kind"] in ("etf", "future")]
    for key, out in (("r5", "rank5"), ("r21", "rank21"), ("r63", "rank63")):
        vals = pd.Series({s: tape[s].get(key) for s in tradeable}, dtype="float64").dropna()
        if len(vals) > 1:
            ranks = vals.rank(pct=True) * 100.0
            for s, v in ranks.items():
                tape[s][out] = _r(v, 1)
    for s in tradeable:
        for k in ("rank5", "rank21", "rank63"):
            tape[s].setdefault(k, None)

    # IV fields for optionable names
    if iv_hist is not None and not iv_hist.empty:
        for sym, grp in iv_hist.groupby("ticker"):
            if sym in tape:
                grp = grp.sort_values("date")
                v = grp["iv30"].to_numpy(float)
                cur = v[-1]
                tape[sym]["iv30"] = _r(cur, 4)
                tape[sym]["iv_rank_1y"] = _pct_rank(v[-252:], cur)
    for s in tape:
        tape[s].setdefault("iv30", None)
        tape[s].setdefault("iv_rank_1y", None)

    quotes = {s: {"close": m["close"], "atr": m["atr"]} for s, m in tape.items()
              if m["kind"] in ("etf", "future")}
    return tape, quotes


# ---------------------------------------------------------------------------
# stress
# ---------------------------------------------------------------------------
def build_stress(px: pd.DataFrame, warnings: list) -> dict:
    """99.9th percentile of rolling h-day forward returns (fractions)."""
    out: dict[str, dict] = {}
    groups = {t: g for t, g in px.groupby("ticker", sort=False)}
    for und in OPTIONABLE:
        g = groups.get(und)
        if g is None or len(g) < 300:
            warnings.append(f"stress: no usable price history for optionable {und}")
            continue
        close = g["Close"].to_numpy(float)
        row = {}
        for h in STRESS_HORIZONS:
            if len(close) <= h + 50:
                continue
            fwd = close[h:] / close[:-h] - 1.0
            fwd = fwd[np.isfinite(fwd)]
            if len(fwd):
                row[str(h)] = _r(np.quantile(fwd, STRESS_Q), 4)
        if len(close) < 1500:
            warnings.append(f"stress: {und} has only {len(close)} bars; 99.9th pct is thin")
        if row:
            out[und] = row
    return out


# ---------------------------------------------------------------------------
# options
# ---------------------------------------------------------------------------
def _latest_snapshots(pos: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    pos = pos.copy()
    pos["_d"] = pd.to_datetime(pos["date"], errors="coerce")
    pos = pos[pos["_d"] <= asof]
    last = pos.groupby("ticker")["_d"].transform("max")
    return pos[pos["_d"] == last]


def _expiry_iso(v) -> str:
    s = str(v)
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}" if re.fullmatch(r"\d{8}", s) else s[:10]


def build_options(pos: pd.DataFrame | None, iv_hist: pd.DataFrame | None,
                  asof: pd.Timestamp, warnings: list) -> dict:
    if pos is None or pos.empty:
        return {}
    snap = _latest_snapshots(pos, asof)
    iv_by = {}
    if iv_hist is not None and not iv_hist.empty:
        for sym, g in iv_hist.groupby("ticker"):
            iv_by[sym] = g.sort_values("date")["iv30"].to_numpy(float)
    out = {}
    for und, g in snap.groupby("ticker"):
        spot = float(g["spot"].iloc[0])
        snap_date = str(g["_d"].iloc[0])[:10]
        atm = {}
        for exp, ge in g.groupby("expiry"):
            ge = ge[(ge["iv"] > 0) & np.isfinite(ge["iv"])]
            if ge.empty:
                continue
            k = ge.iloc[(ge["strike"] - spot).abs().argsort()[:2]]
            atm[_expiry_iso(exp)] = {"dte": int(ge["dte"].iloc[0]), "atm_iv": _r(k["iv"].mean(), 4)}
        # skew on the expiry closest to 30 DTE with usable deltas
        skew = {}
        for exp, ge in sorted(g.groupby("expiry"), key=lambda kv: abs(int(kv[1]["dte"].iloc[0]) - 30)):
            rows = ge.to_dict("records")
            p = osf.nearest_delta(rows, "P", 0.25)
            c = osf.nearest_delta(rows, "C", 0.25)
            if (p and c and abs(abs(p["delta"]) - 0.25) < 0.10 and abs(abs(c["delta"]) - 0.25) < 0.10):
                skew = {"skew_expiry": _expiry_iso(exp), "put_25d_iv": _r(p["iv"], 4),
                        "call_25d_iv": _r(c["iv"], 4), "skew": _r(p["iv"] - c["iv"], 4)}
                break
        v = iv_by.get(und)
        iv30 = float(v[-1]) if v is not None and len(v) else None
        out[und] = {"snapshot_date": snap_date, "spot": _r(spot, 4), "iv30": _r(iv30, 4),
                    "iv_rank_1y": _pct_rank(v[-252:], iv30) if v is not None and iv30 else None,
                    "atm_iv_by_expiry": atm, **skew, "n_quotes": int(len(g))}
        if (asof - pd.Timestamp(snap_date)).days > 5:
            warnings.append(f"options: {und} latest chain snapshot {snap_date} is stale")
    return out


def build_chains(asof: str | None = None, cache_dir: Path | str = rad.CACHE_DIR) -> dict:
    """{underlying: {spot, asof, quotes: {chain_quote_key: {...}}}} from the
    latest positioning snapshot per underlying, expiries 5-200 DTE."""
    cache_dir = Path(cache_dir)
    warnings: list = []
    pos = _read_parquet(cache_dir, "options/positioning_history.parquet", warnings)
    if pos is None:
        return {}
    cutoff = pd.Timestamp(asof) if asof else pd.Timestamp.max
    snap = _latest_snapshots(pos, cutoff)
    snap = snap[(snap["dte"] >= CHAIN_DTE_MIN) & (snap["dte"] <= CHAIN_DTE_MAX)]
    out = {}
    for und, g in snap.groupby("ticker"):
        quotes = {}
        for r in g.itertuples(index=False):
            key = chain_quote_key(_expiry_iso(r.expiry), float(r.strike), str(r.right))
            quotes[key] = _clean({"bid": r.bid, "ask": r.ask, "mid": r.mid, "iv": r.iv,
                                  "delta": r.delta, "con_id": int(r.con_id),
                                  "oi": r.oi, "volume": r.volume}, 5)
        out[und] = {"spot": _r(g["spot"].iloc[0], 4), "asof": str(g["_d"].iloc[0])[:10],
                    "quotes": quotes}
    return out


# ---------------------------------------------------------------------------
# dashboard
# ---------------------------------------------------------------------------
def _state_run(dates: list, periods: list, asof: str) -> dict:
    """Sessions in the current on/off state from on-periods [start, end]."""
    idx = {d: i for i, d in enumerate(dates)}
    if not dates:
        return {}
    last_i = idx.get(asof, len(dates) - 1)
    if periods and idx.get(periods[-1][1], -1) == last_i:
        start = idx.get(periods[-1][0])
        return {"state": "on", "sessions": (last_i - start + 1) if start is not None else None}
    if periods:
        end = idx.get(periods[-1][1])
        return {"state": "off", "sessions": (last_i - end) if end is not None else None}
    return {"state": "off", "sessions": None}


def build_dashboard(cache_dir: Path, asof: str, warnings: list) -> dict:
    p = rad.local_path("shared/site_risk.json", cache_dir)
    if not p.exists():
        warnings.append("missing cache object shared/site_risk.json")
        return {}
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        warnings.append(f"unreadable shared/site_risk.json: {exc}")
        return {}
    dates = d.get("dates") or []
    detail = d.get("signal_detail") or {}
    sigs = []
    for s in d.get("signals") or []:
        name = s.get("name")
        det = detail.get(name) or {}
        met = det.get("metric") or {}
        vals = met.get("values") or []
        cur = (det.get("current") or {}).get("value")

        def chg(k):
            if len(vals) > k and vals[-1] is not None and vals[-1 - k] is not None:
                return _r(vals[-1] - vals[-1 - k], 3)
            return None
        sigs.append({"name": name, "on": bool(s.get("on")), "elevated": bool(s.get("elevated")),
                     "badge": s.get("badge"), "current_value": cur,
                     "metric_key": met.get("key"), "metric_unit": met.get("unit"),
                     "chg_5": chg(5), "chg_21": chg(21),
                     **{f"run_{k}": v for k, v in
                        _state_run(dates, det.get("periods") or [], d.get("asof") or "").items()}})
    fr = {}
    for hz, blk in (d.get("forward_returns") or {}).items():
        rets = {}
        for h, r in (blk.get("returns") or {}).items():
            rets[h] = {"mean_pct": _r(100 * r["mean"]) if r.get("mean") is not None else None,
                       "median_pct": _r(100 * r["median"]) if r.get("median") is not None else None,
                       "p_up": _r(r.get("p_up"), 3), "q10_pct": _r(100 * r["q10"]) if r.get("q10") is not None else None,
                       "q90_pct": _r(100 * r["q90"]) if r.get("q90") is not None else None,
                       "uncond_mean_pct": _r(100 * r["uncond_mean"]) if r.get("uncond_mean") is not None else None,
                       "n": r.get("n")}
        fr[hz] = {"n_episodes": blk.get("n_episodes"), "status": blk.get("status"),
                  "band": [_r(blk.get("band_low"), 1), _r(blk.get("band_high"), 1)],
                  "returns": rets}
    ss = d.get("sizing_state") or {}
    out = {"asof": d.get("asof"), "built_at": d.get("built_at"),
           "dial": {"score": ss.get("score"), "raw_63d": ss.get("raw_63d"), "asof": ss.get("asof")},
           "fragility": d.get("fragility"), "n_active": d.get("n_active"),
           "signals": sigs, "forward_returns": fr, "vol_kpi": d.get("vol_kpi"),
           "price_ctx": d.get("price_ctx")}
    if d.get("asof") != asof:
        warnings.append(f"dashboard asof {d.get('asof')} != state asof {asof}")
    return out


# ---------------------------------------------------------------------------
# small blocks
# ---------------------------------------------------------------------------
def build_vol(px_groups: dict, warnings: list) -> dict:
    out = {}
    for s in VOL_SERIES:
        g = px_groups.get(s)
        if g is None or g.empty:
            warnings.append(f"vol: {s} missing")
            continue
        c = g["Close"].to_numpy(float)
        out[s] = {"last": _r(c[-1], 3),
                  "chg_1d": _r(c[-1] - c[-2], 3) if len(c) > 1 else None,
                  "chg_5d": _r(c[-1] - c[-6], 3) if len(c) > 5 else None,
                  "pctile_1y": _pct_rank(c[-252:], c[-1]), "last_bar": str(g["date"].iloc[-1])[:10]}
    if "^VIX" in out and "^VIX3M" in out and out["^VIX3M"]["last"]:
        out["vix_vix3m_ratio"] = _r(out["^VIX"]["last"] / out["^VIX3M"]["last"], 3)
    return out


def build_rates_fx(px_groups: dict, warnings: list) -> dict:
    out = {}
    for s in RATE_SERIES + FX_SERIES:
        g = px_groups.get(s)
        if g is None or g.empty:
            warnings.append(f"rates_fx: {s} missing")
            continue
        c = g["Close"].to_numpy(float)
        row = {"last": _r(c[-1], 4), "last_bar": str(g["date"].iloc[-1])[:10]}
        if s in RATE_SERIES:   # level in percent: change in bps
            row["chg_5d_bps"] = _r(100 * (c[-1] - c[-6]), 1) if len(c) > 5 else None
            row["chg_21d_bps"] = _r(100 * (c[-1] - c[-22]), 1) if len(c) > 21 else None
        else:
            row["chg_5d_pct"] = _r(100 * (c[-1] / c[-6] - 1), 2) if len(c) > 5 and c[-6] else None
            row["chg_21d_pct"] = _r(100 * (c[-1] / c[-22] - 1), 2) if len(c) > 21 and c[-22] else None
        out[s] = row
    return out


def build_breadth(cache_dir: Path, asof: pd.Timestamp, warnings: list) -> dict:
    df = _read_parquet(cache_dir, "market_breadth.parquet", warnings)
    if df is None or df.empty:
        return {}
    df = df[df.index <= asof].sort_index()
    cols = ["nyse_highs", "nyse_lows", "nasdaq_highs", "nasdaq_lows", "nyse_net", "nasdaq_net"]
    last = df.iloc[-1]
    out = {"date": str(df.index[-1])[:10], "latest": {c: int(last[c]) for c in cols}}
    for k in (5, 21):
        if len(df) > k:
            out[f"chg_{k}d"] = {c: int(last[c] - df[c].iloc[-1 - k]) for c in ("nyse_net", "nasdaq_net")}
            out[f"mean_{k}d"] = {c: _r(df[c].iloc[-k:].mean(), 1) for c in ("nyse_net", "nasdaq_net")}
    if (asof - df.index[-1]).days > 4:
        warnings.append(f"breadth last row {out['date']} is stale")
    return out


def build_putcall(cache_dir: Path, asof: pd.Timestamp, warnings: list) -> dict:
    df = _read_parquet(cache_dir, "cboe_putcall.parquet", warnings)
    if df is None or df.empty:
        return {}
    df = df[df.index <= asof].sort_index()
    out = {"date": str(df.index[-1])[:10]}
    for c in ("total", "index", "equity", "etp", "spx", "oex"):
        s = df[c].dropna()
        if s.empty:
            continue
        ma = s.rolling(10).mean()
        out[c] = {"last": _r(s.iloc[-1], 3), "ma10": _r(ma.iloc[-1], 3),
                  "pctile_1y": _pct_rank(s.to_numpy()[-252:], s.iloc[-1]),
                  "ma10_pctile_1y": _pct_rank(ma.dropna().to_numpy()[-252:], ma.iloc[-1])}
    if (asof - df.index[-1]).days > 4:
        warnings.append(f"putcall last row {out['date']} is stale")
    return out


def build_seasonality(cache_dir: Path, tradeable: list[str], next_session: pd.Timestamp,
                      asof: pd.Timestamp, warnings: list) -> dict:
    p = rad.local_path("atr_seasonal_ranks.parquet", cache_dir)
    if not p.exists():
        warnings.append("missing cache object atr_seasonal_ranks.parquet")
        return {}
    df = pd.read_parquet(p, filters=[("ticker", "in", tradeable)])
    df["Date"] = pd.to_datetime(df["Date"])
    cols = [c for c in df.columns if c.startswith("atr_sznl_")]
    out = {}
    for sym, g in df.groupby("ticker"):
        # ranks are keyed by calendar date; a position entered at the next
        # session sees that date's row (falls back to the last row <= asof).
        row = g[g["Date"] == next_session]
        label = "next_session"
        if row.empty:
            row = g[g["Date"] <= asof].tail(1)
            label = "asof"
        if row.empty:
            continue
        r0 = row.iloc[0]
        out[sym] = {"date": str(r0["Date"])[:10], "for": label,
                    **{c.replace("atr_sznl_", "rank_"): _r(r0[c], 1) for c in cols}}
    return out


def build_events(cache_dir: Path, asof: pd.Timestamp, next_sessions: dict, warnings: list) -> dict:
    out = {"schedule": [], "macro": [], "earnings_next5_count": None, "megacap_earnings": []}
    # Forward schedule (CPI, PPI, NFP, FOMC, opex, VIX expiry, elections) from the
    # repo's public-calendar CSV via macro_calendar. macro_release_history only
    # carries releases once they print, so it never has forward rows.
    try:
        import macro_calendar
        cal = macro_calendar.load_macro_events()
        cal = cal[(cal["date"] > asof) & (cal["date"] <= asof + pd.Timedelta(days=21))]
        for r in cal.sort_values(["date", "time_et"]).itertuples(index=False):
            out["schedule"].append(_clean({"date": r.date, "event": r.event, "detail": r.detail,
                                           "ref_period": r.ref_period, "time_et": r.time_et}))
    except Exception as exc:  # best effort, like every non-tape block
        warnings.append(f"macro schedule unavailable: {exc}")
    if not out["schedule"]:
        warnings.append("no scheduled macro events in the next 21 days (check data/macro_events.csv)")
    mac = _read_parquet(cache_dir, "macro_release_history.parquet", warnings)
    if mac is not None and not mac.empty:
        mac["release_date"] = pd.to_datetime(mac["release_date"])
        win = mac[(mac["release_date"] > asof) & (mac["release_date"] <= asof + pd.Timedelta(days=14))]
        recent = mac[(mac["release_date"] > asof - pd.Timedelta(days=7)) & (mac["release_date"] <= asof)
                     & (mac["country"] == "US") & mac["impact"].fillna("").str.lower().eq("high")]
        out["recent_prints"] = [
            _clean({"event": r.event_name, "date": r.release_date, "actual": r.actual,
                    "consensus": r.consensus, "previous": r.previous,
                    "surprise": getattr(r, "surprise", None)})
            for r in recent.sort_values("release_date").itertuples(index=False)]
        if win.empty:
            pass  # expected: history holds printed releases; the schedule covers the forward view
        else:
            imp = win["impact"].fillna("").str.lower()
            keep = win[(win["country"].isin(["US"]) & imp.isin(["high", "medium"]))
                       | win["event_name"].fillna("").str.contains(CB_PAT)]
            for r in keep.sort_values(["release_date", "time_et"]).itertuples(index=False):
                out["macro"].append(_clean({"event": r.event_name, "date": r.release_date,
                                            "time_et": r.time_et, "country": r.country,
                                            "impact": r.impact, "consensus": r.consensus,
                                            "previous": r.previous}))
    ec = _read_parquet(cache_dir, "earnings_calendar.parquet", warnings,
                       columns=["ticker", "date", "timeOfTheDay"])
    if ec is not None and not ec.empty:
        ec["date"] = pd.to_datetime(ec["date"])
        e5 = ec[(ec["date"] > asof) & (ec["date"] <= next_sessions[5])]
        out["earnings_next5_count"] = int(e5.drop_duplicates(["ticker", "date"]).shape[0])
        e10 = ec[(ec["date"] > asof) & (ec["date"] <= next_sessions[10]) & ec["ticker"].isin(MEGACAPS)]
        for r in e10.drop_duplicates(["ticker", "date"]).sort_values("date").itertuples(index=False):
            out["megacap_earnings"].append(_clean({"ticker": r.ticker, "date": r.date,
                                                   "when": r.timeOfTheDay}))
    return out


# ---------------------------------------------------------------------------
# sleeve / journal
# ---------------------------------------------------------------------------
def _read_journal(path: Path, warnings: list) -> list[dict]:
    if not path.exists():
        return []
    out = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    warnings.append("journal has an unparseable line (skipped)")
    except OSError as exc:
        warnings.append(f"journal unreadable: {exc}")
    return out


def build_sleeve(journal_path: Path, asof: str, tape: dict, warnings: list) -> dict:
    empty = {"capital": SLEEVE_CAPITAL, "nav": SLEEVE_CAPITAL, "cash": SLEEVE_CAPITAL,
             "realized_pnl": 0.0, "positions": {}, "pending": []}
    try:
        import risk_agent_ledger as ledger  # lazy: written concurrently
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"risk_agent_ledger unavailable ({type(exc).__name__}); empty sleeve")
        return empty
    if not journal_path.exists():
        return empty
    try:
        rec = ledger.load(journal_path, pull=False)
        rp = ledger.replay(rec)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"ledger replay failed ({type(exc).__name__}: {exc}); empty sleeve")
        return empty
    out = {"capital": SLEEVE_CAPITAL, "nav": rp.get("nav"), "cash": rp.get("cash"),
           "realized_pnl": rp.get("realized_pnl"), "last_mark_date": rp.get("last_mark_date"),
           "positions": rp.get("positions") or {}, "pending": rp.get("pending") or []}
    return out


def build_decisions(journal_path: Path, warnings: list) -> tuple[list, list]:
    recs = _read_journal(journal_path, warnings)
    decs = [r for r in recs if r.get("kind") == "decision"]
    recent, watch = [], []
    for r in decs[-5:]:
        d = r.get("decision") if isinstance(r.get("decision"), dict) else r
        poss = [{"id": p.get("id"), "action": p.get("action"),
                 "instrument": p.get("instrument")} for p in (d.get("positions") or [])
                if isinstance(p, dict)]
        recent.append({"asof": r.get("asof") or d.get("asof"), "mode": d.get("mode"),
                       "posture": d.get("posture"), "forecasts": d.get("forecasts"),
                       "positions": poss})
    if decs:
        d = decs[-1].get("decision") if isinstance(decs[-1].get("decision"), dict) else decs[-1]
        watch = d.get("watchlist") or []
    return recent, watch


def _read_scoreboard(path: Path, warnings: list):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        warnings.append(f"scoreboard unreadable: {exc}")
        return None


# ---------------------------------------------------------------------------
# main build
# ---------------------------------------------------------------------------
def build_state(asof: str | None = None, cache_dir: Path | str = rad.CACHE_DIR,
                journal_path: Path | str | None = None,
                scoreboard_path: Path | str | None = None) -> dict:
    cache_dir = Path(cache_dir)
    journal_path = Path(journal_path) if journal_path else DEFAULT_JOURNAL
    scoreboard_path = Path(scoreboard_path) if scoreboard_path else DEFAULT_SCOREBOARD
    warnings: list[str] = []

    series_set = {f.series for f in FUTURES.values()}
    wanted = sorted(set(ETFS) | series_set | set(CONTEXT_SERIES) | set(OPTIONABLE) | {"SPY"})
    px = load_prices(cache_dir, wanted, asof)
    spy = px[px["ticker"] == "SPY"]
    if spy.empty:
        raise SystemExit("FATAL: SPY missing from master_prices")
    asof_ts = spy["date"].max()
    asof_s = str(asof_ts)[:10]
    px_groups = {t: g for t, g in px.groupby("ticker", sort=False)}

    nxt = _td_add(asof_ts, 1)
    next_sessions = {k: _td_add(asof_ts, k) for k in (1, 5, 10)}
    opex = []
    y, m = asof_ts.year, asof_ts.month
    while len(opex) < 2:
        d = monthly_opex(y, m)
        if d >= nxt:
            opex.append(str(d)[:10])
        m += 1
        if m > 12:
            y, m = y + 1, 1
    eom = (asof_ts + pd.offsets.MonthEnd(0)).normalize()
    session = {"asof": asof_s, "next_session": str(nxt)[:10], "monthly_opex_next2": opex,
               "days_to_month_end": _sessions_between(asof_ts, eom)}

    iv_hist = _read_parquet(cache_dir, "options/iv_history.parquet", warnings)
    if iv_hist is not None:
        iv_hist = iv_hist[pd.to_datetime(iv_hist["date"]) <= asof_ts]
    tape, quotes = build_tape(px, asof_ts, warnings, iv_hist)
    tradeable = [s for s, m_ in tape.items() if m_["kind"] in ("etf", "future")]

    pos = _read_parquet(cache_dir, "options/positioning_history.parquet", warnings)
    recent, watch = build_decisions(journal_path, warnings)

    state = {
        "schema_version": STATE_SCHEMA, "asof": asof_s,
        "built_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "warnings": warnings, "session": session,
        "sleeve": build_sleeve(journal_path, asof_s, tape, warnings),
        "scoreboard": _read_scoreboard(scoreboard_path, warnings),
        "recent_decisions": recent, "watchlist": watch,
        "dashboard": build_dashboard(cache_dir, asof_s, warnings),
        "vol": build_vol(px_groups, warnings),
        "rates_fx": build_rates_fx(px_groups, warnings),
        "breadth": build_breadth(cache_dir, asof_ts, warnings),
        "putcall": build_putcall(cache_dir, asof_ts, warnings),
        "tape": tape, "quotes": quotes,
        "seasonality": build_seasonality(cache_dir, tradeable, nxt, asof_ts, warnings),
        "events": build_events(cache_dir, asof_ts, next_sessions, warnings),
        "options": build_options(pos, iv_hist, asof_ts, warnings),
        "stress": build_stress(px, warnings),
    }
    try:
        state["data_catalog"] = rad.catalog(cache_dir)
    except Exception as exc:  # noqa: BLE001
        state["data_catalog"] = []
        warnings.append(f"data catalog failed: {exc}")
    return _clean(state)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--asof", default=None)
    ap.add_argument("--out", default=str(ROOT / "data" / "risk_agent_state.json"))
    ap.add_argument("--chains-out", default=str(ROOT / "data" / "risk_agent_chains.json"))
    ap.add_argument("--no-sync", action="store_true")
    a = ap.parse_args(argv)
    if not a.no_sync:
        res = rad.sync()
        bad = {k: v for k, v in res.items() if v in ("failed", "missing")}
        if bad:
            print(f"[risk_agent] sync problems: {bad}", file=sys.stderr)
    state = build_state(a.asof)
    chains = _clean(build_chains(state["asof"]), 5)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(state, separators=(",", ":")), encoding="utf-8")
    cout = Path(a.chains_out)
    cout.write_text(json.dumps(chains, separators=(",", ":")), encoding="utf-8")
    print(f"state: {out} {out.stat().st_size / 1024:.1f} KB; chains: {cout} "
          f"{cout.stat().st_size / 1024:.1f} KB; warnings: {len(state['warnings'])}")
    for w in state["warnings"]:
        print(f"  WARN {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
