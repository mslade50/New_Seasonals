"""Stage A of the PM Weekly: deterministic state.

    python scripts/build_pm_state.py [--asof YYYY-MM-DD] [--out PATH] [--no-sync] [--no-book]

Syncs the PM's allowlisted R2 objects into PM_AGENT_HOME/cache, then writes
PM_AGENT_HOME/state.json: the anchor session, next week's calendar and
resolution date, a weekly recap of the tape, vol, rates/FX, breadth and
put/call, the dashboard as context, CODE-COMPUTED climatology for both claim
types, the PM's own scoreboard and its last briefs, plus the `book` block
(pm_agent_book: live NLV/vol/exposure, the week's fills, the ledger's vol,
exposure and capital efficiency, sleeves, job health) and recent check-ins.

It never reads the Risk Agent's output: the publisher attaches the Risk Agent
readout only after the PM's forecasts have been validated
(docs/claude_ref/pm_agent.md, "Independence").

Market blocks reuse the Risk Agent's pure builders, pointed at the PM cache.
A missing master_prices or SPY/VIX history is fatal; everything else is a
warning.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pm_agent_book as B  # noqa: E402
import pm_agent_data as pad  # noqa: E402
import pm_agent_lab as lab  # noqa: E402
import pm_agent_universe as U  # noqa: E402
import build_risk_agent_state as rab  # noqa: E402  (pure block builders only)
from research_io import read_jsonl  # noqa: E402
from trading_calendar import TRADING_DAY  # noqa: E402


def _r(x, nd=2):
    return rab._r(x, nd)


def build_recap(px: pd.DataFrame, asof: pd.Timestamp, warnings: list) -> dict:
    """Week-to-date (vs the prior week's last close) and 4-week moves."""
    rows = {}
    for sym in U.RECAP_SYMBOLS + U.VOL_SYMBOLS + U.RATE_SYMBOLS:
        if sym not in px.columns:
            warnings.append(f"recap: {sym} absent from master_prices")
            continue
        s = px[sym].dropna()
        s = s[s.index <= asof]
        if len(s) < 30:
            warnings.append(f"recap: {sym} has too little history")
            continue
        if (asof - s.index[-1]).days > 5:
            warnings.append(f"recap: {sym} last bar {s.index[-1].date()} is stale")
            continue
        d0, c0 = lab.prior_week_close(s, asof)
        last = float(s.iloc[-1])
        c20 = float(s.iloc[-21]) if len(s) > 21 else None
        level = sym in U.VOL_SYMBOLS or sym in U.RATE_SYMBOLS
        if level:
            # vol in points; rates (^TNX etc. quote percent) in basis points
            mult = 100.0 if sym in U.RATE_SYMBOLS else 1.0
            row = {"last": _r(last, 3),
                   "chg_week": _r((last - c0) * mult, 2) if c0 is not None else None,
                   "chg_4w": _r((last - c20) * mult, 2) if c20 is not None else None,
                   "unit": "bps" if sym in U.RATE_SYMBOLS else "points",
                   "pctile_1y": rab._pct_rank(s.to_numpy()[-252:], last)}
        else:
            row = {"last": _r(last, 4),
                   "ret_week_pct": _r(100 * (last / c0 - 1)) if c0 else None,
                   "ret_4w_pct": _r(100 * (last / c20 - 1)) if c20 else None,
                   "dist_200d_pct": _r(100 * (last / s.iloc[-200:].mean() - 1)) if len(s) >= 200 else None,
                   "dist_52wh_pct": _r(100 * (last / s.iloc[-252:].max() - 1))}
        row["prior_week_close_date"] = str(d0.date()) if d0 is not None else None
        rows[sym] = row
    ranked = sorted((r["ret_week_pct"], s) for s, r in rows.items()
                    if r.get("ret_week_pct") is not None and s in U.RECAP_SYMBOLS)
    return {"symbols": rows,
            "leaders_week": [s for _, s in ranked[::-1][:5]],
            "laggards_week": [s for _, s in ranked[:5]]}


def build_daily_path(px: pd.DataFrame, asof: pd.Timestamp) -> list[dict]:
    """SPY and VIX day by day over the anchor week (what the tape actually did)."""
    out = []
    monday = asof - pd.Timedelta(days=asof.weekday())
    for sym in ("SPY", "^VIX"):
        if sym not in px.columns:
            continue
        s = px[sym].dropna()
        s = s[s.index <= asof]
        wk = s[s.index >= monday - pd.Timedelta(days=7)]
        prev = None
        for d, v in wk.items():
            if d >= monday and prev is not None:
                out.append({"symbol": sym, "date": str(d.date()), "close": _r(v, 3),
                            "chg": _r(100 * (v / prev - 1), 2) if sym == "SPY" else _r(v - prev, 2),
                            "unit": "pct" if sym == "SPY" else "points"})
            prev = v
    return out


def build_climatology(px: pd.DataFrame, asof: pd.Timestamp, horizon_td: int, vix_last) -> dict:
    out = {}
    for claim, spec in U.CLAIMS.items():
        sym = spec["symbol"]
        kind = "pct" if spec["unit"] == "pct" else "points"
        out[claim] = lab.climatology(px[sym], asof, horizon_td, kind) if sym in px.columns else {"n": 0}
    out["spy_week_return"]["vix_implied"] = lab.implied_band(vix_last, horizon_td)
    return out


def _recent_briefs(journal: Path, k: int = 4) -> list[dict]:
    try:
        recs = read_jsonl(journal)
    except (OSError, ValueError):
        return []
    briefs = [r for r in recs if r.get("kind") in ("brief", "stand_down")][-k:]
    res = {r.get("forecast_id"): r for r in recs if r.get("kind") == "resolution"}
    out = []
    for b in briefs:
        fcs = []
        for f in [r for r in recs if r.get("kind") == "forecast" and r.get("asof") == b.get("asof")]:
            rr = res.get(f.get("forecast_id")) or {}
            fcs.append({k2: f.get(k2) for k2 in ("claim_type", "p_up", "q10", "q90", "resolves_on")}
                       | {"outcome": rr.get("value"), "status": rr.get("status", "open")})
        out.append({"asof": b.get("asof"), "week": b.get("week"),
                    "headline": (b.get("payload") or {}).get("headline"), "forecasts": fcs})
    return out


def build_state(asof: str | None = None, cache_dir: Path | str | None = None,
                journal: Path | str | None = None, scoreboard: Path | str | None = None,
                with_book: bool = True) -> dict:
    cdir = Path(cache_dir or U.cache_dir())
    journal = Path(journal or U.journal_path())
    scoreboard = Path(scoreboard or U.scoreboard_path())
    warnings: list[str] = []

    syms = sorted(set(U.RECAP_SYMBOLS) | set(U.VOL_SYMBOLS) | set(U.RATE_SYMBOLS)
                  | {f for f in rab.FX_SERIES})
    try:
        long_px = rab.load_prices(cdir, syms, asof)
    except SystemExit as exc:
        raise SystemExit(f"FATAL: {exc}")
    px = long_px.pivot_table(index="date", columns="ticker", values="Close", aggfunc="last")
    px = px.astype("float64").sort_index()
    for need in ("SPY", "^VIX"):
        if need not in px.columns or px[need].dropna().empty:
            raise SystemExit(f"FATAL: {need} missing from master_prices")
    asof_ts = px["SPY"].dropna().index.max()
    asof_s = str(asof_ts.date())
    if (pd.Timestamp(asof_s) - px["^VIX"].dropna().index.max()).days > 3:
        warnings.append(f"^VIX last bar {px['^VIX'].dropna().index.max().date()} lags SPY {asof_s}")
    week = lab.target_week(asof_ts)
    groups = {t: g for t, g in long_px.groupby("ticker", sort=False)}
    next_sessions = {k: asof_ts + k * TRADING_DAY for k in (1, 5, 10)}
    vix_last = float(px["^VIX"].dropna().iloc[-1])

    sb = None
    if scoreboard.exists():
        try:
            sb = json.loads(scoreboard.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            warnings.append(f"scoreboard unreadable: {exc}")

    state = {
        "schema_version": U.STATE_SCHEMA, "asof": asof_s, "week": lab.week_key(asof_ts),
        "built_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "warnings": warnings,
        "target_week": week,
        "recap": build_recap(px, asof_ts, warnings),
        "daily_path": build_daily_path(px, asof_ts),
        "vol": rab.build_vol(groups, warnings),
        "rates_fx": rab.build_rates_fx(groups, warnings),
        "breadth": rab.build_breadth(cdir, asof_ts, warnings),
        "putcall": rab.build_putcall(cdir, asof_ts, warnings),
        "events": rab.build_events(cdir, asof_ts, next_sessions, warnings),
        "dashboard": rab.build_dashboard(cdir, asof_s, warnings),
        "climatology": build_climatology(px, asof_ts, week["horizon_td"], vix_last),
        # full-precision raw closes the forecasts resolve against
        "anchors": {spec["symbol"]: {"date": str(px[spec["symbol"]].dropna().index.max().date()),
                                     "close": float(px[spec["symbol"]].dropna().iloc[-1])}
                    for spec in U.CLAIMS.values()},
        "scoreboard": sb,
        "recent_briefs": _recent_briefs(journal),
        "data_catalog": pad.catalog(cdir),
    }
    if with_book:
        monday = asof_ts - pd.Timedelta(days=asof_ts.weekday())
        try:
            state["book"] = B.build_book(asof_s, str(monday.date()), cdir, warnings)
        except Exception as exc:  # noqa: BLE001 - the brief still ships market-only
            warnings.append(f"book block failed: {type(exc).__name__}: {exc}")
        state["checkins"] = _recent_checkins(journal)
    return rab._clean(state)


def _recent_checkins(journal: Path, k: int = 5) -> list[dict]:
    try:
        recs = [r for r in read_jsonl(journal) if r.get("kind") == "check_in"][-k:]
    except (OSError, ValueError):
        return []
    return [{"date": r.get("date"), "n": r.get("n_exceptions"),
             "exceptions": [{"kind": e.get("kind"), "message": e.get("message"),
                             "days_running": e.get("days_running")} for e in r.get("exceptions") or []]}
            for r in recs]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--asof", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--no-book", action="store_true", help="market-only state (no book block)")
    a = ap.parse_args(argv)
    if not a.no_sync:
        res = pad.sync()
        if not a.no_book:
            res.update(B.sync_book()["result"])
        bad = {k: v for k, v in res.items() if v in ("failed", "missing")}
        if bad:
            print(f"[pm_agent] sync problems: {bad}", file=sys.stderr)
    state = build_state(a.asof, with_book=not a.no_book)
    out = Path(a.out) if a.out else U.state_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(state, separators=(",", ":")), encoding="utf-8")
    print(f"state: {out} {out.stat().st_size / 1024:.1f} KB; asof {state['asof']}; "
          f"resolves {state['target_week']['resolves_on']} ({state['target_week']['horizon_td']} td); "
          f"warnings: {len(state['warnings'])}")
    for w in state["warnings"]:
        print(f"  WARN {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
