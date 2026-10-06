"""Extend the frozen Portfolio research replay with the local IBKR minute store.

This is a research producer, never a site build or broker-order process. Run on
the research machine with the retained reference engines and IBKR history store.
New downloads and audit evidence go under artifacts/. The reviewed JSON is the
only promoted output; production is rebuilt separately in GitHub Actions.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import importlib.util
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_intraday_replay as book
from scripts import backtest_legend_ema_futures as futures
from open_breakout.inputs import legacy_score

NY = "America/New_York"
IBKR_ROOT = Path.home() / "OneDrive/trading_ibkr/legend_ema_futures_staging"
SPLICE = pd.Timestamp("2026-09-01", tz="UTC")
WARMUP = pd.Timestamp("2026-06-01", tz="UTC")
FROZEN_ENGINE_HASHES = {
    "Open Breakout": "c4d351bf72f367fdb30421917cbbb7a106ae0177bab9cd11ebf8e2ac58108cef",
    "Legend ETF": "810303bae03998bc97052aa26ecdc60f0cf66d85291c0dec5642cc6b85236a18",
}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fetch_roll_overlap(store, month, output, port, client_id):
    """Fill the old expiry's final full week without modifying the live store."""
    from ib_insync import IB
    date = datetime.strptime(month, "%Y%m")
    if date.month not in (3, 6, 9, 12):
        raise ValueError("A quarterly YYYYMM expiry is required")
    expiry = store.quarterly_expiry(date.year, date.month)
    if expiry >= pd.Timestamp.now(tz=NY).date():
        raise ValueError("Roll overlap backfill is only for completed rolls")
    ib = IB()
    ib.connect("127.0.0.1", port, clientId=client_id, timeout=20, readonly=True)
    try:
        for root in ("ES", "NQ"):
            path = output / f"{root}_roll_overlap_{month}.parquet"
            if path.exists():
                continue
            contract = store.qualify_quarterly(ib, root, date.year, date.month)
            frame = store.fetch_window(ib, contract, root, month, expiry - timedelta(days=7), expiry - timedelta(days=1))
            if frame.empty:
                raise ValueError(f"No {root} {month} overlap returned")
            frame.to_parquet(path)
    finally:
        ib.disconnect()


def volume_front(raw):
    """Use the frozen research's volume leader with a two trade-date lag.

The store retains both expiries at rolls. Its calendar-front helper must not
replace the historical volume-front convention or manufacture an EMA reset.
"""
    local = raw.index.tz_convert(NY)
    x = raw.loc[local.hour != 17].copy()
    local = x.index.tz_convert(NY)
    x["session"] = local.normalize().tz_localize(None) + pd.to_timedelta((local.hour >= 18).astype(int), unit="D")
    volumes = x.groupby(["session", "symbol"]).volume.sum().unstack(fill_value=0).sort_index()
    leader = volumes.idxmax(axis=1)
    if (volumes.max(axis=1) <= 0).any():
        raise ValueError("No volume leader in an IBKR session")
    ties = volumes.eq(volumes.max(axis=1), axis=0).sum(axis=1).gt(1)
    deciding = {}
    for utc_day in sorted(set(x.index.date)):
        prior = volumes.index[volumes.index < pd.Timestamp(utc_day)]
        deciding[utc_day] = prior[-2] if len(prior) >= 2 else None
    selected = []
    for ts, symbol in zip(x.index, x.symbol):
        day = deciding[ts.date()]
        if day is None:
            selected.append(False)  # warm-up only, before the August overlap
        elif ties.loc[day]:
            raise ValueError(f"Ambiguous volume front on {day}")
        else:
            selected.append(symbol == leader.loc[day])
    out = x.loc[selected].drop(columns="session")
    if out.index.has_duplicates:
        raise ValueError("Duplicate selected futures minutes")
    return out


def join_minutes(root, store, through, output):
    frames = []
    for path in sorted((ROOT / "artifacts/databento/parquet").glob("*.parquet")):
        frame = pd.read_parquet(path, columns=["instrument_id", "open", "high", "low", "close", "volume"],
                                filters=[("symbol", "==", f"{root}.v.0"), ("ts_event", ">=", WARMUP)])
        if len(frame):
            frames.append(frame)
    archive = pd.concat(frames).sort_index()
    archive = archive.loc[~archive.index.duplicated(keep="last")]
    raw = store.read_store(root)
    for supplement in sorted(output.glob(f"{root}_roll_overlap*.parquet")):
        raw = store.merge_rows(raw, pd.read_parquet(supplement))
    spliced, overlap = store.splice_ids(raw, archive)
    if not overlap.get("matched"):
        raise ValueError(f"{root}: IBKR/Databento overlap failed: {overlap}")
    front = volume_front(spliced)
    columns = ["instrument_id", "open", "high", "low", "close", "volume"]
    joined = pd.concat([archive.loc[archive.index < SPLICE, columns],
                        front.loc[front.index >= SPLICE, columns]]).sort_index()
    end = (pd.Timestamp(through, tz=NY) + pd.Timedelta(hours=17)).tz_convert("UTC")
    joined = joined.loc[joined.index < end]
    if joined.index.has_duplicates or joined.empty:
        raise ValueError(f"{root}: empty or duplicate joined series")
    vals = joined[["open", "high", "low", "close"]]
    if not np.isfinite(vals.to_numpy()).all() or (vals <= 0).any().any():
        raise ValueError(f"{root}: invalid prices")
    if (vals.high < vals[["open", "close", "low"]].max(axis=1)).any() or (
            vals.low > vals[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError(f"{root}: inconsistent OHLC")
    changed = front.symbol.ne(front.symbol.shift())
    overlap["ibkr_rolls"] = [{"time": str(ts), "contract": str(front.loc[ts, "symbol"])}
                              for ts in front.index[changed] if ts >= SPLICE]
    overlap["through"] = through
    overlap["last_bar"] = str(joined.index[-1])
    return joined, overlap


def cash_schedule(through):
    import exchange_calendars as xc
    cal = xc.get_calendar("XNYS", start="2026-06-01", end=through)
    schedule = cal.schedule.copy()
    schedule.index = schedule.index.tz_localize(None)
    return schedule.rename(columns={"market_close": "close"})


def ob_sessions(minutes, schedule):
    """Frozen full-session TR / RTH eligibility, without a dated research cutoff."""
    frame = minutes.copy()
    frame["ts"] = frame.index.tz_convert(NY).tz_localize(None)
    frame["date"] = frame.ts.dt.normalize()
    clock = frame.ts.dt.hour * 60 + frame.ts.dt.minute
    full = frame.loc[(clock < 1020) | (clock >= 1080)].copy()
    full["session"] = full.date + pd.to_timedelta((full.ts.dt.hour >= 18).astype(int), unit="D")
    ranges = full.groupby("session").agg(high=("high", "max"), low=("low", "min"),
        close=("close", "last"), instrument_id=("instrument_id", "first"),
        instrument_count=("instrument_id", "nunique"))
    previous_close = ranges.close.shift()
    ranges["tr"] = pd.concat([ranges.high - ranges.low, (ranges.high - previous_close).abs(),
                              (ranges.low - previous_close).abs()], axis=1).max(axis=1)
    roll = ranges.instrument_id.ne(ranges.instrument_id.shift()) | ranges.instrument_count.ne(1)
    ranges.loc[roll, "tr"] = np.nan
    atr = ranges.tr.dropna().rolling(20, min_periods=20).mean().shift(1)
    rth = frame.loc[(clock >= 570) & (clock < 960)]
    groups = dict(tuple(rth.groupby("date")))
    result, audit = [], []
    for day, cal in schedule.iterrows():
        rec = {"date": day.strftime("%Y-%m-%d"), "status": "ok"}
        audit.append(rec)
        if cal["close"].tz_convert(NY).hour != 16:
            rec["status"] = "early_close"
            continue
        g = groups.get(day)
        expected = pd.date_range(day + pd.Timedelta(minutes=570), periods=390, freq="min")
        if g is None or not pd.DatetimeIndex(g.ts).equals(expected):
            rec["status"] = "missing_rth_minutes"
            continue
        prior = ranges.index[ranges.index < day]
        prior = prior[-1] if len(prior) else None
        if prior is None or not np.isfinite(ranges.loc[prior, "tr"]):
            rec["status"] = "prior_roll"
            continue
        if g.instrument_id.nunique() != 1 or g.instrument_id.iloc[0] != ranges.loc[prior, "instrument_id"]:
            rec["status"] = "contract_changed_since_prior"
            continue
        tr = float(ranges.loc[prior, "tr"])
        if tr <= 0:
            raise ValueError(f"Nonpositive TR on {day}")
        rec.update(prior_date=str(prior.date()), prior_tr=tr, ratio=tr / atr.get(prior, np.nan))
        result.append((day, tr, list(g[["ts", "open", "high", "low", "close"]].itertuples(index=False, name=None))))
    return result, pd.DataFrame(audit)


def ob_extension(all_minutes, schedule, risk, engine, cutoff, through, output):
    rows, audit_rows = [], []
    for root, spec in book.OB_MARKETS.items():
        sessions, audit = ob_sessions(all_minutes[root], schedule)
        audit.to_csv(output / f"{root}_sessions.csv", index=False)
        missing = audit.loc[audit.date.gt(cutoff) & audit.status.eq("missing_rth_minutes")]
        if len(missing):
            raise ValueError(f"{root}: unvalidated missing sessions: {missing.date.tolist()}")
        lookup = audit.set_index("date")
        previous_r = None
        reference = pd.read_csv(book.OB_CC / f"{root}_base_trades.csv")
        overlap_count = 0
        skip_reference = book.ob_skip_days()[root]
        rules = engine.Rules(risk=.01, multiplier=spec["mult"], tick=.25, leverage=math.inf,
            slip_bps=0, slip_points=.25, fee=.475, min_fee=0, exit_minute=955, entry_cutoff_minute=690)
        for day, tr, bars in sessions:
            key = day.strftime("%Y-%m-%d")
            prior_cash = schedule.index[schedule.index < day]
            if not len(prior_cash):
                continue
            score = legacy_score(risk, prior_cash[-1].strftime("%Y-%m-%d"))
            selected = engine.Rules(**{**vars(rules), "side": "long" if score < 20 else "both"})
            _, trades, _ = engine.simulate_day(bars, tr, 100000., selected)
            ratio = float(lookup.loc[key, "ratio"])
            if key > cutoff and not math.isfinite(ratio):
                raise ValueError(f"{root}: missing prior range warm-up on {key}")
            skipped = math.isfinite(ratio) and ratio >= 1.25 and previous_r is not None and previous_r >= 2
            previous_r = sum(t["net_r"] for t in trades)  # shadow prior, including skipped days
            if "2026-08-03" <= key <= book.OB_SPAN[1]:
                old = reference.loc[reference.date.eq(key)]
                if len(old) != len(trades):
                    raise ValueError(f"{root}/{key}: frozen trade count parity failed")
                for t, old_t in zip(trades, old.itertuples()):
                    for field in ["side", "entry", "exit", "net_r"]:
                        if not math.isclose(float(t[field]), float(getattr(old_t, field)), abs_tol=1e-7):
                            raise ValueError(f"{root}/{key}: frozen {field} parity failed")
                    if t["entry_time"] != old_t.entry_time or t["exit_time"] != old_t.exit_time:
                        raise ValueError(f"{root}/{key}: frozen timestamp parity failed")
                if skipped != (day in skip_reference):
                    raise ValueError(f"{root}/{key}: frozen range-skip parity failed")
                overlap_count += 1
            audit_rows.append({"root": root, "date": key, "score": score, "range_ratio": ratio,
                               "skip": skipped, "shadow_R": previous_r})
            if key <= cutoff or skipped:
                continue
            for t in trades:
                rows.append({**t, "root": root, "date": key, "prior_tr": tr,
                             "usd_per_micro": t["pnl"] / t["qty"]})
        if not overlap_count:
            raise ValueError(f"{root}: no frozen overlap sessions verified")
        print(f"{root}: {overlap_count} frozen overlap sessions verified", flush=True)
    pd.DataFrame(audit_rows).to_csv(output / "open_breakout_decisions.csv", index=False)
    daily_cap, open_cap = book.BASIS * .0075, book.BASIS * .0025
    result = []
    for row in sorted(rows, key=lambda t: (t["entry_time"], t["root"])):
        root = row["root"]
        spec = book.OB_MARKETS[root]
        pc = book.ceil_tick(.25 * row["prior_tr"], .25) * spec["mult"] + 2 * book.OB_FEE_SIDE + spec["mult"]
        planned = min(60, math.floor(book.BASIS * spec["bps"] / 1e4 / pc))
        same = [t for t in result if t["Entry_Date"] == row["date"]]
        reserved = sum(t["reserved_risk"] for t in same)
        opened = sum(t["reserved_risk"] for t in same if t["Exit_Time"] > row["entry_time"])
        qty = min(planned, math.floor(max(0, min(daily_cap - reserved, open_cap - opened)) / pc + 1e-9))
        if qty < 1:
            continue
        ticker = "MNQ" if root == "NQ" else "MES"
        result.append({"trade_id": f"intraday:open_breakout:{row['date']}:{ticker}:{len(same)}",
            "Strategy": "Open Breakout", "Tier": "Intraday", "book": "intraday", "Ticker": ticker,
            "Direction": "Long" if row["side"] == 1 else "Short", "Signal_Date": row["date"],
            "Entry_Date": row["date"], "Exit_Date": row["date"], "Entry_Time": row["entry_time"],
            "Exit_Time": row["exit_time"], "Entry_Price": row["entry"], "Exit_Price": row["exit"],
            "Quantity": qty, "R": round(row["net_r"], 6), "PnL_flat": round(qty * row["usd_per_micro"], 2),
            "Risk_flat": round(.25 * row["prior_tr"] * spec["mult"] * qty, 4),
            "Return_Pct": round(100 * row["side"] * (row["exit"] / row["entry"] - 1), 6),
            "Hold_Days": 0, "Hold_Minutes": (pd.Timestamp(row["exit_time"]) - pd.Timestamp(row["entry_time"])).total_seconds() / 60,
            "Exit_Type": row["reason"], "Open": False, "reserved_risk": qty * pc})
    for t in result:
        t.pop("reserved_risk")
    return result


def legend_candidates(all_minutes, cutoff, through):
    result = []
    for root, minutes in all_minutes.items():
        # The reference futures engine receives New York indexes from its loader.
        # The collector uses UTC; convert before any RTH clock or session tests.
        minutes = minutes.tz_convert(NY)
        bars = futures.build_15_minute_bars(minutes)
        daily = futures.build_daily_sessions(minutes, bars)
        wanted = daily.loc[daily.complete_rth & daily.setup_rth.fillna(False) & daily.trend_ratio.ge(.75)]
        for setup_day, setup in wanted.iterrows():
            entry = setup.next_session
            if pd.isna(entry) or entry.strftime("%Y-%m-%d") <= cutoff or entry.strftime("%Y-%m-%d") > through:
                continue
            if not daily.loc[entry, "complete_rth"] or futures._spans_contract_change(minutes, setup_day, entry):
                continue
            result.append({"root": root, "etf": "SPY" if root == "ES" else "QQQ", "setup_date": setup_day,
                "entry_date": entry, "prior_futures_trend_direction": int(setup.trend_direction),
                "prior_futures_trend_ratio": float(setup.trend_ratio)})
    return pd.DataFrame(result, columns=["root", "etf", "setup_date", "entry_date",
        "prior_futures_trend_direction", "prior_futures_trend_ratio"]).sort_values(["entry_date", "root"])


def verify_legend_overlap(minutes):
    actual = legend_candidates(minutes, "2026-06-30", "2026-08-05")
    reference = pd.read_csv(book.LG / "candidates.csv", parse_dates=["setup_date", "entry_date"])
    reference = reference.loc[reference.root.isin(["ES", "NQ"]) &
        reference.entry_date.between("2026-07-01", "2026-08-05")].sort_values(["entry_date", "root"])
    columns = ["root", "etf", "setup_date", "entry_date", "prior_futures_trend_direction", "prior_futures_trend_ratio"]
    if not len(reference):
        raise ValueError("No frozen Legend overlap candidates")
    pd.testing.assert_frame_equal(actual[columns].reset_index(drop=True), reference[columns].reset_index(drop=True),
                                  check_dtype=False, rtol=1e-8, atol=1e-8)
    return len(actual)


def extend_strategy(strategy, additions, through):
    out = copy.deepcopy(strategy)
    cutoff = strategy["span"][1]
    dates = [d.strftime("%Y-%m-%d") for d in book.sessions((strategy["span"][0], through))]
    if any(t["Exit_Date"] <= cutoff or t["Exit_Date"] > through for t in additions):
        raise ValueError("Extension overlaps retained history or exceeds coverage")
    if any(t["Exit_Date"] not in dates for t in additions):
        raise ValueError("Extension trade is not on a cash session")
    out["trades"].extend(additions)
    ids = [t["trade_id"] for t in out["trades"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate trade IDs")
    old_daily = dict(strategy["daily"])
    out["daily"] = [[d, old_daily.get(d, round(sum(t["PnL_flat"] for t in additions if t["Exit_Date"] == d), 2))] for d in dates]
    for market, old in strategy["by_market"].items():
        ticker = {"NQ": "MNQ", "ES": "MES"}.get(market, market)
        old = dict(old)
        out["by_market"][market] = [[d, old.get(d, round(sum(t["PnL_flat"] for t in additions
            if t["Exit_Date"] == d and t["Ticker"] == ticker), 2))] for d in dates]
    out["span"][1] = through
    out["stats"] = book.stats(pd.Series([v for _, v in out["daily"]], index=pd.to_datetime(dates)),
                               len(out["trades"]), tuple(out["span"]))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--through", required=True, help="Completed cash-session date YYYY-MM-DD")
    p.add_argument("--ibkr-root", type=Path, default=IBKR_ROOT)
    p.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/intraday-replay-refresh")
    p.add_argument("--fetch-etf", action="store_true", help="Read-only IBKR historical backfill")
    p.add_argument("--fetch-roll-overlap", help="Old expiry's final week, e.g. 202609 (read-only IBKR)")
    p.add_argument("--port", type=int, default=7496)
    p.add_argument("--client-id", type=int, default=929171)
    p.add_argument("--write", action="store_true", help="Promote validated research JSON; cloud deploy remains separate")
    args = p.parse_args()
    today = pd.Timestamp.now(tz=NY).date().isoformat()
    if args.through >= today:
        raise ValueError("Use a completed session before today")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    doc = json.loads(book.OUT.read_text(encoding="utf-8"))
    old = {s["id"]: s for s in doc["strategies"]}
    if any(args.through <= s["span"][1] for s in old.values()):
        raise ValueError("Coverage must advance both strategies")
    import cache_io
    for key, file in [("rd2_fragility.parquet", "risk.parquet"),
                      ("intraday/15min/SPY.parquet", "SPY_15min.parquet"),
                      ("intraday/15min/QQQ.parquet", "QQQ_15min.parquet")]:
        if not cache_io.download_to_local(key, str(output / file)):
            raise RuntimeError(f"Canonical R2 input unavailable: {key}")
    risk = pd.read_parquet(output / "risk.parquet")
    schedule = cash_schedule(args.through)
    if pd.Timestamp(args.through) not in schedule.index:
        raise ValueError("Coverage end is not a cash session")
    store = load_module("intraday_ibkr_store", args.ibkr_root / "es_nq_store.py")
    if args.fetch_roll_overlap:
        fetch_roll_overlap(store, args.fetch_roll_overlap, output, args.port, args.client_id)
    for name, path in [("Open Breakout", book.OB_CC / "engine.py"),
                       ("Legend ETF", book.LG / "backtest_futures_signal_etf_execution.py")]:
        if sha256(path) != FROZEN_ENGINE_HASHES[name]:
            raise ValueError(f"{name}: reference engine changed; review before extending its replay")
    engine = load_module("frozen_ob_engine", book.OB_CC / "engine.py")
    etf = load_module("frozen_etf_engine", book.LG / "backtest_futures_signal_etf_execution.py")
    minutes, overlaps = {}, {}
    for root in ["ES", "NQ"]:
        minutes[root], overlaps[root] = join_minutes(root, store, args.through, output)
        print(root, json.dumps(overlaps[root]), flush=True)
    ob = ob_extension(minutes, schedule, risk, engine, old["open_breakout"]["span"][1], args.through, output)
    legend_overlap_count = verify_legend_overlap(minutes)
    candidates = legend_candidates(minutes, old["legend_ema"]["span"][1], args.through)
    candidates.to_csv(output / "legend_candidates.csv", index=False)
    etf.HERE = output
    etf.EVENT_DIR = output / "ibkr_1m_events"
    etf.LOCAL_15M_DIR = output
    print(f"Legend: {len(candidates)} extension candidates", flush=True)
    if len(candidates) and args.fetch_etf:
        fetched = asyncio.run(etf.fetch_ibkr_events(candidates, args.port, args.client_id, 1, None, None))
        if any(r["status"] != "ok" for r in fetched["results"]):
            raise ValueError("ETF backfill incomplete; coverage will not advance")
    if len(candidates):
        # Refresh actions: retained research file may predate September ex-dividends.
        import yfinance as yf
        etf.HERE = output / "actions" / args.through
        etf.HERE.mkdir(parents=True, exist_ok=True)
        etf.ETF_BY_ROOT = {"ES": "SPY", "NQ": "QQQ"}
        yf.set_tz_cache_location(str(etf.HERE / "yf_cache"))
        actions = etf.fetch_corporate_actions()
        if set(actions.loc[actions.dividend.gt(0), "etf"]) != {"SPY", "QQQ"}:
            raise ValueError("ETF dividend history unavailable")
        for candidate in candidates.itertuples():
            seed_frame = pd.read_parquet(output / f"{candidate.etf}_15min.parquet")
            previous = schedule.index[schedule.index < candidate.entry_date][-1]
            expected = schedule.loc[previous, "close"].tz_convert(NY).tz_localize(None) - pd.Timedelta(minutes=15)
            before = seed_frame.loc[pd.to_datetime(seed_frame.ts) < candidate.entry_date]
            if pd.to_datetime(before.ts).max() != expected:
                raise ValueError(f"Stale ETF EMA seed for {candidate.etf}/{candidate.entry_date.date()}")
        seeds = etf.load_local_ema_seeds(candidates)
    lg, decisions = [], []
    for _, candidate in candidates.iterrows():
        path = etf.event_path(candidate.root, candidate.entry_date)
        if not path.exists():
            raise ValueError(f"ETF minutes missing: {path}; rerun with --fetch-etf")
        action = actions.loc[actions.etf.eq(candidate.etf) & actions.date.eq(candidate.entry_date)]
        dividend = float(action.dividend.sum())
        result = etf.simulate_candidate(candidate, pd.read_parquet(path), etf.VARIANTS[0], dividend > 0,
                                         dividend, seeds.get((candidate.etf, candidate.entry_date.normalize())))
        decisions.append(result)
        if result.get("skip_reason") in {"missing_ibkr_window", "incomplete_etf_entry_session", "ema_not_ready"}:
            raise ValueError(f"ETF candidate unvalidated: {result}")
        if not result["traded"] or result["side"] != "long":
            continue
        day = candidate.entry_date.strftime("%Y-%m-%d")
        usd = book.LG_WEIGHT[candidate.etf] * book.BASIS * result["net_return_bps_2bp"] / 1e4
        lg.append({"trade_id": f"intraday:legend_ema:{day}:{candidate.etf}", "Strategy": "Legend EMA",
            "Tier": "Intraday", "book": "intraday", "Ticker": candidate.etf, "Direction": "Long",
            "Signal_Date": candidate.setup_date.strftime("%Y-%m-%d"), "Entry_Date": day, "Exit_Date": day,
            "Entry_Time": str(result["entry_ts"]), "Exit_Time": str(result["exit_ts"]),
            "Entry_Price": result["entry_price"], "Exit_Price": result["exit_price"], "R": None,
            "Risk_flat": None, "PnL_flat": round(usd, 2), "Return_Pct": round(result["net_return_bps_2bp"] / 100, 6),
            "Hold_Days": 0, "Hold_Minutes": result["holding_minutes"], "Exit_Type": result["exit_reason"], "Open": False})
    pd.DataFrame(decisions).to_csv(output / "legend_decisions.csv", index=False)
    updated = copy.deepcopy(doc)
    updated["generated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    updated["strategies"] = [extend_strategy(s, ob if s["id"] == "open_breakout" else lg, args.through)
                               for s in doc["strategies"]]
    for s in updated["strategies"]:
        s["notes"] += f" IBKR extension validated through {args.through}; historical rows retained."
    provenance = {"through": args.through, "overlap": overlaps,
        "retained_snapshot_sha256": sha256(book.OUT), "ob_engine_sha256": sha256(book.OB_CC / "engine.py"),
        "etf_engine_sha256": sha256(book.LG / "backtest_futures_signal_etf_execution.py"),
        "ibkr_inputs_sha256": {p.name: sha256(p) for p in (args.ibkr_root / "data/es_nq_1m").glob("*.parquet")},
        "roll_overlap_sha256": {p.name: sha256(p) for p in output.glob("*_roll_overlap*.parquet")},
        "r2_inputs_sha256": {p.name: sha256(p) for p in output.glob("*.parquet") if p.name in
                             {"risk.parquet", "SPY_15min.parquet", "QQQ_15min.parquet"}},
        "etf_actions_sha256": sha256(etf.HERE / "etf_corporate_actions.csv") if len(candidates) else None,
        "added_trades": {"open_breakout": len(ob), "legend_ema": len(lg)}, "legend_candidates": len(candidates),
        "legend_overlap_candidates": legend_overlap_count}
    updated["refresh"] = provenance
    book.validate(updated)
    candidate_path = output / "intraday_replay.json"
    candidate_path.write_text(json.dumps(updated, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    (output / "validation.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    if args.write:
        book.OUT.write_bytes(candidate_path.read_bytes())
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == "__main__":
    main()
