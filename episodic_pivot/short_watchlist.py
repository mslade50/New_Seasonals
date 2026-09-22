"""Source-replayable ATR Extended Gap Up research supplement; never an order path."""
from __future__ import annotations

import hashlib
import html
import json
import math
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import pandas as pd

from filters import check_signal_live
from indicators import calculate_indicators
from trading_calendar import TRADING_DAY
from .schema import parse_timestamp
from .listed_universe import capture_universe, validate_universe

STRATEGY = "ATR Extended Gap Up"
SCHEMA = "EP_ATR_EXTENDED_SHORT_WATCHLIST_V2"
NY = ZoneInfo("America/New_York")


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def configured_screen() -> dict:
    from strategy_config import STRATEGY_BOOK

    strategy = next(s for s in STRATEGY_BOOK if s["name"] == STRATEGY)
    settings = deepcopy(strategy["settings"])
    # Retain the daily setup criteria independently of the traded universe.
    if not (
        settings["trade_direction"] == "Short"
        and settings["dist_ma_type"] == "SMA 50"
        and settings["dist_logic"] == "Greater Than (>)"
        and settings["use_ma_dist_filter"] and settings["use_vol"]
        and settings["vol_logic"] == ">"
        and settings["use_t1_open_filter"]
        and settings["t1_open_filters"] == [{"logic": ">", "reference": "Close", "atr_offset": 0.5}]
    ):
        raise ValueError("ATR Extended Gap Up configuration needs a watchlist review")
    return json.loads(json.dumps({"strategy": STRATEGY, "settings": settings,
                                  "universe_label": "Fresh US-listed equities, including ADRs; ETFs, preferreds, warrants, rights, units and debt excluded"}))


def screen_from_universe(evidence: dict, target: str) -> dict:
    listings, coverage = validate_universe(evidence, target, now=datetime.now(timezone.utc))
    return {**configured_screen(), "universe": sorted(listings), "listings": listings,
            "universe_coverage": coverage}


def _session(value: str) -> date:
    day = date.fromisoformat(value)
    if not TRADING_DAY.is_on_offset(pd.Timestamp(day)):
        raise ValueError("Short watchlist target must be an NYSE session")
    return day


def normalize_download(frame: pd.DataFrame, symbol: str) -> list[dict]:
    """Select one ticker before flattening Yahoo's (Price, Ticker) columns."""
    if isinstance(frame.columns, pd.MultiIndex):
        ticker_level = "Ticker" if "Ticker" in frame.columns.names else 1
        frame = frame.xs(symbol, level=ticker_level, axis=1)
    frame = frame.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame.columns = [str(c).title() for c in frame.columns]
    columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    frame = frame[columns].dropna(how="all")
    if frame.index.has_duplicates:
        raise ValueError("Duplicate daily source dates")
    return [{"date": str(pd.Timestamp(idx).date()), **{
        c: (float(value) if pd.notna(value) and math.isfinite(float(value)) else None)
        for c, value in zip(columns, row)}} for idx, row in zip(frame.index, frame.to_numpy())]


def analyze_bars(symbol: str, rows: list[dict], target: str, screen: dict) -> dict | None:
    session = _session(target)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("NO_DAILY_HISTORY")
    frame.index = pd.to_datetime(frame.pop("date"), errors="raise")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("DUPLICATE_OR_UNSORTED_DAILY_HISTORY")
    # Today's partial daily candle must never create a setup or change its ATR.
    frame = frame[frame.index < pd.Timestamp(session)]
    prior = pd.Timestamp(session) - TRADING_DAY
    expected = pd.date_range(end=prior, periods=63, freq=TRADING_DAY)
    if len(frame) < 63 or frame.index[-1] != prior or not expected.isin(frame.index).all():
        raise ValueError("STALE_OR_INCOMPLETE_63_SESSION_HISTORY")
    cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    values = frame[cols].apply(pd.to_numeric, errors="raise")
    if values.isna().any().any() or not all(math.isfinite(float(v)) for v in values.to_numpy().ravel()):
        raise ValueError("NONFINITE_DAILY_HISTORY")
    if (values[["Open", "High", "Low", "Close", "Adj Close"]] <= 0).any().any() or (values.Volume < 0).any():
        raise ValueError("NONPOSITIVE_DAILY_HISTORY")
    if ((values.High < values[["Open", "Close", "Low"]].max(axis=1)) |
            (values.Low > values[["Open", "Close", "High"]].min(axis=1))).any():
        raise ValueError("INCONSISTENT_DAILY_OHLC")
    adjusted = values[["Open", "High", "Low", "Close", "Volume"]].copy()
    factors = values["Adj Close"] / values.Close
    adjusted[["Open", "High", "Low", "Close"]] = adjusted[["Open", "High", "Low", "Close"]].mul(factors, axis=0)
    # Cheap exact necessary gates keep broad-universe replay fast. Any survivor
    # still runs through the shared indicator and full live-filter functions.
    settings = screen["settings"]
    volume_mean = adjusted.Volume.tail(63).mean()
    if (adjusted.Close.iloc[-1] < settings["min_price"] or volume_mean < settings["min_vol"]
            or volume_mean <= 0 or adjusted.Volume.iloc[-1] <= settings["vol_thresh"] * volume_mean):
        return None
    indicators = calculate_indicators(adjusted, {}, symbol)
    if not check_signal_live(indicators, screen["settings"], ticker=symbol):
        return None
    last = indicators.iloc[-1]
    raw = values.iloc[-1]
    # Shared filter uses percent extension / ATR percent, not dollar distance / ATR.
    distance = ((last.Close - last.SMA50) / last.SMA50) / (last.ATR / last.Close)
    return {
        "symbol": symbol, "listed_name": screen["listings"][symbol]["company_name"],
        "exchange": screen["listings"][symbol]["exchange"],
        "signal_date": str(prior.date()), "status": "EXTENDED_SHORT_RESEARCH_WATCH",
        "close_raw": float(raw.Close),
        "atr_pct": float(last.ATR_Pct), "extension_score": float(distance),
        "above_sma50_pct": float((last.Close / last.SMA50 - 1) * 100),
        "relative_volume_63": float(last.vol_ratio), "average_volume_63": float(last.vol_ma),
        "return_1d_pct": float((last.Close / indicators.Close.iloc[-2] - 1) * 100),
        "return_5d_pct": float((last.Close / indicators.Close.iloc[-6] - 1) * 100),
        "return_21d_pct": float((last.Close / indicators.Close.iloc[-22] - 1) * 100),
        "borrow_status": "NOT_CHECKED",
    }


def replay_prices(prices: dict, screen: dict, target: str) -> tuple[list[dict], dict]:
    if set(prices) != set(screen["universe"]):
        raise ValueError("Daily history does not account for the full frozen universe")
    candidates, failures, verified = [], {}, 0
    for symbol in screen["universe"]:
        record = prices[symbol]
        if record.get("error"):
            failures[symbol] = str(record["error"])
            continue
        try:
            row = analyze_bars(symbol, record["bars"], target, screen)
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            failures[symbol] = str(exc)[:160]
            continue
        verified += 1
        if row:
            candidates.append(row)
    if not verified:
        raise ValueError("No verified daily histories; short screen unavailable")
    candidates.sort(key=lambda r: (-r["extension_score"], -r["relative_volume_63"], r["symbol"]))
    return candidates, {"requested": len(prices), "verified": verified,
                        "unverified": len(failures), "failures": failures}


def capture(run_dir: Path, target: str, *, download=None) -> dict:
    """Freeze fresh Yahoo daily bars; no stored price database or broker access."""
    import yfinance as yf

    session = _session(target)
    now = datetime.now(timezone.utc)
    if session != now.astimezone(NY).date():
        raise ValueError("Capture must target today's NYSE session")
    run_dir.mkdir(parents=True, exist_ok=False)
    universe = capture_universe(target)
    _write(run_dir / "universe.json", universe)
    screen = screen_from_universe(universe, target)
    yf.set_tz_cache_location(str(run_dir / "yfinance-metadata"))
    download = download or yf.download
    source = {}
    symbols = screen["universe"]
    for offset in range(0, len(symbols), 75):
        batch = symbols[offset:offset + 75]
        yahoo = {s: screen["listings"][s]["yahoo_symbol"] for s in batch}
        try:
            data = download(tickers=sorted(set(yahoo.values())),
                            start=str(session - timedelta(days=400)), end=str(session),
                            auto_adjust=False, repair=True, progress=False, threads=8, timeout=15)
            for symbol in batch:
                try:
                    source[symbol] = {"bars": normalize_download(data, yahoo[symbol])}
                except (ValueError, KeyError, TypeError) as exc:
                    source[symbol] = {"error": type(exc).__name__ + ": daily source unavailable"}
        except Exception as exc:
            for symbol in batch:
                source[symbol] = {"error": type(exc).__name__ + ": daily download unavailable"}
        print(f"Daily histories accounted for: {len(source)}/{len(symbols)}", flush=True)
    _write(run_dir / "prices.json", source)
    candidates, coverage = replay_prices(source, screen, target)
    queue = {"schema": SCHEMA, "session_date": target,
             "captured_at": datetime.now(timezone.utc).isoformat(),
             "source": "YFINANCE_AUTO_ADJUST_FALSE_REPAIR_TRUE_WITH_ADJ_CLOSE",
             "screen": screen, "prices_sha256": digest(source),
             "universe_sha256": digest(universe),
             "candidates": candidates, "coverage": coverage}
    _write(run_dir / "queue.json", queue)
    return queue


def validate_queue(run_dir: Path, target: str) -> dict:
    queue = _read(run_dir / "queue.json")
    if queue.get("schema") != SCHEMA or queue.get("session_date") != target:
        raise ValueError("Short screen schema or session mismatch")
    universe = _read(run_dir / "universe.json")
    if digest(universe) != queue["universe_sha256"]:
        raise ValueError("Listing-universe source hash mismatch")
    screen = screen_from_universe(universe, target)
    if queue["screen"] != screen:
        raise ValueError("Short screen differs from the configured strategy or captured listing universe")
    captured = parse_timestamp(queue["captured_at"])
    if captured.astimezone(NY).date().isoformat() != target or captured > datetime.now(timezone.utc):
        raise ValueError("Short screen capture time is stale or in the future")
    source = _read(run_dir / "prices.json")
    if digest(source) != queue["prices_sha256"]:
        raise ValueError("Short daily source hash mismatch")
    candidates, coverage = replay_prices(source, screen, target)
    if candidates != queue["candidates"] or coverage != queue["coverage"]:
        raise ValueError("Short screen does not replay from retained daily bars")
    return queue


def _https(value: object) -> str:
    if not isinstance(value, str) or urlparse(value).scheme != "https" or not urlparse(value).hostname:
        raise ValueError("News evidence requires an HTTPS URL")
    return value


def _text(record: dict, key: str, minimum=1) -> str:
    value = record.get(key)
    if not isinstance(value, str) or len(value.strip()) < minimum:
        raise ValueError(f"Short review missing {key}")
    return value.strip()


def validate_reviews(queue: dict, reviews: list[dict], sealed_at: str) -> None:
    if not isinstance(reviews, list):
        raise ValueError("Short reviews must be a list")
    symbols = [r.get("symbol") for r in reviews]
    expected = {c["symbol"] for c in queue["candidates"]}
    if len(symbols) != len(set(symbols)) or set(symbols) != expected:
        raise ValueError("Every short candidate needs exactly one completed review")
    start, end = parse_timestamp(queue["captured_at"]), parse_timestamp(sealed_at)
    if end < start or end > datetime.now(timezone.utc):
        raise ValueError("Invalid short review completion time")
    for review in reviews:
        if review.get("research_complete") is not True or review.get("status") not in {"CONTEXT_VERIFIED", "NO_VERIFIED_NEWS"}:
            raise ValueError("Unfinished short research cannot be emailed")
        for key in ("company_name", "news_context", "squeeze_risk", "reason"):
            _text(review, key, 10 if key != "company_name" else 2)
        reviewed = parse_timestamp(_text(review, "reviewed_at"))
        if not start <= reviewed <= end:
            raise ValueError("Short review timestamp outside capture/review window")
        searches = review.get("searches", [])
        if not searches or len(searches) > 4:
            raise ValueError("Short research needs one to four inspected searches")
        for search in searches:
            _text(search, "query")
            _text(search, "observation_ref")
            host = urlparse(_https(search.get("url"))).hostname
            if host not in {"google.com", "www.google.com"}:
                raise ValueError("Short research requires observed Google searches")
            if search.get("outcome") not in {"RESULTS_READ", "NO_RELEVANT_RESULTS"}:
                raise ValueError("Blocked searches are unfinished research")
            if not start <= parse_timestamp(_text(search, "searched_at")) <= reviewed:
                raise ValueError("Search timestamp outside review window")
        sources = review.get("sources", [])
        if len(sources) > 4:
            raise ValueError("Too many short research sources")
        if review["status"] == "CONTEXT_VERIFIED" and not sources:
            raise ValueError("Verified news context needs an opened source")
        if review["status"] == "NO_VERIFIED_NEWS":
            if len({s["query"].casefold() for s in searches}) < 2 or not {
                "COMPANY_NEWS", "PRIMARY_ANNOUNCEMENT"
            }.issubset({s.get("purpose") for s in searches}):
                raise ValueError("No-verified-news requires two distinct purposeful searches")
        for source in sources:
            _https(source.get("url"))
            for key, minimum in (("title", 3), ("content", 80), ("authority_basis", 10), ("observation_ref", 1)):
                _text(source, key, minimum)
            if source.get("capture_kind") != "ARTICLE_BODY":
                raise ValueError("A search snippet is not opened-source news context")
            opened = parse_timestamp(_text(source, "opened_at"))
            if not start <= opened <= reviewed:
                raise ValueError("Opened source outside review window")
            publication = _text(source, "published_at")
            # Context can predate the latest overnight window; preserve its date.
            if len(publication) == 10:
                if date.fromisoformat(publication) > opened.astimezone(NY).date():
                    raise ValueError("Future source publication date")
            elif parse_timestamp(publication) > opened:
                raise ValueError("Future source publication timestamp")


def render_section(packet: dict) -> tuple[str, str]:
    queue = packet["queue"]
    coverage = queue["coverage"]
    settings = queue["screen"]["settings"]
    escape = lambda value: html.escape(str(value), quote=True)
    title = "Parabolic-short watchlist — ATR Extended Gap Up"
    intro = (f"{coverage['verified']}/{coverage['requested']} daily histories verified; "
             f"{coverage['unverified']} unverified and excluded. {queue['screen']['universe_label']}. "
             f"Extension score >{settings['dist_min']:g}; volume >{settings['vol_thresh']:g}× 63-session average. "
             "The extension score is percentage distance above SMA50 divided by ATR%. "
             "Potential short research candidates based on completed daily bars. Borrow and fees are not checked.")
    body = [f'<section style="margin-top:28px;border-top:2px solid #334155;padding-top:16px;font:15px/1.55 Segoe UI,sans-serif;color:#15202b"><h2 style="font-size:24px;margin:0 0 12px">{title}</h2>',
            f'<p style="margin:8px 0">{escape(intro)}</p>']
    plain = [title, intro]
    reviews = {r["symbol"]: r for r in packet["reviews"]}
    if not queue["candidates"]:
        text = "No setups among the verified histories. Unverified histories are not a negative screen result."
        body.append(f"<p>{text}</p>")
        plain.append(text)
    for row in queue["candidates"]:
        review = reviews[row["symbol"]]
        heading = f"{row['symbol']} — {review['company_name']}"
        metrics = (f"As of {row['signal_date']} | prior close ${row['close_raw']:.2f} | "
                   f"1d {row['return_1d_pct']:+.1f}% | 5d {row['return_5d_pct']:+.1f}% | 21d {row['return_21d_pct']:+.1f}% | "
                   f"above SMA50 {row['above_sma50_pct']:+.1f}% | extension {row['extension_score']:.2f} | "
                   f"volume {row['relative_volume_63']:.2f}× | ATR {row['atr_pct']:.2f}%")
        context = review["news_context"]
        if review["status"] == "NO_VERIFIED_NEWS":
            context = "No news verified after completed searches. " + context
        body.extend(['<article style="margin-top:16px;padding:16px;background:#fff;border:1px solid #dbe3ea;border-radius:10px">',
                     f'<h3 style="font-size:20px;text-transform:none;letter-spacing:normal;margin:0 0 10px">{escape(heading)}</h3>',
                     f'<p style="margin:8px 0">{escape(metrics)}</p>',
                     f'<p style="margin:8px 0"><b>News context:</b> {escape(context)}</p>',
                     f'<p style="margin:8px 0"><b>Squeeze risk:</b> {escape(review["squeeze_risk"])}</p>'])
        plain.extend([heading, metrics, "News context: " + context, "Squeeze risk: " + review["squeeze_risk"]])
        for source in review.get("sources", []):
            body.append(f'<p><a href="{escape(source["url"])}">{escape(source["title"])}</a> — {escape(source["published_at"])}</p>')
            plain.append(source["title"] + " — " + source["published_at"] + " — " + source["url"])
        body.append("</article>")
    body.append("</section>")
    return "\n".join(body), "\n\n".join(plain)


def seal(run_dir: Path, notes: Path) -> Path:
    queue = _read(run_dir / "queue.json")
    queue = validate_queue(run_dir, queue["session_date"])
    reviews = _read(notes)
    sealed_at = datetime.now(timezone.utc).isoformat()
    validate_reviews(queue, reviews, sealed_at)
    packet = {"schema": SCHEMA, "queue": queue, "reviews": reviews, "sealed_at": sealed_at}
    _write(run_dir / "watchlist.json", packet)
    section, plain = render_section(packet)
    with (run_dir / "watchlist.html").open("x", encoding="utf-8") as handle:
        handle.write('<!doctype html><html><head><meta charset="utf-8"></head><body>' + section + '</body></html>')
    with (run_dir / "watchlist.md").open("x", encoding="utf-8") as handle:
        handle.write(plain)
    return run_dir / "watchlist.json"


def load_watchlist(path: Path, target: str) -> dict:
    packet = _read(path)
    if packet.get("schema") != SCHEMA:
        raise ValueError("Short watchlist schema mismatch")
    queue = validate_queue(path.parent, target)
    if packet["queue"] != queue:
        raise ValueError("Short watchlist queue was changed")
    validate_reviews(queue, packet["reviews"], packet["sealed_at"])
    section, plain = render_section(packet)
    expected_html = '<!doctype html><html><head><meta charset="utf-8"></head><body>' + section + '</body></html>'
    if (path.parent / "watchlist.html").read_text(encoding="utf-8") != expected_html or (path.parent / "watchlist.md").read_text(encoding="utf-8") != plain:
        raise ValueError("Short watchlist report differs from reviewed evidence")
    return packet
