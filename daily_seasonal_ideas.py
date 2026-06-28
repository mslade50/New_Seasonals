"""
daily_seasonal_ideas.py - daily discretionary trade-idea engine.

Mines the repo's existing data for UNDERUTILIZED statistical/seasonal
situations and presents them as discretionary ideas - deliberately separate
from the live systematic book (it negative-filters against check_signal so it
never re-surfaces a name already being traded).

Pipeline
--------
1. resolve asof (latest trading day with seasonal + price data)
2. load regime context (rd2_environment.json + exposure_state.json), with a
   staleness guard so a stale regime snapshot degrades rather than mis-conditions
3. run each detector channel (scripts/seasonal_detectors.py)
4. negative-filter ticker-level candidates against the live book
5. multiple-comparisons control: Benjamini-Hochberg FDR across the statistical
   (p-valued) candidates; top-N cap per channel
6. emit data/daily_seasonal_ideas.md (committed-markdown digest) + a JSON sidecar

The Claude distillation layer (radar-style variant-perception filtering) and
GHA automation are layered on top of this deterministic engine separately.
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import datetime as _dt

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import scripts.seasonal_edge as se  # noqa: E402

try:
    from scripts import seasonal_detectors as sd  # noqa: E402
except Exception as _e:  # detectors module may not exist yet during bring-up
    sd = None
    print(f"[WARN] seasonal_detectors not importable yet: {_e}")

DATA_DIR = os.path.join(REPO_ROOT, "data")
OUT_MD = os.path.join(DATA_DIR, "daily_seasonal_ideas.md")
OUT_JSON = os.path.join(DATA_DIR, "daily_seasonal_ideas.json")
ENV_JSON = os.path.join(DATA_DIR, "rd2_environment.json")
EXPOSURE_JSON = os.path.join(DATA_DIR, "exposure_state.json")

# Detector channels, in display order. (name, attr, pretty_channel_title)
CHANNELS = [
    ("seasonal", "detect_seasonal", "Seasonal Setups"),
    ("near_miss", "detect_near_miss", "Near-Miss Strategy Signals"),
    ("regime_sleeve", "detect_regime_sleeve", "Regime / Sleeve Tilt"),
    ("cross_asset", "detect_cross_asset", "Cross-Asset Seasonality"),
    ("sentiment", "detect_sentiment", "Sentiment / Positioning"),
    ("analyst", "detect_analyst", "Analyst Rating Revisions"),
]

# Raw channel string each detector stamps on its candidates -> display title.
CHANNEL_TITLES = {
    "detect_seasonal": "Equity Seasonal Tickets",
    "near_miss": "Near-Miss Strategy Signals",
    "detect_near_miss": "Near-Miss Strategy Signals",
    "regime_sleeve": "Regime / Sleeve Tilt",
    "detect_cross_asset": "Macro / Cross-Asset Tickets",
    "sentiment": "Sentiment / Positioning",
    "analyst_grades": "Analyst Rating Revisions",
}

# Scope every ticker-level idea to the curated watch universe (megacap + macro).
# Seasonal/cross-asset are natively scoped; this catches analyst / near-miss.
IDEA_UNIVERSE = set(t.upper() for t in se.IDEA_UNIVERSE)
CHANNEL_ORDER = ["detect_seasonal", "near_miss", "detect_near_miss", "regime_sleeve",
                 "detect_cross_asset", "sentiment", "analyst_grades"]

METHODOLOGY = (
    "Methodology: binomial p tests the realized day-of-year count (k of n years closed the claimed way vs 50%) "
    "for THIS calendar window - a selected, descriptive stat (post-selection optimistic), not an out-of-sample "
    "guarantee. FDR badge = Benjamini-Hochberg multiplicity control across the day's statistical candidates "
    "(strict; 'borderline' is common and expected). Conviction (A/B/C) is driven by the realized cycle + all-years "
    "counts and magnitude. Near-miss is negative-filtered against the live book so it never duplicates a systematic "
    "signal. 'midterm' stats are re-derived from raw prices filtered to year%4==2, since the blended seasonal rank "
    "collapses the cycle and cannot express it."
)

FDR_ALPHA = 0.10
TOPN_PER_CHANNEL = 12
STALE_TRADING_DAYS = 5


# -----------------------------------------------------------------------------
def resolve_asof(override: str | None = None) -> pd.Timestamp:
    """Latest date that has BOTH a seasonal-rank row and a price bar.

    The seasonal parquet carries forward rows through year-end, so we clamp to
    the last date <= today that also has price coverage (via ^GSPC)."""
    if override:
        return pd.Timestamp(override).normalize()
    today = pd.Timestamp(_dt.date.today())
    spx = se.load_one_price("^GSPC")
    if spx is not None and not spx.empty:
        last_px = spx.index.max()
        return min(today, last_px).normalize()
    return today.normalize()


def _read_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] could not read {os.path.basename(path)}: {e}")
        return None


def load_regime(asof: pd.Timestamp) -> dict:
    """Regime context + staleness guard. Returns {label, summary, fragility,
    active_signals, stale, stale_notes, exposure_mult}."""
    env = _read_json(ENV_JSON) or {}
    exp = _read_json(EXPOSURE_JSON) or {}
    out = {"label": None, "summary": None, "fragility": None,
           "active_signals": [], "stale": False, "stale_notes": [],
           "exposure_mult": exp.get("mult")}

    env_date = pd.Timestamp(env["date"]).normalize() if env.get("date") else None
    if env_date is not None:
        gap = int(np.busday_count(env_date.date(), asof.date()))
        if gap > STALE_TRADING_DAYS:
            out["stale"] = True
            out["stale_notes"].append(
                f"regime snapshot {env_date.date()} is {gap} trading days stale - "
                f"regime gate dropped")
            return out  # degrade: do not condition on a stale snapshot

    pc = env.get("price_ctx", {})
    out["label"] = pc.get("regime_label")
    h = env.get("h_scores", {})
    out["fragility"] = h.get("63d")
    active = [k for k, v in (env.get("signals") or {}).items() if v.get("on")]
    out["active_signals"] = active

    bits = []
    if out["label"]:
        bits.append(out["label"])
    if out["fragility"] is not None:
        band = "robust" if out["fragility"] < 33 else ("neutral" if out["fragility"] < 66 else "fragile")
        bits.append(f"fragility 63d {out['fragility']:.0f} ({band})")
    if pc.get("days_since_5pct") is not None:
        bits.append(f"{pc['days_since_5pct']}d since 5% pullback")
    bits.append(f"{len(active)} fragility signals on" + (f" ({', '.join(active)})" if active else ""))
    if exp.get("mult") is not None:
        bits.append(f"core exposure mult {exp['mult']}x")
    out["summary"] = "; ".join(bits)
    return out


# -----------------------------------------------------------------------------
def run_detectors(asof: pd.Timestamp, ctx: dict) -> list[dict]:
    if sd is None:
        print("[ERROR] seasonal_detectors not available - no candidates")
        return []
    out: list[dict] = []
    for key, attr, _title in CHANNELS:
        fn = getattr(sd, attr, None)
        if fn is None:
            print(f"[skip] {key}: {attr} not defined")
            continue
        try:
            cands = fn(asof, ctx) or []
            print(f"[{key}] {len(cands)} candidates")
            out.extend(cands)
        except Exception as e:
            import traceback
            print(f"[WARN] {key} failed: {e}")
            traceback.print_exc()
    return out


def negative_filter(candidates: list[dict], asof: pd.Timestamp, ctx: dict) -> list[dict]:
    """Drop ticker-level candidates that already fire a live STRATEGY_BOOK
    strategy (so the discretionary feed never duplicates the live book).
    'context' candidates pass through untouched."""
    live: set = set()
    if sd is not None and hasattr(sd, "live_book_tickers"):
        try:
            live = set(t.upper() for t in (sd.live_book_tickers(asof, ctx) or set()))
        except Exception as e:
            print(f"[WARN] live_book_tickers failed ({e}) - negative filter skipped")
    if not live:
        return candidates
    kept = []
    dropped = 0
    for c in candidates:
        if c["direction"] != "context" and c["ticker"].upper() in live:
            dropped += 1
            continue
        kept.append(c)
    print(f"[negative-filter] dropped {dropped} live-book duplicates ({len(live)} live names)")
    return kept


def universe_filter(candidates: list[dict]) -> list[dict]:
    """Keep only ticker-level ideas inside the curated watch universe (megacap +
    macro). 'context'/'BOOK' rows (market-level observations) pass through."""
    kept, dropped = [], 0
    for c in candidates:
        if c["direction"] == "context" or c["ticker"] == "BOOK" or c["ticker"].upper() in IDEA_UNIVERSE:
            kept.append(c)
        else:
            dropped += 1
    print(f"[universe-filter] kept {len(kept)}, dropped {dropped} off-universe ideas")
    return kept


def apply_fdr(candidates: list[dict], alpha: float = FDR_ALPHA) -> list[dict]:
    """Benjamini-Hochberg across the day's p-valued candidates (the binomial
    realized-count p). This is a multiplicity BADGE + a mild sort nudge, NOT a
    conviction override: the detectors already set conviction off hard realized
    gates, and these are discretionary ideas, so FDR informs rather than censors.
    Survivors float up and are tagged 'robust'; non-survivors are tagged
    'borderline' and nudged down."""
    idx = [i for i, c in enumerate(candidates) if c.get("p_value") is not None]
    if not idx:
        return candidates
    pvals = [candidates[i]["p_value"] for i in idx]
    reject, crit = se.benjamini_hochberg(pvals, alpha=alpha)
    n_surv = int(np.sum(reject))
    print(f"[FDR] {n_surv}/{len(idx)} statistical candidates robust under BH(alpha={alpha}) "
          f"(crit p={crit:.4f})")
    for j, i in enumerate(idx):
        c = candidates[i]
        if reject[j]:
            c["evidence"] = {**c.get("evidence", {}), "FDR": "robust"}
            c["sort_key"] = c.get("sort_key", 0.0) * 1.15
        else:
            c["evidence"] = {**c.get("evidence", {}), "FDR": "borderline"}
            c["sort_key"] = c.get("sort_key", 0.0) * 0.85
    return candidates


def cap_per_channel(candidates: list[dict], n: int = TOPN_PER_CHANNEL) -> list[dict]:
    by_ch: dict[str, list] = {}
    for c in candidates:
        by_ch.setdefault(c["channel"], []).append(c)
    out = []
    for ch, rows in by_ch.items():
        rows = sorted(rows, key=lambda x: -x.get("sort_key", 0.0))
        out.extend(rows[:n])
    return out


# -----------------------------------------------------------------------------
def build(asof: pd.Timestamp, grades=("A",)) -> tuple[str, dict]:
    regime = load_regime(asof)
    ctx = {"asof": asof, "regime": regime, "min_rr": 2.0, "universe": list(IDEA_UNIVERSE)}
    print(f"[regime] {regime.get('summary')}")

    candidates = run_detectors(asof, ctx)
    candidates = universe_filter(candidates)
    candidates = negative_filter(candidates, asof, ctx)
    candidates = apply_fdr(candidates)
    if grades:
        kept = [c for c in candidates if c.get("conviction") in grades]
        print(f"[grade-filter] showing {'+'.join(grades)} only: {len(kept)}/{len(candidates)} kept")
        candidates = kept
    candidates = cap_per_channel(candidates)

    n_ideas = sum(1 for c in candidates if c["direction"] != "context")
    n_channels = len({c["channel"] for c in candidates})

    # canonical ordering + clean section titles (detectors stamp raw channel keys)
    candidates.sort(key=lambda c: CHANNEL_ORDER.index(c["channel"]) if c["channel"] in CHANNEL_ORDER else 99)
    for c in candidates:
        c["channel"] = CHANNEL_TITLES.get(c["channel"], c["channel"])

    grade_lbl = ("+".join(grades) + "-grade") if grades else "all"
    summary = (f"**{n_ideas} {grade_lbl} setups** flagged across {n_channels} channels. "
               f"These are NOT live-book signals - the systematic scanner trades those separately.")

    meta = {
        "asof": str(asof.date()),
        "regime": regime.get("summary"),
        "summary": summary,
        "stale_notes": regime.get("stale_notes", []),
        "footer": METHODOLOGY,
    }
    md = se.render_markdown(candidates, meta)
    payload = {"meta": meta, "candidates": candidates}
    return md, payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=None, help="override as-of date (YYYY-MM-DD)")
    ap.add_argument("--stdout", action="store_true", help="print markdown to stdout instead of writing")
    ap.add_argument("--grades", default="A", help="conviction grades to show, e.g. A / AB / ABC; 'all' for no filter")
    args = ap.parse_args()

    grades = None if args.grades.lower() == "all" else tuple(args.grades.upper())
    asof = resolve_asof(args.asof)
    print(f"=== daily_seasonal_ideas asof {asof.date()} ===")
    md, payload = build(asof, grades=grades)

    if args.stdout:
        print("\n" + md)
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(md)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n[written] {OUT_MD}")
    print(f"[written] {OUT_JSON}")

    # Forward log: append today's tickets to the append-only track-record ledger
    # (best-effort — a logging failure must never break the digest).
    try:
        from scripts.seasonal_ideas_ledger import append_emitted
        append_emitted(payload, logged_at=str(asof.date()))
    except Exception as _e:
        print(f"[ledger] hook skipped ({_e})")

    # Order staging: write tradeable tickets to the Seasonal / sznl_nostage Sheets
    # tabs for order_staging.py to read (best-effort — save_seasonal_tabs no-ops
    # without Sheets creds, and a failure must never break the digest).
    try:
        from seasonal_order_staging import build_seasonal_rows, save_seasonal_tabs
        _seasonal, _nostage = build_seasonal_rows(payload)
        save_seasonal_tabs(_seasonal, _nostage)
    except Exception as _e:
        print(f"[seasonal-staging] hook skipped ({_e})")


if __name__ == "__main__":
    main()
