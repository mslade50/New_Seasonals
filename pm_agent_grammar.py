"""PM Weekly grammar: the only gate between the agent's brief and delivery.

    validate_brief(payload, ctx) -> {"errors": [...], "warnings": [...], "forecasts": [...]}

ctx: {"asof", "week", "target_week", "climatology", "checks_dir", "vix_last"}

What it enforces (docs/claude_ref/pm_agent.md):
  * Both claim types, exactly once, every week (spy_week_return, vix_week_change).
  * Coherent numbers: 0.03 <= p_up <= 0.97, q10 < q90, VIX q10 above -VIX.
  * Evidence on disk: 00_surface_map.md and every cited script inside today's
    checks folder; a cited script that names a book object or the Risk Agent's
    working files is refused.
  * Small N stays near the base rate: evidence n < 30 forces |p_up - climatology|
    <= 0.05.
  * Propose, never change: rule-change and trading-instruction language is
    refused anywhere in the prose. ASCII only (no emoji, no em dashes).

The resolution contract (anchor close, resolves_on, horizon) and the
climatology come from ctx, never from the payload.

Agent-product module: the book and the Risk Agent must not import it.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import pm_agent_universe as U

MODES = ("brief", "stand_down")
P_MIN, P_MAX = 0.03, 0.97
SMALL_N = 30
SMALL_N_MAX_TILT = 0.05
SPY_Q_BOUND = 25.0          # percent, one week
MAX_RECAP, MIN_RECAP = 8, 3
MAX_WATCH, MAX_QUESTIONS = 5, 3

# Phrases that turn a brief into a rule change or a trade ticket. The PM proposes
# questions; changes go through a written prereg (CLAUDE.md "Pre-registration").
BANNED_PHRASES: tuple[str, ...] = (
    "increase size", "increase sizing", "reduce size", "reduce sizing", "cut size",
    "size up", "size down", "raise the cap", "lower the cap", "add a cap", "remove the cap",
    "change the dial", "retune", "re-tune", "tighten the band", "loosen the band",
    "turn off", "turn on the", "disable the", "switch off", "kill the strategy",
    "you should buy", "you should sell", "we should buy", "we should sell",
    "should go long", "should go short", "buy spy", "sell spy", "short spy",
    "add exposure", "cut exposure", "reduce exposure", "increase exposure",
    "hedge the book", "de-risk the book", "derisk the book",
)
_WORD = re.compile(r"[a-z0-9\-]+")


def _norm(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def lint_text(text: str, where: str, errors: list) -> None:
    if not isinstance(text, str):
        return
    bad = sorted({c for c in text if ord(c) > 126 or (ord(c) < 32 and c not in "\n\t")})
    if bad:
        errors.append(f"{where}: non-ASCII or control characters {''.join(bad)!r} (no emoji, no em dashes)")
    norm = f" {_norm(text)} "
    for p in BANNED_PHRASES:
        if f" {_norm(p)} " in norm:
            errors.append(f"{where}: '{p}' is a rule change or trade instruction; "
                          "the PM states observations and questions only")


def _text(obj: dict, key: str, errors: list, where: str, lo: int = 1, hi: int = 1200) -> str:
    v = obj.get(key) if isinstance(obj, dict) else None
    if not isinstance(v, str) or not (lo <= len(v.strip()) <= hi):
        errors.append(f"{where}.{key}: text of {lo}-{hi} characters required")
        return ""
    lint_text(v, f"{where}.{key}", errors)
    return v


def _num(obj: dict, key: str, errors: list, where: str, lo=None, hi=None) -> float | None:
    v = obj.get(key) if isinstance(obj, dict) else None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        errors.append(f"{where}.{key}: number required")
        return None
    v = float(v)
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        errors.append(f"{where}.{key}: {v} outside [{lo}, {hi}]")
    return v


def script_inside(path: str | None, checks_dir: Path | None) -> Path | None:
    """Resolved path when `path` is an existing file inside checks_dir."""
    if not path or checks_dir is None:
        return None
    root = Path(checks_dir).resolve()
    p = Path(path)
    p = (p if p.is_absolute() else root / p).resolve()
    try:
        p.relative_to(root)
    except ValueError:
        return None
    return p if p.is_file() else None


def forbidden_tokens(source: str) -> list[str]:
    low = source.lower().replace("\\", "/")
    return [t for t in U.FORBIDDEN_SOURCE_TOKENS if t.lower() in low]


def _forecast(f: dict, i: int, ctx: dict, errors: list, warnings: list) -> dict | None:
    where = f"forecasts[{i}]"
    if not isinstance(f, dict):
        errors.append(f"{where}: object required")
        return None
    claim = f.get("claim_type")
    if claim not in U.CLAIMS:
        errors.append(f"{where}.claim_type: one of {sorted(U.CLAIMS)} required")
        return None
    spec = U.CLAIMS[claim]
    p_key, q10_key, q90_key = spec["fields"]
    p = _num(f, p_key, errors, where, P_MIN, P_MAX)
    q10 = _num(f, q10_key, errors, where)
    q90 = _num(f, q90_key, errors, where)
    if q10 is not None and q90 is not None and not q10 < q90:
        errors.append(f"{where}: {q10_key} must be below {q90_key}")
    if claim == "spy_week_return":
        for k, v in ((q10_key, q10), (q90_key, q90)):
            if v is not None and abs(v) > SPY_Q_BOUND:
                errors.append(f"{where}.{k}: |{v}| > {SPY_Q_BOUND}% is not a weekly quantile")
    else:
        vix = ctx.get("vix_last")
        if q10 is not None and vix is not None and q10 <= -float(vix):
            errors.append(f"{where}.{q10_key}: VIX cannot fall {q10} points from {vix}")
    _text(f, "why", errors, where, 40, 900)
    _text(f, "change_my_mind", errors, where, 10, 400)
    _text(f, "basis", errors, where, 3, 300)
    ev = f.get("evidence")
    n = None
    if not isinstance(ev, dict):
        errors.append(f"{where}.evidence: {{summary, n, script}} required")
    else:
        _text(ev, "summary", errors, f"{where}.evidence", 10, 600)
        n = ev.get("n")
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            errors.append(f"{where}.evidence.n: non-negative integer required")
            n = None
        sp = script_inside(ev.get("script"), ctx.get("checks_dir"))
        if sp is None:
            errors.append(f"{where}.evidence.script: must be an existing file inside "
                          f"{ctx.get('checks_dir')}")
        else:
            try:
                bad = forbidden_tokens(sp.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                bad = []
            if bad:
                errors.append(f"{where}.evidence.script reads outside the market-only boundary: {bad}")
    clim = (ctx.get("climatology") or {}).get(claim) or {}
    cp = clim.get("p_up")
    if p is not None and cp is not None:
        tilt = abs(p - float(cp))
        if n is not None and n < SMALL_N and tilt > SMALL_N_MAX_TILT + 1e-9:
            errors.append(f"{where}: evidence n={n} < {SMALL_N} cannot move p_up more than "
                          f"{SMALL_N_MAX_TILT} from the base rate {cp} (moved {tilt:.3f})")
        elif tilt > 0.20:
            warnings.append(f"{where}: p_up {p} is {tilt:.2f} from the base rate {cp}")
    if None in (p, q10, q90):
        return None
    tw = ctx.get("target_week") or {}
    return {"claim_type": claim, "symbol": spec["symbol"], "unit": spec["unit"],
            "p_up": p, "q10": q10, "q90": q90,
            "resolves_on": tw.get("resolves_on"), "horizon_td": tw.get("horizon_td"),
            "climatology": {k: clim.get(k) for k in ("p_up", "q10", "q90", "n", "n_independent")},
            "evidence_n": n, "evidence_script": (ev or {}).get("script") if isinstance(ev, dict) else None}


def validate_brief(payload: Any, ctx: dict) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    out = {"errors": errors, "warnings": warnings, "forecasts": []}
    if not isinstance(payload, dict):
        errors.append("payload: JSON object required")
        return out
    if payload.get("schema_version") != U.SCHEMA_VERSION:
        errors.append(f"schema_version: {U.SCHEMA_VERSION!r} required")
    if str(payload.get("asof")) != str(ctx.get("asof")):
        errors.append(f"asof {payload.get('asof')!r} != state asof {ctx.get('asof')!r}")
    mode = payload.get("mode")
    if mode not in MODES:
        errors.append(f"mode: one of {MODES} required")
        return out
    checks_dir = ctx.get("checks_dir")
    if mode == "stand_down":
        _text(payload, "reason", errors, "stand_down", 10, 600)
        return out

    if checks_dir is None or not (Path(checks_dir) / "00_surface_map.md").is_file():
        errors.append(f"00_surface_map.md missing from {checks_dir} (survey before forecasting)")
    _text(payload, "headline", errors, "headline", 10, 160)

    recap = payload.get("recap")
    if not isinstance(recap, list) or not (MIN_RECAP <= len(recap) <= MAX_RECAP):
        errors.append(f"recap: {MIN_RECAP}-{MAX_RECAP} items required")
    else:
        for i, r in enumerate(recap):
            _text(r, "topic", errors, f"recap[{i}]", 2, 60)
            _text(r, "text", errors, f"recap[{i}]", 20, 700)

    nw = payload.get("next_week")
    if not isinstance(nw, dict):
        errors.append("next_week: {calendar, base_case, alt_case} required")
    else:
        cal = nw.get("calendar")
        if not isinstance(cal, list):
            errors.append("next_week.calendar: list required (may be empty)")
        else:
            for i, c in enumerate(cal):
                if not isinstance(c, str) or not c.strip():
                    errors.append(f"next_week.calendar[{i}]: text required")
                else:
                    lint_text(c, f"next_week.calendar[{i}]", errors)
        _text(nw, "base_case", errors, "next_week", 40, 900)
        _text(nw, "alt_case", errors, "next_week", 40, 900)

    fcs = payload.get("forecasts")
    if not isinstance(fcs, list):
        errors.append("forecasts: list required")
        fcs = []
    seen = [f.get("claim_type") for f in fcs if isinstance(f, dict)]
    for claim in U.CLAIMS:
        if seen.count(claim) != 1:
            errors.append(f"forecasts: exactly one {claim} required (found {seen.count(claim)})")
    extra = [c for c in seen if c not in U.CLAIMS]
    if extra:
        errors.append(f"forecasts: unknown claim types {extra}")
    for i, f in enumerate(fcs):
        rec = _forecast(f, i, ctx, errors, warnings)
        if rec is not None:
            out["forecasts"].append(rec)

    for key, cap, fields in (("watch", MAX_WATCH, ("item", "trigger")),
                             ("questions", MAX_QUESTIONS, ("question", "why_it_matters"))):
        items = payload.get(key, [])
        if not isinstance(items, list) or len(items) > cap:
            errors.append(f"{key}: list of at most {cap} required")
            continue
        for i, it in enumerate(items):
            for fld in fields:
                _text(it, fld, errors, f"{key}[{i}]", 5, 500)
    gaps = payload.get("data_gaps", [])
    if not isinstance(gaps, list):
        errors.append("data_gaps: list required")
    else:
        for i, g in enumerate(gaps):
            lint_text(g if isinstance(g, str) else "", f"data_gaps[{i}]", errors)
    if errors:
        out["forecasts"] = []
    return out
