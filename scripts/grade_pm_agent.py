"""Resolve PM Weekly forecasts and score them against climatology.

    python scripts/grade_pm_agent.py [--no-sync] [--no-push] [--dry-run]

For every journaled forecast without a resolution whose `resolves_on` bar is
in master_prices: SPY is close-to-close percent from the stored anchor close,
VIX is the change in points. Both on RAW closes (the anchor was stored raw at
publish). A forecast whose resolution bar is still missing 7 days after
resolves_on is resolved `void` (data gap), never guessed.

Scoreboard (PM_AGENT_HOME/scoreboard.json, mirrored to R2 pm_agent/):
per claim type, Brier of p_up and the same Brier for the climatology p_up
stored at lock time (skill = 1 - brier / brier_clim), q10/q90 coverage, and
the q10/q90 pinball loss against the climatology quantiles. Forecasts
published late (`scored: false`) are resolved but excluded from the scores.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pm_agent_data as pad  # noqa: E402
import pm_agent_journal as J  # noqa: E402
import pm_agent_universe as U  # noqa: E402
from research_io import write_json  # noqa: E402

VOID_AFTER_DAYS = 7


def load_closes(cache_dir: Path | None = None) -> dict[str, pd.Series]:
    p = pad.local_path("master_prices.parquet", cache_dir)
    syms = sorted({c["symbol"] for c in U.CLAIMS.values()})
    df = pd.read_parquet(p, filters=[("ticker", "in", syms)], columns=["ticker", "date", "Close"])
    df["date"] = pd.to_datetime(df["date"])
    return {t: g.set_index("date")["Close"].astype("float64").sort_index()
            for t, g in df.groupby("ticker")}


def resolve(records: list[dict], closes: dict[str, pd.Series], today: dt.date) -> list[dict]:
    done = {r.get("forecast_id") for r in records if r.get("kind") == "resolution"}
    out = []
    for f in records:
        if f.get("kind") != "forecast" or f.get("forecast_id") in done:
            continue
        s = closes.get(f.get("symbol"))
        ro = pd.Timestamp(f["resolves_on"])
        base = {"kind": "resolution", "forecast_id": f["forecast_id"], "claim_type": f.get("claim_type"),
                "asof": f.get("asof"), "resolves_on": f.get("resolves_on")}
        if s is not None and ro in s.index:
            v1 = float(s.loc[ro])
            a = float(f["anchor_value"])
            value = (v1 / a - 1) * 100 if f.get("unit") == "pct" else v1 - a
            out.append({**base, "status": "resolved", "close": v1, "value": round(value, 4),
                        "up": value > 0, "below_q10": value < f["q10"], "above_q90": value > f["q90"]})
        elif (today - ro.date()).days > VOID_AFTER_DAYS:
            out.append({**base, "status": "void",
                        "reason": f"no {f.get('symbol')} bar on {f.get('resolves_on')} after {VOID_AFTER_DAYS} days"})
    return out


def _pinball(y: float, q: float, tau: float) -> float:
    return max(tau * (y - q), (tau - 1) * (y - q))


def _stats(rows: list[tuple[dict, dict]]) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0}
    br = bc = pin = pinc = 0.0
    below = above = 0
    n_clim = 0
    for f, r in rows:
        y, up = r["value"], 1.0 if r["up"] else 0.0
        br += (f["p_up"] - up) ** 2
        pin += _pinball(y, f["q10"], 0.10) + _pinball(y, f["q90"], 0.90)
        below += r["below_q10"]
        above += r["above_q90"]
        c = f.get("climatology") or {}
        if None not in (c.get("p_up"), c.get("q10"), c.get("q90")):
            n_clim += 1
            bc += (c["p_up"] - up) ** 2
            pinc += _pinball(y, c["q10"], 0.10) + _pinball(y, c["q90"], 0.90)
    out = {"n": n, "brier": br / n, "mean_p_up": sum(f["p_up"] for f, _ in rows) / n,
           "base_rate_up": sum(1.0 for _, r in rows if r["up"]) / n,
           "share_below_q10": below / n, "share_above_q90": above / n,
           "inside_q10_q90": 1 - (below + above) / n, "pinball": pin / n,
           "brier_clim": None, "brier_skill": None, "pinball_clim": None, "pinball_skill": None}
    if n_clim == n:
        out["brier_clim"] = bc / n
        out["pinball_clim"] = pinc / n
        out["brier_skill"] = 1 - out["brier"] / out["brier_clim"] if out["brier_clim"] else None
        out["pinball_skill"] = 1 - out["pinball"] / out["pinball_clim"] if out["pinball_clim"] else None
    return out


def scoreboard(records: list[dict], asof: str) -> dict:
    res = {r["forecast_id"]: r for r in records if r.get("kind") == "resolution"}
    fcs = [r for r in records if r.get("kind") == "forecast"]
    by_claim: dict[str, list] = {c: [] for c in U.CLAIMS}
    by_model: dict[str, dict[str, list]] = {}
    n_void = n_unscored = n_open = 0
    for f in fcs:
        r = res.get(f["forecast_id"])
        if r is None:
            n_open += 1
            continue
        if r.get("status") != "resolved":
            n_void += 1
            continue
        if not f.get("scored", True):
            n_unscored += 1
            continue
        by_claim.setdefault(f["claim_type"], []).append((f, r))
        by_model.setdefault(f.get("model") or "unknown", {}).setdefault(f["claim_type"], []).append((f, r))
    claims = {c: _stats(rows) for c, rows in by_claim.items()}
    recent = []
    for f in fcs[-8:]:
        r = res.get(f["forecast_id"]) or {}
        recent.append({"asof": f.get("asof"), "claim_type": f.get("claim_type"), "p_up": f.get("p_up"),
                       "q10": f.get("q10"), "q90": f.get("q90"), "resolves_on": f.get("resolves_on"),
                       "status": r.get("status", "open"), "value": r.get("value")})
    return {"schema_version": "pm_agent_scoreboard.v1", "asof": asof,
            "built_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "claims": claims, "by_model": {m: {c: _stats(v) for c, v in d.items()} for m, d in by_model.items()},
            "n_open": n_open, "n_void": n_void, "n_unscored": n_unscored, "recent": recent,
            "headline": {c: {k: claims[c].get(k) for k in ("n", "brier", "brier_skill", "inside_q10_q90")}
                         for c in claims}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--journal", default=None)
    ap.add_argument("--scoreboard-out", default=None)
    ap.add_argument("--today", default=None, help="YYYY-MM-DD (tests)")
    a = ap.parse_args(argv)
    push = not (a.no_push or a.dry_run)
    if not a.no_sync:
        pad.sync(["master_prices.parquet"])
    journal = Path(a.journal or U.journal_path())
    records = J.load(journal, pull=push)
    closes = load_closes()
    today = dt.date.fromisoformat(a.today) if a.today else dt.date.today()
    new = resolve(records, closes, today)
    last_bar = max((str(s.index.max().date()) for s in closes.values() if len(s)), default=None)
    sb = scoreboard(records + new, last_bar or str(today))
    print(f"[pm_agent_grade] journal={len(records)} new_resolutions={len(new)} "
          f"open={sb['n_open']} void={sb['n_void']}")
    for c, st in sb["claims"].items():
        if st.get("n"):
            print(f"  {c}: n={st['n']} brier={st['brier']:.3f} skill={st.get('brier_skill')} "
                  f"inside={st['inside_q10_q90']:.2f}")
    if a.dry_run:
        print("[pm_agent_grade] dry run: nothing written")
        return 0
    J.append(new, journal, push=push and bool(new))
    out = Path(a.scoreboard_out or U.scoreboard_path())
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, sb)
    if push:
        import cache_io
        if cache_io.is_configured() and not cache_io.upload_from_local(str(out), U.R2_PREFIX + "scoreboard.json"):
            print("[pm_agent_grade] R2 scoreboard upload failed")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
