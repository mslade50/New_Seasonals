"""PM layer, book side: sync the read surface and compute what a PM looks at.

Every number here is DESCRIPTIVE and carries a basis label:

  live    Primary account, from the daily broker snapshot
          (R2 ops/olv_capacity/<date>.json: NLV, positions, working orders) and
          the canonical fills store (live_fills.parquet). NLV changes are NOT
          flow-adjusted; a day move over FLOW_SUSPECT is excluded from vol and
          flagged as a possible deposit or withdrawal.
  ledger  The modeled book (newest site/builds/<run>/backtest_*.parquet): a
          rebuild of TODAY's config over all history on the flat $750k basis,
          not a fill record. Pre-change notional understates live
          (docs/claude_ref/ledger_and_fills.md). Overflow-tier stats are
          survivorship-biased upper bounds.

There is NO vol target in this book (a target/scaler is a closed negative,
docs/claude_ref/sizing.md). Realised vol is reported against the ledger's own
long-run reference and against the ledger over the same window, never against
a band, and nothing here proposes a size.

Agent-product module: the book and the Risk Agent must not import it.
"""
from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import pm_agent_data as pad
import pm_agent_universe as U

FLAT_BASE = 750_000.0
FLOW_SUSPECT = 0.05          # |daily NLV change| above this: possible flow, excluded
SNAPSHOT_DAYS = 70           # calendar days of broker snapshots kept in the sync
RECEIPT_DAYS = 8
ANN = math.sqrt(252)


def _r(x, nd=2):
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return round(f, nd) if math.isfinite(f) else None


def _read_json(p: Path):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# sync
# ---------------------------------------------------------------------------
def dynamic_keys(today: dt.date, listing: set[str]) -> dict[str, list[str]]:
    """Choose the dated keys to sync from one R2 listing (pure, testable)."""
    lo = today - dt.timedelta(days=SNAPSHOT_DAYS)
    snaps = sorted(k for k in listing if k.startswith("ops/olv_capacity/2") and k.endswith(".json")
                   and str(lo) <= k.rsplit("/", 1)[-1][:10] <= str(today))
    rlo = str(today - dt.timedelta(days=RECEIPT_DAYS))
    receipts = sorted(k for k in listing if k.startswith("automation/receipts/v1/")
                      and k.endswith("/latest.json") and k.split("/")[3] >= rlo)
    deliveries = sorted(k for k in listing
                        if k.startswith(("pitch_delivery_receipts/", "seasonal_agent_delivery_receipts/",
                                         "risk_agent/delivery_receipts/"))
                        and k.rsplit("/", 1)[-1][:10] >= rlo)
    runs: dict[str, set] = {}
    for k in listing:
        if k.startswith(U.LEDGER_PREFIX):
            parts = k.split("/")
            if len(parts) == 4 and parts[3] in U.LEDGER_NAMES:
                runs.setdefault(parts[2], set()).add(parts[3])
    complete = [r for r, names in runs.items() if set(U.LEDGER_NAMES) <= names]
    ledger = []
    if complete:
        best = max(complete, key=lambda r: [int(x) if x.isdigit() else 0 for x in r.split("-")])
        ledger = [f"{U.LEDGER_PREFIX}{best}/{n}" for n in U.LEDGER_NAMES]
    return {"snapshots": snaps, "receipts": receipts, "deliveries": deliveries, "ledger": ledger}


def sync_book(today: dt.date | None = None, cache_dir: Path | None = None,
              with_ledger: bool = True) -> dict:
    """Download the book read surface. Returns {"keys": {...}, "result": {...}}."""
    import cache_io
    today = today or dt.date.today()
    listing = set()
    for prefix in ("ops/olv_capacity/", "automation/receipts/v1/", "pitch_delivery_receipts/",
                   "seasonal_agent_delivery_receipts/", "risk_agent/delivery_receipts/",
                   U.LEDGER_PREFIX if with_ledger else None):
        if prefix:
            listing |= set(cache_io.list_keys(prefix))
    keys = dynamic_keys(today, listing)
    if not with_ledger:
        keys["ledger"] = []
    wanted = list(U.BOOK_KEYS) + [k for v in keys.values() for k in v]
    result = pad.sync(wanted, cache_dir=cache_dir)
    _save_index(keys, cache_dir)
    return {"keys": keys, "result": result}


def _index_path(cache_dir: Path | None) -> Path:
    return Path(cache_dir or U.cache_dir()) / "_book_index.json"


def _save_index(keys: dict, cache_dir: Path | None) -> None:
    p = _index_path(cache_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(keys, indent=1), encoding="utf-8")


def load_index(cache_dir: Path | None = None) -> dict:
    return _read_json(_index_path(cache_dir)) or {}


# ---------------------------------------------------------------------------
# live: broker snapshots
# ---------------------------------------------------------------------------
def primary_account(snapshot: dict) -> dict | None:
    for a in ((snapshot or {}).get("book") or {}).get("accounts") or []:
        if a.get("key") == "primary":
            return a
    return None


def nav_series(snapshots: list[dict]) -> pd.Series:
    """Primary NLV by session from dated snapshots (last observation per session)."""
    rows = {}
    for s in snapshots:
        a = primary_account(s)
        sess = s.get("session")
        if a and sess and isinstance(a.get("nlv"), (int, float)) and a["nlv"] > 0:
            rows[pd.Timestamp(sess)] = float(a["nlv"])
    return pd.Series(rows, dtype="float64").sort_index()


def live_vol(nav: pd.Series) -> dict:
    out = {"basis": "live Primary NLV, not flow-adjusted", "n_snapshots": int(len(nav))}
    if len(nav) < 2:
        return out
    r = nav.pct_change().dropna()
    flows = r[r.abs() > FLOW_SUSPECT]
    clean = r[r.abs() <= FLOW_SUSPECT]
    out.update({
        "first": str(nav.index[0].date()), "last": str(nav.index[-1].date()),
        "nlv_last": _r(nav.iloc[-1], 0), "nlv_first": _r(nav.iloc[0], 0),
        "return_since_first_pct": _r(100 * (nav.iloc[-1] / nav.iloc[0] - 1)),
        "max_drawdown_pct": _r(100 * float((nav / nav.cummax() - 1).min())),
        "suspected_flows": [{"date": str(d.date()), "chg_pct": _r(100 * v)} for d, v in flows.items()],
        "last_day_chg_pct": _r(100 * r.iloc[-1]),
    })
    week = nav[nav.index > nav.index[-1] - pd.Timedelta(days=7)]
    prior = nav[nav.index <= nav.index[-1] - pd.Timedelta(days=7)]
    if len(prior):
        out["week_chg_pct"] = _r(100 * (nav.iloc[-1] / prior.iloc[-1] - 1))
    for n in (10, 20):
        if len(clean) >= n:
            out[f"ann_vol_{n}d_pct"] = _r(100 * clean.iloc[-n:].std(ddof=1) * ANN)
    if len(clean) >= 5:
        out["ann_vol_all_pct"] = _r(100 * clean.std(ddof=1) * ANN)
        out["n_returns"] = int(len(clean))
    out["sessions_this_week"] = int(len(week))
    return out


def _notional(p: dict) -> float:
    mv = p.get("market_value")
    if p.get("sec_type") == "FUT":
        mult = float(p.get("multiplier") or 1.0)
        return float(p.get("position") or 0) * float(p.get("market_price") or 0) * mult
    return float(mv or 0)


def live_exposure(snapshot: dict) -> dict:
    a = primary_account(snapshot)
    if not a or not a.get("nlv"):
        return {"available": False}
    nlv = float(a["nlv"])
    stk = [p for p in a.get("positions") or [] if p.get("sec_type") in ("STK", "FUT")]
    opt = [p for p in a.get("positions") or [] if p.get("sec_type") == "OPT"]
    vals = [(p.get("symbol"), p.get("sec_type"), _notional(p)) for p in stk]
    long_ = sum(v for _, _, v in vals if v > 0)
    short = -sum(v for _, _, v in vals if v < 0)
    top = sorted(vals, key=lambda t: -abs(t[2]))[:6]
    orders = [o for o in a.get("orders") or [] if (o.get("remaining") or 0) > 0]
    o_not = {"BUY": 0.0, "SELL": 0.0}
    for o in orders:
        px = o.get("lmt") or o.get("aux") or 0
        o_not[o.get("action", "BUY") if o.get("action") in o_not else "BUY"] += float(o.get("remaining") or 0) * float(px or 0)
    upnl = sum(float(p.get("unrealized_pnl") or 0) for p in (a.get("positions") or []))
    return {"available": True, "basis": "live Primary broker snapshot", "session": snapshot.get("session"),
            "observed_at": snapshot.get("observed_at"), "nlv": _r(nlv, 0),
            "long_pct": _r(100 * long_ / nlv, 1), "short_pct": _r(100 * short / nlv, 1),
            "gross_pct": _r(100 * (long_ + short) / nlv, 1), "net_pct": _r(100 * (long_ - short) / nlv, 1),
            "n_positions": len(stk), "n_option_legs": len(opt),
            "option_mv_pct": _r(100 * sum(float(p.get("market_value") or 0) for p in opt) / nlv, 2),
            "unrealized_pnl": _r(upnl, 0),
            "top": [{"symbol": s, "type": t, "pct_nlv": _r(100 * v / nlv, 1)} for s, t, v in top],
            "working_orders": len(orders),
            "working_buy_pct": _r(100 * o_not["BUY"] / nlv, 1),
            "working_sell_pct": _r(100 * o_not["SELL"] / nlv, 1)}


# ---------------------------------------------------------------------------
# live: fills
# ---------------------------------------------------------------------------
def fills_window(fills: pd.DataFrame, start: str, end: str) -> dict:
    f = fills[(fills["session_date"].astype(str) >= start) & (fills["session_date"].astype(str) <= end)].copy()
    out = {"basis": "live fills (canonical store)", "start": start, "end": end, "n": int(len(f))}
    if f.empty:
        return out
    f["strategy"] = f["strategy"].fillna("").astype(str).replace("", "untagged")
    f["notional"] = f["qty"].astype(float) * f["price"].astype(float)
    f["sign"] = np.where(f["side"].astype(str).str.upper().str.startswith("B"), 1, -1)
    rows = []
    for (acct, strat), g in f.groupby(["account_key", "strategy"]):
        rows.append({"account": acct, "strategy": strat, "fills": int(len(g)),
                     "buy_notional": _r(g.loc[g["sign"] > 0, "notional"].sum(), 0),
                     "sell_notional": _r(g.loc[g["sign"] < 0, "notional"].sum(), 0),
                     "realized_pnl": _r(g["realized_pnl"].astype(float).sum(), 0),
                     "commission": _r(g["commission"].astype(float).sum(), 2)})
    rows.sort(key=lambda r: -(r["buy_notional"] or 0) - (r["sell_notional"] or 0))
    out.update({"by_strategy": rows[:20],
                "untagged_pct": _r(100 * (f["strategy"] == "untagged").mean(), 1),
                "realized_pnl_total": _r(f["realized_pnl"].astype(float).sum(), 0),
                "commission_total": _r(f["commission"].astype(float).sum(), 2)})
    return out


def fills_health(status: dict | None) -> dict:
    s = status or {}
    gap = (s.get("gap") or {})
    prim = ((s.get("completeness") or {}).get("accounts") or {}).get("primary") or {}
    return {"last_session": s.get("last_session"), "rows": s.get("rows"), "tagged_pct": s.get("tagged_pct"),
            "gap": bool(gap.get("gap")), "gap_reason": gap.get("reason"),
            "primary_complete": prim.get("complete"), "primary_error": prim.get("error")}


# ---------------------------------------------------------------------------
# ledger (modeled book)
# ---------------------------------------------------------------------------
def ledger_vol(daily: pd.DataFrame, asof: str | None = None) -> dict:
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").set_index("date")
    if asof:
        d = d[d.index <= pd.Timestamp(asof)]
    r = d["pnl_flat"].astype(float) / FLAT_BASE
    out = {"basis": "ledger rebuild, flat $750k", "last": str(d.index[-1].date()) if len(d) else None}
    for n in (21, 63, 252):
        if len(r) >= n:
            out[f"ann_vol_{n}d_pct"] = _r(100 * r.iloc[-n:].std(ddof=1) * ANN)
    roll = r.rolling(63).std(ddof=1) * ANN * 100
    ref = roll[roll.index > roll.index[-1] - pd.DateOffset(years=3)].dropna() if len(roll.dropna()) else roll
    if len(ref):
        out["ref_63d_vol_3y_median_pct"] = _r(ref.median())
        out["ref_63d_vol_3y_p90_pct"] = _r(ref.quantile(0.9))
        if f"ann_vol_63d_pct" in out:
            out["pctile_63d_vol_3y"] = _r(100 * (ref < out["ann_vol_63d_pct"]).mean(), 0)
    by_year = r.groupby(r.index.year).agg(lambda x: x.std(ddof=1) * ANN * 100)
    out["vol_by_year_pct"] = {str(y): _r(v) for y, v in by_year.iloc[-4:].items()}
    ytd = r[r.index.year == r.index[-1].year]
    out["ytd_pnl_pct_of_base"] = _r(100 * ytd.sum())
    eq = r.cumsum()
    out["max_dd_252d_pct_of_base"] = _r(100 * float((eq.iloc[-252:] - eq.iloc[-252:].cummax()).min()))
    out["_daily_ret"] = r     # internal; stripped before serialising
    return out


def _direction_sign(s: pd.Series) -> np.ndarray:
    return np.where(s.astype(str).str.lower().str.startswith("s"), -1.0, 1.0)


def ledger_exposure(trades: pd.DataFrame, asof: str | None = None, days: int = 252) -> dict:
    t = trades.copy()
    t["Entry Date"] = pd.to_datetime(t["Entry Date"])
    t["Exit Date"] = pd.to_datetime(t["Exit Date"])
    end = pd.Timestamp(asof) if asof else max(t["Exit Date"].max(), t["Entry Date"].max())
    idx = pd.bdate_range(end=end, periods=days)
    # open trades carry no exit yet: hold them through the window end
    t["Exit Date"] = t["Exit Date"].fillna(end + pd.Timedelta(days=1))
    t = t[(t["Exit Date"] >= idx[0]) & (t["Entry Date"] <= end) & (t["Shares_flat"].fillna(0) > 0)]
    notional = t["Shares_flat"].astype(float).to_numpy() * t["Entry Price"].astype(float).to_numpy()
    sign = _direction_sign(t["Direction"]) if "Direction" in t.columns else np.ones(len(t))
    gross = np.zeros(len(idx))
    net = np.zeros(len(idx))
    ent = t["Entry Date"].to_numpy()
    ex = t["Exit Date"].to_numpy()
    for i, d in enumerate(idx.to_numpy()):
        m = (ent <= d) & (ex >= d)       # held at some point during session d
        gross[i] = notional[m].sum()
        net[i] = (notional[m] * sign[m]).sum()
    g = gross / FLAT_BASE * 100
    n = net / FLAT_BASE * 100
    return {"basis": "ledger rebuild, flat $750k (pre-change notional understates live)",
            "asof": str(end.date()), "gross_pct_last": _r(g[-1], 1), "net_pct_last": _r(n[-1], 1),
            "gross_pct_mean_63d": _r(g[-63:].mean(), 1), "gross_pct_p50_252d": _r(np.median(g), 1),
            "gross_pct_p95_252d": _r(np.quantile(g, 0.95), 1),
            "share_days_gross_lt_25pct": _r(100 * (g < 25).mean(), 1),
            "share_days_flat": _r(100 * (g == 0).mean(), 1)}


def capital_efficiency(trades: pd.DataFrame, asof: str | None = None, days: int = 365) -> dict:
    """Capital Efficiency Ratio per strategy x tier (docs/portfolio_logic.md):
    share of P&L / share of risk, trailing window by exit date and full sample."""
    t = trades.copy()
    t["Exit Date"] = pd.to_datetime(t["Exit Date"])
    end = pd.Timestamp(asof) if asof else t["Exit Date"].max()
    t = t[t["Exit Date"] <= end]

    def table(df):
        tot_p = df["PnL_flat_750k"].astype(float).sum()
        tot_r = df["Risk_flat_750k"].astype(float).sum()
        rows = []
        for (s, tier), g in df.groupby(["Strategy", "Tier"]):
            p, rk = g["PnL_flat_750k"].astype(float).sum(), g["Risk_flat_750k"].astype(float).sum()
            cer = (p / tot_p) / (rk / tot_r) if tot_p and tot_r and rk else None
            rows.append({"strategy": s, "tier": tier, "trades": int(len(g)), "pnl": _r(p, 0),
                         "risk_share_pct": _r(100 * rk / tot_r, 1) if tot_r else None,
                         "pnl_share_pct": _r(100 * p / tot_p, 1) if tot_p else None,
                         "cer": _r(cer), "avg_r": _r(g["R_Multiple"].astype(float).mean())})
        rows.sort(key=lambda r: -(r["risk_share_pct"] or 0))
        return {"total_pnl": _r(tot_p, 0), "total_risk": _r(tot_r, 0), "rows": rows}

    recent = t[t["Exit Date"] > end - pd.Timedelta(days=days)]
    return {"basis": "ledger rebuild, flat $750k; Overflow tier is a survivorship-biased upper bound",
            "asof": str(end.date()), "trailing_days": days,
            "trailing": table(recent), "full": table(t)}


def live_vs_ledger_vol(nav_vol: dict, nav: pd.Series, ledger_ret: pd.Series | None) -> dict:
    if ledger_ret is None or len(nav) < 6:
        return {}
    r = nav.pct_change().dropna()
    r = r[r.abs() <= FLOW_SUSPECT]
    common = r.index.intersection(ledger_ret.index)
    if len(common) < 5:
        return {"n_common": int(len(common))}
    lv, mv = r.loc[common], ledger_ret.loc[common]
    return {"n_common": int(len(common)), "window": [str(common[0].date()), str(common[-1].date())],
            "live_ann_vol_pct": _r(100 * lv.std(ddof=1) * ANN),
            "ledger_ann_vol_pct": _r(100 * mv.std(ddof=1) * ANN),
            "corr": _r(float(np.corrcoef(lv, mv)[0, 1]), 2) if lv.std() > 0 and mv.std() > 0 else None,
            "note": "live is % of actual NLV, ledger % of flat $750k: the ratio mixes both bases"}


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------
def receipts_summary(receipts: list[dict]) -> dict:
    """{run_date: [{job, status, health, phase}]} filtered to anything not plain success."""
    by_day: dict[str, list] = {}
    for j in receipts:
        day = j.get("run_date_et")
        if not day:
            continue
        st, hs = j.get("status"), j.get("health_status")
        if st != "success" or (hs not in (None, "ok", "healthy")):
            by_day.setdefault(day, []).append({"job": j.get("job_id"), "status": st, "health": hs,
                                               "phase": j.get("phase"), "detail": str(j.get("detail") or "")[:200]})
    return dict(sorted(by_day.items()))


def runtime_issues(rt: dict | None) -> list[dict]:
    """Sleeve tasks that should run but did not, honouring each enable flag."""
    rt = rt or {}
    enabled = {"event": rt.get("event_enabled", True), "trend": rt.get("trend_moo_enabled", False),
               "legend": rt.get("legend_enabled", False), "legend_verify": rt.get("legend_enabled", False)}
    out = []
    for name, t in (rt.get("tasks") or {}).items():
        if not enabled.get(name, True):
            continue
        if t.get("state") == "Missing" or (t.get("last_result") not in (0, None)):
            out.append({"task": name, "state": t.get("state"), "last_result": t.get("last_result"),
                        "last_run_at": t.get("last_run_at")})
    return out


def sleeves_summary(cache_dir: Path | None = None) -> dict:
    lp = lambda k: pad.local_path(k, cache_dir)  # noqa: E731
    ex = _read_json(lp("exposure_state.json")) or {}
    tr = _read_json(lp("trend_sleeve_state.json")) or {}
    ev = _read_json(lp("event_sleeve_state.json")) or {}
    ds = _read_json(lp("dial_sleeve_paper.json")) or {}
    ee = _read_json(lp("ops/expected_exit_status.json")) or {}
    return {"exposure_leg": {"asof": ex.get("asof"), "mult": ex.get("mult"), "rule": ex.get("active_rule"),
                             "reason": ex.get("reason")},
            "trend_sleeve": {"asof": tr.get("asof"), "positions": len(tr.get("positions") or {}),
                             "gate": (tr.get("fragility_gate") or {}).get("state")},
            "event_sleeve": {"open": sorted((ev.get("positions") or {}).keys()), "generated": ev.get("generated")},
            "dial_sleeve_paper": {"position": ds.get("position"), "last_evaluated": ds.get("last_evaluated")},
            "expected_exits": {"status": ee.get("status"), "counts": ee.get("counts")}}


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------
def load_snapshots(keys: list[str], cache_dir: Path | None = None) -> list[dict]:
    out = []
    for k in keys:
        j = _read_json(pad.local_path(k, cache_dir))
        if isinstance(j, dict):
            out.append(j)
    return out


def build_book(asof: str, week_start: str, cache_dir: Path | None = None, warnings: list | None = None) -> dict:
    warnings = warnings if warnings is not None else []
    idx = load_index(cache_dir)
    out: dict = {"asof": asof}
    snaps = load_snapshots(idx.get("snapshots") or [], cache_dir)
    nav = nav_series(snaps)
    out["live_vol"] = live_vol(nav)
    latest = _read_json(pad.local_path("ops/olv_capacity/latest.json", cache_dir))
    out["live_exposure"] = live_exposure(latest) if latest else {"available": False}
    if not snaps:
        warnings.append("book: no broker snapshots synced")
    fp = pad.local_path("live_fills.parquet", cache_dir)
    if fp.exists():
        fills = pd.read_parquet(fp)
        out["fills_week"] = fills_window(fills, week_start, asof)
    else:
        warnings.append("book: live_fills.parquet missing")
    out["fills_health"] = fills_health(_read_json(pad.local_path("live_fills_status.json", cache_dir)))
    led = idx.get("ledger") or []
    ledger_ret = None
    if len(led) == 2 and all(pad.local_path(k, cache_dir).exists() for k in led):
        trades = pd.read_parquet(pad.local_path(led[0], cache_dir))
        daily = pd.read_parquet(pad.local_path(led[1], cache_dir))
        lv = ledger_vol(daily)
        ledger_ret = lv.pop("_daily_ret")
        out["ledger_vol"] = lv
        out["ledger_exposure"] = ledger_exposure(trades)
        out["capital_efficiency"] = capital_efficiency(trades)
        try:
            import pyarrow.parquet as pq
            md = pq.read_metadata(pad.local_path(led[0], cache_dir)).metadata or {}
            out["ledger_build"] = {k.decode(): md[k].decode() for k in (b"ledger_build_utc", b"ledger_git_sha")
                                   if k in md}
        except Exception:  # noqa: BLE001
            pass
        out["ledger_run"] = led[0].split("/")[2]
    else:
        warnings.append("book: ledger bundle not synced")
    out["live_vs_ledger_vol"] = live_vs_ledger_vol(out["live_vol"], nav, ledger_ret)
    out["sleeves"] = sleeves_summary(cache_dir)
    out["runtime_issues"] = runtime_issues(_read_json(pad.local_path("ops/sleeve_runtime_status.json", cache_dir)))
    rec = load_snapshots(idx.get("receipts") or [], cache_dir)
    out["job_issues"] = receipts_summary(rec)
    return out
