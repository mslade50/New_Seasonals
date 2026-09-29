"""Best-effort look-back: how did the Daily Pitch's KILLED candidates perform?

Specs were frozen in specs/batch*.json from each kill's title, reason and
check scripts BEFORE any forward price was read (see SPEC_SCHEMA.md). This
script prices them and prints/writes a summary.

    python scratch/kill_lookback/price_kills.py                  # data/master_prices.parquet
    python scratch/kill_lookback/price_kills.py --prices PATH    # another long-format parquet
    python scratch/kill_lookback/price_kills.py --yf             # download from Yahoo instead

Per kill it reports
  ret     : close/open -> close return of the spec, per unit of primary-side notional
            (pair = long leg minus weight x short leg; basket = equal-weight mean)
  z       : ret / (pre-entry 63-session stdev of the spec's daily return x sqrt(h)),
            a vol-normalised "R-like" unit comparable across instruments
  drift   : the same legs' mean h-session return over the 756 sessions before entry
  excess  : ret - drift
Shipped ideas (journal `idea` records + the 09-23 UNG idea) are priced the same
way, so killed vs shipped is apples to apples.

Inference: kills on the same morning share one tape, so the only honest unit is
the pitch DATE. CIs are date-cluster bootstraps, and a "one-per-date" mean is
shown next to the per-kill mean.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
COST_BPS_PER_LEG = 5.0          # round trip, per leg, for the "net" column
BETA_WINDOW = 63
VOL_WINDOW = 63
DRIFT_WINDOW = 756
BOOT = 5000

# 2026-09-23's single shipped idea is only in the email (journal ends 09-18).
EXTRA_SHIPPED = [{
    "idea_id": "2026-09-23-1", "date": "2026-09-23", "grade": "B",
    "title": "Long natural gas for two sessions after an up day on three times normal volume",
    "legs": [{"ticker": "UNG", "side": "LONG", "weight": 1.0}],
    "entry": {"type": "close", "date": "2026-09-23"}, "horizon_td": 2,
}]


# ---------------------------------------------------------------------------
# prices
# ---------------------------------------------------------------------------
def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).capitalize() for c in df.columns]
    return df


def load_parquet(path: Path, tickers: set[str]) -> dict[str, pd.DataFrame]:
    mp = pd.read_parquet(path)
    mp = mp[mp["ticker"].isin(tickers)].copy()
    mp["date"] = pd.to_datetime(mp["date"])
    out = {}
    for t, g in mp.groupby("ticker"):
        g = _norm_cols(g.drop(columns=["ticker"]).sort_values("date").set_index("date"))
        out[t] = g[~g.index.duplicated(keep="last")]
    return out


def load_yf(tickers: set[str], start: str = "2022-06-01") -> dict[str, pd.DataFrame]:
    import yfinance as yf
    out = {}
    for t in sorted(tickers):
        raw = yf.download(t, start=start, progress=False, auto_adjust=True)
        if raw is None or raw.empty:
            continue
        if isinstance(raw.columns, pd.MultiIndex) and "Ticker" in raw.columns.names:
            raw = raw.xs(t, level="Ticker", axis=1)
        df = _norm_cols(raw.copy())
        df.index = pd.to_datetime(df.index).tz_localize(None)
        out[t] = df[~df.index.duplicated(keep="last")]
    return out


# ---------------------------------------------------------------------------
# pricing one spec
# ---------------------------------------------------------------------------
def _fit_beta(y: pd.Series, x: pd.Series) -> float:
    d = pd.concat([y, x], axis=1).dropna()
    if len(d) < 20 or d.iloc[:, 1].var() == 0:
        return 1.0
    return float(np.cov(d.iloc[:, 0], d.iloc[:, 1])[0, 1] / d.iloc[:, 1].var())


def price_spec(spec: dict, px: dict[str, pd.DataFrame]) -> dict:
    legs = spec.get("legs") or []
    h = int(spec.get("horizon_td") or 0)
    if not legs or h <= 0:
        return {"status": "bad_spec"}
    missing = [l["ticker"] for l in legs if l["ticker"] not in px]
    if missing:
        return {"status": "no_data", "missing": missing}
    lead = px[legs[0]["ticker"]]
    cal = lead["Close"].dropna().index
    entry_day = pd.Timestamp(spec["entry"]["date"])
    after = cal[cal >= entry_day]
    if len(after) == 0:
        return {"status": "future"}
    e = after[0]
    ei = cal.get_loc(e)
    xi = ei + h
    closed = xi < len(cal)
    x = cal[min(xi, len(cal) - 1)]
    etype = (spec["entry"].get("type") or "close").lower()

    # daily-return panel for weights / vol / drift, all strictly before entry
    rets = {l["ticker"]: px[l["ticker"]]["Close"].dropna().pct_change() for l in legs}
    # "beta" hedges are fitted against the primary-side basket (numeric weights,
    # normalised), not just the first name, so multi-name legs hedge correctly.
    prim_side = str(legs[0]["side"]).upper() in ("LONG", "BUY")
    prim = [(l, float(l.get("weight", 1.0))) for l in legs
            if (str(l["side"]).upper() in ("LONG", "BUY")) == prim_side
            and not isinstance(l.get("weight", 1.0), str)]
    pw = sum(w for _, w in prim) or 1.0
    lead_r = sum(rets[l["ticker"]] * (w / pw) for l, w in prim) if prim else rets[legs[0]["ticker"]]
    weights, sides = [], []
    for i, l in enumerate(legs):
        w = l.get("weight", 1.0)
        if isinstance(w, str):
            pre = slice(None, e - pd.Timedelta(days=1))
            w = abs(_fit_beta(lead_r.loc[pre].tail(BETA_WINDOW),
                              rets[l["ticker"]].loc[pre].tail(BETA_WINDOW)))
        weights.append(float(w))
        sides.append(1.0 if str(l["side"]).upper() in ("LONG", "BUY") else -1.0)
    prim = sides[0]
    denom = sum(w for w, s in zip(weights, sides) if s == prim) or 1.0

    leg_ret = []
    for l in legs:
        s = px[l["ticker"]]
        c = s["Close"].dropna()
        ce = c[c.index <= e]
        cx = c[c.index <= x]
        if ce.empty or cx.empty:
            return {"status": "no_data", "missing": [l["ticker"]]}
        p0 = float(ce.iloc[-1])
        if etype == "open" and "Open" in s and e in s.index and pd.notna(s.loc[e, "Open"]):
            p0 = float(s.loc[e, "Open"])
        leg_ret.append(float(cx.iloc[-1]) / p0 - 1.0)
    ret = sum(w * sd * r for w, sd, r in zip(weights, sides, leg_ret)) / denom

    panel = pd.concat([rets[l["ticker"]] for l in legs], axis=1).loc[:e].iloc[:-1]
    panel = panel.dropna()
    daily = (panel.values * np.array(weights) * np.array(sides)).sum(axis=1) / denom
    daily = pd.Series(daily, index=panel.index)
    vol = float(daily.tail(VOL_WINDOW).std())
    hist = daily.tail(DRIFT_WINDOW)
    drift = float((1 + hist).rolling(h).apply(np.prod, raw=True).sub(1).mean()) if len(hist) > h else np.nan
    z = ret / (vol * np.sqrt(h)) if vol and vol > 0 else np.nan
    net = ret - COST_BPS_PER_LEG / 1e4 * sum(weights) / denom
    return {"status": "closed" if closed else "open", "entry_session": str(e.date()),
            "exit_session": str(x.date()), "ret": ret, "net": net, "z": z,
            "drift": drift, "excess": ret - drift if pd.notna(drift) else np.nan,
            "weights": [round(w, 3) for w in weights]}


# ---------------------------------------------------------------------------
# reason classes (crude keyword tags on the kill reason, for a breakdown only)
# ---------------------------------------------------------------------------
REASON_CLASSES = [
    ("wrong_sign", r"wrong[- ]sign|runs backwards|opposite|inver|falsified"),
    ("filter_no_filter", r"does not filter|filter(s)? (subtract|out)|gate (subtract|does not|is worth -)|re-?anchor"),
    ("cost", r"\bcost\b|round trip|x a .* pair|x cost"),
    ("concentration", r"concentrat|top two|drop[- ]best|carry \d+% of|% of the total"),
    ("not_live", r"not live|unarmed|not armed|missed its|short of its"),
    ("era_instability", r"sign (flips|instability)|after 2018|2018\+|era"),
    ("registry_collision", r"registry collision|already closed|registry"),
]


def reason_class(reason: str) -> str:
    r = (reason or "").lower()
    for name, pat in REASON_CLASSES:
        if re.search(pat, r):
            return name
    return "other"


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------
def cluster_boot(df: pd.DataFrame, col: str, seed: int = 7) -> tuple[float, float, float]:
    d = df[["date", col]].dropna()
    if d.empty:
        return (np.nan, np.nan, np.nan)
    groups = [g[col].values for _, g in d.groupby("date")]
    rng = np.random.default_rng(seed)
    n = len(groups)
    means = []
    for _ in range(BOOT):
        pick = rng.integers(0, n, n)
        means.append(np.concatenate([groups[i] for i in pick]).mean())
    means = np.array(means)
    return (float(d[col].mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def summarize(df: pd.DataFrame, label: str) -> dict:
    if df.empty:
        return {"label": label, "n": 0}
    m_ret, lo_ret, hi_ret = cluster_boot(df, "ret")
    m_z, lo_z, hi_z = cluster_boot(df, "z")
    m_x, lo_x, hi_x = cluster_boot(df, "excess")
    per_date = df.groupby("date")["z"].mean()
    return {
        "label": label, "n": int(len(df)), "dates": int(df["date"].nunique()),
        "hit": float((df["ret"] > 0).mean()),
        "mean_ret_pct": 100 * m_ret, "ret_ci_pct": (100 * lo_ret, 100 * hi_ret),
        "median_ret_pct": 100 * float(df["ret"].median()),
        "mean_net_pct": 100 * float(df["net"].mean()),
        "mean_excess_pct": 100 * m_x, "excess_ci_pct": (100 * lo_x, 100 * hi_x),
        "mean_z": m_z, "z_ci": (lo_z, hi_z),
        "per_date_mean_z": float(per_date.mean()),
        "dates_positive_z": f"{int((per_date > 0).sum())}/{len(per_date)}",
    }


def fmt(s: dict) -> str:
    if not s.get("n"):
        return f"| {s['label']} | 0 | | | | | | |"
    return ("| {label} | {n} | {dates} | {hit:.0%} | {mean_ret_pct:+.2f}% [{r0:+.2f}, {r1:+.2f}] | "
            "{mean_excess_pct:+.2f}% | {mean_z:+.2f} [{z0:+.2f}, {z1:+.2f}] | {dates_positive_z} |").format(
        r0=s["ret_ci_pct"][0], r1=s["ret_ci_pct"][1], z0=s["z_ci"][0], z1=s["z_ci"][1], **s)


HEADER = ("| cohort | N | dates | hit | mean ret [95% date-cluster CI] | excess vs own drift | "
          "mean z [95% CI] | dates z>0 |\n|---|---|---|---|---|---|---|---|")


# ---------------------------------------------------------------------------
def shipped_specs() -> list[dict]:
    out = []
    for line in open(ROOT / "data" / "pitch_journal.jsonl"):
        r = json.loads(line)
        if r.get("kind") != "idea":
            continue
        sp = r.get("spec") or {}
        legs = [{"ticker": l.get("proxy_ticker") or l["ticker"], "side": l["side"],
                 "weight": l.get("weight", 1.0)} for l in sp.get("legs", [])]
        et = (sp.get("entry") or {}).get("type", "MOC").upper()
        out.append({"idea_id": r["idea_id"], "date": r["date"], "grade": r.get("grade"),
                    "title": r["title"], "legs": legs,
                    "entry": {"type": "open" if et == "MOO" else "close", "date": r["date"]},
                    "horizon_td": (sp.get("exit") or {}).get("time_td") or r.get("horizon_td")})
    return out + EXTRA_SHIPPED


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default=str(ROOT / "data" / "master_prices.parquet"))
    ap.add_argument("--yf", action="store_true", help="download prices from Yahoo instead")
    ap.add_argument("--out", default=str(HERE / "results"))
    a = ap.parse_args()

    kills = {k["kill_id"]: k for k in map(json.loads, open(HERE / "kills.jsonl"))}
    specs = []
    for f in sorted(glob.glob(str(HERE / "specs" / "batch*.json"))):
        specs += json.load(open(f))
    missing_specs = sorted(set(kills) - {s["kill_id"] for s in specs})
    ships = shipped_specs()

    tickers = {l["ticker"] for s in specs if s.get("tradeable") for l in s.get("legs") or []}
    tickers |= {l["ticker"] for s in ships for l in s["legs"]}
    px = load_yf(tickers) if a.yf else load_parquet(Path(a.prices), tickers)
    last = max(df.index.max() for df in px.values()) if px else None
    print(f"priced from {'Yahoo' if a.yf else a.prices}; last bar {last}; "
          f"{len(px)}/{len(tickers)} tickers found")

    rows = []
    for s in specs:
        k = kills.get(s["kill_id"], {})
        base = {"kill_id": s["kill_id"], "date": k.get("date", s["kill_id"][:10]),
                "title": k.get("title"), "axis": k.get("novelty_axis"),
                "reason_class": reason_class(k.get("reason")),
                "tradeable": bool(s.get("tradeable")), "gate_live": s.get("gate_live", True),
                "confidence": s.get("spec_confidence"), "horizon_td": s.get("horizon_td"),
                "legs": json.dumps(s.get("legs")), "entry": json.dumps(s.get("entry"))}
        if not s.get("tradeable"):
            rows.append({**base, "status": "untradeable"})
            continue
        rows.append({**base, **price_spec(s, px)})
    df = pd.DataFrame(rows)

    srows = []
    for s in ships:
        srows.append({"kill_id": s["idea_id"], "date": s["date"], "title": s["title"],
                      "grade": s.get("grade"), **price_spec(s, px)})
    sdf = pd.DataFrame(srows)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "kills_priced.csv", index=False)
    sdf.to_csv(out / "shipped_priced.csv", index=False)

    closed = df[df["status"] == "closed"].copy()
    # near-duplicate collapse: same date, same primary ticker and side -> one row
    def prim(r):
        l = json.loads(r["legs"])[0]
        return f"{l['ticker']}:{str(l['side']).upper()}"
    if not closed.empty:
        closed["prim"] = closed.apply(prim, axis=1)
        dedup = closed.groupby(["date", "prim"], as_index=False)[["ret", "net", "z", "excess"]].mean()
    else:
        dedup = closed
    sclosed = sdf[sdf.get("status", pd.Series(dtype=str)) == "closed"] if not sdf.empty else sdf

    lines = ["# Killed Daily Pitch candidates: best-effort look-back", "",
             f"Last price bar: {last}. Kills: {len(kills)}; specs: {len(specs)}"
             + (f" (MISSING specs: {len(missing_specs)})" if missing_specs else "") + ".",
             "", "Status counts: " + ", ".join(f"{k} {v}" for k, v in df["status"].value_counts().items()),
             "", "## Headline", "", HEADER,
             fmt(summarize(closed, "all closed kills")),
             fmt(summarize(dedup, "kills, near-dupes collapsed")),
             fmt(summarize(closed[closed["gate_live"] != False], "kills whose gate was live")),  # noqa: E712
             fmt(summarize(closed[closed["confidence"].isin(["high", "medium"])], "kills, spec conf high/med")),
             fmt(summarize(sclosed, "SHIPPED ideas, same pricer")),
             "", "## By kill-reason class", "", HEADER]
    for c, g in closed.groupby("reason_class"):
        lines.append(fmt(summarize(g, c)))
    lines += ["", "## By novelty axis", "", HEADER]
    for c, g in closed.groupby(closed["axis"].fillna("(email, untagged)")):
        lines.append(fmt(summarize(g, c)))
    lines += ["", "## By horizon", "", HEADER]
    hb = pd.cut(closed["horizon_td"].astype(float), [0, 1, 3, 5, 10, 999], labels=["1", "2-3", "4-5", "6-10", ">10"])
    for c, g in closed.groupby(hb, observed=True):
        lines.append(fmt(summarize(g, f"h {c}")))
    lines += ["", "## By primary direction", "", HEADER]
    for c, g in closed.groupby(closed["prim"].str.split(":").str[1] if not closed.empty else []):
        lines.append(fmt(summarize(g, c)))
    if not closed.empty:
        best = closed.nlargest(10, "z")[["kill_id", "title", "ret", "z"]]
        worst = closed.nsmallest(10, "z")[["kill_id", "title", "ret", "z"]]
        for name, t in (("Best 10 kills (by z)", best), ("Worst 10 kills (by z)", worst)):
            lines += ["", f"## {name}", "", "| kill | title | ret | z |", "|---|---|---|---|"]
            for _, r in t.iterrows():
                lines.append(f"| {r.kill_id} | {r.title[:90]} | {100*r.ret:+.2f}% | {r.z:+.2f} |")
    lines += ["", "## Shipped ideas, same pricer", "", "| idea | grade | title | ret | z | status |", "|---|---|---|---|---|---|"]
    for _, r in sdf.iterrows():
        lines.append(f"| {r.kill_id} | {r.get('grade')} | {str(r.title)[:80]} | "
                     f"{100*r.get('ret', np.nan):+.2f}% | {r.get('z', np.nan):+.2f} | {r.get('status')} |")
    nd = df[df["status"] == "no_data"]
    if not nd.empty:
        lines += ["", f"No price data for {len(nd)} kills: "
                  + ", ".join(sorted({m for ms in nd['missing'] for m in (ms or [])}))]
    text = "\n".join(lines)
    (out / "summary.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
