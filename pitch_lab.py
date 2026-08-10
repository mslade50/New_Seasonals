"""Shared research library for Daily Pitch falsification checks.

Every morning's check scripts under scratch/pitch_checks/<date>/ import from
here instead of re-deriving price loading, event alignment, forward-return
math, declustering, controls and summary stats. This module is the permanent
home of the machinery the 2026-08-07 run had to invent ad hoc (_common.py /
_engine.py / _study.py in that day's folder), promoted so the statistics that
decide kills are identical from one morning to the next.

READ-ONLY BY CONTRACT: nothing here writes to data/ or touches the live book.
The systematic book must never import this module (same rule as
pitch_grammar): a pitch statistic is never a scanner statistic.

Conventions every check inherits, stated once:

- Returns are FRACTIONS internally; ``summarize()`` reports PERCENT columns
  (``mean_pct`` etc). Passing percent into ``summarize`` double-scales and
  every column comes out 100x wrong -- this exact bug shipped on 2026-08-07.
- Entry convention: signal measured on close D, entry MOC on close D+``lag``
  (lag=1 is the real tradeable order; lag=0 is the naive look for contrast),
  exit on close D+lag+h. ``fwd_lag`` and ``vehicle_ret`` encode this.
- Controls: the instrument's OWN drift over the same span (CTRL-a), all days
  full history (CTRL-b), and the LOCAL +/-126td neighbourhood ex-trigger
  (kills the "triggers just live in a good regime" explanation).
- ATR is Wilder-14 (``pitch_grammar.wilder_atr``), never the scanner's
  simple mean.
- For N < 15 the honest test is the exact sign test (``sign_test``), not a
  t-stat. A 6-0 record is p=0.016 under a coin; report that, not "t
  unavailable".
"""
from __future__ import annotations

import json
import sys
from fractions import Fraction
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pitch_grammar import wilder_atr  # noqa: E402,F401  (re-export, the pitch ATR)

PRICES_PATH = ROOT / "data" / "master_prices.parquet"
EVENTS_PATH = ROOT / "data" / "macro_events.csv"
WATCHLIST_PATH = ROOT / "data" / "pitch_watchlist.json"


# ---------------------------------------------------------------------------
# data access
# ---------------------------------------------------------------------------
def load_prices(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Per-ticker OHLCV frames indexed by date (adjusted basis, as cached)."""
    mp = pd.read_parquet(PRICES_PATH)
    mp = mp[mp["ticker"].isin(tickers)].copy()
    mp["date"] = pd.to_datetime(mp["date"])
    out = {}
    for t, g in mp.groupby("ticker"):
        g = g.drop(columns=["ticker"]).sort_values("date").set_index("date")
        out[t] = g[~g.index.duplicated(keep="last")]
    missing = set(tickers) - set(out)
    if missing:
        print(f"pitch_lab.load_prices: MISSING from cache: {sorted(missing)}")
    return out


def close_panel(tickers: list[str]) -> pd.DataFrame:
    """Aligned close panel, one column per ticker."""
    px = load_prices(tickers)
    return pd.DataFrame({t: px[t]["Close"] for t in px}).dropna(how="all")


def load_events(kinds: list[str] | None = None) -> pd.DataFrame:
    """macro_events.csv (2000..2027): nfp, cpi, ppi, fomc_decision,
    fomc_minutes, opex, quad_witching, vix_expiry, jackson_hole, election."""
    e = pd.read_csv(EVENTS_PATH)
    e["date"] = pd.to_datetime(e["date"])
    if kinds:
        e = e[e["event"].isin(kinds)]
    return e.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# state / signal transforms (match pitch_state definitions exactly)
# ---------------------------------------------------------------------------
def zscore(s: pd.Series, n: int = 10) -> pd.Series:
    """z10 as pitch_state defines it: n-day return over its trailing-year sd."""
    r = s.pct_change(n)
    return (r - r.rolling(252).mean()) / r.rolling(252).std()


def pct_rank(s: pd.Series, n: int, lookback: int = 252) -> pd.Series:
    """Trailing-`lookback` percentile rank of the n-day return (0-100)."""
    r = s.pct_change(n)
    return r.rolling(lookback).rank(pct=True) * 100.0


# ---------------------------------------------------------------------------
# forward returns (the load-bearing conventions)
# ---------------------------------------------------------------------------
def fwd_ret(s: pd.Series, h: int) -> pd.Series:
    """NAIVE close-to-close forward return, aligned to the anchor day (lag=0).
    Use only as the no-lag contrast row; the tradeable number is fwd_lag."""
    return s.shift(-h) / s - 1.0


def fwd_lag(s: pd.Series, h: int, lag: int = 1) -> pd.Series:
    """Forward return entering at close D+lag, exiting h sessions after that.
    lag=1 is the real MOC-tomorrow order."""
    return s.shift(-(lag + h)) / s.shift(-lag) - 1.0


def vehicle_ret(px: pd.DataFrame, legs: list[tuple[str, float]], h: int,
                lag: int = 1) -> pd.Series:
    """Weighted leg combination aligned to the SIGNAL date. legs are
    (ticker, weight) with weight sign carrying the side (short = negative)."""
    out = None
    for tkr, w in legs:
        r = fwd_lag(px[tkr], h, lag)
        out = w * r if out is None else out + w * r
    return out


# ---------------------------------------------------------------------------
# episodes, controls, splits
# ---------------------------------------------------------------------------
def declusters(idx: pd.DatetimeIndex, min_gap_td: int,
               all_dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Keep the first event of each cluster; `min_gap_td` in trading days."""
    pos = pd.Series(range(len(all_dates)), index=all_dates)
    keep, last = [], -10**9
    for d in sorted(idx):
        p = pos.get(d)
        if p is None:
            continue
        if p - last >= min_gap_td:
            keep.append(d)
            last = p
    return pd.DatetimeIndex(keep)


def local_control(all_dates: pd.DatetimeIndex, trig: pd.DatetimeIndex,
                  win: int = 126) -> pd.DatetimeIndex:
    """Every day within +/-`win` td of some trigger, trigger days removed."""
    pos = pd.Series(range(len(all_dates)), index=all_dates)
    keep = np.zeros(len(all_dates), dtype=bool)
    for d in trig:
        p = pos.get(d)
        if p is None:
            continue
        keep[max(0, p - win):min(len(all_dates), p + win + 1)] = True
    out = all_dates[keep]
    return out.difference(trig)


def era_split(dates: pd.DatetimeIndex, vals: np.ndarray,
              cut: str = "2018-01-01") -> list[dict]:
    d = pd.DatetimeIndex(dates)
    m = d < pd.Timestamp(cut)
    return [summarize(np.asarray(vals)[m], f"pre-{cut[:4]}"),
            summarize(np.asarray(vals)[~m], f"{cut[:4]}+")]


def event_in_window(anchor: pd.DatetimeIndex, all_dates: pd.DatetimeIndex,
                    h: int, lag: int = 1,
                    kinds: tuple[str, ...] = ("cpi",)) -> np.ndarray:
    """True when a print of the given kind lands strictly after the entry
    close and on/before the exit close."""
    ev = load_events(list(kinds))["date"].values.astype("datetime64[ns]")
    pos = pd.Series(range(len(all_dates)), index=all_dates)
    out = []
    for d in anchor:
        p = pos.get(d)
        if p is None or p + lag + h >= len(all_dates):
            out.append(False)
            continue
        lo, hi = all_dates[p + lag], all_dates[p + lag + h]
        out.append(bool(((ev > np.datetime64(lo)) & (ev <= np.datetime64(hi))).any()))
    return np.asarray(out, dtype=bool)


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def summarize(vals: np.ndarray, label: str = "") -> dict:
    """FRACTIONS in, PERCENT out. Do not pre-scale the input."""
    v = np.asarray(vals, dtype=float)
    v = v[~np.isnan(v)]
    n = len(v)
    if n == 0:
        return {"label": label, "n": 0}
    sd = v.std(ddof=1) if n > 1 else np.nan
    t = v.mean() / (sd / np.sqrt(n)) if n > 1 and sd > 0 else np.nan
    return {
        "label": label,
        "n": n,
        "mean_pct": 100 * v.mean(),
        "median_pct": 100 * float(np.median(v)),
        "hit": 100 * float((v > 0).mean()),
        "t": t,
        "worst_pct": 100 * v.min(),
        "best_pct": 100 * v.max(),
        "sd_pct": 100 * sd,
    }


def show(rows: list[dict], title: str = "") -> None:
    if title:
        print(f"\n=== {title} ===")
    df = pd.DataFrame(rows)
    if df.empty:
        print("  (empty)")
        return
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].round(3)
    print(df.to_string(index=False))


def sign_test(wins: int, n: int, p: float = 0.5) -> float:
    """One-sided exact binomial: P(>= wins successes in n) under win-prob p.
    THE small-sample statistic. 6-0 -> 0.0156, 8-1 -> 0.0195, 10-2 -> 0.0193.
    A clean record with a large per-event edge and a mechanism needs no
    t-stat; quote this instead.

    The p=0.5 path is exact rational arithmetic. The obvious float form,
    ``sum(comb(n, k) * p**k * (1-p)**(n-k))``, raises OverflowError once n is
    a few hundred, because comb(n, k) becomes a Python int too large to
    convert to float before the tiny probability can scale it back down. That
    is a crash rather than a wrong answer, and it fires exactly when a check
    script measures a CONTROL cell (all Mondays, all days) alongside the small
    conditional cell this function exists for. Values for every n that already
    worked are unchanged.
    """
    if n <= 0 or wins < 0 or wins > n:
        return np.nan
    if p == 0.5:
        # Fraction keeps the big ints exact and rounds once, at the end.
        return float(Fraction(sum(comb(n, k) for k in range(wins, n + 1)),
                              1 << n))
    return float(sum(comb(n, k) * p**k * (1 - p)**(n - k)
                     for k in range(wins, n + 1)))


def bootstrap_p_le0(vals: np.ndarray, n_boot: int = 5000,
                    seed: int = 42) -> float:
    """P(mean <= 0) under a simple iid bootstrap. Use on declustered episodes
    only; day-level overlap inflates it."""
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 3:
        return np.nan
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return float((means <= 0).mean())


def cluster_note(dates: pd.DatetimeIndex, vals: np.ndarray, k: int = 2) -> str:
    """How much of total return sits in the top-k episodes / best years?"""
    v = np.asarray(vals, float)
    if len(v) == 0:
        return "n/a"
    tot = v.sum()
    order = np.argsort(-np.abs(v))[:k]
    top = v[order].sum()
    yrs = pd.DatetimeIndex(dates).year
    by_yr = pd.Series(v).groupby(yrs.values).sum().sort_values(ascending=False)
    share = (top / tot * 100) if tot != 0 else np.nan
    return (f"top{k} episodes {[str(pd.Timestamp(dates[i]).date()) for i in order]} "
            f"= {100*top:+.2f}pp of {100*tot:+.2f}pp total ({share:.0f}%); "
            f"best yrs {dict((y, round(100*r, 1)) for y, r in by_yr.head(2).items())}")


# ---------------------------------------------------------------------------
# the kill battery (stage C round 1, one call)
# ---------------------------------------------------------------------------
def battery(px: pd.DataFrame, mask: pd.Series, legs: list[tuple[str, float]],
            h: int, title: str, cost_bps: float,
            variants: dict[str, pd.Series] | None = None, lag: int = 1,
            min_gap: int | None = None,
            event_kinds: tuple[str, ...] = ("cpi",)) -> None:
    """Full round-1 falsification battery: pattern vs three controls,
    declustered episodes, era split, threshold sensitivity, cost sanity,
    event-in-window split, sign test and concentration note. Prints; the
    check script interprets."""
    ret = vehicle_ret(px, legs, h, lag)
    naive = vehicle_ret(px, legs, h, lag=0)
    valid = ret.notna()
    sig_all = px.index[mask.reindex(px.index, fill_value=False).values
                       & valid.values]
    if len(sig_all) == 0:
        print(f"\n##### {title}: NO TRIGGERS EVER. Dead. #####")
        return
    span = (sig_all[0], sig_all[-1])
    in_span = (px.index >= span[0]) & (px.index <= span[1]) & valid.values

    print(f"\n##### {title}  (h={h} td, entry lag={lag}) #####")
    print(f"legs={legs}  trigger span {span[0].date()} .. {span[1].date()}")

    # 1. pattern + controls
    epi = declusters(sig_all, min_gap or h, px.index)
    ep = ret.loc[epi].values
    loc = local_control(px.index[valid.values], sig_all)
    rows = [summarize(ret.loc[sig_all].values, f"COND day-level (N={len(sig_all)})"),
            summarize(ep, f"COND episodes (N={len(epi)})"),
            summarize(ret[in_span].values, "CTRL-a own drift, same span"),
            summarize(ret[valid].values, "CTRL-b all days, full history"),
            summarize(ret.loc[loc].values, "CTRL-c local +/-126td ex-trigger"),
            summarize(naive.loc[sig_all].values, "COND day-level, lag=0 (no-lag)")]
    show(rows, "1. conditional vs controls")
    ctrl = ret[in_span].values
    if len(ep) > 1:
        se = np.sqrt(ep.var(ddof=1) / len(ep) + ctrl.var(ddof=1) / len(ctrl))
        wins = int((ep > 0).sum())
        print(f"  episode-vs-control diff = {100*(ep.mean()-ctrl.mean()):+.3f}%  "
              f"welch t = {(ep.mean()-ctrl.mean())/se:+.2f}   "
              f"bootstrap P(mean<=0) = {bootstrap_p_le0(ep):.3f}   "
              f"record {wins}-{len(ep)-wins}, sign p = {sign_test(wins, len(ep)):.4f}")
        print(f"  concentration: {cluster_note(epi, ep)}")

    # 2/3. era stability on episodes
    show(era_split(epi, ep), "2+3. episode era split (pre-2018 / 2018+)")
    show(era_split(sig_all, ret.loc[sig_all].values),
         "  day-level era split (for contrast)")
    if len(ep):
        print(f"  worst episode window: {100*ep.min():.2f}%  on "
              f"{epi[int(np.argmin(ep))].date()}   |  worst day-level: "
              f"{100*ret.loc[sig_all].min():.2f}%")
        print("  episode dates:", ", ".join(str(d.date()) for d in epi))

    # 4. threshold sensitivity
    if variants:
        vr = []
        for lbl, m in variants.items():
            s = px.index[m.reindex(px.index, fill_value=False).values
                         & valid.values]
            if len(s) == 0:
                vr.append({"label": lbl, "n": 0})
                continue
            e = declusters(s, min_gap or h, px.index)
            r = summarize(ret.loc[e].values, f"{lbl} (episodes)")
            r["n_days"] = len(s)
            vr.append(r)
        show(vr, "4. threshold sensitivity (episode level)")

    # 5. cost sanity
    n_legs = len(legs)
    rt = cost_bps * n_legs
    if len(ep):
        edge_bps = 100 * ep.mean() * 100
        print(f"\n5. cost: {n_legs} leg(s) x ~{cost_bps} bps round trip = "
              f"{rt} bps. episode mean {100*ep.mean():.3f}% = {edge_bps:.1f} "
              f"bps -> {edge_bps/rt:.1f}x cost (need >=5x)")

    # 6. event in window
    fl = event_in_window(epi, px.index, h, lag, event_kinds)
    show([summarize(ep[fl], f"{'+'.join(event_kinds)} IN window (N={int(fl.sum())})"),
          summarize(ep[~fl], f"{'+'.join(event_kinds)} OUT (N={int((~fl).sum())})")],
         f"6. {'+'.join(event_kinds)}-in-hold split (episodes)")


# ---------------------------------------------------------------------------
# stage C2: develop the survivors
# ---------------------------------------------------------------------------
def horizon_scan(px: pd.DataFrame, dates: pd.DatetimeIndex,
                 legs: list[tuple[str, float]],
                 hs: tuple[int, ...] = (1, 2, 3, 5, 10),
                 lag: int = 1, min_gap: int | None = None) -> list[dict]:
    """Episode-level summary across horizons: where does the edge actually
    concentrate? The pitched horizon is chosen FROM this, never assumed."""
    rows = []
    for h in hs:
        ret = vehicle_ret(px, legs, h, lag)
        valid = ret.dropna().index
        t = pd.DatetimeIndex(dates).intersection(valid)
        epi = declusters(t, min_gap or h, valid)
        r = summarize(ret.loc[epi].values, f"h={h}")
        if r["n"]:
            base = ret.loc[valid]
            r["ctl_all_days_pct"] = round(100 * base.mean(), 3)
            r["edge_pct"] = round(r["mean_pct"] - 100 * base.mean(), 3)
        rows.append(r)
    return rows


def episode_paths(px: pd.DataFrame, dates: pd.DatetimeIndex,
                  legs: list[tuple[str, float]], h: int,
                  lag: int = 1) -> pd.DataFrame:
    """Day-by-day cumulative return of each episode from the entry close
    (rows = episodes, columns = day 1..h). Feed the losers into
    `what_kills_it` so it quotes a number instead of a generic risk."""
    idx = px.index
    pos = pd.Series(range(len(idx)), index=idx)
    out = {}
    for d in pd.DatetimeIndex(dates):
        p = pos.get(d)
        if p is None or p + lag + h >= len(idx):
            continue
        path = None
        for tkr, w in legs:
            c = px[tkr].values
            entry = c[p + lag]
            leg = w * (c[p + lag + 1: p + lag + h + 1] / entry - 1.0)
            path = leg if path is None else path + leg
        out[d] = path
    return pd.DataFrame(out, index=range(1, h + 1)).T


# ---------------------------------------------------------------------------
# watchlist (parked near-misses that carry across mornings)
# ---------------------------------------------------------------------------
def load_watchlist(path: Path | None = None) -> dict:
    """data/pitch_watchlist.json: near-misses parked with the number that
    turns them on. Absent file = empty list, never an error."""
    p = Path(path) if path else WATCHLIST_PATH
    if not p.exists():
        return {"entries": []}
    d = json.loads(p.read_text(encoding="utf-8"))
    d.setdefault("entries", [])
    return d


def save_watchlist(d: dict, path: Path | None = None) -> None:
    p = Path(path) if path else WATCHLIST_PATH
    p.write_text(json.dumps(d, indent=1), encoding="utf-8")
