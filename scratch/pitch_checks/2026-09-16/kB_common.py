"""Shared helpers for the kB_ checks (c3 SPY post-decision, c2 XLU, c4 HYG).

Alignment used everywhere here: the anchor is the DECISION session itself
(position p in the ticker's own valid index); entry = close p (the MOC order
placed on decision day), exit = close p+h. Gates are read on the EVE close p-1.
This is pitch_lab's lag=1 convention with the signal day D = eve = p-1:
fwd_lag(s, h, lag=1) at D = c[p+h]/c[p]-1. Verified by printing 2025-12-10.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

TNX = "^TNX"


def ser(t: str) -> pd.Series:
    return close_panel([t])[t].dropna()


def decisions() -> pd.DatetimeIndex:
    return pd.DatetimeIndex(load_events(["fomc_decision"])["date"])


def vix_exp() -> set:
    return set(pd.DatetimeIndex(load_events(["vix_expiry"])["date"]))


def quad() -> pd.DatetimeIndex:
    return pd.DatetimeIndex(load_events(["quad_witching"])["date"])


def dec_pos(idx: pd.DatetimeIndex, hmax: int = 10, pre: int = 260) -> list:
    """(decision_date, p) where the decision date IS a session of idx."""
    out = []
    pos = pd.Series(range(len(idx)), index=idx)
    for d in decisions():
        p = pos.get(d)
        if p is None or p < pre or p + hmax >= len(idx):
            continue
        out.append((d, int(p)))
    return out


def tdom(idx: pd.DatetimeIndex) -> np.ndarray:
    ym = pd.Series(idx.year * 100 + idx.month, index=idx)
    return ym.groupby(ym.values).cumcount().values + 1


def fwd_arr(c: np.ndarray, h: int) -> np.ndarray:
    r = np.full(len(c), np.nan)
    r[:len(c) - h] = c[h:] / c[:-h] - 1.0
    return r


def tape_z10(s: pd.Series) -> pd.Series:
    """build_pitch_state._metrics_for convention: 10d ret / (21d sd * sqrt10)."""
    vol21 = s.pct_change().rolling(21).std()
    return (s / s.shift(10) - 1.0) / (vol21 * np.sqrt(10))


def tdom_ctrl(idx, c, qs, h, exclude=None):
    """Per-entry matched control: mean h-return over ALL sessions sharing the
    entry's trading-day-of-month (excluding `exclude` positions)."""
    td = tdom(idx)
    r = fwd_arr(c, h)
    ok = ~np.isnan(r)
    if exclude is not None:
        ok[list(exclude)] = False
    means = {j: np.nanmean(r[(td == j) & ok]) for j in np.unique(td)}
    return np.array([means[td[q]] for q in qs])


def tdom_month_ctrl(idx, c, qs, h, exclude=None):
    td = tdom(idx)
    mo = idx.month.values
    r = fwd_arr(c, h)
    ok = ~np.isnan(r)
    if exclude is not None:
        ok[list(exclude)] = False
    out = []
    for q in qs:
        m = (td == td[q]) & (mo == mo[q]) & ok
        out.append(np.nanmean(r[m]) if m.sum() else np.nan)
    return np.array(out)


def placebo(c, ps, h, ks=range(-5, 6)):
    rows = []
    for k in ks:
        v = np.array([c[p + k + h] / c[p + k] - 1.0 for p in ps
                      if 0 <= p + k and p + k + h < len(c)])
        rows.append({"k": k, "n": len(v), "mean_pct": 100 * v.mean(),
                     "hit": 100 * (v > 0).mean()})
    df = pd.DataFrame(rows)
    df["rank"] = df["mean_pct"].rank(ascending=False).astype(int)
    r0 = int(df.loc[df.k == 0, "rank"].iloc[0])
    return df, r0


def line(label, v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return f"{label:<44} n=0"
    w = int((v > 0).sum())
    t = v.mean() / (v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 2 else np.nan
    return (f"{label:<44} n={len(v):3d} mean {100*v.mean():+.3f}% med "
            f"{100*np.median(v):+.3f}% {w}-{len(v)-w} sign p "
            f"{sign_test(w, len(v)):.3f} t {t:+.2f} worst {100*v.min():+.2f}%")


def event_state_cell(tkr: str, s: pd.Series, gates: dict, hs=(1, 2, 3, 5, 10),
                     headline: str | None = None, cost_bps: float = 3.0,
                     show_eps: bool = True) -> dict:
    """gates: label -> boolean Series on s.index (read at the EVE).
    Prints, per horizon: all-FOMC parent, each gated cell with tdom excess,
    the same state on ALL days (lag=1, not event-anchored) and on non-FOMC days.
    Returns {h: {label: vals}} for the headline."""
    idx, c = s.index, s.values
    dp = dec_pos(idx, hmax=max(hs) + 6, pre=252)
    ds = pd.DatetimeIndex([d for d, _ in dp])
    ps = np.array([p for _, p in dp])
    eve = ps - 1
    mid = ds.year % 4 == 2
    excl = set()
    for p in ps:
        excl.update(range(p - 1, p + 4))
    out = {}
    print(f"\n######## {tkr}: {len(ps)} decisions {ds[0].date()}..{ds[-1].date()}")
    for h in hs:
        v = np.array([c[p + h] / c[p] - 1.0 for p in ps])
        ra = fwd_arr(c, h)
        tc = tdom_ctrl(idx, c, ps, h, exclude=excl)
        print(f"\n--- {tkr} h={h} ---")
        print(line("parent: ALL decisions", v),
              f"| tdom exc {100*np.nanmean(v - tc):+.3f}pp")
        print(line("ctrl all days", ra))
        out[h] = {}
        for lbl, g in gates.items():
            gv = g.reindex(idx).fillna(False).values.astype(bool)
            m = gv[eve]
            vv = v[m]
            exc = 100 * np.nanmean((v - tc)[m]) if m.sum() else np.nan
            print(line(f"  FOMC & {lbl}", vv), f"| tdom exc {exc:+.3f}pp"
                  + (f" | mid {100*vv[mid[m]].mean():+.2f}% n{int(mid[m].sum())}"
                     if mid[m].sum() else ""))
            # the same state, any day (entry close D+1), declustered by h
            sd = idx[gv]
            sd = sd[sd < idx[-(h + 2)]]
            epi = declusters(sd, max(h, 5), idx)
            rr = pd.Series(fwd_lag(s, h, 1)).reindex(epi).values
            pset = set(ps.tolist())
            nf = np.array([idx.get_loc(d) + 1 not in pset for d in epi], dtype=bool)
            print(line(f"    state any-day episodes", rr),
                  "|", line("non-FOMC", rr[nf]).split("n=")[1][:60])
            out[h][lbl] = (ds[m], vv, (v - tc)[m])
        if headline is not None:
            m = gates[headline].reindex(idx).fillna(False).values.astype(bool)[eve]
            if m.sum() >= 1:
                df, r0 = placebo(c, ps[m], h)
                print(f"  PLACEBO [{headline}] k=0 rank {r0}/11: "
                      + " ".join(f"{k:+d}:{x:+.2f}" for k, x in zip(df.k, df.mean_pct)))
    if headline is not None and show_eps:
        m = gates[headline].reindex(idx).fillna(False).values.astype(bool)[eve]
        print(f"\n  episodes [{headline}]:")
        for i in np.where(m)[0]:
            p = ps[i]
            print(f"    {ds[i].date()} " + " ".join(
                f"h{h} {100*(c[p+h]/c[p]-1):+.2f}" for h in hs))
    return out


def verify_alignment(idx):
    pos = pd.Series(range(len(idx)), index=idx)
    p = pos[pd.Timestamp("2025-12-10")]
    print(f"ALIGN 2025-12-10: eve(gate)={idx[p-1].date()} entry close="
          f"{idx[p].date()} h=1 exit={idx[p+1].date()} h=2 exit={idx[p+2].date()}"
          f" h=3 exit={idx[p+3].date()}")
