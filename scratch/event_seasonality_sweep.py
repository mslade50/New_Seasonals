"""Event-calendar strategy sweep — SPY/QQQ/IWM/TLT vs data/macro_events.csv.

Sections:
  A. Event-relative day grid: mean daily bps by td offset -5..+5, per event,
     full 2000+ plus era splits (2000-2012 / 2013-2019 / 2020+).
  B. FOMC window returns (pre-drift td-3..0, post-fade td+1..+3) by
     presidential cycle year; one observation per meeting (clustered).
  C. CPI/NFP overnight-into-release vs day-of close reaction, by era.
  D. Opex/quad-witching: expiry week vs post-expiry week, by month;
     September and December cells; midterm-year September.
  E. Jackson Hole: pre/at/post windows (N is tiny — descriptive only).
  F. Elections: pre/post windows, midterm vs presidential.
  G. Combos: pre-FOMC drift conditioned on trailing 5d run-up; Santa window
     conditioned on Dec-FOMC-day sign; turn-of-month with/without NFP.

Output: printed tables + CSVs in scratch/event_sweep_results/.
Run: python scratch/event_seasonality_sweep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from macro_calendar import event_dates, td_offset  # noqa: E402

OUT = ROOT / "scratch" / "event_sweep_results"
OUT.mkdir(exist_ok=True)

TICKERS = ["SPY", "QQQ", "IWM", "TLT"]
EVENTS = ["fomc_decision", "fomc_minutes", "cpi", "nfp", "ppi", "opex",
          "quad_witching", "jackson_hole"]


def load_prices() -> dict[str, pd.DataFrame]:
    mp = pd.read_parquet(ROOT / "data" / "master_prices.parquet",
                         columns=["ticker", "date", "Open", "Close"])
    out = {}
    for t in TICKERS:
        df = (mp[mp["ticker"] == t].set_index("date").sort_index()
              [["Open", "Close"]])
        df.index = pd.to_datetime(df.index).normalize()
        df = df[~df.index.duplicated(keep="last")]
        df = df[df.index >= "2000-01-01"]
        df["ret"] = df["Close"].pct_change()
        df["overnight"] = df["Open"] / df["Close"].shift(1) - 1
        df["intraday"] = df["Close"] / df["Open"] - 1
        out[t] = df
    return out


def tstat(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 2 else np.nan


def cell(x: pd.Series) -> dict:
    x = x.dropna()
    return {"mean_bps": round(x.mean() * 1e4, 1), "t": round(tstat(x), 2),
            "N": len(x), "hit": round((x > 0).mean(), 3) if len(x) else np.nan}


ERAS = [("2000-2012", "2000-01-01", "2012-12-31"),
        ("2013-2019", "2013-01-01", "2019-12-31"),
        ("2020+", "2020-01-01", "2027-12-31"),
        ("full", "2000-01-01", "2027-12-31")]


def section_a(px: dict) -> pd.DataFrame:
    rows = []
    for tkr in TICKERS:
        df = px[tkr]
        for ev in EVENTS:
            off = td_offset(df.index, ev)
            for era, lo, hi in ERAS:
                sl = df.loc[lo:hi]
                osl = off.loc[lo:hi]
                for k in range(-5, 6):
                    c = cell(sl.loc[osl == k, "ret"])
                    rows.append({"ticker": tkr, "event": ev, "era": era,
                                 "td": k, **c})
    return pd.DataFrame(rows)


def window_return(df: pd.DataFrame, anchor: pd.Timestamp,
                  a: int, b: int) -> float:
    """Cumulative close-to-close return from td offset a to b around the
    anchor's session (anchor mapped forward if not a session)."""
    idx = df.index
    p = idx.searchsorted(anchor)
    if p >= len(idx):
        return np.nan
    lo, hi = p + a, p + b
    if lo - 1 < 0 or hi >= len(idx):
        return np.nan
    return float(df["Close"].iloc[hi] / df["Close"].iloc[lo - 1] - 1)


def cycle_year(ts: pd.Timestamp) -> int:
    return ts.year % 4  # 0=election, 1=post, 2=midterm, 3=pre-election


def section_b(px: dict) -> pd.DataFrame:
    df = px["SPY"]
    rows = []
    for name, a, b in [("pre_drift(-3..0)", -3, 0), ("post_fade(+1..+3)", 1, 3)]:
        for dts, label in [(event_dates("fomc_decision"), "all")]:
            dts = dts[(dts >= df.index.min()) & (dts <= df.index.max())]
            w = pd.Series([window_return(df, d, a, b) for d in dts],
                          index=dts)
            rows.append({"window": name, "cycle": "all", **cell(w)})
            for cy in range(4):
                sub = w[[cycle_year(d) == cy for d in w.index]]
                rows.append({"window": name, "cycle": f"y{cy}", **cell(sub)})
            for era, lo, hi in ERAS[:3]:
                rows.append({"window": name, "cycle": era,
                             **cell(w.loc[lo:hi])})
    return pd.DataFrame(rows)


def section_c(px: dict) -> pd.DataFrame:
    rows = []
    for ev in ("cpi", "nfp"):
        dts = event_dates(ev)
        for tkr in ("SPY", "TLT"):
            df = px[tkr]
            off = td_offset(df.index, ev)
            day0 = off == 0
            for era, lo, hi in ERAS:
                sl = df.loc[lo:hi]
                d0 = day0.loc[lo:hi]
                rows.append({"event": ev, "ticker": tkr, "era": era,
                             "leg": "overnight", **cell(sl.loc[d0, "overnight"])})
                rows.append({"event": ev, "ticker": tkr, "era": era,
                             "leg": "intraday", **cell(sl.loc[d0, "intraday"])})
    return pd.DataFrame(rows)


def section_d(px: dict) -> pd.DataFrame:
    df = px["SPY"]
    dts = event_dates("opex")
    dts = dts[(dts >= df.index.min()) & (dts <= df.index.max())]
    rows = []
    frames = []
    for d in dts:
        frames.append({"date": d, "month": d.month,
                       "quad": d.month in (3, 6, 9, 12),
                       "midterm": cycle_year(d) == 2,
                       "expiry_wk": window_return(df, d, -4, 0),
                       "post_wk": window_return(df, d, 1, 5)})
    w = pd.DataFrame(frames)
    for leg in ("expiry_wk", "post_wk"):
        rows.append({"cut": "all months", "leg": leg, **cell(w[leg])})
        rows.append({"cut": "quad", "leg": leg, **cell(w.loc[w.quad, leg])})
        for mo in (3, 6, 9, 12):
            rows.append({"cut": f"month={mo}", "leg": leg,
                         **cell(w.loc[w.month == mo, leg])})
        rows.append({"cut": "Sep+midterm", "leg": leg,
                     **cell(w.loc[(w.month == 9) & w.midterm, leg])})
        rows.append({"cut": "non-quad months", "leg": leg,
                     **cell(w.loc[~w.quad, leg])})
    return pd.DataFrame(rows)


def section_e(px: dict) -> pd.DataFrame:
    rows = []
    for tkr in ("SPY", "TLT"):
        df = px[tkr]
        dts = event_dates("jackson_hole")
        dts = dts[(dts >= df.index.min()) & (dts <= df.index.max())]
        for name, a, b in [("into(-2..0)", -2, 0), ("keynote(0..0)", 0, 0),
                           ("after(+1..+5)", 1, 5)]:
            w = pd.Series([window_return(df, d, a, b) for d in dts], index=dts)
            rows.append({"ticker": tkr, "window": name, **cell(w)})
            rows.append({"ticker": tkr, "window": name + " 2013+",
                         **cell(w.loc["2013-01-01":])})
    return pd.DataFrame(rows)


def section_f(px: dict) -> pd.DataFrame:
    df = px["SPY"]
    dts = event_dates("election")
    dts = dts[(dts >= df.index.min()) & (dts <= df.index.max())]
    rows = []
    for name, a, b in [("pre(-5..0)", -5, 0), ("post(+1..+5)", 1, 5),
                       ("post(+1..+21)", 1, 21)]:
        w = pd.Series([window_return(df, d, a, b) for d in dts], index=dts)
        rows.append({"window": name, "cut": "all", **cell(w)})
        mid = w[[d.year % 4 == 2 for d in w.index]]
        pres = w[[d.year % 4 == 0 for d in w.index]]
        rows.append({"window": name, "cut": "midterm", **cell(mid)})
        rows.append({"window": name, "cut": "presidential", **cell(pres)})
    return pd.DataFrame(rows)


def section_g(px: dict) -> pd.DataFrame:
    df = px["SPY"]
    rows = []

    # G1: pre-FOMC drift conditioned on trailing 5d return percentile
    dts = event_dates("fomc_decision")
    dts = dts[(dts >= df.index.min()) & (dts <= df.index.max())]
    r5 = df["Close"].pct_change(5)
    pct = r5.rolling(252).rank(pct=True) * 100
    recs = []
    for d in dts:
        p = df.index.searchsorted(d)
        if p < 260 or p + 3 >= len(df):
            continue
        anchor_pct = pct.iloc[p - 4]  # known before the drift window starts
        recs.append({"date": d, "runup_pct": anchor_pct,
                     "pre": window_return(df, d, -3, 0),
                     "post": window_return(df, d, 1, 3)})
    g = pd.DataFrame(recs).dropna()
    for leg in ("pre", "post"):
        rows.append({"combo": f"G1 fomc {leg}", "cut": "runup>80",
                     **cell(g.loc[g.runup_pct > 80, leg])})
        rows.append({"combo": f"G1 fomc {leg}", "cut": "runup<20",
                     **cell(g.loc[g.runup_pct < 20, leg])})
        rows.append({"combo": f"G1 fomc {leg}", "cut": "mid",
                     **cell(g.loc[(g.runup_pct >= 20) & (g.runup_pct <= 80),
                                  leg])})

    # G2: Santa window (post-Dec-FOMC close -> year-end) by FOMC-day sign
    dec = [d for d in dts if d.month == 12]
    recs = []
    for d in dec:
        p = df.index.searchsorted(d)
        if p >= len(df):
            continue
        ye = df.index.searchsorted(pd.Timestamp(f"{d.year}-12-31"),
                                   side="right") - 1
        if ye <= p:
            continue
        recs.append({"year": d.year, "fomc_day": df["ret"].iloc[p],
                     "to_ye": float(df["Close"].iloc[ye]
                                    / df["Close"].iloc[p] - 1)})
    g2 = pd.DataFrame(recs)
    rows.append({"combo": "G2 santa", "cut": "fomc_day>0",
                 **cell(g2.loc[g2.fomc_day > 0, "to_ye"])})
    rows.append({"combo": "G2 santa", "cut": "fomc_day<0",
                 **cell(g2.loc[g2.fomc_day < 0, "to_ye"])})
    rows.append({"combo": "G2 santa", "cut": "all",
                 **cell(g2["to_ye"])})

    # G3: turn-of-month window (last session .. +3) with vs without NFP inside
    nfp = set(event_dates("nfp"))
    me = df.groupby([df.index.year, df.index.month]).tail(1).index
    recs = []
    for d in me:
        p = df.index.searchsorted(d)
        if p + 4 >= len(df):
            continue
        win = df.index[p: p + 4]
        recs.append({"date": d,
                     "has_nfp": bool(nfp & set(win)),
                     "tom": float(df["Close"].iloc[p + 3]
                                  / df["Close"].iloc[p - 1] - 1)})
    g3 = pd.DataFrame(recs)
    rows.append({"combo": "G3 TOM", "cut": "with NFP",
                 **cell(g3.loc[g3.has_nfp, "tom"])})
    rows.append({"combo": "G3 TOM", "cut": "no NFP",
                 **cell(g3.loc[~g3.has_nfp, "tom"])})
    return pd.DataFrame(rows)


def main() -> None:
    px = load_prices()
    for t in TICKERS:
        print(f"{t}: {px[t].index.min():%Y-%m-%d} .. "
              f"{px[t].index.max():%Y-%m-%d} ({len(px[t])} bars)")

    secs = {"A_day_grid": section_a(px), "B_fomc_cycle": section_b(px),
            "C_release_legs": section_c(px), "D_opex": section_d(px),
            "E_jackson_hole": section_e(px), "F_elections": section_f(px),
            "G_combos": section_g(px)}
    for name, frame in secs.items():
        frame.to_csv(OUT / f"{name}.csv", index=False)
        print(f"\n===== {name} ({len(frame)} rows) =====")
        if name == "A_day_grid":
            # print only the interesting slice: SPY full-era
            sl = frame[(frame.ticker == "SPY") & (frame.era == "full")]
            print(sl.pivot(index="td", columns="event",
                           values="mean_bps").to_string())
            print("t-stats:")
            print(sl.pivot(index="td", columns="event",
                           values="t").to_string())
        else:
            print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
