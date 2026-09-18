"""Shared panel helpers for the k2 checker (C1 bank shock, C5 failed thrust)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

ETFS = {"CEF", "DBC", "DIA", "DX-Y.NYB", "EEM", "EFA", "EWJ", "EWZ", "FXI", "GDX",
        "GLD", "HYG", "IBB", "IEF", "IHI", "ITA", "ITB", "IWM", "IYR", "KRE", "LQD",
        "OIH", "QQQ", "SLV", "SMH", "SPY", "SVXY", "TLT", "UNG", "USO", "UUP", "UVXY",
        "VNQ", "XBI", "XHB", "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE",
        "XLU", "XLV", "XLY", "XME", "XOP", "XRT"}

SECTOR_ETF = {"Technology": "XLK", "Financial Services": "XLF", "Healthcare": "XLV",
              "Consumer Defensive": "XLP", "Consumer Cyclical": "XLY",
              "Utilities": "XLU", "Energy": "XLE", "Basic Materials": "XLB",
              "Industrials": "XLI", "Communication Services": "XLC",
              "Real Estate": "IYR"}


def tape_names() -> list[str]:
    t = json.loads((ROOT / "data" / "pitch_tape.json").read_text(encoding="utf-8"))
    return sorted(t["tickers"])


def single_names() -> dict[str, str]:
    sm = pd.read_parquet(ROOT / "data" / "sector_map.parquet")
    sec = dict(zip(sm["ticker"], sm["sector"]))
    out = {}
    for t in tape_names():
        if t in ETFS or t.startswith("^") or "=" in t or "." in t:
            continue
        out[t] = sec.get(t, "UNKNOWN")
    return out


def load_panels(tickers: list[str], start: str = "2000-01-01"):
    px = load_prices(sorted(set(tickers)))
    cal = px["SPY"].index
    cal = cal[cal >= pd.Timestamp(start)]
    P = {f: pd.DataFrame({t: px[t][f].reindex(cal) for t in px})
         for f in ["Open", "High", "Low", "Close"]}
    return px, cal, P


def derive(px: dict, cal: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    shock, atr, atrp, r5, ret1, gap = {}, {}, {}, {}, {}, {}
    for t, d in px.items():
        d = d.dropna(subset=["Close"])
        if len(d) < 30:
            continue
        a = pd.Series(np.asarray(wilder_atr(d["High"], d["Low"], d["Close"]),
                                 dtype=float), index=d.index)
        shock[t] = (d["Close"].diff() / a.shift(1)).reindex(cal)
        atr[t] = a.reindex(cal)
        atrp[t] = (a / d["Close"]).reindex(cal)
        r5[t] = pct_rank(d["Close"], 5).reindex(cal)
        ret1[t] = (d["Close"] / d["Close"].shift(1) - 1).reindex(cal)
        o = d["Open"].where(d["Open"] > 0)
        gap[t] = (o / d["Close"].shift(1) - 1).reindex(cal)
    return {k: pd.DataFrame(v) for k, v in
            dict(shock=shock, atr=atr, atrp=atrp, r5=r5, ret1=ret1, gap=gap).items()}


def rolling_beta(ret1: pd.DataFrame, bench: str, win: int = 252) -> pd.DataFrame:
    m = ret1[bench]
    var = m.rolling(win, min_periods=200).var()
    return pd.DataFrame({t: ret1[t].rolling(win, min_periods=200).cov(m) / var
                         for t in ret1.columns})


def fwd_panel(C: pd.DataFrame, h: int, lag: int = 1) -> pd.DataFrame:
    return C.shift(-(lag + h)) / C.shift(-lag) - 1.0


def lookup(panel: pd.DataFrame, dates, names) -> np.ndarray:
    ri = panel.index.get_indexer(pd.DatetimeIndex(dates))
    ci = panel.columns.get_indexer(list(names))
    return panel.to_numpy(dtype=float)[ri, ci]


def events_from_mask(mask: pd.DataFrame, names) -> pd.DataFrame:
    m = mask[list(names)].fillna(False).astype(bool)
    st = m.stack()
    st = st[st]
    return pd.DataFrame({"date": st.index.get_level_values(0),
                         "name": st.index.get_level_values(1)})


def date_series(ev: pd.DataFrame, col: str) -> pd.Series:
    e = ev.dropna(subset=[col])
    return e.groupby("date")[col].mean().sort_index()


def date_stats(ev: pd.DataFrame, col: str, cal: pd.DatetimeIndex, gap: int,
               label: str, ctrl: float | None = None) -> dict:
    s = date_series(ev, col)
    if len(s) == 0:
        return {"label": label, "n": 0}
    keep = declusters(s.index, gap, cal)
    v = s.loc[keep].values
    r = summarize(v, label)
    r["name_days"] = int(ev[col].notna().sum())
    r["dates"] = len(s)
    w = int((v > 0).sum())
    r["rec"] = f"{w}-{len(v) - w}"
    r["sign_p"] = round(sign_test(w, len(v)), 4)
    if ctrl is not None:
        r["ctrl_pct"] = round(100 * ctrl, 3)
        r["excess_pp"] = round(r["mean_pct"] - 100 * ctrl, 3)
    return r
