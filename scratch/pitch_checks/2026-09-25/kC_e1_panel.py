"""E1 panel: reuse 2026-09-21 kB_common (LIQ universe, earnings calendar, r5/r21/r63
ranks, 252d SPY beta) rebuilt through the 2026-09-24 close into today's folder, plus
SMH beta and print timing coverage. Import this module; run it to print coverage."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "2026-09-21"))
sys.path.insert(0, str(HERE.parents[2]))
import kB_common as kc  # noqa: E402
from pitch_lab import *  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

kc.CACHE = HERE / "_kC_e1_panel.pkl"
SEMIS = ["MU", "NVDA", "AMD", "AVGO", "INTC", "QCOM", "TXN", "AMAT", "LRCX", "KLAC", "MCHP", "ADI",
         "NXPI", "MRVL", "ON", "MPWR", "TSM", "ASML", "SWKS", "QRVO", "TER", "WDC", "STX", "SMCI", "ARM"]


def get_panel():
    P = kc.build_panel()
    if "beta_smh" not in P:
        mp = pd.read_parquet(ROOT / "data" / "master_prices.parquet", columns=["ticker", "date", "Close"])
        s = mp[mp.ticker == "SMH"].copy()
        s["date"] = pd.to_datetime(s["date"])
        smh = s.drop_duplicates("date").set_index("date")["Close"].sort_index().reindex(P["C"].index)
        P["C"]["SMH"] = smh
        ret = P["C"].pct_change(fill_method=None)
        x = ret["SMH"]
        cols = [c for c in SEMIS if c in P["C"].columns]
        mx = ret[cols].rolling(252, min_periods=200).mean()
        my = x.rolling(252, min_periods=200).mean()
        mxy = ret[cols].mul(x, axis=0).rolling(252, min_periods=200).mean()
        vy = x.rolling(252, min_periods=200).var(ddof=0)
        P["beta_smh"] = (mxy - mx.mul(my, axis=0)).div(vy, axis=0)
        pd.to_pickle(P, kc.CACHE)
    return P


def timing():
    e = pd.read_parquet(ROOT / "data" / "earnings_calendar.parquet", columns=["ticker", "date", "timeOfTheDay"])
    e["date"] = pd.to_datetime(e["date"])
    return e


if __name__ == "__main__":
    P = get_panel()
    C = P["C"]
    print("panel", C.shape, C.index[0].date(), C.index[-1].date(), "LIQ singles", len(kc.LIQ),
          "in panel", len([t for t in kc.LIQ if t in C.columns]))
    d = C.index[-1]
    for t in ["MU", "SMH"]:
        if t in P["r63"].columns:
            print(t, d.date(), "close", round(C[t].iloc[-1], 2), "r5", round(P["r5"][t].iloc[-1], 1),
                  "r21", round(P["r21"][t].iloc[-1], 1), "r63", round(P["r63"][t].iloc[-1], 1),
                  "63d ret", round(100 * (C[t].iloc[-1] / C[t].iloc[-64] - 1), 2))
    e = timing()
    print("timeOfTheDay counts:", e.timeOfTheDay.value_counts(dropna=False).head(6).to_dict())
    print("MU prints since 2000 with prices:", int(((e.ticker == "MU") & (e.date >= "2000-01-01")).sum()))
    live = [t for t in kc.LIQ if t in C.columns and P["r63"][t].iloc[-1] <= 5 and P["r5"][t].iloc[-1] >= 70]
    print("LIQ names in the E1 state on", d.date(), ":", live)
