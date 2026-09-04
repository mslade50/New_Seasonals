"""C15 -- intraday shape of the ME-0 session, and the RAW-price overnight.

First use of data/intraday (15min, 2003+, ET, UNADJUSTED) in a pitch check.
Bars run 09:30..15:45 inclusive (26 per session); the 15:45 bar covers
15:45-16:00, so its close is the last print of the session.

Four objects:
 A. last-hour (15:00 -> close) return on ME-0 vs every other session.  If the
    closing-auction flow story is real the last hour must be distinguishable.
 B. last-hour VOLUME share on ME-0 vs other sessions.  Elevated volume with an
    UNCHANGED return is absorption without impact, which falsifies the
    price-impact premise directly.
 C. the RAW overnight (unadjusted close -> unadjusted next open) against the
    ADJUSTED overnight from master_prices.  The gap is the dividend accrual,
    which is what an adjusted-basis month-end study silently books as alpha.
 D. regression of the overnight on the ME-0 last-hour move.  A reversal
    mechanism needs a NEGATIVE slope, and it needs to be MORE negative on ME-0
    than on the all-session control.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pitch_lab import *  # noqa: E402,F403
from pitch_lab import load_prices, summarize, show, sign_test  # noqa: E402
import intraday_data as idl  # noqa: E402

VEH = ["SPY", "IWM", "QQQ"]


def month_end_dates(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    ym = pd.Series(idx.year * 100 + idx.month, index=idx)
    return pd.DatetimeIndex(ym.groupby(ym.values).apply(lambda s: s.index[-1]).values)


def session_frame(tkr: str) -> pd.DataFrame:
    """One row per session: open, p1500, close, vol_total, vol_lasthour."""
    b = idl.get_intraday(tkr)
    if b.empty:
        return pd.DataFrame()
    b = b.copy()
    b["d"] = b["ts"].dt.normalize()
    b["t"] = b["ts"].dt.time
    rows = []
    for d, g in b.groupby("d", sort=True):
        g = g.sort_values("ts")
        tt = list(g["t"])
        if len(g) < 20:
            continue                      # half day / broken session
        o = float(g["open"].iloc[0])
        c = float(g["close"].iloc[-1])
        # price at 15:00 = OPEN of the 15:00 bar
        m = g["t"].astype(str) == "15:00:00"
        if not m.any():
            continue
        p15 = float(g.loc[m, "open"].iloc[0])
        lh = g[g["ts"].dt.hour >= 15]
        rows.append({"date": d, "o": o, "p15": p15, "c": c,
                     "vol": float(g["volume"].sum()),
                     "vol_lh": float(lh["volume"].sum()),
                     "first_t": str(tt[0]), "last_t": str(tt[-1])})
    f = pd.DataFrame(rows).set_index("date").sort_index()
    f["r_open_1500"] = f["p15"] / f["o"] - 1.0
    f["r_lasthour"] = f["c"] / f["p15"] - 1.0
    f["r_intraday"] = f["c"] / f["o"] - 1.0
    f["vol_share_lh"] = f["vol_lh"] / f["vol"]
    f["raw_on"] = f["o"].shift(-1) / f["c"] - 1.0     # raw close -> raw next open
    return f


def main() -> None:
    daily = load_prices(VEH)
    print("=" * 78)
    print("C15  intraday shape of the ME-0 session (15min cache, first pitch use)")
    print("=" * 78)

    for t in VEH:
        f = session_frame(t)
        if f.empty:
            print(f"{t}: NO INTRADAY DATA -- cannot grade")
            continue
        d = daily[t]
        me = month_end_dates(d.index)
        me = pd.DatetimeIndex([x for x in me if x in f.index])
        is_me = f.index.isin(me)

        print(f"\n----- {t}: {len(f)} sessions {f.index[0].date()} .. "
              f"{f.index[-1].date()};  {int(is_me.sum())} of them ME-0 -----")
        print(f"  bar grid: first {f['first_t'].mode()[0]}  last "
              f"{f['last_t'].mode()[0]}")

        # ---- A. last hour ------------------------------------------------
        show([summarize(f.loc[is_me, "r_lasthour"].values, "ME-0 last hour"),
              summarize(f.loc[~is_me, "r_lasthour"].values, "other sessions"),
              summarize(f.loc[is_me, "r_open_1500"].values, "ME-0 open->15:00"),
              summarize(f.loc[~is_me, "r_open_1500"].values, "other open->15:00"),
              summarize(f.loc[is_me, "r_intraday"].values, "ME-0 open->close"),
              summarize(f.loc[~is_me, "r_intraday"].values, "other open->close")],
             f"A. {t} intraday decomposition")
        a, b = f.loc[is_me, "r_lasthour"], f.loc[~is_me, "r_lasthour"]
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        print(f"  last-hour MEAN diff {100*(a.mean()-b.mean()):+.4f}%  "
              f"welch t {(a.mean()-b.mean())/se:+.2f}")
        print(f"  last-hour SD  ME-0 {100*a.std(ddof=1):.4f}%  vs other "
              f"{100*b.std(ddof=1):.4f}%  ratio {a.std(ddof=1)/b.std(ddof=1):.3f}")
        print(f"  last-hour |move| ME-0 {100*a.abs().mean():.4f}%  vs other "
              f"{100*b.abs().mean():.4f}%  ratio {a.abs().mean()/b.abs().mean():.3f}")

        # ---- B. volume ----------------------------------------------------
        va, vb = f.loc[is_me, "vol_share_lh"], f.loc[~is_me, "vol_share_lh"]
        rv = f["vol"] / f["vol"].rolling(21, min_periods=10).median()
        print(f"B. last-hour VOLUME share  ME-0 {100*va.mean():.2f}%  vs other "
              f"{100*vb.mean():.2f}%   (ratio {va.mean()/vb.mean():.3f})")
        print(f"   whole-session volume vs its own 21d median: ME-0 "
              f"{rv[is_me].mean():.3f}x  vs other {rv[~is_me].mean():.3f}x")

        # ---- C. raw vs adjusted overnight ---------------------------------
        adj_on = (d["Open"].shift(-1) / d["Close"] - 1.0)
        j = pd.concat({"raw": f["raw_on"], "adj": adj_on}, axis=1).dropna()
        jm = j.loc[j.index.isin(me)]
        print(f"C. overnight, RAW (unadjusted intraday) vs ADJUSTED "
              f"(master_prices), 2003+ overlap N={len(j)}")
        show([summarize(jm["raw"].values, f"ME-0 RAW overnight (N={len(jm)})"),
              summarize(jm["adj"].values, f"ME-0 ADJ overnight (N={len(jm)})"),
              summarize(j["raw"].values, "all sessions RAW"),
              summarize(j["adj"].values, "all sessions ADJ")],
             f"   {t} raw vs adjusted overnight")
        ex_raw = 100 * 100 * (jm["raw"].mean() - j["raw"].mean())
        ex_adj = 100 * 100 * (jm["adj"].mean() - j["adj"].mean())
        print(f"   ME-0 EXCESS over unconditional: RAW {ex_raw:+.2f} bps  vs "
              f"ADJ {ex_adj:+.2f} bps   -> dividend/adjustment share "
              f"{100*(ex_adj-ex_raw)/ex_adj if ex_adj else float('nan'):.0f}%")
        w = int((jm["raw"] > 0).sum())
        base = float((j["raw"] > 0).mean())
        print(f"   RAW record {w}-{len(jm)-w} ({100*w/len(jm):.1f}%) vs own base "
              f"{100*base:.1f}%,  sign p = {sign_test(w, len(jm), p=base):.4f}")
        print(f"   RAW cost multiple at 5 bps: {abs(ex_raw)/5:.2f}x")

        # ---- D. reversal regression --------------------------------------
        k = pd.concat({"lh": f["r_lasthour"], "on": f["raw_on"]}, axis=1).dropna()
        km = k.loc[k.index.isin(me)]
        for lbl, s in [("ME-0 only", km), ("ALL sessions (control)", k)]:
            x, y = s["lh"].values, s["on"].values
            if len(x) < 5:
                continue
            sl, ic = np.polyfit(x, y, 1)
            yh = sl * x + ic
            ss = ((y - yh) ** 2).sum()
            r2 = 1 - ss / ((y - y.mean()) ** 2).sum()
            sse = np.sqrt(ss / (len(x) - 2) / ((x - x.mean()) ** 2).sum())
            print(f"D. {lbl:24s} N={len(x):5d}  slope {sl:+.4f}  "
                  f"t {sl/sse:+.2f}  R2 {r2:.4f}")

        # E. does the ME-0 last hour predict its OWN next-open in the way the
        #    story needs, split by the SIGN of the last-hour move?
        up = km["lh"] > 0
        show([summarize(km.loc[up, "on"].values, f"ME-0 last hour UP (N={int(up.sum())})"),
              summarize(km.loc[~up, "on"].values,
                        f"ME-0 last hour DOWN (N={int((~up).sum())})")],
             f"E. {t} overnight conditioned on the sign of the ME-0 last hour")


if __name__ == "__main__":
    main()
