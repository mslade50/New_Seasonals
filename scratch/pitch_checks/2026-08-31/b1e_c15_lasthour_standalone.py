"""C15 round 2 -- the ME-0 LAST HOUR as a standalone short, which is the only
cell either candidate produced that is distinguishable inside its own window.

SPY's ME-0 15:00->close pays -0.065% against +0.004% on other sessions
(welch t -2.52); IWM -0.128% vs +0.009% (t -4.35).  Before that is a trade it
owes: era split, month-of-year (today is an August turn), midterm split, the
month-position placebo ladder, a cost bar, and the reference class.

Entry would be a 15:00 market order, exit MOC.  Round trip ~4 bps: one
non-auction market order into a liquid ETF plus one MOC.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pitch_lab import *  # noqa: E402,F403
from pitch_lab import load_prices, summarize, show, sign_test  # noqa: E402
import intraday_data as idl  # noqa: E402

VEH = ["SPY", "IWM", "QQQ", "XLK", "XLF", "EFA", "EEM", "XLE", "XLV", "XLU"]
COST_BPS = 4.0


def me_dates(idx):
    ym = pd.Series(idx.year * 100 + idx.month, index=idx)
    return pd.DatetimeIndex(
        ym.groupby(ym.values).apply(lambda s: s.index[-1]).values)


def lasthour(tkr: str) -> pd.Series:
    b = idl.get_intraday(tkr)
    if b.empty:
        return pd.Series(dtype=float)
    b = b.copy()
    b["d"] = b["ts"].dt.normalize()
    out = {}
    for d, g in b.groupby("d", sort=True):
        if len(g) < 20:
            continue
        g = g.sort_values("ts")
        m = g["ts"].dt.strftime("%H:%M") == "15:00"
        if not m.any():
            continue
        p15 = float(g.loc[m, "open"].iloc[0])
        out[d] = float(g["close"].iloc[-1]) / p15 - 1.0
    return pd.Series(out).sort_index()


def main() -> None:
    daily = load_prices(VEH)
    print("=" * 78)
    print("C15 standalone: SHORT the ME-0 last hour (15:00 -> close)")
    print("=" * 78)

    eff, var, rows = [], [], []
    for t in VEH:
        lh = lasthour(t)
        if lh.empty or t not in daily:
            print(f"{t}: no intraday")
            continue
        me = me_dates(daily[t].index)
        me = pd.DatetimeIndex([d for d in me if d in lh.index])
        s = lh.reindex(me).dropna()
        other = lh.drop(me, errors="ignore")
        # SHORT: pnl = -return
        ex = -(s.mean() - other.mean())
        se = np.sqrt(s.var(ddof=1) / len(s) + other.var(ddof=1) / len(other))
        eff.append(ex)
        var.append(se ** 2)
        rows.append({"ticker": t, "n": len(s),
                     "me0_lh_bp": round(1e4 * s.mean(), 2),
                     "other_bp": round(1e4 * other.mean(), 2),
                     "SHORT_excess_bp": round(1e4 * ex, 2),
                     "t": round(ex / se, 2),
                     "short_win": round(100 * (s < 0).mean(), 1)})
    df = pd.DataFrame(rows).sort_values("SHORT_excess_bp", ascending=False)
    print("\n--- REFERENCE CLASS (run before round 2) ---")
    print(df.to_string(index=False))
    eff, var = np.array(eff), np.array(var)
    w = 1 / var
    fe = float((w * eff).sum() / w.sum())
    Q = float((w * (eff - fe) ** 2).sum())
    dfree = len(eff) - 1
    from scipy.stats import chi2
    print(f"  fixed-effect common SHORT excess {1e4*fe:+.2f} bps "
          f"(t {fe/np.sqrt(1/w.sum()):+.2f}); Cochran Q {Q:.2f}/{dfree} df "
          f"p {1-chi2.cdf(Q, dfree):.4f}; I2 "
          f"{max(0, 100*(Q-dfree)/Q):.1f}%")

    for t in ("SPY", "IWM"):
        lh = lasthour(t)
        me = me_dates(daily[t].index)
        me = pd.DatetimeIndex([d for d in me if d in lh.index])
        s = -lh.reindex(me).dropna()          # SHORT pnl, fractions
        other = -lh.drop(me, errors="ignore")
        base = other.mean()
        print(f"\n===== {t} SHORT the ME-0 last hour  (N={len(s)}, "
              f"{s.index[0].date()}..{s.index[-1].date()}) =====")
        out = []
        for lbl, m in [("ALL ME-0", np.ones(len(s), bool)),
                       ("pre-2013", s.index.year < 2013),
                       ("2013+", s.index.year >= 2013),
                       ("2018+", s.index.year >= 2018),
                       ("2020+", s.index.year >= 2020),
                       ("MIDTERM", s.index.year % 4 == 2),
                       ("non-midterm", s.index.year % 4 != 2),
                       ("AUGUST", s.index.month == 8),
                       ("AUG x MIDTERM (live)",
                        (s.index.month == 8) & (s.index.year % 4 == 2))]:
            r = summarize(s[m].values, lbl)
            if r["n"]:
                r["excess_bp"] = round(1e4 * (r["mean_pct"] / 100 - base), 2)
                r["x_cost"] = round(abs(r["excess_bp"]) / COST_BPS, 2)
            out.append(r)
        show(out, f"{t} era / midterm / month splits")
        v = s.values
        wct = int((v > 0).sum())
        bp = float((other > 0).mean())
        print(f"  record {wct}-{len(v)-wct} vs own base {100*bp:.1f}%, sign p "
              f"{sign_test(wct, len(v), p=bp):.4f}")
        net = v - COST_BPS / 1e4
        print(f"  NET of {COST_BPS} bps: {1e4*net.mean():+.2f} bps/trade, cum "
              f"{100*net.sum():+.2f}pp over {len(v)} trades")
        # month-of-year scan
        mm = [{"m": pd.Timestamp(2020, k, 1).strftime("%b"),
               "n": int((s.index.month == k).sum()),
               "bp": round(1e4 * s[s.index.month == k].mean(), 2)}
              for k in range(1, 13)]
        print("  by month (bps, SHORT):",
              ", ".join(f"{d['m']} {d['bp']:+.1f}" for d in mm))

        # placebo ladder on month position
        idx = lh.index
        pos = pd.Series(range(len(idx)), index=idx)
        mep = [int(pos[d]) for d in me]
        lad = []
        for k in range(-5, 4):
            d = idx[[p + k for p in mep if 0 <= p + k < len(idx)]]
            r = summarize(-lh.reindex(d).dropna().values,
                          f"ME{k:+d}" if k else "ME-0 (TRUE)")
            if r["n"]:
                r["excess_bp"] = round(1e4 * (r["mean_pct"] / 100 - base), 2)
            lad.append(r)
        show(lad, f"{t} SHORT-last-hour placebo ladder")
        o = sorted([(x["label"], x.get("excess_bp", -1e9)) for x in lad],
                   key=lambda z: -z[1])
        print(f"  TRUE anchor ranks {[i for i,(l,_) in enumerate(o) if 'TRUE' in l][0]+1}"
              f" of {len(o)}: {[l for l,_ in o]}")


if __name__ == "__main__":
    main()
