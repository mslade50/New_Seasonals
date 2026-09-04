"""Follow-ups on the trend prototype: robustness + fragility timing + book fit."""
from pathlib import Path
import numpy as np
import pandas as pd

from tf_backtest import (ROOT, UNIVERSE, BONDS, COST_PER_SIDE,
                         load_prices, build_signals, base_weights, stats, fmt)

pd.set_option("future.no_silent_downcasting", True)


def run_lagged(px, irx, sig, lag=1):
    """lag=1 is the base (decide t, hold t+1); lag=2 = one extra month delay."""
    m = px.resample("ME").last()
    aret = m.pct_change()
    rf = (irx.resample("ME").last() / 100.0).reindex(aret.index).ffill() / 12.0
    base = base_weights(px, sig)
    on = sig.fillna(False).astype(float)
    w = base * on
    w_held = w.shift(lag).fillna(0.0)
    port = (w_held * aret).sum(axis=1) + (1 - w_held.sum(axis=1)) * rf.shift(1).fillna(0)
    cost = (w - w.shift(1)).abs().sum(axis=1).shift(lag).fillna(0.0) * COST_PER_SIDE
    net = port - cost
    n_elig = sig.notna().sum(axis=1)
    start = n_elig[n_elig >= 3].index.min()
    return net[net.index >= start], rf


def main():
    px, irx = load_prices()
    sigs = build_signals(px)
    sig = sigs["combo"]

    r1, rf = run_lagged(px, irx, sig, lag=1)
    r2, _ = run_lagged(px, irx, sig, lag=2)
    print("EXECUTION-LAG ROBUSTNESS (combo L/F invvol, net)")
    print(fmt(stats(r1, rf, "base (trade at signal close)")))
    print(fmt(stats(r2, rf, "1 FULL MONTH delayed exec")))

    # t-stat of monthly excess return (monthly is already the cluster unit)
    ex = (r1 - rf.reindex(r1.index)).dropna()
    t = ex.mean() / ex.std() * np.sqrt(len(ex))
    print(f"\nExcess-return t-stat (monthly, N={len(ex)}): t = {t:.2f}, "
          f"mean {ex.mean()*100:+.3f}%/mo")

    # ---- asset-group attribution ----
    m = px.resample("ME").last()
    aret = m.pct_change()
    base = base_weights(px, sig)
    w_held = (base * sig.fillna(False).astype(float)).shift(1).fillna(0.0)
    contrib = (w_held * aret).fillna(0.0)
    contrib = contrib[contrib.index >= r1.index.min()]
    groups = {"equities": ["SPY","QQQ","IWM","EFA","EEM","FXI","VNQ"],
              "bonds": BONDS, "commods": ["GLD","SLV","DBC","USO"], "dollar": ["UUP"]}
    print("\nASSET-GROUP CONTRIBUTION (avg %/mo, full sample and 2020+):")
    for g, tks in groups.items():
        c = contrib[tks].sum(axis=1)
        print(f"  {g:<9} full {c.mean()*100:+.3f}%/mo   2020+ {c[c.index>='2020'].mean()*100:+.3f}%/mo")
    print("  per-asset full-sample avg %/mo:")
    for tk in UNIVERSE:
        print(f"    {tk:<4} {contrib[tk].mean()*100:+.3f}")

    # ---- book series, clean window ----
    tr = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
    tr = tr.dropna(subset=["Exit Date", "PnL_flat_750k"])
    book = (tr.set_index("Exit Date")["PnL_flat_750k"]
              .groupby(pd.Grouper(freq="ME")).sum() / 750_000.0)
    lo, hi = "2003-01-31", "2026-06-30"   # complete months with book activity
    both = pd.DataFrame({"sleeve": r1, "book": book}).loc[lo:hi].dropna()
    print(f"\nCORR, clean window {lo}..{hi} N={len(both)}: "
          f"{both['sleeve'].corr(both['book']):+.3f}")
    dn = both[both["book"] < 0]
    print(f"  book<0 months N={len(dn)}: corr {dn['sleeve'].corr(dn['book']):+.3f}, "
          f"sleeve {dn['sleeve'].mean()*100:+.2f}%/mo, hit {(dn['sleeve']>0).mean()*100:.0f}%")

    # ---- combined book + sleeve (same $750k, sleeve as overlay on idle cash) ----
    print("\nBOOK vs BOOK+SLEEVE (additive monthly returns on flat $750k):")
    for lbl, s in [("book alone", both["book"]),
                   ("book + sleeve", both["book"] + both["sleeve"]),
                   ("book + 2x sleeve", both["book"] + 2*both["sleeve"])]:
        cum = s.cumsum()
        mdd = (cum - cum.cummax()).min()
        print(f"  {lbl:<17} mean {s.mean()*100:+.2f}%/mo  vol {s.std()*np.sqrt(12)*100:5.2f}%ann  "
              f"worst mo {s.min()*100:+.2f}%  maxDD(add) {mdd*100:+.2f}%  "
              f"Sharpe~ {s.mean()/s.std()*np.sqrt(12):.2f}")

    # ---- fragility lead/lag ----
    frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
    ma10 = frag["63d"].rolling(10, min_periods=1).mean()
    m_mean = ma10.groupby(pd.Grouper(freq="ME")).mean()
    flag = (m_mean >= 50)
    flag = flag[flag.index >= "2016-07-31"]
    sl = r1.reindex(flag.index)
    bk = book.reindex(flag.index)
    spy_m = px["SPY"].resample("ME").last().pct_change().reindex(flag.index)
    print("\nFRAGILITY LEAD/LAG (month-mean MA10>=50, 2016-07+):")
    for k in [0, 1, 2, 3]:
        f_k = flag.shift(k).fillna(False).astype(bool)
        s_hi, s_lo = sl[f_k], sl[~f_k]
        b_hi = bk[f_k]
        spy_hi = spy_m[f_k]
        print(f"  t+{k}: N={f_k.sum():>2}  sleeve {s_hi.mean()*100:+.2f}%/mo (vs {s_lo.mean()*100:+.2f} else, "
              f"hit {(s_hi>0).mean()*100:.0f}%)  book {b_hi.mean()*100:+.2f}%/mo  SPY {spy_hi.mean()*100:+.2f}%/mo")
    # window: high-frag month OR any of following 3 months
    f_win = flag.copy()
    for k in [1, 2, 3]:
        f_win = f_win | flag.shift(k).fillna(False).astype(bool)
    print(f"  t..t+3 union: N={f_win.sum()}  sleeve {sl[f_win].mean()*100:+.2f}%/mo  "
          f"book {bk[f_win].mean()*100:+.2f}%/mo  SPY {spy_m[f_win].mean()*100:+.2f}%/mo")

    # per-month detail of the 16 concurrent high-frag months
    print("\n  concurrent high-frag month detail (sleeve / book / SPY %):")
    for dt in flag[flag].index:
        print(f"    {dt.strftime('%Y-%m')}: sleeve {sl.get(dt, np.nan)*100:+6.2f}  "
              f"book {bk.get(dt, np.nan)*100:+6.2f}  SPY {spy_m.get(dt, np.nan)*100:+6.2f}")

    # position count / order profile
    on_ct = sig.fillna(False).sum(axis=1)
    w = base * sig.fillna(False).astype(float)
    chg = ((w > 0) != (w.shift(1) > 0)).sum(axis=1)
    print(f"\nEXECUTION PROFILE: avg positions {on_ct[on_ct.index>=r1.index.min()].mean():.1f}, "
          f"avg on/off flips per month {chg[chg.index>=r1.index.min()].mean():.1f}")


if __name__ == "__main__":
    main()
