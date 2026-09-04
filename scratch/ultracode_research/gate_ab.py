"""Step-7 pre-work gates for the trend-following pilot.

GATE A: same-close vs next-open execution for the ex-bonds 12-ETF combo L/F.
GATE B: Jul-Sep / midterm-year dead-zone fill, sleeve vs book.
Plus re-verification of the two integration numbers (book corr, high-frag loss).

Engine is a fresh reimplementation of the tf_backtest.py rules with one
difference for the next-open leg: asset period returns are open-to-open at the
first trading day of each month (signal still computed on month-end close).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[2]

UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "FXI", "VNQ",
            "TLT", "IEF", "LQD", "HYG",
            "GLD", "SLV", "DBC", "USO", "UUP"]
BONDS = ["TLT", "IEF", "LQD", "HYG"]
EXB = [t for t in UNIVERSE if t not in BONDS]
COST_PER_SIDE = 0.0005
CAP = 0.20
LAST_FULL = pd.Timestamp("2026-06-30")   # exclude partial 2026-07


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    mp = pd.read_parquet(ROOT / "data" / "master_prices.parquet",
                         columns=["ticker", "date", "Open", "Close"])
    sub = mp[mp.ticker.isin(UNIVERSE + ["^IRX"])]
    close = sub.pivot(index="date", columns="ticker", values="Close").sort_index().astype("float64")
    openp = sub.pivot(index="date", columns="ticker", values="Open").sort_index().astype("float64")
    irx = close.pop("^IRX").ffill()
    openp = openp.drop(columns=["^IRX"])
    close = close[UNIVERSE]
    openp = openp[UNIVERSE]
    return close, openp, irx


def build_weights_and_signals(close: pd.DataFrame, universe: list[str]):
    px = close[universe]
    m = px.resample("ME").last()
    mom = m.shift(1) / m.shift(12) - 1.0
    ma10 = m - m.rolling(10).mean()
    elig = m.notna() & m.shift(12).notna()
    combo = ((mom > 0) & (ma10 > 0)).where(elig)          # NaN = ineligible
    vol63 = px.pct_change().rolling(63).std() * np.sqrt(252)
    vol_m = vol63.resample("ME").last().clip(lower=0.04)
    inv = (1.0 / vol_m).where(combo.notna()).fillna(0.0)
    tot = inv.sum(axis=1)
    base = inv.div(tot.replace(0, np.nan), axis=0).fillna(0.0).clip(upper=CAP)
    w = base * combo.fillna(False).astype(float)
    n_elig = combo.notna().sum(axis=1)
    start = n_elig[n_elig >= 3].index.min()
    return w, start


def run_sleeve(close: pd.DataFrame, openp: pd.DataFrame, irx: pd.Series,
               universe: list[str], mode: str) -> pd.DataFrame:
    """mode='close': close-to-close (prototype convention).
    mode='open': signal at month-end close, trade next trading day's open;
    period return = open(first td of month j+1)/open(first td of month j)-1
    attributed to holding month j."""
    w, start = build_weights_and_signals(close, universe)
    m = close[universe].resample("ME").last()
    if mode == "close":
        aret = m.pct_change()
    else:
        fo = openp[universe].resample("ME").first()   # open of first td of month
        aret = fo.shift(-1) / fo - 1.0
    rf = (irx.resample("ME").last() / 100.0).reindex(aret.index).ffill() / 12.0
    w_held = w.shift(1).fillna(0.0)
    port = (w_held * aret).sum(axis=1)
    net_exp = w_held.sum(axis=1)
    cash = (1.0 - net_exp) * rf.shift(1).fillna(0.0)
    turnover = (w - w.shift(1)).abs().sum(axis=1).shift(1).fillna(0.0)
    cost = turnover * COST_PER_SIDE
    out = pd.DataFrame({"gross": port + cash, "net": port + cash - cost,
                        "rf": rf, "turnover": turnover})
    out = out.loc[(out.index >= start) & (out.index <= LAST_FULL)]
    return out


def stat_line(r: pd.Series, rf: pd.Series, label: str) -> dict:
    r = r.dropna()
    curve = (1 + r).cumprod()
    yrs = len(r) / 12.0
    cagr = curve.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(12)
    ex = r - rf.reindex(r.index).fillna(0.0)
    sharpe = ex.mean() / r.std() * np.sqrt(12)
    dd = (curve / curve.cummax() - 1).min()
    t = ex.mean() / ex.std() * np.sqrt(len(ex))
    return {"label": label, "N": len(r), "CAGR": cagr, "Vol": vol,
            "Sharpe": sharpe, "MaxDD": dd, "t_excess": t}


def fmt(d: dict) -> str:
    return (f"{d['label']:<34} N={d['N']:>3}  CAGR {d['CAGR']*100:6.2f}%  "
            f"Vol {d['Vol']*100:5.2f}%  Sharpe {d['Sharpe']:5.2f}  "
            f"MaxDD {d['MaxDD']*100:6.1f}%  t={d['t_excess']:.2f}")


def per_year(r: pd.Series) -> pd.Series:
    return r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)


def window_stats(r: pd.Series, mask: pd.Series, label: str) -> dict:
    x, o = r[mask], r[~mask]
    t, p = sps.ttest_ind(x, o, equal_var=False)
    return {"label": label, "N_in": len(x), "avg_in": x.mean(), "hit_in": (x > 0).mean(),
            "N_out": len(o), "avg_out": o.mean(), "hit_out": (o > 0).mean(),
            "welch_t": t, "p": p}


def wfmt(d: dict) -> str:
    return (f"{d['label']:<28} N={d['N_in']:>3} avg {d['avg_in']*100:+.3f}%/mo hit {d['hit_in']*100:4.1f}%"
            f"  | other N={d['N_out']:>3} avg {d['avg_out']*100:+.3f}% hit {d['hit_out']*100:4.1f}%"
            f"  | Welch t={d['welch_t']:+.2f} p={d['p']:.3f}")


def main() -> None:
    close, openp, irx = load()

    print("=" * 110)
    print("GATE A -- EX-BONDS 12-ETF combo L/F inv-vol: same-close vs next-open (net of 5bps/side)")
    print("=" * 110)
    res = {}
    for uni, uname in [(EXB, "EXBONDS"), (UNIVERSE, "FULL16")]:
        for mode in ["close", "open"]:
            res[(uname, mode)] = run_sleeve(close, openp, irx, uni, mode)
    for uname in ["EXBONDS", "FULL16"]:
        for mode in ["close", "open"]:
            r = res[(uname, mode)]
            print(fmt(stat_line(r["net"], r["rf"], f"{uname} {mode}-exec net")))
        rc, ro = res[(uname, "close")]["net"], res[(uname, "open")]["net"]
        both = pd.DataFrame({"c": rc, "o": ro}).dropna()
        d = both["o"] - both["c"]
        print(f"  paired diff (open - close): {d.mean()*100:+.4f}%/mo, "
              f"t={d.mean()/d.std()*np.sqrt(len(d)):+.2f}, N={len(d)}, corr {both['c'].corr(both['o']):.4f}")

    print("\nPer-year net returns, EX-BONDS (same-close | next-open):")
    yc = per_year(res[("EXBONDS", "close")]["net"])
    yo = per_year(res[("EXBONDS", "open")]["net"])
    for y in yc.index:
        print(f"  {y}: {yc[y]*100:+7.2f}%  | {yo.get(y, np.nan)*100:+7.2f}%   diff {(yo.get(y, np.nan)-yc[y])*100:+.2f}pp")

    # sanity vs prototype (prototype included partial 2026-07; also check with it)
    print("\nSanity: prototype reported EXBONDS close-exec CAGR 6.75/Vol 6.32/Sharpe 0.79/DD -10.4 "
          "(incl partial 2026-07); FULL16 5.22/4.03/0.86/-4.5")

    # ================= GATE B =================
    print("\n" + "=" * 110)
    print("GATE B -- dead-zone fill: Jul-Sep months, midterm years (year%4==2), intersection")
    print("=" * 110)
    # book monthly
    tr = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet").dropna(
        subset=["Exit Date", "PnL_flat_750k"])
    book_d = (tr.set_index("Exit Date")["PnL_flat_750k"]
                .groupby(pd.Grouper(freq="ME")).sum())
    book_r = (tr.set_index("Exit Date")["R_Multiple"].dropna()
                .groupby(pd.Grouper(freq="ME")).sum())
    win = pd.date_range("2003-01-31", LAST_FULL, freq="ME")
    book_pct = (book_d / 750_000.0).reindex(win).fillna(0.0)
    book_r = book_r.reindex(win).fillna(0.0)

    series = {
        "sleeve EXBONDS close": res[("EXBONDS", "close")]["net"],
        "sleeve EXBONDS open": res[("EXBONDS", "open")]["net"],
        "sleeve FULL16 close": res[("FULL16", "close")]["net"],
        "book %750k": book_pct,
        "book sum R": book_r,
    }
    for name, s in series.items():
        s = s.dropna()
        jas = s.index.month.isin([7, 8, 9])
        mid = (s.index.year % 4) == 2
        both_m = jas & mid
        unit = "R" if "R" in name else "%"
        print(f"\n--- {name} (N={len(s)}, {s.index.min():%Y-%m}..{s.index.max():%Y-%m}) ---")
        scale = 1.0 if unit == "R" else 100.0
        overall = s.mean() * scale
        print(f"  overall: avg {overall:+.3f}{unit}/mo, hit {(s>0).mean()*100:.1f}%")
        for lbl, msk in [("Jul-Sep", pd.Series(jas, s.index)),
                         ("midterm yrs", pd.Series(mid, s.index)),
                         ("Jul-Sep x midterm", pd.Series(both_m, s.index))]:
            d = window_stats(s, msk, lbl)
            if unit == "R":
                print(f"  {lbl:<20} N={d['N_in']:>3} avg {d['avg_in']:+.2f}R/mo hit {d['hit_in']*100:4.1f}%"
                      f" | other avg {d['avg_out']:+.2f}R hit {d['hit_out']*100:4.1f}%"
                      f" | t={d['welch_t']:+.2f} p={d['p']:.3f}")
            else:
                print("  " + wfmt(d))

    # midterm-year clustering: per-year sleeve returns, midterm vs not
    print("\nYear-level view (EXBONDS close, compounded per calendar year):")
    yr = per_year(res[("EXBONDS", "close")]["net"])
    midy = yr[(yr.index % 4) == 2]
    othy = yr[(yr.index % 4) != 2]
    t, p = sps.ttest_ind(midy, othy, equal_var=False)
    print(f"  midterm years (N={len(midy)}): {', '.join(f'{y}:{v*100:+.1f}%' for y, v in midy.items())}")
    print(f"  midterm avg {midy.mean()*100:+.2f}%/yr vs other {othy.mean()*100:+.2f}%/yr, "
          f"Welch t={t:+.2f} p={p:.3f} (year-clustered, N={len(midy)}/{len(othy)})")

    # ============ integration re-verification ============
    print("\n" + "=" * 110)
    print("RE-VERIFY: book correlation and high-fragility months")
    print("=" * 110)
    for uname in ["FULL16", "EXBONDS"]:
        for mode in ["close", "open"] if uname == "EXBONDS" else ["close"]:
            r = res[(uname, mode)]["net"]
            both = pd.DataFrame({"sleeve": r, "book": book_pct}).dropna()
            print(f"{uname} {mode}: corr to book = {both['sleeve'].corr(both['book']):+.3f} "
                  f"(N={len(both)}, {both.index.min():%Y-%m}..{both.index.max():%Y-%m})")

    frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
    ma10 = frag["63d"].rolling(10, min_periods=1).mean()
    m_mean = ma10.groupby(pd.Grouper(freq="ME")).mean()
    for uname in ["FULL16", "EXBONDS"]:
        for mode in (["close", "open"] if uname == "EXBONDS" else ["close"]):
            r = res[(uname, mode)]["net"]
            flag = (m_mean >= 50).reindex(r.index)
            sub = r[flag == True]   # noqa: E712
            oth = r[(flag == False)]  # noqa: E712
            t, p = sps.ttest_ind(sub, oth, equal_var=False)
            print(f"{uname} {mode}: high-frag N={len(sub)} avg {sub.mean()*100:+.2f}%/mo "
                  f"hit {(sub>0).mean()*100:.0f}% vs other (2016-07+ only N={len(oth)}) "
                  f"{oth.mean()*100:+.2f}%/mo  t={t:+.2f} p={p:.3f}")
            if uname == "EXBONDS" and mode == "close":
                print("   high-frag months: " + ", ".join(d.strftime("%Y-%m") for d in sub.index))

    # save monthly series for the writeup
    out = pd.DataFrame({
        "exb_close": res[("EXBONDS", "close")]["net"],
        "exb_open": res[("EXBONDS", "open")]["net"],
        "full16_close": res[("FULL16", "close")]["net"],
        "book_pct": book_pct, "book_R": book_r,
    })
    out.to_parquet(ROOT / "scratch" / "ultracode_research" / "gate_ab_series.parquet")
    print("\nsaved gate_ab_series.parquet")


if __name__ == "__main__":
    main()
