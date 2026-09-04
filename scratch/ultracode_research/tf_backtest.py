"""Multi-asset trend-following prototype on liquid ETFs.

Rules (stated up front):
- Universe (16 ETFs, dynamic inception): equities SPY QQQ IWM EFA EEM FXI VNQ,
  bonds/credit TLT IEF LQD HYG, commodities GLD SLV DBC USO, dollar UUP.
- Monthly rebalance at month-end close (signal computed on that close, traded
  at that close -- standard monthly convention; live = MOC on last session).
- Signals (per asset, monthly closes):
    mom12_1 : P[t-1m]/P[t-12m] - 1 > 0            (12-1 momentum sign)
    ma10    : P[t] > SMA(10 monthly closes)        (price vs 10-month MA)
    combo   : both ON
- Eligibility: asset needs 13 monthly closes (12m lookback) + 63d daily vol.
- Weights: slot-based inverse-vol. Each eligible asset gets a base slot
  w_i = (1/vol63_i)/sum(1/vol63_j) over ALL eligible assets, capped 20%
  (excess to cash). Signal OFF -> slot sits in T-bills (long/flat) or goes
  short -w_i (long/short). Cash earns ^IRX/100/12 on (1 - net exposure).
- Costs: 5 bps per side on target-weight turnover (0.0005 * sum|dw|).
- All prices dividend-adjusted (total-return basis). Signals are relative
  levels recomputed per run -> scale-invariant per CLAUDE.md rule.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "FXI", "VNQ",
            "TLT", "IEF", "LQD", "HYG",
            "GLD", "SLV", "DBC", "USO", "UUP"]
BONDS = ["TLT", "IEF", "LQD", "HYG"]
COST_PER_SIDE = 0.0005
CAP = 0.20


def load_prices() -> tuple[pd.DataFrame, pd.Series]:
    mp = pd.read_parquet(ROOT / "data" / "master_prices.parquet",
                         columns=["ticker", "date", "Close"])
    sub = mp[mp.ticker.isin(UNIVERSE + ["^IRX"])]
    wide = sub.pivot(index="date", columns="ticker", values="Close").sort_index()
    irx = wide.pop("^IRX").ffill()
    wide = wide[[c for c in UNIVERSE if c in wide.columns]].astype(float)
    return wide, irx


def build_signals(px: pd.DataFrame) -> dict[str, pd.DataFrame]:
    m = px.resample("ME").last()
    mom12_1 = m.shift(1) / m.shift(12) - 1.0
    ma10 = m - m.rolling(10).mean()
    elig = m.notna() & m.shift(12).notna()
    sig = {
        "mom12_1": (mom12_1 > 0),
        "ma10": (ma10 > 0),
        "combo": (mom12_1 > 0) & (ma10 > 0),
    }
    return {k: v.where(elig) for k, v in sig.items()}  # NaN = ineligible


def base_weights(px: pd.DataFrame, elig_mask: pd.DataFrame,
                 equal_weight: bool = False) -> pd.DataFrame:
    dret = px.pct_change()
    vol63 = dret.rolling(63).std() * np.sqrt(252)
    vol_m = vol63.resample("ME").last().clip(lower=0.04)  # floor avoids inf
    if equal_weight:
        inv = elig_mask.notna().astype(float) * elig_mask.notna()
        inv = elig_mask.notna().astype(float)
    else:
        inv = (1.0 / vol_m).where(elig_mask.notna())
        inv = inv.fillna(0.0) * elig_mask.notna()
    tot = inv.sum(axis=1)
    w = inv.div(tot.replace(0, np.nan), axis=0).fillna(0.0)
    return w.clip(upper=CAP)  # excess slot -> cash


def run(px: pd.DataFrame, irx: pd.Series, sig: pd.DataFrame,
        long_short: bool = False, equal_weight: bool = False,
        universe: list[str] | None = None) -> pd.DataFrame:
    if universe is not None:
        px = px[universe]
        sig = sig[universe]
    m = px.resample("ME").last()
    aret = m.pct_change()                       # asset month returns
    rf_ann = (irx.resample("ME").last() / 100.0).reindex(aret.index).ffill()
    rf = rf_ann / 12.0
    base = base_weights(px, sig, equal_weight=equal_weight)
    on = sig.fillna(False).astype(float)
    if long_short:
        direction = on * 2.0 - 1.0              # +1 on, -1 off
        direction = direction.where(sig.notna(), 0.0)
        w = base * direction
    else:
        w = base * on
    # weights decided at month-end t apply to month t+1 returns
    w_held = w.shift(1).fillna(0.0)
    port_asset = (w_held * aret).sum(axis=1)
    net_exp = w_held.sum(axis=1)
    cash_ret = (1.0 - net_exp) * rf.shift(1).fillna(0.0)
    turnover = (w - w.shift(1)).abs().sum(axis=1).shift(1).fillna(0.0)
    cost = turnover * COST_PER_SIDE
    out = pd.DataFrame({
        "gross": port_asset + cash_ret,
        "net": port_asset + cash_ret - cost,
        "turnover": turnover,
        "net_exp": net_exp,
        "rf": rf,
    })
    # start when at least 3 assets eligible
    n_elig = sig.notna().sum(axis=1)
    start = n_elig[n_elig >= 3].index.min()
    return out.loc[out.index >= start]


def stats(r: pd.Series, rf: pd.Series | None = None, label: str = "") -> dict:
    r = r.dropna()
    if len(r) == 0:
        return {}
    curve = (1 + r).cumprod()
    yrs = len(r) / 12.0
    cagr = curve.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(12)
    ex = r - (rf.reindex(r.index).fillna(0.0) if rf is not None else 0.0)
    sharpe = ex.mean() / r.std() * np.sqrt(12) if r.std() > 0 else np.nan
    dd = (curve / curve.cummax() - 1).min()
    return {"label": label, "months": len(r), "CAGR": cagr, "Vol": vol,
            "Sharpe": sharpe, "MaxDD": dd}


def fmt(d: dict) -> str:
    return (f"{d['label']:<28} N={d['months']:>4}  CAGR {d['CAGR']*100:6.2f}%  "
            f"Vol {d['Vol']*100:5.2f}%  Sharpe {d['Sharpe']:5.2f}  MaxDD {d['MaxDD']*100:6.1f}%")


def main() -> None:
    px, irx = load_prices()
    sigs = build_signals(px)

    results: dict[str, pd.DataFrame] = {}
    specs = [
        ("mom12_1 L/F invvol", "mom12_1", False, False, None),
        ("ma10 L/F invvol", "ma10", False, False, None),
        ("combo L/F invvol", "combo", False, False, None),
        ("combo L/S invvol", "combo", True, False, None),
        ("combo L/F equalwt", "combo", False, True, None),
        ("combo L/F invvol EX-BONDS", "combo", False, False,
         [t for t in UNIVERSE if t not in BONDS]),
        ("mom12_1 L/F invvol EX-BONDS", "mom12_1", False, False,
         [t for t in UNIVERSE if t not in BONDS]),
    ]
    print("=" * 100)
    print("FULL-SAMPLE (net of 5bps/side) -- all specs")
    print("=" * 100)
    for name, sk, ls, ew, uni in specs:
        res = run(px, irx, sigs[sk], long_short=ls, equal_weight=ew, universe=uni)
        results[name] = res
        print(fmt(stats(res["net"], res["rf"], name)))
        print(fmt(stats(res["gross"], res["rf"], name + " [gross]")))
        print(f"{'':28}   avg monthly turnover {res['turnover'].mean()*100:.1f}%, "
              f"avg net exposure {res['net_exp'].mean()*100:.0f}%")

    # SPY benchmark
    spy_m = px["SPY"].resample("ME").last().pct_change()
    prim = results["combo L/F invvol"]
    spy_al = spy_m.reindex(prim.index)
    print("\n" + fmt(stats(spy_al, prim["rf"], "SPY buy&hold (same window)")))

    # ---- primary spec deep dive ----
    prim_name = "combo L/F invvol"
    r = results[prim_name]["net"]
    rf = results[prim_name]["rf"]
    print("\n" + "=" * 100)
    print(f"PRIMARY SPEC: {prim_name} -- per-year net returns")
    print("=" * 100)
    yr = r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)
    spy_yr = spy_al.groupby(spy_al.index.year).apply(lambda x: (1 + x).prod() - 1)
    for y in yr.index:
        print(f"  {y}: sleeve {yr[y]*100:7.2f}%   SPY {spy_yr.get(y, np.nan)*100:7.2f}%")

    print("\nSub-periods (net):")
    for lo, hi, lbl in [("2003", "2012", "2003-2012"), ("2013", "2019", "2013-2019"),
                        ("2020", "2027", "2020+")]:
        seg = r[(r.index >= lo) & (r.index < str(int(hi) + 1))]
        seg = r[(r.index.year >= int(lo)) & (r.index.year <= int(hi))]
        print("  " + fmt(stats(seg, rf, lbl)))
        exb = results["combo L/F invvol EX-BONDS"]["net"]
        seg2 = exb[(exb.index.year >= int(lo)) & (exb.index.year <= int(hi))]
        print("  " + fmt(stats(seg2, rf, lbl + " EX-BONDS")))

    # ---- crisis months ----
    print("\n" + "=" * 100)
    print("CRISIS WINDOWS (primary spec, net, monthly)")
    print("=" * 100)
    for lbl, lo, hi in [("2008", "2008-01", "2008-12"),
                        ("2020 Jan-Apr", "2020-01", "2020-04"),
                        ("2022", "2022-01", "2022-12")]:
        seg = r[lo:hi]
        sspy = spy_al[lo:hi]
        cum = (1 + seg).prod() - 1
        cums = (1 + sspy).prod() - 1
        print(f"\n{lbl}: sleeve {cum*100:+.1f}% vs SPY {cums*100:+.1f}%")
        for dt in seg.index:
            print(f"  {dt.strftime('%Y-%m')}: sleeve {seg[dt]*100:+6.2f}%  SPY {sspy[dt]*100:+6.2f}%")

    # ---- book correlation ----
    print("\n" + "=" * 100)
    print("CORRELATION TO EXISTING BOOK (monthly PnL_flat_750k by exit month / 750k)")
    print("=" * 100)
    tr = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
    tr = tr.dropna(subset=["Exit Date", "PnL_flat_750k"])
    book = (tr.set_index("Exit Date")["PnL_flat_750k"]
              .groupby(pd.Grouper(freq="ME")).sum() / 750_000.0)
    book = book.reindex(r.index).fillna(0.0)
    both = pd.DataFrame({"sleeve": r, "book": book}).dropna()
    print(f"Full overlap {both.index.min().strftime('%Y-%m')}..{both.index.max().strftime('%Y-%m')} "
          f"N={len(both)}: corr = {both['sleeve'].corr(both['book']):+.3f}")
    dn = both[both["book"] < 0]
    print(f"Months book<0 (N={len(dn)}): corr = {dn['sleeve'].corr(dn['book']):+.3f}, "
          f"sleeve avg {dn['sleeve'].mean()*100:+.2f}%/mo, book avg {dn['book'].mean()*100:+.2f}%/mo")
    worst = both.nsmallest(12, "book")
    print("12 worst book months:")
    for dt, row in worst.iterrows():
        print(f"  {dt.strftime('%Y-%m')}: book {row['book']*100:+6.2f}%  sleeve {row['sleeve']*100:+6.2f}%")

    # ---- high-fragility months (2016-07+) ----
    print("\n" + "=" * 100)
    print("HIGH-FRAGILITY MONTHS (63d MA10 >= 50, 2016-07+)")
    print("=" * 100)
    frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
    ma10 = frag["63d"].rolling(10, min_periods=1).mean()
    m_mean = ma10.groupby(pd.Grouper(freq="ME")).mean()
    m_any = ma10.groupby(pd.Grouper(freq="ME")).max()
    m_start = ma10.groupby(pd.Grouper(freq="ME")).first()
    for lbl, series in [("month-mean>=50", m_mean), ("any-day>=50", m_any),
                        ("month-start>=50", m_start)]:
        flag = (series >= 50).reindex(both.index)
        sub = both[flag == True]  # noqa: E712
        oth = both[(flag == False)]  # noqa: E712
        if len(sub) == 0:
            print(f"{lbl}: no months")
            continue
        print(f"{lbl}: N={len(sub)} high-frag months | "
              f"sleeve avg {sub['sleeve'].mean()*100:+.2f}%/mo (hit {(sub['sleeve']>0).mean()*100:.0f}%) "
              f"vs other months {oth['sleeve'].mean()*100:+.2f}%/mo | "
              f"book avg {sub['book'].mean()*100:+.2f}%/mo vs other {oth['book'].mean()*100:+.2f}%/mo")
        print("   high-frag month list: " +
              ", ".join(d.strftime('%Y-%m') for d in sub.index))

    # signals-on count over time (context)
    print("\nAvg # assets ON (combo): ",
          f"{sigs['combo'].fillna(False).sum(axis=1).loc[r.index].mean():.1f} of "
          f"{sigs['combo'].notna().sum(axis=1).loc[r.index].mean():.1f} eligible")

    # save monthly series for reuse
    outp = ROOT / "scratch" / "ultracode_research" / "tf_monthly_series.parquet"
    pd.DataFrame({"sleeve_net": r, "sleeve_gross": results[prim_name]["gross"],
                  "book": book, "spy": spy_al}).to_parquet(outp)
    print(f"\nsaved {outp}")


if __name__ == "__main__":
    main()
