"""Adversarial verification of the trend-following prototype claims.

Independent reimplementation from the rules in trend-following.md section 1.
Does NOT import or reuse tf_backtest.py.
"""
import numpy as np
import pandas as pd

UNIV = ["SPY", "QQQ", "IWM", "EFA", "EEM", "FXI", "VNQ",
        "TLT", "IEF", "LQD", "HYG",
        "GLD", "SLV", "DBC", "USO", "UUP"]
BONDS = ["TLT", "IEF", "LQD", "HYG"]
COST = 0.0005  # 5 bps per side, charged on sum |dw| of invested weights

mp = pd.read_parquet("data/master_prices.parquet")
mp = mp[mp["ticker"].isin(UNIV + ["^IRX"])]
px = mp.pivot(index="date", columns="ticker", values="Close").sort_index()
px = px.loc["1999-01-01":"2026-06-30"]

irx_d = px["^IRX"]
p = px[UNIV]

# month-end closes
me = p.resample("ME").last()
irx_me = irx_d.resample("ME").last().ffill()
mret = me.pct_change()  # return of month t (from close t-1 to close t)

# 63d daily vol, annualized, sampled at month end
dret = p.pct_change()
vol63 = dret.rolling(63).std() * np.sqrt(252)
vol_me = vol63.resample("ME").last()

# signals at month end t
mom = me.shift(1) / me.shift(12) - 1.0          # 12-1 momentum
ma10 = me > me.rolling(10).mean()
sig_mom = mom > 0
sig_combo = sig_mom & ma10

# eligibility: 13 monthly closes available (so mom is computable) + vol available
nobs = me.notna().cumsum()
elig = me.notna() & (nobs >= 13) & vol_me.notna()

cash_mo = (irx_me / 100.0 / 12.0)  # bill yield for the FOLLOWING month, set at t


def run(signal: pd.DataFrame, universe: list[str], long_short: bool = False,
        equal_weight: bool = False, delay: int = 0) -> pd.DataFrame:
    sig = signal[universe]
    el = elig[universe]
    v = vol_me[universe]
    rows = []
    prev_w = pd.Series(0.0, index=universe)
    for t in me.index:
        e = el.loc[t]
        if e.sum() < 3:
            continue
        if equal_weight:
            iw = pd.Series(np.where(e, 1.0 / e.sum(), 0.0), index=universe)
        else:
            inv = (1.0 / v.loc[t]).where(e, 0.0)
            iw = inv / inv.sum()
        iw = iw.clip(upper=0.20)  # excess to cash
        s = sig.loc[t].where(e, False)
        if long_short:
            w = iw * np.where(s, 1.0, -1.0) * e
        else:
            w = iw * s.astype(float)
        rows.append((t, w, iw, s, e))
    out = []
    wlist = {t: w for t, w, _, _, _ in rows}
    ts = [t for t, *_ in rows]
    for i, (t, w, iw, s, e) in enumerate(rows):
        j = i - delay
        w_use = wlist[ts[j]] if j >= 0 else pd.Series(0.0, index=universe)
        # return realized in the NEXT month
        nxt = me.index[me.index.get_loc(t) + 1] if me.index.get_loc(t) + 1 < len(me.index) else None
        if nxt is None:
            continue
        r = mret.loc[nxt, universe].fillna(0.0)
        gross_inv = float((w_use * r).sum())
        cash_w = 1.0 - float(w_use.abs().sum()) if long_short else 1.0 - float(w_use.sum())
        cash_w = max(cash_w, 0.0)
        cash_r = float(cash_mo.loc[t]) if not np.isnan(cash_mo.loc[t]) else 0.0
        turnover = float((w_use - prev_w).abs().sum())
        prev_w = w_use
        net = gross_inv + cash_w * cash_r - COST * turnover
        out.append({"month": nxt, "net": net, "gross": gross_inv + cash_w * cash_r,
                    "turnover": turnover, "n_pos": int((w_use != 0).sum()),
                    "cash_ret": cash_r})
    df = pd.DataFrame(out).set_index("month")
    return df


def stats(r: pd.Series, cash: pd.Series | None = None) -> dict:
    r = r.dropna()
    n = len(r)
    eq = (1 + r).cumprod()
    yrs = n / 12.0
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(12)
    if cash is not None:
        ex = r - cash.reindex(r.index).fillna(0.0)
    else:
        ex = r
    sharpe = ex.mean() / ex.std() * np.sqrt(12)
    t = ex.mean() / (ex.std() / np.sqrt(n))
    dd = (eq / eq.cummax() - 1).min()
    return {"N": n, "CAGR": cagr, "vol": vol, "Sharpe": sharpe, "t_excess": t,
            "maxDD": dd, "mean_ex_mo": ex.mean(), "start": r.index[0].strftime("%Y-%m"),
            "end": r.index[-1].strftime("%Y-%m")}


def show(name: str, d: dict) -> None:
    print(f"{name:38s} N={d['N']:4d} {d['start']}..{d['end']}  CAGR={d['CAGR']*100:5.2f}%  "
          f"vol={d['vol']*100:5.2f}%  Sharpe={d['Sharpe']:.2f}  t={d['t_excess']:.2f}  "
          f"maxDD={d['maxDD']*100:5.1f}%")


print("=" * 100)
print("CLAIM 1: primary spec full-sample")
prim = run(sig_combo, UNIV)
cash_series = prim["cash_ret"]
show("combo L/F inv-vol NET", stats(prim["net"], cash_series))
show("combo L/F inv-vol GROSS", stats(prim["gross"], cash_series))
print(f"avg turnover {prim['turnover'].mean()*100:.1f}%/mo, avg positions {prim['n_pos'].mean():.1f}")

print()
print("CLAIM 2: crisis years")
net = prim["net"]
spy_mo = mret["SPY"].reindex(net.index)


def yr(series, y):
    s = series[series.index.year == y]
    return (1 + s).prod() - 1


for y in [2008, 2022]:
    print(f"{y}: sleeve {yr(net, y)*100:+.1f}%  SPY {yr(spy_mo, y)*100:+.1f}%")
febmar = net.loc["2020-02":"2020-03"]
febmar_spy = spy_mo.loc["2020-02":"2020-03"]
print(f"2020 Feb-Mar: sleeve {((1+febmar).prod()-1)*100:+.1f}%  SPY {((1+febmar_spy).prod()-1)*100:+.1f}%")

print()
print("Per-year sleeve returns (spot check vs their table):")
for y in sorted(set(net.index.year)):
    print(f"  {y}: {yr(net, y)*100:+.1f}%", end="")
print()

print()
print("CLAIM 3: correlation to book")
tr = pd.read_parquet("data/backtest_trades_full.parquet")
book = tr.groupby(pd.to_datetime(tr["Exit Date"]).dt.to_period("M"))["PnL_flat_750k"].sum() / 750000.0
book.index = book.index.to_timestamp("M")
win = pd.date_range("2003-01-31", "2026-06-30", freq="ME")
b = book.reindex(win).fillna(0.0)
s = net.reindex(win)
mask = s.notna()
b, s = b[mask], s[mask]
print(f"N={len(b)}  corr={b.corr(s):+.3f}")
losing = b < 0
print(f"losing book months N={losing.sum()}: sleeve avg {s[losing].mean()*100:+.2f}%/mo, "
      f"hit {(s[losing] > 0).mean()*100:.0f}%, corr in losers {b[losing].corr(s[losing]):+.3f}")
worst12 = b.nsmallest(12).index
print(f"12 worst book months: sleeve positive {(s.loc[worst12] > 0).sum()}/12, "
      f"worst sleeve {s.loc[worst12].min()*100:+.1f}%")

print()
print("CLAIM 4: high-fragility months (63d MA10 >= 50, month mean, 2016-07+)")
frag = pd.read_parquet("data/rd2_fragility.parquet")["63d"].rolling(10, min_periods=1).mean()
fm = frag.resample("ME").mean()
fm = fm.loc["2016-07-31":]
hi = fm >= 50
print(f"high-frag months N={hi.sum()}: {list(fm.index[hi].strftime('%Y-%m'))}")
s16 = net.reindex(fm.index)
b16 = book.reindex(fm.index).fillna(0.0)
spy16 = spy_mo.reindex(fm.index)
for lag in [0, 1, 2, 3]:
    flag = hi.shift(lag).fillna(False)
    n = int(flag.sum())
    sl = s16[flag]
    print(f"t+{lag}: N={n}  sleeve {sl.mean()*100:+.2f}%/mo (hit {(sl>0).mean()*100:.0f}%)  "
          f"other {s16[~flag].mean()*100:+.2f}%  book {b16[flag].mean()*100:+.2f}%  "
          f"SPY {spy16[flag].mean()*100:+.2f}%")
# significance of the concurrent split
from scipy import stats as sps
tt = sps.ttest_ind(s16[hi].dropna(), s16[~hi].dropna(), equal_var=False)
print(f"concurrent split Welch t={tt.statistic:+.2f} p={tt.pvalue:.3f}")

print()
print("CLAIM 5: sub-period Sharpes, full vs ex-bonds")
exb = [t for t in UNIV if t not in BONDS]
prim_exb = run(sig_combo, exb)
for lo, hi_ in [("2003-01", "2012-12"), ("2013-01", "2019-12"), ("2020-01", "2026-06")]:
    d1 = stats(prim["net"].loc[lo:hi_], prim["cash_ret"].loc[lo:hi_])
    d2 = stats(prim_exb["net"].loc[lo:hi_], prim_exb["cash_ret"].loc[lo:hi_])
    print(f"{lo}..{hi_}: full16 Sharpe {d1['Sharpe']:.2f} CAGR {d1['CAGR']*100:.2f}%  |  "
          f"ex-bonds Sharpe {d2['Sharpe']:.2f} CAGR {d2['CAGR']*100:.2f}%")
show("ex-bonds full sample NET", stats(prim_exb["net"], prim_exb["cash_ret"]))

print()
print("CLAIM 6: long/short variant")
ls = run(sig_combo, UNIV, long_short=True)
show("combo L/S inv-vol NET", stats(ls["net"], ls["cash_ret"]))

print()
print("CLAIM 7: portfolio effect at 1x / 2x (flat 750k basis)")
for mult in [0, 1, 2]:
    comb = b + mult * s
    sh = comb.mean() / comb.std() * np.sqrt(12)
    print(f"book + {mult}x sleeve: mean {comb.mean()*100:+.2f}%/mo  ann vol "
          f"{comb.std()*np.sqrt(12)*100:.2f}%  Sharpe {sh:.2f}  worst month {comb.min()*100:+.1f}%")

print()
print("CLAIM 8: execution + delay")
print(f"avg positions {prim['n_pos'].mean():.1f}, avg turnover {prim['turnover'].mean()*100:.1f}%/mo "
      f"(~${prim['turnover'].mean()*750000:,.0f}/mo at 750k)")
# flips per month: signal state changes among eligible assets
flips = (sig_combo.where(elig) != sig_combo.where(elig).shift(1)) & elig & elig.shift(1).fillna(False)
flips_m = flips.sum(axis=1).loc[prim.index[0]:]
print(f"avg signal flips/mo {flips_m.mean():.1f}")
d1 = run(sig_combo, UNIV, delay=1)
show("combo L/F delayed 1 full month", stats(d1["net"], d1["cash_ret"]))

print()
print("Other spec variants (their table 2):")
m = run(sig_mom, UNIV)
show("mom12-1 L/F inv-vol", stats(m["net"], m["cash_ret"]))
ma = run(pd.DataFrame(ma10), UNIV)
show("ma10 L/F inv-vol", stats(ma["net"], ma["cash_ret"]))
ew = run(sig_combo, UNIV, equal_weight=True)
show("combo L/F equal-weight", stats(ew["net"], ew["cash_ret"]))
spy_full = spy_mo.dropna()
show("SPY B&H same window", stats(spy_full, prim["cash_ret"]))
