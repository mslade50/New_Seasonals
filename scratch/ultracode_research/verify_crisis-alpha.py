"""Adversarial verification of the crisis-alpha track findings.

Independent recompute — does not import or reuse any ca_*.py code.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

ROOT = Path(r"C:\Users\McKinley Slade\dev\New_Seasonals")
NAV = 750_000.0

# ---------------------------------------------------------------- data
mp = pd.read_parquet(ROOT / "data" / "master_prices.parquet",
                     columns=["ticker", "date", "Close"])
mp["date"] = pd.to_datetime(mp["date"])


def close(tk: str) -> pd.Series:
    s = mp[mp["ticker"] == tk].set_index("date")["Close"].sort_index()
    return s[~s.index.duplicated()]


uvxy = close("UVXY")
spy = close("SPY")
vix3m = close("^VIX3M")
irx = close("^IRX")

fr = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
basis = fr["63d"].rolling(10, min_periods=1).mean()  # live sizing basis

tr = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
tr["Signal Date"] = pd.to_datetime(tr["Signal Date"])
tr["Exit Date"] = pd.to_datetime(tr["Exit Date"])

print("=" * 70)
print("UVXY window:", uvxy.index[0].date(), "->", uvxy.index[-1].date())

# ------------------------------------------------- claim 1: VXX-proxy B&H
lev = pd.Series(np.where(uvxy.index < pd.Timestamp("2018-02-28"), 2.0, 1.5),
                index=uvxy.index)
r_uvxy = uvxy.pct_change()
r_proxy = (r_uvxy / lev).dropna()
cum = (1 + r_proxy).cumprod()
yrs = (r_proxy.index[-1] - r_proxy.index[0]).days / 365.25
cagr = cum.iloc[-1] ** (1 / yrs) - 1
vol = r_proxy.std() * np.sqrt(252)
sharpe = r_proxy.mean() / r_proxy.std() * np.sqrt(252)
dd = (cum / cum.cummax() - 1).min()
print(f"\n[1] VXX-proxy B&H {r_proxy.index[0].date()}..{r_proxy.index[-1].date()}: "
      f"CAGR {cagr:+.1%}/yr, vol {vol:.0%}, Sharpe {sharpe:+.2f}, maxDD {dd:.0%}, "
      f"total {cum.iloc[-1]-1:+.1%}")
# raw UVXY for reference
cum_u = (1 + r_uvxy.dropna()).cumprod()
cagr_u = cum_u.iloc[-1] ** (1 / yrs) - 1
print(f"    UVXY raw CAGR {cagr_u:+.1%}/yr")

# ------------------------------------------------- gate construction
def build_gate(b: pd.Series, thr_on: float = 55.0, thr_off: float = 50.0) -> pd.Series:
    on = False
    out = []
    for v in b.values:
        if not on and v >= thr_on:
            on = True
        elif on and v < thr_off:
            on = False
        out.append(on)
    return pd.Series(out, index=b.index)


gate55 = build_gate(basis, 55, 50)
gate50 = build_gate(basis, 50, 45)

# episodes (gaps <= 10 td merged)
def episodes(g: pd.Series, merge_td: int = 10):
    idx = g.index
    on_idx = np.where(g.values)[0]
    if len(on_idx) == 0:
        return []
    eps = []
    start = on_idx[0]
    prev = on_idx[0]
    for i in on_idx[1:]:
        if i - prev > merge_td:
            eps.append((idx[start], idx[prev]))
            start = i
        prev = i
    eps.append((idx[start], idx[prev]))
    return eps


eps = episodes(gate55)
print(f"\n[gate] thr55/off50 episodes (merged<=10td): {len(eps)}")
for a, b_ in eps:
    peak = basis.loc[a:b_].max()
    print(f"    {a.date()} .. {b_.date()}  peak {peak:.0f}")

# ------------------------------------------------- claim 3: tactical VXXP 5%
# gate at close t -> position effective t+1, close-to-close returns
g_on_price = gate55.reindex(r_proxy.index).ffill().fillna(False).astype(bool)
pos = g_on_price.shift(1).fillna(False).astype(float)  # effective next session
w = 0.05
pnl_gross = pos * r_proxy * w * NAV
turn = pos.diff().abs().fillna(pos.iloc[0])
cost = turn * w * NAV * 0.0010  # 10 bps/side
pnl_net = (pnl_gross - cost)
# window 2016-08..2026-06 monthly; but total over full gated period
sleeve_daily = pnl_net.loc["2016-07-01":]
n_on = int(pos.loc["2016-07-01":].sum())
n_days = len(pos.loc["2016-07-01":])
entries = int((pos.diff() == 1).sum())
total = sleeve_daily.sum()
curve = sleeve_daily.cumsum()
sleeve_dd = (curve - curve.cummax()).min()
print(f"\n[3] tactical VXXP 5% thr55: total ${total:,.0f}  "
      f"({total/NAV/9.9*100:+.2f}%/yr approx), in-market {n_on}/{n_days} "
      f"({n_on/n_days:.0%}), entries(round trips) {entries}, sleeve maxDD ${sleeve_dd:,.0f}")

# thr50 / thr60 variants
for thr, off in [(50, 45), (60, 55)]:
    g = build_gate(basis, thr, off).reindex(r_proxy.index).ffill().fillna(False)
    p = g.shift(1).fillna(False).astype(float)
    pn = (p * r_proxy * w * NAV - p.diff().abs().fillna(0) * w * NAV * 0.0010)
    print(f"    thr{thr}: total ${pn.loc['2016-07-01':].sum():,.0f}")

# ------------------------------------------------- claim 4: monthly t-stat
mret = sleeve_daily.loc["2016-08-01":"2026-06-30"].resample("ME").sum() / NAV * 100
mret = mret[(mret.index >= "2016-08-01") & (mret.index <= "2026-06-30")]
basis_m = basis.resample("ME").mean()
hif = basis_m.reindex(mret.index) >= 50
t, p = stats.ttest_1samp(mret, 0)
print(f"\n[4] monthly sleeve N={len(mret)}: mean {mret.mean():+.3f}%/mo, "
      f"t={t:+.2f}, p={p:.2f}")
hi = mret[hif.values]
t2, p2 = stats.ttest_1samp(hi, 0)
print(f"    hi-frag months N={len(hi)}: mean {hi.mean():+.3f}%/mo, "
      f"hit {(hi>0).mean():.0%}, t={t2:+.2f} p={p2:.2f}")

# book monthly R for correlation
tr16 = tr[tr["Exit Date"] >= "2016-07-01"]
book_mR = tr16.groupby(tr16["Exit Date"].dt.to_period("M"))["R_Multiple"].sum()
book_mR.index = book_mR.index.to_timestamp("M")
common = mret.index.intersection(book_mR.index)
corr = np.corrcoef(mret.reindex(common), book_mR.reindex(common))[0, 1]
book_mPnL = tr16.groupby(tr16["Exit Date"].dt.to_period("M"))["PnL_flat_750k"].sum()
book_mPnL.index = book_mPnL.index.to_timestamp("M")
corr2 = np.corrcoef(mret.reindex(common), book_mPnL.reindex(common))[0, 1]
print(f"    corr(sleeve, book monthly R) = {corr:+.2f}; vs book $ = {corr2:+.2f}")

# ------------------------------------------------- claim 5: LOEO on episodes
ep_pnl = {}
for a, b_ in eps:
    # sleeve pnl attributable: position days from a..(b_ + few days for exit)
    win = sleeve_daily.loc[a:b_ + pd.Timedelta(days=7)]
    # more precise: assign each nonzero pnl day to the episode containing its gate-on driver
    ep_pnl[(a, b_)] = win[win != 0].sum()
tot_ep = sum(ep_pnl.values())
print(f"\n[5] episode-attributed sleeve PnL sums to ${tot_ep:,.0f} (vs total ${total:,.0f})")
worst_loeo = None
for k, v in ep_pnl.items():
    loeo = total - v
    if worst_loeo is None or loeo < worst_loeo[1]:
        worst_loeo = (k, loeo)
    print(f"    {k[0].date()}..{k[1].date()}: ep ${v:,.0f}  LOEO ${total - v:,.0f}")
print(f"    paid episodes: {sum(1 for v in ep_pnl.values() if v > 0)}/{len(ep_pnl)}; "
      f"worst LOEO ${worst_loeo[1]:,.0f} (drop {worst_loeo[0][0].date()})")

# ------------------------------------------------- puts (claims 2, 6)
def bs_put(S, K, T, r, iv):
    if T <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * iv * iv) * T) / (iv * np.sqrt(T))
    d2 = d1 - iv * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def put_iv(S, K, v3m):
    otm = max(0.0, (S - K) / S * 100)
    return (v3m + 0.40 * otm) / 100


px = pd.DataFrame({"S": spy, "v3m": vix3m, "irx": irx}).loc["2016-07-01":].ffill().dropna()
dates = px.index


def run_put_overlay(gated: bool, gate: pd.Series | None, otm_frac=0.95,
                    tenor_td=63, roll_td=5, haircut=0.05):
    """Returns daily $ PnL series of a 1x-NAV 3M put overlay."""
    pnl = pd.Series(0.0, index=dates)
    holding = None  # dict K, expiry_i, shares, last_mark
    if gated:
        g_eff = gate.reindex(dates).ffill().fillna(False).shift(1).fillna(False).astype(bool)
    else:
        g_eff = pd.Series(True, index=dates)
    for i, d in enumerate(dates):
        S, v3m_, r_ = px.at[d, "S"], px.at[d, "v3m"], px.at[d, "irx"] / 100
        want = bool(g_eff.loc[d])
        if holding is not None:
            T = max(holding["expiry_i"] - i, 0) / 252
            mark = bs_put(S, holding["K"], T, r_, put_iv(S, holding["K"], v3m_))
            pnl.loc[d] += (mark - holding["last_mark"]) * holding["shares"]
            holding["last_mark"] = mark
            if not want:
                pnl.loc[d] -= mark * haircut * holding["shares"]  # sell haircut
                holding = None
            elif holding["expiry_i"] - i <= roll_td:
                pnl.loc[d] -= mark * haircut * holding["shares"]
                holding = None
                # immediate re-buy below
        if want and holding is None:
            K = otm_frac * S
            T = tenor_td / 252
            price = bs_put(S, K, T, r_, put_iv(S, K, v3m_))
            shares = NAV / S
            pnl.loc[d] -= price * haircut * shares  # buy haircut
            holding = {"K": K, "expiry_i": i + tenor_td, "shares": shares,
                       "last_mark": price}
    return pnl


ao = run_put_overlay(gated=False, gate=None)
yrs_p = (dates[-1] - dates[0]).days / 365.25
print(f"\n[2] always-on 3M 5%OTM put 1xNAV 2016-07+: total ${ao.sum():,.0f} "
      f"= {ao.sum()/NAV/yrs_p*100:+.2f}%/yr of NAV")

p55 = run_put_overlay(gated=True, gate=gate55)
p50 = run_put_overlay(gated=True, gate=gate50)
print(f"[6] gated put thr55: total ${p55.sum():,.0f}; thr50: ${p50.sum():,.0f}")
# LOEO for thr55 puts
p55_ep = {}
for a, b_ in eps:
    win = p55.loc[a:b_ + pd.Timedelta(days=7)]
    p55_ep[(a, b_)] = win[win != 0].sum()
t55 = p55.sum()
for k, v in sorted(p55_ep.items()):
    print(f"    put ep {k[0].date()}..{k[1].date()}: ${v:,.0f}  LOEO ${t55 - v:,.0f}")

# ------------------------------------------------- claim 10: book curve
daily_book = tr16.groupby("Exit Date")["PnL_flat_750k"].sum().sort_index()
cumb = daily_book.cumsum()
mdd = (cumb - cumb.cummax()).min()
print(f"\n[10] book realized 2016-07+: total ${cumb.iloc[-1]:,.0f}, maxDD ${mdd:,.0f}")
bm = book_mPnL.loc["2016-07-31":"2026-06-30"]
sh_base = bm.mean() / bm.std() * np.sqrt(12)
print(f"     monthly Sharpe (2016-07..2026-06) = {sh_base:.2f}, worst month "
      f"${book_mPnL.min():,.0f} ({book_mPnL.idxmin():%Y-%m})")

# ------------------------------------------------- claim 7: gate coverage
worst12 = book_mPnL.loc[:"2026-06-30"].nsmallest(12)
gate_m = gate55.resample("ME").max().astype(bool)
print("\n[7] book 12 worst months vs gate:")
non = 0
for d, v in worst12.items():
    on = bool(gate_m.reindex([d]).fillna(False).iloc[0])
    print(f"    {d:%Y-%m}: ${v:,.0f}  gate-on-any-day={on}")
    non += on
print(f"    gate on in {non}/12")
print(f"    Volmageddon window 2018-01-15..02-28 basis max: "
      f"{basis.loc['2018-01-15':'2018-02-28'].max():.1f}")
cov_ep = [e for e in eps if e[0] < pd.Timestamp("2020-03-15") < e[1] + pd.Timedelta(days=60)]
g20 = gate55.loc["2020-02-01":"2020-04-30"]
off_dates = g20[~g20 & g20.shift(1).fillna(False)].index
print(f"    2020 gate-off date(s): {[d.date() for d in off_dates]}")
print(f"    2022-04..2022-10 basis max: {basis.loc['2022-04-01':'2022-10-31'].max():.1f}")

# ------------------------------------------------- claims 8/9: throttle + integration
sig_frag = basis.reindex(
    pd.date_range(basis.index[0], basis.index[-1], freq="D")).ffill(limit=5)


def throttle_mult(f):
    if pd.isna(f):
        return 1.0
    return 1.0 - 0.5 * min(max((f - 50) / 10, 0), 1)


t16 = tr[tr["Signal Date"] >= "2016-07-05"].copy()
fvals = sig_frag.reindex(t16["Signal Date"]).values
t16["mult"] = [1.0 if s == "Overbot Vol Spike" else throttle_mult(f)
               for s, f in zip(t16["Strategy"], fvals)]
t16["pnl_thr"] = t16["PnL_flat_750k"] * t16["mult"]
thr_cost = t16["pnl_thr"].sum() - t16["PnL_flat_750k"].sum()
print(f"\n[9] throttle cost 2016-07+ (signal-dated trades): ${thr_cost:,.0f}")

# episode book PnL (signal-dated inside window)
print("\n[8] episode book PnL (trades signaled inside gate window):")
five_worst = sorted(eps, key=lambda e: -basis.loc[e[0]:e[1]].max())[:5]
agg_b = agg_t = agg_h = 0.0
for a, b_ in eps:
    m = (t16["Signal Date"] >= a) & (t16["Signal Date"] <= b_)
    bpnl = t16.loc[m, "PnL_flat_750k"].sum()
    tpnl = t16.loc[m, "pnl_thr"].sum()
    hpnl = ep_pnl.get((a, b_), 0.0)
    tag = " *5worst*" if (a, b_) in five_worst else ""
    print(f"    {a.date()}..{b_.date()}: book ${bpnl:,.0f} thr ${tpnl:,.0f} "
          f"vxxp ${hpnl:,.0f}{tag}")
    if (a, b_) in five_worst:
        agg_b += bpnl; agg_t += tpnl; agg_h += hpnl
print(f"    5-worst agg: baseline ${agg_b:,.0f}, throttle ${agg_t:,.0f}, "
      f"throttle+hedge ${agg_t + agg_h:,.0f}")

# integration Sharpes
thr_m = t16.groupby(t16["Exit Date"].dt.to_period("M"))["pnl_thr"].sum()
thr_m.index = thr_m.index.to_timestamp("M")
sleeve_m_full = sleeve_daily.resample("ME").sum()
win = bm.index  # 2016-07..2026-06
base_m = bm
thr_mm = thr_m.reindex(win).fillna(0)
hedge_mm = sleeve_m_full.reindex(win).fillna(0)


def sh(x):
    return x.mean() / x.std() * np.sqrt(12)


print(f"\n     Sharpe baseline {sh(base_m):.2f} | +throttle {sh(thr_mm):.2f} | "
      f"throttle+VXXP {sh(thr_mm + hedge_mm):.2f} | baseline+VXXP "
      f"{sh(base_m + hedge_mm):.2f}")
cthr = t16.groupby("Exit Date")["pnl_thr"].sum().sort_index().cumsum()
print(f"     throttle curve maxDD ${(cthr - cthr.cummax()).min():,.0f}; "
      f"totals: base ${t16['PnL_flat_750k'].sum():,.0f} thr ${t16['pnl_thr'].sum():,.0f} "
      f"thr+hedge ${t16['pnl_thr'].sum() + total:,.0f}")
