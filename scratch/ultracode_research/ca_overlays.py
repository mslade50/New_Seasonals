"""Crisis-alpha track: overlay prototypes.

Overlays tested (all sized as % of a flat $750k book NAV, daily sim):

1. Tactical long-vol: de-levered UVXY ("VXX-proxy", 1x short-term VIX futures)
   gated by frag63_ma10 >= T (T in 50/55/60). Enter next session after the gate
   turns on, exit next session after it turns off. Also raw UVXY, and always-on
   baselines to show roll bleed. Cost 10 bps/side.
2. Tactical / permanent GLD, TLT, IEF. Cost 5 bps/side.
3. BS-priced SPY put overlay: 3M 5%-OTM puts on 1x NAV notional, gated at T,
   IV = VIX3M + skew adjustment, 5% premium haircut each way. Put-spread variant
   (long 95%, short 85%) sells the richer wing to finance.

Conventions: signal at close t -> position effective day t+1 (close-to-close
returns), matching stage-next-open live workflow. Sleeve returns reported as
% of the flat 750k base. Gate uses 5-point hysteresis on exit (off below T-5)
to avoid churn; no-hysteresis variant checked.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
NAV = 750_000.0

panel = pd.read_parquet(HERE / "ca_prices.parquet")
close = panel["Close"]
frag = pd.read_parquet(HERE / "ca_frag.parquet")["frag63_ma10"]
book_mon = pd.read_parquet(HERE / "ca_book_monthly.parquet")
if not isinstance(book_mon.index, pd.PeriodIndex):
    # parquet round-trips PeriodIndex as int64 ordinals (months since 1970-01)
    book_mon.index = pd.PeriodIndex.from_ordinals(book_mon.index.values, freq="M")

# ---------------------------------------------------------------- instruments
ret = close.pct_change()

# de-levered UVXY -> 1x short-term VIX futures proxy ("VXXP")
lev = pd.Series(np.where(close.index < pd.Timestamp("2018-02-28"), 2.0, 1.5),
                index=close.index)
ret["VXXP"] = ret["UVXY"] / lev

rf_daily = (close["^IRX"] / 100 / 252).reindex(close.index).ffill().fillna(0)


def perf_stats(daily_ret: pd.Series, label: str) -> dict:
    r = daily_ret.dropna()
    if len(r) < 60:
        return {}
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(252)
    ex = r - rf_daily.reindex(r.index).fillna(0)
    sharpe = ex.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan
    dd = (eq / eq.cummax() - 1).min()
    return {"label": label, "start": r.index.min().date(), "end": r.index.max().date(),
            "CAGR%": cagr * 100, "Vol%": vol * 100, "Sharpe": sharpe, "MaxDD%": dd * 100,
            "TotRet%": (eq.iloc[-1] - 1) * 100}


# --------------------------------------------------------- tactical ETF sleeve
def tactical_sleeve(asset: str, thr: float, weight: float, cost_bps: float,
                    hysteresis: float = 5.0, always_on: bool = False) -> pd.Series:
    """Daily sleeve return (% of NAV). Gate known at close t, held day t+1."""
    a_ret = ret[asset].dropna()
    idx = a_ret.index.intersection(frag.index)
    if always_on:
        pos = pd.Series(1.0, index=a_ret.index)
        idx = a_ret.index
    else:
        f = frag.reindex(idx).ffill()
        raw_on = pd.Series(np.nan, index=idx)
        raw_on[f >= thr] = 1.0
        raw_on[f < thr - hysteresis] = 0.0
        gate = raw_on.ffill().fillna(0.0)
        pos = gate.shift(1).fillna(0.0)  # effective next session
    a = a_ret.reindex(pos.index).fillna(0.0)
    sleeve = pos * a * weight
    turnover = pos.diff().abs().fillna(pos.abs()) * weight
    sleeve -= turnover * cost_bps / 1e4
    return sleeve


# ------------------------------------------------------------- BS put overlay
def bs_put(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def skew_iv(vix3m_pct: float, moneyness: float) -> float:
    """IV with a linear skew: +4 vol pts per 10% below spot. 5% OTM -> +2, 15% -> +6."""
    otm = max(0.0, 1.0 - moneyness)
    return vix3m_pct / 100.0 + 0.40 * otm


def put_overlay(thr: float, spread: bool, notional_frac: float = 1.0,
                tenor_td: int = 63, otm: float = 0.05, otm_short: float = 0.15,
                haircut: float = 0.05, hysteresis: float = 5.0,
                always_on: bool = False, roll_before_td: int = 5) -> pd.Series:
    """Daily PnL (% of NAV) of a gated 3M put (or put spread) on 1x NAV notional."""
    spy = close["SPY"].dropna()
    vix3m = close["^VIX3M"].reindex(spy.index).ffill()
    irx = (close["^IRX"] / 100).reindex(spy.index).ffill().fillna(0.04)
    idx = spy.index[spy.index >= (frag.index.min() if not always_on else vix3m.first_valid_index())]
    idx = idx[idx >= vix3m.first_valid_index()]

    if always_on:
        gate = pd.Series(1.0, index=idx)
    else:
        f = frag.reindex(idx).ffill()
        raw_on = pd.Series(np.nan, index=idx)
        raw_on[f >= thr] = 1.0
        raw_on[f < thr - hysteresis] = 0.0
        gate = raw_on.ffill().fillna(0.0)
    want = gate.shift(1).fillna(0.0)  # act next session

    pnl = pd.Series(0.0, index=idx)
    pos = None  # dict: K_long, K_short, expiry_i, n_shares, mark
    dates = list(idx)
    for i, d in enumerate(dates):
        S = spy.loc[d]
        iv_base = vix3m.loc[d]
        r = irx.loc[d]

        def price_leg(K, days_left):
            T = max(days_left, 0) / 252.0
            sig = skew_iv(iv_base, K / S)
            return bs_put(S, K, T, r, sig)

        if pos is not None:
            days_left = pos["expiry_i"] - i
            mark = price_leg(pos["K_long"], days_left)
            if spread:
                mark -= price_leg(pos["K_short"], days_left)
            pnl.loc[d] += (mark - pos["mark"]) * pos["n_shares"] / NAV
            pos["mark"] = mark
            # exit / expiry / roll
            if want.loc[d] == 0:
                pnl.loc[d] -= abs(mark) * haircut * pos["n_shares"] / NAV
                pos = None
            elif days_left <= 0:
                pos = None  # settled at intrinsic via mark
            elif days_left <= roll_before_td:
                pnl.loc[d] -= abs(mark) * haircut * pos["n_shares"] / NAV
                pos = None  # roll: re-enter below

        if pos is None and want.loc[d] == 1:
            K_long = round(S * (1 - otm))
            K_short = round(S * (1 - otm_short))
            prem = price_leg(K_long, tenor_td)
            if spread:
                prem -= price_leg(K_short, tenor_td)
            n_shares = notional_frac * NAV / S
            cost = abs(prem) * haircut
            pnl.loc[d] -= cost * n_shares / NAV
            pos = {"K_long": K_long, "K_short": K_short, "expiry_i": i + tenor_td,
                   "n_shares": n_shares, "mark": prem}
    return pnl


# ---------------------------------------------------------------- evaluation
def monthly(s: pd.Series) -> pd.Series:
    return s.dropna().groupby(s.dropna().index.to_period("M")).sum()


def eval_overlay(name: str, sleeve: pd.Series, start=None) -> dict:
    s = sleeve.dropna()
    if start:
        s = s[s.index >= start]
    mon = monthly(s)
    # align with book
    common = mon.index.intersection(book_mon.index)
    common = common[(common >= pd.Period("2016-08")) & (common <= pd.Period("2026-06"))]
    m = mon.reindex(common).fillna(0)
    b = book_mon["ret"].reindex(common).fillna(0)
    corr = m.corr(b) if len(common) > 12 else np.nan
    # high-frag months: month mean of frag >= 50
    fm = frag.groupby(frag.index.to_period("M")).mean()
    hi_months = fm[fm >= 50].index.intersection(common)
    lo_months = common.difference(hi_months)
    out = {
        "name": name, "N_mo": len(common),
        "ann_ret_%": m.mean() * 12 * 100,
        "ann_vol_%": m.std() * np.sqrt(12) * 100,
        "corr_book": corr,
        "hiFrag_N": len(hi_months),
        "hiFrag_avg_%/mo": m.reindex(hi_months).mean() * 100 if len(hi_months) else np.nan,
        "hiFrag_hit_%": (m.reindex(hi_months) > 0).mean() * 100 if len(hi_months) else np.nan,
        "calm_avg_%/mo": m.reindex(lo_months).mean() * 100,
        "tot_ret_%": m.sum() * 100,
    }
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    results = []

    # --- 1. always-on baselines (the honest bleed) -------------------------
    print("=" * 100)
    print("ALWAYS-ON baselines (buy & hold, max available window) — the roll-decay reality check")
    print("=" * 100)
    rows = []
    for asset, label in [("VXXP", "VXX-proxy (1x ST VIX fut, de-levered UVXY, 2011-10+)"),
                         ("UVXY", "UVXY raw (2x->1.5x, 2011-10+)"),
                         ("GLD", "GLD (2004-11+)"), ("TLT", "TLT (2002-07+)"),
                         ("IEF", "IEF (2002-07+)"), ("DBC", "DBC (2006-02+)")]:
        rows.append(perf_stats(ret[asset].dropna(), label))
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    ao_put = put_overlay(thr=0, spread=False, always_on=True)
    ao_stats = perf_stats(ao_put + 0, "always-on 3M 5%OTM put (1x NAV, %NAV PnL)")
    print("\nAlways-on 5% OTM 3M put on 1x NAV (2016-07+ window, %/yr of NAV): "
          f"{monthly(ao_put).mean()*12*100:+.2f}%/yr")

    # --- 2. tactical overlays, gated 2016-07+ ------------------------------
    print("\n" + "=" * 100)
    print("TACTICAL overlays gated by frag63_MA10 (2016-08..2026-06 monthly eval)")
    print("=" * 100)
    specs = []
    for thr in (50, 55, 60):
        specs.append((f"VXX-proxy 2%NAV thr{thr}", tactical_sleeve("VXXP", thr, 0.02, 10)))
        specs.append((f"VXX-proxy 5%NAV thr{thr}", tactical_sleeve("VXXP", thr, 0.05, 10)))
    specs.append(("UVXY 2%NAV thr55", tactical_sleeve("UVXY", 55, 0.02, 10)))
    for thr in (50, 55):
        specs.append((f"TLT 20%NAV thr{thr}", tactical_sleeve("TLT", thr, 0.20, 5)))
        specs.append((f"GLD 10%NAV thr{thr}", tactical_sleeve("GLD", thr, 0.10, 5)))
    specs.append(("IEF 20%NAV thr55", tactical_sleeve("IEF", 55, 0.20, 5)))
    specs.append(("GLD 10%NAV always", tactical_sleeve("GLD", 0, 0.10, 5, always_on=True)))
    specs.append(("TLT 20%NAV always", tactical_sleeve("TLT", 0, 0.20, 5, always_on=True)))

    for thr in (50, 55):
        specs.append((f"Put 5%OTM 1xNAV thr{thr}", put_overlay(thr=thr, spread=False)))
        specs.append((f"PutSpread 95/85 1xNAV thr{thr}", put_overlay(thr=thr, spread=True)))

    for name, sleeve in specs:
        results.append(eval_overlay(name, sleeve))
    df = pd.DataFrame(results)
    print(df.round(2).to_string(index=False))

    # save the sleeves for the integration script
    keep = {
        "vxxp2_55": tactical_sleeve("VXXP", 55, 0.02, 10),
        "vxxp5_55": tactical_sleeve("VXXP", 55, 0.05, 10),
        "put_55": put_overlay(thr=55, spread=False),
        "putspread_55": put_overlay(thr=55, spread=True),
        "put_50": put_overlay(thr=50, spread=False),
        "tlt20_55": tactical_sleeve("TLT", 55, 0.20, 5),
        "gld10_55": tactical_sleeve("GLD", 55, 0.10, 5),
    }
    pd.DataFrame(keep).to_parquet(HERE / "ca_sleeves.parquet")
    print("\nsaved ca_sleeves.parquet")
