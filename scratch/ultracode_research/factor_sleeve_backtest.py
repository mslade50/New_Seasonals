"""Factor-exposure sleeve prototype — ETF-implementable tilts, static vs
fragility-conditional.

Rules (stated up front):
- Prices: adjusted daily closes. Factor ETFs (MTUM/QUAL/USMV/VLUE/SPLV/SPHQ/
  RSP/SPY/BIL) from yfinance (auto_adjust=True) saved to factor_etf_prices.parquet;
  long-history defensive proxies (XLP/XLU/XLV, TLT, GLD) from
  data/master_prices.parquet (also adjusted). Last partial bar (run date) dropped.
- Rebalance: MONTHLY. Signals read at month-end close, executed at that same
  close (optimistic by one overnight; noted as bias — tested T+1-open-free since
  we only have closes, magnitude is small at this turnover).
- Costs: 5 bps per side on every dollar traded (10 bps for a full switch).
- Fragility dial: data/rd2_fragility.parquet '63d' col, dropna, 10d MA
  (live sizing basis), ffilled to calendar, read as-of month-end (limit 5d).
- Sharpe: monthly returns annualized, rf=0 (BIL CAGR reported for context).

Variants:
  Static: SPY, USMV, QUAL, MTUM, VLUE, SPLV, MQUV blend (25% each),
          defensive-proxy EW(XLP,XLU,XLV) for 2003+ context, RSP, SPHQ.
  Conditional: hold BASE; when dial >= THR at month-end, hold DEF next month.
          Grid: THR in {40,44,50,55}, DEF in {USMV, TLT, BIL}, BASE in {SPY, MQUV}.
  Daily variant: enter DEF when dial >= 50, exit < 45 (hysteresis), T+1 close.

Outputs: perf table, per-year table, correlation to book monthly R,
high-fragility-month behavior (63d MA10 >= 50 month-mean, 2016-08+), LOYO.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN_DATE = pd.Timestamp("2026-07-02")
COST_PER_SIDE = 0.0005  # 5 bps

# ---------------------------------------------------------------- prices
fac = pd.read_parquet(HERE / "factor_etf_prices.parquet")
fac.index = pd.to_datetime(fac.index).normalize()
fac = fac[fac.index < RUN_DATE]  # drop partial run-date bar

mp = pd.read_parquet(ROOT / "data" / "master_prices.parquet",
                     columns=["ticker", "date", "Close"])
mp["date"] = pd.to_datetime(mp["date"]).dt.normalize()
prox = (mp[mp["ticker"].isin(["XLP", "XLU", "XLV", "TLT", "GLD", "SPY"])]
        .pivot(index="date", columns="ticker", values="Close")
        .sort_index())
prox = prox[prox.index < RUN_DATE]

# sanity: master_prices SPY vs yfinance SPY (both adjusted) should track
chk = pd.concat([fac["SPY"].rename("yf"), prox["SPY"].rename("mp")], axis=1).dropna()
ratio = (chk["yf"] / chk["mp"])
print(f"[sanity] SPY yf/mp ratio drift last 3y: "
      f"{ratio.iloc[-756:].std() / ratio.iloc[-756:].mean():.5f} (want ~0)")

px = fac.join(prox[["XLP", "XLU", "XLV", "TLT", "GLD"]], how="outer").sort_index()
mret = px.resample("ME").last().pct_change()  # monthly returns per instrument

# ---------------------------------------------------------------- fragility
frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
frag.index = pd.to_datetime(frag.index).normalize()
dial = frag["63d"].dropna().rolling(10, min_periods=1).mean()
dial_daily = dial.reindex(pd.date_range(dial.index.min(), dial.index.max()),
                          method="ffill", limit=5)
dial_me = dial_daily.resample("ME").last()          # month-end reading
dial_mmean = dial_daily.resample("ME").mean()       # month-average (for labeling)

FRAG_START = pd.Period("2016-09", "M")  # first full month after 10d-MA warmup

# ---------------------------------------------------------------- book series
tr = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
tr["Exit Date"] = pd.to_datetime(tr["Exit Date"])
book_m = (tr.groupby(tr["Exit Date"].dt.to_period("M"))["PnL_flat_750k"].sum()
          / 750_000.0)
book_m.name = "book"


# ---------------------------------------------------------------- engines
def perf(r: pd.Series, name: str) -> dict:
    r = r.dropna()
    if len(r) < 12:
        return {}
    eq = (1 + r).cumprod()
    yrs = len(r) / 12.0
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(12)
    dd = (eq / eq.cummax() - 1).min()
    return {"name": name, "start": str(r.index[0]), "end": str(r.index[-1]),
            "n_mo": len(r), "CAGR%": 100 * cagr, "Vol%": 100 * vol,
            "Sharpe": (r.mean() * 12) / (vol if vol > 0 else np.nan),
            "MaxDD%": 100 * dd}


def static_blend(weights: dict, start=None) -> pd.Series:
    """Monthly-rebalanced fixed-weight blend with drift-based rebal costs."""
    cols = list(weights)
    r = mret[cols].dropna()
    if start:
        r = r[r.index.to_period("M") >= start]
    w = pd.Series(weights, dtype=float)
    out = []
    for dt, row in r.iterrows():
        gross = float((w * row).sum())
        w_drift = w * (1 + row) / (1 + gross)
        cost = float((w - w_drift).abs().sum()) * COST_PER_SIDE  # trade back to target
        out.append(gross - cost)
    return pd.Series(out, index=r.index.to_period("M"))


def conditional(base: str | dict, defensive: str, thr: float,
                start=FRAG_START) -> tuple[pd.Series, float, int]:
    """Hold base; if month-end dial >= thr, hold defensive next month.
    base may be a ticker or a weight dict (blend held statically when risk-on).
    Returns (monthly returns, frac months defensive, n switches)."""
    if isinstance(base, str):
        base_r = mret[base]
    else:
        base_r = static_blend(base)
        base_r.index = base_r.index.to_timestamp("M")  # align to ME stamps
        base_r = base_r.reindex(mret.index)
    def_r = mret[defensive]
    sig = (dial_me >= thr).astype(float)          # decided at month-end t
    hold_def = sig.shift(1).reindex(mret.index)   # applied over month t+1
    df = pd.DataFrame({"b": base_r, "d": def_r, "h": hold_def}).dropna()
    df = df[df.index.to_period("M") >= start]
    ret = np.where(df["h"] == 1, df["d"], df["b"])
    switches = df["h"].diff().abs().fillna(0)
    cost = switches * 2 * COST_PER_SIDE           # full switch = sell+buy
    r = pd.Series(ret - cost.values, index=df.index.to_period("M"))
    return r, float(df["h"].mean()), int(switches.sum())


def daily_hysteresis(base: str, defensive: str, hi=50.0, lo=45.0) -> tuple[pd.Series, float, int]:
    """Daily: enter defensive when dial >= hi, exit when < lo. T+1 close exec."""
    both = px[[base, defensive]].dropna()
    d = dial.reindex(both.index, method="ffill", limit=5)
    state = np.zeros(len(both))
    s = 0
    for i, v in enumerate(d.values):
        if np.isnan(v):
            state[i] = s
            continue
        if s == 0 and v >= hi:
            s = 1
        elif s == 1 and v < lo:
            s = 0
        state[i] = s
    hold = pd.Series(state, index=both.index).shift(1).fillna(0)  # T+1
    rets = both.pct_change()
    r = np.where(hold == 1, rets[defensive], rets[base])
    switch = hold.diff().abs().fillna(0)
    r = pd.Series(r - switch.values * 2 * COST_PER_SIDE, index=both.index)
    r = r[r.index >= "2016-08-01"]
    rm = (1 + r).resample("ME").prod() - 1
    rm.index = rm.index.to_period("M")
    return rm, float(hold[hold.index >= "2016-08-01"].mean()), int(switch[switch.index >= "2016-08-01"].sum())


# ---------------------------------------------------------------- run statics
rows = []
STATICS = {
    "SPY (b&h)": {"SPY": 1.0},
    "USMV": {"USMV": 1.0},
    "QUAL": {"QUAL": 1.0},
    "MTUM": {"MTUM": 1.0},
    "VLUE": {"VLUE": 1.0},
    "SPLV": {"SPLV": 1.0},
    "SPHQ": {"SPHQ": 1.0},
    "RSP": {"RSP": 1.0},
    "MQUV blend 25x4": {"MTUM": .25, "QUAL": .25, "USMV": .25, "VLUE": .25},
    "EW XLP/XLU/XLV": {"XLP": 1 / 3, "XLU": 1 / 3, "XLV": 1 / 3},
    "BIL (cash ref)": {"BIL": 1.0},
}
static_series = {}
for nm, w in STATICS.items():
    s = static_blend(w)
    static_series[nm] = s
    rows.append(perf(s, nm))
print("\n=== STATIC (max available history, monthly rebal, 5bps/side) ===")
print(pd.DataFrame(rows).round(2).to_string(index=False))

# common-window comparison 2013-08+ (all four factors live)
rows = []
CW = pd.Period("2013-08", "M")
for nm in STATICS:
    s = static_series[nm]
    rows.append(perf(s[s.index >= CW], nm))
print("\n=== STATIC, common window 2013-08+ ===")
print(pd.DataFrame(rows).round(2).to_string(index=False))

# ---------------------------------------------------------------- conditionals
print("\n=== CONDITIONAL (2016-09+): hold base, month-end dial>=thr -> defensive next month ===")
rows = []
for base_nm, base in [("SPY", "SPY"),
                      ("MQUV", {"MTUM": .25, "QUAL": .25, "USMV": .25, "VLUE": .25})]:
    for dfn in ["USMV", "TLT", "BIL"]:
        for thr in [40, 44, 50, 55]:
            r, fdef, nsw = conditional(base, dfn, thr)
            p = perf(r, f"{base_nm}->{dfn} thr{thr}")
            p["%def"] = 100 * fdef
            p["switches"] = nsw
            rows.append(p)
# static references over identical window
for nm in ["SPY (b&h)", "USMV", "MQUV blend 25x4"]:
    s = static_series[nm]
    p = perf(s[s.index >= FRAG_START], nm + " [ref]")
    p["%def"] = np.nan
    p["switches"] = 0
    rows.append(p)
cond_tbl = pd.DataFrame(rows).drop(columns=["start", "end"])
print(cond_tbl.round(2).to_string(index=False))

# daily hysteresis variant
rm, fdef, nsw = daily_hysteresis("SPY", "USMV")
p = perf(rm, "SPY->USMV daily hyst 50/45")
p["%def"] = 100 * fdef
p["switches"] = nsw
print("\ndaily-hysteresis variant:")
print(pd.DataFrame([p]).round(2).to_string(index=False))

# ---------------------------------------------------------------- per-year table
def yearly(r: pd.Series) -> pd.Series:
    r = r.dropna()
    return r.groupby(r.index.year).apply(lambda s: 100 * ((1 + s).prod() - 1))

main = {
    "SPY": static_series["SPY (b&h)"],
    "USMV": static_series["USMV"],
    "MQUV": static_series["MQUV blend 25x4"],
    "EW_defsec": static_series["EW XLP/XLU/XLV"],
    "SPY->USMV t50": conditional("SPY", "USMV", 50)[0],
    "SPY->BIL t50": conditional("SPY", "BIL", 50)[0],
}
yt = pd.DataFrame({k: yearly(v) for k, v in main.items()})
print("\n=== PER-YEAR RETURNS (%) ===")
print(yt.round(1).to_string())

# ---------------------------------------------------------------- book correlation
print("\n=== CORRELATION to book monthly return (PnL_flat_750k / 750k, exit month) ===")
for k, v in main.items():
    j = pd.concat([v, book_m], axis=1).dropna()
    if len(j) < 12:
        continue
    c_full = j.corr().iloc[0, 1]
    j16 = j[j.index >= FRAG_START]
    c16 = j16.corr().iloc[0, 1]
    print(f"{k:16s} full-overlap corr={c_full:+.2f} (N={len(j)} mo)   2016-09+ corr={c16:+.2f} (N={len(j16)})")
# active return of timing overlay vs book
act = (main["SPY->USMV t50"] - main["SPY"].reindex(main["SPY->USMV t50"].index)).dropna()
ja = pd.concat([act.rename("act"), book_m], axis=1).dropna()
print(f"timing ACTIVE ret (rot minus SPY) corr to book: {ja.corr().iloc[0, 1]:+.2f} (N={len(ja)})")

# ---------------------------------------------------------------- high-frag months
print("\n=== HIGH-FRAGILITY MONTHS (month-mean 63d MA10 >= 50, 2016-09+) ===")
hi_mo = dial_mmean[dial_mmean >= 50].index.to_period("M")
hi_mo = hi_mo[hi_mo >= FRAG_START]
print(f"N high-frag months: {len(hi_mo)}  -> {list(hi_mo.astype(str))}")
cmp_rows = []
for k, v in {**main, "book": book_m}.items():
    vv = v[v.index >= FRAG_START]
    hi = vv[vv.index.isin(hi_mo)]
    lo = vv[~vv.index.isin(hi_mo)]
    cmp_rows.append({"series": k, "hiN": len(hi), "hi_avg%": 100 * hi.mean(),
                     "hi_med%": 100 * hi.median(), "hi_tot%": 100 * ((1 + hi).prod() - 1),
                     "lo_avg%": 100 * lo.mean()})
print(pd.DataFrame(cmp_rows).round(2).to_string(index=False))

# entered-at-high variant: month started with dial >= 50 (prior month-end)
hi_mo2 = dial_me[dial_me.shift(0) >= 50].index.to_period("M") + 1
hi_mo2 = hi_mo2[hi_mo2 >= FRAG_START]
print(f"\nalt def — months FOLLOWING a >=50 month-end reading: N={len(hi_mo2)} {list(hi_mo2.astype(str))}")
for k, v in {**main, "book": book_m}.items():
    hi = v[v.index.isin(hi_mo2)].dropna()
    print(f"  {k:14s} avg={100*hi.mean():+.2f}%  N={len(hi)}")

# ---------------------------------------------------------------- LOYO on timing
print("\n=== LOYO: SPY->USMV thr50 active return (vs SPY), 2016-09+ ===")
act_full = act.dropna()
print(f"all: mean act={100*act_full.mean():+.3f}%/mo, tot={100*((1+act_full).prod()-1):+.1f}%, N={len(act_full)}")
for y in sorted(set(act_full.index.year)):
    a = act_full[act_full.index.year != y]
    print(f"  drop {y}: mean={100*a.mean():+.3f}%/mo  tot={100*((1+a).prod()-1):+.1f}%")

# does USMV-SPY monthly spread relate to dial level at all? (regression, no rule)
spread = (mret["USMV"] - mret["SPY"]).dropna()
spread.index = spread.index.to_period("M")
dlev = dial_me.copy()
dlev.index = dlev.index.to_period("M")
jj = pd.concat([spread.rename("sp"), dlev.shift(1).rename("dial")], axis=1).dropna()
from scipy import stats
sl, ic, rv, pv, se = stats.linregress(jj["dial"], jj["sp"])
print(f"\nUSMV-SPY next-month spread vs prior month-end dial: slope={100*sl:+.4f}%/pt "
      f"r={rv:+.2f} p={pv:.3f} N={len(jj)}")
