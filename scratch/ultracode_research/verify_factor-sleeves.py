"""Adversarial verification of the factor-sleeves study.

Independent recompute: fresh yfinance data (verify_fs_prices.parquet, fetched by
verify_fetch_fs.py), own monthly-return / dial / episode machinery. Nothing reused
from the researcher's scripts.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

root = Path(r"C:\Users\McKinley Slade\dev\New_Seasonals")
ur = root / "scratch" / "ultracode_research"

px = pd.read_parquet(ur / "verify_fs_prices.parquet")
px = px.loc[:"2026-06-30"]  # drop July partial month

mret = px.resample("ME").last().pct_change()
mret = mret.loc["2000-02":"2026-06"]

def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(12))

def cagr(r: pd.Series) -> float:
    r = r.dropna()
    yrs = len(r) / 12
    return float((1 + r).prod() ** (1 / yrs) - 1)

def maxdd(r: pd.Series) -> float:
    w = (1 + r.dropna()).cumprod()
    return float((w / w.cummax() - 1).min())

print("=" * 70)
print("CLAIM 1: common-window 2013-08+ Sharpes")
cw = mret.loc["2013-08":]
mquv = cw[["MTUM", "QUAL", "USMV", "VLUE"]].mean(axis=1)
for name, s in [("SPY", cw["SPY"]), ("USMV", cw["USMV"]), ("QUAL", cw["QUAL"]),
                ("MTUM", cw["MTUM"]), ("VLUE", cw["VLUE"]), ("SPHQ", cw["SPHQ"]),
                ("MQUV", mquv)]:
    print(f"  {name:5s} Sharpe {sharpe(s):.2f}  CAGR {cagr(s)*100:5.1f}%  N={s.dropna().shape[0]}")

# ---------------- fragility dial ----------------
frag = pd.read_parquet(root / "data" / "rd2_fragility.parquet")
dial = frag["63d"].rolling(10, min_periods=1).mean()

dial_me = dial.resample("ME").last()          # month-end reading (signal)
dial_mmean = dial.resample("ME").mean()       # month-mean (for high-frag month def)

print("=" * 70)
print("CLAIM 2: high-fragility months (month-mean dial >= 50), 2016-09+")
hf = dial_mmean.loc["2016-09":"2026-06"]
hf_months = hf[hf >= 50].index
print(f"  N high-frag months = {len(hf_months)}")
print("  months:", [m.strftime("%Y-%m") for m in hf_months])

led = pd.read_parquet(root / "data" / "backtest_trades_full.parquet")
book_pnl = led.groupby(pd.to_datetime(led["Exit Date"]).dt.to_period("M"))["PnL_flat_750k"].sum()
book = (book_pnl / 750_000).to_timestamp("M") if hasattr(book_pnl.index, "to_timestamp") else book_pnl
book.index = book_pnl.index.to_timestamp(how="end").normalize()
book = book.reindex(pd.date_range("2003-01-31", "2026-06-30", freq="ME")).fillna(0.0)

win = mret.loc["2016-09":"2026-06"]
mquv_w = win[["MTUM", "QUAL", "USMV", "VLUE"]].mean(axis=1)
book_w = book.loc["2016-09-01":"2026-06-30"]

is_hf = win.index.isin(hf_months)
for name, s in [("SPY", win["SPY"]), ("USMV", win["USMV"]), ("MQUV", mquv_w),
                ("book", book_w)]:
    s = s.copy()
    s.index = s.index.normalize()
    hf_mask = pd.Series(s.index.isin(hf_months.normalize()), index=s.index)
    print(f"  {name:5s} high-frag {s[hf_mask].mean()*100:+.2f}%/mo (N={hf_mask.sum()}) "
          f"| other {s[~hf_mask].mean()*100:+.2f}%/mo")

print("=" * 70)
print("CLAIM 3: combined book + 0.5x sleeve, 2016-09+ monthly")
bw = book_w.copy()
bw.index = bw.index.normalize()
print(f"  book alone: avg {bw.mean()*100:+.2f}%/mo Sharpe {sharpe(bw):.2f} maxDD {maxdd(bw)*100:.1f}%")
for name, s in [("SPY", win["SPY"]), ("USMV", win["USMV"]), ("MQUV", mquv_w)]:
    s = s.copy(); s.index = s.index.normalize()
    comb = bw + 0.5 * s.reindex(bw.index)
    print(f"  book+0.5x {name:5s}: avg {comb.mean()*100:+.2f}%/mo Sharpe {sharpe(comb):.2f} "
          f"maxDD {maxdd(comb)*100:.1f}%")

# ---------------- fragility-timed rotation ----------------
COST = 0.0010  # 10 bps per full switch

def rotation(base: str, defn: str, thr: float, w: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Position for month m set by dial month-end reading at m-1."""
    sig = dial_me.reindex(w.index.normalize(), method=None)
    sig = dial_me.shift(1).reindex(w.index)  # prior month-end reading
    defensive = (sig >= thr).fillna(False)
    r = w[base].copy()
    r[defensive] = w[defn][defensive]
    switch = defensive != defensive.shift(1).fillna(False)
    r = r - switch.astype(float) * COST
    return r, defensive

def episode_stats(active: pd.Series, defensive: pd.Series, label: str) -> None:
    # group contiguous defensive months into episodes; attach the exit-cost month
    eps = []
    cur = []
    for i, (dt, d) in enumerate(defensive.items()):
        if d:
            cur.append(dt)
        elif cur:
            eps.append(cur); cur = []
    if cur:
        eps.append(cur)
    sums = []
    print(f"  {label}: {int(defensive.sum())} defensive months in {len(eps)} episodes")
    for ep in eps:
        a = active.loc[ep[0]:ep[-1]].sum()
        # add exit-cost month (first non-defensive month after the episode)
        after = active.index[active.index > ep[-1]]
        if len(after):
            a += active.loc[after[0]]
        sums.append(a)
        print(f"    {ep[0]:%Y-%m}..{ep[-1]:%Y-%m} n={len(ep)} active {a*100:+.1f}%")
    sums = np.array(sums)
    t = sums.mean() / (sums.std(ddof=1) / np.sqrt(len(sums)))
    p = 2 * stats.t.sf(abs(t), len(sums) - 1)
    print(f"    episode-clustered t={t:+.2f} p={p:.3f} (N={len(sums)}), "
          f"total active {sums.sum()*100:+.1f}%")
    # monthly t on defensive-month actives
    am = active[defensive]
    tm = am.mean() / (am.std(ddof=1) / np.sqrt(len(am)))
    pm = 2 * stats.t.sf(abs(tm), len(am) - 1)
    print(f"    monthly t={tm:+.2f} p={pm:.3f} (N={len(am)})")

print("=" * 70)
print("CLAIMS 4+5: fragility-timed rotation, 2016-09+ (thr=50, 10bps/switch)")
for defn in ["USMV", "BIL"]:
    r, d = rotation("SPY", defn, 50, win)
    active = r - win["SPY"]
    print(f" SPY->{defn} thr50: Sharpe {sharpe(r):.2f} (SPY {sharpe(win['SPY']):.2f}) "
          f"CAGR {cagr(r)*100:.1f}% vs SPY {cagr(win['SPY'])*100:.1f}% "
          f"maxDD {maxdd(r)*100:.1f}% | %def {d.mean()*100:.0f}%")
    print(f"   total active (sum) {active.sum()*100:+.1f}% | "
          f"terminal wealth diff {((1+r).prod()-(1+win['SPY']).prod())*100:+.1f}pp")
    episode_stats(active, d, f"SPY->{defn}")
    # LOYO on total active
    loyo = {}
    for yr in sorted(set(active.index.year)):
        loyo[yr] = active[active.index.year != yr].sum() * 100
    worst = min(loyo, key=loyo.get)
    print(f"   LOYO total-active: full {active.sum()*100:+.1f}% | "
          + " ".join(f"ex{y}:{v:+.1f}" for y, v in loyo.items()))

print("=" * 70)
print("CLAIM 6: correlations to book, 2016-09+")
for name, s in [("SPY", win["SPY"]), ("USMV", win["USMV"]), ("MQUV", mquv_w)]:
    s = s.copy(); s.index = s.index.normalize()
    print(f"  corr(book, {name}) = {bw.corr(s.reindex(bw.index)):+.2f}")
r_bil, d_bil = rotation("SPY", "BIL", 50, win)
act = (r_bil - win["SPY"]).copy(); act.index = act.index.normalize()
print(f"  corr(book, BIL-rotation ACTIVE) = {bw.corr(act.reindex(bw.index)):+.2f}")
# full-history SPY corr
spy_all = mret["SPY"].copy(); spy_all.index = spy_all.index.normalize()
bfull = book.copy(); bfull.index = bfull.index.normalize()
ix = spy_all.index.intersection(bfull.index)
print(f"  corr(book, SPY) full 2003+ = {bfull[ix].corr(spy_all[ix]):+.2f} (N={len(ix)})")
