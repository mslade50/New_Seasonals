"""C10 recon: what IS the date convention in earnings_calendar.parquet?

Never assume. Three tests:
 1. spot-check the four live names against known 2026-08 report dates
 2. how many rows land on a fiscal quarter-end (the pre-2016 FMP style)
 3. cross-sectional |return| on day -1 / 0 / +1 / +2 vs the name's own norm --
    a BMO print prints its reaction on day 0, an AMC print on day +1.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

pd.set_option("display.width", 220)

e = pd.read_parquet(ROOT / "data" / "earnings_calendar.parquet")
print("rows", len(e), "tickers", e.ticker.nunique())

print("\n--- 1. the four live names, 2026-08 ---")
sel = e[(e.date >= "2026-08-01") & (e.date <= "2026-08-31")
        & e.ticker.isin(["WMT", "TJX", "ROST", "TGT"])]
print(sel.to_string(index=False))

print("\n--- 2. rows landing on a quarter-end date ---")
qe = e.date.dt.is_quarter_end
print(f" quarter-end rows: {int(qe.sum())} of {len(e)}  "
      f"({100*qe.mean():.1f}%);  with eps_est present: {int((qe & e.eps_est.notna()).sum())}")
print(" by year:")
print(e.assign(qe=qe).groupby(e.date.dt.year)["qe"].mean().mul(100).round(1).tail(20).to_string())
print(f" rows with eps_est NaN: {int(e.eps_est.isna().sum())}  "
      f"({100*e.eps_est.isna().mean():.1f}%)")

# --- 3. reaction-day location ------------------------------------------------
mp = pd.read_parquet(ROOT / "data" / "master_prices.parquet")
mp["date"] = pd.to_datetime(mp["date"])
cal = pd.DatetimeIndex(sorted(mp.loc[mp.ticker == "SPY", "date"].unique()))
px = mp.pivot_table(index="date", columns="ticker", values="Close", aggfunc="last")
px = px.reindex(cal)
vol = mp.pivot_table(index="date", columns="ticker", values="Volume", aggfunc="last").reindex(cal)
ret = px.pct_change(fill_method=None)
print(f"\n price panel {px.index[0].date()}..{px.index[-1].date()}  tickers {px.shape[1]}")

ev = e[e.eps_est.notna() & e.eps_surprise_pct.notna()].copy()
ev = ev[ev.ticker.isin(px.columns)]
ev = ev[(ev.date >= px.index[0]) & (ev.date <= px.index[-1])]
pos = pd.Series(range(len(cal)), index=cal)
# snap each event date to the next trading day on or after it
snap = cal[np.searchsorted(cal.values, ev.date.values, side="left").clip(0, len(cal) - 1)]
ev["p"] = pos.reindex(snap).values
ev = ev.dropna(subset=["p"])
ev["p"] = ev["p"].astype(int)
print(f" usable events: {len(ev)}  tickers {ev.ticker.nunique()}  "
      f"{ev.date.min().date()}..{ev.date.max().date()}")

rows = []
for off in (-2, -1, 0, 1, 2, 3):
    a, rv = [], []
    for t, g in ev.groupby("ticker"):
        r = ret[t]
        base = r.abs().rolling(63).median()
        v = vol[t]
        vbase = v.rolling(63).median()
        idx = (g.p.values + off)
        idx = idx[(idx >= 0) & (idx < len(cal))]
        d0 = cal[idx]
        a.append((r.reindex(d0).abs() / base.reindex(d0)).values)
        rv.append((v.reindex(d0) / vbase.reindex(d0)).values)
    a = np.concatenate(a); rv = np.concatenate(rv)
    rows.append({"offset": off, "n": int(np.isfinite(a).sum()),
                 "|ret| / own 63d median |ret|": round(float(np.nanmedian(a)), 3),
                 "volume / own 63d median": round(float(np.nanmedian(rv)), 3)})
show(rows, "3. where does the print reaction land relative to the parquet date?")
