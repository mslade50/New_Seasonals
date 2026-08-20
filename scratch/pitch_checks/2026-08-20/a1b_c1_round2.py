"""C1 round 2: behaviour AT today's extreme, gradient, concentration,
survivorship, the fragility-dial redundancy check, and book overlap.
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

pd.set_option("display.width", 220)

TAPE = json.load(open(ROOT / "data/pitch_tape.json"))["tickers"]
UNIV = [t for t in TAPE if not t.startswith("^") and "=" not in t and "-" not in t]
pan = close_panel(UNIV)
rets = pan.pct_change(fill_method=None)
nok = rets.notna().sum(axis=1)
cs = rets.std(axis=1)[nok >= 120]
csr = cs.rolling(252).apply(lambda a: (a[:-1] < a[-1]).mean(), raw=True)

px = close_panel(["SPY"])
d = px.index
spy1 = px["SPY"].pct_change(fill_method=None).reindex(cs.index)
SHORT = [("SPY", -1.0)]
TODAY_CSR = csr.iloc[-1]
print(f"today's PIT csr = {TODAY_CSR*100:.1f}")

m_join = ((csr >= 0.90) & (spy1.abs() <= 0.005)).reindex(d).fillna(False).astype(bool)
epi = declusters(d[m_join], 5, d)

# ---------------------------------------------------------------- 1. by year
print("\n===== 1. concentration: by-year episode totals (short SPY) =====")
for h in (3, 10):
    ret = vehicle_ret(px, SHORT, h)
    e = pd.DatetimeIndex(epi).intersection(ret.dropna().index)
    v = ret.loc[e]
    by = v.groupby(v.index.year).agg(["count", "sum", "mean"])
    by[["sum", "mean"]] = (by[["sum", "mean"]] * 100).round(2)
    print(f"\n h={h}:")
    print(by.to_string())
    for drop in ([2008], [2008, 2020], [2008, 2002, 2020]):
        k = v[~v.index.year.isin(drop)]
        print(f"   ex-{drop}: N={len(k)} mean={k.mean()*100:+.3f}% "
              f"hit={(k>0).mean()*100:.1f}% signp={sign_test(int((k>0).sum()), len(k)):.4f}")

# ---------------------------------------------------------------- 2. AT today's extreme
print("\n\n===== 2. the cell AT today's reading (registry 2026-08-18 probe) =====")
rows = []
bands = [(0.50, 0.70), (0.70, 0.80), (0.80, 0.85), (0.85, 0.90), (0.90, 0.95), (0.95, 1.01)]
for lo, hi in bands:
    m = ((csr >= lo) & (csr < hi) & (spy1.abs() <= 0.005)).reindex(d).fillna(False).astype(bool)
    for h in (3, 10):
        ret = vehicle_ret(px, SHORT, h)
        e = declusters(pd.DatetimeIndex(d[m]).intersection(ret.dropna().index), 5, ret.dropna().index)
        r = summarize(ret.loc[e].values, f"csr [{lo:.2f},{hi:.2f}) h={h}")
        r["live"] = "<== TODAY" if lo <= TODAY_CSR < hi else ""
        rows.append(r)
show(rows, "quiet-index cell by dispersion band")

# ---------------------------------------------------------------- 3. gradient
print("\n===== 3. distance-from-the-extreme gradient WITHIN the trigger set =====")
for h in (3, 10):
    ret = vehicle_ret(px, SHORT, h)
    e = pd.DatetimeIndex(epi).intersection(ret.dropna().index)
    x = csr.reindex(e).values * 100
    y = ret.loc[e].values * 100
    ok = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[ok], y[ok]
    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x)
    se = np.sqrt((resid ** 2).sum() / (len(x) - 2) / ((x - x.mean()) ** 2).sum())
    print(f" h={h}: slope {b:+.4f} pp per pctile pt (t={b/se:+.2f}), N={len(x)}, "
          f"fitted at today's {TODAY_CSR*100:.1f} = {a + b*TODAY_CSR*100:+.3f}% "
          f"(cell mean {y.mean():+.3f}%)")
    print(f"    trigger csr distribution: min {x.min():.1f} p25 {np.percentile(x,25):.1f} "
          f"median {np.percentile(x,50):.1f} max {x.max():.1f}; today's pctile within triggers = "
          f"{(x < TODAY_CSR*100).mean()*100:.1f}")

# ---------------------------------------------------------------- 4. survivorship
print("\n\n===== 4. survivorship: alternative cross-sections =====")
SECT = ["XLK", "XLV", "XLF", "XLY", "XLP", "XLE", "XLI", "XLB", "XLU", "XLRE", "XLC"]
sp = close_panel(SECT)
sr = sp.pct_change(fill_method=None)
cs_s = sr.std(axis=1)[sr.notna().sum(axis=1) >= 8]
csr_s = cs_s.rolling(252).apply(lambda a: (a[:-1] < a[-1]).mean(), raw=True)
print(f" sector-ETF cross-sectional sd today {cs_s.iloc[-1]*100:.3f}%, PIT rank {csr_s.iloc[-1]*100:.1f}")
# fixed membership: names with a bar on the first common date
first_ok = pan.notna().all(axis=1)
fixed = [c for c in pan.columns if pan[c].first_valid_index() is not None
         and pan[c].first_valid_index() <= pd.Timestamp("2001-06-01")]
print(f" names with data by 2001-06: {len(fixed)} of {len(UNIV)}")
pf = pan[fixed]
rf = pf.pct_change(fill_method=None)
cs_f = rf.std(axis=1)
csr_f = cs_f.rolling(252).apply(lambda a: (a[:-1] < a[-1]).mean(), raw=True)
print(f" fixed-membership cs sd today {cs_f.iloc[-1]*100:.3f}%, PIT rank {csr_f.iloc[-1]*100:.1f}")

rows = []
for nm, rk in [("full 210-name tape", csr), ("fixed-membership (2001)", csr_f),
               ("11 sector ETFs", csr_s)]:
    m = ((rk >= 0.90) & (spy1.reindex(rk.index).abs() <= 0.005)).reindex(d).fillna(False).astype(bool)
    for h in (3, 10):
        ret = vehicle_ret(px, SHORT, h)
        e = declusters(pd.DatetimeIndex(d[m]).intersection(ret.dropna().index), 5, ret.dropna().index)
        r = summarize(ret.loc[e].values, f"{nm} h={h}")
        base = ret.dropna()
        r["edge_pp"] = round(r.get("mean_pct", np.nan) - 100 * base.mean(), 3)
        rows.append(r)
show(rows, "panel-definition sensitivity")

# ---------------------------------------------------------------- 5. fragility dial
print("\n\n===== 5. registry collision: is this the dispersion COMPONENT or the composite? =====")
frag = pd.read_parquet(ROOT / "data/rd2_fragility.parquet")
ma = frag["63d"].rolling(10).mean()
print(f" dial ma10(63d) today = {ma.iloc[-1]:.1f}; series starts {frag.index.min().date()}; "
      f"PIT-only from 2026-07-02")
ma_al = ma.reindex(d)
m_dial = (ma_al >= 50).fillna(False).astype(bool)
ov = m_join & ma_al.notna()
print(f" cell days with a dial reading: {int(ov.sum())} of {int(m_join.sum())} "
      f"({100*ov.sum()/max(1,m_join.sum()):.0f}%) -- everything before 2016-07 is unmeasurable here")
if ov.sum():
    print(f" of those, dial ma10(63d) >= 50 on {int((m_join & m_dial).sum())} "
          f"({100*(m_join & m_dial).sum()/ov.sum():.0f}%)")
    pit = pd.Timestamp("2026-07-02")
    print(f" cell days on the PIT-only vintage (>= 2026-07-02): {int((m_join & (d >= pit)).sum())}")
rows = []
for nm, m in [("dial>=50 ALONE", m_dial),
              ("dispersion cell ALONE (2016+)", m_join & ma_al.notna()),
              ("cell AND dial>=50", m_join & m_dial),
              ("dial>=50 WITHOUT the cell", m_dial & ~m_join)]:
    for h in (3, 10):
        ret = vehicle_ret(px, SHORT, h)
        e = declusters(pd.DatetimeIndex(d[m.values]).intersection(ret.dropna().index), 5, ret.dropna().index)
        r = summarize(ret.loc[e].values, f"{nm} h={h}")
        rows.append(r)
show(rows, "dial vs dispersion (2016+ only, mixed vintage)")

# ---------------------------------------------------------------- 6. book overlap
print("\n\n===== 6. book overlap on trigger days =====")
led = pd.read_parquet(ROOT / "data/backtest_trades_full.parquet")
led["Signal Date"] = pd.to_datetime(led["Signal Date"])
trig = set(pd.DatetimeIndex(d[m_join]))
on = led[led["Signal Date"].isin(trig)]
print(f" book trades signalled on the 162 trigger days: {len(on)} of {len(led)} "
      f"({100*len(on)/len(led):.2f}%; trigger days are {100*m_join.sum()/len(d):.2f}% of sessions)")
if len(on):
    print(on.groupby("Direction")["PnL_flat_750k"].agg(["count", "mean", "sum"]).to_string())
    print("\n by strategy:")
    print(on.groupby(["Strategy", "Direction"]).size().to_string())
