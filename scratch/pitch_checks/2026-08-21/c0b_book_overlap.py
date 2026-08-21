"""Book overlap, all three candidates, against data/backtest_trades_full.parquet."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change

pd.set_option("display.width", 250)
L = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
L["Signal Date"] = pd.to_datetime(L["Signal Date"])

# ---------------------------------------------------------------- C2
print("===== C2: what does the book trade on the credit-unconfirmed washout? =====")
px = close_panel(["SPY", "HYG"])
px = px.loc[px["HYG"].notna()]
spy5r = pct_rank(px["SPY"], 5)
hyg_off = px["HYG"] / rolling_on_valid(px["HYG"], lambda x: x.rolling(252).max()) - 1.0
hyg5 = _valid_pct_change(px["HYG"], 5)
trig_hi = px.index[((spy5r <= 10) & (hyg_off >= -0.005)).fillna(False).values]
trig_rt = px.index[((spy5r <= 10) & (hyg5 >= -0.005)).fillna(False).values]
for nm, tr in (("52w-high form", trig_hi), ("HYG-return form", trig_rt)):
    sub = L[L["Signal Date"].isin(tr)]
    print(f"\n {nm}: {len(tr)} trigger days -> {len(sub)} ledger trades")
    if len(sub):
        g = sub.groupby(["Strategy", "Direction"]).agg(
            n=("R_Multiple", "size"), avgR=("R_Multiple", "mean"),
            totR=("R_Multiple", "sum")).round(3).sort_values("n", ascending=False)
        print(g.head(12).to_string())
        print(f"  ALL: n={len(sub)} avgR {sub.R_Multiple.mean():.3f} "
              f"long {int((sub.Direction=='Long').sum())} / short "
              f"{int((sub.Direction=='Short').sum())}")
        idx = sub[sub.Ticker.isin(["SPY", "QQQ", "^GSPC", "^NDX", "IWM"])]
        print(f"  index-vehicle trades: n={len(idx)} avgR "
              f"{idx.R_Multiple.mean() if len(idx) else float('nan'):.3f} "
              f"({int((idx.Direction=='Long').sum())} long)")

# ---------------------------------------------------------------- C7
print("\n\n===== C7: what does the book do on bank-breadth BROKEN trigger days? =====")
BANKS = ["JPM", "BAC", "C", "WFC", "GS", "MS", "USB", "PNC", "TFC", "SCHW", "STT"]
raw = load_prices(BANKS + ["SPY", "XLF", "KRE"])
d = raw["SPY"]["Close"].dropna().index
R5 = pd.DataFrame({t: pct_rank(raw[t]["Close"].dropna(), 5).reindex(d) for t in BANKS if t in raw})
R63 = pd.DataFrame({t: pct_rank(raw[t]["Close"].dropna(), 63).reindex(d) for t in BANKS if t in raw})
nv = R5.notna().sum(axis=1)
brd = (R5 <= 20).sum(axis=1) / nv.replace(0, np.nan)
m = ((brd >= 0.70) & (R63.median(axis=1) < 70) & (nv >= 8)).fillna(False)
tr = d[m.values]
print(f" {len(tr)} trigger days")
sub = L[L["Signal Date"].isin(tr)]
print(f" ledger trades on those days: {len(sub)}, avgR {sub.R_Multiple.mean():.3f}")
g = sub.groupby(["Strategy", "Direction"]).agg(n=("R_Multiple", "size"),
                                               avgR=("R_Multiple", "mean")).round(3)
print(g.sort_values("n", ascending=False).head(15).to_string())
fin = sub[sub.Ticker.isin(BANKS + ["XLF", "KRE"])]
print(f"\n trades IN the bank complex on trigger days: n={len(fin)} "
      f"avgR {fin.R_Multiple.mean() if len(fin) else float('nan'):.3f}")
if len(fin):
    print(fin.groupby(["Strategy", "Direction"]).agg(
        n=("R_Multiple", "size"), avgR=("R_Multiple", "mean")).round(3).to_string())
allfin = L[L.Ticker.isin(BANKS + ["XLF", "KRE"])]
print(f" book's lifetime bank-complex trades: n={len(allfin)} avgR "
      f"{allfin.R_Multiple.mean():.3f}, "
      f"{int((allfin.Direction=='Long').sum())} long / {int((allfin.Direction=='Short').sum())} short")
print(allfin.groupby(["Strategy", "Direction"]).agg(
    n=("R_Multiple", "size"), avgR=("R_Multiple", "mean")).round(3)
    .sort_values("n", ascending=False).head(10).to_string())

# ---------------------------------------------------------------- C10
print("\n\n===== C10: does the book trade names inside their own print? =====")
e = pd.read_parquet(ROOT / "data" / "earnings_calendar.parquet")
e = e[e.eps_est.notna()]
emap = {t: pd.DatetimeIndex(sorted(g.date.unique())) for t, g in e.groupby("ticker")}
cal = pd.DatetimeIndex(sorted(raw["SPY"]["Close"].dropna().index))
pos = pd.Series(range(len(cal)), index=cal)
rows = []
for _, r in L.iterrows():
    t = r["Ticker"]
    if t not in emap:
        continue
    sd = r["Signal Date"]
    if sd not in pos.index:
        continue
    p = pos[sd]
    ed = emap[t]
    near = ed[(ed >= cal[max(0, p - 12)]) & (ed <= cal[min(len(cal) - 1, p + 12)])]
    if len(near) == 0:
        off = 99
    else:
        offs = [int(pos.get(cal[np.searchsorted(cal.values, np.datetime64(x), "left")
                                .clip(0, len(cal) - 1)]) - p) for x in near]
        off = min(offs, key=abs)
    rows.append({"Strategy": r["Strategy"], "Direction": r["Direction"],
                 "off": off, "R": r["R_Multiple"]})
o = pd.DataFrame(rows)
print(f" ledger trades on tickers with earnings coverage: {len(o)}")
close = o[o.off.abs() <= 1]
print(f" signals within +/-1 session of a print: {len(close)} "
      f"({100*len(close)/len(o):.2f}%)")
if len(close):
    print(close.groupby(["Strategy", "Direction"]).agg(
        n=("R", "size"), avgR=("R", "mean")).round(3).sort_values("n", ascending=False).to_string())
print("\n OVS blackout confirmation, |offset| distribution for Overbot Vol Spike:")
ovs = o[o.Strategy == "Overbot Vol Spike"]
print(f"  n={len(ovs)}  |off|<=10: {int((ovs.off.abs()<=10).sum())}  "
      f"|off|<=1: {int((ovs.off.abs()<=1).sum())}  min |off| = "
      f"{int(ovs.off.abs().min()) if len(ovs) else 'n/a'}")
print("\n by strategy, share of signals within +/-1 of a print:")
print(o.assign(c=o.off.abs() <= 1).groupby("Strategy")["c"].agg(["size", "sum", "mean"])
      .assign(mean=lambda x: (100 * x["mean"]).round(2)).sort_values("sum", ascending=False)
      .head(15).to_string())
