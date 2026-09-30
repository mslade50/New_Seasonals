"""RED-TEAM for C3 (long UNG after vol-confirmed thrust). Independent recompute
from raw master_prices (own index arithmetic, not vehicle_ret), split/volume
integrity, 2009 creation-halt window, NG=F cross-check, roll timing, EIA
storage split, loser paths in ATR units, book overlap."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
mp = pd.read_parquet(ROOT / "data" / "master_prices.parquet")
mp["date"] = pd.to_datetime(mp["date"])


def frame(t):
    g = mp[mp.ticker == t].drop(columns="ticker").sort_values("date")
    g = g[~g.date.duplicated(keep="last")].set_index("date")
    return g.astype(float)


def wilder_atr(d, n=14):
    pc = d["Close"].shift(1)
    tr = pd.concat([d["High"] - d["Low"], (d["High"] - pc).abs(),
                    (d["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def nymex_expiry(year, month):
    """NG: 3rd-last business day of the month before delivery (approx: weekdays
    minus US federal holidays)."""
    from pandas.tseries.holiday import USFederalHolidayCalendar
    hol = USFederalHolidayCalendar().holidays(f"{year}-01-01", f"{year}-12-31")
    days = pd.bdate_range(f"{year}-{month:02d}-01",
                          pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0))
    days = days[~days.isin(hol)]
    return days[-3]


u = frame("UNG")
c, v = u["Close"], u["Volume"]
idx = c.index
r1 = c.pct_change()
vr_incl = v / v.rolling(63).mean()
vr_ex = v / v.shift(1).rolling(63).mean()
atr = wilder_atr(u)

print("LIVE rows:")
print(pd.DataFrame({"close": c, "r1%": 100 * r1, "vr_incl": vr_incl,
                    "vr_ex": vr_ex, "atr": atr}).tail(4).round(4).to_string())

# ---------------- 2. split / volume integrity ----------------
print("\n=== 2. split/volume integrity ===")
med = v.rolling(20).median()
step = (med.shift(-20) / med)  # median of next 20 vs prior 20
dv = (v * c).rolling(20).median()
dstep = dv.shift(-20) / dv
s = pd.DataFrame({"vol_step": step, "dollarvol_step": dstep}).dropna()
print("largest 20d-median VOLUME steps (next20/prev20):")
print(s.sort_values("vol_step").iloc[list(range(4)) + list(range(-4, 0))].round(2))
for sd in ["2018-02-22", "2024-01-24", "2024-01-25"]:
    sd = pd.Timestamp(sd)
    p = idx.searchsorted(sd)
    pre = v.iloc[p - 20:p].median()
    post = v.iloc[p:p + 20].median()
    pc_ = c.iloc[p - 1]
    print(f"  around {sd.date()}: vol med pre {pre:,.0f} post {post:,.0f} "
          f"ratio {post/pre:.2f}; close d-1 {pc_:.3f} d {c.iloc[p]:.3f} "
          f"(1d {100*(c.iloc[p]/pc_-1):+.2f}%)")
big = r1.abs() > 0.25
print("  any |1d| > 25% in adjusted close (split artifact?):",
      [(d.date(), round(100 * x, 1)) for d, x in r1[big].items()])

# ---------------- 1. independent recompute ----------------
trig = (r1 >= 0.05) & (vr_incl >= 3)
cand = list(idx[trig.fillna(False).values])
pos = {d: i for i, d in enumerate(idx)}
H = 2
eps, last = [], -10**9
for d in cand:
    p = pos[d]
    if p + 1 + H >= len(idx):
        continue  # no forward data (today's live signal)
    if p - last >= H:
        eps.append(d)
        last = p
rows = []
for d in eps:
    p = pos[d]
    ent, ex = c.iloc[p + 1], c.iloc[p + 1 + H]
    rows.append({
        "signal": d, "wd": d.day_name()[:3], "sig_r%": 100 * r1.iloc[p],
        "vr": vr_incl.iloc[p], "vr_ex": vr_ex.iloc[p], "sig_close": c.iloc[p],
        "entry_date": idx[p + 1], "entry": ent, "exit_date": idx[p + 1 + H],
        "exit": ex, "t1_r%": 100 * (ent / c.iloc[p] - 1),
        "h2%": 100 * (ex / ent - 1), "atr_sig": atr.iloc[p],
        "h2_atr": (ex - ent) / atr.iloc[p],
        "t2_vs_sig%": 100 * (c.iloc[p + 2] / c.iloc[p] - 1),
        "t2_vs_entry%": 100 * (c.iloc[p + 2] / ent - 1),
        "mae_atr": (u["Low"].iloc[p + 2:p + 2 + H].min() - ent) / atr.iloc[p],
        "thu_in_hold": any(idx[p + k].weekday() == 3 for k in (2, 3)),
    })
E = pd.DataFrame(rows).set_index("signal")
pd.set_option("display.width", 250)
print("\n=== 1. episode list (entry close t+1, exit close t+3) ===")
print(E.drop(columns=["exit_date"]).round(3).to_string())
x = E["h2%"].values / 100


def line(vals, lab):
    vals = np.asarray(vals, float)
    w = int((vals > 0).sum())
    n = len(vals)
    if n == 0:
        return f"{lab}: N=0"
    return (f"{lab}: N={n} mean {100*vals.mean():+.2f}% median "
            f"{100*np.median(vals):+.2f}% rec {w}-{n-w} sign p "
            f"{sign_test(w, n):.4f} boot P<=0 {bootstrap_p_le0(vals):.3f}")


print("\n" + line(x, "ALL episodes h=2"))
print(f"entry-day (t+1) mean {E['t1_r%'].mean():+.2f}% (NOT in h2)")
chk = vehicle_ret(pd.DataFrame({"UNG": c}), [("UNG", 1.0)], 2).reindex(E.index)
print("max |recompute - vehicle_ret| (pp):",
      float((100 * chk - E["h2%"]).abs().max()))

# ---------------- 2b. compromised episodes ----------------
split_dates = [pd.Timestamp("2018-02-22"), pd.Timestamp("2024-01-24")]
comp = pd.Series(False, index=E.index)
reason = {}
for d in E.index:
    p = pos[d]
    for sd in split_dates:
        q = idx.searchsorted(sd)
        if 0 <= p - q < 63:
            comp[d] = True
            reason[d] = f"within 63td after split {sd.date()}"
    if pd.Timestamp("2009-06-15") <= d <= pd.Timestamp("2009-12-31"):
        comp[d] = True
        reason[d] = "2009 creation halt / NAV premium window"
    if p < 63 + pos[idx[0]] + 5:
        comp[d] = True
        reason[d] = "63d mean incomplete"
print("\ncompromised:", {k.date(): r for k, r in reason.items()})
print(line(x[~comp.values], "EX-compromised h=2"))
print(line(x[E.index.year != 2009], "EX-all-2009 h=2"))
print(line(x[E.index >= "2010-01-01"], "2010+ h=2"))

# vol definition fragility
trig_ex = (r1 >= 0.05) & (vr_ex >= 3)
extra = trig_ex & ~trig
print("\nextra days under ex-today 63d mean (vr_ex>=3 but vr_incl<3):")
for d in idx[extra.fillna(False).values]:
    p = pos[d]
    if p + 3 < len(idx):
        print(f"   {d.date()} vr_ex {vr_ex.iloc[p]:.2f} vr_incl "
              f"{vr_incl.iloc[p]:.2f} h2 {100*(c.iloc[p+3]/c.iloc[p+1]-1):+.2f}%")

# ---------------- 3. NG=F cross-check ----------------
print("\n=== 3. NG=F front on the same signal dates ===")
ng = frame("NG=F")["Close"]
ngr = []
for d in E.index:
    if d not in ng.index:
        ngr.append((d, np.nan, np.nan, "missing"))
        continue
    q = ng.index.get_loc(d)
    ent_d, ex_d = ng.index[q + 1], ng.index[q + 3]
    # roll seam: an NG expiry falls on [ent_d, ex_d)
    seam = False
    for y, m in {(ent_d.year, ent_d.month), (ex_d.year, ex_d.month)}:
        e_ = nymex_expiry(y, m)
        if ent_d <= e_ < ex_d:
            seam = True
    ngr.append((d, 100 * (ng.iloc[q] / ng.iloc[q - 1] - 1),
                100 * (ng.iloc[q + 3] / ng.iloc[q + 1] - 1), "SEAM" if seam else ""))
N = pd.DataFrame(ngr, columns=["signal", "ng_sig%", "ng_h2%", "seam"]).set_index("signal")
N["ung_h2%"] = E["h2%"]
print(N.round(2).to_string())
ok = N[(N.seam == "") & N["ng_h2%"].notna()]
print(line(ok["ng_h2%"].values / 100, "NG=F h=2 (no seam)"))
print(line(ok["ung_h2%"].values / 100, "UNG h=2 same dates"))
print("corr UNG vs NG h2:", round(ok[["ng_h2%", "ung_h2%"]].corr().iloc[0, 1], 3))

# ---------------- 4. roll timing ----------------
print("\n=== 4. roll timing (bd from signal to front NG expiry) ===")
bd2x = []
for d in E.index:
    e_ = nymex_expiry(d.year, d.month)
    if d > e_:
        nm = d + pd.offsets.MonthBegin(1)
        e_ = nymex_expiry(nm.year, nm.month)
    bd2x.append(len(pd.bdate_range(d, e_)) - 1)
E["bd_to_exp"] = bd2x
# UNG roll: 4 days starting ~2 weeks (~10 bd) before expiry -> bd_to_exp ~7..10
E["in_roll"] = E["bd_to_exp"].between(6, 11)
print(E[["bd_to_exp", "in_roll", "h2%"]].T.to_string())
print(line(x[E["in_roll"].values], "signal in/near UNG roll (6-11bd)"))
print(line(x[~E["in_roll"].values], "outside roll"))
e_now = nymex_expiry(2026, 9)
print(f"Oct-26 NG expiry {e_now.date()}; bd from 09-22 = "
      f"{len(pd.bdate_range('2026-09-22', e_now)) - 1}; UNG roll (2wk before, "
      f"4 days) ~ {(e_now - pd.offsets.BDay(10)).date()}..{(e_now - pd.offsets.BDay(7)).date()}")

# ---------------- 5. storage split ----------------
print("\n=== 5. EIA storage (Thursday) inside t+2..t+3 ===")
print(E.groupby("wd")["h2%"].agg(["count", "mean"]).round(2))
th = E["thu_in_hold"].values
print(line(x[th], "EIA-Thu INSIDE hold"))
print(line(x[~th], "EIA-Thu OUTSIDE hold"))
print("inside-hold episodes:", [(d.date(), round(r, 2)) for d, r in
                                E.loc[E.thu_in_hold, "h2%"].items()])
# control: all UNG days, h=2 lag1, split by Thu-inside
allr = (c.shift(-3) / c.shift(-1) - 1)
thu_all = pd.Series([any((p + k) < len(idx) and idx[p + k].weekday() == 3
                         for k in (2, 3)) for p in range(len(idx))], index=idx)
print(f"unconditional UNG h=2 lag1: Thu-inside {100*allr[thu_all].mean():+.3f}% "
      f"/ outside {100*allr[~thu_all].mean():+.3f}%")

# ---------------- 7. sizing ----------------
print("\n=== 7. h=2 outcomes in ATR units (Wilder-14 at signal close) ===")
print(E["h2_atr"].describe().round(2).to_string())
print("worst 5 h2_atr:", E["h2_atr"].nsmallest(5).round(2).to_dict())
print("worst 5 MAE_atr (intra-hold lows):", E["mae_atr"].nsmallest(5).round(2).to_dict())
for k in (1.5, 2.0, 2.5):
    print(f"  k={k}: worst h2 = {E['h2_atr'].min()/k:+.2f}R, "
          f"mean {E['h2_atr'].mean()/k:+.2f}R, worst MAE {E['mae_atr'].min()/k:+.2f}R")

# ---------------- what-kills-it level at t+2 ----------------
print("\n=== t+2 close vs signal close (the 09-24 checkpoint) vs final ===")
print(E[["t2_vs_sig%", "t2_vs_entry%", "h2%"]].sort_values("t2_vs_sig%").round(2).to_string())

# ---------------- 8. book overlap ----------------
bt = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
tcol = [cc for cc in bt.columns if cc.lower() in ("ticker", "symbol")][0]
hit = bt[bt[tcol].astype(str).str.upper().isin(
    ["UNG", "NG=F", "BOIL", "KOLD", "UNL", "FCG", "GAZ"])]
print(f"\n=== 8. book ledger natgas rows: {len(hit)} ===")
if len(hit):
    print(hit[[tcol]].value_counts())
