"""GAP 6: cost of the WP1 margin-feasibility guard, replayed on the ledger.

Stylised PM rates from the brief's WP1 table: single stock / sector ETF 15%,
broad index ETF 8%, small-cap index ETF 10%, 3x ETF long 45% / short 90%,
short stock under $16.67 max(15%, $5/share), under $5 100%.
Req_proj(d) = m x [ Req_open(d) + 1.10 x Req_new(d) ] where Req_open = positions
open at the morning of d (entered before d, not yet exited), Req_new = that day's
NEW entries (ledger fills; unfilled staged limits are invisible, so Req_new is a
lower bound), m = GRM / 1.5 (1.25 for GRM 1.875). Marks = entry notional (flat
$750k shares x entry price). Trim factor f = (0.70 NLV - m Req_open) / (1.10 m
Req_new), clipped [0, 1]; f = 0 when m Req_open > 0.85 NLV (WP1 alarm rule).
Foregone PnL = sum over the day's entries of m x PnL x (1 - f). Path dependence
(a trimmed leg lowers later Req_open) is ignored -> trim counts are slightly
conservative. Writes gap6_guard_cost.json.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SIZ = HERE.parent
ROOT = SIZ.parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(SIZ))
import strategy_config as sc  # noqa: E402
from flow_conditional_lib import load_ledger, build_trade_mtm  # noqa: E402

NAV = 750_000.0; LIVE_NLV = 632_000.0; GRM_NOW = 1.5
M = 1.875 / GRM_NOW
LEV3X_SHORT = float(os.environ.get("GAP6_LEV3X_SHORT", "0.90"))
TAG = "" if LEV3X_SHORT == 0.90 else f"_lev3xshort{int(LEV3X_SHORT*100)}"
OUT: dict = {"m": M, "lev3x_short_rate": LEV3X_SHORT, "bases": {"base_750k": NAV, "live_nlv_632k": LIVE_NLV}}

led = load_ledger()
alias = {"^GSPC": "SPY", "^NDX": "QQQ"}
led["tk"] = led["Ticker"].map(lambda t: alias.get(t, t))
LEV3X = set(sc.LEV3X_ALL)
BROAD = {"SPY", "QQQ", "DIA", "VOO", "IVV", "MDY", "IJH", "RSP", "EFA", "EEM", "VEA", "VWO"}
SMALL = {"IWM", "IJR", "IWN", "IWO", "VB", "VBR", "VTWO"}
SECTOR = {"XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC", "GLD", "SLV", "TLT", "IEF", "LQD", "HYG", "USO", "UNG", "GDX", "GDXJ", "XOP", "XBI", "SMH", "KRE", "XME", "XRT", "XHB", "ITB", "IYR", "VNQ", "FXI", "EWZ", "EWJ", "EWY", "EWT", "EWG", "EWU", "EWC", "EWW", "EWA", "EWH", "INDA", "KWEB", "ARKK", "IBB", "OIH", "TAN", "URA", "LIT", "DBC", "DBA", "UUP", "FXE", "FXY", "SLX", "COPX", "PPLT", "PALL", "CORN", "WEAT", "SOYB", "WOOD", "REMX", "SIL", "JETS", "PHO", "MOO", "BOTZ", "SOXX", "HACK", "IGV", "VGT", "XLG", "SPHB", "SPLV", "VXX", "UVXY", "SVXY", "TLH", "TBT", "TMF", "BND", "AGG"}


def rate(tk: str, direction: str, price: float) -> float:
    if tk in LEV3X:
        return 0.45 if direction == "Long" else LEV3X_SHORT
    if tk in BROAD:
        return 0.08
    if tk in SMALL:
        return 0.10
    if tk in SECTOR:
        return 0.15
    if direction == "Short":
        if price < 5:
            return 1.0
        if price < 16.67:
            return max(0.15, 5.0 / price)
    return 0.15


led["notional"] = led["EntryPrice"] * led["Shares"]
led["rate"] = [rate(t, d, p) for t, d, p in zip(led["tk"], led["Direction"], led["EntryPrice"])]
led["req"] = led["notional"] * led["rate"]
idx = pd.bdate_range("2003-01-01", "2026-09-01")
n = len(idx)
req_open = np.zeros(n); req_new = np.zeros(n); gross = np.zeros(n)
d0 = idx.searchsorted(led["Entry Date"].values); d1 = idx.searchsorted(led["ExitDate"].values)
for a, b, q, g in zip(d0, d1, led["req"].values, led["notional"].values):
    req_new[a] += q
    req_open[a + 1:b + 1] += q      # open at the morning of d for entry < d <= exit
    gross[a:b + 1] += g
req_open_s = pd.Series(req_open, index=idx); req_new_s = pd.Series(req_new, index=idx); gross_s = pd.Series(gross, index=idx)
years_full = (idx[-1] - pd.Timestamp("2003-01-21")).days / 365.25
years_16 = (idx[-1] - pd.Timestamp("2016-01-01")).days / 365.25
blended = float(led["req"].sum() / led["notional"].sum())
print(f"trades {len(led)}; blended rate on trade notional {blended:.1%}; class shares:",
      led.assign(k=np.select([led.tk.isin(LEV3X), led.tk.isin(BROAD), led.tk.isin(SMALL), led.tk.isin(SECTOR)], ["lev3x", "broad", "small", "sector"], "single")).groupby("k")["notional"].sum().div(led.notional.sum()).round(3).to_dict())
OUT["blended_rate"] = blended

# daily PnL from per-trade MTM (flat basis, m=1)
days, MTM = build_trade_mtm(led)
daily = pd.Series(MTM.sum(axis=0), index=days).reindex(idx).fillna(0.0)
print("MTM reconciliation residual:", float(abs(MTM.sum(axis=1) - led.PnL.values).max()))
entry_pos = {i: [] for i in range(n)}
for i, a in enumerate(d0):
    entry_pos[a].append(i)
pnl_arr = led["PnL"].values


def replay(m: float, nlv: float, lo: float, hi: float, start: str) -> dict:
    ro, rn = m * req_open, m * req_new
    proj = ro + 1.10 * rn
    w = idx >= start
    yrs = (idx[-1] - pd.Timestamp(start)).days / 365.25
    relief_off = (proj > lo * nlv) & w
    trim = (proj > hi * nlv) & w
    alarm = (ro > 0.85 * nlv) & w
    f = np.ones(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        ft = (hi * nlv - ro) / (1.10 * rn)
    f[trim] = np.clip(np.nan_to_num(ft[trim], nan=0.0), 0.0, 1.0)
    f[alarm] = 0.0
    trim_eff = (f < 1.0) & w
    foregone = 0.0; rows = []
    for i in np.where(trim_eff)[0]:
        ids = entry_pos[i]
        day_pnl = m * pnl_arr[ids].sum()
        fg = m * pnl_arr[ids].sum() * (1 - f[i])
        foregone += fg
        rows.append(dict(date=str(idx[i].date()), req_proj_pct_nlv=float(proj[i] / nlv), req_open_pct_nlv=float(ro[i] / nlv), f=float(f[i]),
                         n_new=len(ids), new_entry_pnl_at_m=float(day_pnl), foregone=float(fg),
                         book_pnl_day_at_m=float(m * daily.values[i]), gross_pct_base=float(m * gross[i] / NAV)))
    book_pnl = m * daily.values[w].sum()
    return dict(nlv=nlv, lo=lo, hi=hi, start=start, years=yrs, relief_off_days=int(relief_off.sum()), relief_off_per_yr=float(relief_off.sum() / yrs),
                trim_days=int(trim_eff.sum()), trims_per_yr=float(trim_eff.sum() / yrs), alarm_days=int(alarm.sum()),
                foregone_total=float(foregone), foregone_per_yr=float(foregone / yrs), book_pnl_per_yr=float(book_pnl / yrs),
                foregone_pct_book_pnl=float(foregone / book_pnl) if book_pnl else np.nan,
                p99_proj_pct_nlv=float(np.quantile(proj[w] / nlv, 0.99)), max_proj_pct_nlv=float(proj[w].max() / nlv), max_date=str(idx[w][proj[w].argmax()].date()),
                trim_rows=rows)


print("\n=== main: GRM 1.875 (m=1.25), lines 60/70, both bases, 2003+ and 2016+ ===")
OUT["main"] = {}
for base_lab, nlv in [("base_750k", NAV), ("live_nlv_632k", LIVE_NLV)]:
    for start in ["2003-01-21", "2016-01-01"]:
        r = replay(M, nlv, 0.60, 0.70, start)
        OUT["main"][f"{base_lab}|{start[:4]}+"] = r
        print(f"{base_lab:14s} {start[:4]}+: reliefs-off {r['relief_off_per_yr']:.1f} d/yr | trims {r['trims_per_yr']:.2f}/yr ({r['trim_days']} days, {r['alarm_days']} alarm) | "
              f"foregone ${r['foregone_per_yr']:,.0f}/yr = {r['foregone_pct_book_pnl']:.2%} of book PnL (${r['book_pnl_per_yr']:,.0f}/yr) | p99 proj {r['p99_proj_pct_nlv']:.0%} max {r['max_proj_pct_nlv']:.0%} ({r['max_date']})")
        if start.startswith("2003"):
            for row in r["trim_rows"]:
                print(f"     {row['date']} proj {row['req_proj_pct_nlv']:.0%} open {row['req_open_pct_nlv']:.0%} f={row['f']:.2f} n_new={row['n_new']:2d} new-entry PnL {row['new_entry_pnl_at_m']:>9,.0f} foregone {row['foregone']:>8,.0f} book day {row['book_pnl_day_at_m']:>9,.0f} gross {row['gross_pct_base']:.0%}")

print("\n=== sensitivity to the lines (m=1.25) ===")
OUT["sensitivity"] = []
for base_lab, nlv in [("base_750k", NAV), ("live_nlv_632k", LIVE_NLV)]:
    for lo, hi in [(0.55, 0.65), (0.60, 0.70), (0.65, 0.75), (0.70, 0.80)]:
        for start in ["2003-01-21", "2016-01-01"]:
            r = replay(M, nlv, lo, hi, start)
            OUT["sensitivity"].append({k: v for k, v in r.items() if k != "trim_rows"} | dict(base=base_lab))
            print(f"{base_lab:14s} {int(lo*100)}/{int(hi*100)} {start[:4]}+: reliefs-off {r['relief_off_per_yr']:5.1f} d/yr | trims {r['trims_per_yr']:.2f}/yr | foregone ${r['foregone_per_yr']:>7,.0f}/yr = {r['foregone_pct_book_pnl']:.2%} of book PnL")

print("\n=== GRM sensitivity at 60/70 (2003+) ===")
OUT["grm_sens"] = []
for grm in [1.5, 1.875, 2.25, 2.5, 3.0]:
    for base_lab, nlv in [("base_750k", NAV), ("live_nlv_632k", LIVE_NLV)]:
        r = replay(grm / GRM_NOW, nlv, 0.60, 0.70, "2003-01-21")
        OUT["grm_sens"].append({k: v for k, v in r.items() if k != "trim_rows"} | dict(base=base_lab, grm=grm))
        print(f"GRM {grm:<5} {base_lab:14s}: reliefs-off {r['relief_off_per_yr']:5.1f} d/yr | trims {r['trims_per_yr']:.2f}/yr | foregone ${r['foregone_per_yr']:>7,.0f}/yr = {r['foregone_pct_book_pnl']:.2%} | max proj {r['max_proj_pct_nlv']:.0%}")

# ------------------------------------------------------------ gross > 100% NAV share (verify the 15% / 49% claim)
print("\n=== share of days / PnL with gross > x% of the $750k base at m=1 ===")
OUT["gross_tail"] = {}
sd = json.load(open(ROOT / "dist/data/strategy_daily.json"))
tot_sd = pd.Series(sd["total_flat"], index=pd.to_datetime(sd["dates"]), dtype=float)
for lab, series in [("ledger_mtm_2003+", daily[daily.index >= "2003-01-21"]), ("ledger_mtm_2016+", daily[daily.index >= "2016-01-01"]),
                    ("strategy_daily_2016+_to_0807", tot_sd[tot_sd.index >= "2016-01-01"])]:
    g = gross_s.reindex(series.index).fillna(0.0)
    OUT["gross_tail"][lab] = {}
    for x in [0.8, 1.0, 1.5, 2.0]:
        mask = g > x * NAV
        OUT["gross_tail"][lab][f"{x:g}"] = dict(days_share=float(mask.mean()), pnl_share=float(series[mask].sum() / series.sum()))
        print(f"  {lab:30s} gross > {x:.0%}: {mask.mean():5.1%} of days, {series[mask].sum()/series.sum():5.1%} of PnL")
# same using the requirement rather than gross: days where m*Req_proj > 70% NLV and their PnL share
for base_lab, nlv in [("base_750k", NAV), ("live_nlv_632k", LIVE_NLV)]:
    proj = M * (req_open + 1.10 * req_new)
    for start in ["2003-01-21", "2016-01-01"]:
        w = idx >= start
        dd = daily.values
        for lab, thr in [("relief_off>60", 0.60), ("trim>70", 0.70)]:
            mask = (proj > thr * nlv) & w
            OUT["gross_tail"][f"{base_lab}|{start[:4]}+|{lab}"] = dict(days_share=float(mask.sum() / w.sum()), pnl_share=float(dd[mask].sum() / dd[w].sum()))
            print(f"  {base_lab} {start[:4]}+ {lab}: {mask.sum()/w.sum():5.1%} of days carry {dd[mask].sum()/dd[w].sum():5.1%} of book PnL (m=1.25)")

# per-strategy share of the foregone PnL (main case, both bases, 2003+)
OUT["foregone_by_strategy"] = {}
for base_lab, nlv in [("base_750k", NAV), ("live_nlv_632k", LIVE_NLV)]:
    r = OUT["main"][f"{base_lab}|2003+"]
    acc = {}
    for row in r["trim_rows"]:
        i = idx.get_loc(pd.Timestamp(row["date"]))
        for j in entry_pos[i]:
            acc[led.Strategy.iloc[j]] = acc.get(led.Strategy.iloc[j], 0.0) + M * pnl_arr[j] * (1 - row["f"])
    OUT["foregone_by_strategy"][base_lab] = acc
    print(f"foregone by strategy ({base_lab}): {dict(sorted(((k, round(v)) for k, v in acc.items()), key=lambda x: -abs(x[1])))}")
json.dump(OUT, open(HERE / f"gap6_guard_cost{TAG}.json", "w"), indent=1, default=float)
print(f"wrote gap6_guard_cost{TAG}.json (3x short rate {LEV3X_SHORT})")
