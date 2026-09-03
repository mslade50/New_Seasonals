"""Re-read of the post-freeze OOS window on the current ledger vintage.

Same per-strategy RULE_FREEZE dates as estimation_haircut_05_oos.py. Adds:
  - open-position handling (rows booked 'Time' at the ledger's last bar are
    marks, not exits) -> pooled stats with and without them
  - leave-one-strategy-out and leave-one-ticker-out on the pooled ratio
  - a within-2026 control: Jan-2026 -> freeze vs freeze -> Aug-28 (same year,
    same regime, only the freeze differs), so '2026 is a bad year' and 'the
    rules were fitted' can be told apart
  - the midterm-year control (Jan-Aug of 2006/10/14/18/22 vs the rest)
  - trade-level vs day-block vs ticker-cluster CIs
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = Path(r"C:/Users/McKinley Slade/dev/New_Seasonals")
OUT = ROOT / "scratch/ultracode_sizing_2026-09-02"
sys.path.insert(0, str(OUT))
RNG = np.random.default_rng(7)
pd.set_option("display.width", 260)

led = pd.read_parquet(ROOT / "data/backtest_trades_full.parquet")
led = led[led["PnL_flat_750k"].notna()].copy()
led["Signal Date"] = pd.to_datetime(led["Signal Date"])
led["Exit Date"] = pd.to_datetime(led["Exit Date"])
LAST_BAR = led["Exit Date"].max()
led["open_mark"] = led["Exit Date"].eq(LAST_BAR) & (led["Exit Type"] == "Time")
import pyarrow.parquet as pq
md = {k.decode(): v.decode() for k, v in (pq.read_schema(ROOT / "data/backtest_trades_full.parquet").metadata or {}).items()}

RULE_FREEZE = {
    "Overbot Vol Spike": "2026-06-05", "Oversold Low Volume": "2026-07-20", "52wh Breakout": "2026-07-07",
    "Weak Close Decent Sznls": "2026-05-19", "LT Trend ST OS": "2026-05-01", "St OS Sznl": "2026-04-23",
    "3x ETF Overbot Fade": "2026-07-28", "3x Bear ETF Overbot Fade": "2026-07-28", "3x Leader Gap Fade": "2026-07-10",
    "Indices Oversold Bounce": "2026-05-12", "SPY QQQ MonFri Reversion": "2026-06-09", "Sector BO": "2026-05-01",
    "Monday Dip": "2026-05-12", "ATR Extended Gap Up": "2026-05-11", "Monthly Weak Close": "2026-07-31",
}
led["freeze"] = led["Strategy"].map(RULE_FREEZE).map(pd.Timestamp)
led["post"] = led["Signal Date"] >= led["freeze"]
led["y2026"] = led["Signal Date"] >= "2026-01-01"


def st(df):
    r = df["R_Multiple"].to_numpy(float)
    n = len(r)
    if n == 0:
        return {"N": 0, "avgR": None}
    sd = r.std(ddof=1) if n > 1 else np.nan
    return {"N": n, "avgR": float(r.mean()), "sdR": float(sd), "t": float(r.mean() / sd * np.sqrt(n)) if n > 1 and sd > 0 else None,
            "sumR": float(r.sum()), "win": float((r > 0).mean()), "pnl": float(df["PnL_flat_750k"].sum())}


def mix_ratio(oos, ins):
    """OOS avgR over the IS avgR expected at the OOS strategy mix."""
    mix = oos.groupby("Strategy").size()
    is_means = ins.groupby("Strategy")["R_Multiple"].mean()
    exp = float((mix * is_means.reindex(mix.index)).sum() / mix.sum())
    return float(oos["R_Multiple"].mean() / exp), exp


def boots(oos, exp, key):
    grp = oos.groupby(key)["R_Multiple"].apply(lambda s: s.to_numpy())
    arr = list(grp.values)
    out = []
    for _ in range(4000):
        pick = RNG.choice(len(arr), len(arr))
        out.append(np.concatenate([arr[i] for i in pick]).mean() / exp)
    out = np.array(out)
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))], float((out < 0.5).mean())


res = {"ledger_vintage": md, "last_signal": str(led["Signal Date"].max().date()), "last_bar": str(LAST_BAR.date()), "rule_freeze": RULE_FREEZE}
oos_all = led[led["post"]]
ins_all = led[~led["post"]]
oos_cl = oos_all[~oos_all["open_mark"]]
print("post-freeze rows:", len(oos_all), "of which still-open marks:", int(oos_all["open_mark"].sum()))
for name, oos in (("all", oos_all), ("closed_only", oos_cl)):
    ratio, exp = mix_ratio(oos, ins_all)
    ci_t, p_t = boots(oos, exp, oos.index)
    ci_d, p_d = boots(oos, exp, "Signal Date")
    ci_k, p_k = boots(oos, exp, "Ticker")
    res[f"pooled_{name}"] = {"OOS": st(oos), "IS_avgR_at_mix": exp, "ratio": ratio, "ci_trade": ci_t, "ci_dayblock": ci_d, "ci_ticker": ci_k, "P_ratio_below_half_dayblock": p_d}
    print(f"\n[{name}] OOS N={len(oos)} avgR={oos['R_Multiple'].mean():.3f} IS@mix={exp:.3f} ratio={ratio:.2f} CI trade {ci_t} day {ci_d} ticker {ci_k}")

# per strategy pre/post
rows = []
for s, g in led.groupby("Strategy"):
    a, b = g[~g["post"]], g[g["post"]]
    bc = b[~b["open_mark"]]
    rows.append({"Strategy": s, "freeze": RULE_FREEZE[s], "N_IS": len(a), "avgR_IS": round(a["R_Multiple"].mean(), 3),
                 "N_OOS": len(b), "avgR_OOS": round(b["R_Multiple"].mean(), 3) if len(b) else None, "sumR_OOS": round(b["R_Multiple"].sum(), 2),
                 "N_OOS_closed": len(bc), "avgR_OOS_closed": round(bc["R_Multiple"].mean(), 3) if len(bc) else None,
                 "N_2026_pre_freeze": int(((g["Signal Date"] >= "2026-01-01") & ~g["post"]).sum()),
                 "avgR_2026_pre_freeze": round(g[(g["Signal Date"] >= "2026-01-01") & ~g["post"]]["R_Multiple"].mean(), 3) if ((g["Signal Date"] >= "2026-01-01") & ~g["post"]).any() else None})
tab = pd.DataFrame(rows).sort_values("N_OOS", ascending=False)
print("\n", tab.to_string(index=False))
res["per_strategy"] = tab.to_dict("records")

# leave-one-strategy-out / leave-one-ticker-out on the pooled closed-only ratio
loo = {}
for s in oos_cl["Strategy"].unique():
    o = oos_cl[oos_cl["Strategy"] != s]
    loo[s] = {"N": len(o), "ratio": round(mix_ratio(o, ins_all)[0], 3)}
res["leave_one_strategy_out"] = loo
print("\nLOSO ratio (closed-only):", loo)
lto = {}
for t in oos_cl["Ticker"].value_counts().head(8).index:
    o = oos_cl[oos_cl["Ticker"] != t]
    lto[t] = {"N_dropped": int((oos_cl["Ticker"] == t).sum()), "sumR_dropped": round(oos_cl.loc[oos_cl["Ticker"] == t, "R_Multiple"].sum(), 2), "ratio_without": round(mix_ratio(o, ins_all)[0], 3)}
res["leave_one_ticker_out"] = lto
print("LOTO ratio (closed-only, top tickers):", lto)

# within-2026 control: same year, before vs after the freeze, per strategy mix-matched
pre26 = led[led["y2026"] & ~led["post"]]
post26 = led[led["y2026"] & led["post"] & ~led["open_mark"]]
ratio_pre26, exp_pre26 = mix_ratio(pre26, led[~led["y2026"]])
ratio_post26, exp_post26 = mix_ratio(post26, led[~led["y2026"]])
res["within_2026"] = {"Jan2026_to_freeze": {**st(pre26), "ratio_vs_pre2026_at_mix": ratio_pre26, "IS_at_mix": exp_pre26},
                      "freeze_to_Aug28_closed": {**st(post26), "ratio_vs_pre2026_at_mix": ratio_post26, "IS_at_mix": exp_post26}}
print(f"\nwithin-2026: Jan->freeze N={len(pre26)} avgR={pre26['R_Multiple'].mean():.3f} ratio={ratio_pre26:.2f} | freeze->Aug28 (closed) N={len(post26)} avgR={post26['R_Multiple'].mean():.3f} ratio={ratio_post26:.2f}")
# 2026 whole year vs pre-2026, and the other midterm years' Jan-Aug at the 2026 mix
y26 = led[led["y2026"] & ~led["open_mark"]]
r26, e26 = mix_ratio(y26, led[~led["y2026"]])
mid = {}
for y in (2006, 2010, 2014, 2018, 2022):
    w = led[(led["Signal Date"] >= f"{y}-01-01") & (led["Signal Date"] <= f"{y}-08-28")]
    rest = led[(led["Signal Date"] < f"{y}-01-01") | (led["Signal Date"] > f"{y}-12-31")]
    if len(w):
        mid[y] = {"N": len(w), "avgR": round(w["R_Multiple"].mean(), 3), "ratio_vs_rest_at_mix": round(mix_ratio(w, rest)[0], 2)}
res["y2026_vs_pre2026"] = {"N": len(y26), "avgR": float(y26["R_Multiple"].mean()), "ratio": r26, "IS_at_mix": e26}
res["midterm_JanAug_controls"] = mid
print("2026 (closed) vs pre-2026 ratio:", round(r26, 2), "| midterm Jan-Aug controls:", mid)
# monthly path of 2026 (does one month carry it?)
y26m = led[led["y2026"]].groupby(led["Signal Date"].dt.to_period("M")).agg(N=("R_Multiple", "size"), avgR=("R_Multiple", "mean"), sumR=("R_Multiple", "sum"))
print("\n2026 by signal month:\n", y26m.round(3).to_string())
res["y2026_by_month"] = {str(k): {"N": int(v["N"]), "avgR": float(v["avgR"]), "sumR": float(v["sumR"])} for k, v in y26m.iterrows()}
(HERE / "oos_reread.json").write_text(json.dumps(res, indent=1, default=str))
