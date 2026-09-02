"""In-sample vs out-of-sample edge per strategy.

Three OOS definitions, from weakest to strictest:
  (1) post-repo: Signal Date >= 2025-12-18 (strategy_config.py created; every
      rule was designed on data before this, then TUNED through 2026-08 on the
      full history, so this is only pseudo-OOS)
  (2) post-rule-freeze: Signal Date >= the last commit that changed the
      strategy's SIGNAL/ENTRY/EXIT rules (hand-classified from the git log;
      sizing-only commits such as frag bands / ladders / derates are ignored
      because they do not change the signal set)
  (3) post-any-change: Signal Date >= the last commit touching the dict at all
      (estimation_haircut_freeze_dates.json)
Per strategy and pooled: N, avgR, sdR, t, sumR, PnL; bootstrap CI on the pooled
OOS/IS ratio. Also era decay 2003-09 / 2010-16 / 2017-21 / 2022-25 / 2026.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:/Users/McKinley Slade/dev/New_Seasonals")
OUT = ROOT / "scratch/ultracode_sizing_2026-09-02"
RNG = np.random.default_rng(7)

led = pd.read_parquet(ROOT / "data/backtest_trades_full.parquet")
led = led[led["PnL_flat_750k"].notna()].copy()
led["Signal Date"] = pd.to_datetime(led["Signal Date"])
freeze_any = json.load(open(OUT / "estimation_haircut_freeze_dates.json"))

# hand classification of the last SIGNAL-RULE change (see git log of strategy_config.py)
RULE_FREEZE = {
    "Overbot Vol Spike": "2026-06-05",        # 06-02 barbell revert, 06-05 EOD-DD weekday gate (exit rule)
    "Oversold Low Volume": "2026-07-20",      # vol-confirmed stop package (exit rule); 06-24 fill window
    "52wh Breakout": "2026-07-07",            # entry limit -0.5 -> -0.25 ATR
    "Weak Close Decent Sznls": "2026-05-19",  # last filter change (05-05 dropped MOC entry)
    "LT Trend ST OS": "2026-05-01",           # tightened setup; 06-17 is bps only
    "St OS Sznl": "2026-04-23",               # major config rewrite; 07-30 is a sizing derate
    "3x ETF Overbot Fade": "2026-07-28",      # stacking allowed (changes trade count)
    "3x Bear ETF Overbot Fade": "2026-07-28", # created 07-07, stacking 07-28
    "3x Leader Gap Fade": "2026-07-10",       # created
    "Indices Oversold Bounce": "2026-05-12",  # rework
    "SPY QQQ MonFri Reversion": "2026-06-09", # Monday-gap kill filter
    "Sector BO": "2026-05-01",                # created (07-01 audit fix = parity)
    "Monday Dip": "2026-05-12",               # 05-05 created, 05-11/12 open-gap filter
    "ATR Extended Gap Up": "2026-05-11",      # created
    "Monthly Weak Close": "2026-07-31",       # created
}
REPO_DATE = "2025-12-18"
LAST = led["Signal Date"].max()


def stats(df: pd.DataFrame) -> dict:
    r = df["R_Multiple"].to_numpy(float)
    n = len(r)
    if n == 0:
        return {"N": 0}
    m = float(r.mean())
    sd = float(r.std(ddof=1)) if n > 1 else float("nan")
    return {"N": n, "avgR": m, "sdR": sd, "t": float(m / sd * np.sqrt(n)) if n > 1 and sd > 0 else None,
            "sumR": float(r.sum()), "pnl": float(df["PnL_flat_750k"].sum()), "win": float((r > 0).mean())}


res = {"ledger_last_signal": str(LAST.date()), "repo_date": REPO_DATE, "rule_freeze": RULE_FREEZE, "per_strategy": {}}
rows = []
for s, g in led.groupby("Strategy"):
    d_rule = pd.Timestamp(RULE_FREEZE[s])
    d_any = pd.Timestamp(freeze_any[s]["last_change"])
    d_repo = pd.Timestamp(REPO_DATE)
    e = {
        "IS_pre_repo": stats(g[g["Signal Date"] < d_repo]),
        "OOS_post_repo": stats(g[g["Signal Date"] >= d_repo]),
        "IS_pre_rule_freeze": stats(g[g["Signal Date"] < d_rule]),
        "OOS_post_rule_freeze": stats(g[g["Signal Date"] >= d_rule]),
        "OOS_post_any_change": stats(g[g["Signal Date"] >= d_any]),
        "rule_freeze": str(d_rule.date()), "any_change": str(d_any.date()),
        "oos_months_rule": float((LAST - d_rule).days / 30.4),
    }
    res["per_strategy"][s] = e
    rows.append({"Strategy": s, "freeze": e["rule_freeze"], "N_IS": e["IS_pre_rule_freeze"]["N"], "avgR_IS": e["IS_pre_rule_freeze"].get("avgR"),
                 "N_OOS": e["OOS_post_rule_freeze"]["N"], "avgR_OOS": e["OOS_post_rule_freeze"].get("avgR"), "sumR_OOS": e["OOS_post_rule_freeze"].get("sumR"),
                 "N_post_repo": e["OOS_post_repo"]["N"], "avgR_post_repo": e["OOS_post_repo"].get("avgR"), "avgR_pre_repo": e["IS_pre_repo"].get("avgR")})
tab = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(tab.to_string(index=False))

# pooled OOS (each strategy's own freeze) vs its own in-sample, R-weighted by strategy mix
oos = pd.concat([g[g["Signal Date"] >= pd.Timestamp(RULE_FREEZE[s])] for s, g in led.groupby("Strategy")])
ins = pd.concat([g[g["Signal Date"] < pd.Timestamp(RULE_FREEZE[s])] for s, g in led.groupby("Strategy")])
post_repo = led[led["Signal Date"] >= pd.Timestamp(REPO_DATE)]
pre_repo = led[led["Signal Date"] < pd.Timestamp(REPO_DATE)]
res["pooled"] = {"OOS_rule": stats(oos), "IS_rule": stats(ins), "post_repo": stats(post_repo), "pre_repo": stats(pre_repo)}
print("\npooled OOS (rule freeze):", res["pooled"]["OOS_rule"])
print("pooled IS  (rule freeze):", res["pooled"]["IS_rule"])
print("pooled post-repo:", res["pooled"]["post_repo"])
print("pooled pre-repo:", res["pooled"]["pre_repo"])

# expected IS avgR for the OOS strategy mix (so the mix does not confound the ratio)
mix = oos.groupby("Strategy").size()
is_means = ins.groupby("Strategy")["R_Multiple"].mean()
expected = float((mix * is_means.reindex(mix.index)).sum() / mix.sum())
res["pooled"]["IS_avgR_at_OOS_mix"] = expected
ratio = res["pooled"]["OOS_rule"]["avgR"] / expected
res["pooled"]["OOS_over_IS_mix_ratio"] = ratio
# bootstrap the OOS mean (trade-level; signals cluster by day, so also a day-block bootstrap)
r = oos["R_Multiple"].to_numpy(float)
boots = np.array([RNG.choice(r, len(r)).mean() for _ in range(4000)])
days = oos.groupby("Signal Date")["R_Multiple"].apply(list)
dboots = []
for _ in range(4000):
    pick = RNG.choice(len(days), len(days))
    vals = np.concatenate([days.iloc[i] for i in pick])
    dboots.append(vals.mean())
dboots = np.array(dboots)
res["pooled"]["OOS_avgR_ci95_trade"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
res["pooled"]["OOS_avgR_ci95_dayblock"] = [float(np.percentile(dboots, 2.5)), float(np.percentile(dboots, 97.5))]
res["pooled"]["OOS_ratio_ci95_dayblock"] = [float(np.percentile(dboots, 2.5) / expected), float(np.percentile(dboots, 97.5) / expected)]
res["pooled"]["P_oos_mean_below_half_IS_dayblock"] = float((dboots < 0.5 * expected).mean())
print(f"\nIS avgR at OOS mix {expected:.3f}; OOS avgR {res['pooled']['OOS_rule']['avgR']:.3f}; ratio {ratio:.2f}; day-block CI {res['pooled']['OOS_ratio_ci95_dayblock']}")
print("P(OOS mean < 0.5 x IS):", res["pooled"]["P_oos_mean_below_half_IS_dayblock"])

# same for post-repo (longer window, pseudo-OOS)
mix2 = post_repo.groupby("Strategy").size()
is2 = pre_repo.groupby("Strategy")["R_Multiple"].mean()
exp2 = float((mix2 * is2.reindex(mix2.index)).sum() / mix2.sum())
r2 = post_repo["R_Multiple"].to_numpy(float)
days2 = post_repo.groupby("Signal Date")["R_Multiple"].apply(list)
db2 = []
for _ in range(4000):
    pick = RNG.choice(len(days2), len(days2))
    db2.append(np.concatenate([days2.iloc[i] for i in pick]).mean())
db2 = np.array(db2)
res["pooled"]["post_repo_IS_avgR_at_mix"] = exp2
res["pooled"]["post_repo_ratio"] = float(r2.mean() / exp2)
res["pooled"]["post_repo_ratio_ci95_dayblock"] = [float(np.percentile(db2, 2.5) / exp2), float(np.percentile(db2, 97.5) / exp2)]
print(f"post-repo: IS-at-mix {exp2:.3f}, OOS {r2.mean():.3f}, ratio {r2.mean()/exp2:.2f}, CI {res['pooled']['post_repo_ratio_ci95_dayblock']}")

# era decay per strategy (evidence about non-stationarity, separate from selection)
eras = [("2003-09", "2003-01-01", "2009-12-31"), ("2010-16", "2010-01-01", "2016-12-31"), ("2017-21", "2017-01-01", "2021-12-31"), ("2022-25", "2022-01-01", "2025-12-17"), ("2026", "2025-12-18", "2027-01-01")]
era_tab = {}
for s, g in led.groupby("Strategy"):
    era_tab[s] = {}
    for name, a, b in eras:
        sub = g[(g["Signal Date"] >= a) & (g["Signal Date"] <= b)]
        era_tab[s][name] = {"N": int(len(sub)), "avgR": float(sub["R_Multiple"].mean()) if len(sub) else None}
res["era_table"] = era_tab
book_era = {}
for name, a, b in eras:
    sub = led[(led["Signal Date"] >= a) & (led["Signal Date"] <= b)]
    book_era[name] = stats(sub)
res["book_era"] = book_era
print("\nbook by era:", {k: (v["N"], round(v["avgR"], 3)) for k, v in book_era.items()})
print(pd.DataFrame({s: {k: (v["N"], None if v["avgR"] is None else round(v["avgR"], 2)) for k, v in e.items()} for s, e in era_tab.items()}).T.to_string())

(OUT / "estimation_haircut_oos.json").write_text(json.dumps(res, indent=1, default=str))
