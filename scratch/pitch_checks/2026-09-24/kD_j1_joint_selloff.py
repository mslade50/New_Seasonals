"""kD J1 round 1: long a 60/40 SPY/TLT basket the session after a JOINT
stock-bond down day (SPY <= -0.5% AND TLT <= -1.25% same session), h=1..5.
Pre-specified sign LONG (reversal). Job: kill it.

Gate attribution is the whole test:
  (a) SPY leg on joint days vs SPY <= -0.5% days alone (and the complement,
      SPY down but TLT NOT down)
  (b) TLT leg on joint days vs TLT <= -1.25% days alone (and complement)
  (c) 60/40 on joint days vs all-days 60/40
Plus: trailing-63d SPY/TLT correlation regime, pre/post 2018 and 2022.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

TK = ["SPY", "TLT", "^TNX", "^MOVE"]
raw = load_prices(TK)
IDX = raw["TLT"].index
px = close_panel(TK).reindex(IDX)
px = px.dropna(subset=["SPY", "TLT"])
IDX = px.index
rs = px["SPY"].pct_change()
rt = px["TLT"].pct_change()
corr63 = rs.rolling(63).corr(rt)

print(f"panel {IDX[0].date()} .. {IDX[-1].date()}  n={len(IDX)}")
print(f"LIVE {IDX[-1].date()}: SPY {100*rs.iloc[-1]:+.2f}%  TLT {100*rt.iloc[-1]:+.2f}%  "
      f"corr63 {corr63.iloc[-1]:+.3f}  SPY close {px['SPY'].iloc[-1]:.2f} TLT {px['TLT'].iloc[-1]:.2f}")
for t in ("SPY", "TLT"):
    a = wilder_atr(raw[t]["High"].to_numpy(), raw[t]["Low"].to_numpy(), raw[t]["Close"].to_numpy())
    print(f"  {t} Wilder-14 ATR {a[-1]:.3f} ({100*a[-1]/raw[t]['Close'].iloc[-1]:.2f}%)")

S_T, T_T = -0.005, -0.0125
sdown = (rs <= S_T)
tdown = (rt <= T_T)
joint = (sdown & tdown).fillna(False)
print(f"\njoint days N={int(joint.sum())}  SPY-down N={int(sdown.sum())}  TLT-down N={int(tdown.sum())}")
print("joint dates by year:", pd.Series(IDX[joint.values].year).value_counts().sort_index().to_dict())

L6040 = [("SPY", 0.6), ("TLT", 0.4)]
LS = [("SPY", 1.0)]
LT = [("TLT", 1.0)]


def cell(mask, legs, h, lbl, gap=None):
    ret = vehicle_ret(px, legs, h, 1)
    ok = ret.notna()
    d = IDX[(mask.reindex(IDX, fill_value=False) & ok).values]
    e = declusters(d, gap or h, IDX)
    v = ret.loc[e].values
    r = summarize(v, lbl)
    if r["n"]:
        w = int((v > 0).sum())
        r["rec"] = f"{w}-{len(v)-w}"
        r["sign_p"] = sign_test(w, len(v))
    return r, e, v


def all_days(legs, h, lbl):
    ret = vehicle_ret(px, legs, h, 1)
    return summarize(ret.dropna().values, lbl)


# ---------------------------------------------------------------- round 1 battery, 60/40
for h in (1, 3, 5):
    battery(px, joint, L6040, h, f"J1 60/40 after joint down day", cost_bps=1.0,
            variants={
                "SPY<=-0.3 & TLT<=-1.25": (rs <= -0.003) & tdown,
                "SPY<=-0.8 & TLT<=-1.25": (rs <= -0.008) & tdown,
                "SPY<=-0.5 & TLT<=-1.0": sdown & (rt <= -0.010),
                "SPY<=-0.5 & TLT<=-1.5": sdown & (rt <= -0.015),
                "SPY-down alone": sdown,
                "TLT-down alone": tdown,
                "SPY-down & TLT NOT down": sdown & ~tdown,
                "TLT-down & SPY NOT down": tdown & ~sdown,
            },
            event_kinds=("nfp",))

# ---------------------------------------------------------------- gate attribution per leg, h=1..5
print("\n\n######## GATE ATTRIBUTION (episodes, declustered at h) ########")
for h in (1, 2, 3, 4, 5):
    rows = []
    rows.append(cell(joint, L6040, h, "60/40 | JOINT")[0])
    rows.append(all_days(L6040, h, "60/40 | all days"))
    rows.append(cell(sdown, L6040, h, "60/40 | SPY-down alone")[0])
    rows.append(cell(tdown, L6040, h, "60/40 | TLT-down alone")[0])
    rows.append(cell(joint, LS, h, "SPY leg | JOINT")[0])
    rows.append(cell(sdown, LS, h, "SPY leg | SPY-down alone")[0])
    rows.append(cell(sdown & ~tdown, LS, h, "SPY leg | SPY-down, TLT NOT")[0])
    rows.append(all_days(LS, h, "SPY leg | all days"))
    rows.append(cell(joint, LT, h, "TLT leg | JOINT")[0])
    rows.append(cell(tdown, LT, h, "TLT leg | TLT-down alone")[0])
    rows.append(cell(tdown & ~sdown, LT, h, "TLT leg | TLT-down, SPY NOT")[0])
    rows.append(all_days(LT, h, "TLT leg | all days"))
    show([{k: r.get(k) for k in ("label", "n", "mean_pct", "median_pct", "hit", "t", "rec", "sign_p", "worst_pct")}
          for r in rows], f"h={h}")

# ---------------------------------------------------------------- regime + era splits (60/40, and each leg)
print("\n\n######## REGIME / ERA SPLITS (episodes) ########")
pos_c = corr63 > 0
for h in (1, 3, 5):
    rows = []
    for lbl, m in [("JOINT corr63>0", joint & pos_c), ("JOINT corr63<=0", joint & ~pos_c),
                   ("SPY-down alone corr63>0", sdown & pos_c), ("SPY-down alone corr63<=0", sdown & ~pos_c),
                   ("TLT-down alone corr63>0", tdown & pos_c), ("TLT-down alone corr63<=0", tdown & ~pos_c),
                   ("ALL days corr63>0", pos_c), ("ALL days corr63<=0", ~pos_c & corr63.notna())]:
        for legs, ln in [(L6040, "6040"), (LS, "SPY"), (LT, "TLT")]:
            r, _, _ = cell(m, legs, h, f"{ln} | {lbl}", gap=(h if "ALL" not in lbl else 1))
            rows.append(r)
    show([{k: r.get(k) for k in ("label", "n", "mean_pct", "hit", "t", "rec", "sign_p")} for r in rows],
         f"corr regime h={h}")

    rows = []
    for legs, ln in [(L6040, "6040"), (LS, "SPY"), (LT, "TLT")]:
        r, e, v = cell(joint, legs, h, ln)
        d = pd.DatetimeIndex(e)
        for lab, sel in [("<2018", d < "2018-01-01"), ("2018-2021", (d >= "2018-01-01") & (d < "2022-01-01")),
                         (">=2022", d >= "2022-01-01"), ("<2022", d < "2022-01-01")]:
            rr = summarize(v[sel], f"{ln} JOINT {lab}")
            if rr["n"]:
                w = int((v[sel] > 0).sum())
                rr["rec"] = f"{w}-{rr['n']-w}"
                rr["sign_p"] = sign_test(w, rr["n"])
            rows.append(rr)
        # all-days control split at 2022 for the same leg
        ret = vehicle_ret(px, legs, h, 1).dropna()
        rows.append(summarize(ret[ret.index < "2022-01-01"].values, f"{ln} ALL <2022"))
        rows.append(summarize(ret[ret.index >= "2022-01-01"].values, f"{ln} ALL >=2022"))
    show([{k: r.get(k) for k in ("label", "n", "mean_pct", "hit", "t", "rec", "sign_p")} for r in rows],
         f"era split h={h}")

# ---------------------------------------------------------------- per-episode list h=5 with corr
print("\n\n######## JOINT episodes h=5 (60/40, SPY, TLT) ########")
_, e5, _ = cell(joint, L6040, 5, "x")
r6 = vehicle_ret(px, L6040, 5, 1)
r5s = vehicle_ret(px, LS, 5, 1)
r5t = vehicle_ret(px, LT, 5, 1)
for d in e5:
    print(f"  {d.date()}  SPY {100*rs[d]:+.2f} TLT {100*rt[d]:+.2f} corr63 {corr63[d]:+.2f}  "
          f"6040 {100*r6[d]:+.2f}  SPY {100*r5s[d]:+.2f}  TLT {100*r5t[d]:+.2f}")

# ---------------------------------------------------------------- book overlap
print("\n\n######## BOOK OVERLAP ########")
tr = pd.read_parquet(Path(__file__).resolve().parents[3] / "data" / "backtest_trades_full.parquet")
spy = tr[(tr["Ticker"] == "SPY") & (tr["Direction"].astype(str).str.lower().str.startswith("l"))].copy()
spy["Signal Date"] = pd.to_datetime(spy["Signal Date"])
first = spy["Signal Date"].min()
jd = IDX[joint.values]
jd = jd[jd >= first]
hit = [d for d in jd if (spy["Signal Date"] == d).any()]
print(f"ledger SPY longs from {first.date()}: {len(spy)} trades; joint days since then {len(jd)}; "
      f"joint days with a book SPY long signalled same day: {len(hit)}")
for d in hit:
    print("   ", d.date(), spy.loc[spy['Signal Date'] == d, 'Strategy'].tolist())
print("SPY-down days (<=-0.5%) since ledger start:", int((sdown[IDX >= first]).sum()),
      " with book SPY long signal:", int(sum((spy['Signal Date'] == d).any() for d in IDX[(sdown & (IDX >= first)).values])))
print("book signals on 2026-09-23:", tr[pd.to_datetime(tr['Signal Date']) == pd.Timestamp('2026-09-23')][['Strategy', 'Ticker']].values.tolist())
