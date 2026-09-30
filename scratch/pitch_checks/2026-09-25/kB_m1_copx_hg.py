"""kB M1 round 1: long COPX against beta-HG=F after copper miners lag the metal by
>= 8pp over 21 sessions with HG=F within 3% of its 252 high. FCX (2000+) and
SCCO as long-history proxies. Pre-specified LONG miner residual (catch-up).

Pair construction: a daily-rebalanced synthetic pair index on the joint calendar,
r_pair_t = r_miner_t - beta_{t-1} * r_metal_t with beta from the trailing 252
daily returns (ex-ante, no look-ahead). The two-factor residual (miner on metal
AND SPY) is built the same way and reported for attribution only.
Reference class: the same form on GDX/GLD, GDXJ/GLD, XME/HG=F, SCCO/HG=F.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

TK = ["COPX", "FCX", "SCCO", "HG=F", "GDX", "GDXJ", "GLD", "XME", "SPY", "TECK"]
raw = load_prices(TK)
cl = {t: raw[t]["Close"].dropna() for t in TK}


def pair_index(miner: str, metal: str, spy: bool = False) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    cols = [miner, metal] + (["SPY"] if spy else [])
    df = pd.concat([cl[c] for c in cols], axis=1, keys=cols).dropna()
    r = df.pct_change()
    if not spy:
        b = (r[miner].rolling(252).cov(r[metal]) / r[metal].rolling(252).var()).shift(1)
        rp = r[miner] - b * r[metal]
        bet = pd.DataFrame({"b_metal": b})
    else:
        y, X = r[miner].values, r[[metal, "SPY"]].values
        B = np.full((len(r), 2), np.nan)
        for i in range(253, len(r)):
            yy, XX = y[i - 252:i], X[i - 252:i]
            ok = ~np.isnan(yy) & ~np.isnan(XX).any(1)
            XX1 = np.column_stack([np.ones(ok.sum()), XX[ok]])
            B[i] = np.linalg.lstsq(XX1, yy[ok], rcond=None)[0][1:]
        bet = pd.DataFrame(B, index=r.index, columns=["b_metal", "b_spy"])
        rp = r[miner] - bet["b_metal"] * r[metal] - bet["b_spy"] * r["SPY"]
    idx = rp.dropna().index
    p = (1 + rp.loc[idx]).cumprod()
    return p, df.loc[idx], bet.loc[idx]


def state(df: pd.DataFrame, miner: str, metal: str, lag_pp: float = 0.08,
          near: float = 0.03, lb: int = 21) -> pd.Series:
    mr = df[miner] / df[miner].shift(lb) - 1
    tr = df[metal] / df[metal].shift(lb) - 1
    hi = df[metal].rolling(252).max()
    return ((mr - tr) <= -lag_pp) & (df[metal] >= (1 - near) * hi)


# ---------- live premise
print("=== 0. live premise ===")
for m, mt in [("COPX", "HG=F"), ("FCX", "HG=F"), ("SCCO", "HG=F"), ("XME", "HG=F"), ("GDX", "GLD")]:
    df = pd.concat([cl[m], cl[mt], cl["SPY"]], axis=1, keys=[m, mt, "SPY"]).dropna()
    mr = df[m].iloc[-1] / df[m].iloc[-22] - 1
    tr = df[mt].iloc[-1] / df[mt].iloc[-22] - 1
    sr = df["SPY"].iloc[-1] / df["SPY"].iloc[-22] - 1
    off = df[mt].iloc[-1] / df[mt].rolling(252).max().iloc[-1] - 1
    print(f"{m:5s} 21d {100*mr:+6.2f}%  {mt} 21d {100*tr:+6.2f}%  lag {100*(mr-tr):+6.2f}pp  "
          f"{mt} off 252 hi {100*off:+.2f}%  SPY 21d {100*sr:+.2f}%  on {df.index[-1].date()}")

for miner, metal in [("COPX", "HG=F"), ("FCX", "HG=F")]:
    p, df, bet = pair_index(miner, metal)
    p2, df2, bet2 = pair_index(miner, metal, spy=True)
    print(f"\n\n################ {miner} vs beta-{metal} ################")
    print(f"ex-ante beta on {metal} today {bet['b_metal'].iloc[-1]:.3f}; 2-factor b_metal "
          f"{bet2['b_metal'].iloc[-1]:.3f} b_spy {bet2['b_spy'].iloc[-1]:.3f}")
    px = pd.DataFrame({"PAIR": p})
    px2 = pd.DataFrame({"RESID2": p2})
    cell = state(df, miner, metal).reindex(px.index).fillna(False).astype(bool)
    print("cell live today:", bool(cell.iloc[-1]), " cell days:", int(cell.sum()))
    mr = df[miner] / df[miner].shift(21) - 1
    tr = df[metal] / df[metal].shift(21) - 1
    hi = df[metal] >= 0.97 * df[metal].rolling(252).max()
    lagv = (mr - tr)
    spy21 = (cl["SPY"] / cl["SPY"].shift(21) - 1).reindex(px.index)
    variants = {
        "PARENT: lag<=-8pp any metal level": (lagv <= -0.08),
        "PARENT: metal within 3% of hi, no lag": hi,
        "COMPLEMENT: lag<=-8pp & metal NOT near hi": (lagv <= -0.08) & ~hi,
        "NB lag<=-6pp": (lagv <= -0.06) & hi,
        "NB lag<=-10pp": (lagv <= -0.10) & hi,
        "NB lag<=-12pp": (lagv <= -0.12) & hi,
        "NB near 2%": (lagv <= -0.08) & (df[metal] >= 0.98 * df[metal].rolling(252).max()),
        "NB near 5%": (lagv <= -0.08) & (df[metal] >= 0.95 * df[metal].rolling(252).max()),
        "NB lookback 10d": state(df, miner, metal, 0.06, 0.03, 10),
        "NB lookback 42d": state(df, miner, metal, 0.10, 0.03, 42),
        "cell & SPY 21d < 0": state(df, miner, metal).reindex(px.index) & (spy21 < 0),
        "cell & SPY 21d >= 0": state(df, miner, metal).reindex(px.index) & (spy21 >= 0),
    }
    variants = {k: v.reindex(px.index).fillna(False).astype(bool) for k, v in variants.items()}
    for h in (5, 10):
        battery(px, cell, [("PAIR", 1.0)], h, f"M1 long {miner} vs beta-{metal} (ex-ante, daily reb.)",
                6.0, variants=variants, min_gap=10, event_kinds=("nfp", "cpi"))
    for h in (5, 10):
        r2 = vehicle_ret(px2, [("RESID2", 1.0)], h)
        c2 = state(df2, miner, metal).reindex(px2.index).fillna(False).astype(bool)
        s = px2.index[(c2 & r2.notna()).values]
        e = declusters(s, 10, px2.index)
        rr = summarize(r2.loc[e].values, f"2-factor resid (metal+SPY) h={h} eps")
        rr["ctrl_all"] = 100 * r2.mean()
        v = r2.loc[e].values
        rr["sign_p"] = sign_test(int((v > 0).sum()), len(v))
        show([rr], f"{miner} 2-factor residual h={h}")

# ---------- reference class: same form on other miner/metal pairs, h=5 and 10
print("\n\n=== reference class: same form (lag<=-8pp over 21d, metal within 3% of 252 hi), long miner vs ex-ante beta metal ===")
rows = []
for miner, metal in [("COPX", "HG=F"), ("FCX", "HG=F"), ("SCCO", "HG=F"), ("TECK", "HG=F"),
                     ("XME", "HG=F"), ("GDX", "GLD"), ("GDXJ", "GLD")]:
    p, df, bet = pair_index(miner, metal)
    px = pd.DataFrame({"PAIR": p})
    cell = state(df, miner, metal).reindex(px.index).fillna(False).astype(bool)
    for h in (5, 10):
        r = vehicle_ret(px, [("PAIR", 1.0)], h)
        s = px.index[(cell & r.notna()).values]
        e = declusters(s, 10, px.index)
        v = r.loc[e].values
        rr = summarize(v, f"{miner}/{metal} h={h}")
        if rr["n"]:
            rr["ctrl_all"] = 100 * r.mean()
            rr["edge"] = rr["mean_pct"] - rr["ctrl_all"]
            rr["sign_p"] = sign_test(int((v > 0).sum()), len(v))
            rr["pre18"] = 100 * np.nanmean(v[e < "2018-01-01"]) if (e < "2018-01-01").any() else np.nan
            rr["post18"] = 100 * np.nanmean(v[e >= "2018-01-01"]) if (e >= "2018-01-01").any() else np.nan
        rows.append(rr)
show(rows, "reference class")
