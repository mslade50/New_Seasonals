"""Return profile of OLV trades where vol_ratio_10d_rank < 15 persisted
for >10 consecutive days ending on the signal date.

vol_ratio_10d_rank = expanding(min_periods=252) pct-rank of
(10d mean Volume / 63d mean Volume) * 100  -- matches indicators.py.
"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RANK_MIN_PERIODS = 252
THRESH = 15.0
STRAT = "Oversold Low Volume"

trades = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
olv = trades[trades["Strategy"] == STRAT].copy()
olv["Signal Date"] = pd.to_datetime(olv["Signal Date"])
print(f"OLV trades in ledger: {len(olv)}  tiers={olv['Tier'].value_counts().to_dict()}")

mp = pd.read_parquet(ROOT / "data" / "master_prices.parquet")
mp["date"] = pd.to_datetime(mp["date"])

# overflow tier names may live in overflow_prices.parquet
ovf_path = ROOT / "data" / "overflow_prices.parquet"
price_frames = {}


def vol_rank_run(df):
    df = df.sort_values("date")
    vma63 = df["Volume"].rolling(63).mean()
    vma10 = df["Volume"].rolling(10).mean()
    ratio = vma10 / vma63
    rank = ratio.expanding(min_periods=RANK_MIN_PERIODS).rank(pct=True) * 100.0
    below = (rank < THRESH).fillna(False)
    # consecutive run length ending on each row
    grp = (~below).cumsum()
    run = below.groupby(grp).cumsum()
    out = pd.DataFrame({"date": df["date"].values,
                        "vr_rank": rank.values,
                        "run": run.values})
    return out


# build per-ticker rank/run lookups for tickers we need
need = set(olv["Ticker"].unique())
have_master = set(mp["ticker"].unique())
src_master = need & have_master
missing = need - have_master

ovf = None
if ovf_path.exists():
    ovf = pd.read_parquet(ovf_path)
    ovf["date"] = pd.to_datetime(ovf["date"])

lookups = {}
for tk in sorted(need):
    sub = mp[mp["ticker"] == tk]
    if len(sub) < RANK_MIN_PERIODS and ovf is not None and "ticker" in ovf.columns:
        sub2 = ovf[ovf["ticker"] == tk]
        if len(sub2) > len(sub):
            sub = sub2
    if len(sub) >= 63:
        lookups[tk] = vol_rank_run(sub).set_index("date")

# attach run length at signal date to each trade
runs, ranks = [], []
for _, r in olv.iterrows():
    lk = lookups.get(r["Ticker"])
    if lk is None or r["Signal Date"] not in lk.index:
        runs.append(np.nan); ranks.append(np.nan); continue
    row = lk.loc[r["Signal Date"]]
    runs.append(row["run"]); ranks.append(row["vr_rank"])
olv["vr_rank_at_signal"] = ranks
olv["lowvol_run"] = runs

matched = olv.dropna(subset=["lowvol_run"]).copy()
print(f"matched to price history: {len(matched)} / {len(olv)} "
      f"(unmatched tickers: {sorted(need - set(lookups))[:10]})")
print(f"rank<15 at signal among matched: "
      f"{(matched['vr_rank_at_signal'] < THRESH).mean():.1%}")


def profile(df, label):
    if len(df) == 0:
        print(f"\n{label}: (no trades)"); return
    ret = df["Return_Pct"]; r = df["R_Multiple"]
    print(f"\n{label}  (N={len(df)})")
    print(f"  win rate (R>0)   : {(r > 0).mean():6.1%}")
    print(f"  avg R            : {r.mean():+.3f}    median R: {r.median():+.3f}")
    print(f"  avg return %     : {ret.mean():+.3f}    median %: {ret.median():+.3f}")
    print(f"  total R          : {r.sum():+.1f}")
    print(f"  avg hold (days)  : {(df['Exit Date'] - df['Entry Date']).dt.days.mean():.1f}")
    print(f"  exit type mix    : {df['Exit Type'].value_counts().to_dict()}")


print("\n" + "=" * 60)
print("RUN-LENGTH DISTRIBUTION (consecutive days rank<15 incl. signal day)")
print(matched["lowvol_run"].describe().to_string())
for lo in [1, 5, 10, 11, 15, 20]:
    print(f"  run >= {lo:2d}: {int((matched['lowvol_run'] >= lo).sum()):4d} trades")

print("\n" + "=" * 60)
profile(matched, "ALL matched OLV trades")
profile(matched[matched["lowvol_run"] <= 10], "Run <= 10 days")
profile(matched[matched["lowvol_run"] > 10], ">>> Run > 10 days (the ask)")

# tier split for the target subset
tgt = matched[matched["lowvol_run"] > 10]
for tier in tgt["Tier"].unique():
    profile(tgt[tgt["Tier"] == tier], f"Run>10, tier={tier}")
