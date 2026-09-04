import numpy as np
import pandas as pd

ROOT = r"C:\Users\McKinley Slade\dev\New_Seasonals"

# my rv series rebuilt
px = pd.read_parquet(ROOT + r"\data\master_prices.parquet",
                     filters=[("ticker", "in", ["SPY"])])
px["date"] = pd.to_datetime(px["date"])
spy = px.set_index("date")["Close"].sort_index()
ret = spy.pct_change()
rv21 = ret.rolling(21).std() * np.sqrt(252)
rv_pct_756 = rv21.rolling(756).rank(pct=True)
rv_pct_252 = rv21.rolling(252).rank(pct=True)
rv_pct_504 = rv21.rolling(504).rank(pct=True)
rv_pct_full = rv21.rank(pct=True)
rv_pct_exp = rv21.expanding().rank(pct=True)

j = pd.read_parquet(ROOT + r"\scratch\ultracode_research\proxy_joined.parquet")
print("their joined frame cols:", j.columns.tolist())
print("their N:", len(j))
dcol = [c for c in j.columns if "ignal" in c or c.lower() == "date"]
print("date-ish cols:", dcol)
sd = pd.to_datetime(j[dcol[0]])
comp = pd.DataFrame({"sig": sd, "their_rv": j["rv21_pct"].values})
comp["mine_756"] = rv_pct_756.reindex(sd, method="ffill").values
comp["mine_252"] = rv_pct_252.reindex(sd, method="ffill").values
comp["mine_504"] = rv_pct_504.reindex(sd, method="ffill").values
comp["mine_full"] = rv_pct_full.reindex(sd, method="ffill").values
comp["mine_exp"] = rv_pct_exp.reindex(sd, method="ffill").values
for c in ["mine_756", "mine_252", "mine_504", "mine_full", "mine_exp"]:
    d = (comp["their_rv"] - comp[c]).abs()
    print(f"{c}: mean|diff|={d.mean():.4f} max={d.max():.4f} corr={comp['their_rv'].corr(comp[c]):.4f}")

# where do they disagree most vs 756?
comp["diff"] = comp["their_rv"] - comp["mine_756"]
big = comp[comp["diff"].abs() > 0.15]
print("\nrows with |their - mine756| > 0.15:", len(big))
print(big.groupby(sd.dt.year)["diff"].agg(["count", "mean"]))

# my rv_pct in their claimed toxic windows
for a, b in [("2021-05-01", "2022-01-31"), ("2024-04-01", "2024-04-30"),
             ("2024-08-01", "2024-12-31")]:
    w = rv_pct_756.loc[a:b]
    tw = comp[(comp.sig >= a) & (comp.sig <= b)]
    print(f"\n{a}..{b}: my daily rv_pct756 mean={w.mean():.3f} max={w.max():.3f}; "
          f"their trade-level rv mean={tw['their_rv'].mean():.3f} max={tw['their_rv'].max():.3f}")

# their frag col vs threshold and their toxic cell recomputed from their frame
fcol = [c for c in j.columns if "frag" in c.lower()]
print("\nfrag cols in their frame:", fcol)
fh = j[fcol[0]] >= 50
rv_thr = j["rv21_pct"].quantile(1 - fh.mean())
print("their frame: frag-hi N =", fh.sum(), " rv_thr(matched hi tail) =", round(rv_thr, 3))
tox = j[fh & (j["rv21_pct"] >= rv_thr)]
print("their-frame toxic cell N =", len(tox), " avgR =", tox["R_Multiple"].mean() if "R_Multiple" in j else "?")
print("toxic years:", pd.to_datetime(tox[dcol[0]]).dt.year.value_counts().to_dict())
