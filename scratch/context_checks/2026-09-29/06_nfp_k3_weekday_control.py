"""02 follow-up. Most k3 h1 sessions are Wednesdays. Re-control the month's-last-session split and the
position-adjusted cell on weekday x month position, then check era, concentration and the quarter-end subset."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

px = load_prices(["SPY", "^GSPC", "^VIX", "IWM", "QQQ"])
nfp = load_events(["nfp"])["date"]
nfp = nfp[nfp <= pd.Timestamp("2026-09-29")]

for tk in ["SPY", "^GSPC", "^VIX", "IWM"]:
    c = px[tk]["Close"].astype(float).dropna()
    idx = c.index
    r = c.pct_change()
    per = pd.Series(idx.to_period("M"), index=idx)
    fe = per.groupby(per.values).cumcount(ascending=False)
    fs = per.groupby(per.values).cumcount() + 1
    comp = pd.Series(idx.to_period("M") < pd.Period("2026-09", "M"), index=idx)
    pos, _ = anchor_positions(idx, nfp, offset=-2)
    sess = idx[pos]
    sess = sess[sess < idx[-1]]
    is_k3 = pd.Series(idx.isin(sess), index=idx)
    wd = pd.Series(idx.dayofweek, index=idx)
    print(f"\n######## {tk} ########")
    k3_last = r[is_k3 & (fe == 0)].dropna()
    print("k3-on-last weekdays:", pd.Series(k3_last.index.dayofweek).value_counts().to_dict())
    ctrl_last_wed = r[~is_k3 & (fe == 0) & comp & (wd == 2)].dropna()
    ctrl_last = r[~is_k3 & (fe == 0) & comp].dropna()
    rows = [summarize(k3_last.values, "last session, NFP 2 td later"),
            summarize(k3_last[k3_last.index.dayofweek == 2].values, "  ...Wednesdays only"),
            summarize(ctrl_last.values, "last session, no NFP 2 td later"),
            summarize(ctrl_last_wed.values, "  ...Wednesdays only"),
            summarize(r[~is_k3 & comp & (wd == 2)].values, "all other Wednesdays")]
    show(rows, "month's last session")
    a, b = k3_last.values, ctrl_last.values
    wt = (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    print("Welch t (k3-last vs other last):", round(wt, 2))
    print("sign p of k3-last up count vs the other-last up rate:",
          round(sign_test(int((a > 0).sum()), len(a), float((b > 0).mean())), 4))
    show(era_split(k3_last.index, k3_last.values), "k3-last era")
    print(cluster_note(k3_last.index, k3_last.values))
    trimmed = k3_last.drop(k3_last.abs().nlargest(2).index)
    print("k3-last without the two largest moves:", round(100 * trimmed.mean(), 3), int((trimmed > 0).sum()), "of", len(trimmed))
    # weekday x position adjusted full cell
    cat_pos = np.where(fe == 0, "last", np.where(fe == 1, "2nd-last", np.where(fs <= 2, "first12", np.where(fs <= 5, "td35", "mid"))))
    key = pd.Series([f"{p}|{d}" for p, d in zip(cat_pos, wd.values)], index=idx)
    base = r[~is_k3 & comp].groupby(key[~is_k3 & comp]).mean()
    adj = (r[is_k3] - key[is_k3].map(base)).dropna()
    show([summarize(adj.values, "k3 h1 minus weekday x position control")] + era_split(adj.index, adj.values), "weekday x position adjusted")
    qe = k3_last[k3_last.index.month.isin([3, 6, 9, 12])]
    print("quarter-end k3-last:", len(qe), "up", int((qe > 0).sum()), "mean", round(100 * qe.mean(), 3))
