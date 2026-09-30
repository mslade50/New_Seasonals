import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

PROX = ["SPY", "IWM", "TLT", "HYG", "GLD", "SLV", "USO", "DX-Y.NYB", "EFA", "^VIX", "SVXY"]
FEAT_T = ["SPY", "IWM", "^TNX", "DX-Y.NYB", "^VIX", "^MOVE", "GLD"]
raw = close_panel(sorted(set(PROX + FEAT_T)))
spy_idx = raw["SPY"].dropna().index
px = raw.reindex(spy_idx).ffill(limit=3)
idx = px.index

F = pd.DataFrame(index=idx)
F["spy_off_hi"] = px["SPY"] / rolling_on_valid(px["SPY"], lambda x: x.rolling(252).max()) - 1
F["iwm_r63"] = pct_rank(px["IWM"], 63)
F["tnx_r63"] = pct_rank(px["^TNX"], 63)
F["dx_r21"] = pct_rank(px["DX-Y.NYB"], 21)
F["vix"] = px["^VIX"]
F["move_r21"] = pct_rank(px["^MOVE"], 21)
F["gld_off_hi"] = px["GLD"] / rolling_on_valid(px["GLD"], lambda x: x.rolling(252).max()) - 1
_fr = pd.read_parquet(Path(__file__).resolve().parents[3] / "data" / "rd2_fragility.parquet")
frag = _fr["63d"].rolling(10).mean()  # main_score is populated only from 2026-09-17; ma10(63d) is the dial
F8 = F.copy()
F8["frag_ma10"] = frag.reindex(idx)

TODAY = idx[-1]


def zstd(F):
    mu = F.rolling(1260, min_periods=504).mean()
    sd = F.rolling(1260, min_periods=504).std()
    return (F - mu) / sd


def eligible(Z, start="2004-01-01", excl_td=63, anchor_pos=None):
    ok = Z.notna().all(axis=1).values & (Z.index >= pd.Timestamp(start))
    pos = np.arange(len(Z))
    ap = len(Z) - 1 if anchor_pos is None else anchor_pos
    ok &= np.abs(pos - ap) > excl_td
    return ok


def knn(Z, anchor_pos, k=20, gap=10, start="2004-01-01", cols=None):
    Zc = Z if cols is None else Z[cols]
    ok = eligible(Zc, start, anchor_pos=anchor_pos)
    x0 = Zc.iloc[anchor_pos].values
    d = np.sqrt(((Zc.values - x0) ** 2).sum(axis=1))
    d[~ok] = np.inf
    order = np.argsort(d)
    picked = []
    for p in order:
        if not np.isfinite(d[p]):
            break
        if all(abs(p - q) >= gap for q in picked):
            picked.append(p)
            if len(picked) == k:
                break
    return np.array(picked), d


FWD = {h: pd.DataFrame({t: fwd_lag(px[t], h) for t in PROX}) for h in (5, 10)}


def grid(picks, span_ok):
    rows = []
    for h in (5, 10):
        fw = FWD[h]
        for t in PROX:
            v = fw[t].values[picks]
            v = v[~np.isnan(v)]
            base = fw[t].values[span_ok]
            base = base[~np.isnan(base)]
            if len(v) < 3:
                rows.append({"h": h, "px": t, "n": len(v)})
                continue
            w = int((v > 0).sum())
            cons = max(w, len(v) - w)
            rows.append({"h": h, "px": t, "n": len(v), "rec": f"{w}-{len(v)-w}",
                         "mean": 100 * v.mean(), "drift": 100 * base.mean(),
                         "edge": 100 * (v.mean() - base.mean()),
                         "edge_t": (v.mean() - base.mean()) / (base.std() / np.sqrt(len(v))),
                         "cons": cons / len(v), "sign_p2": min(1.0, 2 * sign_test(cons, len(v)))})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    Z = zstd(F)
    ap = len(Z) - 1
    print("today raw features:")
    print(F.iloc[-1].round(4).to_string())
    print("today z:", Z.iloc[-1].round(2).to_dict())
    print(f"frag ma10 (main_score) today {F8['frag_ma10'].iloc[-1]:.1f}; frag coverage from {frag.index[0].date()} "
          f"(recompute vintage before 2026-07-02, PIT after)")
    picks, dist = knn(Z, ap, 20)
    ok = eligible(Z, anchor_pos=ap)
    dd = dist[np.isfinite(dist)]
    print(f"\nk=20 declustered nearest (weights all 1.0 on 7 z-features), eligible pool {ok.sum()} days; "
          f"nearest dist {dist[picks[0]]:.2f}, 20th {dist[picks[-1]]:.2f}, median pool dist {np.median(dd):.2f}")
    nb = F.iloc[picks].copy()
    nb["dist"] = dist[picks]
    print(nb.round(3).to_string())
    print("\nneighbour medians vs today:")
    print(pd.DataFrame({"today": F.iloc[-1], "nb_median": nb.drop(columns='dist').median(),
                        "nb_min": nb.drop(columns='dist').min(), "nb_max": nb.drop(columns='dist').max()}).round(3))
    print("neighbour years:", pd.Series(Z.index[picks].year).value_counts().sort_index().to_dict())
    g = grid(picks, ok)
    print("\n=== analogue grid k=20 (lag-1 MOC, fwd from entry close) ===")
    print(g.round(3).to_string(index=False))
