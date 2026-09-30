import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

px = close_panel(["QQQ", "IWM", "SPY"]).dropna()
idx = px.index
spread = (px["QQQ"] / px["QQQ"].shift(21) - 1) - (px["IWM"] / px["IWM"].shift(21) - 1)
sp_rank = spread.rolling(252).rank(pct=True) * 100
iwm_r63 = pct_rank(px["IWM"], 63)
qqq_hi = px["QQQ"] / px["QQQ"].rolling(252).max() - 1
r = px.pct_change()
beta = (r["IWM"].rolling(63).cov(r["QQQ"]) / r["QQQ"].rolling(63).var()).clip(0.3, 1.5)
vix_like = r["QQQ"].rolling(21).std() * np.sqrt(252)

cells = {
    "rank>=99.5 (pre-spec)": sp_rank >= 99.5,
    "rank>=99": sp_rank >= 99,
    "rank>=98": sp_rank >= 98,
    "abs>=10pp": spread >= 0.10,
    "abs>=8pp": spread >= 0.08,
    "rank>=99.5 & IWM r63<=5": (sp_rank >= 99.5) & (iwm_r63 <= 5),
    "rank>=99.5 & QQQ<=1% off hi": (sp_rank >= 99.5) & (qqq_hi >= -0.01),
    "live joint: rank>=99 & r63<=5 & QQQ<=1% off": (sp_rank >= 99) & (iwm_r63 <= 5) & (qqq_hi >= -0.01),
}

rows = []
for h in (5, 10):
    ed = vehicle_ret(px, [("IWM", 1.0), ("QQQ", -1.0)], h)
    # beta-matched: long IWM $1, short QQQ $beta (beta measured at signal)
    bm = fwd_lag(px["IWM"], h) - beta * fwd_lag(px["QQQ"], h)
    for form, ret in (("eq$", ed), ("beta", bm)):
        valid = ret.dropna().index
        pre = valid < pd.Timestamp("2018-01-01")
        d_pre, d_post = ret.loc[valid[pre]].mean(), ret.loc[valid[~pre]].mean()
        for lbl, m in cells.items():
            s = valid[m.reindex(valid, fill_value=False).values]
            e = declusters(s, h, valid)
            v = ret.loc[e].values
            ep = e < pd.Timestamp("2018-01-01")
            w = int((v > 0).sum())
            rows.append({"h": h, "form": form, "cell": lbl, "n": len(v),
                         "rec": f"{w}-{len(v)-w}", "sign_p": round(sign_test(w, len(v)), 3),
                         "mean": round(100 * v.mean(), 3),
                         "xs_drift": round(100 * (v.mean() - ret.loc[valid].mean()), 3),
                         "pre_n": int(ep.sum()), "pre_xs": round(100 * (v[ep].mean() - d_pre), 3),
                         "post_n": int((~ep).sum()),
                         "post_xs": round(100 * (v[~ep].mean() - d_post), 3) if (~ep).sum() else np.nan,
                         "post_rec": f"{int((v[~ep]>0).sum())}-{int((v[~ep]<=0).sum())}",
                         "top2_share": cluster_note(e, v).split("(")[1].split(")")[0] if len(v) > 2 else ""})
pd.set_option("display.width", 250)
print(pd.DataFrame(rows).to_string(index=False))

# abs>=10pp episodes: which years / vol regime
m = spread >= 0.10
s = idx[m.values]
e = declusters(s, 10, idx)
print("\nabs>=10pp h=10 episode dates + QQQ 21d realized vol:")
ret10 = vehicle_ret(px, [("IWM", 1.0), ("QQQ", -1.0)], 10)
print(pd.DataFrame({"ret10": (100 * ret10.loc[e]).round(2), "qqq_rv21": (100 * vix_like.loc[e]).round(1),
                    "spread": (100 * spread.loc[e]).round(1)}).to_string())
print(f"today QQQ rv21 {100*vix_like.iloc[-1]:.1f}")
