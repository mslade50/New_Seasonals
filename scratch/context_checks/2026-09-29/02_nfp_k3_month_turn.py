"""E:nfp k3: the session two days before payrolls. NFP sits early in the month, so that session
lands on the month turn. Split it by month position and test it against the same month
positions without payrolls two days later. Tomorrow's h1 is the month's (and quarter's) last session."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

px = load_prices(["SPY", "^GSPC", "QQQ", "IWM", "^VIX", "TLT"])
nfp = load_events(["nfp"])["date"]
nfp = nfp[nfp <= pd.Timestamp("2026-09-29")]


def month_pos(idx: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    per = pd.Series(idx.to_period("M"), index=idx)
    from_end = per.groupby(per.values).cumcount(ascending=False)
    from_start = per.groupby(per.values).cumcount() + 1
    return from_start, from_end


for tk in ["SPY", "^GSPC", "QQQ", "IWM", "^VIX"]:
    c = px[tk]["Close"].astype(float).dropna()
    idx = c.index
    r = c.pct_change()
    fs, fe = month_pos(idx)
    complete = idx.to_period("M") < pd.Period("2026-09", "M")
    pos, kept = anchor_positions(idx, nfp, offset=-2)  # the h1 session itself: 2 td before NFP
    sess = idx[pos]
    sess = sess[sess < idx[-1]]
    v = r.reindex(sess)
    cat = pd.Series(np.where(fe.reindex(sess).values == 0, "last",
                    np.where(fe.reindex(sess).values == 1, "2nd-last",
                    np.where(fs.reindex(sess).values <= 2, "first 1-2",
                    np.where(fs.reindex(sess).values <= 5, "td 3-5", "td 6+")))), index=sess)
    print(f"\n######## {tk}: k3 h1 session (2 td before NFP), {sess[0].date()} to {sess[-1].date()} ########")
    rows = [summarize(v.values, "all k3 h1")]
    for k in ["last", "2nd-last", "first 1-2", "td 3-5", "td 6+"]:
        rows.append(summarize(v[cat == k].values, f"k3 h1 on {k}"))
    show(rows, "by month position of the h1 session")
    # controls: the same month positions on all sessions, complete months, excluding the NFP-k3 sessions
    ok = pd.Series(complete, index=idx) & ~idx.isin(sess)
    ctrl = [summarize(r[ok & (fe == 0)].values, "ctrl: all last sessions"),
            summarize(r[ok & (fe == 1)].values, "ctrl: all 2nd-last"),
            summarize(r[ok & (fs <= 2)].values, "ctrl: all first 1-2"),
            summarize(r[ok & (fs >= 3) & (fs <= 5)].values, "ctrl: all td 3-5"),
            summarize(r[ok & (fs >= 6) & (fe >= 2)].values, "ctrl: all mid-month"),
            summarize(r[ok].values, "ctrl: all days")]
    show(ctrl, "controls (no NFP two sessions later)")
    last = v[cat == "last"].dropna()
    if len(last):
        print("k3 h1 on the month's last session:", [(str(d.date()), round(100 * x, 2)) for d, x in last.items()])
        print("record", int((last > 0).sum()), "up of", len(last), "sign p up", round(sign_test(int((last > 0).sum()), len(last)), 4),
              "sign p down", round(sign_test(int((last < 0).sum()), len(last)), 4))
        q = last[last.index.month.isin([3, 6, 9, 12])]
        print("  ...quarter-end subset:", [(str(d.date()), round(100 * x, 2)) for d, x in q.items()])
    # is the k3 effect anything beyond month position? regress-lite: k3 minus its own position control
    base = {k: r[ok & m].mean() for k, m in [("last", fe == 0), ("2nd-last", fe == 1), ("first 1-2", fs <= 2),
                                              ("td 3-5", (fs >= 3) & (fs <= 5)), ("td 6+", (fs >= 6) & (fe >= 2))]}
    adj = v - cat.map(base)
    s = summarize(adj.dropna().values, "k3 h1 minus same-position control")
    show([s], "position-adjusted")
    show(era_split(adj.dropna().index, adj.dropna().values), "position-adjusted era")
    print("counts by position:", cat.value_counts().to_dict())
