"""c3 round 1 (+ the mandatory round-2 ladder, run up front because the whole
lane lives or dies on it): short a liquid single stock at/near its 52w low
(close within 3% of the 252-session closing min AND r21 <= 15) from T-8 to
the pre-print close T-1, pooled.

Timing, matched to the live NKE case: signal close = T-9 (09-18), entry MOC
T-8 (09-21), exit MOC T-1 (09-30) -> h=7. T = first session on/after the
calendar's announcement date. No BMO/AMC field: T-1 exit never holds a print.

Returns: short raw = -r; SPY-rel = -(r - r_spy); beta-hedged = -(r - b*r_spy)
with b the name's 252d beta at the signal close. Every summary is WEEK-
CLUSTERED (cross-sectional mean per entry week) because many names share a
calendar date. Survivorship: the cache holds today's names only; missing
delistings are LOSERS, so a short-loser edge here is understated, not flattered.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kB_common import *  # noqa

H = 7
P = build_panel()
C, beta, lodist, r21, r63 = P["C"], P["beta"], P["lodist"], P["r21"], P["r63"]
idx = C.index
spy = C["SPY"].values
E = P["E"]
LIQ_U = [t for t in LIQ if t in C.columns]
print(f"panel {C.shape}, {idx[0].date()}..{idx[-1].date()}; LIQ singles {len(LIQ_U)}")

qe_sess = np.zeros(len(idx), bool)
per = idx.to_period("M")
last_of_month = np.r_[per[1:] != per[:-1], False]
qe_sess[(last_of_month) & np.isin(idx.month, [3, 6, 9, 12])] = True
qe_cum = np.cumsum(qe_sess)


def build(univ, k_list=range(-10, 6)):
    Ev = event_positions(E, idx, univ)
    ci = np.array([C.columns.get_loc(t) for t in Ev.ticker])
    pT = Ev.pT.values
    V = C.values
    B, LD, R21, R63 = beta.values, lodist.values, r21.values, r63.values
    out = []
    for k in k_list:
        sig = pT + k - 9
        ent = sig + 1
        ex = ent + H
        ok = (sig >= 252) & (ex < len(idx))
        s, e, x, c = sig[ok], ent[ok], ex[ok], ci[ok]
        rn = V[x, c] / V[e, c] - 1.0
        rs = spy[x] / spy[e] - 1.0
        b = B[s, c]
        df = pd.DataFrame({
            "k": k, "ticker": C.columns[c], "sig_pos": s,
            "entry_date": idx[e], "T": idx[np.minimum(pT[ok], len(idx) - 1)],
            "lodist": LD[s, c], "r21": R21[s, c], "r63": R63[s, c],
            "short": -rn, "rel": -(rn - rs), "bh": -(rn - b * rs), "beta": b,
            "qe_in": (qe_cum[x] - qe_cum[e]) > 0,
            "spy200": P["spy_above200"].values[s],
        })
        out.append(df)
    D = pd.concat(out, ignore_index=True)
    D = D.dropna(subset=["short", "bh", "lodist", "r21"])
    D["year"] = D.entry_date.dt.year
    return D


def gate(D, ld=0.03, rr=15.0):
    return D[(D.lodist <= ld) & (D.r21 <= rr)]


for name, univ in [("LIQ", LIQ_U), ("BROAD", [t for t in C.columns if t != "SPY"])]:
    D = build(univ)
    D0 = D[D.k == 0]
    G0 = gate(D0)
    print(f"\n{'=' * 78}\n[{name}] events at k=0: {len(D0)}, gated {len(G0)} "
          f"({G0.entry_date.dt.to_period('W-FRI').nunique()} weeks)\n{'=' * 78}")
    rows = []
    for col in ("short", "rel", "bh"):
        rows.append(cl_stat(G0, col, f"GATED short T-8->T-1 [{col}]"))
    for col in ("short", "rel", "bh"):
        rows.append(cl_stat(D0, col, f"ALL prints short T-8->T-1 [{col}]"))
    show(rows, f"[{name}] 1. pattern vs all-events baseline (week-clustered)")

    # own unconditional drift and the NO-PRINT placebo (same gate, no print near)
    U = univ
    Cu = C[U]
    fwd = Cu.shift(-(1 + H)) / Cu.shift(-1) - 1.0
    spf = C["SPY"].shift(-(1 + H)) / C["SPY"].shift(-1) - 1.0
    bh_all = -(fwd - beta[U].mul(spf, axis=0))
    pm = print_mask(event_positions(E, idx, U), idx, U)
    near = pm.astype(float).rolling(H + 4, min_periods=1).sum().shift(-(H + 2)) > 0
    # coverage: only between a ticker's first and last calendar event
    first = E[E.ticker.isin(U)].groupby("ticker").date.min()
    last = E[E.ticker.isin(U)].groupby("ticker").date.max()
    cov = pd.DataFrame({t: (idx >= first.get(t, idx[-1])) & (idx <= last.get(t, idx[0]))
                        for t in U}, index=idx)
    g = (lodist[U] <= 0.03) & (r21[U] <= 15)
    plc = g & ~near & cov
    fwd_s = -fwd
    rel_all = -(fwd.sub(spf, axis=0))

    def panel_week(mask, val):
        m = mask & val.notna()
        st = val.where(m).stack()
        st.index.names = ["date", "ticker"]
        st = st.reset_index()
        st.columns = ["date", "ticker", "v"]
        # entry date = next session after the signal row
        pos = idx.get_indexer(st.date)
        st["entry_date"] = idx[np.minimum(pos + 1, len(idx) - 1)]
        return st

    rows = []
    own = fwd_s.where(cov).stack()
    rows.append(stat(own.groupby(own.index.get_level_values(0).to_period("W-FRI")).mean().values,
                     "OWN drift: short every name, every day, h=7 [short]"))
    ownb = bh_all.where(cov).stack()
    rows.append(stat(ownb.groupby(ownb.index.get_level_values(0).to_period("W-FRI")).mean().values,
                     "OWN drift [bh]"))
    for col, val in (("short", fwd_s), ("rel", rel_all), ("bh", bh_all)):
        st = panel_week(plc, val)
        st = st[st.date >= pd.Timestamp("1999-12-01")]
        s = week_cluster(st.entry_date, st.v)
        r = stat(s.values, f"NO-PRINT placebo, same gate [{col}]")
        r["n_obs"] = len(st)
        rows.append(r)
        if col == "bh":
            plc_week_bh = s
        if col == "short":
            plc_week_sh = s
    show(rows, f"[{name}] 1b. own drift + the NO-PRINT placebo (week-clustered)")

    # paired by week: print cell minus no-print placebo in the SAME week
    for col, pw in (("short", plc_week_sh), ("bh", plc_week_bh)):
        gw = week_cluster(G0.entry_date, G0[col])
        j = pd.concat([gw.rename("print"), pw.rename("noprint")], axis=1, join="inner")
        d = (j["print"] - j["noprint"]).values
        r = stat(d, f"PAIRED same-week print minus no-print [{col}]")
        show([r], f"[{name}] 1c. paired difference")

    # era / regime / QE split on the gated cell (bh)
    rows = []
    for lbl, m in [("pre-2018", G0.year < 2018), ("2018+", G0.year >= 2018),
                   ("midterm yrs", G0.year % 4 == 2), ("non-midterm", G0.year % 4 != 2),
                   ("SPY > 200d", G0.spy200), ("SPY < 200d", ~G0.spy200.astype(bool)),
                   ("QE inside window", G0.qe_in), ("NO QE in window", ~G0.qe_in.astype(bool)),
                   ("Sep/Oct prints", G0["T"].dt.month.isin([9, 10]))]:
        for col in ("short", "bh"):
            rows.append(cl_stat(G0[m], col, f"{lbl} [{col}]"))
    show(rows, f"[{name}] 2. era / regime / quarter-end split (gated, week-clustered)")
    if len(G0):
        gw = week_cluster(G0.entry_date, G0.bh)
        print("  worst week-cluster (bh): %.2f%% on %s; worst single obs %.2f%% (%s %s)" % (
            100 * gw.min(), gw.idxmin(), 100 * G0.bh.min(),
            G0.loc[G0.bh.idxmin(), "ticker"], G0.loc[G0.bh.idxmin(), "entry_date"].date()))
        print("  concentration:", cluster_note(gw.index.to_timestamp(), gw.values))

    # gate variants (definition neighbours)
    rows = []
    for ld, rr in [(0.01, 15), (0.02, 15), (0.03, 15), (0.05, 15), (0.03, 10),
                   (0.03, 25), (0.03, 100), (1.0, 15), (1.0, 5), (0.05, 30)]:
        Gv = gate(D0, ld, rr)
        r = cl_stat(Gv, "bh", f"lodist<={ld:.2f} & r21<={rr:g} [bh]")
        rows.append(r)
    Gr63 = D0[(D0.lodist <= 0.03) & (D0.r63 <= 20)]
    rows.append(cl_stat(Gr63, "bh", "lodist<=0.03 & r63<=20 [bh]"))
    show(rows, f"[{name}] 2b. definition neighbours (k=0, week-clustered)")

    # OFFSET PLACEBO LADDER k=-10..+5 (anchor shifted, same gate at shifted signal)
    rows = []
    for k in range(-10, 6):
        Gk = gate(D[D.k == k])
        r = cl_stat(Gk, "bh", f"k={k:+d}")
        r2 = cl_stat(Gk, "short", "")
        r["short_mean_pct"] = r2.get("mean_pct")
        rows.append(r)
    L = pd.DataFrame(rows)
    L["rank_bh"] = L.mean_pct.rank(ascending=False).astype(int)
    L["rank_short"] = L.short_mean_pct.rank(ascending=False).astype(int)
    show(L[["label", "n", "n_obs", "mean_pct", "t", "hit", "rec", "rank_bh",
            "short_mean_pct", "rank_short"]].to_dict("records"),
         f"[{name}] 3. OFFSET LADDER (gated, bh; k=0 is the true anchor, k>0 holds through the print)")
    k0 = L[L.label == "k=+0"].iloc[0]
    others = L[L.label != "k=+0"]
    print(f"  true anchor rank (bh) {int(k0.rank_bh)} of {len(L)}; ladder mean ex-true "
          f"{others.mean_pct.mean():+.3f}%, true {k0.mean_pct:+.3f}%, true minus placebo "
          f"{k0.mean_pct - others.mean_pct.mean():+.3f}pp; pre-print-only rungs k<=0 "
          f"mean {L[L.label.isin([f'k={k:+d}' for k in range(-10, 0)])].mean_pct.mean():+.3f}%")

# ---------------------------------------------------------------------------
# live: prints in the next 10 sessions for LIQ, with state on 09-18
# ---------------------------------------------------------------------------
last = idx[-1]
fut = pd.bdate_range(last + pd.Timedelta(days=1), periods=12)
Ef = E[(E.date > last) & (E.date <= fut[-1]) & E.ticker.isin(LIQ_U)]
print(f"\nLIVE (signal {last.date()}): LIQ prints through {fut[-1].date()}")
for _, r in Ef.sort_values("date").iterrows():
    t = r.ticker
    kk = int(np.searchsorted(fut, r.date)) + 1  # sessions after the signal close
    print(f"  {t:5s} {r.date.date()} (T = signal+{kk}; entry 09-21 -> T-1 is h={kk - 2})  "
          f"lodist {100 * lodist[t].iloc[-1]:.2f}%  r21 {r21[t].iloc[-1]:.1f}  "
          f"r63 {r63[t].iloc[-1]:.1f}  beta {beta[t].iloc[-1]:.2f}  "
          f"c3 gate {'ON' if (lodist[t].iloc[-1] <= 0.03 and r21[t].iloc[-1] <= 15) else 'off'}")
