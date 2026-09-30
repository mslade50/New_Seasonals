"""c7 round 1: short SVXY (long vol) from the September opex+1 close across October
in election years (presidential and midterm), gated on ^VIX within 15% of its 252
low. SVXY is two instruments (-1x until 2018-02-27, -0.5x after): pre-break daily
returns are scaled by 0.5 into a synthetic -0.5x series. Rows: all years, election
vs odd years, VIX spot back to 2000, all months (seasonal row), SPY-hedged residual
(MANDATORY), unconditional short-SVXY carry over the same horizon, offset ladder."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kC_common import *  # noqa

BREAK = pd.Timestamp("2018-02-28")
px = nyse_panel(["SVXY", "SPY", "^VIX", "^VIX3M"])
idx = px.index
r = dret(px["SVXY"])
r_adj = r.where(idx >= BREAK, 0.5 * r)
S = (1 + r_adj.fillna(0)).cumprod()
S[idx < px["SVXY"].first_valid_index()] = np.nan
px["S05"] = S
rs = dret(px["SPY"])
m_pre = (idx < BREAK) & rs.notna().values & r_adj.notna().values
b_pre = np.polyfit(rs[m_pre].values, r_adj[m_pre].values, 1)[0]
m_post = (idx >= BREAK) & rs.notna().values & r_adj.notna().values
b_post = np.polyfit(rs[m_post].values, r_adj[m_post].values, 1)[0]
print(f"synthetic -0.5x SVXY beta on SPY: pre-break {b_pre:.2f}, post-break {b_post:.2f}")
beta_era = pd.Series(np.where(idx < BREAK, b_pre, b_post), index=idx)
# SPY-hedged SHORT SVXY index: -(r_S - beta*r_SPY), daily rebalanced
hres = -(r_adj - beta_era * rs)
H = (1 + hres.fillna(0)).cumprod()
H[idx < px["SVXY"].first_valid_index() + pd.Timedelta(days=2)] = np.nan
px["HSHORT"] = H
px["SHORTS"] = 1.0 / S  # not a tradeable index; we compute short returns directly below

vix = px["^VIX"]
vlow = rolling_on_valid(vix, lambda x: x.rolling(252).min())
near_low = vix / vlow - 1.0
opex = to_sessions(load_events(["opex"])["date"], idx)
vixexp = to_sessions(load_events(["vix_expiry"])["date"], idx)
pos = pd.Series(range(len(idx)), index=idx)
print(f"LIVE {idx[-1].date()}: VIX {vix.iloc[-1]:.2f}, {100*near_low.iloc[-1]:.2f}% above 252 low; "
      f"VIX/VIX3M {vix.iloc[-1]/px['^VIX3M'].iloc[-1]:.3f}")


def seg(series, p0, p1):
    s = series.values
    if p1 >= len(s) or np.isnan(s[p0]) or np.isnan(s[p1]):
        return np.nan
    return s[p1] / s[p0] - 1.0


rows = []
for y in range(2000, 2026):
    sep = [d for d in opex if d.year == y and d.month == 9]
    if not sep:
        continue
    sig = pos[sep[0]]
    ent = sig + 1
    oct_opex = [d for d in opex if d.year == y and d.month == 10][0]
    oct_vx = [d for d in vixexp if d.year == y and d.month == 10][0]
    oct_last = idx[(idx.year == y) & (idx.month == 10)][-1]
    exits = {"h5": ent + 5, "h10": ent + 10, "h15": ent + 15, "octopex": pos[oct_opex],
             "octvx": pos[oct_vx], "octend": pos[oct_last]}
    row = {"year": y, "elect": y % 2 == 0, "midterm": y % 4 == 2, "pres": y % 4 == 0,
           "vix_sig": round(vix.iloc[sig], 2), "near_low_pct": round(100 * near_low.iloc[sig], 1),
           "h_octvx": exits["octvx"] - ent}
    for k, p1 in exits.items():
        row[f"shortS_{k}"] = -seg(px["S05"], ent, p1)
        row[f"hedged_{k}"] = seg(px["HSHORT"], ent, p1)
        row[f"spy_{k}"] = seg(px["SPY"], ent, p1)
        row[f"vix_{k}"] = seg(vix, ent, p1)
    rows.append(row)
Y = pd.DataFrame(rows)

cols = ["year", "vix_sig", "near_low_pct", "h_octvx", "shortS_h10", "shortS_octvx", "hedged_h10",
        "hedged_octvx", "spy_octvx", "vix_h10", "vix_octvx"]
Z = Y.copy()
for c in cols[4:]:
    Z[c] = (100 * Z[c]).round(2)
print("\n=== per-year, entry = Sep opex+1 close (shortS = short synthetic -0.5x SVXY; hedged = SPY-residual short) ===")
print(Z[cols].to_string(index=False))


def cell(vals, label):
    v = np.asarray(vals, float)
    v = v[~np.isnan(v)]
    s = summarize(v, label)
    if len(v):
        w = int((v > 0).sum())
        s["rec"] = f"{w}-{len(v)-w}"
        s["sign_p"] = round(sign_test(w, len(v)), 4)
    return s


svx = Y[Y.year >= 2012]
for k in ["h5", "h10", "h15", "octopex", "octvx", "octend"]:
    out = []
    for lbl, fr in [("election yrs", svx[svx.elect]), ("odd yrs", svx[~svx.elect]),
                    ("midterm", svx[svx.midterm]), ("presidential", svx[svx.pres]),
                    ("election & VIX<=15% of low", svx[svx.elect & (svx.near_low_pct <= 15)]),
                    ("ANY yr & VIX<=15% of low", svx[svx.near_low_pct <= 15]),
                    ("all yrs", svx), ("post-break election", svx[svx.elect & (svx.year >= 2018)])]:
        out.append(cell(fr[f"shortS_{k}"], f"shortS {lbl}"))
        out.append(cell(fr[f"hedged_{k}"], f"HEDGED {lbl}"))
    show(out, f"SVXY 2012-2025, exit {k}")

# VIX spot back to 2000 (mechanism row: does spot VIX rise across election Octobers?)
out = []
for k in ["h10", "octvx", "octend"]:
    for lbl, fr in [("election 2000-24", Y[Y.elect]), ("odd 2001-25", Y[~Y.elect]),
                    ("midterm", Y[Y.midterm]), ("presidential", Y[Y.pres]),
                    ("election pre-2012", Y[Y.elect & (Y.year < 2012)]),
                    ("any yr VIX<=15% of low", Y[Y.near_low_pct <= 15])]:
        out.append(cell(Y.loc[fr.index, f"vix_{k}"], f"VIX {k} {lbl}"))
show(out, "^VIX spot % change from Sep opex+1 close")

# unconditional short-SVXY carry at matching horizons (the cost of being short)
out = []
fS = {h: -(px["S05"].shift(-h) / px["S05"] - 1) for h in (10, 22)}
fH = {h: px["HSHORT"].shift(-h) / px["HSHORT"] - 1 for h in (10, 22)}
for h in (10, 22):
    for lbl, m in [("all days 2012+", idx >= "2012-01-01"), ("post-break", idx >= BREAK),
                   ("pre-break", (idx < BREAK) & (idx >= "2012-01-01"))]:
        out.append(summarize(fS[h][m].dropna().values, f"UNCOND short S05 h={h} {lbl}"))
        out.append(summarize(fH[h][m].dropna().values, f"UNCOND hedged short h={h} {lbl}"))
show(out, "unconditional short-SVXY carry (day-level, overlapping)")

# seasonal row: same construction from EVERY month's opex+1 over h=22, by month
out = []
for mth in range(1, 13):
    vs, vh, vv = [], [], []
    for d in opex:
        if d.month != mth or d.year < 2012:
            continue
        ent = pos[d] + 1
        if ent + 22 >= len(idx):
            continue
        vs.append(-seg(px["S05"], ent, ent + 22))
        vh.append(seg(px["HSHORT"], ent, ent + 22))
        vv.append(seg(vix, ent, ent + 22))
    a, b_, c = cell(vs, f"m{mth:02d} shortS"), cell(vh, "h"), cell(vv, "v")
    out.append({"month": mth, "n": a["n"], "shortS_mean": round(a["mean_pct"], 2), "shortS_rec": a["rec"],
                "hedged_mean": round(b_["mean_pct"], 2), "hedged_rec": b_["rec"],
                "vix_mean": round(c["mean_pct"], 2), "vix_rec": c["rec"]})
print("\n=== seasonal row: opex+1 -> +22 sessions by opex month, 2012-2025 (Sep row = today's window) ===")
print(pd.DataFrame(out).to_string(index=False))
