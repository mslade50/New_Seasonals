"""c8 round 1: long yen (short JPY=X) from QE-7 to the quarter-end close in March and
September (Japan fiscal year / half-year ends). Pre-specified falsifications:
(a) Mar+Sep must beat Jun+Dec; (b) ordinary month-ends control; (c) reversal QE ->
QE+5; (d) era (pre-2013 / 2013-2021 / 2022+) and midterm splits; (e) carry cost
(US 3m minus a JP short-rate proxy). Plus the W56 DX row and an FOMC-in-window split.
Long-yen return = JPY_entry / JPY_exit - 1 (USD P&L of holding yen)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-09-17"))
from kC_common import *  # noqa
from k1_common import month_end_positions  # noqa  (09-17 helper: last NYSE session of each month)

px = nyse_panel(["JPY=X", "DX-Y.NYB", "^IRX"], ffill=("JPY=X", "DX-Y.NYB", "^IRX"))
idx = px.index
J = px["JPY=X"].values
D = px["DX-Y.NYB"].values
IRX = px["^IRX"].values
fomc = pd.DatetimeIndex(load_events(["fomc_decision"])["date"])
me = month_end_positions(idx)
print(f"LIVE {idx[-1].date()}: USDJPY {J[-1]:.2f}; ^IRX {IRX[-1]:.2f}; last completed ME {idx[me[-1]].date()}")


def jp_rate(d: pd.Timestamp) -> float:
    # BoJ policy-rate proxy (pct): ~0 until 2024-03, 0.1 to 2024-07, 0.25 to 2025-01, 0.5 after
    if d < pd.Timestamp("2024-03-19"):
        return 0.0
    if d < pd.Timestamp("2024-07-31"):
        return 0.1
    if d < pd.Timestamp("2025-01-24"):
        return 0.25
    return 0.5


ENTRY_OFF, H = -7, 7       # entry at QE-7 close, exit QE close (today: 09-21 -> 09-30)
rows = []
for m in me:
    e, x = m + ENTRY_OFF, m
    if e < 1 or x + 5 >= len(idx):
        continue
    d = idx[m]
    ly = J[e] / J[x] - 1.0
    carry = (IRX[e] - jp_rate(idx[e])) / 100 * H / 252
    f_in = bool(((fomc > idx[e]) & (fomc <= idx[x])).any())
    rows.append({"me": d, "year": d.year, "month": d.month, "qe": d.month in (3, 6, 9, 12),
                 "fiscal": d.month in (3, 9), "midterm": d.year % 4 == 2,
                 "ly": ly, "carry": carry, "ly_net": ly - carry,
                 "rev5": J[x] / J[x + 5] - 1.0,          # long yen QE -> QE+5 (reversal row: expect < 0)
                 "dx": D[x] / D[e] - 1.0, "fomc_in": f_in})
A = pd.DataFrame(rows)


def cell(vals, label):
    v = np.asarray(vals, float)
    v = v[~np.isnan(v)]
    s = summarize(v, label)
    if len(v):
        w = int((v > 0).sum())
        s["rec"] = f"{w}-{len(v)-w}"
        s["sign_p"] = round(sign_test(w, len(v)), 4)
    return s


out = []
for era, fr in [("2000+", A), ("2008+", A[A.year >= 2008])]:
    for lbl, m in [("Mar+Sep (fiscal)", fr.fiscal), ("Jun+Dec (non-fiscal QE)", fr.qe & ~fr.fiscal),
                   ("ordinary ME", ~fr.qe), ("September", fr.month == 9), ("March", fr.month == 3),
                   ("June", fr.month == 6), ("December", fr.month == 12)]:
        out.append(cell(fr.loc[m, "ly"], f"{era} {lbl} gross"))
        out.append(cell(fr.loc[m, "ly_net"], f"{era} {lbl} NET of carry"))
show(out, "long yen QE-7 -> QE (gross, and net of US-JP carry)")

f = A[A.fiscal]
o = A[~A.qe]
d1, t1 = welch(f.ly, A[A.qe & ~A.fiscal].ly)
d2, t2 = welch(f.ly, o.ly)
print(f"\nMar+Sep minus Jun+Dec {d1:+.3f}pp (t {t1:+.2f});  Mar+Sep minus ordinary ME {d2:+.3f}pp (t {t2:+.2f})")
s9 = A[A.month == 9]
d3, t3 = welch(s9.ly, o.ly)
print(f"September minus ordinary ME {d3:+.3f}pp (t {t3:+.2f})")

out = []
for lbl, m in [("Sep pre-2013", (A.month == 9) & (A.year < 2013)),
               ("Sep 2013-2021", (A.month == 9) & (A.year >= 2013) & (A.year <= 2021)),
               ("Sep 2022+", (A.month == 9) & (A.year >= 2022)),
               ("Mar+Sep pre-2013", A.fiscal & (A.year < 2013)),
               ("Mar+Sep 2013-2021", A.fiscal & (A.year >= 2013) & (A.year <= 2021)),
               ("Mar+Sep 2022+", A.fiscal & (A.year >= 2022)),
               ("ordinary ME 2022+", ~A.qe & (A.year >= 2022)),
               ("Sep midterm", (A.month == 9) & A.midterm), ("Sep non-midterm", (A.month == 9) & ~A.midterm),
               ("Mar+Sep midterm", A.fiscal & A.midterm),
               ("Sep FOMC inside window", (A.month == 9) & A.fomc_in),
               ("Sep no FOMC in window", (A.month == 9) & ~A.fomc_in)]:
    out.append(cell(A.loc[m, "ly"], lbl))
show(out, "era / midterm / FOMC splits, long yen QE-7 -> QE gross")

out = []
for lbl, m in [("Mar+Sep", A.fiscal), ("Jun+Dec", A.qe & ~A.fiscal), ("ordinary ME", ~A.qe),
               ("September", A.month == 9), ("March", A.month == 3)]:
    out.append(cell(A.loc[m, "rev5"], f"REVERSAL long yen QE -> QE+5 {lbl}"))
show(out, "(c) reversal row: long yen after the close (pressure lifts -> expect NEGATIVE)")

print("\nSeptember by year (long yen QE-7 -> QE %, net %, FOMC in window):")
print(", ".join(f"{r.year}:{100*r.ly:+.2f}/{100*r.ly_net:+.2f}{'F' if r.fomc_in else ''}"
                for r in A[A.month == 9].itertuples()))

# offset ladder: window [QE-7+k, QE+k], fiscal months, k=-5..+5
pos_me = {idx[m]: m for m in me}
lad = []
for k in range(-5, 6):
    vf, vo = [], []
    for m in me:
        e, x = m + ENTRY_OFF + k, m + k
        if e < 1 or x >= len(idx):
            continue
        v = J[e] / J[x] - 1.0
        (vf if idx[m].month in (3, 9) else vo if idx[m].month not in (6, 12) else []).append(v)
    cf, co = cell(vf, ""), cell(vo, "")
    lad.append({"k": k, "fiscal_mean": round(cf["mean_pct"], 3), "fiscal_rec": cf["rec"],
                "ordME_mean": round(co["mean_pct"], 3), "diff_pp": round(cf["mean_pct"] - co["mean_pct"], 3)})
L = pd.DataFrame(lad)
L["rank_fiscal"] = L.fiscal_mean.rank(ascending=False).astype(int)
L["rank_diff"] = L.diff_pp.rank(ascending=False).astype(int)
print("\n=== offset ladder, Mar+Sep long yen [QE-7+k, QE+k] (k=0 is the pitched window) ===")
print(L.to_string(index=False))

# W56 row: DX QE-9 -> QE, quarters with NO FOMC decision inside the window vs ordinary MEs, 2008+
w = []
for m in me:
    e, x = m - 9, m
    if e < 1 or idx[m].year < 2008:
        continue
    f_strict = bool(((fomc > idx[e]) & (fomc <= idx[x])).any())      # 09-17 definition (after entry close)
    f_incl = bool(((fomc >= idx[e]) & (fomc <= idx[x])).any())       # entry-day decision counted too
    w.append({"qe": idx[m].month in (3, 6, 9, 12), "dx": D[x] / D[e] - 1.0, "f_strict": f_strict, "f_incl": f_incl})
W = pd.DataFrame(w)
out = []
for defn in ["f_strict", "f_incl"]:
    q0 = W[W.qe & ~W[defn]].dx
    om = W[~W.qe].dx
    om0 = W[~W.qe & ~W[defn]].dx
    dd, tt = welch(q0, om)
    dd0, tt0 = welch(q0, om0)
    out.append(cell(q0, f"W56 {defn}: QE no-FOMC DX QE-9->QE 2008+"))
    out.append(cell(om, "   ordinary ME 2008+ (all)"))
    out.append(cell(om0, "   ordinary ME 2008+ no FOMC"))
    print(f"W56 [{defn}] no-FOMC QE minus ordinary ME = {dd:+.3f}pp (t {tt:+.2f}); "
          f"minus no-FOMC ordinary ME = {dd0:+.3f}pp (t {tt0:+.2f})  -> threshold +0.25pp")
show(out, "W56 row (DX-Y.NYB long, QE-9 -> QE)")
