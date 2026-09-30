"""C5 round 1: long USDJPY (JPY=X, yen per dollar) after the Japanese fiscal
half-year / year-end book close (March and September quarter-ends).

Entries: QE-1 close (today's live form, 09-29) and QE close (tomorrow's run).
Exits: QE+3, QE+4, QE+5. Long USDJPY = JPY=X[x] / JPY=X[e] - 1.
Groups: September, March, Mar+Sep vs Jun+Dec vs ordinary month-ends.
Parent to beat: DX QE -> QE+5 over ALL quarters (+0.228%, 62-44, registry).
Mechanism check in its own window: repatriation INTO the close predicts yen
strength (USDJPY down) from QE-7 to QE in Mar/Sep.
NYSE calendar, FX forward-filled across NYSE holidays (the 09-21 convention).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kB_common import *  # noqa

px = nyse_panel(["JPY=X", "DX-Y.NYB"], ffill=("JPY=X", "DX-Y.NYB"))
idx = px.index
J, D = px["JPY=X"].values, px["DX-Y.NYB"].values
me = month_end_positions(idx)
nfp = load_events(["nfp"])["date"]
print(f"LIVE {idx[-1].date()}: USDJPY {J[-1]:.2f}; 5d {100*(J[-1]/J[-6]-1):+.2f}%  21d {100*(J[-1]/J[-22]-1):+.2f}%;"
      f" DX 21d {100*(D[-1]/D[-22]-1):+.2f}%;  last completed ME {idx[me[-1]].date()}")

rows = []
for m in me:
    if m - 7 < 0 or m + 5 >= len(idx):
        continue
    d = idx[m]
    r = {"me": d, "year": d.year, "month": d.month, "qe": d.month in (3, 6, 9, 12), "fiscal": d.month in (3, 9)}
    for e_off in (-1, 0):
        for x_off in (3, 4, 5):
            r[f"J{e_off}_{x_off}"] = span_ret(J, m + e_off, m + x_off)
            r[f"D{e_off}_{x_off}"] = span_ret(D, m + e_off, m + x_off)
    r["J_pre7"] = span_ret(J, m - 7, m)      # into the close
    r["D_pre7"] = span_ret(D, m - 7, m)
    r["J_pre1"] = span_ret(J, m - 1, m)      # the QE session itself
    r["nfp_in"] = bool(((nfp > idx[m - 1]) & (nfp <= idx[m + 5])).any())
    rows.append(r)
A = pd.DataFrame(rows)
groups = [("September", A.month == 9), ("March", A.month == 3), ("Mar+Sep", A.fiscal),
          ("Jun+Dec", A.qe & ~A.fiscal), ("all QE", A.qe), ("ordinary ME", ~A.qe)]

out = []
for col in ("J-1_4", "J-1_5", "J0_3", "J0_4", "J0_5"):
    for lbl, m in groups:
        out.append(cell(A.loc[m, col], f"USDJPY {col} {lbl}"))
show(out, "long USDJPY by entry (QE-1 / QE) and exit (QE+3..+5); col = J<entry>_<exit>")

# all-days drift at the same horizons
jd = pd.Series(J, index=idx)
show([cell(fwd_ret(jd, h).dropna().values, f"USDJPY all days h={h}") for h in (3, 4, 5, 6)], "CTRL: USDJPY own drift")

# parent: DX QE -> QE+5 all quarters, and DX in the same fiscal months
show([cell(A.loc[A.qe, "D0_5"], "PARENT DX QE->QE+5 all quarters"),
      cell(A.loc[A.fiscal, "D0_5"], "DX QE->QE+5 Mar+Sep"),
      cell(A.loc[A.qe & ~A.fiscal, "D0_5"], "DX QE->QE+5 Jun+Dec"),
      cell(A.loc[A.month == 9, "D0_5"], "DX QE->QE+5 September"),
      cell(A.loc[A.fiscal, "D-1_5"], "DX QE-1->QE+5 Mar+Sep"),
      cell(A.loc[A.qe, "J0_5"], "USDJPY QE->QE+5 all quarters")],
     "parent comparison")
for col in ("J0_5", "J-1_5", "J-1_4"):
    d1, t1 = welch(A.loc[A.fiscal, col], A.loc[A.qe & ~A.fiscal, col])
    d2, t2 = welch(A.loc[A.fiscal, col], A.loc[~A.qe, col])
    d3, t3 = welch(A.loc[A.fiscal, col], A.loc[A.qe, "D0_5"])
    print(f"  {col}: Mar+Sep minus Jun+Dec {d1:+.3f}pp (t {t1:+.2f}); minus ordinary ME {d2:+.3f}pp (t {t2:+.2f}); "
          f"minus DX-all-QE parent {d3:+.3f}pp (t {t3:+.2f})")
# yen-specific residual: USDJPY minus DX, same window
A["res0_5"] = A["J0_5"] - A["D0_5"]
A["res-1_5"] = A["J-1_5"] - A["D-1_5"]
show([cell(A.loc[m, c], f"USDJPY-DX {c} {lbl}") for c in ("res0_5", "res-1_5")
      for lbl, m in groups[:4] + [groups[5]]], "yen-specific residual (USDJPY minus DX)")

# mechanism in its own window: repatriation INTO the close => USDJPY DOWN from QE-7 to QE
show([cell(A.loc[m, "J_pre7"], f"USDJPY QE-7->QE {lbl}") for lbl, m in groups] +
     [cell(A.loc[A.fiscal, "J_pre1"], "USDJPY QE-1->QE (the close day) Mar+Sep")],
     "mechanism check: into the close (repatriation predicts < 0)")

# era split, fiscal
for col in ("J-1_5", "J0_5"):
    f = A[A.fiscal]
    show([cell(f.loc[f.year < 2018, col], f"Mar+Sep {col} pre-2018"), cell(f.loc[f.year >= 2018, col], f"Mar+Sep {col} 2018+"),
          cell(f.loc[(f.year >= 2013) & (f.year < 2018), col], f"Mar+Sep {col} 2013-2017"),
          cell(A.loc[(A.month == 9) & (A.year < 2018), col], f"Sep {col} pre-2018"),
          cell(A.loc[(A.month == 9) & (A.year >= 2018), col], f"Sep {col} 2018+"),
          cell(A.loc[(~A.qe) & (A.year >= 2018), col], f"ordinary ME {col} 2018+")], f"era split {col}")

f = A[A.fiscal].copy()
print("\nconcentration Mar+Sep J-1_5:", cluster_note(pd.DatetimeIndex(f.me), f["J-1_5"].values))
print("concentration Mar+Sep J0_5:", cluster_note(pd.DatetimeIndex(f.me), f["J0_5"].values))
print(f"worst Mar+Sep J-1_5 {100*f['J-1_5'].min():.2f}% on {f.loc[f['J-1_5'].idxmin(), 'me'].date()}")
print("\nSeptember by year (QE-1->QE+5 %, QE->QE+5 %, QE-7->QE %):")
s = A[A.month == 9]
print(" ".join(f"{r.year}:{100*r['J-1_5']:+.2f}/{100*r['J0_5']:+.2f}/{100*r['J_pre7']:+.2f}" for _, r in s.iterrows()))

# NFP inside the hold
show([cell(A.loc[A.fiscal & A.nfp_in, "J-1_5"], "Mar+Sep J-1_5 NFP inside"),
      cell(A.loc[A.fiscal & ~A.nfp_in, "J-1_5"], "Mar+Sep J-1_5 NFP outside")], "NFP in hold (QE-1 -> QE+5)")
print(f"\ncost ~2 bp round trip (6J futures). Mar+Sep QE-1->QE+5 mean {1e4*A.loc[A.fiscal, 'J-1_5'].mean():.1f} bp; "
      f"over ordinary ME {1e4*(A.loc[A.fiscal, 'J-1_5'].mean()-A.loc[~A.qe, 'J-1_5'].mean()):.1f} bp")
