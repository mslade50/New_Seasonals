"""UUP and the 10-year both closed at 52-week highs today (UUP highest since 2025-01-17,
10y highest since 2007-07-06), with SPY 1.1% under its high. When the dollar and the 10y
break out on the same session, what do stocks, EM, gold and bonds do next? Control: the 10y
at a 52w high WITHOUT the dollar at one."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

px = close_panel(["UUP", "DX-Y.NYB", "^TNX", "SPY", "QQQ", "IWM", "EEM", "GLD", "TLT", "^VIX"])
idx = px["SPY"].dropna().index
px = px.reindex(idx)


def at_high(s):
    return s >= s.rolling(252, min_periods=240).max() - 1e-12


tnx_hi = at_high(px["^TNX"])
uup_hi = at_high(px["UUP"])
dxy_hi = at_high(px["DX-Y.NYB"])
spy_near = px["SPY"] >= px["SPY"].rolling(252, min_periods=240).max() * 0.98
print("today: tnx_hi", bool(tnx_hi.iloc[-1]), "uup_hi", bool(uup_hi.iloc[-1]), "dxy_hi", bool(dxy_hi.iloc[-1]), "spy within 2%", bool(spy_near.iloc[-1]))
valid_uup = px["UUP"].rolling(252, min_periods=240).max().notna()


def report(mask, label, gap=21, subjects=("SPY", "IWM", "EEM", "GLD", "TLT", "UUP")):
    trig = idx[mask.fillna(False).values]
    trig = trig[trig < idx[-1]]
    epi = declusters(trig, gap, idx)
    ctl = local_control(idx, epi, 126)
    print(f"\n##### {label}: raw {len(trig)}, declustered({gap}) {len(epi)}")
    print("   dates:", [str(d.date()) for d in epi])
    rows = []
    for t in subjects:
        for h in (5, 21, 63):
            r = fwd_ret(px[t], h)
            v = r.reindex(epi).dropna()
            if len(v) == 0:
                continue
            row = summarize(v.values, f"{t} h{h}")
            row["rec"] = f"{int((v > 0).sum())}-{int((v < 0).sum())}"
            k = int((v > 0).sum()) if row["hit"] >= 50 else int((v < 0).sum())
            row["sign_p"] = round(sign_test(k, len(v)), 4)
            row["local"] = 100 * r.reindex(ctl).mean()
            rows.append(row)
    tb = (px["^TNX"].shift(-21) - px["^TNX"]) * 100
    v = tb.reindex(epi).dropna()
    print(f"   10y 21d fwd: mean {v.mean():+.1f}bp, up {int((v > 0).sum())}/{len(v)}")
    show(rows, label)
    for t, h in (("SPY", 21), ("EEM", 21), ("GLD", 21)):
        v = fwd_ret(px[t], h).reindex(epi).dropna()
        eras = era_split(v.index, v.values)
        print(f"   era {t} h{h}:", [(e['label'], e['n'], round(e.get('mean_pct', np.nan), 2), round(e.get('hit', np.nan), 1)) for e in eras])
        print(f"   cluster {t} h{h}:", cluster_note(v.index, v.values))
    return epi


both = tnx_hi & uup_hi
e_both = report(both, "10y AND UUP at 52w closing highs, same session")
e_tnx_only = report(tnx_hi & ~uup_hi & valid_uup, "CONTROL: 10y at 52w high, UUP not")
e_all3 = report(both & spy_near, "10y AND UUP at highs, SPY within 2% of its high")
e_dxy = report(tnx_hi & dxy_hi, "10y AND DX-Y.NYB at highs (1999+, stamping caveat)")

# per-episode table for the headline cell
rows = []
for d in e_both:
    p = idx.get_loc(d)
    row = {"date": d.date(), "tnx": round(px["^TNX"].iloc[p], 2),
           "spy_vs_hi": round(100 * (px["SPY"].iloc[p] / px["SPY"].iloc[max(0, p - 251):p + 1].max() - 1), 1)}
    for t in ("SPY", "EEM", "GLD", "UUP"):
        r = fwd_ret(px[t], 21)
        row[f"{t}_21"] = round(100 * r.iloc[p], 2) if not np.isnan(r.iloc[p]) else None
    rows.append(row)
print("\n", pd.DataFrame(rows).to_string(index=False))
