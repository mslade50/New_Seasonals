"""MOVE +21.5% in one session (78.56 -> 95.45). How rare, and what followed:
MOVE itself, TLT, the 10y, SPY, VIX. Sunday told the 21-day MOVE climb in a calm VIX;
tonight is the single-session jump."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

px = close_panel(["^MOVE", "^TNX", "TLT", "SPY", "IWM", "^VIX", "^GSPC"])
mv = px["^MOVE"].dropna()
idx = mv.index
r = mv.pct_change()
print("MOVE history from", idx[0].date(), "n", len(idx), "today", round(100 * r.iloc[-1], 2), "level", mv.iloc[-1])
rank = (r.abs() >= abs(r.iloc[-1])).sum()
print("sessions with |ret| >= today's:", rank)
top = r.sort_values(ascending=False).head(25)
print("largest one-day MOVE rises:\n", [(str(d.date()), round(100 * v, 1), round(mv.loc[d], 1)) for d, v in top.items()])
prev = r.loc[: idx[-2]]
ge = prev[prev >= r.iloc[-1] - 1e-9]
print("previous day >= today:", ge.index[-1].date() if len(ge) else None)
vix = px["^VIX"].reindex(idx)
print("today VIX", vix.iloc[-1], "VIX ret", round(100 * vix.pct_change().iloc[-1], 2))

tnx = px["^TNX"].reindex(idx)
fwd_tnx = lambda h: (tnx.shift(-h) - tnx) * 100

def cell(mask, label, gap=5):
    trig = idx[mask.reindex(idx).fillna(False).values]
    trig = trig[trig < idx[-1]]
    epi = declusters(trig, gap, idx)
    ctl = local_control(idx, epi, 126)
    print(f"\n##### {label}: raw {len(trig)}, epi {len(epi)}")
    print("   ", [(str(d.date()), round(100 * r.loc[d], 1), round(mv.loc[d], 0)) for d in epi])
    rows = []
    for nm in ("^MOVE", "TLT", "SPY", "IWM", "^VIX"):
        s = px[nm].reindex(idx)
        for h in (1, 5, 21):
            f = fwd_ret(s, h)
            e = [d for d in epi if not np.isnan(f.get(d, np.nan))]
            row = summarize(f.loc[e].values, f"{nm} h{h}")
            dn = int((f.loc[e] < 0).sum())
            row["down"] = f"{dn}/{len(e)}"
            row["p_dn"] = sign_test(dn, len(e))
            row["p_up"] = sign_test(len(e) - dn, len(e))
            row["local"] = 100 * f.loc[ctl].mean()
            rows.append(row)
    show(rows, label)
    for h in (1, 5, 21):
        f = fwd_tnx(h).loc[epi].dropna()
        print(f"   TNX h{h}: mean {f.mean():+.1f}bp median {f.median():+.1f} up {(f > 0).sum()}/{len(f)}")
    for nm, h in (("^MOVE", 5), ("^MOVE", 21), ("SPY", 5)):
        f = fwd_ret(px[nm].reindex(idx), h).loc[epi].dropna()
        for part in era_split(f.index, f.values):
            print(f"   era {nm} h{h}:", part["label"], part["n"], round(part.get("mean_pct", np.nan), 2), "hit", round(part.get("hit", np.nan), 1))
        print(f"   cluster {nm} h{h}:", cluster_note(f.index, f.values))
    return epi

cell(r >= 0.15, "MOVE +15% or more in a session")
cell(r >= 0.12, "MOVE +12% or more in a session")
cell((r >= 0.12) & (vix < 20), "MOVE +12%+ with VIX under 20")
cell((r >= 0.12) & (vix.pct_change() < 0.10), "MOVE +12%+ with VIX up less than 10%")
