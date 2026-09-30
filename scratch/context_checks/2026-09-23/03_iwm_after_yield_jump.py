"""02 found IWM weak the session after a 12bp+ 10y jump (t -3.2 over 137 episodes).
Is it real, era-stable, relative to SPY, and does it depend on how IWM closed that day?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

px = close_panel(["^TNX", "SPY", "IWM", "QQQ", "^RUT", "^GSPC"])
tnx = px["^TNX"].dropna()
idx = tnx.index
bp = tnx.diff() * 100
iwm = px["IWM"].reindex(idx)
spy = px["SPY"].reindex(idx)
rut = px["^RUT"].reindex(idx)
iwm_r, spy_r = iwm.pct_change(), spy.pct_change()
hi = tnx >= tnx.rolling(252, min_periods=200).max() - 1e-9
print("today: IWM", round(100 * iwm_r.iloc[-1], 2), "SPY", round(100 * spy_r.iloc[-1], 2), "bp", round(bp.iloc[-1], 1))

f_iwm = fwd_ret(iwm, 1)
f_rel = fwd_ret(iwm, 1) - fwd_ret(spy, 1)
f_iwm5 = fwd_ret(iwm, 5)
f_rut = fwd_ret(rut, 1)

def cell(mask, label, gap=5):
    trig = idx[mask.reindex(idx).fillna(False).values]
    trig = trig[trig < idx[-1]]
    epi = declusters(trig, gap, idx)
    ctl = local_control(idx, epi, 126)
    out = []
    for nm, f in (("IWM h1", f_iwm), ("IWM-SPY h1", f_rel), ("IWM h5", f_iwm5), ("RUT h1", f_rut)):
        e = [d for d in epi if not np.isnan(f.get(d, np.nan))]
        r = summarize(f.loc[e].values, nm)
        dn = int((f.loc[e] < 0).sum())
        r["down"] = f"{dn}/{len(e)}"
        r["sign_p_dn"] = sign_test(dn, len(e))
        r["local"] = 100 * f.loc[ctl].mean()
        r["all"] = 100 * f.mean()
        out.append(r)
    show(out, f"{label}  (raw {len(trig)}, epi {len(epi)})")
    v = f_iwm.loc[epi].dropna()
    for part in era_split(v.index, v.values):
        print("   era IWM h1:", part["label"], part["n"], round(part.get("mean_pct", np.nan), 3), "hit", round(part.get("hit", np.nan), 1), "t", round(part.get("t", np.nan), 2))
    vr = f_rel.loc[epi].dropna()
    for part in era_split(vr.index, vr.values):
        print("   era IWM-SPY h1:", part["label"], part["n"], round(part.get("mean_pct", np.nan), 3), "hit", round(part.get("hit", np.nan), 1), "t", round(part.get("t", np.nan), 2))
    print("   cluster IWM h1:", cluster_note(v.index, v.values))
    by5 = pd.Series(v.values, index=v.index).groupby((v.index.year // 5) * 5).agg(["count", "mean", lambda s: (s < 0).mean()])
    print("   by 5y block (n, mean, share down):\n", by5.round(4).to_string())
    return epi

e_all = cell(bp >= 12, "10y +12bp any level")
e_all_ex08 = cell((bp >= 12) & ~((idx >= "2008-09-01") & (idx <= "2009-06-30")), "10y +12bp, GFC window removed")
cell((bp >= 12) & (iwm_r <= -0.01), "10y +12bp and IWM down 1%+ that day")
cell((bp >= 12) & (iwm_r <= -0.01) & (spy_r > iwm_r + 0.005), "10y +12bp, IWM down 1%+ and trailing SPY by 0.5pp+")
cell((bp >= 12) & (iwm_r > -0.01), "10y +12bp and IWM not down 1%")
cell((bp >= 12) & hi, "10y +12bp at a 52w high")
cell((bp >= 12) & (tnx >= 3.0), "10y +12bp with the 10y at 3%+")
# control: is it just 'IWM after an IWM down day'? same IWM move without the yield jump
cell((iwm_r <= -0.015) & (bp < 5), "IWM down 1.5%+ with 10y NOT up 5bp (control)")
cell((iwm_r <= -0.015) & (bp >= 10), "IWM down 1.5%+ with 10y up 10bp+")
