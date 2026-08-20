"""C1 round 1: short SPY on the index-quiet / components-wild state.

Reproduces the recon mask from scratch, states TODAY's reading against the
gate before anything else, then runs the battery on the SHORT vehicle and the
two gate-attribution legs (dispersion alone / quiet index alone).
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

pd.set_option("display.width", 220)

TAPE = json.load(open(ROOT / "data/pitch_tape.json"))["tickers"]
UNIV = [t for t in TAPE if not t.startswith("^") and "=" not in t and "-" not in t]
print(f"tape names used for the cross-section: {len(UNIV)}")

pan = close_panel(UNIV)
rets = pan.pct_change()
nok = rets.notna().sum(axis=1)
cs = rets.std(axis=1)[nok >= 120]          # cross-sectional sd of daily returns
csr = cs.rolling(252).apply(lambda a: (a[:-1] < a[-1]).mean(), raw=True)

px = close_panel(["SPY", "^VIX"])
d = px.index
spy1 = px["SPY"].pct_change().reindex(cs.index)

print("\n===== 0. TODAY, before any statistic =====")
print(f"  cs sd last bar {cs.index[-1].date()}: {cs.iloc[-1]*100:.3f}%")
print(f"  trailing-252 pctile of cs sd : {csr.iloc[-1]*100:.1f}   (gate needs >= 90)")
print(f"  FULL-history pctile of cs sd : {(cs < cs.iloc[-1]).mean()*100:.1f}")
print(f"  SPY 1d: {spy1.iloc[-1]*100:+.3f}%   (gate needs |ret| <= 0.5%)")
print(f"  -> gate fires today? {bool((csr.iloc[-1] >= 0.90) and abs(spy1.iloc[-1]) <= 0.005)}")
print(f"  n names with data today: {int(nok.iloc[-1])}")

# masks
m_join = ((csr >= 0.90) & (spy1.abs() <= 0.005)).reindex(d).fillna(False)
m_disp = (csr >= 0.90).reindex(d).fillna(False)
m_quiet = (spy1.abs() <= 0.005).reindex(d).fillna(False)

for nm, m in [("JOINT", m_join), ("dispersion alone", m_disp), ("quiet index alone", m_quiet)]:
    e = declusters(d[m], 5, d)
    print(f"  {nm:<20} days={int(m.sum()):>5}  episodes(gap5)={len(e):>4}")

SHORT = [("SPY", -1.0)]
variants = {
    "csr>=0.95 & |spy|<=0.5%": ((csr >= 0.95) & (spy1.abs() <= 0.005)).reindex(d).fillna(False),
    "csr>=0.85 & |spy|<=0.5%": ((csr >= 0.85) & (spy1.abs() <= 0.005)).reindex(d).fillna(False),
    "csr>=0.90 & |spy|<=0.3%": ((csr >= 0.90) & (spy1.abs() <= 0.003)).reindex(d).fillna(False),
    "csr>=0.90 & |spy|<=1.0%": ((csr >= 0.90) & (spy1.abs() <= 0.010)).reindex(d).fillna(False),
    "dispersion ALONE": m_disp,
    "quiet index ALONE": m_quiet,
}

for h in (3, 5, 10):
    battery(px, m_join, SHORT, h, f"C1 SHORT SPY, joint cell", cost_bps=2.0,
            variants=variants if h == 3 else None, min_gap=5)

# ------------------------------------------------------------------ gate attribution
print("\n\n===== GATE ATTRIBUTION (short SPY, episode level, min_gap 5) =====")
rows = []
for nm, m in [("JOINT disp>=90 & quiet", m_join),
              ("dispersion ALONE >=90", m_disp),
              ("quiet index ALONE", m_quiet),
              ("dispersion>=90 & NOT quiet", (m_disp & ~m_quiet)),
              ("quiet & dispersion<90", (m_quiet & ~m_disp))]:
    for h in (1, 3, 5, 10):
        ret = vehicle_ret(px, SHORT, h)
        valid = ret.dropna().index
        t = d[m.values] if hasattr(m, "values") else d[m]
        t = pd.DatetimeIndex(t).intersection(valid)
        e = declusters(t, 5, valid)
        r = summarize(ret.loc[e].values, f"{nm} h={h}")
        base = ret.loc[valid]
        r["ctrl_all"] = round(100 * base.mean(), 3)
        r["edge_pp"] = round(r.get("mean_pct", np.nan) - 100 * base.mean(), 3)
        rows.append(r)
show(rows, "gate attribution")
