"""kB: book overlap for c3 / c9 / c4 live names.
(1) open or just-signalled ledger positions in the live names;
(2) historical: how often the book was LONG a c3-gated name inside the T-8..T-1
    window (a short would lean against the book), by strategy.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kB_common import *  # noqa
import strategy_config as sc

L = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet",
                    columns=["Strategy", "Ticker", "Direction", "Signal Date", "Entry Date",
                             "Exit Date", "R_Multiple"])
live = ["NKE", "GIS", "PAYX", "CAG", "COST", "AON", "CMS", "LOW", "MCD", "PEG", "PEP",
        "TAP", "VMC", "WHR", "EXC", "PCG", "SRE", "VFC", "DTE", "PPL", "SYK"]
recent = L[L.Ticker.isin(live) & (L["Exit Date"] >= "2026-09-10")]
print("ledger rows in live names exiting on/after 2026-09-10:")
print(recent.to_string(index=False) if len(recent) else "  none")
print("\nledger rows in live names, signal date 2026-09-01+:")
r2 = L[L.Ticker.isin(live) & (L["Signal Date"] >= "2026-09-01")]
print(r2.to_string(index=False) if len(r2) else "  none")

short_strats = [k for k, v in sc.STRATEGY_BOOK.items()] if isinstance(sc.STRATEGY_BOOK, dict) else \
    [s.get("name") for s in sc.STRATEGY_BOOK]
print("\nbook strategies:", short_strats)
print("\nledger direction by strategy:")
print(L.groupby(["Strategy", "Direction"]).size().to_string())

# historical overlap with c3-gated events (LIQ, k=0)
P = build_panel()
C, lodist, r21 = P["C"], P["lodist"], P["r21"]
idx = C.index
E = P["E"]
LIQ_U = [t for t in LIQ if t in C.columns]
Ev = event_positions(E, idx, LIQ_U)
sig = Ev.pT.values - 9
ok = sig >= 252
Ev = Ev[ok]
sig = sig[ok]
ld = np.array([lodist[t].values[s] for t, s in zip(Ev.ticker, sig)])
rr = np.array([r21[t].values[s] for t, s in zip(Ev.ticker, sig)])
G = Ev[(ld <= 0.03) & (rr <= 15)].copy()
G["ent"] = idx[sig[(ld <= 0.03) & (rr <= 15)] + 1]
G["ex"] = idx[np.minimum(G.pT.values - 1, len(idx) - 1)]
hits = []
for _, g in G.iterrows():
    m = L[(L.Ticker == g.ticker) & (L["Entry Date"] <= g.ex) & (L["Exit Date"] >= g.ent)]
    for _, x in m.iterrows():
        hits.append((g.ticker, g.ent.date(), x.Strategy, x.Direction))
print(f"\nc3-gated LIQ events {len(G)}; book positions overlapping the window: {len(hits)}")
if hits:
    print(pd.DataFrame(hits, columns=["ticker", "entry", "strategy", "dir"])
          .groupby(["strategy", "dir"]).size().to_string())
