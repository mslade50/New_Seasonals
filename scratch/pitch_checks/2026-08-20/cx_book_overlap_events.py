"""Round-1 leftovers for C5 / C6 / C9: book overlap and the two volatility
events that sit INSIDE today's 10 td hold (opex 08-21 at +1, Jackson Hole
08-28 at +6).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

# ------------------------------------------------------------ book overlap
print("######## BOOK OVERLAP ########")
try:
    import strategy_config as sc
    for t in ["KWEB", "FXI", "EEM", "TIP", "IEF", "UUP", "GLD"]:
        liq = t in getattr(sc, "LIQUID_PLUS_COMMODITIES", [])
        csv = t in getattr(sc, "CSV_UNIVERSE", [])
        print(f"  {t:<6} liquid={liq}  full_universe={csv}")
except Exception as e:                                   # noqa: BLE001
    print("  strategy_config unavailable:", e)

led = ROOT / "data" / "backtest_trades_full.parquet"
if led.exists():
    tr = pd.read_parquet(led)
    col = "Ticker" if "Ticker" in tr.columns else tr.columns[0]
    for t in ["KWEB", "FXI", "EEM", "TIP", "IEF", "UUP", "GLD"]:
        sub = tr[tr[col] == t]
        print(f"  ledger trades in {t:<6}: {len(sub)}"
              + (f"  strategies {sorted(sub['Strategy_Name'].unique())}"
                 if len(sub) and "Strategy_Name" in sub.columns else ""))
else:
    print("  no ledger parquet at", led)

# --------------------------------------------------- events inside the hold
print("\n\n######## opex / jackson_hole INSIDE THE HOLD ########")
px = close_panel(["DX-Y.NYB", "KWEB", "TIP", "IEF", "GLD"])
d = px.index
dx_r = pct_rank(px["DX-Y.NYB"], 21)
gl_r = pct_rank(px["GLD"], 21)
m5 = (dx_r <= 2).reindex(d).fillna(False)
m9 = ((dx_r <= 5) & (gl_r >= 85)).reindex(d).fillna(False)

specs = [("C5 KWEB long", m5, [("KWEB", 1.0)], 10),
         ("C5 KWEB long", m5, [("KWEB", 1.0)], 5),
         ("C6 DXY long", (dx_r <= 2).reindex(d).fillna(False),
          [("DX-Y.NYB", 1.0)], 10),
         ("C9 TIP/IEF dur-neutral", m9, [("TIP", 1.0), ("IEF", -0.698)], 10)]
for lbl, m, legs, h in specs:
    r = vehicle_ret(px, legs, h)
    e = declusters(pd.DatetimeIndex(
        [x for x in d[m.values] if not np.isnan(r.get(x, np.nan))]), 21, d)
    v = r.reindex(e).dropna()
    for kinds in [("opex",), ("jackson_hole",)]:
        fl = event_in_window(v.index, d, h, 1, kinds)
        a, b = v.values[fl], v.values[~fl]
        if len(a) < 2 or len(b) < 2:
            print(f"  {lbl} h={h} {kinds[0]:<13} too few on one side "
                  f"(IN={len(a)} OUT={len(b)})")
            continue
        print(f"  {lbl} h={h} {kinds[0]:<13} IN N={len(a)} "
              f"{100*a.mean():+.3f}% hit {100*(a>0).mean():.0f}%  |  "
              f"OUT N={len(b)} {100*b.mean():+.3f}% hit {100*(b>0).mean():.0f}%")
