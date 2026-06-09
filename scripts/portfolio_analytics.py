"""
portfolio_analytics.py — portfolio-level figures + stats from the trade ledger.

Reads:
  data/backtest_trades_full.parquet   (one row per executed trade)
  data/backtest_daily_pnl.parquet     (daily MTM, compounded + flat bases)

Writes to reports/portfolio/:
  portfolio_stats.md         human-readable headline + risk stats (both bases)
  per_strategy.csv           per (Strategy, Tier) trade + R + $ contribution
  per_year.csv               yearly R / trades / win% / tier split
  fig_equity_drawdown.png    equity curves (comp log + flat) + underwater DD
  fig_monthly_heatmap.png    flat monthly returns (% of $750k) year x month
  fig_yearly_R_by_tier.png   stacked yearly total R, Liquid vs Overflow
  fig_strategy_R.png         total R by strategy (Liquid vs Overflow split)
  fig_R_distribution.png     R-multiple histogram
  fig_cum_R_by_strategy.png  cumulative R over time per strategy
  fig_rolling_sharpe.png     rolling 1y Sharpe (compounded)
  fig_concurrency.png        open positions over time (gross + by direction)

Dollar guidance: PnL_flat_750k sizes every trade off a fixed $750k (era-
comparable); PnL_compounded follows the growing-equity path. CAGR uses
compounded; R / Return_Pct are sizing-invariant.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
TRADES = os.path.join(_ROOT, "data", "backtest_trades_full.parquet")
DAILY = os.path.join(_ROOT, "data", "backtest_daily_pnl.parquet")
OUT = os.path.join(_ROOT, "reports", "portfolio")
START_EQ = 750000.0
TRADING_DAYS = 252

C_LIQ, C_OF = "#2c7fb8", "#de8a5a"   # liquid / overflow colors


def _curve_stats(pnl, equity, compounded):
    eq = equity.astype(float)
    if compounded:
        ret = pnl / eq.shift(1).fillna(START_EQ)
    else:
        ret = pnl / START_EQ
    ret = ret.replace([np.inf, -np.inf], 0).fillna(0)
    n = len(eq)
    years = n / TRADING_DAYS if n else 0
    total_ret = eq.iloc[-1] / START_EQ - 1.0
    cagr = (eq.iloc[-1] / START_EQ) ** (1 / years) - 1 if (years > 0 and eq.iloc[-1] > 0) else np.nan
    ann_vol = ret.std() * np.sqrt(TRADING_DAYS)
    sharpe = (ret.mean() * TRADING_DAYS) / ann_vol if ann_vol > 0 else np.nan
    dn = ret[ret < 0]
    dvol = dn.std() * np.sqrt(TRADING_DAYS) if len(dn) else np.nan
    sortino = (ret.mean() * TRADING_DAYS) / dvol if dvol and dvol > 0 else np.nan
    dd = (eq - eq.cummax()) / eq.cummax()
    maxdd = dd.min()
    # longest underwater stretch (days below prior peak)
    underwater = (eq < eq.cummax()).astype(int)
    longest = cur = 0
    for v in underwater.values:
        cur = cur + 1 if v else 0
        longest = max(longest, cur)
    ann_ret = ret.mean() * TRADING_DAYS
    return {
        "Years": round(years, 1),
        "Total return": f"{total_ret*100:,.0f}%",
        "Ann return": f"{ann_ret*100:.1f}%",
        "CAGR": (f"{cagr*100:.1f}%" if pd.notna(cagr) else "n/a"),
        "Ann vol": f"{ann_vol*100:.1f}%",
        "Sharpe": round(sharpe, 2) if pd.notna(sharpe) else "n/a",
        "Sortino": round(sortino, 2) if pd.notna(sortino) else "n/a",
        "Max DD": f"{maxdd*100:.1f}%",
        "Longest DD (days)": int(longest),
        "% up days": f"{(ret > 0).mean()*100:.1f}%",
        "Final equity": f"${eq.iloc[-1]:,.0f}",
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    tr = pd.read_parquet(TRADES)
    for c in ["Signal Date", "Entry Date", "Exit Date"]:
        tr[c] = pd.to_datetime(tr[c])
    daily = pd.read_parquet(DAILY)
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.set_index("date").sort_index()

    tr["year"] = tr["Signal Date"].dt.year
    R = tr["R_Multiple"]

    # ------------------------------------------------------------------ stats
    comp = _curve_stats(daily["pnl_compounded"], daily["equity_compounded"], True)
    flat = _curve_stats(daily["pnl_flat"], daily["equity_flat"], False)

    def _pf(s):
        g = s[s > 0].sum(); l = -s[s < 0].sum()
        return (g / l) if l > 0 else float("inf")

    lines = []
    lines.append("# Portfolio Analytics — full strategy book\n")
    lines.append(f"_Source: `data/backtest_trades_full.parquet` "
                 f"({len(tr):,} trades, {tr['Ticker'].nunique()} tickers, "
                 f"{tr['Signal Date'].min().date()} -> {tr['Signal Date'].max().date()})_\n")
    lines.append("## Trade-level (sizing-invariant)\n")
    lines.append(f"- Trades: **{len(tr):,}**  |  Win rate: **{(tr['PnL_compounded']>0).mean()*100:.1f}%**  "
                 f"|  Total R: **{R.sum():.0f}**  |  Avg R: **{R.mean():.3f}**  |  PF (R): **{_pf(R):.2f}**")
    lines.append(f"- Long: {int((tr['Direction']=='Long').sum())} trades, "
                 f"{R[tr['Direction']=='Long'].sum():.0f}R  |  "
                 f"Short: {int((tr['Direction']=='Short').sum())} trades, "
                 f"{R[tr['Direction']=='Short'].sum():.0f}R")
    lines.append(f"- Liquid: {int((tr['Tier']=='Liquid').sum())} trades, {R[tr['Tier']=='Liquid'].sum():.0f}R  |  "
                 f"Overflow: {int((tr['Tier']=='Overflow').sum())} trades, {R[tr['Tier']=='Overflow'].sum():.0f}R\n")
    lines.append("## Portfolio curve — two sizing bases\n")
    keys = list(comp.keys())
    lines.append("| Metric | Compounded (from $750k) | Flat ($750k/trade) |")
    lines.append("|---|--:|--:|")
    for k in keys:
        lines.append(f"| {k} | {comp[k]} | {flat[k]} |")
    lines.append("\n_Compounded = realistic growing-equity path. Flat = constant-risk "
                 "(stationary), better for comparing eras and judging edge. CAGR is only "
                 "meaningful on the compounded basis._\n")

    # per-strategy / per-tier
    rows = []
    for (s, t), g in tr.groupby(["Strategy", "Tier"]):
        gr = g["R_Multiple"]
        rows.append({
            "Strategy": s, "Tier": t, "Trades": len(g),
            "Win%": round((g["PnL_compounded"] > 0).mean() * 100, 1),
            "Tot_R": round(gr.sum(), 1), "Avg_R": round(gr.mean(), 3),
            "PF_R": round(_pf(gr), 2),
            "PnL_flat_750k": round(g["PnL_flat_750k"].sum()),
            "Pct_of_total_R": round(gr.sum() / R.sum() * 100, 1),
        })
    per_strat = pd.DataFrame(rows).sort_values("Tot_R", ascending=False)
    per_strat.to_csv(os.path.join(OUT, "per_strategy.csv"), index=False)

    yr_rows = []
    for y, g in tr.groupby("year"):
        gr = g["R_Multiple"]
        yr_rows.append({
            "Year": y, "Trades": len(g), "Win%": round((g["PnL_compounded"] > 0).mean() * 100, 1),
            "Tot_R": round(gr.sum(), 1),
            "Liquid_R": round(gr[g["Tier"] == "Liquid"].sum(), 1),
            "Overflow_R": round(gr[g["Tier"] == "Overflow"].sum(), 1),
        })
    per_year = pd.DataFrame(yr_rows)
    per_year.to_csv(os.path.join(OUT, "per_year.csv"), index=False)

    lines.append("## Per-strategy (sorted by total R)\n")
    lines.append("| Strategy | Tier | Trades | Win% | Tot R | Avg R | PF | PnL_flat | %R |")
    lines.append("|---|---|--:|--:|--:|--:|--:|--:|--:|")
    for _, r in per_strat.iterrows():
        lines.append(f"| {r.Strategy} | {r.Tier} | {int(r.Trades)} | {r['Win%']} | "
                     f"{r.Tot_R} | {r.Avg_R} | {r.PF_R} | ${int(r.PnL_flat_750k):,} | {r.Pct_of_total_R}% |")
    with open(os.path.join(OUT, "portfolio_stats.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # ----------------------------------------------------------------- figures
    plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.25, "axes.axisbelow": True})

    # 1. equity + drawdown
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), height_ratios=[2.2, 1], sharex=True)
    ax1.plot(daily.index, daily["equity_compounded"], color="#1a9850", lw=1.3, label="Compounded (log)")
    ax1.set_yscale("log"); ax1.set_ylabel("Compounded equity (log)", color="#1a9850")
    axb = ax1.twinx()
    axb.plot(daily.index, daily["equity_flat"], color="#762a83", lw=1.1, alpha=0.8, label="Flat $750k/trade")
    axb.set_ylabel("Flat equity ($)", color="#762a83"); axb.grid(False)
    ax1.set_title("Portfolio equity — compounded (log) vs flat constant-risk")
    l1, lb1 = ax1.get_legend_handles_labels(); l2, lb2 = axb.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc="upper left", fontsize=8)
    eqf = daily["equity_flat"]; ddf = (eqf - eqf.cummax()) / eqf.cummax() * 100
    ax2.fill_between(daily.index, ddf, 0, color="#d73027", alpha=0.6)
    ax2.set_ylabel("Drawdown % (flat)"); ax2.set_title("Underwater (flat basis)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_equity_drawdown.png")); plt.close(fig)

    # 2. monthly heatmap (flat % of 750k)
    m = (daily["pnl_flat"].resample("ME").sum() / START_EQ * 100)
    mt = pd.DataFrame({"y": m.index.year, "mo": m.index.month, "v": m.values})
    piv = mt.pivot(index="y", columns="mo", values="v")
    fig, ax = plt.subplots(figsize=(11, max(4, 0.32 * len(piv))))
    vmax = np.nanmax(np.abs(piv.values))
    im = ax.imshow(piv.values, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(12)); ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
    ax.set_yticks(range(len(piv))); ax.set_yticklabels(piv.index)
    for i in range(len(piv)):
        for j in range(12):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6.5,
                        color="black")
    ax.set_title("Monthly P&L (% of $750k, flat basis)")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="% of $750k")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_monthly_heatmap.png")); plt.close(fig)

    # 3. yearly R by tier (stacked)
    yp = per_year.set_index("Year")[["Liquid_R", "Overflow_R"]]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(yp.index, yp["Liquid_R"], color=C_LIQ, label="Liquid")
    ax.bar(yp.index, yp["Overflow_R"], bottom=yp["Liquid_R"], color=C_OF, label="Overflow")
    ax.axhline(0, color="k", lw=0.8); ax.set_ylabel("Total R"); ax.legend()
    ax.set_title("Yearly total R by tier")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_yearly_R_by_tier.png")); plt.close(fig)

    # 4. strategy R (split L/O)
    pst = per_strat.copy()
    pst["label"] = pst["Strategy"] + " [" + pst["Tier"].str[0] + "]"
    pst = pst.sort_values("Tot_R")
    fig, ax = plt.subplots(figsize=(10, max(4, 0.42 * len(pst))))
    ax.barh(pst["label"], pst["Tot_R"],
            color=[C_OF if t == "Overflow" else C_LIQ for t in pst["Tier"]])
    ax.axvline(0, color="k", lw=0.8); ax.set_xlabel("Total R")
    ax.set_title("Total R by strategy x tier  (blue=Liquid, orange=Overflow)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_strategy_R.png")); plt.close(fig)

    # 5. R distribution
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(R.clip(-3, 5), bins=60, color="#4575b4", alpha=0.85)
    ax.axvline(0, color="k", lw=0.8); ax.axvline(R.mean(), color="#d73027", ls="--", label=f"mean {R.mean():.2f}R")
    ax.set_xlabel("R multiple (clipped [-3,5])"); ax.set_ylabel("trades"); ax.legend()
    ax.set_title("Per-trade R distribution")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_R_distribution.png")); plt.close(fig)

    # 6. cumulative R by strategy over time (by exit date)
    fig, ax = plt.subplots(figsize=(12, 6))
    tr_sorted = tr.sort_values("Exit Date")
    for s in tr.groupby("Strategy")["R_Multiple"].sum().sort_values(ascending=False).index:
        g = tr_sorted[tr_sorted["Strategy"] == s]
        ax.plot(g["Exit Date"], g["R_Multiple"].cumsum(), lw=1.2, label=s)
    ax.set_ylabel("Cumulative R"); ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.set_title("Cumulative R by strategy (by exit date)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_cum_R_by_strategy.png")); plt.close(fig)

    # 7. rolling 1y Sharpe (compounded)
    retc = (daily["pnl_compounded"] / daily["equity_compounded"].shift(1).fillna(START_EQ)).fillna(0)
    roll = retc.rolling(TRADING_DAYS)
    rsharpe = (roll.mean() * TRADING_DAYS) / (roll.std() * np.sqrt(TRADING_DAYS))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily.index, rsharpe, color="#0571b0", lw=1)
    ax.axhline(0, color="k", lw=0.8); ax.axhline(1, color="#1a9850", ls="--", lw=0.8, label="Sharpe=1")
    ax.set_ylabel("Rolling 1y Sharpe"); ax.legend()
    ax.set_title("Rolling 252-day Sharpe (compounded)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_rolling_sharpe.png")); plt.close(fig)

    # 8. concurrency: open positions over time, gross + by direction
    bdays = pd.date_range(tr["Entry Date"].min(), tr["Exit Date"].max(), freq="B")
    def _open_count(sub):
        delta = pd.Series(0, index=bdays, dtype=float)
        ent = sub["Entry Date"].dt.normalize().value_counts()
        ext = (sub["Exit Date"].dt.normalize() + pd.offsets.BDay(1)).value_counts()
        for d, c in ent.items():
            if d in delta.index: delta[d] += c
        for d, c in ext.items():
            if d in delta.index: delta[d] -= c
        return delta.cumsum()
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(bdays, _open_count(tr), color="k", lw=0.9, label="All open")
    ax.plot(bdays, _open_count(tr[tr["Direction"] == "Long"]), color=C_LIQ, lw=0.8, alpha=0.8, label="Long")
    ax.plot(bdays, _open_count(tr[tr["Direction"] == "Short"]), color="#d73027", lw=0.8, alpha=0.8, label="Short")
    ax.set_ylabel("Open positions"); ax.legend()
    ax.set_title("Concurrent open positions over time")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_concurrency.png")); plt.close(fig)

    # ---- console summary ----
    print("=" * 70)
    print("PORTFOLIO ANALYTICS")
    print("=" * 70)
    print(f"  {len(tr):,} trades | total {R.sum():.0f}R | avg {R.mean():.3f}R | "
          f"win {(tr['PnL_compounded']>0).mean()*100:.1f}% | PF(R) {_pf(R):.2f}")
    print(f"  Compounded: CAGR {comp['CAGR']}, Sharpe {comp['Sharpe']}, MaxDD {comp['Max DD']}, "
          f"final {comp['Final equity']}")
    print(f"  Flat:       Sharpe {flat['Sharpe']}, MaxDD {flat['Max DD']}, "
          f"total {flat['Total return']}")
    print(f"\n  Wrote stats + 8 figures -> {OUT}")
    print("Done.")


if __name__ == "__main__":
    main()
