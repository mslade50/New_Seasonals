"""
backtester_html_report.py — render the strat_backtester page's outputs as a
static, interactive HTML file, driven off the persistent ledger (no re-run).

Reuses the PAGE'S OWN helper functions so the views match the Streamlit page.

Enhancements over the page:
  - Starting capital scaled to $10,000 (fractional-share basis). Compounded
    sizing is scale-invariant, so % / Sharpe / Max-DD / R are identical to any
    starting capital; only dollar labels differ (scaled by 10k/750k).
  - Equity MTM and strategy charts shown for FULL history AND trailing 12 months.
  - Strategy charts plot CUMULATIVE R (sizing-invariant) instead of PnL.
  - Tables are interactive (DataTables: click-to-sort, type-to-filter, paging).

Output: reports/portfolio/backtester_view.html
"""
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import data_provider
from strategy_config import ACCOUNT_VALUE, SPOT_TO_TRADEABLE
from pages.strat_backtester import (
    get_daily_mtm_series,
    calculate_annual_stats,
    calculate_performance_stats,
    calculate_mark_to_market_curve,
    build_strategy_correlation_matrix,
    calculate_daily_exposure,
)

LEDGER = os.path.join(_ROOT, "data", "backtest_trades_full.parquet")
OUT_HTML = os.path.join(_ROOT, "reports", "portfolio", "backtester_view.html")
PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.27.0.min.js"
JQUERY_CDN = "https://code.jquery.com/jquery-3.7.1.min.js"
DT_CSS = "https://cdn.datatables.net/1.13.8/css/jquery.dataTables.min.css"
DT_JS = "https://cdn.datatables.net/1.13.8/js/jquery.dataTables.min.js"
MIN_SAMPLE = 200
DARK = "plotly_dark"
START_EQ = 10000.0                     # display starting capital
SCALE = START_EQ / float(ACCOUNT_VALUE)  # 10k / 750k


def load_raw_sig_df():
    df = pd.read_parquet(LEDGER)
    df = df.rename(columns={
        "Signal Date": "Date",
        "Entry Price": "Price",
        "PnL_compounded": "PnL",
        "Risk_compounded": "Risk $",
        "Equity_at_Signal": "Equity at Signal",
    })
    for c in ["Date", "Entry Date", "Exit Date", "Time Stop"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    # Scale dollar/share columns to the $10k start. Compounded sizing is
    # scale-invariant; this is the 750k path viewed at 1/75 (fractional shares).
    for c in ["Shares", "PnL", "Risk $", "Equity at Signal"]:
        df[c] = df[c].astype(float) * SCALE
    return df


def load_master(tickers):
    uniq = sorted(set(tickers) | {"SPY"})
    md = data_provider.get_history(uniq, start="2000-01-01")
    for k in list(md.keys()):
        ck = k.replace(".", "-")
        if ck not in md:
            md[ck] = md[k]
    return md


# ---------------------------------------------------------------- html helpers
def fmt_table(df, fmts=None, cls="dt-small", index=False):
    """Format columns per str.format templates, emit a DataTables-ready table."""
    d = df.copy()
    fmts = fmts or {}
    for c, tmpl in fmts.items():
        if c in d.columns:
            d[c] = d[c].map(lambda v, t=tmpl: ("" if pd.isna(v) else t.format(v)))
    return d.to_html(index=index, escape=False, border=0, classes=cls, na_rep="")


def metric_cards(items):
    cells = "".join(
        f'<div class="card"><div class="mlabel">{lbl}</div>'
        f'<div class="mval">{val}</div></div>' for lbl, val in items)
    return f'<div class="cards">{cells}</div>'


def section(title, *html_parts):
    return f'<h2>{title}</h2>\n' + "\n".join(html_parts)


def fig_html(fig):
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       config={"displayModeBar": False})


def equity_fig(df_eq, log, hline=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_eq.index, y=df_eq["Equity"], mode="lines",
                             name="Equity", line=dict(color="#00FF00", width=2)))
    if hline is not None:
        fig.add_hline(y=hline, line_dash="dash", line_color="gray",
                      annotation_text=f"Start: ${hline:,.0f}")
    fig.update_layout(template=DARK, height=400, margin=dict(l=10, r=10, t=30, b=10),
                      yaxis_type=("log" if log else "linear"), yaxis_title="Equity ($)",
                      yaxis=dict(tickformat="$,.0f"))
    return fig


def cum_r_fig(trades, eligible):
    f = trades[trades["Strategy"].isin(eligible)]
    piv = f.pivot_table(index="Exit Date", columns="Strategy", values="R_Multiple", aggfunc="sum")
    fig = go.Figure()
    if not piv.empty:
        cum = piv.fillna(0).cumsum()
        for col in cum.columns:
            fig.add_trace(go.Scatter(x=cum.index, y=cum[col], mode="lines", name=str(col)))
    fig.update_layout(template=DARK, height=400, margin=dict(l=10, r=10, t=30, b=10),
                      hovermode="x unified", yaxis_title="Cumulative R",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig


def main():
    eq = START_EQ
    sig_df = load_raw_sig_df()
    start_date = sig_df["Date"].min()
    today = pd.Timestamp.today().normalize()
    print(f"Loaded {len(sig_df)} trades, scaled to ${eq:,.0f} start (x{SCALE:.4f})")
    print("Loading prices ...")
    md = load_master(sig_df["Ticker"].unique().tolist())

    parts = []
    total_pnl = sig_df["PnL"].sum()
    final_eq = eq + total_pnl
    parts.append(section(
        "Backtester View — full strategy book (liquid + overflow)",
        f'<p class="sub">{len(sig_df):,} trades | {sig_df["Ticker"].nunique()} tickers | '
        f'{sig_df["Date"].min().date()} &rarr; {sig_df["Date"].max().date()} | '
        f'starting capital ${eq:,.0f} '
        f'<span class="cap">(scaled from $750k; fractional shares; %/Sharpe/R identical)</span></p>'))

    # ---- Dynamic Sizing Analysis ----
    eqgrow = sig_df["Equity at Signal"].max() / sig_df["Equity at Signal"].min()
    parts.append(section(
        "Dynamic Sizing Analysis",
        metric_cards([
            ("Avg Risk/Trade", f"${sig_df['Risk $'].mean():,.0f}"),
            ("Risk Range", f"${sig_df['Risk $'].min():,.0f} - ${sig_df['Risk $'].max():,.0f}"),
            ("Final Equity", f"${final_eq:,.0f} ({(final_eq/eq-1)*100:,.0f}%)"),
            ("Peak/Min Equity Ratio", f"{eqgrow:.2f}x"),
        ])))

    # ---- Annual Performance ----
    port_daily = get_daily_mtm_series(sig_df, md, start_date=start_date)
    annual_df = calculate_annual_stats(port_daily, eq, trades_df=sig_df)
    ann_fmt = {"Trades": "{:,.0f}", "Total Return ($)": "${:,.0f}", "Total Return (%)": "{:.1%}",
               "Max Drawdown": "{:.1%}", "Sharpe Ratio": "{:.2f}", "Sortino Ratio": "{:.2f}",
               "Std Dev": "{:.1%}"}
    parts.append(section("Annual Performance", fmt_table(annual_df, ann_fmt, cls="dt-small")))

    # ---- Strategy Metrics ----
    stats_df = calculate_performance_stats(sig_df, md, eq, start_date=start_date)
    st_fmt = {"Trades": "{:.0f}", "Total PnL": "${:,.0f}", "Profit Factor": "{:.2f}",
              "SQN": "{:.2f}", "Sharpe (TIM)": "{:.2f}", "Sortino (TIM)": "{:.2f}"}
    parts.append(section(
        "Strategy Metrics",
        '<p class="cap">Sharpe / Sortino are time-in-market (only days each strategy '
        'has open positions, normalized by starting equity).</p>',
        fmt_table(stats_df, st_fmt, cls="dt-small")))

    # ---- charts: full + trailing 12m ----
    df_eq = calculate_mark_to_market_curve(sig_df, md, eq, start_date=start_date)
    end_date = df_eq.index.max() if not df_eq.empty else sig_df["Exit Date"].max()
    cutoff = end_date - pd.DateOffset(years=1)

    strat_counts = sig_df.groupby("Strategy").size()
    eligible = strat_counts[strat_counts >= MIN_SAMPLE].index.tolist()
    excluded = strat_counts[strat_counts < MIN_SAMPLE]
    excl_note = (f'<p class="cap">Hidden (n &lt; {MIN_SAMPLE} over full history): '
                 + ", ".join(f"{s} ({c})" for s, c in excluded.items()) + "</p>") if len(excluded) else ""

    fig_eq_full = equity_fig(df_eq, log=True, hline=eq) if not df_eq.empty else go.Figure()
    fig_r_full = cum_r_fig(sig_df, eligible)
    parts.append(section(
        "Full history &mdash; Equity (MTM, log) &amp; Cumulative R by strategy",
        '<div class="row2">'
        f'<div>{fig_html(fig_eq_full)}</div>'
        f'<div>{fig_html(fig_r_full)}{excl_note}</div></div>'))

    df_eq_12 = df_eq[df_eq.index >= cutoff] if not df_eq.empty else df_eq
    trades_12 = sig_df[sig_df["Exit Date"] >= cutoff]
    fig_eq_12 = equity_fig(df_eq_12, log=False) if not df_eq_12.empty else go.Figure()
    fig_r_12 = cum_r_fig(trades_12, eligible)
    parts.append(section(
        f"Trailing 12 months ({cutoff.date()} &rarr; {end_date.date()}) &mdash; Equity (MTM) &amp; Cumulative R",
        '<div class="row2">'
        f'<div>{fig_html(fig_eq_12)}</div>'
        f'<div>{fig_html(fig_r_12)}</div></div>'))

    # ---- Strategy Correlation Matrix ----
    corr_df, _, _ = build_strategy_correlation_matrix(sig_df, md, min_trades=30, mode="calendar")
    corr_part = '<p class="cap">Need >=2 strategies with >=30 trades.</p>'
    div_html = ""
    if corr_df is not None and not corr_df.empty and len(corr_df) >= 2:
        cvals = corr_df.copy()
        np.fill_diagonal(cvals.values, np.nan)
        fig_corr = px.imshow(cvals.values, x=cvals.columns, y=cvals.index, text_auto=".2f",
                             aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                             labels=dict(color="Correlation"))
        fig_corr.update_layout(template=DARK, height=max(380, 42 * len(cvals) + 130),
                               margin=dict(l=130, r=30, t=20, b=130), xaxis=dict(tickangle=45))
        corr_part = fig_html(fig_corr)
        avg_corr = cvals.mean(axis=1).sort_values()
        max_corr = cvals.max(axis=1)
        max_with = cvals.idxmax(axis=1)
        rows = []
        for s in avg_corr.index:
            v = avg_corr[s]
            bg = ("#1a4d1a" if v < 0.2 else "#4d4d1a" if v < 0.4 else "#663f1a" if v < 0.6 else "#661a1a")
            fg = ("#9fe09f" if v < 0.2 else "#e0e09f" if v < 0.4 else "#e0b89f" if v < 0.6 else "#e09f9f")
            rows.append(f"<tr><td>{s}</td>"
                        f'<td style="background:{bg};color:{fg};text-align:right">{v:.2f}</td>'
                        f'<td style="text-align:right">{max_corr[s]:.2f}</td><td>{max_with[s]}</td></tr>')
        div_html = ('<p><b>Diversification score</b> — avg correlation vs other strategies. '
                    'Lower = better diversifier.</p>'
                    '<table class="dataframe dt-small"><thead><tr><th>Strategy</th><th>Avg Corr (vs others)</th>'
                    '<th>Max Corr</th><th>Most Correlated With</th></tr></thead><tbody>'
                    + "".join(rows) + "</tbody></table>")
    parts.append(section("Strategy Correlation Matrix (Daily P&amp;L)", corr_part, div_html))

    # ---- Exposure Over Time ----
    exp_df = calculate_daily_exposure(sig_df, starting_equity=eq)
    exp_part = '<p class="cap">No exposure data.</p>'
    if not exp_df.empty:
        colors = {"Long Exposure %": "#00CC00", "Short Exposure %": "#CC0000",
                  "Net Exposure %": "#0066CC", "Gross Exposure %": "#FF9900"}
        fig_exp = go.Figure()
        for col in exp_df.columns:
            fig_exp.add_trace(go.Scatter(x=exp_df.index, y=exp_df[col], mode="lines", name=col,
                                         line=dict(color=colors.get(col, "#888"), width=1.5)))
        fig_exp.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1)
        fig_exp.update_layout(template=DARK, height=400, margin=dict(l=10, r=10, t=30, b=10),
                              hovermode="x unified", yaxis_title="Exposure (%)",
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        exp_part = fig_html(fig_exp) + metric_cards([
            ("Avg Gross Exposure", f"{exp_df['Gross Exposure %'].mean():.1f}%"),
            ("Max Gross Exposure", f"{exp_df['Gross Exposure %'].max():.1f}%"),
            ("Avg Net Exposure", f"{exp_df['Net Exposure %'].mean():.1f}%"),
            ("Max Net Exposure", f"{exp_df['Net Exposure %'].max():.1f}%"),
        ])
    parts.append(section("Exposure Over Time (% of Equity)", exp_part))

    # ---- Cross-Strategy Signal Overlap ----
    ov = sig_df[["Date", "Strategy", "Ticker", "Action", "Risk $", "R_Multiple"]].copy()
    ov["Date"] = pd.to_datetime(ov["Date"]).dt.normalize()
    ov["TradedAs"] = ov["Ticker"].map(lambda t: SPOT_TO_TRADEABLE.get(t, t))
    grp = ov.groupby(["Date", "TradedAs"])["Strategy"].nunique().reset_index(name="NumStrats")
    keys = grp[grp["NumStrats"] >= 2][["Date", "TradedAs"]]
    if len(keys) == 0:
        ov_part = '<p class="cap">No same-date / same-tradeable overlaps.</p>'
    else:
        hits = ov.merge(keys, on=["Date", "TradedAs"], how="inner")
        pivot = hits.pivot_table(index=["Date", "TradedAs"], columns="Strategy", values="Action",
                                 aggfunc=lambda s: " / ".join(sorted(set(s)))).reset_index()
        pivot = pivot.sort_values("Date", ascending=False).rename(columns={"TradedAs": "Traded As"})
        # Per pair: count co-fire instances AND sum the two strategies' R on
        # those shared (Date, TradedAs) instances (R is sizing-invariant).
        pair_days = defaultdict(int)
        pair_r = defaultdict(float)
        for (_d, _t), g in hits.groupby(["Date", "TradedAs"]):
            r_by_strat = g.groupby("Strategy")["R_Multiple"].sum()
            ss = sorted(r_by_strat.index)
            for i in range(len(ss)):
                for j in range(i + 1, len(ss)):
                    key = (ss[i], ss[j])
                    pair_days[key] += 1
                    pair_r[key] += float(r_by_strat[ss[i]] + r_by_strat[ss[j]])
        if pair_days:
            psum = pd.DataFrame([
                {"Strategy A": a, "Strategy B": b, "Co-Fire Days": pair_days[(a, b)],
                 "Cum R (A+B)": pair_r[(a, b)],
                 "Avg R/Day": pair_r[(a, b)] / pair_days[(a, b)]}
                for (a, b) in pair_days
            ]).sort_values("Co-Fire Days", ascending=False)
        else:
            psum = pd.DataFrame()
        ov_part = (f'<p><b>{len(keys)} overlap days</b> across {keys["TradedAs"].nunique()} tickers.</p>'
                   '<div class="row2"><div>' + fmt_table(pivot, {"Date": "{:%Y-%m-%d}"}, cls="dt-big")
                   + '</div><div><b>Strategy pair co-fire frequency</b>'
                   '<p class="cap">Cum R (A+B) = combined R of both strategies on their shared '
                   'co-fire instances. Avg R/Day = per co-fire event.</p>'
                   + fmt_table(psum, {"Cum R (A+B)": "{:.1f}", "Avg R/Day": "{:.2f}"}, cls="dt-small")
                   + "</div></div>")
    parts.append(section("Cross-Strategy Signal Overlap (same date, same tradeable)", ov_part))

    # ---- Current Exposure (open positions) ----
    open_df = sig_df[sig_df["Time Stop"] >= today].copy() if "Time Stop" in sig_df.columns else sig_df.iloc[0:0]
    if not open_df.empty:
        cur, opnl, mv = [], [], []
        for r in open_df.itertuples():
            t = r.Ticker.replace(".", "-")
            tdf = md.get(t)
            last = float("nan")
            if tdf is not None and not tdf.empty:
                tmp = tdf.copy()
                if isinstance(tmp.columns, pd.MultiIndex):
                    tmp.columns = [c[0] if isinstance(c, tuple) else c for c in tmp.columns]
                tmp.columns = [c.capitalize() for c in tmp.columns]
                last = float(tmp["Close"].iloc[-1])
            p = (last - r.Price) * r.Shares if r.Action == "BUY" else (r.Price - last) * r.Shares
            cur.append(last); opnl.append(p); mv.append(last * r.Shares)
        open_df["Current Price"] = cur; open_df["Open PnL"] = opnl; open_df["Mkt Value"] = mv
        tl = open_df[open_df["Action"] == "BUY"]["Mkt Value"].sum()
        ts = open_df[open_df["Action"] == "SELL SHORT"]["Mkt Value"].sum()
        oe_cards = metric_cards([
            ("# Positions", f"{len(open_df)}"),
            ("Total Long", f"${tl:,.0f}"), ("Total Short", f"${ts:,.0f}"),
            ("Net Exposure", f"${tl-ts:,.0f}"), ("Total Open PnL", f"${open_df['Open PnL'].sum():,.0f}"),
        ])
        oc = ["Entry Date", "Time Stop", "Strategy", "Ticker", "Action", "Price",
              "Current Price", "Shares", "Mkt Value", "PnL", "ATR", "Risk $", "Equity at Signal"]
        oc = [c for c in oc if c in open_df.columns]
        oe_tbl = fmt_table(open_df[oc], {
            "Entry Date": "{:%Y-%m-%d}", "Time Stop": "{:%Y-%m-%d}", "Price": "${:,.2f}",
            "Current Price": "${:,.2f}", "PnL": "${:,.2f}", "Mkt Value": "${:,.2f}", "ATR": "{:.2f}",
            "Shares": "{:,.2f}", "Risk $": "${:,.2f}", "Equity at Signal": "${:,.0f}"}, cls="dt-small")
        parts.append(section("Current Exposure (Active Positions)", oe_cards, oe_tbl))
    else:
        parts.append(section("Current Exposure (Active Positions)",
                             '<p class="cap">No active positions (Time Stop &gt;= today).</p>'))

    # ---- Trade Log (all rows; interactive) ----
    disp = ["Date", "Entry Date", "Exit Date", "Exit Type", "Strategy", "Tier", "Ticker", "Action",
            "Entry Criteria", "Signal Close", "T+1 Open", "Price", "Shares", "R_Multiple", "PnL", "ATR",
            "Equity at Signal", "Risk $"]
    disp = [c for c in disp if c in sig_df.columns]
    tl_df = sig_df[disp].sort_values("Date", ascending=False)
    parts.append(section(
        "Trade Log (all %d trades — sortable / searchable)" % len(sig_df),
        '<p class="cap">Click a column header to sort; type in the search box (e.g. a strategy '
        'name or ticker) to filter.</p>',
        fmt_table(tl_df, {
            "Price": "${:,.2f}", "Signal Close": "${:,.2f}", "T+1 Open": "${:,.2f}", "PnL": "${:,.2f}",
            "R_Multiple": "{:.2f}", "Date": "{:%Y-%m-%d}", "Entry Date": "{:%Y-%m-%d}",
            "Exit Date": "{:%Y-%m-%d}", "ATR": "{:.2f}", "Equity at Signal": "${:,.0f}",
            "Risk $": "${:,.2f}", "Shares": "{:,.2f}"}, cls="dt-big")))

    # ---- assemble ----
    css = """
    body{background:#0e1117;color:#e6e6e6;font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:0;padding:24px 32px;}
    h1{font-size:22px;} h2{font-size:18px;border-bottom:1px solid #2a2f3a;padding-bottom:6px;margin-top:34px;}
    .sub{color:#9aa0aa;margin-top:-6px;} .cap{color:#8a909a;font-size:12px;}
    .cards{display:flex;flex-wrap:wrap;gap:12px;margin:10px 0;}
    .card{background:#1a1f2b;border:1px solid #2a2f3a;border-radius:8px;padding:10px 16px;min-width:150px;}
    .mlabel{color:#9aa0aa;font-size:12px;} .mval{font-size:18px;font-weight:600;margin-top:4px;}
    .row2{display:grid;grid-template-columns:1fr 1fr;gap:18px;} @media(max-width:1100px){.row2{grid-template-columns:1fr;}}
    table.dataframe{border-collapse:collapse;width:100%;font-size:12px;margin:8px 0;color:#e6e6e6;}
    table.dataframe th{background:#1a1f2b;color:#cfd3da;text-align:right;padding:5px 8px;border-bottom:1px solid #2a2f3a;}
    table.dataframe td{padding:4px 8px;border-bottom:1px solid #20242e;text-align:right;}
    table.dataframe td:nth-child(1),table.dataframe th:nth-child(1){text-align:left;}
    /* DataTables dark theme */
    .dataTables_wrapper{color:#cfd3da;font-size:12px;margin-bottom:14px;}
    .dataTables_filter input,.dataTables_length select{background:#1a1f2b;color:#e6e6e6;border:1px solid #2a2f3a;border-radius:4px;padding:2px 6px;}
    .dataTables_wrapper .dataTables_paginate .paginate_button{color:#cfd3da!important;}
    .dataTables_wrapper .dataTables_paginate .paginate_button.current{background:#1a1f2b!important;border:1px solid #2a2f3a;color:#fff!important;}
    table.dataTable tbody tr{background:#0e1117;} table.dataTable.stripe tbody tr.odd{background:#141925;}
    table.dataTable thead th{border-bottom:1px solid #2a2f3a;}
    """
    init_js = """
    $(function(){
      $('table.dt-small').each(function(){ $(this).DataTable({paging:false,searching:false,info:false,order:[]}); });
      $('table.dt-big').each(function(){ $(this).DataTable({pageLength:25,order:[],scrollX:true,
        lengthMenu:[[25,50,100,-1],[25,50,100,'All']]}); });
    });
    """
    body = "\n".join(parts)
    html = (f"<!doctype html><html><head><meta charset='utf-8'><title>Backtester View</title>"
            f"<link rel='stylesheet' href='{DT_CSS}'>"
            f"<script src='{JQUERY_CDN}'></script>"
            f"<script src='{PLOTLY_CDN}'></script>"
            f"<script src='{DT_JS}'></script>"
            f"<style>{css}</style></head>"
            f"<body><h1>Strategy Backtester — HTML view</h1>{body}"
            f"<script>{init_js}</script></body></html>")
    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {OUT_HTML}  ({len(html)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
