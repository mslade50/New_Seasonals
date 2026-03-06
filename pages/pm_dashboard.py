"""
PM Dashboard — Situation Room
===============================
Single-page view combining environment + portfolio + decisions.

Reads all data from cached files (parquet + JSON) populated by
daily_risk_report.py. No risk_dashboard_v2 import needed.

Data: All cached — no yfinance downloads needed. Should render in <3 seconds.
"""

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os
import sys
import json
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
try:
    st.set_page_config(
        page_title="PM Dashboard",
        page_icon="🎯",
        layout="wide",
    )
except Exception:
    pass

# ---------------------------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

DATA_DIR = os.path.join(parent_dir, "data")
ENV_CACHE = os.path.join(DATA_DIR, "rd2_environment.json")
FRAG_CACHE = os.path.join(DATA_DIR, "rd2_fragility.parquet")

# ---------------------------------------------------------------------------
# IMPORTS — Strategy config (portfolio)
# ---------------------------------------------------------------------------
try:
    from strategy_config import _STRATEGY_BOOK_RAW, ACCOUNT_VALUE
except ImportError:
    _STRATEGY_BOOK_RAW = []
    ACCOUNT_VALUE = 750000

# ---------------------------------------------------------------------------
# TICKER → SECTOR MAPPING (inline, covers LIQUID_UNIVERSE)
# ---------------------------------------------------------------------------
TICKER_SECTOR = {
    # Technology
    'AAPL': 'Technology', 'ADBE': 'Technology', 'ADI': 'Technology', 'ADP': 'Technology',
    'ADSK': 'Technology', 'AMAT': 'Technology', 'AMD': 'Technology', 'AVGO': 'Technology',
    'CRM': 'Technology', 'CSCO': 'Technology', 'GOOG': 'Technology', 'IBM': 'Technology',
    'INTC': 'Technology', 'META': 'Technology', 'MSFT': 'Technology', 'MU': 'Technology',
    'NVDA': 'Technology', 'ORCL': 'Technology', 'QCOM': 'Technology', 'TXN': 'Technology',
    'QQQ': 'Technology', 'SMH': 'Technology', 'XLK': 'Technology',
    # Healthcare
    'ABT': 'Healthcare', 'AMGN': 'Healthcare', 'BAX': 'Healthcare', 'BDX': 'Healthcare',
    'BMY': 'Healthcare', 'CVS': 'Healthcare', 'GILD': 'Healthcare', 'HUM': 'Healthcare',
    'JNJ': 'Healthcare', 'LLY': 'Healthcare', 'MDT': 'Healthcare', 'MRK': 'Healthcare',
    'PFE': 'Healthcare', 'REGN': 'Healthcare', 'SYK': 'Healthcare', 'TMO': 'Healthcare',
    'UNH': 'Healthcare', 'IBB': 'Healthcare', 'IHI': 'Healthcare', 'XBI': 'Healthcare',
    'XLV': 'Healthcare',
    # Financials
    'AIG': 'Financials', 'ALL': 'Financials', 'AXP': 'Financials', 'BAC': 'Financials',
    'BK': 'Financials', 'C': 'Financials', 'GS': 'Financials', 'HIG': 'Financials',
    'JPM': 'Financials', 'KEY': 'Financials', 'MET': 'Financials', 'MMC': 'Financials',
    'MS': 'Financials', 'PGR': 'Financials', 'RF': 'Financials', 'SCHW': 'Financials',
    'STT': 'Financials', 'TRV': 'Financials', 'USB': 'Financials', 'WFC': 'Financials',
    'KRE': 'Financials', 'XLF': 'Financials',
    # Consumer Discretionary
    'AMZN': 'Consumer Disc', 'DIS': 'Consumer Disc', 'F': 'Consumer Disc',
    'HD': 'Consumer Disc', 'LOW': 'Consumer Disc', 'MCD': 'Consumer Disc',
    'NKE': 'Consumer Disc', 'ROST': 'Consumer Disc', 'SBUX': 'Consumer Disc',
    'TGT': 'Consumer Disc', 'TJX': 'Consumer Disc', 'WHR': 'Consumer Disc',
    'XLY': 'Consumer Disc', 'XRT': 'Consumer Disc', 'XHB': 'Consumer Disc',
    'ITB': 'Consumer Disc',
    # Consumer Staples
    'CAG': 'Consumer Staples', 'CL': 'Consumer Staples', 'COST': 'Consumer Staples',
    'CPB': 'Consumer Staples', 'GIS': 'Consumer Staples', 'HRL': 'Consumer Staples',
    'HSY': 'Consumer Staples', 'K': 'Consumer Staples', 'KMB': 'Consumer Staples',
    'KO': 'Consumer Staples', 'KR': 'Consumer Staples', 'MO': 'Consumer Staples',
    'PEP': 'Consumer Staples', 'PG': 'Consumer Staples', 'SYY': 'Consumer Staples',
    'TAP': 'Consumer Staples', 'TSN': 'Consumer Staples', 'WMT': 'Consumer Staples',
    'XLP': 'Consumer Staples',
    # Industrials
    'BA': 'Industrials', 'CAT': 'Industrials', 'CSX': 'Industrials', 'DE': 'Industrials',
    'DOV': 'Industrials', 'ECL': 'Industrials', 'EMR': 'Industrials', 'FDX': 'Industrials',
    'GD': 'Industrials', 'GE': 'Industrials', 'GPC': 'Industrials', 'HON': 'Industrials',
    'ITW': 'Industrials', 'LMT': 'Industrials', 'LUV': 'Industrials', 'MAS': 'Industrials',
    'MMM': 'Industrials', 'NOC': 'Industrials', 'NSC': 'Industrials', 'PAYX': 'Industrials',
    'PH': 'Industrials', 'RHI': 'Industrials', 'ROK': 'Industrials', 'RTX': 'Industrials',
    'SNA': 'Industrials', 'SWK': 'Industrials', 'UNP': 'Industrials', 'VMC': 'Industrials',
    'WM': 'Industrials', 'XLI': 'Industrials', 'ITA': 'Industrials',
    # Energy
    'COP': 'Energy', 'CVX': 'Energy', 'EOG': 'Energy', 'HAL': 'Energy',
    'OXY': 'Energy', 'SLB': 'Energy', 'VLO': 'Energy', 'WMB': 'Energy',
    'XLE': 'Energy', 'XOM': 'Energy',
    # Utilities
    'AEP': 'Utilities', 'CMS': 'Utilities', 'CNP': 'Utilities', 'D': 'Utilities',
    'DTE': 'Utilities', 'DUK': 'Utilities', 'ED': 'Utilities', 'EIX': 'Utilities',
    'ETR': 'Utilities', 'EXC': 'Utilities', 'FE': 'Utilities', 'NEE': 'Utilities',
    'PCG': 'Utilities', 'PEG': 'Utilities', 'PNW': 'Utilities', 'PPL': 'Utilities',
    'SO': 'Utilities', 'SRE': 'Utilities', 'XLU': 'Utilities',
    # Materials
    'ADM': 'Materials', 'APD': 'Materials', 'FCX': 'Materials', 'GLW': 'Materials',
    'IP': 'Materials', 'LIN': 'Materials', 'NEM': 'Materials', 'NUE': 'Materials',
    'PPG': 'Materials', 'SHW': 'Materials', 'XLB': 'Materials', 'XME': 'Materials',
    # Real Estate
    'IYR': 'Real Estate', 'PSA': 'Real Estate', 'SPG': 'Real Estate',
    'VNQ': 'Real Estate', 'XLRE': 'Real Estate',
    # Communication Services
    'CMCSA': 'Communication', 'T': 'Communication', 'VZ': 'Communication',
    'XLC': 'Communication',
    # Broad Index
    'SPY': 'Index', 'DIA': 'Index', 'IWM': 'Index',
    'AON': 'Financials', 'LEG': 'Consumer Disc', 'HPQ': 'Technology', 'V': 'Financials',
    'VFC': 'Consumer Disc',
}


# ---------------------------------------------------------------------------
# HELPER: LOAD ENVIRONMENT DATA
# ---------------------------------------------------------------------------

def _load_environment():
    """Compute risk signals and fragility from cached data. Returns dict or None."""
    if not os.path.exists(ENV_CACHE):
        return None
    try:
        with open(ENV_CACHE, 'r') as f:
            snapshot = json.load(f)

        # Convert signals dict to match expected format
        signals_ordered = {}
        for name, sig in snapshot.get('signals', {}).items():
            signals_ordered[name] = sig

        return {
            'price_ctx': snapshot.get('price_ctx', {}),
            'signals_ordered': signals_ordered,
            'h_scores': snapshot.get('h_scores'),
            'active_count': sum(1 for s in signals_ordered.values() if s.get('on')),
        }
    except Exception:
        return None


def _load_portfolio():
    """Load cached sig_df from backtest. Returns (sig_df, open_df) or (None, None)."""
    cache_path = os.path.join(DATA_DIR, "backtest_sig_df.parquet")
    if not os.path.exists(cache_path):
        return None, None
    try:
        sig_df = pd.read_parquet(cache_path)
        for col in ['Date', 'Entry Date', 'Exit Date', 'Time Stop']:
            if col in sig_df.columns:
                sig_df[col] = pd.to_datetime(sig_df[col])

        today = pd.Timestamp(datetime.date.today())
        open_mask = sig_df['Exit Date'] >= today
        open_df = sig_df[open_mask].copy()
        return sig_df, open_df
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# PANEL BUILDERS
# ---------------------------------------------------------------------------

def _render_panel_environment(env):
    """Panel 1: The Environment."""
    st.markdown("### 🌍 The Environment")

    if env is None:
        st.warning("Risk Dashboard data not available. Run Risk Dashboard V2 or daily_risk_report.py to populate cache.")
        return

    # Price context banner
    p = env['price_ctx']
    cols = st.columns(5)
    cols[0].metric("SPY", f"${p['price']:.2f}")
    ret_12m = p.get('ret_12m')
    cols[1].metric("12M Return", f"{ret_12m:+.1%}" if ret_12m is not None else "N/A")
    ext = p.get('extension_200d')
    cols[2].metric("vs 200d SMA", f"{ext:+.1%}" if ext is not None else "N/A")
    dd = p.get('drawdown')
    cols[3].metric("Drawdown", f"{dd:+.1%}" if dd is not None else "N/A")
    regime = p.get('regime_label', 'Unknown')
    cols[4].markdown(f"**Regime:** {regime}")

    # Fragility dials (inline — no risk_dashboard import needed)
    h_scores = env.get('h_scores')
    if h_scores:
        dial_cols = st.columns(3)
        for i, (key, label) in enumerate([('5d', '5-Day'), ('21d', '21-Day'), ('63d', '63-Day')]):
            score = float(h_scores.get(key, 0))
            bar_color = '#00CC00' if score < 20 else '#7FCC00' if score < 40 else '#FFD700' if score < 60 else '#FF8C00' if score < 80 else '#CC0000'
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=min(score, 100),
                title={'text': label, 'font': {'size': 13}},
                number={'font': {'size': 32}},
                gauge={'axis': {'range': [0, 100], 'tickvals': [0, 25, 50, 75, 100],
                                'ticktext': ['Robust', '', 'Neutral', '', 'Fragile'], 'tickfont': {'size': 9}},
                       'bar': {'color': bar_color, 'thickness': 0.3}, 'bgcolor': 'rgba(0,0,0,0)',
                       'steps': [{'range': [0, 20], 'color': 'rgba(0,204,0,0.12)'},
                                 {'range': [20, 40], 'color': 'rgba(0,204,0,0.06)'},
                                 {'range': [40, 60], 'color': 'rgba(255,215,0,0.08)'},
                                 {'range': [60, 80], 'color': 'rgba(255,140,0,0.10)'},
                                 {'range': [80, 100], 'color': 'rgba(204,0,0,0.12)'}]},
            ))
            fig.update_layout(height=180, margin=dict(l=15, r=15, t=35, b=5),
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            dial_cols[i].plotly_chart(fig, width="stretch")

    # Active signals summary
    signals = env['signals_ordered']
    active = env['active_count']
    st.markdown(f"**Active Signals:** {active}/6")
    active_names = [name for name, sig in signals.items() if sig.get('on')]
    if active_names:
        for name in active_names:
            sig = signals[name]
            detail = sig.get('detail', '') or sig.get('summary', '')
            st.markdown(f"- **{name}:** {detail}")
    else:
        st.markdown("_All signals clear._")


def _render_panel_portfolio(sig_df, open_df):
    """Panel 2: The Portfolio."""
    st.markdown("### 📊 The Portfolio")

    if sig_df is None:
        st.warning("No backtest data cached. Run the Strategy Backtester first.")
        return

    if open_df is None or open_df.empty:
        st.info("No open positions.")
    else:
        st.markdown(f"**Open Positions:** {len(open_df)}")
        display_cols = ['Strategy', 'Ticker', 'Action', 'Entry Date', 'Exit Date',
                       'Price', 'Shares', 'Risk $']
        available = [c for c in display_cols if c in open_df.columns]
        st.dataframe(
            open_df[available].sort_values('Entry Date', ascending=False).style.format({
                'Price': '${:.2f}', 'Risk $': '${:,.0f}', 'Shares': '{:.0f}',
                'Entry Date': '{:%Y-%m-%d}', 'Exit Date': '{:%Y-%m-%d}',
            }),
            width="stretch", height=min(len(open_df) * 40 + 40, 300),
        )

    # Sector exposure
    if open_df is not None and not open_df.empty:
        open_df_sect = open_df.copy()
        open_df_sect['Sector'] = open_df_sect['Ticker'].map(TICKER_SECTOR).fillna('Other')
        open_df_sect['Dollar Exposure'] = open_df_sect['Price'] * open_df_sect['Shares']
        sector_exp = open_df_sect.groupby('Sector')['Dollar Exposure'].sum().sort_values(ascending=True)

        if not sector_exp.empty:
            col1, col2 = st.columns(2)
            with col1:
                fig = go.Figure(go.Bar(
                    x=sector_exp.values, y=sector_exp.index,
                    orientation='h',
                    marker_color='#0066CC',
                    hovertemplate='%{y}: $%{x:,.0f}<extra></extra>',
                ))
                fig.update_layout(
                    height=max(200, len(sector_exp) * 30),
                    margin=dict(l=10, r=10, t=25, b=10),
                    title=dict(text='Sector Exposure ($)', font=dict(size=13)),
                    xaxis=dict(tickformat='$,.0f'),
                )
                st.plotly_chart(fig, width="stretch")

            with col2:
                # Net/gross exposure
                long_exp = open_df_sect[open_df_sect['Action'] == 'BUY']['Dollar Exposure'].sum()
                short_exp = open_df_sect[open_df_sect['Action'] == 'SELL SHORT']['Dollar Exposure'].sum()
                gross = long_exp + short_exp
                net = long_exp - short_exp
                st.metric("Gross Exposure", f"${gross:,.0f}", delta=f"{gross/ACCOUNT_VALUE:.1%} of account")
                st.metric("Net Exposure", f"${net:,.0f}", delta=f"{net/ACCOUNT_VALUE:+.1%} of account")

                # Factor tilt
                long_count = len(open_df_sect[open_df_sect['Action'] == 'BUY'])
                short_count = len(open_df_sect[open_df_sect['Action'] == 'SELL SHORT'])
                st.metric("Long/Short Count", f"{long_count}L / {short_count}S")

                # Strategy type mix
                strat_types = {}
                for _, row in open_df_sect.iterrows():
                    strat_name = row['Strategy']
                    for s in _STRATEGY_BOOK_RAW:
                        if s['name'] == strat_name:
                            stype = s.get('setup', {}).get('type', 'Unknown')
                            strat_types[stype] = strat_types.get(stype, 0) + 1
                            break
                if strat_types:
                    type_str = " | ".join(f"{k}: {v}" for k, v in sorted(strat_types.items()))
                    st.markdown(f"**Strategy Mix:** {type_str}")


def _render_panel_algo_book(sig_df, open_df):
    """Panel 3: The Algo Book."""
    st.markdown("### 📖 The Algo Book")

    if sig_df is None:
        st.warning("No backtest data cached.")
        return

    today = pd.Timestamp(datetime.date.today())

    # Recent entries (last 5 business days)
    recent_cutoff = today - pd.tseries.offsets.BDay(5)
    recent = sig_df[sig_df['Entry Date'] >= recent_cutoff].sort_values('Entry Date', ascending=False)
    st.markdown(f"**Recent Entries** (last 5 days): {len(recent)}")
    if not recent.empty:
        rcols = ['Strategy', 'Ticker', 'Action', 'Entry Date', 'Price', 'Shares']
        available = [c for c in rcols if c in recent.columns]
        st.dataframe(
            recent[available].style.format({
                'Price': '${:.2f}', 'Shares': '{:.0f}', 'Entry Date': '{:%Y-%m-%d}',
            }),
            width="stretch", height=min(len(recent) * 40 + 40, 200),
        )
    else:
        st.caption("No entries in last 5 days.")

    # Upcoming time exits
    if open_df is not None and not open_df.empty and 'Exit Date' in open_df.columns:
        exit_cutoff = today + pd.tseries.offsets.BDay(5)
        upcoming_exits = open_df[open_df['Exit Date'] <= exit_cutoff].sort_values('Exit Date')
        st.markdown(f"**Upcoming Exits** (next 5 days): {len(upcoming_exits)}")
        if not upcoming_exits.empty:
            ecols = ['Strategy', 'Ticker', 'Action', 'Entry Date', 'Exit Date', 'Price', 'Shares']
            available = [c for c in ecols if c in upcoming_exits.columns]
            st.dataframe(
                upcoming_exits[available].style.format({
                    'Price': '${:.2f}', 'Shares': '{:.0f}',
                    'Entry Date': '{:%Y-%m-%d}', 'Exit Date': '{:%Y-%m-%d}',
                }),
                width="stretch", height=min(len(upcoming_exits) * 40 + 40, 200),
            )
        else:
            st.caption("No positions expiring in next 5 days.")

    # Strategy utilization
    if open_df is not None and not open_df.empty:
        st.markdown("**Strategy Utilization**")
        util_rows = []
        for s in _STRATEGY_BOOK_RAW:
            sname = s['name']
            max_pos = s['settings'].get('max_total_positions', 99)
            active = len(open_df[open_df['Strategy'] == sname])
            if active > 0 or max_pos < 50:
                util_rows.append({
                    'Strategy': sname,
                    'Active': active,
                    'Max': max_pos,
                    'Utilization': f"{active}/{max_pos}",
                })
        if util_rows:
            st.dataframe(pd.DataFrame(util_rows), width="stretch", height=min(len(util_rows) * 40 + 40, 250))


def _render_panel_decision_queue(env, open_df):
    """Panel 4: Decision Queue — rules-based alerts."""
    st.markdown("### 🚨 Decision Queue")

    alerts = []

    # Fragility alerts
    if env and env.get('h_scores'):
        frag_21d = env['h_scores'].get('21d', 0)
        if frag_21d >= 90:
            alerts.append(('🔴', f"21d Fragility at {frag_21d:.0f} — Consider hedge overlay (see playbook)"))
        elif frag_21d >= 70:
            min_mult = max(0.5, 1.0 - (frag_21d / 100) * 0.5)
            alerts.append(('🟠', f"21d Fragility at {frag_21d:.0f} — Consider reducing new position sizes to {min_mult:.0%}"))

    # Signal count alerts
    if env:
        active_count = env.get('active_count', 0)
        if active_count >= 3:
            active_names = [name for name, sig in env['signals_ordered'].items() if sig.get('on')]
            alerts.append(('🟡', f"{active_count} risk signals active: {', '.join(active_names)}"))

    # Sector concentration
    if open_df is not None and not open_df.empty:
        open_df_sect = open_df.copy()
        open_df_sect['Sector'] = open_df_sect['Ticker'].map(TICKER_SECTOR).fillna('Other')
        open_df_sect['Dollar Exposure'] = open_df_sect['Price'] * open_df_sect['Shares']
        total_exp = open_df_sect['Dollar Exposure'].sum()
        if total_exp > 0:
            sector_pcts = open_df_sect.groupby('Sector')['Dollar Exposure'].sum() / total_exp
            max_sector = sector_pcts.idxmax()
            max_pct = sector_pcts.max()
            if max_pct > 0.40:
                alerts.append(('🟡', f"Portfolio is {max_pct:.0%} concentrated in {max_sector}"))

    # Display
    if alerts:
        for icon, msg in alerts:
            st.markdown(f"{icon} {msg}")
    else:
        st.success("Environment is benign and portfolio is balanced.")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    st.title("🎯 PM Dashboard — Situation Room")
    st.caption("Environment + Portfolio + Algo Book + Decision Queue | All data from cache — no downloads.")
    st.markdown("---")

    # Load data
    with st.spinner("Loading environment data..."):
        env = _load_environment()
    sig_df, open_df = _load_portfolio()

    # Decision Queue at the top (most actionable)
    _render_panel_decision_queue(env, open_df)
    st.markdown("---")

    # Environment + Portfolio side by side
    col_env, col_port = st.columns(2)
    with col_env:
        _render_panel_environment(env)
    with col_port:
        _render_panel_portfolio(sig_df, open_df)

    st.markdown("---")

    # Algo Book full width
    _render_panel_algo_book(sig_df, open_df)


if __name__ == "__main__":
    main()
