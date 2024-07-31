import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# URL of the raw Excel file in your GitHub repository
file_url = 'https://github.com/mslade50/New_Seasonals/raw/main/Trade%20Details%20(MS).xlsx'

@st.cache_data
def load_data():
    df = pd.read_excel(file_url, sheet_name='All Trades')
    return df

# Refresh data button
if st.button('Refresh Data'):
    st.cache_data.clear()
    st.experimental_rerun()

df = load_data()
date_threshold = pd.Timestamp('2023-01-31')

# Filter the DataFrame
df = df[df['Closing Date'] > date_threshold]
df['Rolling PnL'] = df['PnL'].cumsum()

st.title("Trade Analysis Dashboard")

# Filters
type_filter = st.multiselect('Filter by Type', options=df['Type'].unique())
kind_filter = st.multiselect('Filter by Kind', options=df['Kind'].unique())
instrument_filter = st.multiselect('Filter by Instrument', options=df['Instrument'].unique())
direction_filter = st.multiselect('Filter by Direction', options=df['Direction'].unique())
year_filter = st.multiselect('Filter by Year', options=df['Year'].unique())
account_filter = st.multiselect('Filter by Account', options=df['Account'].unique())

dff = df.copy()
if type_filter:
    dff = dff[dff['Type'].isin(type_filter)]
if kind_filter:
    dff = dff[dff['Kind'].isin(kind_filter)]
if instrument_filter:
    dff = dff[dff['Instrument'].isin(instrument_filter)]
if direction_filter:
    dff = dff[dff['Direction'].isin(direction_filter)]
if year_filter:
    dff = dff[dff['Year'].isin(year_filter)]
if account_filter:
    dff = dff[dff['Account'].isin(account_filter)]

# Equity Curve
dff['Rolling PnL'] = dff['PnL'].cumsum()
fig1 = go.Figure(data=[go.Scatter(x=dff['Closing Date'], y=dff['Rolling PnL'], mode='lines', name='Equity Curve')])
st.plotly_chart(fig1)

# Trade PnL Bar Chart
trade_pnl_summary = dff.groupby('Ticker')['PnL'].sum().reset_index()
fig2 = go.Figure(data=[go.Bar(x=trade_pnl_summary['Ticker'], y=trade_pnl_summary['PnL'], name='Total Trade PnL')])
st.plotly_chart(fig2)

# Stats Table
total_trades = len(dff)
winning_trades = len(dff[dff['PnL'] > 0])
winning_pct = winning_trades / total_trades if total_trades > 0 else 0
avg_win = dff[dff['PnL'] > 0]['PnL'].mean() if winning_trades > 0 else 0
avg_loss = dff[dff['PnL'] < 0]['PnL'].mean() if total_trades - winning_trades > 0 else 0
avg_win_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else 0

avg_win_pct = dff[dff['PnL'] > 0]['PnL %'].mean() if winning_trades > 0 else 0
avg_loss_pct = dff[dff['PnL'] < 0]['PnL %'].mean() if total_trades - winning_trades > 0 else 0
avg_win_loss_pct_ratio = avg_win_pct / abs(avg_loss_pct) if avg_loss_pct != 0 else 0

stats_data = {
    "Total Trades": total_trades,
    "Winning %": f"{winning_pct:.2%}",
    "Average Win ($)": f"${avg_win:.2f}",
    "Average Win (%)": f"{avg_win_pct*100:.2f}%",
    "Average Loss ($)": f"${avg_loss:.2f}",
    "Average Loss (%)": f"{avg_loss_pct*100:.2f}%",
    "Average Win/Loss Ratio ($)": f"{avg_win_ratio:.2f}",
    "Average Win/Loss Ratio (%)": f"{avg_win_loss_pct_ratio:.2f}"
}

st.write(pd.DataFrame(list(stats_data.items()), columns=['Metric', 'Value']))
