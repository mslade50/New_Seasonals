import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import plotly.graph_objs as go
import streamlit as st
from datetime import date, timedelta

def compute_atr(df, window=14):
    df = df.copy()
    df["previous_close"] = df["Close"].shift(1)
    df["TR"] = df[["High", "previous_close"]].max(axis=1) - df[["Low", "previous_close"]].min(axis=1)
    df["ATR"] = df["TR"].rolling(window=window).mean()
    df["ATR%"] = (df["ATR"] / df["Close"]) * 100
    return df

def summarize_data(df, include_atr=True):
    if df.empty:
        # Return with ATR and % Pos if include_atr is True, else still return these columns as NaN
        if include_atr:
            return {
                "Avg Return (%)": np.nan, 
                "Median Daily Return (%)": np.nan, 
                "Avg ATR%": np.nan,
                "% Pos": np.nan
            }
        else:
            return {
                "Avg Return (%)": np.nan, 
                "Median Daily Return (%)": np.nan, 
                "Avg ATR%": np.nan,  # even if include_atr=False, we return the column as NaN for consistency
                "% Pos": np.nan
            }
    daily_returns = df["log_return"] * 100  # convert to percentage
    pos_percentage = (daily_returns > 0).sum() / len(daily_returns) * 100
    if include_atr:
        return {
            "Avg Return (%)": daily_returns.mean(),
            "Median Daily Return (%)": daily_returns.median(),
            "Avg ATR%": df["ATR%"].mean(),
            "% Pos": pos_percentage
        }
    else:
        # Even if include_atr=False, let's still return ATR% as NaN so all tables have same columns
        return {
            "Avg Return (%)": daily_returns.mean(),
            "Median Daily Return (%)": daily_returns.median(),
            "Avg ATR%": np.nan,  # no ATR in original daily request, but we add it as NaN for consistency
            "% Pos": pos_percentage
        }

def get_current_trading_info():
    # Determine today's trading day_of_month and week_of_month_5day
    today = dt.date.today()
    start_of_month = dt.date(today.year, today.month, 1)
    current_data = yf.download("SPY", start=start_of_month, end=today + timedelta(days=1)) 
    if not current_data.empty:
        current_data["trading_day_of_month"] = np.arange(1, len(current_data) + 1)
        current_data["week_of_month_5day"] = (current_data["trading_day_of_month"] - 1) // 5 + 1
        # Merge week 5 into week 4
        current_data.loc[current_data["week_of_month_5day"] > 4, "week_of_month_5day"] = 4
        current_trading_day_of_month = current_data["trading_day_of_month"].iloc[-1]
        current_week_of_month = current_data["week_of_month_5day"].iloc[-1]
        return current_trading_day_of_month, current_week_of_month
    else:
        return None, None

def seasonals_chart(ticker, cycle_label, show_tables):
    cycle_start_mapping = {
        "Election": 1952,
        "Pre-Election": 1951,
        "Post-Election": 1953,
        "Midterm": 1950
    }
    cycle_start = cycle_start_mapping[cycle_label]

    end_date = dt.datetime(2023, 12, 30)
    this_yr_end = dt.date.today() + timedelta(days=1)
    spx1 = yf.Ticker(ticker)
    spx = spx1.history(period="max", end=end_date)

    if spx.empty:
        st.error(f"No data found for {ticker}.")
        return

    spx["log_return"] = np.log(spx["Close"] / spx["Close"].shift(1))
    spx["year"] = spx.index.year
    spx["month"] = spx.index.month
    spx["trading_day_of_month"] = spx.groupby([spx.index.year, spx.index.month]).cumcount() + 1
    spx["week_of_month_5day"] = (spx["trading_day_of_month"] - 1) // 5 + 1

    years_in_cycle = [cycle_start + i * 4 for i in range(19)]
    cycle_data = spx[spx["year"].isin(years_in_cycle)]
    cycle_data = compute_atr(cycle_data)

    # Merge week 5 into week 4
    cycle_data.loc[cycle_data["week_of_month_5day"] > 4, "week_of_month_5day"] = 4

    current_year_data = yf.download(ticker, start=dt.datetime(this_yr_end.year, 1, 1), end=this_yr_end)
    if not current_year_data.empty:
        current_year_data["log_return"] = np.log(current_year_data["Close"] / current_year_data["Close"].shift(1))
        this_year_path = (
            current_year_data["log_return"]
            .cumsum()
            .apply(np.exp) - 1
        )
        current_trading_day = len(current_year_data)
    else:
        this_year_path = pd.Series(dtype=float)
        current_trading_day = None

    cycle_data["day_count"] = cycle_data.groupby("year").cumcount() + 1
    avg_path = (
        cycle_data.groupby("day_count")["log_return"]
        .mean()
        .cumsum()
        .apply(np.exp) - 1
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=avg_path.index,
        y=avg_path.values,
        mode="lines",
        name=f"Avg Path ({cycle_label})",
        line=dict(color="orange")
    ))

    fig.add_trace(go.Scatter(
        x=avg_path.index,
        y=this_year_path.values,
        mode="lines",
        name="This Year",
        line=dict(color="green")
    ))

    avg_path_y_value = avg_path[current_trading_day] if current_trading_day is not None and current_trading_day in avg_path.index else None
    if avg_path_y_value is not None:
        fig.add_trace(go.Scatter(
            x=[current_trading_day],
            y=[avg_path_y_value],
            mode="markers",
            name="Current Day on Avg Path",
            marker=dict(color="white", size=7),
            showlegend=False
        ))

    fig.update_layout(
        title=f"{ticker} - {cycle_label} Cycle Average",
        xaxis_title="Trading Day",
        yaxis_title="Cumulative Return",
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    st.plotly_chart(fig)

    now = dt.date.today()
    current_month = now.month
    next_month = current_month + 1 if current_month < 12 else 1

    if show_tables:
        # Handle January ATR by only considering second half of January
        cycle_data["day_of_month"] = cycle_data.index.day
        cycle_data.loc[(cycle_data["month"] == 1) & (cycle_data["day_of_month"] <= 15), "ATR%"] = np.nan

        # Monthly returns
        cycle_data["monthly_return"] = cycle_data.groupby(["year", "month"])["log_return"].transform("sum")

        # Table 1: Monthly Summary Stats
        # Compute stats and add % Pos
        monthly_group = cycle_data.groupby(["year", "month"])
        # Compute % Pos for each month/year
        monthly_pos = monthly_group.apply(lambda g: (g["log_return"] > 0).mean()*100).reset_index(name="% Pos")
        monthly_returns_by_month = cycle_data.groupby("month")["monthly_return"].agg(["mean", "median"])
        monthly_returns_by_month["mean"] = monthly_returns_by_month["mean"] * 100
        monthly_returns_by_month["median"] = monthly_returns_by_month["median"] * 100
        monthly_returns_by_month.columns = ["Avg Monthly Return (%)", "Median Monthly Return (%)"]
        atr_by_year_month = monthly_group["ATR%"].mean().reset_index()
        atr_by_month = atr_by_year_month.groupby("month")["ATR%"].mean().to_frame("Avg ATR%")

        # Compute overall % Pos by month
        monthly_pos_by_month = monthly_pos.groupby("month")["% Pos"].mean()

        summary_table_1 = monthly_returns_by_month.join(atr_by_month, on="month")
        summary_table_1 = summary_table_1.join(monthly_pos_by_month, on="month")

        st.subheader("Table 1: Monthly Summary Stats (All Cycle Years)")
        st.dataframe(summary_table_1.style.format({
            "Avg Monthly Return (%)": "{:.1f}%",
            "Median Monthly Return (%)": "{:.1f}%",
            "Avg ATR%": "{:.1f}%",
            "% Pos": "{:.1f}%"
        }))

        # Table 2: Current Month Weekly Stats
        current_month_data = cycle_data[cycle_data["month"] == current_month]
        current_month_weeks = sorted(current_month_data["week_of_month_5day"].unique())
        current_month_week_stats = []
        for w in current_month_weeks:
            w_data = current_month_data[current_month_data["week_of_month_5day"] == w]
            stats = summarize_data(w_data, include_atr=True)
            stats["Week"] = w
            current_month_week_stats.append(stats)
        current_month_week_df = pd.DataFrame(current_month_week_stats).set_index("Week")

        st.subheader("Table 2: Current Month Weekly Stats (5-Day Weeks Merged)")
        st.dataframe(current_month_week_df.style.format({
            "Avg Return (%)": "{:.1f}%",
            "Median Daily Return (%)": "{:.1f}%",
            "Avg ATR%": "{:.1f}%",
            "% Pos": "{:.1f}%"
        }))

        # Table 3: Next Month Weekly Stats
        next_month_data = cycle_data[cycle_data["month"] == next_month]
        if not next_month_data.empty:
            next_month_weeks = sorted(next_month_data["week_of_month_5day"].unique())
            next_month_week_stats = []
            for w in next_month_weeks:
                w_data = next_month_data[next_month_data["week_of_month_5day"] == w]
                stats = summarize_data(w_data, include_atr=True)
                stats["Week"] = w
                next_month_week_stats.append(stats)
            next_month_week_df = pd.DataFrame(next_month_week_stats).set_index("Week")
        else:
            next_month_week_df = pd.DataFrame(columns=["Avg Return (%)","Median Daily Return (%)","Avg ATR%","% Pos"])

        st.subheader("Table 3: Next Month Weekly Stats (5-Day Weeks Merged)")
        st.dataframe(next_month_week_df.style.format({
            "Avg Return (%)": "{:.1f}%",
            "Median Daily Return (%)": "{:.1f}%",
            "Avg ATR%": "{:.1f}%",
            "% Pos": "{:.1f}%"
        }))

        # Table 4: Current Month Daily Stats
        # Include ATR and % Pos here as well
        current_month_days = sorted(current_month_data["trading_day_of_month"].unique())
        current_month_day_stats = []
        for d in current_month_days:
            d_data = current_month_data[current_month_data["trading_day_of_month"] == d]
            # Now we include ATR and % Pos as well
            stats = summarize_data(d_data, include_atr=True)
            stats["Day"] = d
            current_month_day_stats.append(stats)
        current_month_day_df = pd.DataFrame(current_month_day_stats).set_index("Day")

        st.subheader("Table 4: Current Month Daily Stats")
        st.dataframe(current_month_day_df.style.format({
            "Avg Return (%)": "{:.1f}%",
            "Median Daily Return (%)": "{:.1f}%",
            "Avg ATR%": "{:.1f}%",
            "% Pos": "{:.1f}%"
        }))

        # Table 5: Next Month Daily Stats
        next_month_data = cycle_data[cycle_data["month"] == next_month]
        if not next_month_data.empty:
            next_month_days = sorted(next_month_data["trading_day_of_month"].unique())
            next_month_day_stats = []
            for d in next_month_days:
                d_data = next_month_data[next_month_data["trading_day_of_month"] == d]
                stats = summarize_data(d_data, include_atr=True)
                stats["Day"] = d
                next_month_day_stats.append(stats)
            next_month_day_df = pd.DataFrame(next_month_day_stats).set_index("Day")
        else:
            next_month_day_df = pd.DataFrame(columns=["Avg Return (%)","Median Daily Return (%)","Avg ATR%","% Pos"])

        st.subheader("Table 5: Next Month Daily Stats")
        st.dataframe(next_month_day_df.style.format({
            "Avg Return (%)": "{:.1f}%",
            "Median Daily Return (%)": "{:.1f}%",
            "Avg ATR%": "{:.1f}%",
            "% Pos": "{:.1f}%"
        }))

    # Print current trading day/week of the month at the end
    current_day_of_month, current_week_of_month = get_current_trading_info()
    if current_day_of_month is not None and current_week_of_month is not None:
        st.write(f"Today is the {current_day_of_month}-th trading day of this month and we are currently in week {current_week_of_month} of this month.")
    else:
        st.write("Unable to determine the current trading day/week of the month.")
        

st.title("Presidential Cycle Seasonality Chart")

ticker = st.text_input("Enter a stock ticker:", value="AAPL")
cycle_label = st.selectbox(
    "Select the presidential cycle type:",
    ["Election", "Pre-Election", "Post-Election", "Midterm"]
)

show_tables = st.sidebar.checkbox("Show Summary Tables")

if st.button("Plot"):
    try:
        seasonals_chart(ticker, cycle_label, show_tables)
    except Exception as e:
        st.error(f"Error generating chart: {e}")
