def plot_seasonal_paths(ticker, cycle_label, stats_row=None):
    # --- Stats Header ---
    if stats_row is not None:
        def get_val(col): return stats_row.get(col, np.nan)
        st.caption(f"📊 **Historical {cycle_label} Stats (from Screener)**")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("5d Avg", f"{get_val('Seas_Cyc_Avg_5d'):.2f}%")
        c2.metric("5d Med", f"{get_val('Seas_Cyc_Med_5d'):.2f}%")
        c3.metric("5d Win%", f"{get_val('Seas_Cyc_Win_5d'):.0f}%")
        c4.metric("21d Avg", f"{get_val('Seas_Cyc_Avg_21d'):.2f}%")
        c5.metric("21d Med", f"{get_val('Seas_Cyc_Med_21d'):.2f}%")
        c6.metric("21d Win%", f"{get_val('Seas_Cyc_Win_21d'):.0f}%")

    st.subheader(f"📈 {ticker} Seasonal Average Path: {cycle_label}")
    
    # Fetch Data
    end_date = dt.datetime.now()
    spx = yf.download(ticker, period="max", end=end_date, progress=False)
    
    if spx.empty:
        st.error(f"No data found for {ticker}.")
        return

    if isinstance(spx.columns, pd.MultiIndex):
        spx.columns = spx.columns.get_level_values(0)

    # Engineering
    spx["log_return"] = np.log(spx["Close"] / spx["Close"].shift(1))
    spx["year"] = spx.index.year
    spx["day_count"] = spx.groupby("year").cumcount() + 1 

    # --- SPLIT DATA: Historical vs Current ---
    current_year = date.today().year
    spx_historical = spx[spx["year"] < current_year].copy()
    df_current_year = spx[spx["year"] == current_year].copy()

    # --- Cycle Path Calculation ---
    if cycle_label == "All Years":
        cycle_data = spx_historical.copy()
        line_name = "All Years Avg Path"
    else:
        cycle_start = CYCLE_START_MAPPING.get(cycle_label, 1953)
        years_in_cycle = [cycle_start + i * 4 for i in range((current_year - cycle_start) // 4 + 1)] 
        cycle_data = spx_historical[spx_historical["year"].isin(years_in_cycle)].copy()
        line_name = f"Avg Path ({cycle_label})"
        
    avg_path = (cycle_data.groupby("day_count")["log_return"].mean().cumsum().apply(np.exp) - 1)

    # --- Realized YTD Path ---
    realized_path = pd.Series(dtype=float)
    if not df_current_year.empty:
        realized_path = df_current_year.set_index("day_count")["log_return"].cumsum().apply(np.exp) - 1

    # --- Identify Best Buy/Sell Points ---
    # We look for local extrema in the AVG PATH
    best_buys = find_optimal_points(avg_path, is_buy=True, top_n=3, min_separation=7)
    best_sells = find_optimal_points(avg_path, is_buy=False, top_n=3, min_separation=7)

    # --- Plotting ---
    fig = go.Figure()
    
    # 1. Seasonal Average Line
    fig.add_trace(go.Scatter(x=avg_path.index, y=avg_path.values, mode="lines", name=line_name, line=dict(color="orange", width=3)))

    # 2. Realized YTD Line
    if not realized_path.empty:
        fig.add_trace(go.Scatter(
            x=realized_path.index, 
            y=realized_path.values, 
            mode="lines", 
            name=f"{current_year} Realized", 
            line=dict(color="#39FF14", width=2)
        ))

    # 3. Buy/Sell Vertical Lines
    for day, val in best_buys:
        fig.add_vline(x=day, line_width=1, line_dash="solid", line_color="#00FF00", opacity=0.6)
        # Optional: Add small marker on the line
        fig.add_trace(go.Scatter(
            x=[day], y=[val], mode='markers', 
            marker=dict(color='#00FF00', size=6, symbol='triangle-up'),
            name="Best Buy", showlegend=False, hoverinfo='x+y'
        ))

    for day, val in best_sells:
        fig.add_vline(x=day, line_width=1, line_dash="dot", line_color="red", opacity=0.6)
        # Optional: Add small marker on the line
        fig.add_trace(go.Scatter(
            x=[day], y=[val], mode='markers', 
            marker=dict(color='red', size=6, symbol='triangle-down'),
            name="Best Sell", showlegend=False, hoverinfo='x+y'
        ))

    # --- All Years Overlay ---
    if cycle_label != "All Years":
        all_avg_path = (spx_historical.groupby("day_count")["log_return"].mean().cumsum().apply(np.exp) - 1)
        fig.add_trace(go.Scatter(x=all_avg_path.index, y=all_avg_path.values, mode="lines", name="All Years Avg Path", line=dict(color="lightblue", width=1, dash='dash')))

    # --- Current Day Markers ---
    current_day_count = realized_path.index[-1] if not realized_path.empty else None

    if current_day_count:
        val_t = avg_path.get(current_day_count)
        val_t5 = avg_path.get(current_day_count + 5)
        val_t21 = avg_path.get(current_day_count + 21)

        if val_t is not None:
            fig.add_trace(go.Scatter(x=[current_day_count], y=[val_t], mode="markers", name="Curr Day", marker=dict(color="red", size=10, line=dict(width=2, color='white'))))
        if val_t5 is not None:
            fig.add_trace(go.Scatter(x=[current_day_count + 5], y=[val_t5], mode="markers", name="T+5", marker=dict(color="#00FF00", size=8, symbol="circle")))
        if val_t21 is not None:
            fig.add_trace(go.Scatter(x=[current_day_count + 21], y=[val_t21], mode="markers", name="T+21", marker=dict(color="#00BFFF", size=8, symbol="circle")))
        
    fig.update_layout(
        xaxis_title="Trading Day of Year", yaxis_title="Cumulative Return (%)", yaxis_tickformat=".2%",
        plot_bgcolor="black", paper_bgcolor="black", font=dict(color="white"), height=500,
        showlegend=True, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(0,0,0,0)")
    )
    st.plotly_chart(fig, use_container_width=True)
