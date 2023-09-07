import requests
import os
import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px

st.title("Forward Ranks")
def download_db():
    url = "https://github.com/mslade50/New_Seasonals/raw/main/ticker_data.db"
    r = requests.get(url)
    with open("ticker_data.db", 'wb') as f:
        f.write(r.content)

def app():
    download_db()

    conn = sqlite3.connect("ticker_data.db")

    query = """
    SELECT *
    FROM ticker_results
    WHERE (Ticker, timestamp) IN (
        SELECT Ticker, MAX(timestamp)
        FROM ticker_results
        GROUP BY Ticker
    );
    """

    df = pd.read_sql_query(query, conn)
    st.write(df.head())
    fig1 = px.bar(df, x='Ticker', y='F5', title='F5 Data')
    st.plotly_chart(fig1)

    fig2 = px.bar(df, x='Ticker', y='F21', title='F21 Data')
    st.plotly_chart(fig2)

    fig3 = px.bar(df, x='Ticker', y='F63', title='F63 Data')
    st.plotly_chart(fig3)

    fig4 = px.bar(df, x='Ticker', y='Vol_Change', title='Vol_Change Data')
    st.plotly_chart(fig4)
