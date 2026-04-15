import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, timedelta
import math
from scipy import stats

# -- Page configuration ----------------------------------
# Kept exactly as it appears in the guide
st.set_page_config(page_title="Stock Analyzer", layout="wide")
st.title("Stock Analysis Dashboard")

# -- Sidebar: user inputs --------------------------------
st.sidebar.header("Settings")

ticker = st.sidebar.text_input("Stock Ticker", value="AAPL").upper().strip()

# Feature 1: Custom Date Range
col_side1, col_side2 = st.sidebar.columns(2)
start_date = col_side1.date_input("Start Date", date.today() - timedelta(days=365))
end_date = col_side2.date_input("End Date", date.today())

if start_date >= end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()

# Feature 6: Risk-free rate
rf_rate = st.sidebar.number_input("Risk-Free Rate (%)", value=4.0, step=0.1) / 100

# Feature 7: Rolling volatility window
vol_window = st.sidebar.slider("Rolling Volatility Window (days)", 10, 120, 30)

# -- Data download ----------------------------------------
@st.cache_data(show_spinner="Fetching data...", ttl=3600)
def load_data(ticker: str, start: date, end: date) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False)
    return df

# -- Main logic -------------------------------------------
if ticker:
    try:
        df = load_data(ticker, start_date, end_date)
    except Exception as e:
        st.error(f"Failed to download data: {e}")
        st.stop()

    if df.empty:
        st.error(f"No data found for **{ticker}**.")
        st.stop()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # -- Compute derived columns -------------------------
    df["Daily Return"] = df["Close"].pct_change()
    df["Cumulative Return"] = (1 + df["Daily Return"]).cumprod()
    df["Rolling Volatility"] = df["Daily Return"].rolling(window=vol_window).std() * math.sqrt(252)

    # -- Key metrics --------------------------------------
    latest_close = float(df["Close"].iloc[-1])
    total_return = float(df["Cumulative Return"].iloc[-1] - 1)
    ann_volatility = float(df["Daily Return"].std() * math.sqrt(252))
    
    # Sharpe Ratio
    sharpe = (total_return - rf_rate) / ann_volatility if ann_volatility != 0 else 0
    
    # Drawdown
    rolling_max = df["Close"].cummax()
    drawdown = (df["Close"] - rolling_max) / rolling_max
    max_drawdown = float(drawdown.min())

    st.subheader(f"{ticker} — Key Metrics")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latest Close", f"${latest_close:,.2f}")
    col2.metric("Total Return", f"{total_return:.2%}")
    col3.metric("Annualized Volatility", f"{ann_volatility:.2%}")
    col4.metric("Sharpe Ratio", f"{sharpe:.2f}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Max Drawdown", f"{max_drawdown:.2%}")
    col6.metric("Skewness", f"{stats.skew(df['Daily Return'].dropna()):.2f}")
    col7.metric("Kurtosis", f"{stats.kurtosis(df['Daily Return'].dropna()):.2f}")
    col8.metric("Period High", f"${float(df['Close'].max()):,.2f}")

    st.divider()

    # -- Tabs for Visuals --------------------------------
    tab1, tab2, tab3 = st.tabs(["Price Action", "Volatility", "Distribution Analysis"])

    with tab1:
        st.subheader("Closing Price")
        fig_price = px.line(df, x=df.index, y="Close", template="plotly_white")
        st.plotly_chart(fig_price, use_container_width=True)
        
        st.subheader("Cumulative Return")
        fig_cum = px.line(df, x=df.index, y="Cumulative Return", template="plotly_white")
        st.plotly_chart(fig_cum, use_container_width=True)

    with tab2:
        st.subheader("Rolling Annualized Volatility")
        fig_roll_vol = px.line(df, x=df.index, y="Rolling Volatility", template="plotly_white")
        fig_roll_vol.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_roll_vol, use_container_width=True)

    with tab3:
        # Feature 8: Distribution Analysis & Normality Test
        st.subheader("Daily Returns Distribution")
        returns_clean = df["Daily Return"].dropna()
        
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=returns_clean, nbinsx=60, name="Daily Returns",
            histnorm="probability density", marker_color="mediumpurple", opacity=0.75
        ))

        # Normal curve overlay
        x_range = np.linspace(float(returns_clean.min()), float(returns_clean.max()), 200)
        mu, sigma = float(returns_clean.mean()), float(returns_clean.std())
        fig_hist.add_trace(go.Scatter(
            x=x_range, y=stats.norm.pdf(x_range, mu, sigma),
            mode="lines", name="Normal Distribution", line=dict(color="red", width=2)
        ))

        fig_hist.update_layout(xaxis_title="Daily Return", yaxis_title="Density", template="plotly_white")
        st.plotly_chart(fig_hist, use_container_width=True)

        # Jarque-Bera Test Result
        jb_stat, jb_pvalue = stats.jarque_bera(returns_clean)
        st.info(f"**Jarque-Bera test:** statistic = {jb_stat:.2f}, p-value = {jb_pvalue:.4f}")
        st.caption("A p-value < 0.05 indicates the returns are likely NOT normally distributed.")

    with st.expander("View Raw Data"):
        st.dataframe(df)

else:
    st.info("Enter a stock ticker in the sidebar to get started.")