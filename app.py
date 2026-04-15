import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, timedelta
import math
from scipy import stats

# -- Page configuration --
st.set_page_config(page_title="Ultimate Stock Analyzer", layout="wide")
st.title("📈 Ultimate Stock Analysis Dashboard")

# -- Sidebar: user inputs --
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL").upper().strip()

# Feature 1: Custom Date Range
col_side1, col_side2 = st.sidebar.columns(2)
start_date = col_side1.date_input("Start Date", date.today() - timedelta(days=365))
end_date = col_side2.date_input("End Date", date.today())

if start_date >= end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()

# Feature 6: Risk-free rate for Sharpe Ratio
rf_rate = st.sidebar.number_input("Risk-Free Rate (%)", value=4.0, step=0.1) / 100

# Feature 7: Rolling volatility window
vol_window = st.sidebar.slider("Rolling Vol Window (Days)", 10, 120, 30)

# -- Data download --
@st.cache_data(show_spinner="Fetching data...", ttl=3600)
def load_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

# -- Main logic --
if ticker:
    try:
        df = load_data(ticker, start_date, end_date)
        
        if df.empty:
            st.error(f"No data found for {ticker}.")
            st.stop()

        # Calculations & Derived Columns
        df["Daily Return"] = df["Close"].pct_change()
        df["Cumulative Return"] = (1 + df["Daily Return"]).cumprod()
        
        # Performance Metrics
        latest_close = float(df["Close"].iloc[-1])
        total_ret = float(df["Cumulative Return"].iloc[-1] - 1)
        ann_vol = float(df["Daily Return"].std() * math.sqrt(252))
        
        # Sharpe Ratio
        sharpe = (total_ret - rf_rate) / ann_vol if ann_vol != 0 else 0
        
        # Max Drawdown
        rolling_max = df["Close"].cummax()
        drawdown = (df["Close"] - rolling_max) / rolling_max
        max_drawdown = float(drawdown.min())

        # -- Display Metrics --
        st.subheader(f"{ticker} Performance Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Latest Close", f"${latest_close:,.2f}")
        m2.metric("Total Return", f"{total_ret:.2%}")
        m3.metric("Annualized Vol", f"{ann_vol:.2%}")
        m4.metric("Sharpe Ratio", f"{sharpe:.2f}")

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Max Drawdown", f"{max_drawdown:.2%}")
        m6.metric("Skewness", f"{stats.skew(df['Daily Return'].dropna()):.2f}")
        m7.metric("Kurtosis", f"{stats.kurtosis(df['Daily Return'].dropna()):.2f}")
        m8.metric("Data Points", len(df))

        st.divider()

        # -- Tabs for Charts --
        tab1, tab2, tab3 = st.tabs(["Price & Returns", "Volatility", "Distribution (Feature 8)"])

        with tab1:
            # Price Chart
            fig_price = px.line(df, x=df.index, y="Close", title=f"{ticker} Price Action")
            st.plotly_chart(fig_price, use_container_width=True)
            
            # Cumulative Returns
            fig_cum = px.line(df, x=df.index, y="Cumulative Return", title="Growth of $1 Investment")
            st.plotly_chart(fig_cum, use_container_width=True)

        with tab2:
            # Feature 7: Rolling Volatility
            df["Rolling Vol"] = df["Daily Return"].rolling(vol_window).std() * math.sqrt(252)
            fig_vol = px.line(df, x=df.index, y="Rolling Vol", title=f"{vol_window}-Day Rolling Volatility")
            fig_vol.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig_vol, use_container_width=True)

        with tab3:
            # Feature 8: Distribution Analysis & Normality Test
            returns_clean = df["Daily Return"].dropna()
            
            fig_dist = go.Figure()
            # Histogram
            fig_dist.add_trace(go.Histogram(
                x=returns_clean, nbinsx=50, name="Actual Returns",
                histnorm="probability density", marker_color="mediumpurple", opacity=0.7
            ))
            
            # Fitted Normal Curve
            x_range = np.linspace(returns_clean.min(), returns_clean.max(), 100)
            mu, sigma = returns_clean.mean(), returns_clean.std()
            fig_dist.add_trace(go.Scatter(
                x=x_range, y=stats.norm.pdf(x_range, mu, sigma),
                mode="lines", name="Normal Distribution", line=dict(color="red", width=2)
            ))
            
            fig_dist.update_layout(title="Returns Distribution vs. Normal Curve", template="plotly_white")
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Jarque-Bera Test
            jb_stat, jb_p = stats.jarque_bera(returns_clean)
            st.info(f"**Jarque-Bera Normality Test:** Statistic: {jb_stat:.2f}, p-value: {jb_p:.4f}")
            if jb_p < 0.05:
                st.warning("Result: Reject Normality (Returns are not normally distributed).")
            else:
                st.success("Result: Fail to reject normality.")

        with st.expander("View Raw Data"):
            st.dataframe(df)

    except Exception as e:
        st.error(f"Analysis Error: {e}")
else:
    st.info("Enter a ticker in the sidebar to begin.")