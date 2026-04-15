import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, timedelta
from scipy import stats
import math

# -- Page configuration ----------------------------------
st.set_page_config(page_title="Stock Analyzer", layout="wide")
st.title("Stock Analysis Dashboard")

# -- Sidebar: user inputs --------------------------------
st.sidebar.header("Settings")

# 2.1.1 Ticker entry
ticker_input = st.sidebar.text_input("Stock Tickers (separate by commas)", value="AAPL, MSFT, GOOGL")
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

# 2.1.2 Date range selection
col_side_a, col_side_b = st.sidebar.columns(2)
start_date = col_side_a.date_input("Start Date", date.today() - timedelta(days=365*2))
end_date = col_side_b.date_input("End Date", date.today())

# Sliders (Moving Average added here)
ma_window = st.sidebar.slider("Moving Average Window (days)", 5, 200, 50)
vol_window = st.sidebar.slider("Rolling Volatility Window (days)", 10, 120, 30)
rf_rate = st.sidebar.number_input("Risk-Free Rate (%)", value=4.0, step=0.1) / 100

# Validation
if len(tickers) < 2 or len(tickers) > 5:
    st.sidebar.error("Please enter between 2 and 5 tickers.")
    st.stop()

if (end_date - start_date).days < 365:
    st.sidebar.error("The date range must be at least 1 year.")
    st.stop()

# -- 2.1.3 & 2.1.4 Data download and handling ------------
@st.cache_data(show_spinner="Fetching data...", ttl=3600)
def load_data(ticker_list, start, end):
    all_tickers = list(set(ticker_list + ["^GSPC"]))
    try:
        data = yf.download(all_tickers, start=start, end=end, progress=False, auto_adjust=True)
        if data.empty:
            return None, "No data found."
        
        if isinstance(data.columns, pd.MultiIndex):
            df_price = data['Close']
        else:
            df_price = data[['Close']]

        missing_pct = df_price.isnull().mean()
        too_much_missing = missing_pct[missing_pct > 0.05].index.tolist()
        df_clean = df_price.drop(columns=too_much_missing).dropna()
        
        return df_clean, too_much_missing
    except Exception as e:
        return None, str(e)

df_price, dropped_tickers = load_data(tickers, start_date, end_date)

if df_price is None:
    st.error(f"Error: {dropped_tickers}")
    st.stop()

if dropped_tickers:
    st.warning(f"Tickers dropped: {', '.join(dropped_tickers)}")

# Calculations
df_returns = df_price.pct_change().dropna()
user_stocks = [t for t in df_price.columns if t != "^GSPC"]

# -- Tabs ------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Price & Returns", "Risk & Distribution", "Correlation", "Portfolio Explorer"])

# --- TAB 1: Price and Return Analysis ---
with tab1:
    st.subheader("Price Chart")
    sel_stocks = st.multiselect("Select stocks to view:", options=user_stocks, default=user_stocks)
    
    if sel_stocks:
        fig_p = go.Figure()
        for s in sel_stocks:
            # Add Closing Price
            fig_p.add_trace(go.Scatter(x=df_price.index, y=df_price[s], name=f"{s} Price"))
            # Add Moving Average (Feature 9)
            ma_series = df_price[s].rolling(window=ma_window).mean()
            fig_p.add_trace(go.Scatter(x=df_price.index, y=ma_series, name=f"{s} {ma_window}d MA", line=dict(dash='dot')))
        
        fig_p.update_layout(xaxis_title="Date", yaxis_title="Price ($)")
        st.plotly_chart(fig_p, use_container_width=True)

    st.subheader("Summary Statistics (Annualized)")
    stats_df = pd.DataFrame(index=df_price.columns)
    stats_df['Mean Return'] = df_returns.mean() * 252
    stats_df['Volatility'] = df_returns.std() * np.sqrt(252)
    stats_df['Skewness'] = df_returns.skew()
    stats_df['Kurtosis'] = df_returns.kurtosis()
    stats_df['Min Return'] = df_returns.min()
    stats_df['Max Return'] = df_returns.max()
    st.table(stats_df.style.format("{:.2%}")
             .format("{:.2f}", subset=['Skewness', 'Kurtosis']))

    st.subheader("Cumulative Wealth Index")
    ew_rets = df_returns[user_stocks].mean(axis=1)
    wealth = (1 + df_returns).cumprod() * 10000
    wealth["Equal Weight Portfolio"] = (1 + ew_rets).cumprod() * 10000
    st.plotly_chart(px.line(wealth, title="Growth of $10,000"), use_container_width=True)

# --- TAB 2: Risk and Distribution ---
with tab2:
    st.subheader("Rolling Volatility")
    rolling_v = df_returns[user_stocks].rolling(window=vol_window).std() * np.sqrt(252)
    st.plotly_chart(px.line(rolling_v, title=f"{vol_window}-Day Rolling Volatility"), use_container_width=True)

    st.divider()
    d_stock = st.selectbox("Select stock for distribution analysis:", user_stocks)
    c_a, c_b = st.columns(2)
    
    with c_a:
        p_choice = st.radio("Plot Type:", ["Histogram", "Q-Q Plot"])
        s_rets = df_returns[d_stock].dropna()
        if p_choice == "Histogram":
            m, s = stats.norm.fit(s_rets)
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=s_rets, histnorm='probability density', name="Actual"))
            x_v = np.linspace(s_rets.min(), s_rets.max(), 100)
            fig.add_trace(go.Scatter(x=x_v, y=stats.norm.pdf(x_v, m, s), mode='lines', name='Normal', line=dict(color='red')))
            st.plotly_chart(fig, use_container_width=True)
        else:
            (osm, osr), (slope, intercept, r) = stats.probplot(s_rets, dist="norm")
            fig_q = go.Figure()
            fig_q.add_trace(go.Scatter(x=osm, y=osr, mode='markers', name='Quantiles'))
            fig_q.add_trace(go.Scatter(x=osm, y=slope*osm + intercept, mode='lines', name='Normal Line'))
            st.plotly_chart(fig_q, use_container_width=True)
    with c_b:
        st.subheader("Normality Test")
        jb_s, jb_p = stats.jarque_bera(s_rets)
        st.write(f"Statistic: {jb_s:.2f}, P-Value: {jb_p:.4f}")
        if jb_p < 0.05: st.error("Rejects normality (p < 0.05)")
        else: st.success("Fails to reject normality (p >= 0.05)")

    st.subheader("Returns Comparison (Box Plot)")
    st.plotly_chart(px.box(df_returns[user_stocks]), use_container_width=True)

# --- TAB 3: Correlation ---
with tab3:
    st.subheader("Correlation Heatmap")
    st.plotly_chart(px.imshow(df_returns[user_stocks].corr(), text_auto=".2f", color_continuous_scale="RdBu_r"), use_container_width=True)

    col_1, col_2 = st.columns(2)
    with col_1:
        st_a = st.selectbox("Stock A", user_stocks, index=0)
        st_b = st.selectbox("Stock B", user_stocks, index=1)
        st.plotly_chart(px.scatter(df_returns, x=st_a, y=st_b), use_container_width=True)
    with col_2:
        c_wind = st.slider("Correlation Window (Days)", 30, 120, 60)
        r_corr = df_returns[st_a].rolling(c_wind).corr(df_returns[st_b])
        st.plotly_chart(px.line(y=r_corr, x=df_returns.index, title="Rolling Correlation"), use_container_width=True)

# --- TAB 4: Portfolio Explorer ---
with tab4:
    st.subheader("Two-Asset Portfolio Analysis")
    p_a = st.selectbox("Asset A", user_stocks, index=0, key="pa")
    p_b = st.selectbox("Asset B", user_stocks, index=1, key="pb")
    w_a = st.slider(f"Weight on {p_a}", 0.0, 1.0, 0.5)
    
    # Financial math
    r_a, r_b = df_returns[p_a].mean() * 252, df_returns[p_b].mean() * 252
    s_a, s_b = df_returns[p_a].std() * np.sqrt(252), df_returns[p_b].std() * np.sqrt(252)
    cov_ab = df_returns[[p_a, p_b]].cov().iloc[0,1] * 252
    
    p_ret = w_a * r_a + (1-w_a) * r_b
    p_vol = np.sqrt((w_a**2 * s_a**2) + ((1-w_a)**2 * s_b**2) + (2 * w_a * (1-w_a) * cov_ab))
    
    m1, m2 = st.columns(2)
    m1.metric("Portfolio Return", f"{p_ret:.2%}")
    m2.metric("Portfolio Volatility", f"{p_vol:.2%}")
    
    # Risk Curve
    w_r = np.linspace(0, 1, 100)
    v_r = [np.sqrt((w**2 * s_a**2) + ((1-w)**2 * s_b**2) + (2 * w * (1-w) * cov_ab)) for w in w_r]
    f_curve = go.Figure()
    f_curve.add_trace(go.Scatter(x=w_r, y=v_r, mode='lines', name='Risk Curve'))
    f_curve.add_trace(go.Scatter(x=[w_a], y=[p_vol], mode='markers', marker=dict(size=12, color='red'), name='Current Mix'))
    st.plotly_chart(f_curve, use_container_width=True)
    st.info("The dip in the curve shows how diversification lowers overall risk.")

# -- Methodology --
with st.sidebar.expander("About this App"):
    st.write("Source: Yahoo Finance. Annualization: 252 days. Wealth Index: $10,000.")