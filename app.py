import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, timedelta
from scipy import stats

# -- Page configuration --
st.set_page_config(page_title="Stock Analyzer", layout="wide")
st.title("Stock Analysis Dashboard")

# -- Sidebar --
st.sidebar.header("Settings")
ticker_input = st.sidebar.text_input("Stock Tickers (separate by commas)", value="NVDA, AAPL, KO")
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

col_side_a, col_side_b = st.sidebar.columns(2)
start_date = col_side_a.date_input("Start Date", date.today() - timedelta(days=365*2))
end_date = col_side_b.date_input("End Date", date.today())

ma_window = st.sidebar.slider("Moving Average Window (days)", 5, 200, 50)
vol_window = st.sidebar.slider("Rolling Volatility Window (days)", 10, 120, 30)
rf_rate = st.sidebar.number_input("Risk-Free Rate (%)", value=4.0, step=0.1) / 100

if len(tickers) < 2 or len(tickers) > 5:
    st.sidebar.error("Please enter between 2 and 5 tickers.")
    st.stop()

# -- Fail-Safe Data Download Logic --
@st.cache_data(show_spinner="Fetching market data...", ttl=3600)
def load_data(ticker_list, start, end):
    all_tickers = list(set(ticker_list + ["^GSPC"]))
    df_final = pd.DataFrame()
    dropped = []
    
    try:
        data = yf.download(all_tickers, start=start, end=end, progress=False, auto_adjust=True)
        if not data.empty:
            df_final = data['Close'] if isinstance(data.columns, pd.MultiIndex) else data[['Close']]
    except:
        pass

    for t in all_tickers:
        if t not in df_final.columns or df_final[t].isnull().all():
            try:
                single_data = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
                if not single_data.empty:
                    df_final[t] = single_data['Close']
                else:
                    dropped.append(t)
            except:
                dropped.append(t)

    if df_final.empty: return None, "No data found."
    df_clean = df_final.dropna(thresh=int(len(df_final)*0.9), axis=1).dropna()
    return df_clean, dropped

df_price, dropped_tickers = load_data(tickers, start_date, end_date)

if df_price is None:
    st.error("Critical Error: Data fetch failed.")
    st.stop()

user_stocks = [t for t in df_price.columns if t != "^GSPC"]
df_returns = df_price.pct_change().dropna()

tab1, tab2, tab3, tab4 = st.tabs(["Price & Returns", "Risk & Distribution", "Correlation", "Portfolio Explorer"])

# --- TAB 1 ---
with tab1:
    st.subheader("Price Analysis & Moving Averages")
    sel_stocks = st.multiselect("Select stocks:", options=user_stocks, default=user_stocks)
    if sel_stocks:
        fig_p = go.Figure()
        for s in sel_stocks:
            fig_p.add_trace(go.Scatter(x=df_price.index, y=df_price[s], name=f"{s} Price"))
            ma_series = df_price[s].rolling(window=ma_window).mean()
            fig_p.add_trace(go.Scatter(x=df_price.index, y=ma_series, name=f"{s} {ma_window}d MA", line=dict(dash='dot')))
        st.plotly_chart(fig_p, use_container_width=True)

    st.subheader("Annualized Summary Statistics")
    stats_df = pd.DataFrame(index=df_price.columns)
    stats_df['Mean Return'] = df_returns.mean() * 252
    stats_df['Volatility'] = df_returns.std() * np.sqrt(252)
    stats_df['Skewness'] = df_returns.skew()
    stats_df['Kurtosis'] = df_returns.kurtosis()
    st.table(stats_df.style.format("{:.2%}").format("{:.2f}", subset=['Skewness', 'Kurtosis']))

    st.subheader("Wealth Index ($10,000)")
    st.plotly_chart(px.line((1 + df_returns).cumprod() * 10000), use_container_width=True)

# --- TAB 2 ---
with tab2:
    st.subheader("Rolling Volatility")
    rolling_v = df_returns[user_stocks].rolling(window=vol_window).std() * np.sqrt(252)
    st.plotly_chart(px.line(rolling_v), use_container_width=True)
    
    d_stock = st.selectbox("Detailed Analysis for:", user_stocks)
    s_rets = df_returns[d_stock].dropna()
    c1, c2 = st.columns(2)
    with c1:
        (osm, osr), (slope, intercept, r) = stats.probplot(s_rets, dist="norm")
        fig_q = go.Figure()
        fig_q.add_trace(go.Scatter(x=osm, y=osr, mode='markers', name='Quantiles'))
        fig_q.add_trace(go.Scatter(x=osm, y=slope*osm + intercept, mode='lines', name='Normal'))
        st.plotly_chart(fig_q, use_container_width=True)
    with c2:
        jb_s, jb_p = stats.jarque_bera(s_rets)
        st.metric("P-Value", f"{jb_p:.4f}")
        if jb_p < 0.05: st.error("Non-Normal")
        else: st.success("Normal Distribution")

# --- TAB 3 ---
with tab3:
    st.subheader("Correlation Heatmap")
    st.plotly_chart(px.imshow(df_returns[user_stocks].corr(), text_auto=".2f"), use_container_width=True)
    
    s1 = st.selectbox("X-Axis Stock", user_stocks, index=0)
    s2 = st.selectbox("Y-Axis Stock", user_stocks, index=1)
    # FIX: Removed trendline="ols" to prevent statsmodels error
    st.plotly_chart(px.scatter(df_returns, x=s1, y=s2), use_container_width=True)

# --- TAB 4 ---
with tab4:
    st.subheader("Two-Asset Portfolio Explorer")
    p1, p2 = st.selectbox("Asset 1", user_stocks, index=0, key="p1"), st.selectbox("Asset 2", user_stocks, index=1, key="p2")
    w = st.slider(f"Allocation to {p1}", 0.0, 1.0, 0.5)
    
    r1, r2 = df_returns[p1].mean() * 252, df_returns[p2].mean() * 252
    v1, v2 = df_returns[p1].std() * np.sqrt(252), df_returns[p2].std() * np.sqrt(252)
    cv = df_returns[[p1, p2]].cov().iloc[0,1] * 252
    
    port_ret = w * r1 + (1-w) * r2
    port_vol = np.sqrt((w**2 * v1**2) + ((1-w)**2 * v2**2) + (2*w*(1-w)*cv))
    
    st.metric("Portfolio Risk (Vol)", f"{port_vol:.2%}")
    
    w_vals = np.linspace(0, 1, 100)
    v_vals = [np.sqrt((i**2 * v1**2) + ((1-i)**2 * v2**2) + (2*i*(1-i)*cv)) for i in w_vals]
    fig_ef = go.Figure()
    fig_ef.add_trace(go.Scatter(x=w_vals, y=v_vals, mode='lines'))
    fig_ef.add_trace(go.Scatter(x=[w], y=[port_vol], mode='markers', marker=dict(size=12, color='red')))
    st.plotly_chart(fig_ef, use_container_width=True)

with st.sidebar.expander("Notes"):
    st.write("Source: Yahoo Finance | Benchmark: ^GSPC")