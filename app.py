# app.py
# -------------------------------------------------------
# A simple Streamlit stock analysis dashboard.
# Run with:  uv run streamlit run app.py
# -------------------------------------------------------
import numpy as np
from scipy import stats
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta
import math

# -- Page configuration ----------------------------------
# st.set_page_config must be the FIRST Streamlit command in the script.
# If you add any other st.* calls above this line, you'll get an error.
st.set_page_config(page_title="Stock Analyzer", layout="wide")
st.title("Stock Analysis Dashboard")

# -- Sidebar: user inputs --------------------------------
st.sidebar.header("Settings")

# Ticker entry with validation
ticker_input = st.sidebar.text_area(
    "Stock Tickers",
    value="AAPL, MSFT",
    height=80,
    help="Enter 2-5 stock ticker symbols, separated by commas or line breaks"
)

def validate_tickers(input_string: str) -> tuple[list[str], list[str]]:
    """
    Validate ticker input.
    Returns: (valid_tickers, error_messages)
    """
    errors = []
    
    # Split by comma or newline and clean up
    tickers = []
    for line in input_string.replace(',', '\n').split('\n'):
        cleaned = line.strip().upper()
        if cleaned:
            tickers.append(cleaned)
    
    # Check count
    if len(tickers) < 2:
        errors.append(f"At least 2 tickers required. You provided {len(tickers)}.")
    elif len(tickers) > 5:
        errors.append(f"No more than 5 tickers allowed. You provided {len(tickers)}.")
    
    # Validate ticker format (alphanumeric, 1-5 characters)
    valid_tickers = []
    for t in tickers:
        if not t.isalpha() or len(t) < 1 or len(t) > 5:
            errors.append(f"Invalid ticker: '{t}'. Tickers must be 1-5 letters only.")
        else:
            valid_tickers.append(t)
    
    return valid_tickers, errors

# Validate and process tickers
valid_tickers, validation_errors = validate_tickers(ticker_input)

if validation_errors:
    for error in validation_errors:
        st.sidebar.error(error)
    st.stop()

tickers = valid_tickers

# Default date range: one year back from today
default_start = date.today() - timedelta(days=365)
start_date = st.sidebar.date_input("Start Date", value=default_start, min_value=date(1970, 1, 1))
end_date = st.sidebar.date_input("End Date", value=date.today(), min_value=date(1970, 1, 1))

# Validate that the date range makes sense
if start_date >= end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()

# Enforce minimum range of 1 year
min_range_days = 365
date_range_days = (end_date - start_date).days
if date_range_days < min_range_days:
    st.sidebar.error(f"Date range must be at least {min_range_days} days (1 year). Currently: {date_range_days} days.")
    st.stop()

# Let the user pick a moving-average window
ma_window = st.sidebar.slider(
    "Moving Average Window (days)", min_value=5, max_value=200, value=50, step=5
)
# Risk-free rate for Sharpe ratio calculation
risk_free_rate = st.sidebar.number_input(
    "Risk-Free Rate (%)", min_value=0.0, max_value=20.0, value=4.5, step=0.1
) / 100

# Rolling volatility window
vol_window = st.sidebar.slider(
    "Rolling Volatility Window (days)", min_value=10, max_value=120, value=30, step=5
)

# -- Data download ----------------------------------------
# We wrap the download in st.cache_data so repeated runs with
# the same inputs don't re-download every time. The ttl (time-to-live)
# ensures the cache expires after one hour so data stays fresh.
@st.cache_data(show_spinner="Fetching data...", ttl=3600)
def load_data(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Download daily data from Yahoo Finance for a given date range."""
    df = yf.download(ticker, start=start, end=end, progress=False)
    return df

def compute_metrics(df: pd.DataFrame, ticker: str, ma_window: int, vol_window: int, risk_free_rate: float):
    """Compute all metrics for a given ticker."""
    # Compute derived columns
    df["Daily Return"] = df["Adj Close"].pct_change()
    df[f"{ma_window}-Day MA"] = df["Adj Close"].rolling(window=ma_window).mean()
    df["Cumulative Return"] = (1 + df["Daily Return"]).cumprod() - 1
    df["Rolling Volatility"] = df["Daily Return"].rolling(window=vol_window).std() * math.sqrt(252)

    # Calculate key metrics
    latest_close = float(df["Adj Close"].iloc[-1])
    total_return = float(df["Cumulative Return"].iloc[-1])
    avg_daily_ret = float(df["Daily Return"].mean())
    volatility = float(df["Daily Return"].std())
    ann_volatility = volatility * math.sqrt(252)
    ann_return = avg_daily_ret * 252
    sharpe = (ann_return - risk_free_rate) / ann_volatility if ann_volatility > 0 else 0
    skewness = float(df["Daily Return"].skew())
    kurtosis = float(df["Daily Return"].kurtosis())
    max_close = float(df["Adj Close"].max())
    min_close = float(df["Adj Close"].min())

    metrics = {
        "df": df,
        "latest_close": latest_close,
        "total_return": total_return,
        "avg_daily_ret": avg_daily_ret,
        "volatility": volatility,
        "ann_volatility": ann_volatility,
        "ann_return": ann_return,
        "sharpe": sharpe,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "max_close": max_close,
        "min_close": min_close,
    }
    return metrics

# -- Main logic -------------------------------------------
if tickers:
    st.subheader(f"Analyzing: {', '.join(tickers)}")
    
    # Load data for all tickers and benchmark
    all_data = {}
    failed_tickers = []
    
    # First, download benchmark (S&P 500)
    try:
        benchmark_df = load_data("^GSPC", start_date, end_date)
        if benchmark_df.empty:
            st.warning("No data found for S&P 500 benchmark (^GSPC). Benchmark comparison will not be available.")
            benchmark_df = None
        else:
            # Flatten any multi-level columns
            if isinstance(benchmark_df.columns, pd.MultiIndex):
                benchmark_df.columns = benchmark_df.columns.get_level_values(0)
            
            # Rename Close to Adj Close if Adj Close doesn't exist (for consistency)
            if "Adj Close" not in benchmark_df.columns and "Close" in benchmark_df.columns:
                benchmark_df["Adj Close"] = benchmark_df["Close"]
            
            # Check that required columns exist
            if "Adj Close" not in benchmark_df.columns:
                st.warning("S&P 500 benchmark data missing required price data. Benchmark comparison will not be available.")
                benchmark_df = None
            else:
                all_data["^GSPC"] = benchmark_df
    except Exception as e:
        st.warning(f"Failed to download S&P 500 benchmark (^GSPC): {e}. Benchmark comparison will not be available.")
        benchmark_df = None
    
    # Download each selected ticker
    for ticker in tickers:
        try:
            df = load_data(ticker, start_date, end_date)
        except Exception as e:
            failed_tickers.append((ticker, str(e)))
            continue

        if df.empty:
            failed_tickers.append((ticker, "No data found for this ticker symbol"))
            continue

        # Check for minimum data points
        min_data_points = 200  # Allow for market closures and holidays
        if len(df) < min_data_points:
            failed_tickers.append((ticker, f"Insufficient data: only {len(df)} trading days available (minimum {min_data_points} required)"))
            continue

        # Flatten any multi-level columns that yfinance sometimes returns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Rename Close to Adj Close if Adj Close doesn't exist (for consistency)
        if "Adj Close" not in df.columns and "Close" in df.columns:
            df["Adj Close"] = df["Close"]
        
        # Check that required columns exist
        if "Adj Close" not in df.columns or "Volume" not in df.columns:
            failed_tickers.append((ticker, "Missing required price or volume data"))
            continue
        
        # Store only the Adj Close price and Volume for analysis
        all_data[ticker] = df[["Adj Close", "Volume"]].copy()

    # Display errors if any tickers failed
    if failed_tickers:
        st.error("Failed to load data for the following ticker(s):")
        for ticker, reason in failed_tickers:
            st.error(f"  • {ticker}: {reason}")
        
        # Stop if all tickers failed
        if len(failed_tickers) == len(tickers):
            st.stop()
        else:
            st.info(f"Proceeding with {len(tickers) - len(failed_tickers)} valid ticker(s).")
    
    # Update tickers list to only include successfully loaded tickers
    tickers = [t for t in tickers if t not in [ft[0] for ft in failed_tickers]]
    
    if not tickers:
        st.stop()

    # Compute metrics for each ticker and benchmark
    ticker_metrics = {}
    for ticker, df in all_data.items():
        if ma_window > len(df):
            st.warning(
                f"({ticker}) The selected {ma_window}-day window is longer than the "
                f"available data ({len(df)} trading days). The moving average "
                "line won't appear — try a shorter window or a wider date range."
            )
        ticker_metrics[ticker] = compute_metrics(df, ticker, ma_window, vol_window, risk_free_rate)

    # -- Summary Statistics Table --
    st.subheader("Summary Statistics")
    
    # Prepare data for the summary table
    summary_data = []
    for ticker in tickers + (["^GSPC"] if "^GSPC" in ticker_metrics else []):
        metrics = ticker_metrics[ticker]
        df = metrics["df"]
        returns_clean = df["Daily Return"].dropna()
        
        summary_data.append({
            "Ticker": ticker,
            "Annualized Mean Return": f"{metrics['ann_return']:.2%}",
            "Annualized Volatility": f"{metrics['ann_volatility']:.2%}",
            "Skewness": f"{metrics['skewness']:.3f}",
            "Kurtosis": f"{metrics['kurtosis']:.3f}",
            "Min Daily Return": f"{returns_clean.min():.2%}",
            "Max Daily Return": f"{returns_clean.max():.2%}"
        })
    
    # Display the summary table
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.divider()

    # -- Combined Price Chart --
    st.subheader("Combined Price Chart")
    
    # Multi-select widget for stock selection
    selected_tickers = st.multiselect(
        "Select stocks to display:",
        options=tickers,
        default=tickers,  # All selected by default
        key="price_chart_selector"
    )
    
    if selected_tickers:
        fig = go.Figure()
        
        # Add each selected stock to the chart
        for ticker in selected_tickers:
            df = all_data[ticker]
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df["Adj Close"],
                    mode="lines", name=ticker,
                    line=dict(width=2)
                )
            )
        
        # Add S&P 500 benchmark if available
        if "^GSPC" in all_data:
            benchmark_df = all_data["^GSPC"]
            fig.add_trace(
                go.Scatter(
                    x=benchmark_df.index, y=benchmark_df["Adj Close"],
                    mode="lines", name="S&P 500 (^GSPC)",
                    line=dict(width=2, dash="dot", color="gray")
                )
            )
        
        fig.update_layout(
            yaxis_title="Adjusted Close Price (USD)", 
            xaxis_title="Date",
            template="plotly_white", 
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please select at least one stock to display the chart.")

    st.divider()

    # -- Cumulative Wealth Index Chart --
    st.subheader("Cumulative Wealth Index ($10,000 Investment)")
    
    fig_wealth = go.Figure()
    initial_investment = 10000
    
    # Add individual stock wealth indices
    for ticker in tickers:
        df = all_data[ticker]
        # Calculate cumulative wealth: start with $10,000 and compound daily returns
        wealth_index = initial_investment * (1 + df["Daily Return"]).cumprod()
        fig_wealth.add_trace(
            go.Scatter(
                x=df.index, y=wealth_index,
                mode="lines", name=ticker,
                line=dict(width=2)
            )
        )
    
    # Add S&P 500 benchmark if available
    if "^GSPC" in all_data:
        benchmark_df = all_data["^GSPC"]
        benchmark_wealth = initial_investment * (1 + benchmark_df["Daily Return"]).cumprod()
        fig_wealth.add_trace(
            go.Scatter(
                x=benchmark_df.index, y=benchmark_wealth,
                mode="lines", name="S&P 500 (^GSPC)",
                line=dict(width=2, dash="dot", color="gray")
            )
        )
    
    # Add equal-weight portfolio
    if len(tickers) > 1:
        # Get all daily returns for the selected period
        portfolio_returns = pd.DataFrame()
        for ticker in tickers:
            portfolio_returns[ticker] = all_data[ticker]["Daily Return"]
        
        # Calculate equal-weight portfolio return (average of daily returns)
        portfolio_returns["Equal_Weight"] = portfolio_returns[tickers].mean(axis=1)
        
        # Calculate cumulative wealth for equal-weight portfolio
        portfolio_wealth = initial_investment * (1 + portfolio_returns["Equal_Weight"]).cumprod()
        
        fig_wealth.add_trace(
            go.Scatter(
                x=portfolio_returns.index, y=portfolio_wealth,
                mode="lines", name="Equal-Weight Portfolio",
                line=dict(width=3, dash="dash", color="black")
            )
        )
    
    fig_wealth.update_layout(
        yaxis_title="Portfolio Value ($)", 
        xaxis_title="Date",
        template="plotly_white", 
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    st.plotly_chart(fig_wealth, use_container_width=True)

    st.divider()

    # Display all tickers together
    for ticker in tickers:
        metrics = ticker_metrics[ticker]
        df = metrics["df"]

        # -- Key metrics section --
        st.subheader(f"{ticker} — Key Metrics")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Latest Close", f"${metrics['latest_close']:,.2f}")
        col2.metric("Total Return", f"{metrics['total_return']:.2%}")
        col3.metric("Annualized Return", f"{metrics['ann_return']:.2%}")
        col4.metric("Sharpe Ratio", f"{metrics['sharpe']:.2f}")

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Annualized Volatility (σ)", f"{metrics['ann_volatility']:.2%}")
        col6.metric("Skewness", f"{metrics['skewness']:.2f}")
        col7.metric("Excess Kurtosis", f"{metrics['kurtosis']:.2f}")
        col8.metric("Avg Daily Return", f"{metrics['avg_daily_ret']:.4%}")

        col9, col10, _, _ = st.columns(4)
        col9.metric("Period High", f"${metrics['max_close']:,.2f}")
        col10.metric("Period Low", f"${metrics['min_close']:,.2f}")

        st.divider()

        # -- Volume chart --
        st.subheader("Daily Trading Volume")

        fig_vol = go.Figure()
        fig_vol.add_trace(
            go.Bar(x=df.index, y=df["Volume"], name="Volume",
                   marker_color="steelblue", opacity=0.7)
        )
        fig_vol.update_layout(
            yaxis_title="Shares Traded", xaxis_title="Date",
            template="plotly_white", height=350
        )
        st.plotly_chart(fig_vol, use_container_width=True)

        # -- Daily returns distribution --
        st.subheader("Distribution of Daily Returns")

        returns_clean = df["Daily Return"].dropna()

        fig_hist = go.Figure()
        fig_hist.add_trace(
            go.Histogram(
                x=returns_clean, nbinsx=60,
                marker_color="mediumpurple", opacity=0.75,
                name="Daily Returns", histnorm="probability density"
            )
        )

        # Overlay a fitted normal distribution curve
        x_range = np.linspace(float(returns_clean.min()), float(returns_clean.max()), 200)
        mu = float(returns_clean.mean())
        sigma = float(returns_clean.std())
        fig_hist.add_trace(
            go.Scatter(
                x=x_range, y=stats.norm.pdf(x_range, mu, sigma),
                mode="lines", name="Normal Distribution",
                line=dict(color="red", width=2)
            )
        )

        fig_hist.update_layout(
            xaxis_title="Daily Return", yaxis_title="Density",
            template="plotly_white", height=350
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # Display normality test results
        jb_stat, jb_pvalue = stats.jarque_bera(returns_clean)
        st.caption(
            f"Jarque-Bera test: statistic = {jb_stat:.2f}, p-value = {jb_pvalue:.4f} — "
            f"{'Fail to reject normality (p > 0.05)' if jb_pvalue > 0.05 else 'Reject normality (p <= 0.05)'}"
        )

        # -- Daily returns over time --
        st.subheader("Daily Returns Over Time")

        fig_returns = go.Figure()
        fig_returns.add_trace(
            go.Scatter(
                x=df.index, y=df["Daily Return"],
                mode="lines", name="Daily Return",
                line=dict(color="orange", width=1)
            )
        )
        fig_returns.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_returns.update_layout(
            yaxis_title="Daily Return", yaxis_tickformat=".2%",
            xaxis_title="Date", template="plotly_white", height=350
        )
        st.plotly_chart(fig_returns, use_container_width=True)

        # -- Cumulative return chart --
        st.subheader("Cumulative Return Over Time")

        fig_cum = go.Figure()
        fig_cum.add_trace(
            go.Scatter(
                x=df.index, y=df["Cumulative Return"],
                mode="lines", name="Cumulative Return",
                fill="tozeroy", line=dict(color="teal")
            )
        )
        fig_cum.update_layout(
            yaxis_title="Cumulative Return", yaxis_tickformat=".0%",
            xaxis_title="Date", template="plotly_white", height=400
        )
        st.plotly_chart(fig_cum, use_container_width=True)

        # -- Rolling volatility chart --
        st.subheader("Rolling Annualized Volatility")

        fig_roll_vol = go.Figure()
        fig_roll_vol.add_trace(
            go.Scatter(
                x=df.index, y=df["Rolling Volatility"],
                mode="lines", name=f"{vol_window}-Day Rolling Vol",
                line=dict(color="crimson", width=1.5)
            )
        )
        fig_roll_vol.update_layout(
            yaxis_title="Annualized Volatility", yaxis_tickformat=".0%",
            xaxis_title="Date", template="plotly_white", height=400
        )
        st.plotly_chart(fig_roll_vol, use_container_width=True)

        # -- Raw data (expandable) --
        with st.expander("View Raw Data"):
            st.dataframe(df.tail(60), use_container_width=True)

        st.divider()  # Separator between tickers

else:
    st.info("Enter stock tickers in the sidebar to get started.")
