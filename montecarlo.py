import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(page_title="Monte Carlo Dashboard", layout="wide")
st.title("📊 Monte Carlo Simulation Dashboard")

# === Sidebar Inputs ===
with st.sidebar:
    st.header("Global Settings")
    n_days = st.slider("Simulation Days", 30, 365, 252)
    n_simulations = st.slider("Number of Simulations", 100, 5000, 1000)
    use_implied_vol = st.radio("Volatility Source", ["Implied", "Manual"], index=0)
    manual_sigma = st.number_input("Manual Volatility (e.g. 0.25)", value=0.25)
    show_debug = st.checkbox("Show Debug Info")
    use_capm = st.checkbox("Use CAPM for Expected Return")
    if use_capm:
        market_return = st.number_input("Expected Market Return", value=0.08)
        risk_free_rate = st.number_input("Risk-Free Rate", value=0.03)

# === Utility Functions ===
@st.cache_data

def fetch_data(ticker):
    try:
        df = yf.download(ticker, period="6mo", progress=False)
        if df.empty:
            st.warning(f"⚠️ Data for {ticker} returned empty. This might be due to rate limits or a bad connection.")
            return None
        return df
    except Exception as e:
        st.error(f"❌ Failed to fetch {ticker}: {e}")
        return None

def fetch_full_history(ticker):
    try:
        df = yf.download(ticker, period="10y", progress=False)
        if df.empty:
            st.warning(f"⚠️ Long-term data for {ticker} returned empty. May be rate-limited or a data issue.")
            return None
        return df
    except Exception as e:
        st.error(f"❌ Failed to fetch full history for {ticker}: {e}")
        return None

def get_implied_volatility(ticker):
    try:
        stock = yf.Ticker(ticker)
        options = stock.options
        if not options:
            return None
        expiry = options[0]
        calls = stock.option_chain(expiry).calls
        spot = stock.history(period="1d")["Close"].iloc[-1]
        calls["diff"] = np.abs(calls["strike"] - spot)
        iv = calls.sort_values("diff").iloc[0]["impliedVolatility"]
        return iv
    except:
        return None

def get_beta(ticker):
    try:
        return yf.Ticker(ticker).info.get("beta", 1.0)
    except:
        return 1.0

def simulate(S0, mu, sigma, n_days, n_simulations):
    dt = 1/252
    paths = np.zeros((n_days, n_simulations))
    paths[0] = S0
    for t in range(1, n_days):
        z = np.random.standard_normal(n_simulations)
        paths[t] = paths[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    return paths

def get_historical_returns(data):
    try:
        today = data.index[-1]
        one_year_ago = today - timedelta(days=365)
        five_years_ago = today - timedelta(days=5*365)

        one_year_return = None
        five_year_cagr = None

        if one_year_ago in data.index:
            one_year_return = (data.loc[today, "Close"] / data.loc[one_year_ago, "Close"]) - 1
        else:
            close_one_year = data[data.index <= one_year_ago]["Close"].iloc[-1]
            one_year_return = (data["Close"].iloc[-1] / close_one_year) - 1

        if five_years_ago in data.index:
            five_year_cagr = (data.loc[today, "Close"] / data.loc[five_years_ago, "Close"]) ** (1/5) - 1
        else:
            close_five_year = data[data.index <= five_years_ago]["Close"].iloc[-1]
            five_year_cagr = (data["Close"].iloc[-1] / close_five_year) ** (1/5) - 1

        return float(one_year_return), float(five_year_cagr)
    except:
        return None, None

# === Tabs ===
tab1, tab2 = st.tabs(["📈 Single Stock Simulation", "📊 Portfolio Simulation"])

# === Single Stock Simulation ===
with tab1:
    st.header("Single Stock Simulation")
    tickers = st.text_input("Enter tickers (comma-separated)", "AAPL, MSFT")

    if tickers:
        ticker_list = [t.strip().upper() for t in tickers.split(",")]
        single_tabs = st.tabs(ticker_list)

        for i, ticker in enumerate(ticker_list):
            with single_tabs[i]:
                try:
                    data = fetch_data(ticker)
                    full_data = fetch_full_history(ticker)
                    if data is None or data.empty:
                        st.warning(f"No data for {ticker}")
                        continue

                    current_price = float(data["Close"].iloc[-1])
                    st.subheader(f"{ticker} — Current Price: ${current_price:.2f}")

                    sigma = get_implied_volatility(ticker) if use_implied_vol == "Implied" else manual_sigma
                    fallback_sigma = 0.20
                    if sigma is None or sigma < 0.05:
                        sigma = fallback_sigma
                        if show_debug:
                            st.warning(f"{ticker}: Using fallback volatility {sigma:.2f}")

                    beta = get_beta(ticker)
                    mu = (risk_free_rate + beta * (market_return - risk_free_rate)) if use_capm else 0.0

                    paths = simulate(current_price, mu, sigma, n_days, n_simulations)
                    ending_prices = paths[-1]

                    mean_price = np.mean(ending_prices)
                    median_price = np.median(ending_prices)
                    p25 = np.percentile(ending_prices, 25)
                    p75 = np.percentile(ending_prices, 75)
                    prob_up = np.mean(ending_prices > current_price)

                    one_year_return, five_year_cagr = get_historical_returns(full_data)

                    col1, col2 = st.columns([2, 1])
                    with col2:
                        st.metric("Mean Ending Price", f"${mean_price:.2f}")
                        st.metric("Median", f"${median_price:.2f}")
                        st.metric("25th Percentile", f"${p25:.2f}")
                        st.metric("75th Percentile", f"${p75:.2f}")
                        st.metric("Probability > Current", f"{prob_up:.2%}")
                        st.metric("1Y Historical Return", f"{float(one_year_return):.2%}" if one_year_return is not None else "N/A")
                        st.metric("5Y CAGR", f"{float(five_year_cagr):.2%}" if five_year_cagr is not None else "N/A")
                    with col1:
                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.plot(paths, linewidth=0.7)
                        ax.set_title(f"Monte Carlo Simulation for {ticker}")
                        ax.set_xlabel("Days")
                        ax.set_ylabel("Simulated Price")
                        st.pyplot(fig)

                except Exception as e:
                    st.error(f"Error processing {ticker}: {e}")

# === Portfolio Simulation ===
with tab2:
    st.header("Portfolio Simulation")
    st.markdown("### Portfolio Composition")
    portfolio_df = st.data_editor(
    pd.DataFrame({"Ticker": ["AAPL", "MSFT", "TSLA", "FIXED_INCOME"], "Weight": [0.3, 0.3, 0.2, 0.2], "Yield": [None, None, None, 0.05]}),
    num_rows="dynamic",
    use_container_width=True,
    key="portfolio_input"
)
    ticker_list = portfolio_df["Ticker"].str.upper().tolist()
    weight_list = portfolio_df["Weight"].tolist()

    if len(ticker_list) != len(weight_list):
        st.error("Number of tickers and weights must match.")
    elif not np.isclose(sum(weight_list), 1.0):
        st.error("Weights must sum to 1.0")
    else:
        all_paths = []
        all_data = []
        fallback_sigma = 0.20

        for i, ticker in enumerate(ticker_list):
            try:
                if ticker == "FIXED_INCOME":
                    yield_value = portfolio_df.loc[i, "Yield"] or 0.04
                    sim = np.full((n_days, n_simulations), 100 * (1 + yield_value) ** (np.arange(n_days) / 252).reshape(-1, 1))
                    all_paths.append(sim)
                    all_data.append(None)
                    continue
                data = fetch_data(ticker)
                full_data = fetch_full_history(ticker)
                if data is None or data.empty:
                    st.warning(f"No data for {ticker}")
                    continue
                current_price = float(data["Close"].iloc[-1])

                sigma = get_implied_volatility(ticker) if use_implied_vol == "Implied" else manual_sigma
                if sigma is None or sigma < 0.05:
                    sigma = fallback_sigma
                    if show_debug:
                        st.warning(f"{ticker}: Using fallback volatility {sigma:.2f}")

                beta = get_beta(ticker)
                mu = (risk_free_rate + beta * (market_return - risk_free_rate)) if use_capm else 0.0

                paths = simulate(current_price, mu, sigma, n_days, n_simulations)
                all_paths.append(paths)
                all_data.append(full_data)

            except Exception as e:
                st.error(f"Error processing {ticker}: {e}")

        if all_paths:
            normalized_paths = [paths / paths[0] for paths in all_paths]
            weighted_paths = [w * norm for w, norm in zip(weight_list, normalized_paths)]
            portfolio_paths = sum(weighted_paths) * 100

            ending_values = portfolio_paths[-1]
            mean_end = np.mean(ending_values)
            median_end = np.median(ending_values)
            p5 = np.percentile(ending_values, 5)
            p95 = np.percentile(ending_values, 95)
            prob_above_100 = np.mean(ending_values > 100)
            std_dev = np.std(ending_values)
            sharpe_ratio = (mean_end - 100) / std_dev if std_dev > 0 else 0
            downside_returns = ending_values[ending_values < 100]
            downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1
            sortino_ratio = (mean_end - 100) / downside_std if downside_std > 0 else 0

            try:
                min_length = min([len(d) for d in all_data])
                aligned_data = [d["Close"].iloc[-min_length:].pct_change().dropna() for d in all_data if d is not None]
                combined_returns = sum(w * r for w, d, r in zip(weight_list, all_data, aligned_data) if d is not None)

                prod_1y = (1 + combined_returns[-252:]).prod()
                total_return_1y = float(prod_1y.item()) if hasattr(prod_1y, "item") else float(prod_1y - 1)

                prod_5y = (1 + combined_returns[-1260:]).prod()
                cagr_5y = float(prod_5y ** (1/5) - 1)
            except:
                total_return_1y = None
                cagr_5y = None

            col1, col2 = st.columns([2, 1])
            with col1:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(portfolio_paths, linewidth=0.5, alpha=0.7)
                ax.set_title("Simulated Portfolio Value Over Time")
                ax.set_xlabel("Days")
                ax.set_ylabel("Portfolio Value ($)")
                st.pyplot(fig)
            with col2:
                st.subheader("📈 Portfolio Summary")
                st.metric("Mean Ending Value", f"${mean_end:.2f}")
                st.metric("Median Ending Value", f"${median_end:.2f}")
                st.metric("5th Percentile", f"${p5:.2f}")
                st.metric("95th Percentile", f"${p95:.2f}")
                st.metric("Probability > $100", f"{prob_above_100:.2%}")
                st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                st.metric("Sortino Ratio", f"{sortino_ratio:.2f}")
                st.metric("1Y Historical Return", f"{float(total_return_1y):.2%}" if total_return_1y is not None else "N/A")
                st.metric("5Y CAGR", f"{float(cagr_5y):.2%}" if cagr_5y is not None else "N/A")
