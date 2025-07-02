import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd

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
        return yf.download(ticker, period="6mo", progress=False)
    except:
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

                    col1, col2 = st.columns([2, 1])
                    with col2:
                        st.metric("Mean Ending Price", f"${mean_price:.2f}")
                        st.metric("Median", f"${median_price:.2f}")
                        st.metric("25th Percentile", f"${p25:.2f}")
                        st.metric("75th Percentile", f"${p75:.2f}")
                        st.metric("Probability > Current", f"{prob_up:.2%}")
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
    tickers_input = st.text_input("Enter tickers for portfolio (comma-separated)", "AAPL, MSFT, TSLA")
    weights_input = st.text_input("Enter weights (comma-separated, must sum to 1)", "0.4, 0.4, 0.2")

    ticker_list = [t.strip().upper() for t in tickers_input.split(",")]
    weight_list = [float(w.strip()) for w in weights_input.split(",")]

    if len(ticker_list) != len(weight_list):
        st.error("Number of tickers and weights must match.")
    elif not np.isclose(sum(weight_list), 1.0):
        st.error("Weights must sum to 1.0")
    else:
        all_paths = []
        fallback_sigma = 0.20

        for ticker in ticker_list:
            try:
                data = fetch_data(ticker)
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

            if show_debug:
                st.write("Tickers:", ticker_list)
                st.write("Weights:", weight_list)
                st.write("Volatilities:", [get_implied_volatility(t) for t in ticker_list])
