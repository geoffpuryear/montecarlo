import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="Monte Carlo Simulator", layout="wide")
st.title("📈 Monte Carlo Stock Price Simulator")

# === Sidebar Controls ===
with st.sidebar:
    st.header("Simulation Settings")
    tickers = st.text_input("Enter tickers (comma-separated)", "AAPL, MSFT")
    n_days = st.slider("Simulation Days", 30, 365, 252)
    n_simulations = st.slider("Number of Simulations", 100, 5000, 1000)

    market_adjust = st.checkbox("Use CAPM for Expected Return")
    if market_adjust:
        market_return = st.number_input("Expected Market Return", value=0.08)
        risk_free_rate = st.number_input("Risk-Free Rate", value=0.03)

    use_implied_vol = st.radio("Volatility Source", ["Implied", "Manual"], index=0)
    manual_sigma = st.number_input("Manual Volatility (e.g. 0.25)", value=0.25)

    show_debug = st.checkbox("Show Debug Info")

# === Data Utilities ===
@st.cache_data
def fetch_data(ticker):
    return yf.download(ticker, period="6mo", progress=False)

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
        paths[t] = paths[t-1] * np.exp((mu - 0.5 * sigma**2)*dt + sigma*np.sqrt(dt)*z)
    return paths

# === Main Dashboard Logic ===
if tickers:
    ticker_list = [t.strip().upper() for t in tickers.split(",")]
    tabs = st.tabs(ticker_list)

    for i, ticker in enumerate(ticker_list):
        with tabs[i]:
            try:
                data = fetch_data(ticker)
                current_price = float(data["Close"].iloc[-1])
                st.subheader(f"{ticker} — Current Price: ${current_price:.2f}")

                # === Volatility logic
                sigma = get_implied_volatility(ticker) if use_implied_vol == "Implied" else manual_sigma
                if sigma is None:
                    st.warning(f"Could not retrieve implied volatility for {ticker}. Using manual or fallback value.")
                    sigma = manual_sigma
                if sigma < 0.05:
                    st.warning(f"Implied volatility for {ticker} is suspiciously low. Setting fallback value of 0.20.")
                    sigma = 0.20  # fallback volatility

                # === Return logic
                beta = get_beta(ticker)
                mu = (risk_free_rate + beta * (market_return - risk_free_rate)) if market_adjust else 0.0

                if show_debug:
                    st.code(f"mu: {mu:.4f}, sigma: {sigma:.4f}, beta: {beta:.2f}")

                # === Simulate
                paths = simulate(current_price, mu, sigma, n_days, n_simulations)
                ending_prices = paths[-1]

                mean_price = np.mean(ending_prices)
                median_price = np.median(ending_prices)
                p25 = np.percentile(ending_prices, 25)
                p75 = np.percentile(ending_prices, 75)
                prob_up = np.mean(ending_prices > current_price)

                col1, col2, col3 = st.columns(3)
                col1.metric("📈 Mean Ending Price", f"${mean_price:.2f}")
                col2.metric("🔻 25th Percentile", f"${p25:.2f}")
                col3.metric("🔺 75th Percentile", f"${p75:.2f}")

                col4, col5 = st.columns(2)
                col4.metric("📊 Median", f"${median_price:.2f}")
                col5.metric("💡 Probability > Current", f"{prob_up:.2%}")

                # === Plot
                fig, ax = plt.subplots()
                ax.plot(paths, linewidth=0.7)
                ax.set_title(f"Monte Carlo Simulation for {ticker}")
                ax.set_xlabel("Days")
                ax.set_ylabel("Simulated Price")
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error processing {ticker}: {e}")
