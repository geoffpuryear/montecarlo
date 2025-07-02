import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd

st.title("Monte Carlo Stock Price Simulator")

tickers = st.text_input("Enter tickers (e.g. AAPL, MSFT, JPM):")

n_days = st.slider("Simulation Days", 30, 365, 252)
n_simulations = st.slider("Number of Simulations", 100, 5000, 1000)

market_adjust = st.checkbox("Adjust expected return using CAPM?")
if market_adjust:
    market_return = st.number_input("Expected Market Return", value=0.0)
    risk_free_rate = st.number_input("Risk-Free Rate", value=0.0)

use_implied_vol = st.radio("Volatility Source", ["Implied", "Manual"], index=0)
manual_sigma = st.number_input("Manual Volatility (e.g. 0.25)", value=0.25)

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

if tickers:
    ticker_list = [t.strip().upper() for t in tickers.split(",")]
    tabs = st.tabs(ticker_list)

    for i, ticker in enumerate(ticker_list):
        with tabs[i]:
            try:
                data = fetch_data(ticker)
                current_price = float(data["Close"].iloc[-1])
                st.subheader(f"{ticker}")
                st.metric("Current Price", f"${current_price:.2f}")

                sigma = get_implied_volatility(ticker) if use_implied_vol == "Implied" else manual_sigma
                if sigma is None or sigma < 1e-5:
                    st.warning("Implied volatility unavailable or too small. Using manual input.")
                    sigma = manual_sigma

                beta = get_beta(ticker)
                mu = (risk_free_rate + beta * (market_return - risk_free_rate)) if market_adjust else 0.0

                st.write(f"Volatility: {sigma:.2%}")
                if market_adjust:
                    st.write(f"Expected return via CAPM: {mu:.2%} (Beta = {beta:.2f})")

                paths = simulate(current_price, mu, sigma, n_days, n_simulations)
                ending_prices = paths[-1]
                prob_up = np.mean(ending_prices > current_price)

                st.metric("Probability > Current Price", f"{prob_up:.2%}")
                st.metric("Mean Ending Price", f"${np.mean(ending_prices):.2f}")
                st.metric("Median Ending Price", f"${np.median(ending_prices):.2f}")
                st.metric("25th Percentile", f"${np.percentile(ending_prices, 25):.2f}")
                st.metric("75th Percentile", f"${np.percentile(ending_prices, 75):.2f}")

                fig, ax = plt.subplots()
                ax.plot(paths)
                ax.set_title(f"Monte Carlo Simulation: {ticker}")
                ax.set_xlabel("Days")
                ax.set_ylabel("Price")
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error with {ticker}: {e}")
