import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="Monte Carlo Stock Simulator", layout="wide")
st.title("📈 Monte Carlo Simulation for Stock Price Forecasting")

# Sidebar Inputs
st.sidebar.header("Simulation Inputs")
tickers = st.sidebar.text_input("Stock Tickers (comma-separated)", value="AAPL,MSFT").upper().split(',')
days = st.sidebar.slider("Simulation Time Horizon (days)", 30, 365, 252)
sims = st.sidebar.slider("Number of Simulations", 100, 10000, 1000, step=100)
vol_source = st.sidebar.selectbox("Volatility Source", ["Manual", "Historical (21d)", "Implied (ATM)"], index=2)
manual_mu = st.sidebar.number_input("Expected Annual Return (μ, if Manual)", value=0.10)
manual_sigma = st.sidebar.number_input("Expected Volatility (σ, if Manual)", value=0.30)

# Market condition toggle
adjust_for_market = st.sidebar.checkbox("Adjust Expected Return Using Market Scenario")
if adjust_for_market:
    market_return = st.sidebar.number_input("Assumed Market Return over Period", value=-0.10)
    risk_free_rate = st.sidebar.number_input("Risk-Free Rate", value=0.00)

@st.cache_data

def fetch_data(ticker):
    return yf.download(ticker, period="6mo")

def estimate_historical_vol(data):
    returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
    return returns.std() * np.sqrt(252)

def get_implied_vol(ticker, current_price):
    try:
        opt = yf.Ticker(ticker).option_chain()
        calls = opt.calls
        atm_strike = calls['strike'].sub(current_price).abs().idxmin()
        return calls.loc[atm_strike, 'impliedVolatility']
    except Exception:
        return None

def get_beta(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info.get("beta", 1.0)  # default to 1.0 if missing
    except Exception:
        return 1.0

def run_simulation(S0, mu, sigma, days, sims):
    dt = 1/252
    paths = np.zeros((days, sims))
    paths[0] = S0
    for t in range(1, days):
        z = np.random.standard_normal(sims)
        paths[t] = paths[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    return paths

for ticker in tickers:
    with st.expander(f"📊 Simulation for {ticker.strip()}"):
        try:
            data = fetch_data(ticker.strip())
            current_price = float(data["Close"].iloc[-1])
            st.write(f"### Current Price of {ticker.strip()}: ${current_price:.2f}")

            if vol_source == "Manual":
                mu = manual_mu
                sigma = manual_sigma
            elif vol_source == "Historical (21d)":
                sigma = estimate_historical_vol(data)
                mu = manual_mu
                st.write(f"Estimated Historical Volatility (21d): {sigma:.2%}")
            elif vol_source == "Implied (ATM)":
                sigma = get_implied_vol(ticker.strip(), current_price)
                mu = manual_mu
                if sigma:
                    st.write(f"ATM Implied Volatility: {sigma:.2%}")
                else:
                    st.warning("Could not retrieve implied volatility. Falling back to manual volatility.")
                    sigma = manual_sigma

            if adjust_for_market:
                beta = get_beta(ticker.strip())
                mu = risk_free_rate + beta * (market_return - risk_free_rate)
                st.write(f"Adjusted μ using CAPM (β = {beta:.2f}): {mu:.2%}")

            # Run simulation
            price_paths = run_simulation(current_price, mu, sigma, days, sims)
            ending_prices = price_paths[-1]

            # Plotting paths
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(price_paths, lw=0.8, alpha=0.6)
            ax.set_title(f"Monte Carlo Simulation of {ticker.strip()} Over {days} Days ({sims} Paths)")
            ax.set_xlabel("Days")
            ax.set_ylabel("Price")
            st.pyplot(fig)

            # Probability metrics
            prob_above = np.mean(ending_prices > current_price)
            mean_price = np.mean(ending_prices)
            median_price = np.median(ending_prices)
            percentile_25 = np.percentile(ending_prices, 25)
            percentile_75 = np.percentile(ending_prices, 75)

            st.metric("Probability Ends Above Current Price", f"{prob_above * 100:.2f}%")
            st.metric("Mean Ending Price", f"${mean_price:.2f}")
            st.metric("Median Ending Price", f"${median_price:.2f}")
            st.metric("25th Percentile Price", f"${percentile_25:.2f}")
            st.metric("75th Percentile Price", f"${percentile_75:.2f}")

            # Histogram
            fig2, ax2 = plt.subplots()
            ax2.hist(ending_prices, bins=50, alpha=0.7)
            ax2.axvline(current_price, color='r', linestyle='--', label='Current Price')
            ax2.set_title("Histogram of Simulated Final Prices")
            ax2.legend()
            st.pyplot(fig2)

        except Exception as e:
            st.error(f"Failed to retrieve or simulate data for {ticker.strip()}: {e}")
