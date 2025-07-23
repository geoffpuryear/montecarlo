import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Streamlit page config
st.set_page_config(page_title="Monte Carlo Stock Simulator", layout="wide")

st.title("📈 Monte Carlo Stock Price Simulator")

# Sidebar inputs
tickers = st.sidebar.text_input("Enter stock tickers (comma-separated)", value="AAPL,MSFT").upper().split(",")
start_date = st.sidebar.date_input("Start date for historical data", value=datetime(2020, 1, 1))
sim_days = st.sidebar.slider("Days to simulate", 30, 365, 252)
num_simulations = st.sidebar.slider("Number of simulations", 100, 5000, 1000, step=100)
recession = st.sidebar.checkbox("🔻 Recession scenario (lower returns, higher volatility)")

# Simulation function
def run_simulation(ticker, sims, days, recession_mode):
    df = yf.download(ticker, start=start_date)
    if df.empty:
        st.warning(f"Could not retrieve data for {ticker}")
        return None

    df = df['Adj Close']
    log_returns = np.log(df / df.shift(1)).dropna()
    mu = log_returns.mean()
    sigma = log_returns.std()

    if recession_mode:
        mu *= 0.5
        sigma *= 1.25

    last_price = df.iloc[-1]
    dt = 1
    price_paths = np.zeros((days, sims))
    price_paths[0] = last_price

    for t in range(1, days):
        z = np.random.standard_normal(sims)
        price_paths[t] = price_paths[t - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)

    return price_paths, last_price, mu, sigma

# Run simulation for each ticker
for ticker in tickers:
    ticker = ticker.strip()
    st.subheader(f"📊 Simulation for {ticker}")
    
    result = run_simulation(ticker, num_simulations, sim_days, recession)
    if result is None:
        continue

    price_paths, last_price, mu, sigma = result

    # Plot simulations
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(price_paths, linewidth=0.5, alpha=0.6)
    ax.set_title(f"{ticker} - Monte Carlo Simulation ({num_simulations} runs)")
    ax.set_ylabel("Price")
    ax.set_xlabel("Days")
    ax.axhline(y=last_price, color='black', linestyle='--', linewidth=1, label="Starting Price")
    ax.legend()
    st.pyplot(fig)

    # Summary stats
    final_prices = price_paths[-1]
    st.write(f"**Final price distribution for {ticker}:**")
    st.write(f"Expected mean: ${final_prices.mean():.2f}")
    st.write(f"5th percentile: ${np.percentile(final_prices, 5):.2f}")
    st.write(f"95th percentile: ${np.percentile(final_prices, 95):.2f}")
    st.write(f"Estimated daily return: {mu:.4f}")
    st.write(f"Estimated daily volatility: {sigma:.4f}")
