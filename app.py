import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st # Import streamlit!
import yfinance as yf


def load_from_ticker(ticker: str):
    info = yf.Ticker(ticker).info
    return {
        "initial_rev": info["totalRevenue"] / 1e6,  # € m
        "ebit_margin": info["ebitMargin"],
        "tax": 0.25,  # fallback
        # …
    }

def download_to_excel(series: pd.Series, path="dcf_simulations.xlsx"):
    series.to_frame("Intrinsic value (€ m)")\
          .to_excel(path, index=False)
    
# --- DCF Calculation Function (COPY THIS ENTIRE FUNCTION FROM YOUR NOTEBOOK) ---
def calculate_dcf(initial_revenue, revenue_growth_rate, ebit_margin, tax_rate,
                  reinvestment_rate, wacc, terminal_growth_rate, forecast_years):
    """
    Calculates the intrinsic value of a company using a Discounted Cash Flow (DCF) model.
    Includes explicit forecast period and Terminal Value using Gordon Growth Model.
    """

    # Initialize lists to store annual projected values
    revenues = []
    ebits = []
    nopats = []
    reinvestments = []
    fcffs = []
    discount_factors = [] # Discount factor for each year: 1 / (1 + WACC)^year

    current_revenue = initial_revenue # Start with the initial revenue

    # --- Project Free Cash Flows for each forecast year ---
    for year in range(1, forecast_years + 1):
        # Calculate revenue for the current year
        if year > 1:
            current_revenue = revenues[-1] * (1 + revenue_growth_rate) # Grow from previous year's revenue

        # Calculate financial components
        current_ebit = current_revenue * ebit_margin
        current_nopat = current_ebit * (1 - tax_rate) # Net Operating Profit After Tax
        current_reinvestment = current_ebit * reinvestment_rate # Simplified reinvestment
        current_fcff = current_nopat - current_reinvestment

        # Store the calculated values for this year
        revenues.append(current_revenue)
        ebits.append(current_ebit)
        nopats.append(current_nopat)
        reinvestments.append(current_reinvestment)
        fcffs.append(current_fcff)

        # Calculate and store the discount factor for this year
        discount_factor = 1 / ((1 + wacc) ** year)
        discount_factors.append(discount_factor)

    # Convert lists to NumPy arrays for efficient calculations
    revenues = np.array(revenues)
    ebits = np.array(ebits)
    nopats = np.array(nopats)
    reinvestments = np.array(reinvestments)
    fcffs = np.array(fcffs)
    discount_factors = np.array(discount_factors)

    # --- Calculate Present Value of Forecast Period FCFFs ---
    pv_fcffs_individual = fcffs * discount_factors
    sum_pv_fcffs = np.sum(pv_fcffs_individual)

    # --- Calculate Terminal Value (TV) ---
    # FCFF in the year immediately following the forecast period (Year N+1)
    fcff_next_year = fcffs[-1] * (1 + terminal_growth_rate)

    # Gordon Growth Model for Terminal Value
    if wacc <= terminal_growth_rate:
        # In Streamlit, print statements might not show up clearly.
        # Consider st.warning("...") or a more robust error handling if this were a production app.
        terminal_value = 0 # Prevent division by zero or negative denominator
    else:
        terminal_value = fcff_next_year / (wacc - terminal_growth_rate)

    # --- Calculate Present Value of Terminal Value ---
    # The discount factor for the last forecast year (Year N)
    discount_factor_last_year = discount_factors[-1]
    pv_terminal_value = terminal_value * discount_factor_last_year

    # --- Calculate Total Intrinsic Value ---
    intrinsic_value = sum_pv_fcffs + pv_terminal_value

    return intrinsic_value, revenues, fcffs, discount_factors, pv_fcffs_individual, terminal_value, pv_terminal_value

# --- Streamlit App Layout and Logic (COPY ALL OF THIS) ---
st.set_page_config(layout="centered") # Optional: makes the app content centered
st.title("Monte Carlo DCF Valuation Tool 💰")
st.write("Simulate company intrinsic value under uncertainty using a Discounted Cash Flow (DCF) model.")

st.sidebar.header("Valuation Assumptions")

initial_revenue = st.sidebar.number_input(
    "Initial Revenue (€ Millions)",
    min_value=1.0, value=100.0, step=10.0, format="%.2f",
    help="Current annual revenue of the company."
)
ebit_margin = st.sidebar.slider(
    "EBIT Margin (%)",
    min_value=0.0, max_value=0.50, value=0.20, step=0.01, format="%.2f",
    help="Earnings Before Interest & Taxes as a percentage of revenue."
)
tax_rate = st.sidebar.slider(
    "Tax Rate (%)",
    min_value=0.0, max_value=0.50, value=0.25, step=0.01, format="%.2f",
    help="Effective corporate tax rate."
)
reinvestment_rate = st.sidebar.slider(
    "Reinvestment Rate (%)",
    min_value=0.0, max_value=0.50, value=0.15, step=0.01, format="%.2f",
    help="Simplified reinvestment rate (e.g., % of EBIT for CapEx, D&A, NWC changes)."
)

forecast_years = st.sidebar.slider(
    "Forecast Years",
    min_value=3, max_value=10, value=5, step=1,
    help="Number of years for explicit cash flow forecast."
)

st.sidebar.header("Monte Carlo Simulation Parameters")

num_simulations = st.sidebar.slider(
    "Number of Simulations",
    min_value=1000, max_value=100000, value=10000, step=1000,
    help="How many times to run the DCF model with random inputs."
)

st.sidebar.subheader("Revenue Growth Rate Distribution")
rev_growth_mean = st.sidebar.slider(
    "Mean Revenue Growth (%)",
    min_value=0.0, max_value=0.20, value=0.05, step=0.005, format="%.3f",
    help="Average annual revenue growth rate during the forecast period."
)
rev_growth_std = st.sidebar.slider(
    "Std Dev Revenue Growth (%)",
    min_value=0.0, max_value=0.10, value=0.01, step=0.001, format="%.3f",
    help="Volatility (standard deviation) of revenue growth."
)

st.sidebar.subheader("WACC Distribution")
wacc_mean = st.sidebar.slider(
    "Mean WACC (%)",
    min_value=0.05, max_value=0.20, value=0.10, step=0.005, format="%.3f",
    help="Average Weighted Average Cost of Capital."
)
wacc_std = st.sidebar.slider(
    "Std Dev WACC (%)",
    min_value=0.0, max_value=0.05, value=0.015, step=0.001, format="%.3f",
    help="Volatility (standard deviation) of WACC."
)

st.sidebar.subheader("Terminal Growth Rate")
terminal_growth_rate = st.sidebar.slider(
    "Terminal Growth Rate (%)",
    min_value=0.0, max_value=0.05, value=0.02, step=0.005, format="%.3f",
    help="Stable growth rate for cash flows beyond the forecast period."
)

# --- Perform Monte Carlo Simulation ---
simulated_values = []

with st.spinner(f"Running {num_simulations} simulations..."):
    for i in range(num_simulations):
        current_rev_growth_rate = np.random.normal(rev_growth_mean, rev_growth_std)
        current_wacc = np.random.normal(wacc_mean, wacc_std)

        if current_wacc <= terminal_growth_rate:
            current_wacc = terminal_growth_rate + 0.001

        intrinsic_value_this_sim, _, _, _, _, _, _ = calculate_dcf(
            initial_revenue=initial_revenue,
            revenue_growth_rate=current_rev_growth_rate,
            ebit_margin=ebit_margin,
            tax_rate=tax_rate,
            reinvestment_rate=reinvestment_rate,
            wacc=current_wacc,
            terminal_growth_rate=terminal_growth_rate,
            forecast_years=forecast_years
        )

        simulated_values.append(intrinsic_value_this_sim)

valuation_series = pd.Series(simulated_values)
st.success("Simulation complete! 🎉")

# --- Display Results ---
st.header("Simulation Results")

st.subheader("Summary Statistics")
st.dataframe(valuation_series.describe().round(2))

st.subheader("Key Valuation Metrics")
lower_bound = valuation_series.quantile(0.05)
median_value = valuation_series.quantile(0.50)
upper_bound = valuation_series.quantile(0.95)
mean_value = valuation_series.mean()

st.metric(label="Mean Intrinsic Value", value=f"€{mean_value:.2f}")
st.metric(label="Median Intrinsic Value", value=f"€{median_value:.2f}")
st.metric(label="90% Confidence Interval", value=f"€{lower_bound:.2f} - €{upper_bound:.2f}")

st.subheader("Distribution of Intrinsic Values")
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(valuation_series, bins=50, edgecolor='black', alpha=0.7, color='skyblue')

ax.axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label=f'Mean: €{mean_value:.2f}')
ax.axvline(median_value, color='green', linestyle='dashed', linewidth=2, label=f'Median: €{median_value:.2f}')
ax.axvline(lower_bound, color='purple', linestyle='dashed', linewidth=1, label=f'90% CI: €{lower_bound:.2f}')
ax.axvline(upper_bound, color='purple', linestyle='dashed', linewidth=1)

ax.set_title('Distribution of Intrinsic Values from Monte Carlo DCF Simulation')
ax.set_xlabel('Intrinsic Value (€)')
ax.set_ylabel('Frequency')
ax.grid(axis='y', alpha=0.75)
ax.legend()

st.pyplot(fig)
