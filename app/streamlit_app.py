import os
import sys

import numpy as np
import streamlit as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dcf.models import Assumptions  # noqa: E402
from dcf.simulate import run_monte_carlo  # noqa: E402

st.set_page_config(page_title="Monte Carlo DCF", page_icon="💹", layout="wide")
st.title("Monte Carlo DCF")
st.caption("Quick Monte Carlo valuation (10k–100k paths)")
st.write("✅ App is running. Use the sidebar and click **Run simulation**.")

with st.sidebar:
    st.subheader("Assumptions")
    n = st.slider("Simulations", 1_000, 100_000, 10_000, step=1_000)
    years = st.slider("Forecast years", 4, 10, 6)
    rev0 = st.number_input("Revenue (t=0)", value=1_000.0, step=50.0)
    shares = st.number_input("Diluted shares (mm)", value=100.0)
    net_debt = st.number_input("Net debt", value=0.0)
    wacc = st.slider("WACC", 0.05, 0.15, 0.09, 0.001)
    g = st.slider("Terminal growth g", -0.01, 0.05, 0.02, 0.001)
    with st.expander("Advanced"):
        ebit_m = st.slider("EBIT margin (mean)", 0.05, 0.30, 0.15, 0.005)
        rev_g = st.slider("Revenue growth (mean)", 0.00, 0.25, 0.08, 0.005)
        tax = st.slider("Tax rate", 0.00, 0.40, 0.25, 0.01)
        da = st.slider("D&A % sales", 0.00, 0.10, 0.03, 0.005)
        capex = st.slider("CapEx % sales", 0.00, 0.15, 0.05, 0.005)
        nwc = st.slider("NWC % sales", 0.00, 0.25, 0.10, 0.005)

assump = Assumptions(
    years=years,
    revenue0=rev0,
    shares_out=shares,
    net_debt=net_debt,
    wacc=wacc,
    term_g=g,
    ebit_margin_mean=ebit_m,
    rev_growth_mean=rev_g,
    tax_rate=tax,
    da_pct_sales=da,
    capex_pct_sales=capex,
    nwc_pct_sales=nwc,
)

if st.button("Run simulation", type="primary", use_container_width=True):
    vals = run_monte_carlo(assump, n_sims=n, seed=42)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean", f"{np.mean(vals):.2f}")
    col2.metric("Median", f"{np.median(vals):.2f}")
    col3.metric("P5", f"{np.percentile(vals, 5):.2f}")
    col4.metric("P95", f"{np.percentile(vals, 95):.2f}")

    st.bar_chart(vals)
