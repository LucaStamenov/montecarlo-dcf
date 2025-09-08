import numpy as np

from .deterministics import fcff_path


def _discount_cashflows(cfs, wacc):
    t = np.arange(1, len(cfs) + 1)
    return np.sum(cfs / (1 + wacc) ** t)


def _terminal_value(last_rev, ebit_margin, tax_rate, da_pct, capex_pct, nwc_pct, wacc, g):
    ebit = last_rev * ebit_margin
    nopat = ebit * (1 - tax_rate)
    da = last_rev * da_pct
    capex = last_rev * capex_pct
    d_nwc = last_rev * nwc_pct * g
    fcff_next = nopat + da - capex - d_nwc
    return fcff_next * (1 + g) / (wacc - g)


def run_monte_carlo(assump, n_sims=10_000, seed=42):
    rng = np.random.default_rng(seed)

    growths = rng.normal(assump.rev_growth_mean, 0.04, size=(n_sims, assump.years))
    ebit_margins = rng.normal(assump.ebit_margin_mean, 0.03, size=(n_sims, assump.years))
    waccs = rng.normal(assump.wacc, 0.01, size=n_sims)
    gs = np.clip(rng.normal(assump.term_g, 0.005, size=n_sims), -0.01, 0.04)

    values = np.empty(n_sims, dtype=np.float64)
    for i in range(n_sims):
        fcff = fcff_path(
            assump.revenue0,
            growths[i],
            ebit_margins[i],
            assump.tax_rate,
            assump.da_pct_sales,
            assump.capex_pct_sales,
            assump.nwc_pct_sales,
        )
        pv_fcff = _discount_cashflows(fcff, waccs[i])

        last_rev = assump.revenue0 * np.prod(1 + growths[i])
        tv = _terminal_value(
            last_rev,
            ebit_margins[i, -1],
            assump.tax_rate,
            assump.da_pct_sales,
            assump.capex_pct_sales,
            assump.nwc_pct_sales,
            waccs[i],
            gs[i],
        )
        pv_tv = tv / (1 + waccs[i]) ** assump.years

        ev = pv_fcff + pv_tv
        eq = ev - assump.net_debt
        values[i] = eq / assump.shares_out

    return values
