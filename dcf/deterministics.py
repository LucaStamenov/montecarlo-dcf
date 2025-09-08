import numpy as np


def fcff_path(revenue0, growths, ebit_margins, tax_rate, da_pct, capex_pct, nwc_pct):
    rev = [revenue0]
    for g in growths:
        rev.append(rev[-1] * (1 + g))
    rev = np.array(rev[1:], dtype=np.float64)

    ebit = rev * ebit_margins
    nopat = ebit * (1 - tax_rate)
    da = rev * da_pct
    capex = rev * capex_pct
    nwc = rev * nwc_pct

    d_nwc = np.diff(np.concatenate([[revenue0 * nwc_pct], nwc]))
    fcff = nopat + da - capex - d_nwc
    return fcff
