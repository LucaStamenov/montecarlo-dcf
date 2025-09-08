from dataclasses import dataclass

import numpy as np


@dataclass
class Assumptions:
    years: int = 6
    revenue0: float = 1_000.0
    shares_out: float = 100.0
    net_debt: float = 0.0
    wacc: float = 0.09
    term_g: float = 0.02
    rev_growth_mean: float = 0.08
    ebit_margin_mean: float = 0.15
    tax_rate: float = 0.25
    da_pct_sales: float = 0.03
    capex_pct_sales: float = 0.05
    nwc_pct_sales: float = 0.10


@dataclass
class Results:
    values_per_share: np.ndarray
