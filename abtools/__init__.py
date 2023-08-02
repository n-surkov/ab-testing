from .data_generators import get_lognorm, get_sales
from .methods import (
    # Проверки без данных с предпериода
    check_ttest_naive, check_ttest_avg, check_bootstrap, check_poisson, check_delta_method,
    # Проверки с данными с предпериода
    check_cuped, check_normed, check_linearization, check_normed_delta_method
)
from .plots import plot_pvalue_ecdf
