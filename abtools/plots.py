import matplotlib.pyplot as plt
import seaborn as sns
# from tqdm.notebook import tqdm
# from scipy import stats
from typing import List


def plot_pvalue_ecdf(pvalues: List[float], title: str = None) -> plt.figure:
    """
    Отрисовка распределения p-value

    При отсутствии эффекта:
    * распределение на левом графике должно быть равномерным
    * синия линия правго графика совпадать с чёрной

    Parameters
    ----------
    pvalues: List[float], список p-values для отрисовки
    title: Optional[str] (default=None), заголовок графика

    Returns
    -------
    fig: plt.figure, график
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    if title:
        plt.suptitle(title)

    sns.histplot(pvalues, ax=ax1, bins=20, stat='density')
    ax1.plot([0, 1], [1, 1], 'k--')
    ax1.set_xlabel('p-value')

    sns.ecdfplot(pvalues, ax=ax2)
    ax2.plot([0, 1], [0, 1], 'k--')
    ax2.set_xlabel('p-value')
    ax2.set_ylabel('Probability')
    ax2.grid()

    return fig


# def plot_two_graphs(
#         list_pvalue: List[float],
#         list_delta: List[float],
#         list_real_delta: List[float],
#         metric_name='mean user metric delta',
#         suptitle=None
# ):
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
#     if suptitle:
#         plt.suptitle(suptitle)
#
#     sns.ecdfplot(list_pvalue, ax=ax1)
#     ax1.set_title('Функция распределения p-value')
#     ax1.set_xlabel('p-value')
#     ax1.set_ylabel('Probability')
#     ax1.grid()
#
#     ax2.scatter(list_real_delta, list_delta, s=1)
#     ax2.set_title('Разность метрик между группами')
#     ax2.set_xlabel('real ration metric delta')
#     ax2.set_ylabel(metric_name)
#     ax2.grid()
#
#     return fig
