"""
Набор функций для проверки гипотез для данных типа:

data = List[np.array]
    [магазин1_продажи1, магазин1_продажи2, ...],
    [магазин2_продажи1, магазин2_продажи2, ...],
]

, где data -- список массивов продаж каждого магазина:
* элемент списка -- продажи одного магазина
* элемент массива -- продажи в конкретный день/неделю

Метрикой, для которой производится проверка является средние продажи магазина в день/неделю.
"""
import numpy as np
import pandas as pd
from typing import List
from scipy import stats

"""
-----------------------------------------------------------------------------------------------------------------------
----------------------------Методы, не использующие данных с препериода------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------
"""


def check_ttest_naive(a: List[np.array], b: List[np.array]) -> (float, float):
    """
    Лобовой способ сравнения продаж.
    Сравниваем средние продолжительности сессий по всей контрольной и пилотной группам.

    Получается плохо, так как данные внутри одного магазина являются зависимыми.

    Parameters
    ----------
    a: List[np.array], список множеств продаж магазинов контрольной группы
    b: List[np.array], список множеств продаж магазинов пилотной группы

    Returns
    -------
    pvalue: float, pvalue
    delta: float, разница средних в группах
    """

    a = np.concatenate(a)
    b = np.concatenate(b)

    _, pvalue = stats.ttest_ind(a, b)
    delta = np.mean(b) - np.mean(a)

    return pvalue, delta


def check_ttest_avg(a: List[np.array], b: List[np.array]) -> (float, float):
    """
    Первоначально производим агрегацию данных по магазинам.
    Т.е. для каждого магазина считаем средние продажи и затем считаем статистики.

    * Данные внутри магазина перестают быть зависимыми, так как мы получаем 1 чиселку для одного магазина.
    * Минус в том, что метрика получаются не сонаправленной средним продажам во всех магазинах группы.
    * Иными словами, могут возникать ситуации, когда средние продажи в группе выросли,
    а метрика данной функции показывает падение, и наоборот.

    Parameters
    ----------
    a: List[np.array], список множеств продаж магазинов контрольной группы
    b: List[np.array], список множеств продаж магазинов пилотной группы

    Returns
    -------
    pvalue: float, pvalue
    delta: float, разница средних в группах
    """

    a = [np.mean(u) for u in a]
    b = [np.mean(u) for u in b]

    _, pvalue = stats.ttest_ind(a, b)
    delta = np.mean(b) - np.mean(a)

    return pvalue, delta


def check_bootstrap(
        a: List[np.array],
        b: List[np.array],
        n_boot: int = 1000,
        return_diffs: bool = False,
        w_a: List[float] = None,
        w_b: List[float] = None,
) -> (float, float):
    """
    Проверка гипотезы методом бутстрепа.

    Бутстреп является хорошим методом проверки.
    Его минусом является вычислительная сложность, считается на порядок дольше других методов.

    Parameters
    ----------
    a: List[np.array], список множеств продаж магазинов контрольной группы
    b: List[np.array], список множеств продаж магазинов пилотной группы
    n_boot: Optional[int] (default=1000), количество итераций бутстрепа
    return_diffs: Optional[bool] (default=False), возвращать ли список нагенерированных разностей
    w_a: List[np.array] (default=None), список весов магазинов контрольной группы. По-умолчанию равные веса для всех семплов.
    w_b: List[np.array] (default=None), список весов магазинов пилотной группы. По-умолчанию равные веса для всех семплов.

    Returns
    -------
    pvalue: float, pvalue
    delta: float, разница средних в группах
    list_diff: Optional[List[float]], список набутстрепленных дельт
    """

    len_a = len(a)
    len_b = len(b)
    group_len = min(len_a, len_b)

    w_a = np.ones(len_a) if w_a is None else w_a
    w_b = np.ones(len_b) if w_b is None else w_b

    # Расчитываем суммарные продажи в магазинах и количество дней/недель продаж
    def _agg_data(data, weights):
        len_data = len(weights)
        sum_count = np.zeros((len_data, 3))
        sum_count[:, 0] = np.array([np.sum(u) for u in data])
        sum_count[:, 1] = np.array([len(u) for u in data])
        sum_count[:, 2] = np.array(weights) / np.array(weights).sum()
        return sum_count

    a_indices = np.arange(len_a)
    b_indices = np.arange(len_b)

    a_sum_count = _agg_data(a, w_a)
    b_sum_count = _agg_data(b, w_b)

    # Бутстрепим
    def _boot(data, indices):
        index = np.random.choice(indices, group_len, p=data[:, 2])
        boot = data[index, :-1]
        metric = boot[:, 0].sum() / boot[:, 1].sum()
        return metric

    list_diff = []
    for _ in range(n_boot):
        a_boot_metric = _boot(a_sum_count, a_indices)
        b_boot_metric = _boot(b_sum_count, b_indices)
        list_diff.append(b_boot_metric - a_boot_metric)

    delta = ((b_sum_count[:, 0] * b_sum_count[:, 2]).sum() / (b_sum_count[:, 1] * b_sum_count[:, 2]).sum()) \
            - ((a_sum_count[:, 0] * a_sum_count[:, 2]).sum() / (a_sum_count[:, 1] * a_sum_count[:, 2]).sum())
    std = np.std(list_diff)

    pvalue = 2 * (1 - stats.norm.cdf(np.abs(delta / std)))

    if return_diffs:
        return pvalue, delta, list_diff
    else:
        return pvalue, delta


def check_delta_method(a: List[np.array], b: List[np.array]) -> (float, float):
    """
    Проверка гипотезы с помощью дельта-метода.
    Приводится просто для того, чтобы помнить, что такой способ есть.

    Плюсы:
    * он более чувствителен, чем усреднение
    * даёт сонаправленную метрику

    Минусы:
    * нельзя применять методы снижения дисперсии, так как мы не считаем метрику в разрезе магазина

    Parameters
    ----------
    a: List[np.array], список множеств продаж магазинов контрольной группы
    b: List[np.array], список множеств продаж магазинов пилотной группы

    Returns
    -------
    pvalue: float, pvalue
    delta: float, разница средних в группах
    """

    dict_stats = {'a': {'data': a}, 'b': {'data': b}}
    for key, dict_ in dict_stats.items():
        data = dict_['data']
        dict_['x'] = np.array([np.sum(row) for row in data])
        dict_['y'] = np.array([len(row) for row in data])
        dict_['metric'] = np.sum(dict_['x']) / np.sum(dict_['y'])
        dict_['len'] = len(data)
        dict_['mean_x'] = np.mean(dict_['x'])
        dict_['mean_y'] = np.mean(dict_['y'])
        dict_['std_x'] = np.std(dict_['x'])
        dict_['std_y'] = np.std(dict_['y'])
        dict_['cov_xy'] = np.cov(dict_['x'], dict_['y'])[0, 1]
        dict_['var_metric'] = (
            (dict_['std_x'] ** 2) / (dict_['mean_y'] ** 2)
            + (dict_['mean_x'] ** 2) / (dict_['mean_y'] ** 4) * (dict_['std_y'] ** 2)
            - 2 * dict_['mean_x'] / (dict_['mean_y'] ** 3) * dict_['cov_xy']
        ) / dict_['len']

    var = dict_stats['b']['var_metric'] + dict_stats['a']['var_metric']
    delta = dict_stats['b']['metric'] - dict_stats['a']['metric']
    statistic = delta / np.sqrt(var)
    pvalue = 2 * (1 - stats.norm.cdf(np.abs(statistic)))

    return pvalue, delta


def check_linearization(a: List[np.array], b: List[np.array]) -> (float, float):
    """
    Проверка гипотезы с помощью метода линеаризации.
    Преимущество этого метода относительно дельта-метода в том, что в нём можно использовать CUPED.

    Parameters
    ----------
    a: List[np.array], список множеств продаж магазинов контрольной группы
    b: List[np.array], список множеств продаж магазинов пилотной группы

    Returns
    -------
    pvalue: float, pvalue
    delta: float, разница средних в группах
    """
    a_x = np.array([np.sum(row) for row in a])
    a_y = np.array([len(row) for row in a])
    b_x = np.array([np.sum(row) for row in b])
    b_y = np.array([len(row) for row in b])

    coef = np.sum(a_x) / np.sum(a_y)

    a_lin = a_x - coef * a_y
    b_lin = b_x - coef * b_y

    delta = np.mean(b_lin) - np.mean(a_lin)
    _, pvalue = stats.ttest_ind(a_lin, b_lin)

    return pvalue, delta


"""
-----------------------------------------------------------------------------------------------------------------------
----------------------------Методы, использующие данные с препериода---------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------
"""


def _calc_theta(t_control: np.array, t_pilot: np.array, t_control_cov: np.array, t_pilot_cov: np.array) -> float:
    """Вычисляем Thete для CUPED.

    t_control -- значение метрик на тестовом периоде в магазинах контрольной группы
    t_pilot -- значение метрик на тестовом периоде в магазинах пилотной группы
    t_control_before -- значение ковариант магазинов на препилотном периоде в контрольной группе
    t_pilot_before -- значение ковариант магазинов на тестовом препилотном в пилотной группе
    """
    t = np.hstack([t_control, t_pilot])
    t_cov = np.hstack([t_control_cov, t_pilot_cov])
    covariance = np.cov(t_cov, t)[0, 1]
    variance = t_cov.var()
    theta = covariance / variance

    return theta


def check_cuped(a: List[np.array], a_before: List[np.array],
                b: List[np.array], b_before: List[np.array],
                print_stats: bool = False) -> (float, float):
    """
    Проверка гипотезы с помощью метода линеаризации с использование CUPED.

    Parameters
    ----------
    a: List[np.array], список множеств продаж магазинов контрольной группы
    a_before: List[np.array], список множеств продаж магазинов контрольной группы на препилотном периоде
    b: List[np.array], список множеств продаж магазинов пилотной группы
    b_before: List[np.array], список множеств продаж магазинов пилотной группы на препилотном периоде
    print_stats: Optional[bool] (default=False), вывод промежуточных статистик в stdout

    Returns
    -------
    pvalue: float, pvalue
    delta: float, разница средних в группах
    """
    # Создадим датафреймы данных
    df_pilot = pd.DataFrame({
        'x': [x.sum() for x in b],
        'y': [len(x) for x in b],
        'x_before': [x.sum() for x in b_before],
        'y_before': [len(x) for x in b_before]
    })

    df_control = pd.DataFrame({
        'x': [x.sum() for x in a],
        'y': [len(x) for x in a],
        'x_before': [x.sum() for x in a_before],
        'y_before': [len(x) for x in a_before]
    })

    # Линеаризация и нормализация данных
    coef_lin = df_control['x'].sum() / df_control['y'].sum()
    if print_stats:
        print(f'Коэффициент линеаризации -- {coef_lin:.2f}')

    for df, data in zip([df_pilot, df_control], [b, a]):
        df['metric_lin'] = df['x'] - coef_lin * df['y']
        df['metric_lin_before'] = df['x_before'] - coef_lin * df['y_before']

    if print_stats:
        _, p = stats.ttest_ind(df_pilot['metric_lin'], df_control['metric_lin'])
        print(f'p-value на линеаризованных данных -- {p:.4f}')

    theta = _calc_theta(df_control['metric_lin'], df_pilot['metric_lin'], df_control['metric_lin_before'],
                        df_pilot['metric_lin_before'])
    if print_stats:
        print(f'Theta для CUPED -- {theta:.2f}')

    for df in [df_pilot, df_control]:
        df['metric_cuped'] = df['metric_lin'] - theta * df['metric_lin_before']

    delta = df_pilot['metric_cuped'].mean() - df_control['metric_cuped'].mean()
    _, pvalue = stats.ttest_ind(df_pilot['metric_cuped'], df_control['metric_cuped'])

    return pvalue, delta


def check_normed(a: List[np.array], a_before: List[np.array],
                 b: List[np.array], b_before: List[np.array]) -> (float, float):
    """
    Проверка гипотезы с помощью метода нормализации данных на препериод.

    Parameters
    ----------
    a: List[np.array], список множеств продаж магазинов контрольной группы
    a_before: List[np.array], список множеств продаж магазинов контрольной группы на препилотном периоде
    b: List[np.array], список множеств продаж магазинов пилотной группы
    b_before: List[np.array], список множеств продаж магазинов пилотной группы на препилотном периоде

    Returns
    -------
    pvalue: float, pvalue
    delta: float, разница средних в группах
    """

    def _norm_data(data, data_before):
        """Нормализация данных на препериод
        """
        mean_before = [max(x.sum(), 1) / len(x) for x in data_before]
        return [np.mean(x / m) for x, m in zip(data, mean_before)]

    a_normed = _norm_data(a, a_before)
    b_normed = _norm_data(b, b_before)

    delta = np.mean(b_normed) - np.mean(a_normed)
    _, pvalue = stats.ttest_ind(a_normed, b_normed)

    return pvalue, delta


def check_normed_delta_method(a: List[np.array], a_before: List[np.array],
                              b: List[np.array], b_before: List[np.array]) -> (float, float):
    """
    Проверка гипотезы дельта-методом на нормализованных данных.

    Parameters
    ----------
    a: List[np.array], список множеств продаж магазинов контрольной группы
    a_before: List[np.array], список множеств продаж магазинов контрольной группы на препилотном периоде
    b: List[np.array], список множеств продаж магазинов пилотной группы
    b_before: List[np.array], список множеств продаж магазинов пилотной группы на препилотном периоде

    Returns
    -------
    pvalue: float, pvalue
    delta: float, разница средних в группах
    """

    dict_stats = {
        'a': {'data': [x / (max(x_before.sum(), 1) / len(x_before)) for x, x_before in zip(a, a_before)]},
        'b': {'data': [x / (max(x_before.sum(), 1) / len(x_before)) for x, x_before in zip(b, b_before)]}
    }
    for key, dict_ in dict_stats.items():
        data = dict_['data']
        dict_['x'] = np.array([np.sum(row) for row in data])
        dict_['y'] = np.array([len(row) for row in data])
        dict_['metric'] = np.sum(dict_['x']) / np.sum(dict_['y'])
        dict_['len'] = len(data)
        dict_['mean_x'] = np.mean(dict_['x'])
        dict_['mean_y'] = np.mean(dict_['y'])
        dict_['std_x'] = np.std(dict_['x'])
        dict_['std_y'] = np.std(dict_['y'])
        dict_['cov_xy'] = np.cov(dict_['x'], dict_['y'])[0, 1]
        dict_['var_metric'] = (
            (dict_['std_x'] ** 2) / (dict_['mean_y'] ** 2)
            + (dict_['mean_x'] ** 2) / (dict_['mean_y'] ** 4) * (dict_['std_y'] ** 2)
            - 2 * dict_['mean_x'] / (dict_['mean_y'] ** 3) * dict_['cov_xy']
        ) / dict_['len']

    var = dict_stats['b']['var_metric'] + dict_stats['a']['var_metric']
    delta = dict_stats['b']['metric'] - dict_stats['a']['metric']
    statistic = delta / np.sqrt(var)
    pvalue = 2 * (1 - stats.norm.cdf(np.abs(statistic)))

    return pvalue, delta