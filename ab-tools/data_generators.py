"""
Генераторы данных для проверки гипотез
"""
import numpy as np
from typing import Tuple, List
from scipy import stats


def get_lognorm(mean: float, std: float, size: int) -> List[float]:
    """
    Функция получения данных логнормального распределения с заданными средним и стандартным отклонением
    Parameters
    ----------
    mean: float, среднее
    std: float, стандартное отклонение
    size: int, размер выборки

    Returns
    -------
    output: List[float], массив сгенерированных из логнормального распределения точек
    """
    k = (std / mean) ** 2 + 1
    s = -np.log(k / 2)
    scale = mean / (k ** 0.5)
    dist = stats.lognorm(s=s, scale=scale)
    return list(dist.rvs(size=size))


# def get_session_duration(size: int, effect: float = 0., s_cnt: Tuple[int] = (1, 6),
#                          mean_sales: int = 100, sales_std: int = 20, noise_std: int = 10,
#                          seed: int = None) -> List[np.array]:
#     """Генерация продаж
#
#     Parameters
#     ----------
#     size: int, количество магазинов.
#     effect: float, размер эффекта, насколько изменились продажи, относительно базовых.
#     s_cnt: tuple[int], минимальное и максимальное значение количества дней/недель продаж для каждого магазина.
#     mean_sales: int, среднее значение продаж магазинов в группе.
#     sales_std: int, дисперсия продаж магазинов в группе.
#     noise_std: int, дисперсия шума в продажах внутри магазина.
#     seed: int, состояние генератора случайных чисел
#
#     Returns
#     -------
#     sales: List[np.array], список продаж в магазинах:
#         элемент списка -- продажи одного магазина
#         элемент массива -- продажи в один день/неделю
#     """
#     if seed:
#         np.random.seed(seed)
#
#     def _gen_shop_sales(mean):
#         shop_size = np.random.randint(*s_cnt)
#         sales = np.random.normal(loc=mean, scale=noise_std, size=shop_size).round()
#         sales = np.where(sales > 0, sales, 0)
#         return sales
#
#     mean_durations = np.random.normal(loc=mean_sales, scale=sales_std, size=size) * (1 + effect)
#     return [_gen_shop_sales(mean) for mean in mean_durations]


def get_sales(size: int, effect: float = 0., s_cnt: Tuple[int] = (1, 6),
              mean_sales: int = 100, sales_std: int = 20, noise_std: int = 10,
              preperiod_effect: float = 0., s_cnt_before: Tuple[int] = None,
              seed: int = None) -> (List[np.array], List[np.array]):
    """Генерация продаж на пилотном и препилотном периодах

    Parameters
    ----------
    size: int, количество магазинов.
    effect: float, размер эффекта, насколько изменились продажи, относительно базовых.
    s_cnt: tuple[int], минимальное и максимальное значение количества дней/недель продаж для каждого магазина.
    mean_sales: int, среднее значение продаж магазинов в группе.
    sales_std: int, дисперсия продаж магазинов в группе.
    noise_std: int, дисперсия шума в продажах внутри магазина.
    s_cnt_before: tuple[int], минимальное и максимальное значение количества дней/недель продаж для каждого магазина на препериоде.
    preperiod_effect: float, эффект относительно препериода.
    seed: int, состояние генератора случайных чисел.

    Returns
    -------
    sales: (List[np.array], List[np.array]), списки продаж на препериоде и пилотном периоде:
        элемент списка -- продажи одного магазина
        элемент массива -- продажи в один день/неделю
    """
    if s_cnt_before is None:
        s_cnt_before = s_cnt
    if seed:
        np.random.seed(seed)

    mean_shops_sales = np.random.normal(loc=mean_sales, scale=sales_std, size=size)
    count_shops_sales = np.random.randint(*s_cnt, size)
    count_shops_sales_before = np.random.randint(*s_cnt_before, size)

    data = []
    data_before = []
    zip_data = zip(mean_shops_sales, count_shops_sales, count_shops_sales_before)

    for mean_shop_sales, count_shop_sales, count_shop_sales_before in zip_data:
        noise_pilot = np.random.normal(0, noise_std, count_shop_sales)
        mean_sales_pilot = mean_shop_sales * (1 + effect)
        pilot_sales = (mean_sales_pilot + noise_pilot).round()
        pilot_sales = np.where(pilot_sales > 0, pilot_sales, 0)
        data.append(pilot_sales)

        noise_prepilot = np.random.normal(0, noise_std, count_shop_sales_before)
        mean_sales_prepilot = mean_shop_sales * (1 - preperiod_effect)
        prepilot_sales = (mean_sales_prepilot + noise_prepilot).round()
        prepilot_sales = np.where(prepilot_sales > 0, prepilot_sales, 0)
        data_before.append(prepilot_sales)

    return data_before, data