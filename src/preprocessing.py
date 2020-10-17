import numpy as np


def replace_missing_values_mean():
    pass


def replace_missing_values_mice():
    pass


def up_sampling():
    pass


def down_sampling():
    pass


def remove_outliers(values, iqr_factor: float = 2.5):
    """

    :param values: list of values for which the outlier should be removed
    :param iqr_factor:
    :return:
    """
    lq, uq, iqr = _get_iqr(values)
    return [v for v in values if lq - iqr_factor * iqr < v < uq + iqr_factor * iqr]


def clip_outliers(values, iqr_factor: float = 2.5):
    """
    instead of removing it sets outlier to the lower and the upper bound that is defined by the 25th and 75th
    percent quantile subtracted/added by the iqr time a defined iqr_factor

    :param values: list of values for which the outlier should be clipped
    :param iqr_factor:
    :return:
    """
    lq, uq, iqr = _get_iqr(values)
    values = np.array(values)
    lower_bound, upper_bound = lq - iqr_factor * iqr, uq + iqr_factor * iqr
    values[values < lower_bound] = lower_bound
    values[values < upper_bound] = upper_bound
    return values


def _get_iqr(values):
    lq, uq = np.nanpercentile(values, [25, 75])
    iqr = uq - lq
    return lq, uq, iqr


if __name__ == "__main__":
    values = [i for i in range(1000)]
    values.extend([-1000, 2000])
    clipped_values = remove_outliers(values, 1.5)
    print(min(clipped_values))
    print(max(clipped_values))