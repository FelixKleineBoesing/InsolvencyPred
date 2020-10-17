import numpy as np
import pandas as pd


def replace_missing_values_mean():
    pass


def replace_missing_values_mice():
    pass


def _sampling(data, method: str = "up"):
    """

    :param data:
    :return:
    """
    assert method in ["up", "down"]
    if method == "up":
        factor = -1
    else:
        factor = 1
    class_neg = data.loc[data["class"] == 0, :]
    class_pos = data.loc[data["class"] == 1, :]

    count_neg = class_neg.shape[0]
    count_pos = class_pos.shape[0]
    args = {}
    if factor * count_neg < factor * count_pos:
        if count_neg > count_pos:
            args["replace"] = True
        down_sampled_df = class_pos.sample(count_neg, **args)
        sampled_data = pd.concat((down_sampled_df, class_neg))
    else:
        if count_pos > count_neg:
            args["replace"] = True
        down_sampled_df = class_neg.sample(count_pos, replace=True)
        sampled_data = pd.concat((down_sampled_df, class_pos))
    return sampled_data


def up_sampling(data):
    return _sampling(data, "up")


def down_sampling(data: pd.DataFrame):
    return _sampling(data, "down")


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

    data = pd.DataFrame({"class": [1, 1, 0, 1]})
    upsampled_data = up_sampling(data)
    downsampled_data = down_sampling(data)