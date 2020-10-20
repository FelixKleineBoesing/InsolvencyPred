import abc
import numpy as np
import pandas as pd
from typing import Union
from sklearn.decomposition import PCA as PCASK


def _sampling(data, label, method: str = "up"):
    """

    :param data:
    :return:
    """
    pd.set_option('chained_assignment', None)
    assert method in ["up", "down"]
    data["class"] = label
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
    sampled_data.reset_index(inplace=True, drop=True)
    sampled_data = sampled_data.iloc[np.random.permutation(sampled_data.shape[0]),:]
    sampled_data.reset_index(inplace=True, drop=True)
    sampled_label = sampled_data.pop("class")
    return sampled_data, sampled_label


def remove_outliers(values, iqr_factor: float = 2.5):
    """

    :param values: list of values for which the outlier should be removed
    :param iqr_factor:
    :return:
    """
    lq, uq, iqr = _get_iqr(values)
    return [v for v in values if lq - iqr_factor * iqr < v < uq + iqr_factor * iqr]


def clip_outliers(values: pd.DataFrame, iqr_factor: float = 2.5):
    """
    instead of removing it sets outlier to the lower and the upper bound that is defined by the 25th and 75th
    percent quantile subtracted/added by the iqr time a defined iqr_factor

    :param values: list of values for which the outlier should be clipped
    :param iqr_factor:
    :return:
    """
    for col in values.columns:
        lq, uq, iqr = _get_iqr(values[col])
        lower_bound, upper_bound = lq - iqr_factor * iqr, uq + iqr_factor * iqr
        values.loc[values[col] < lower_bound, col] = lower_bound
        values.loc[values[col] > upper_bound, col] = upper_bound
    return values


def _get_iqr(values):
    lq, uq = np.nanpercentile(values, [25, 75])
    iqr = uq - lq
    return lq, uq, iqr


class Preprocessor(abc.ABC):

    @abc.abstractmethod
    def process(self, train_data, test_data=None, train_label=None) -> (pd.DataFrame, pd.DataFrame,
                                                                        Union[pd.Series, list]):
        """
        This method may mutate the train and test data and return them afterwards

        :param train_data:
        :param test_data:
        :param train_label:
        :return: a tuple that contains the train and test dataframes and the label
        """
        pass


class PCA(Preprocessor):

    def __init__(self, variance_threshold: float = 0.95):
        self.variance_threshold = variance_threshold

    def process(self, train_data, test_data=None, train_label=None) -> (pd.DataFrame, pd.DataFrame,
                                                                        Union[pd.Series, list]):
        pca = PCASK(n_components=self.variance_threshold)
        pca.fit(train_data)
        train_data = pd.DataFrame(pca.transform(train_data))
        test_data = pd.DataFrame(pca.transform(test_data))

        return train_data, test_data, train_label


class CorrelationRemover(Preprocessor):

    def __init__(self, number_dropped_features: int = 10, squared: bool = False):
        self.number_dropped_features = number_dropped_features
        self.squared = squared

    def process(self, train_data, test_data=None, train_label=None) -> (pd.DataFrame, pd.DataFrame,
                                                                        Union[pd.Series, list]):
        assert train_data.shape[1] > self.number_dropped_features

        for i in range(self.number_dropped_features):
            corrs = train_data.corr()
            if self.squared:
                corrs = np.sqrt(corrs)
            sum_corrs = corrs.sum(axis=1)
            col_to_remove = corrs.index[np.nanargmax(np.abs(sum_corrs))]
            train_data = train_data.drop(col_to_remove, axis=1)
            if test_data is not None:
                test_data = test_data.drop(col_to_remove, axis=1)
        return train_data, test_data, train_label


class ReSampler(Preprocessor):

    def __init__(self, method: str = "up"):
        self.method = method

    def process(self, train_data, test_data=None, train_label=None) -> (pd.DataFrame, pd.DataFrame):
        data, label = _sampling(train_data, train_label, self.method)
        return data, test_data, label


class MeanReplacement(Preprocessor):

    def process(self, train_data, test_data=None, train_label=None) -> (pd.DataFrame, pd.DataFrame):
        for col in train_data.columns:
            mean_train = np.nanmean(train_data.loc[np.isfinite(train_data[col]), col])
            train_data.loc[~np.isfinite(train_data[col]), col] = mean_train
            if test_data is not None:
                test_data.loc[~np.isfinite(test_data[col]), col] = mean_train
        return train_data, test_data, train_label


class Standardizer(Preprocessor):

    def process(self, train_data, test_data=None, train_label=None) -> (pd.DataFrame, pd.DataFrame):
        mean_train = np.nanmean(train_data[np.isfinite(train_data)], axis=0)
        sd_train = np.nanstd(train_data[np.isfinite(train_data)], axis=0)

        train_data = (train_data - mean_train) / sd_train
        if test_data is not None:
            test_data = (test_data - mean_train) / sd_train
        return train_data, test_data, train_label


if __name__ == "__main__":
    values = [i for i in range(1000)]
    values.extend([-1000, 2000])
    clipped_values = remove_outliers(values, 1.5)
    print(min(clipped_values))
    print(max(clipped_values))
    data = pd.DataFrame({"class": [1, 1, 0, 1, 0,0], "a": [1, 2, 3, 4, 5, 6]})
    data_na = pd.DataFrame({"class": [1, 1, 0, 1, 0,0], "a": [1, 2, 3, 4, np.NaN, np.Inf]})
    mean_replacer = MeanReplacement()
    mean_replacer.process(data_na)

    correlation = CorrelationRemover(1)
    correlation.process(data_na)


