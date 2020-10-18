import abc
import pandas as pd


class FeatureSelector(abc.ABC):

    @abc.abstractmethod
    def run_selection(self, data: pd.DataFrame, ):
        pass

    def get_selected_features(self):
        pass

    def get_search_trajectory(self):
        pass


class CorrelationSelector(FeatureSelector):
    """
    This FeatureSelector removes one feature in each run until the
    """
    def __init__(self):
        pass

    def run_selection(self, data, ):
        pass

