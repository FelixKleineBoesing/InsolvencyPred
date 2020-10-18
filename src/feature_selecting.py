import abc


class FeatureSelector(abc.ABC):

    def __init__(self):
        pass

    @abc.abstractmethod
    def run_selection(self):
        pass

    def get_selected_features(self):
        pass

    def get_search_trajectory(self):
        pass