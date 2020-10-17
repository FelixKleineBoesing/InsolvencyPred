import abc
import pandas as pd
import numpy as np
import xgboost as xgb
from typing import Union


class Model(abc.ABC):

    @abc.abstractmethod
    def train(self, train_data: pd.DataFrame, label: Union[list, pd.Series]):
        pass

    @abc.abstractmethod
    def predict(self, test_data: pd.DataFrame):
        pass


class XGBoostModel(Model):

    def __init__(self, n_rounds: int = 10, eta: float = 0.3, lambda_: float = 0,
                 additional_booster_params: dict = None):
        """

        :param n_rounds:
        :param eta:
        :param lambda_:
        :param additional_booster_params:
        """
        self.params = {"num_boost_round": n_rounds, "params":
            {"eta": eta, "lambda": lambda_, "objective": "binary:logistic"}}
        if additional_booster_params is not None:
            self.params.update(additional_booster_params)
        self._model = None

    def train(self, train_data: pd.DataFrame, label: Union[list, pd.Series], val_share: float = 0.0):
        """

        :param train_data:
        :param label:
        :param val_share:
        :return:
        """
        if val_share > 0.0:
            number_obs = train_data.shape[0]
            val_indices = np.random.choice(np.arange(number_obs), int(val_share*number_obs))
            val_matrix = xgb.DMatrix(data=train_data.iloc[val_indices, :],
                                     label=[v for i, v in enumerate(label) if i in val_indices])
            train_matrix = xgb.DMatrix(data=train_data.iloc[~train_data.index.isin(val_indices), :],
                                       label=[v for i, v in enumerate(label) if i not in val_indices])
            eval_list = [(train_matrix, "train"), (val_matrix, "eval")]
        else:
            train_matrix = xgb.DMatrix(data=train_data, label=label)
            eval_list = [(train_matrix, "train")]
        self._model = xgb.train(**self.params, dtrain=train_matrix, evals=eval_list)

    def predict(self, test_data: pd.DataFrame):
        assert self._model is not None, "Model must be trained first!"
        test_matrix = xgb.DMatrix(data=test_data)
        return self._model.predict(test_matrix)


if __name__ == "__main__":
    model = XGBoostModel()
    data = pd.DataFrame({"a": [1, 2, 3], "b": [7, 5, 2]})
    label = [0, 1, 1]
    model.train(data, label, 0.0)