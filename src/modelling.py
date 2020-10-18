import abc
import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.linear_model import LogisticRegression as LR
from typing import Union, List


class Model(abc.ABC):

    def __init__(self):
        if not hasattr(self, "_model"):
            self._model = None

    @abc.abstractmethod
    def train(self, train_data: pd.DataFrame, label: Union[list, pd.Series]):
        pass

    def predict(self, test_data: pd.DataFrame):
        assert self._model is not None, "Model must be trained first!"
        return self._predict(test_data)

    @abc.abstractmethod
    def _predict(self, test_data: pd.DataFrame):
        pass


class XGBoost(Model):

    def __init__(self, n_rounds: int = 10, eta: float = 0.3, lambda_: float = 0,
                 additional_booster_params: dict = None, val_share: float = 0.0):
        """

        :param n_rounds:
        :param eta:
        :param lambda_:
        :param additional_booster_params:
        """
        super().__init__()
        self.val_share = val_share
        self.params = {"num_boost_round": n_rounds, "params":
            {"eta": eta, "lambda": lambda_, "objective": "binary:logistic", "eval_metric": "logloss"}}
        if additional_booster_params is not None:
            if "params" in additional_booster_params:
                self.params["params"].update(additional_booster_params["params"])
            del additional_booster_params["params"]
            self.params.update(additional_booster_params)

    def train(self, train_data: pd.DataFrame, label: Union[list, pd.Series]):
        """

        :param train_data:
        :param label:
        :return:
        """
        if self.val_share > 0.0:
            number_obs = train_data.shape[0]
            val_indices = np.random.choice(np.arange(number_obs), int(self.val_share*number_obs), replace=False)
            val_matrix = xgb.DMatrix(data=train_data.iloc[val_indices, :],
                                     label=[v for i, v in enumerate(label) if i in val_indices])
            train_matrix = xgb.DMatrix(data=train_data.iloc[[i for i in range(number_obs) if i not in val_indices], :],
                                       label=[v for i, v in enumerate(label) if i not in val_indices])
            eval_list = [(train_matrix, "train"), (val_matrix, "eval")]
        else:
            train_matrix = xgb.DMatrix(data=train_data, label=label)
            eval_list = [(train_matrix, "train")]
        self._model = xgb.train(**self.params, dtrain=train_matrix, evals=eval_list)

    def _predict(self, test_data: pd.DataFrame):
        assert self._model is not None, "Model must be trained first!"
        test_matrix = xgb.DMatrix(data=test_data)
        return self._model.predict(test_matrix)


class LogisticRegression(Model):

    def __init__(self, random_state: int = 0, penalty: str = "l2", max_iter: int = 100):
        assert penalty in ["l2", "l1"]
        self._model = LR(random_state=random_state, penalty=penalty, max_iter=max_iter)
        super().__init__()

    def train(self, train_data: pd.DataFrame, label: Union[list, pd.Series]):
        nas = np.sum(~np.isfinite(train_data), axis=0)
        assert np.sum(nas) == 0, "There are nas in the following columns: {}".format(nas[nas > 0])
        self._model.fit(train_data, label)

    def _predict(self, test_data: pd.DataFrame):
        return self._model.predict_proba(test_data)[:, 1]


class CrossValidation:

    def __init__(self, folds: int = 5):
        self.folds = folds

    def run(self, data: pd.DataFrame, label: Union[pd.Series, list], model: Model,
            preprocessors: List['Preprocessor'] = None):
        """

        :param data:
        :param label:
        :param model: instance of class Model
        :param preprocessors: can be a list of preprocessors that take the training and test data and labels,
            mutate and return it afterwards
        :return:
        """
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        test_indices = np.array_split(indices, self.folds)
        test_indices = [sorted(l) for l in test_indices]
        predictions = np.array([np.NaN for _ in range(len(indices))])
        for i in range(self.folds):
            train_data, test_data = data.iloc[[k for k in range(data.shape[0]) if k not in test_indices[i]], :], \
                                    data.iloc[test_indices[i], :]
            train_label, test_label = [v for j, v in enumerate(label) if j not in test_indices[i]], \
                                      [v for j, v in enumerate(label) if j in test_indices[i]]
            if preprocessors is not None:
                for preprocessor in preprocessors:
                    train_data, test_data, train_label = preprocessor.process(train_data=train_data,
                                                                              test_data=test_data,
                                                                              train_label=train_label)
            print("Label Distribution: Train: {}, Test: {}".format(np.mean(train_label), np.mean(test_label)))
            model.train(train_data=train_data, label=train_label)
            predictions[test_indices[i]] = model.predict(test_data)
        return predictions


if __name__ == "__main__":
    model = XGBoost()
    data = pd.DataFrame({"a": [1, 2, 3], "b": [7, 5, 2]})
    label = [0, 1, 1]
    tdata = pd.DataFrame({"a": [4], "b": [1]})
    tlabel = [1]

    model.train(data, label)
    print(model.predict(tdata))

    lr_model = LogisticRegression()
    lr_model.train(data, label)
    print(lr_model.predict(tdata))

    cv = CrossValidation(5).run(data, label, LogisticRegression(), preprocessors=None)

