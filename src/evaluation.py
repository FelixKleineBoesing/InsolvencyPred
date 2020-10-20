import numpy as np
import json
import os
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize


def get_auc(probs, actuals):
    return roc_auc_score(actuals, probs)


def get_f1_score(preds, actuals):
    preds = np.array(preds)
    actuals = np.array(actuals)
    precision = get_precision(preds, actuals)
    recall = get_recall(preds, actuals)
    dividend = (precision + recall)
    if dividend == 0:
        return 0
    else:
        return (2 * precision * recall) / (precision + recall)


def get_accuracy(preds, actuals):
    preds = np.array(preds)
    actuals = np.array(actuals)
    return np.sum(preds == actuals) / len(preds)


def get_weighted_accuracy(preds, actuals, weight: float):
    """

    :param preds:
    :param actuals:
    :param weight: weight for the positive class (cost for 1/cost for 0)
    :return:
    """
    preds = np.array(preds)
    actuals = np.array(actuals)
    return (np.sum(np.logical_and(preds == 1, actuals == 1)) * weight +
            np.sum(np.logical_and(preds == 0, actuals == 0))) / (np.sum(actuals==1) * weight + np.sum(actuals==0))


def get_threshold_for_optim_cost(probs, actuals, weight: float, steps: int = 500):
    """

    :param probs:
    :param actuals:
    :param weight: positive class cost  negative class cost
    :return:
    """
    probs = np.array(probs)
    actuals = np.array(actuals)

    def optim_func(threshold):
        preds = probs > float(threshold)
        positive_cost = np.sum(np.logical_and(preds == 0, actuals == 1)) * weight
        negative_cost = np.sum(np.logical_and(preds == 1, actuals == 0))
        return negative_cost + positive_cost

    # The function is a non continous function therefore its hard to optimize this one
    # This is the very naive implementation. In Case of a larges size of observation this could take more time and
    # should therefore be approximated with a polynomial function which can be optimized
    costs = []
    for i in range(steps):
        cost = optim_func(i / steps)
        costs.append(cost)

    return np.argmin(costs) / steps, np.min(costs)


def get_recall(preds, actuals):
    preds = np.array(preds)
    actuals = np.array(actuals)
    dividend = (np.sum(np.logical_and(preds == 1, actuals == 1)) +
                np.sum(np.logical_and(preds == 0, actuals == 1)))
    if dividend == 0:
        return 0
    else:
        return np.sum(np.logical_and(preds == 1, actuals == 1)) / dividend


def get_precision(preds, actuals):
    preds = np.array(preds)
    actuals = np.array(actuals)
    return np.sum(np.logical_and(preds == 1, actuals == 1)) / (np.sum(np.logical_and(preds == 1, actuals == 1)) +
                                                               np.sum(np.logical_and(preds == 1, actuals == 0)))


def get_all_measures(probs, actuals, threshold):
    preds = [1 if v > threshold else 0 for v in probs]
    measures = {
        "auc": get_auc(probs, actuals),
        "f1": get_f1_score(preds, actuals),
        "acc": get_accuracy(preds, actuals),
        "recall": get_recall(preds, actuals),
        "precision": get_precision(preds, actuals)
    }
    return measures


class EvaluationTracker:

    def __init__(self, file_store: str = "../data/eval.json"):
        self.file_store = file_store
        self._current_run = None
        if os.path.exists(file_store):
            with open(file_store, "r") as f:
                self._track_cache = json.load(f)
        else:
            track_cache = {"runs": {},
                           "last_run_id": {}}
            with open(file_store, "w") as f:
                json.dump(track_cache, f)
            self._track_cache = track_cache

    def add_parameter(self, name: str, value):
        pass

    def add_metric(self, name: str, value):
        pass

    def add_artifact(self):
        pass

    def start_tracking(self, tracking_name: str):
        if tracking_name in self._track_cache["runs"]:
            tracking_id = self._track_cache["last_run_id"][tracking_name] + 1
        else:
            tracking_id = 0
        self._current_run = (tracking_name, tracking_id)
        self._track_cache[self._current_run] = {"params": {"name": [], "value": []},
                                                "metric": {"name": [], "value": []}}

    def stop_tracking(self):
        pass


if __name__ == "__main__":
    print(get_weighted_accuracy([1, 0, 1, 0, 0, 1], actuals=[1, 0, 0, 0, 0, 1], weight=10))
    print(get_threshold_for_optim_cost([0.8, 0, 0.7, 0, 0, 0.9], actuals=[1, 0, 0, 0, 0, 1], weight=10))