import numpy as np
import json
import os
from sklearn.metrics import roc_auc_score


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