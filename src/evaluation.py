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


def get_params_for_best_measure_overall(measure: str = "auc", path_to_measure_file: str = "../"):
    with open(path_to_measure_file, "r") as f:
        measures = json.load(f)

    year_values = {}
    for m in measures:
        for key in m.keys():
            if key in year_values:
                if year_values[key][measure] < m[key][measure]:
                    year_values[key] = m[key]
            else:
                year_values[key] = m[key]

    return year_values


def get_params_for_best_model(measure: str = "auc", target_year: int = 1, path_to_measure_file: str = "../"):
    with open(path_to_measure_file, "r") as f:
        measures = json.load(f)

    max_measure = None
    index = None
    for i, m in enumerate(measures):
        if max_measure is not None:
            if m["Year1"][measure] > max_measure:
                max_measure = m["Year{}".format(target_year)]
                index = i
    return measures[index]


if __name__ == "__main__":
    print(get_weighted_accuracy([1, 0, 1, 0, 0, 1], actuals=[1, 0, 0, 0, 0, 1], weight=10))
    print(get_threshold_for_optim_cost([0.8, 0, 0.7, 0, 0, 0.9], actuals=[1, 0, 0, 0, 0, 1], weight=10))
    print(get_params_for_best_measure_overall("auc", "../data/cleaned_data/recorded_measures_xgb.json"))