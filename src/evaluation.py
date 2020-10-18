import numpy as np


def get_auc(probs, actuals):
    # TODO
    pass


def get_roc_curve(probs, actuals):
    # TODO
    pass


def get_f1_score(preds, actuals):
    precision = get_precision(preds, actuals)
    recall = get_recall(preds, actuals)
    return (2 * precision * recall) / (precision + recall)


def get_accuracy(preds, actuals):
    return (preds == actuals) / len(preds)


def get_recall(preds, actuals):
    return np.sum(np.logical_and(preds == 1, actuals == 1)) / (np.sum(np.logical_and(preds == 1, actuals == 1)) +
                                                               np.sum(np.logical_and(preds == 1, actuals == 0)))


def get_precision(preds, actuals):
    return np.sum(np.logical_and(preds == 1, actuals == 1)) / (np.sum(np.logical_and(preds == 1, actuals == 1)) +
                                                               np.sum(np.logical_and(preds == 0, actuals == 1)))