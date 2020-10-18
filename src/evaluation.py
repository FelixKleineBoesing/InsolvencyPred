import numpy as np
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
                np.sum(np.logical_and(preds == 1, actuals == 0)))
    if dividend == 0:
        return 0
    else:
        return np.sum(np.logical_and(preds == 1, actuals == 1)) / dividend


def get_precision(preds, actuals):
    preds = np.array(preds)
    actuals = np.array(actuals)
    return np.sum(np.logical_and(preds == 1, actuals == 1)) / (np.sum(np.logical_and(preds == 1, actuals == 1)) +
                                                               np.sum(np.logical_and(preds == 0, actuals == 1)))