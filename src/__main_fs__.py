import json
import os
import pandas as pd

from cleaning import load_arff_files
from evaluation import get_auc
from feature_selecting import GreedyForwardSelector, CorrelationSelector
from modelling import CrossValidation, XGBoost
from preprocessing import ReSampler


def main_greedy(data, pred_function, early_stopping=5, tolerance=0.001, verbose=True, max_processes=8):
    gfs = GreedyForwardSelector()
    label = data.pop("class")
    features, measures = gfs.run_selection(data=data, label=label, prediction_function=pred_function,
                                          early_stopping_iter=early_stopping,
                                          tolerance=tolerance, verbose=verbose, max_processes=max_processes)

    return features, measures


def main_corr(data, pred_function, early_stopping=5, tolerance=0.001, verbose=True):
    gfs = CorrelationSelector()
    label = data.pop("class")
    features, measures = gfs.run_selection(data=data, label=label, prediction_function=pred_function,
                                          early_stopping_iter=early_stopping,
                                          tolerance=tolerance, verbose=verbose)

    return features, measures


class pred_function:

    def __init__(self, cost_weight):
        self.cost_weight = cost_weight

    def __call__(self, data, label):
        cv = CrossValidation(folds=4)
        model = XGBoost(val_share=0.0, n_rounds=50, lambda_=5,
                        additional_booster_params={"params": {"max_depth": 4, "subsample": 0.7,
                                                              "colsample_bytree": 0.7}}) #, "scale_pos_weight": (df.shape[0] - np.sum(np.sum(df["class])) / np.sum(df["class])

        probs = cv.run(data=data, label=label, model=model, preprocessors=[ReSampler("down")])
        return get_auc(probs, actuals=label)


if __name__ == "__main__":
    pd.set_option('chained_assignment', None)
    cost_weight = 13
    features_path = "../data/cleaned_data/features.json"
    dfs, file_names = load_arff_files("../data/raw_data/")

    if os.path.exists(features_path):
        with open(features_path, "r") as f:
            features = json.load(f)
    else:
        features = {"corr_selector": {}, "greedy_selector": {}}

    for i, df in enumerate(dfs):
        corr_features, corr_measure = main_corr(data=df.copy(), pred_function=pred_function(cost_weight))

        features["corr_selector"]["Year{}".format(i + 1)] = {"features": corr_features, "auc": corr_measure}
        with open(features_path, "w") as f:
            json.dump(features, f)
        print(corr_features)

    for i, df in enumerate(dfs):
        features_greedy, greedy_measure = main_greedy(data=df.copy(), pred_function=pred_function(cost_weight))
        features["greedy_selector"]["Year{}".format(i + 1)] = {"features": features_greedy, "auc": greedy_measure}

        with open(features_path, "w") as f:
            json.dump(features, f)
        print(features_greedy)

