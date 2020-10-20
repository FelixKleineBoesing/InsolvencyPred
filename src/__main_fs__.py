import pandas as pd
import functools
from cleaning import load_arff_files
from evaluation import get_weighted_accuracy, get_auc
from feature_selecting import GreedyForwardSelector
from modelling import CrossValidation, XGBoost
from preprocessing import ReSampler


def main(data, pred_function, early_stopping=5, tolerance=0.001, verbose=True, max_processes=8):
    gfs = GreedyForwardSelector()
    label = data.pop("class")
    features, measure = gfs.run_selection(data=data, label=label, prediction_function=pred_function, early_stopping_iter=early_stopping,
                      tolerance=tolerance, verbose=verbose, max_processes=max_processes)

    return features, measure


class pred_function:

    def __init__(self, cost_weight):
        self.cost_weight = cost_weight

    def __call__(self, data, label):
        cv = CrossValidation(folds=4)
        model = XGBoost(val_share=0.0, n_rounds=20, lambda_=0,
                        additional_booster_params={"params": {"max_depth": 4, "subsample": 0.7,
                                                              "colsample_bytree": 0.7}}) #, "scale_pos_weight": (df.shape[0] - np.sum(np.sum(df["class])) / np.sum(df["class])

        probs = cv.run(data=data, label=label, model=model, preprocessors=[ReSampler("down")])
        return get_auc(probs, actuals=label)


if __name__ == "__main__":
    pd.set_option('chained_assignment', None)
    cost_weight = 13
    dfs, file_names = load_arff_files("../data/raw_data/")
    features, measure = main(dfs[0], pred_function=pred_function(cost_weight), max_processes=4)
    print(features)