import pandas as pd
import numpy as np
from cleaning import load_arff_files
from modelling import LogisticRegression, XGBoost, CrossValidation
from preprocessing import Standardizer, MeanReplacement, clip_outliers, ReSampler, CorrelationRemover, PCA
from evaluation import  get_f1_score, get_accuracy, get_auc
from visualization import get_roc_curve


def main():
    probs_lr = []
    probs_xgb = []

    dfs, file_names = load_arff_files("../data/raw_data/")

    cv = CrossValidation(folds=10)
    for i, df in enumerate(dfs):
        print("#####   {} Year   ####".format(i + 1))
        df = df.copy()
        label = df.pop("class")
        # this is to kick only to kick most extreme outlier
        df = clip_outliers(df.copy(), 5.5)
        model = LogisticRegression(max_iter=100)
        probs_lr.append(cv.run(data=df, label=label, model=model,
                               preprocessors=[MeanReplacement(), Standardizer(), PCA(0.9)])) #

        preds = [1 if p > 0.5 else 0 for p in probs_lr[i]]
        print("F1, Baseline: {}, LogisticRegression: {}".format(get_f1_score(np.random.choice([0, 1], size=len(label)), label),
                                                                get_f1_score(preds, label)))
        print("Acc, Baseline: {}, LogisticRegression: {}".format(get_accuracy([0 for _ in range(len(label))], label),
                                                                 get_accuracy(preds, label)))
        print("AUC, Baseline: {}, LogisticRegression: {}".format(get_auc(np.random.choice([0, 1], size=len(label)), label),
                                                                 get_auc(probs_lr[i], label)))
    get_roc_curve(probs_lr[i], label)

    for i, df in enumerate(dfs):
        print("#####   {} Year   ####".format(i + 1))
        df = df.copy()
        label = df.pop("class")
        # this is to kick only to kick most extreme outlier
        #df = clip_outliers(df.copy(), 5.5)
        model = XGBoost(val_share=0.2, n_rounds=20, lambda_=0,
                        additional_booster_params={"params": {"max_depth": 4}})
        probs_xgb.append(cv.run(data=df, label=label, model=model, preprocessors=[MeanReplacement(), PCA(0.9)]))

        preds = [1 if p > 0.5 else 0 for p in probs_xgb[i]]
        print("F1, Baseline: {}, Xgboost: {}".format(get_f1_score(np.random.choice([0, 1], size=len(label)), label),
                                                     get_f1_score(preds, label)))
        print("Acc, Baseline: {}, Xgboost: {}".format(get_accuracy([0 for _ in range(len(label))], label),
                                                      get_accuracy(preds, label)))
        print("AUC, Baseline: {}, Xgboost: {}".format(get_auc(np.random.choice([0, 1], size=len(label)), label),
                                                      get_auc(probs_xgb[i], label)))
    get_roc_curve(probs_xgb[i], label)


if __name__ == "__main__":
    pd.set_option('chained_assignment', None)
    main()