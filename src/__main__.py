import pandas as pd
import numpy as np
from cleaning import load_arff_files
from modelling import LogisticRegression, XGBoost, CrossValidation
from preprocessing import Standardizer, MeanReplacement, clip_outliers, ReSampler, CorrelationRemover, PCA
from evaluation import get_f1_score, get_accuracy, get_auc, get_precision, get_recall, get_all_measures
from visualization import get_roc_curve


def main():
    probs_lr = []
    probs_xgb = []

    dfs, file_names = load_arff_files("../data/raw_data/")

    cv = CrossValidation(folds=10)
    all_measures = []

    for i, df in enumerate(dfs):
        print("#####   {} Year   ####".format(i + 1))
        df = df.copy()
        label = df.pop("class")
        # this is to kick only to kick most extreme outlier
        df = clip_outliers(df.copy(), 5.5)
        model = LogisticRegression(max_iter=1000)
        probs = cv.run(data=df, label=label, model=model,
                       preprocessors=[MeanReplacement(), Standardizer(), PCA(0.98)])
        probs_lr.append(probs) #
        measures = get_all_measures(probs, label, 0.5)
        measures["name"] = "Year {}".format(i + 1)
        measures["method"] = "LogisticRegression"
        all_measures.append(measures)
        base_line_measures = get_all_measures(np.random.choice([0, 1], size=len(label)), label, 0.5)
        print("F1, Baseline: {}, LogisticRegression: {}".format(base_line_measures["f1"], measures["f1"]))
        print("Acc, Baseline: {}, LogisticRegression: {}".format(base_line_measures["acc"], measures["acc"]))
        print("AUC, Baseline: {}, LogisticRegression: {}".format(base_line_measures["auc"], measures["auc"]))
        print("Recall, Baseline: {}, LogisticRegression: {}".format(base_line_measures["recall"], measures["recall"]))
        print("Precision, Baseline: {}, LogisticRegression: {}".format(base_line_measures["precision"],
                                                                       measures["precision"]))
    get_roc_curve(probs_lr[i], label)

    for i, df in enumerate(dfs):
        print("#####   {} Year   ####".format(i + 1))
        df = df.copy()
        label = df.pop("class")
        model = XGBoost(val_share=0.2, n_rounds=10, lambda_=0,
                        additional_booster_params={"params": {"max_depth": 4, "subsample": 0.7,
                                                              "colsample_bytree": 0.7}}) #, "scale_pos_weight": (df.shape[0] - np.sum(np.sum(df["class])) / np.sum(df["class])
        probs = cv.run(data=df, label=label, model=model, preprocessors=[])
        probs_xgb.append(probs)

        measures = get_all_measures(probs, label, 0.5)
        base_line_measures = get_all_measures(np.random.choice([0, 1], size=len(label)), label, 0.5)
        measures["name"] = "Year {}".format(i + 1)
        measures["method"] = "XGBoost"
        all_measures.append(measures)

        print("F1, Baseline: {}, XGBoost: {}".format(base_line_measures["f1"], measures["f1"]))
        print("Acc, Baseline: {}, XGBoost: {}".format(get_accuracy([1 for _ in range(len(label))], label),
                                                      measures["acc"]))
        print("AUC, Baseline: {}, XGBoost: {}".format(base_line_measures["auc"], measures["auc"]))
        print("Recall, Baseline: {}, XGBoost: {}".format(base_line_measures["recall"], measures["recall"]))
        print("Precision, Baseline: {}, XGBoost: {}".format(base_line_measures["precision"],
                                                                       measures["precision"]))
    get_roc_curve(probs_xgb[i], label)


if __name__ == "__main__":
    pd.set_option('chained_assignment', None)
    main()