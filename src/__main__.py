import copy

import pandas as pd
import numpy as np
import json
import os
from cleaning import load_arff_files
from modelling import LogisticRegression, XGBoost, CrossValidation
from preprocessing import Standardizer, MeanReplacement, clip_outliers, ReSampler, CorrelationRemover, PCA
from evaluation import get_f1_score, get_accuracy, get_auc, get_precision, get_recall, get_all_measures, \
    get_threshold_for_optim_cost, get_weighted_accuracy
from visualization import get_roc_curve


def main_lr(cost_weight=1, primary_measure: str = "Weighted Accuracy"):
    probs_lr = []
    params = {"model": {"max_iter": 1000}, "preprocessors": [MeanReplacement(), Standardizer(), PCA(0.999),
                                                             ReSampler("down")]}

    dfs, file_names = load_arff_files("../data/raw_data/")

    cv = CrossValidation(folds=10)
    all_measures = {}
    if os.path.exists("../data/cleaned_data/recorded_measures_lr.json"):
        with open("../data/cleaned_data/recorded_measures_lr.json", "r") as f:
            recorded_measures = json.load(f)
    else:
        recorded_measures = None

    for i, df in enumerate(dfs):
        print("#####   {} Year   ####".format(i + 1))
        df = df.copy()
        label = df.pop("class")
        # this is to kick only to kick most extreme outlier
        df = clip_outliers(df.copy(), 5.5)
        model = LogisticRegression(max_iter=params["model"]["max_iter"])

        probs = cv.run(data=df, label=label, model=model,
                       preprocessors=params["preprocessors"])
        probs_lr.append(probs)
        measures = get_all_measures(probs, label, 0.5)
        measures["importance"] = model._model.coef_.tolist()
        base_line_measures = get_all_measures(np.random.choice([0, 1], size=len(label)), label, 0.5)
        print("F1, Baseline: {}, LogisticRegression: {}".format(base_line_measures["f1"], measures["f1"]))
        print("Acc, Baseline: {}, LogisticRegression: {}".format(base_line_measures["acc"], measures["acc"]))
        print("AUC, Baseline: {}, LogisticRegression: {}".format(base_line_measures["auc"], measures["auc"]))
        print("Recall, Baseline: {}, LogisticRegression: {}".format(base_line_measures["recall"], measures["recall"]))
        print("Precision, Baseline: {}, LogisticRegression: {}".format(base_line_measures["precision"],
                                                                       measures["precision"]))

        threshold, costs = get_threshold_for_optim_cost(probs, label, weight=cost_weight)

        print("Threshold: {}".format(threshold))
        print("Costs: {}".format(costs))
        weighted_accuracy = get_weighted_accuracy([1 if p > threshold else 0 for p in probs], label, cost_weight)
        measures["Weighted Accuracy"] = weighted_accuracy
        measures["params"] = copy.deepcopy(params)
        measures["params"]["preprocessors"] = [(p.__class__.__name__, str(p.__dict__)) for p in params["preprocessors"]]
        measures["probs"] = probs.tolist()
        measures["label"] = label.tolist()
        print("Weighted Accuracy, Baseline: {}, LogisticRegression: {}".format(
            get_weighted_accuracy(np.random.choice([0, 1], len(label)), label, cost_weight),
            weighted_accuracy
        ))
        print("All measures with cost optimal threshold: {}".format(get_all_measures(probs, label, threshold)))
        all_measures["LR Year {}".format(i + 1)] = measures
        if recorded_measures is not None:
            if measures[primary_measure] > np.max([m["LR Year {}".format(i + 1)][primary_measure]
                                                   for m in recorded_measures]):
                print("Found new best measure {} with value {}".format(primary_measure, measures[primary_measure]))

    if recorded_measures is None:
        recorded_measures = [all_measures]
    else:
        recorded_measures.append(all_measures)
    with open("../data/cleaned_data/recorded_measures_lr.json", "w") as f:
        json.dump(recorded_measures, f)

    get_roc_curve(probs_lr[i], label)


def main_xgb(cost_weight=1, primary_measure: str = "Weighted Accuracy"):
    probs_xgb = []
    dfs, file_names = load_arff_files("../data/raw_data/")
    params = {"model": dict(val_share=0.2, n_rounds=8, lambda_=5,
                            additional_booster_params={"params": {"max_depth": 4, "subsample": 0.5,
                                                                  "colsample_bytree": 0.5}},
                            verbose=True), "preprocessors": []}

    cv = CrossValidation(folds=10)
    all_measures = {}
    if os.path.exists("../data/cleaned_data/recorded_measures_xgb.json"):
        with open("../data/cleaned_data/recorded_measures_xgb.json", "r") as f:
            recorded_measures = json.load(f)
    else:
        recorded_measures = None

    for i, df in enumerate(dfs):
        print("#####   {} Year   ####".format(i + 1))
        df = df.copy()
        label = df.pop("class")
        model = XGBoost(**params["model"]) #, "scale_pos_weight": (df.shape[0] - np.sum(np.sum(df["class])) / np.sum(df["class])
        probs = cv.run(data=df, label=label, model=model, preprocessors=params["preprocessors"])
        probs_xgb.append(probs)

        measures = get_all_measures(probs, label, 0.5)
        base_line_measures = get_all_measures(np.random.choice([0, 1], size=len(label)), label, 0.5)
        measures["importance"] = model._model.get_score(importance_type="gain")

        print("F1, Baseline: {}, XGBoost: {}".format(base_line_measures["f1"], measures["f1"]))
        print("Acc, Baseline: {}, XGBoost: {}".format(get_accuracy([1 for _ in range(len(label))], label),
                                                      measures["acc"]))
        print("AUC, Baseline: {}, XGBoost: {}".format(base_line_measures["auc"], measures["auc"]))
        print("Recall, Baseline: {}, XGBoost: {}".format(base_line_measures["recall"], measures["recall"]))
        print("Precision, Baseline: {}, XGBoost: {}".format(base_line_measures["precision"],
                                                            measures["precision"]))
        threshold, costs = get_threshold_for_optim_cost(probs, label, weight=cost_weight)
        print("Threshold: {}".format(threshold))
        print("Costs: {}".format(costs))
        weighted_accuracy = get_weighted_accuracy([1 if p > threshold else 0 for p in probs], label, cost_weight)
        measures["Weighted Accuracy"] = weighted_accuracy
        measures["params"] = copy.deepcopy(params)
        measures["params"]["preprocessors"] = [(p.__class__.__name__, p.__dict__) for p in params["preprocessors"]]
        measures["probs"] = probs.tolist()
        measures["label"] = label.tolist()
        print("Weighted Accuracy, Baseline: {}, XGB: {}".format(
            get_weighted_accuracy(np.random.choice([0, 1], len(label)), label, cost_weight),
            weighted_accuracy
        ))
        print("All measures with cost optimal threshold: {}".format(get_all_measures(probs, label, threshold)))
        all_measures["XGB Year {}".format(i + 1)] = measures
        if recorded_measures is not None:
            if measures[primary_measure] > np.max([m["XGB Year {}".format(i + 1)][primary_measure]
                                                   for m in recorded_measures]):
                print("Found new best measure {} with value {}".format(primary_measure, measures[primary_measure]))

    if recorded_measures is None:
        recorded_measures = [all_measures]
    else:
        recorded_measures.append(all_measures)
    with open("../data/cleaned_data/recorded_measures_xgb.json", "w") as f:
        json.dump(recorded_measures, f)

    get_roc_curve(probs_xgb[i], label)


if __name__ == "__main__":
    pd.set_option('chained_assignment', None)
    cost_weight = 20
    main_lr(cost_weight=cost_weight)
    #main_xgb(cost_weight=cost_weight)