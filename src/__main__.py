import pandas as pd
from cleaning import load_arff_files
from modelling import LogisticRegression, XGBoost, CrossValidation
from preprocessing import Standardizer, MeanReplacement, clip_outliers


def main():
    predictions_lr = []

    dfs, file_names = load_arff_files("../data/raw_data/")

    cv = CrossValidation(folds=5)
    for df in dfs:
        df = df.copy()
        label = df.pop("class")
        # this is to kick only to kick most extreme outlier
        df = clip_outliers(df.copy(), 5.5)
        model = LogisticRegression(max_iter=200)
        predictions_lr.append(cv.run(data=df, label=label, model=model, preprocessors=[MeanReplacement(),
                                                                                       Standardizer()]))






if __name__ == "__main__":
    pd.set_option('chained_assignment', None)
    main()