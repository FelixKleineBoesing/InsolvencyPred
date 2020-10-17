import pandas as pd
from typing import List


def get_class_dist_timeseries(labels: List[pd.Series]):
    """
    calculates the number of occurences of

    :param labels: labels of bankruptcy datasets. Must be chronological ordered
    :return:
    """
    stats = {"Customers": [], "Bankrupt customers": [], "Share of bankrupt customers": []}
    for label in labels:
        bankrupts = label.value_counts()[1]
        stats["Customers"].append(len(label))
        stats["Bankrupt customers"].append(bankrupts)
        stats["Share of bankrupt customers"].append(bankrupts / len(label))

    stats["Year"] = ["Year {}".format(i) for i in range(1, len(labels) + 1)]
    return pd.DataFrame(stats)


if __name__ == "__main__":
    dfs = [pd.Series([1, 0, 1]), pd.Series([1, 0, 1, 0])]
    print(get_class_dist_timeseries(dfs))

