from scipy.io import arff
import pandas as pd
import numpy as np
import os


def load_arff_files(dir: str = "data/raw_data/",
                    attribute_file_path: str = "../data/raw_data/attribute_information.txt"):
    """
    loads and returns all arff files in the given dir

    :param dir: directory path which stores the arff files
    :param attribute_file_path: path to attribute file that describes the column names
    :return:
    """
    column_names = _get_column_name_mapping(attribute_file_path)

    files = [dir + file for file in os.listdir(dir) if ".arff" in file]
    arrf_data = [arff.loadarff(file) for file in files]
    dataframes = [pd.DataFrame(data[0]) for data in arrf_data]
    for i, frame in enumerate(dataframes):
        frame["class"] = frame["class"].astype(int)
        indices = np.arange(frame.shape[0])
        np.random.shuffle(indices)
        frame = frame.iloc[indices, :]
        frame.rename(column_names, axis=1, inplace=True)
        frame = frame.reset_index(drop=True)
        dataframes[i] = frame
    return dataframes, files


def _get_column_name_mapping(file_path: str):
    with open(file_path, "r") as f:
        column_names = {}
        for line in f.readlines():
            name_short, name_long = line.split("\t")
            name_long = name_long.replace("\n", "")
            column_names[name_short] = name_long # "".join([c for c in name_long if c.isalnum()])
    column_names = {"Attr" + key[1:]: val for key, val in column_names.items()}
    return column_names


if __name__ == "__main__":
    frames, file_names = load_arff_files("../data/raw_data/")
