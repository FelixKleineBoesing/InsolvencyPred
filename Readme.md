# Data Science Assessment (Bankruptcy)

### Setup

This project is build with python 3.7. Click [here](https://www.python.org/downloads/release/python-370/) to download python 3.7.

```
git clone https://github.com/FelixKleineBoesing/AccentureApplication.git
cd AccentureApplication
pip3 install -r reqirements.txt
jupyter notebook
```

Now put the .arff files inside the directory data/raw_data. If it doesn´t exist, you should create the dir.

Then open the jupyter notebook with http://localhost:8888 in your preffered browser (I hope that its not IE :) )

Note for Mac OS User: If you are using python 3.8 and the latest Mac Os you will perhaps encounter a pickle problem
in the GreedyForwardSelector. It should work since I put the task on module level, but I´m not 100% sure. 
You can still use the Selector with max_processes = 1


### Overview

__The main files are:__

- \_\_main__.py (Main Script for the training of xgboost and logistic regression)
- \_\_main_fs__.py (Main Script for Feature Selection)
- Main.ipynb (Notebook which include both scripts above as well as exploration and evaluation)

__These scripts use some custom functions that are located in:__

- cleaning.py (import and cleans the raw data)
- evaluation.py (measurements)
- feature_selecting.py (Feature Selectors)
- misc.py (helper functions)
- modelling.py (Algorithms and CV Framekwork)
- preprocessing.py (Preprocessing functions like PCA, Resampling, etc.)
- visualization.py (roc curve)

