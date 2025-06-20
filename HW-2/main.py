import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from Sourabh_Pandit_code import *
import Sourabh_Pandit_code

import sys

##TODO:  remove this
from itertools import product
##TODO: Hyperparam index
hyp_param_list_idx = None



replace_options = [False, True]
max_depths = range(2, 10, 1)
sample_splits = range(2, 10, 1)
entropy_options = [False, True]
hyp_param_list = list(product(replace_options,max_depths, sample_splits, entropy_options))


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 main.py <hyp_param_list_idx>")
        return

    # Parse the hyp_param_list_idx from command line
    try:
        arg_idx = int(sys.argv[1])
    except ValueError:
        print("Index must be an integer.")
        return

    # Set the global var in Sourabh_Pandit_code.py
    Sourabh_Pandit_code.hyp_param_list_idx = arg_idx
    Sourabh_Pandit_code.hyp_param_list = hyp_param_list

    #print (f"Main hyp_index = {arg_idx}")
    #print (f"Main SP hyp_index = {Sourabh_Pandit_code.hyp_param_list_idx }")
    #print (f"Main SP hyp_list len = {len(hyp_param_list)}")

    loader = DataLoader("./", 42)

    loader.data_prep()
    #print(DataLoader.st_categorical_cols)
    #print(DataLoader.st_categorical_indices)
    #print(DataLoader.st_non_categorical_cols)
    #print(DataLoader.st_non_categorical_indices)

    loader.data_split()

    X_train, y_train = loader.extract_features_and_label(loader.data_train)
    X_valid, y_valid = loader.extract_features_and_label(loader.data_valid)

    #print(f"Train shape: {X_train.shape}, {y_train.shape}")
    #print(f"Valid shape: {X_valid.shape}, {y_valid.shape}")

    decision_tree = ClassificationTree(random_state = 42);

    decision_tree.build_tree(X_train, y_train)
    y_pred = decision_tree.predict(X_valid)

    accuracy  = (y_pred == y_valid).mean()
    prec = precision(y_valid, y_pred)
    rec =  recall(y_valid, y_pred)
    f1 = compute_f1_score(y_valid, y_pred)
    #print("HYP: Validation Accuracy:", accuracy)

    #print(f"HYP: Validation F1-score: {f1:.4f} for {hyp_param_list[arg_idx]}")

    #print("HYP: Precision:", precision(y_valid, y_pred))
    #print("HYP: Validatio: Recall:", recall(y_valid, y_pred))

    print(f"HYP: A/P/R/F1: {accuracy:.4f}, {prec:.4f}, {rec:.4f}, {f1:.4f} for {hyp_param_list[arg_idx]}")
    print("Hello World!")

if __name__ == "__main__":
     main()
