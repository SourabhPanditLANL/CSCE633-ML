import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from Sourabh_Pandit_code import *

def main():
    print("Hello World!")

    loader = DataLoader("./", 42)
    df = loader.data
    print((df['pdays'] == 0).sum())

    loader.data_prep()
    #loader.data_split()


    #X_train, y_train = loader.extract_features_and_label(loader.data_train)
    #X_valid, y_valid = loader.extract_features_and_label(loader.data_valid)


    #decision_tree = ClassificationTree(random_state = 42);
    #decision_tree.build_tree(X_train, y_train)

if __name__ == "__main__":
     main()
