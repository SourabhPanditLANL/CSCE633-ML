import numpy as np
from xgboost import XGBClassifier
from Sourabh_Pandit_HW2 import ClassificationTree, DataLoader, train_XGBoost, my_best_model


def run_train_predict_with_decision_tree():

    loader = DataLoader("./bank.csv", 42)
    loader.data_prep()
    loader.data_split()
    print("\nFirst 10 rows/samples of the training data")
    print(loader.data_train.head(11))

    print("\nNow actually running train and predict with my Decison Tree")

    X_train, y_train = loader.extract_features_and_label(loader.data_train)
    X_valid, y_valid = loader.extract_features_and_label(loader.data_valid)

    print(f"Main X_train {X_train.shape}, y_train {y_train.shape}", flush=True)
    print(f"Main X_valid {X_valid.shape}, y_valid {y_valid.shape}", flush=True)

    decision_tree = ClassificationTree(random_state = 42);
    decision_tree.build_tree(X_train, y_train)

    y_pred = decision_tree.predict(X_valid)

    accuracy  = (y_pred == y_valid).mean()
    prec = ClassificationTree.precision(y_valid, y_pred)
    rec =  ClassificationTree.recall(y_valid, y_pred)
    f1 = ClassificationTree.compute_f1_score(y_valid, y_pred)

    #print(f"\tSimple Train and Predict Results: Accuracy Precison Recall F1-Score: ",
    #      f"\n{accuracy:8.4f}, {prec:8.4f}, {rec:8.4f}, {f1:8.4f}"
    #      , flush=True)
    print(f"\tAccuracy  : {accuracy:.4f}")
    print(f"\tPrecision : {prec:.4f}")
    print(f"\tRecall    : {rec:.4f}")
    print(f"\tF1 Score  : {f1:.4f}", flush=True)

def run_grid_search():
    ClassificationTree.grid_search()

def run_xgboost():

    print(f"\n\tSearch for best alpha with XG boost")
    results = train_XGBoost()
    best_alpha = results["best_alpha"]

    loader = DataLoader("./bank.csv", random_state=42)
    loader.data_prep()
    loader.data_split()

    X_train, y_train = loader.extract_features_and_label(loader.data_train)
    X_valid, y_valid = loader.extract_features_and_label(loader.data_valid)

    # Initialize XGBClassifier with the best_alpha returned from train_XGBoost()
    print(f"\nNow Running the best XG Boost model with alpha = {best_alpha}")
    #my_best_model = XGBClassifier(
    #    max_depth = 5,
    #  n_estimators=100,
    #   eval_metric='logloss',
    #   reg_lambda=best_alpha,
    #   random_state=42,
    #   n_jobs = 1
    #

    my_best_model.fit(X_train, y_train)

    # Predict
    y_pred = my_best_model.predict(X_valid)

    # Evaluate
    accuracy = (y_pred == y_valid).mean()
    prec = ClassificationTree.precision(y_valid, y_pred)
    rec = ClassificationTree.recall(y_valid, y_pred)
    f1 = ClassificationTree.compute_f1_score(y_valid, y_pred)

    print(f"\n\tmy_best_model Evaluation:")
    print(f"\t\tAccuracy  : {accuracy:.4f}")
    print(f"\t\tPrecision : {prec:.4f}")
    print(f"\t\tRecall    : {rec:.4f}")
    print(f"\t\tF1 Score  : {f1:.4f}")

    # Predict probabilities for positive class
    y_prob = my_best_model.predict_proba(X_valid)[:, 1]
    ClassificationTree.plot_roc_curve(y_valid, y_prob, "roc_auc.png")

def main():
    #print("\nFirst running train and predict with my Decison Tree")
    #run_train_predict_with_decision_tree()

    #print("\nSecond: Running XG boost")
    #run_xgboost()

    print("\nFinally: GridSearch for hyperparame tuning for -Accuracy, Precision, Recall, F1-Score")
    run_grid_search()


if __name__ == "__main__":
     main()
