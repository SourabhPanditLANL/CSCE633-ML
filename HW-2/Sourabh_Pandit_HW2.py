import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier


# Remove - begin
import sys                          ##TODO:  remove this
from itertools import product       ##TODO: REmove this

replace_unknown = [False, True]
max_depths = range(3, 8, 1)
sample_split_size = range(4, 20, 1)
use_entropy = [False] #, True]
drop_cols = [False, True]
hyp_list = list(product(replace_unknown, max_depths, sample_split_size, use_entropy, drop_cols))
hyp_idx = 100000

hardcoded = False       ##TODO: Delete this flag
# Remove - end

'''
General Instructions:

1. Do not use any additional libraries. Your code will be tested in a pre-built environment with only
the library above available.

2. You are expected to fill in the skeleton code precisely as per provided. On top of skeleton code given,
you may write whatever deemed necessary to complete the assignment. For example, you may define additional
default arguments, class parameters, or methods to help you complete the assignment.

3. Some initial steps or definition are given, aiming to help you getting started. As long as you follow
the argument and return type, you are free to change them as you see fit.

4. Your code should be free of compilation errors. Compilation errors will result in 0 marks.
'''


'''
Problem A-1: Data Preprocessing and EDA
'''
class DataLoader:
    # Static variables
    st_categorical_cols = []
    st_categorical_indices = []
    '''
    This class will be used to load the data and perform initial data processing. Fill in functions.
    You are allowed to add any additional functions which you may need to check the data. This class
    will be tested on the pre-built enviornment with only numpy and pandas available.
    '''

    def __init__(self, data_root: str, random_state: int):
        '''
        Inialize the DataLoader class with the data_root path.
        Load data as pd.DataFrame, store as needed and initialize other variables.
        All dataset should save as pd.DataFrame.
        '''
        self.random_state = random_state
        np.random.seed(self.random_state)

        self.data = pd.read_csv(f"{data_root}/bank.csv", delimiter=';')
        self.data_train = None
        self.data_valid = None

        ##TODO: Hyperparam and flag
        if hardcoded == True:
            self.replace_unknown = False
            self.drop_cols = False
        else:
            self.replace_unknown = hyp_list[hyp_idx][0]
            self.drop_cols = hyp_list[hyp_idx][4]


    def data_split(self) -> None:
        '''
        You are asked to split the training data into train/valid datasets on the ratio of 80/20.
        Add the split datasets to self.data_train, self.data_valid. Both of the split should still be pd.DataFrame.
        '''

        ## Train and Validation Split using Strata: Separate class 0 and class 1
        class_0 = self.data[self.data['y'] == 0]
        class_1 = self.data[self.data['y'] == 1]

        class_0 = class_0.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        class_1 = class_1.sample(frac=1, random_state=self.random_state).reset_index(drop=True)

        split_0 = int(0.8 * len(class_0))
        split_1 = int(0.8 * len(class_1))

        train_0 = class_0.iloc[:split_0]
        valid_0 = class_0.iloc[split_0:]

        train_1 = class_1.iloc[:split_1]
        valid_1 = class_1.iloc[split_1:]

        # Combine and shuffle
        self.data_train = pd.concat(
                [train_0, train_1]).sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        self.data_valid = pd.concat(
                [valid_0, valid_1]).sample(frac=1, random_state=self.random_state).reset_index(drop=True)


    def data_prep(self) -> None:
        '''
        You are asked to drop any rows with missing values and map categorical variables to numeric values.
        '''

        df = self.data.copy()                   # Work on a safe copy
        df.columns = df.columns.str.strip()     # Clean column names

        print(f"data_prep: shape before cleaning: {df.shape}")

        # Drop uninformative columns
        #TODO; check if duration and default should be dropped or kept

        for col in ['day', 'duration', 'default']:
            if self.drop_cols  and col in df.columns:
                df.drop(columns=[col], inplace=True)

        if self.replace_unknown:
            for col in df.select_dtypes(include=['object']).columns:
                unknown_count = (df[col] == 'unknown').sum()
                unknown_ratio = unknown_count / len(df)
                if unknown_count > 0 and unknown_ratio < 0.05:
                    mode = df[df[col] != 'unknown'][col].mode()
                    if not mode.empty:
                        df.loc[:, col] = df[col].replace('unknown', mode[0])

        df.dropna(inplace=True)                 # Drop any remaining NaNs
        df.reset_index(drop=True, inplace=True)
        print(f"data_prep: shape after dropna: {df.shape}")

        # Replace -1 in 'pdays' with 0 (or keep if model can learn from it)
        #TODO: is the even needed?
        #if 'pdays' in df.columns:
        #    df['pdays'] = df['pdays'].replace(-1, 0)

        #Encode categorical variables using pd.factorize
        for col in df.select_dtypes(include=['object']).columns:
            df[col], _ = pd.factorize(df[col])


        #TODO: enable/disable this
        #One Hot Encoding
        #df['y'] = df['y'].map({'yes': 1, 'no': 0})
        '''
        X = df.drop('y', axis=1)
        y = df['y']

        X_encoded = pd.get_dummies(X, drop_first=True)
        df = pd.concat([X_encoded, y], axis=1)
        '''

        DataLoader.st_categorical_cols = df.select_dtypes(include=['object']).columns
        DataLoader.st_categorical_indices = [df.columns.get_loc(col) for col in DataLoader.st_categorical_cols]

        print(f"data_prep: final shape: {df.shape}")
        self.data = df


    def extract_features_and_label(self, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        '''
        This function will be called multiple times to extract features and labels from train/valid/test
        data.

        Expected return:
            X_data: np.ndarray of shape (n_samples, n_features) - Extracted features
            y_data: np.ndarray of shape (n_samples,) - Extracted labels
        '''

        return (data['y'].values, data.drop(columns=['y']).values)

'''
Porblem A-2: Classification Tree Inplementation
'''
class ClassificationTree:
    '''
    You are asked to implement a simple classification tree from scratch. This class will be tested on the
    pre-built enviornment with only numpy and pandas available.

    You may add more variables and functions to this class as you see fit.
    '''
    class Node:
        '''
        A data structure to represent a node in the tree.
        '''
        def __init__(self, split=None, left=None, right=None, prediction=None):
            '''
            split: tuple - (feature_idx, split_value, is_categorical)
                - For numerical features: split_value is the threshold
                - For categorical features: split_value is a set of categories for the left branch
            left: Node - Left child node
            right: Node - Right child node
            prediction: (any) - Prediction value if the node is a leaf
            '''
            self.split = split
            self.left = left
            self.right = right
            self.prediction = prediction

            if split is not None:
                self.feature_idx = split[0]
                self.split_value = split[1]
                self.is_categorical = split[2]

        def is_leaf(self):
            return self.prediction is not None

    def __init__(self, random_state: int):

        self.random_state = random_state
        np.random.seed(self.random_state)
        self.tree_root = None
        self.categorical_indices = DataLoader.st_categorical_indices

        ##TODO: Hyperparam and flag
        if hardcoded == True:
            self.max_depth = 5
            self.min_samples_split = 10
            self.use_entropy = False
        else:
            self.max_depth = hyp_list[hyp_idx][1]
            self.min_samples_split = hyp_list[hyp_idx][2]
            self.use_entropy = hyp_list[hyp_idx][3]

        #print (f"CHECK: {self.max_depth}: {self.min_samples_split}: {self.use_entropy}")

    def split_crit(self, y: np.ndarray) -> float:
        '''
        Computes impurity of labels y using the specified method.

        Args:
            y (np.ndarray): array of labels
            method (str): "gini" or "entropy" (default: "entropy")

        Returns:
            float: impurity score
        '''

        if self.use_entropy == True:
            return self.entropy(y)
        else:
            return self.gini_index(y)


    def build_tree(self, X: np.ndarray, y: np.ndarray) -> None:
        '''
        Public method to initiate tree construction. Stores the root node.
        '''
        self.tree_root = self._build_tree_recursive(X, y, depth=0)


    def _build_tree_recursive(self, X: np.ndarray, y: np.ndarray, depth: int):
        '''
        Recursively builds the decision tree.

        Returns:
            Node - The constructed node (either split node or leaf node)
        '''

        # Pure node: all labels are the same
        if np.unique(y).size == 1:
            return self.Node(prediction=y[0])

        # Stop condition: minimum samples or maximum depth reached
        if len(y) < self.min_samples_split or depth >= self.max_depth:
            majority_class = np.bincount(y.astype(int)).argmax()
            return self.Node(prediction=majority_class)

        # Find the best split
        split_result = self.search_best_split(X, y)
        if split_result is None:
            majority_class = np.bincount(y.astype(int)).argmax()
            return self.Node(prediction=majority_class)

        feature_index, split_value, is_categorical = split_result

        # Apply the split
        X_left, y_left, X_right, y_right = self.split(X, y, feature_index, split_value, is_categorical)
        if X_left is None or X_right is None:
            majority_class = np.bincount(y.astype(int)).argmax()
            return self.Node(prediction=majority_class)

        # Recursively build the left and right subtrees
        left_child = self._build_tree_recursive(X_left, y_left, depth + 1)
        right_child = self._build_tree_recursive(X_right, y_right, depth + 1)

        # Return a split node
        return self.Node(split=(feature_index, split_value, is_categorical),
                         left=left_child,
                         right=right_child,
                         prediction=None)


    def search_best_split(self, X: np.ndarray, y: np.ndarray):
        '''
        Search for the best feature and split value.

        Returns:
            (feature_index, split_value, is_categorical) if a split is found,
            else None
        '''
        '''
        best_gain = -1
        best_split = None
        parent_impurity = self.split_crit(y)

        n_samples, n_features = X.shape

        for feature_index in range(n_features):
            feature_values = X[:, feature_index]
            unique_values = np.unique(feature_values)

            is_categorical = feature_index in self.categorical_indices

            if is_categorical:
                # Try 1-vs-rest splits
                for val in unique_values:
                    left_set = {val}
                    X_left, y_left, X_right, y_right = self.split(X, y, feature_index, left_set, is_categorical=True)
                    if X_left is None or X_right is None:
                        continue

                    n_left, n_right = len(y_left), len(y_right)
                    weighted_impurity = (
                        n_left * self.split_crit(y_left) +
                        n_right * self.split_crit(y_right)
                    ) / (n_left + n_right)

                    gain = parent_impurity - weighted_impurity
                    if gain > best_gain:
                        best_gain = gain
                        best_split = (feature_index, left_set, True)

            else:
                # Try thresholds between sorted values
                sorted_vals = np.sort(unique_values)
                for i in range(1, len(sorted_vals)):
                    threshold = (sorted_vals[i - 1] + sorted_vals[i]) / 2
                    X_left, y_left, X_right, y_right = self.split(X, y, feature_index, threshold, is_categorical=False)
                    if X_left is None or X_right is None:
                        continue

                    n_left, n_right = len(y_left), len(y_right)
                    weighted_impurity = (
                        n_left * self.split_crit(y_left) +
                        n_right * self.split_crit(y_right)
                    ) / (n_left + n_right)

                    gain = parent_impurity - weighted_impurity
                    if gain > best_gain:
                        best_gain = gain
                        best_split = (feature_index, threshold, False)

        return best_split  # or None if no good split found
        '''

    def search_best_split(self, X: np.ndarray, y: np.ndarray):
        best_gain = -1
        best_split = None

        n_samples, n_features = X.shape

        for feature_index in range(n_features):
            feature_values = X[:, feature_index]
            unique_values = np.unique(feature_values)

            is_categorical = feature_index in self.categorical_indices

            if is_categorical:
                for val in unique_values:
                    left_set = {val}
                    X_left, y_left, X_right, y_right = self.split(X, y, feature_index, left_set, is_categorical=True)
                    if X_left is None or X_right is None:
                        continue

                    gain = self.information_gain(y, y_left, y_right)

                    if gain > best_gain:
                        best_gain = gain
                        best_split = (feature_index, left_set, True)

            else:
                sorted_vals = np.sort(unique_values)
                for i in range(1, len(sorted_vals)):
                    threshold = (sorted_vals[i - 1] + sorted_vals[i]) / 2
                    X_left, y_left, X_right, y_right = self.split(X, y, feature_index, threshold, is_categorical=False)
                    if X_left is None or X_right is None:
                        continue

                    gain = self.information_gain(y, y_left, y_right)

                    if gain > best_gain:
                        best_gain = gain
                        best_split = (feature_index, threshold, False)

        return best_split

        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        '''
        Predict classes for multiple samples.

        Args:
            X: numpy array with the same columns as the training data

        Returns:
            np.ndarray: Array of predictions
        '''
        return np.array([self._predict_one(x, self.tree_root) for x in X])

    def _predict_one(self, x: np.ndarray, node) -> int:
        '''
        Recursively predict the class for a single sample.

        Args:
            x (np.ndarray): Single input sample
            node (Node): Current node in the tree

        Returns:
            int: Predicted class
        '''
        if node.is_leaf():
            return node.prediction

        if node.is_categorical:
            if x[node.feature_idx] in node.split_value:
                return self._predict_one(x, node.left)
            else:
                return self._predict_one(x, node.right)
        else:
            if x[node.feature_idx] <= node.split_value:
                return self._predict_one(x, node.left)
            else:
                return self._predict_one(x, node.right)



    def entropy(self, y):
        entropy = 0
        class_labels = np.unique(y)

        for cls in class_labels:
            p_cls = len(y[y == cls])/len(y)
            entropy -= p_cls * np.log2(p_cls)

        return entropy

    def gini_index(self, y):
        gini = 0
        class_labels = np.unique(y)

        for cls in class_labels:
            p_cls = len(y[y == cls])/len(y)
            gini += p_cls**2

        return (1 - gini)

    def split(self, X: np.ndarray, y: np.ndarray, feature_index: int, split_value, is_categorical: bool):
        '''
        Splits the dataset (X, y) based on the feature at feature_index.

        If is_categorical is True:
            - split_value is a set of category values for the left branch
        If is_categorical is False:
            - split_value is a numeric threshold

        Returns:
            X_left, y_left, X_right, y_right
        '''
        dataset = np.concatenate((X, y.reshape(-1, 1)), axis=1)

        if is_categorical:
            left_rows = [row for row in dataset if row[feature_index] in split_value]
            right_rows = [row for row in dataset if row[feature_index] not in split_value]
        else:
            left_rows = [row for row in dataset if row[feature_index] <= split_value]
            right_rows = [row for row in dataset if row[feature_index] > split_value]

        dataset_left = np.array(left_rows)
        dataset_right = np.array(right_rows)

        if len(dataset_left) == 0 or len(dataset_right) == 0:
            return None, None, None, None

        X_left, y_left = dataset_left[:, :-1], dataset_left[:, -1]
        X_right, y_right = dataset_right[:, :-1], dataset_right[:, -1]

        return X_left, y_left, X_right, y_right


    def variance(y):
        return y.var()

    def information_gain(self, y, y_left, y_right):
        '''
        Computes information gain from a proposed split.

        Args:
            y: Full label array before the split
            y_left, y_right: Label arrays after split
            method: "gini" or "entropy"

        Returns:
            Information gain (float)
        '''
        parent_impurity = self.split_crit(y)
        n = len(y)
        n_left, n_right = len(y_left), len(y_right)

        if n_left == 0 or n_right == 0:
            return 0.0

        child_impurity = (
            n_left * self.split_crit(y_left) +
            n_right * self.split_crit(y_right)
        ) / n

        return parent_impurity - child_impurity



def train_XGBoost() -> dict:
    '''
    See instruction for implementation details. This function will be tested on the pre-built enviornment
    with numpy, pandas, xgboost available.
    '''
    pass


'''
Initialize the following variable with the best model you have found. This model will be used in testing
in our pre-built environment.
'''
my_best_model = XGBClassifier()


def precision(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def recall(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def compute_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    '''
    Compute the F1-score for binary classification (0/1 labels).

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        F1-score as a float
    '''
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    if tp + fp == 0 or tp + fn == 0:
        return 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1






def main():
    global hyp_idx

    for hyp_idx in range(0,len(hyp_list)):
        loader = DataLoader("./", 42)
        loader.data_prep()
        loader.data_split()

        X_train, y_train = loader.extract_features_and_label(loader.data_train)
        X_valid, y_valid = loader.extract_features_and_label(loader.data_valid)


        decision_tree = ClassificationTree(random_state = 42);
    exit(0)

    decision_tree.build_tree(X_train, y_train)
    y_pred = decision_tree.predict(X_valid)

    accuracy  = (y_pred == y_valid).mean()
    prec = precision(y_valid, y_pred)
    rec =  recall(y_valid, y_pred)
    f1 = compute_f1_score(y_valid, y_pred)
    #print("HYP: Validation Accuracy:", accuracy)

    #print(f"HYP: Validation F1-score: {f1:.4f} for {hyp_list[arg_idx]}")

    #print("HYP: Precision:", precision(y_valid, y_pred))
    #print("HYP: Validatio: Recall:", recall(y_valid, y_pred))

    print(f"HYP: A/P/R/F1: {accuracy:.4f}, {prec:.4f}, {rec:.4f}, {f1:.4f} for {hyp_list[arg_idx]}")
    print("Hello World!")

if __name__ == "__main__":
     main()
