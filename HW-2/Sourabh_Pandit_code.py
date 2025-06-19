import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

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

        #TODO: Q1:Should init method also load the data and inititlize the data frame?
        self.data = pd.read_csv(f"{data_root}/bank.csv", delimiter=';') # .data = pd.DataFrame()
        self.data_train = None
        self.data_valid = None
        self.debug_print("init()")

    def data_split(self) -> None:
        '''
        You are asked to split the training data into train/valid datasets on the ratio of 80/20.
        Add the split datasets to self.data_train, self.data_valid. Both of the split should still be pd.DataFrame.
        '''

        # Shuffle the data
        shuffled_data = self.data.sample(frac=1, random_state=self.random_state).reset_index(drop=True)

        split_idx = int(0.8 * len(shuffled_data))
        self.data_train = shuffled_data.iloc[:split_idx]
        self.data_valid = shuffled_data.iloc[split_idx:]
        self.debug_print("data_split()")

    def data_prep(self) -> None:
        '''
        You are asked to drop any rows with missing values and map categorical variables to numeric values.
        '''
        print(f"data_prep: shape before : {self.data.shape}")

        # Replace "unknown" with mode in each categorical column
        self.debug_print("data_prep()-1 ")
        self.data = self.data.copy()  # ← Add this line first

        categorical_cols = self.data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'poutcome' and col != 'contact':
                mode = self.data[self.data[col] != 'unknown'][col].mode()
                if not mode.empty:
                    self.data.loc[:, col] = self.data[col].replace('unknown', mode[0])

        self.debug_print("data_prep()-2 ")

        self.data.dropna(inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        print(f"data_prep: shape after: {self.data.shape}")
        #self.data = self.data[~self.data.isin(['unknown']).any(axis=1)]
        print(f"data_prep: shape after after: {self.data.shape}")

        # Encode categorical variables to integers
        for col in categorical_cols:
            self.data[col], _ = pd.factorize(self.data[col])

        self.debug_print("data_prep()-3")


    def extract_features_and_label(self, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        '''
        This function will be called multiple times to extract features and labels from train/valid/test
        data.

        Expected return:
            X_data: np.ndarray of shape (n_samples, n_features) - Extracted features
            y_data: np.ndarray of shape (n_samples,) - Extracted labels
        '''

        y_data = data['y'].values
        X_data = data.drop(columns=['y']).values
        print(f"Shape: {X_data.shape}, {y_data.shape}")
        return (X_data, y_data)


    def debug_print(self, caller_func):
        print(f"\nDEBUG-1: {caller_func} - self.data = {self.data.shape}")
        if (self.data_train is None or self.data_valid is None):
            print(f"DEBUG-2: {caller_func} - self.data_train = {type(self.data_train)}: self.data_valid = {type(self.data_valid)}")
        else:
            print(f"DEBUG-2: {caller_func} - self.data_train = {self.data_train.shape}: self.data_valid = {self.data_valid.shape}")

        #for col in self.data.columns:
        #    print (f"\nDEBUG-3: {caller_func} - :\n{self.data[col].value_counts(dropna=False)}")

        print(f"DEBUG-3: {caller_func}: Remaining 'unknown's: {(self.data == 'unknown').sum().sum()}")

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

            self.feature_idx = split[0]
            self.split_value = split[1]
            self.is_categorical = split[2]

        def is_leaf(self):
            return self.prediction is not None

    def __init__(self, random_state: int):

        self.random_state = random_state
        np.random.seed(self.random_state)
        self.max_depth = 3
        self.min_samples_split = 2

        self.tree_root = None

    def split_crit(self, y: np.ndarray) -> float:
        '''
        Implement the impurity measure of your choice here. Return the impurity value.
        '''
        return self.entropy(y)

    def build_tree(self, X: np.ndarray, y: np.ndarray) -> None:
        '''
        Implement the tree building algorithm here. You can recursivly call this function to build the
        tree. After building the tree, store the root node in self.tree_root.
        '''

        num_samples, num_features = np.shape(X)

        # data_split until stopping conditions are met

        #if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
        if num_samples >= self.min_samples_split:
            # find the best data_split
            best_split = self.search_best_split(X, y)

            # check if information gain is positive
            if best_split["info_gain"]>0:
                # recur left
                #left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                left_subtree = self.build_tree(best_split["dataset_left"])
                # recur right
                #right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                right_subtree = self.build_tree(best_split["dataset_right"])
                # return decision node
                return Node(best_split["feature_idx"], best_split["threshold"],
                            left_subtree, right_subtree, best_split["info_gain"])

        # compute leaf node
        leaf_value = self.calculate_leaf_value(y)
        # return leaf node
        return Node(value=leaf_value)


    def search_best_split(self, X: np.ndarray, y: np.ndarray):
        '''
        Implement the search for best split here.

        Expected return:
        - tuple(int, float): Best feature index and split value
        - None: If no split is found
        '''

        # dictionary to store the best data_split
        best_split = {}
        max_info_gain = -float("inf")
        num_samples, num_features = np.shape(X)

        # loop over all the features
        for feature_idx in range(num_features):
            feature_values = X[:, feature_idx]
            possible_thresholds = np.unique(feature_values)
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # get current data_split
                dataset_left, dataset_right = self.data_split(dataset, feature_idx, threshold)
                # check if childs are not null
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # compute information gain
                    curr_info_gain = self.split_crit(y, left_y, right_y, "gini")
                    # update the best data_split if needed
                    if curr_info_gain>max_info_gain:
                        best_split["feature_idx"] = feature_idx
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain

        # return best data_split
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
        pass

    def entropy(seld, y):
        entropy = 0
        class_labels = np.unique(y)

        for cls in class_labels:
            p_cls = len(y[y == cls])/len(y)
            entropy += p_cls * np.logs(p_cls)

        return entropy

    def gini_index(self, y):
        gini = 0
        class_labels = np.unique(y)

        for cls in class_labels:
            p_cls = len(y[y == cls])/len(y)
            gini += p_cls**2

        return (1 - gini)

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

