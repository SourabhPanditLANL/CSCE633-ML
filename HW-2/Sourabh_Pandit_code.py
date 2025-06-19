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

        self.data = pd.read_csv(f"{data_root}/bank.csv", delimiter=';')
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
        df = self.data.copy()  # Work on a safe copy

        # Clean column names
        df.columns = df.columns.str.strip()

        # Drop leakage-prone/uninformative columns
        #TODO; check if duration and default should be dropped or kept
        cols_to_drop = ['day', 'duration', 'default']
        for col in cols_to_drop:
            if col in df.columns:
                print(f"DROPPING: {col}")
                df.drop(columns=[col], inplace=True)

        print(f"data_prep: shape before cleaning: {df.shape}")
        total = len(df)

        # Replace "unknown" if it occurs in <5% of a column
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            unknown_count = (df[col] == 'unknown').sum()
            unknown_ratio = unknown_count / total
            if unknown_count > 0 and unknown_ratio < 0.05:
                mode = df[df[col] != 'unknown'][col].mode()
                if not mode.empty:
                    df.loc[:, col] = df[col].replace('unknown', mode[0])

        # Drop any remaining NaNs
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        print(f"data_prep: shape after dropna: {df.shape}")

        # Replace -1 in 'pdays' with 0 (or keep if model can learn from it)
        if 'pdays' in df.columns:
            df['pdays'] = df['pdays'].replace(-1, 0)

        # Optionally store column type indices
        self.categorical_cols = df.select_dtypes(include=['object']).columns
        self.categorical_indices = [df.columns.get_loc(col) for col in self.categorical_cols]

        self.non_categorical_cols = df.columns.difference(self.categorical_cols)

        # Encode categorical variables using pd.factorize
        for col in df.select_dtypes(include=['object']).columns:
            df[col], _ = pd.factorize(df[col])

        # Final assignment back to self
        self.data = df
        self.non_categorical_indices = [df.columns.get_loc(col) for col in self.non_categorical_cols]

        print(f"data_prep: final shape: {df.shape}")

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


    def debug_print(self, caller):
        print(f"\nDEBUG-1: {caller} - self.data = {self.data.shape}")
        if (self.data_train is None or self.data_valid is None):
            print(f"DEBUG-2: {caller} - train = {type(self.data_train)}: valid = {type(self.data_valid)}")
        else:
            print(f"DEBUG-2: {caller} - train = {self.data_train.shape}: valid = {self.data_valid.shape}")

        #for col in self.data.columns:
        #    print (f"\nDEBUG-3: {caller} - :\n{self.data[col].value_counts(dropna=False)}")

        print(f"DEBUG-3: {caller}: Remaining 'unknown's: {(self.data == 'unknown').sum().sum()}")

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

    #def split_crit(self, y: np.ndarray) -> float:
    def split_crit(self, y: np.ndarray, method: str = "gini") -> float:
        '''
        Computes impurity of labels y using the specified method.

        Args:
            y (np.ndarray): array of labels
            method (str): "gini" or "entropy" (default: "gini")

        Returns:
            float: impurity score
        '''
        if method == "entropy":
            return self.entropy(y)
        else:
            return self.gini_index(y)

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

    def Information_gain(y, mask, func=entropy):
        pass



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

