import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from typing import Tuple, List

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

class DataProcessor:
    def __init__(self, data_root: str):
        """Initialize data processor with paths to train and test data.

        Args:
            data_root: root path to data directory
        """
        self.data_root = data_root

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load training and test data from CSV files.

        Returns:
            Tuple containing training and test dataframes
        """
        # TODO: Implement data loading
        train_data_csv = self.data_root + "/" + "data_train_25s.csv"
        test_data_csv = self.data_root  + "/" + "data_test_25s.csv"

        df_train = pd.read_csv(train_data_csv)
        df_test = pd.read_csv(test_data_csv)

        return (df_train, df_test)

    def check_missing_values(self, data: pd.DataFrame) -> int:
        """Count number of missing values in dataset.

        Args:
            data: Input dataframe

        Returns:
            Number of missing values
        """
        # TODO: Implement missing value check
        return data.isnull().sum().sum()

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with missing values.

        Args:
            data: Input dataframe

        Returns:
            Cleaned dataframe
        """
        # TODO: Implement data cleaning
        data.dropna(inplace=True)
        data.reset_index(drop=True, inplace=True)
        return data

    def extract_features_labels(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features and labels from dataframe, convert to numpy arrays.

        Args:
            data: Input dataframe

        Returns:
            Tuple of feature matrix X and label vector y
        """
        # TODO: Implement feature/label extraction

        X = data.iloc[:, :-1].to_numpy()
        y = data.iloc[:, -1].to_numpy()
        return (X, y)


    def get_pearson_corr(self, data: pd.DataFrame) -> None:
        """Return Pearson correlation coefficient for two features

        Args:
            data: Input dataframe
            features1_idx: index for features 1
            features2_idx: index for features 2

        Returns:
            returns nothing
        """

        # Pearson's correlation matrix
        corr_matrix = data.corr(method='pearson')
        plt.figure(figsize=(12, 10))

        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": .75}
        )

        plt.title("Pearson Correlation Heatmap", fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig("feature_heatmap.png", dpi=300, bbox_inches='tight')
        plt.show()

    def draw_histogram(self, data: pd.DataFrame) -> None:
        """Draw histogram for all features and the target variable

        Args:
            data: Input dataframe

        Returns:
            returns Nothing
        """
        fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 12))  # 4 rows x 3 columns
        axes = axes.flatten()  # Flatten to 1D array for easy indexing

        for i, feature in enumerate(data.columns):
            axes[i].hist(data[feature], bins=20, edgecolor='black')
            axes[i].set_title(f'Histogram of {feature}')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Frequency')

        plt.tight_layout()
        plt.savefig("features_histogram.png", dpi=300, bbox_inches='tight')
        plt.show()

    def draw_scatter_plot(self, data: pd.DataFrame, idx1: int, idx2: int) -> None:
        """Extract features and labels from dataframe, convert to numpy arrays.

        Args:
            data: Input dataframe
            features1_idx: index for features 1
            features2_idx: index for features 2

        Returns:
            returns Nothing
        """
        feature1 = data.columns[idx1]
        feature2 = data.columns[idx2]

        plt.figure(figsize=(8, 6))
        plt.scatter(data[feature1], data[feature2], alpha=0.6, edgecolors='k')

        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.title(f'Scatter Plot: {feature1} vs {feature2}')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("scatter_plot.png", dpi=300, bbox_inches='tight')
        plt.show()

    def normalize_ndarray(self, X: np.ndarray) -> np.ndarray:
        """
            Normalize each feature (column) in the NumPy array to the range [0, 1].

        Args:
            X: NumPy array of shape (n_samples, n_features)

        Returns:
            A new NumPy array with normalized values
        """
        X_normalized = X.copy().astype(float)  # Make a float copy for division

        for i in range(X.shape[1]):
            col = X[:, i]
            min_val = np.min(col)
            max_val = np.max(col)
            if min_val != max_val:
                X_normalized[:, i] = (col - min_val) / (max_val - min_val)
            else:
                X_normalized[:, i] = 0.0

        return X_normalized


    def create_binary_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        """
            Normalize each feature (column) in the NumPy array to the range [0, 1].

        Args:
            X: NumPy array of shape (n_samples, n_features)

        Returns:
            A new NumPy array with normalized values
        """

        data['binary_label'] = (data['PT08.S1(CO)'] > 1000).astype(int)

        data.drop(columns=['PT08.S1(CO)'], axis=1, inplace=True)
        return data


class LinearRegression:
    def __init__(self):
        """Initialize linear regression model.

        Args:
            learning_rate: Learning rate for gradient descent
            max_iter: Maximum number of iterations
            l2_lambda: L2 regularization strength
        """


        self.weights = None # We do not know #features yet.
        self.bias = 0.0
        self.learning_rate = 0.001
        self.max_iter = 10000
        self.l2_lambda = 0.0     # SP-Check
        self.losses = list()

    def fit(self, X: np.ndarray, y: np.ndarray) -> list[float]:
        """Train linear regression model.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            List of loss values
        """
        # TODO: Implement linear regression training
        n_samples, n_features = X.shape
        if self.weights is None:
            self.weights = np.ones(n_features) * 0.01

        for _ in range(self.max_iter):
            y_pred = self.predict(X)
            error = y - y_pred

            self.losses.append(self.criterion(y, y_pred))

            dw = -2  * X.T @ error / n_samples
            db = -2 * np.sum(error) / n_samples

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with trained model.

        Args:
            X: Feature matrix

        Returns:
            Predicted values
        """
        # TODO: Implement linear regression prediction
        return X @ self.weights + self.bias

    def criterion(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate MSE loss.

        Args:
            y_true: True target values
            y_pred: Predicted values

        Returns:
            Loss value
        """
        # TODO: Implement loss function
        loss = np.mean((y_true - y_pred) ** 2)
        return loss

    def metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate RMSE.

        Args:
            y_true: True target values
            y_pred: Predicted values

        Returns:
            Metric value
        """
        # TODO: Implement RMSE calculation
        return np.sqrt(self.criterion(y_true, y_pred))

class LogisticRegression:
    def __init__(self, learning_rate: float = 0.01, max_iter: int = 50000, prob_threshold: float=0.5):
        """Initialize logistic regression model.

        Args:
            learning_rate: Learning rate for gradient descent
            max_iter: Maximum number of iterations
            prob_threshold: Probability Threshold to classify binary output as 1 or 0
        """
        self.weights = None
        self.bias = 0
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.losses = list()
        self.prob_threshold = prob_threshold

    def set_prob_threshold(self, prob_thresh: float) -> None:
        self.prob_threshold = prob_thresh

    def set_learning_rate(self, learning_rate: float) -> None:
        self.learning_rate= learning_rate

    def set_max_iter(self, max_iter: float) -> None:
        self.max_iter = max_iter

    def fit(self, X: np.ndarray, y: np.ndarray) -> list[float]:
        """Train logistic regression model with normalization and L2 regularization.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            List of loss values
        """
        # TODO: Implement logistic regression training

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        for _ in range(1, self.max_iter):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)
            loss = self.criterion(y, y_pred)
            self.losses.append(loss)

            dw = np.dot(X.T, (y_pred -y)) / n_samples
            db = np.sum(y_pred -y) / n_samples

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Calculate prediction probabilities using normalized features.

        Args:
            X: Feature matrix

        Returns:
            Prediction probabilities
        """
        # TODO: Implement logistic regression prediction probabilities
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with trained model.

        Args:
            X: Feature matrix

        Returns:
            Predicted values
        """
        # TODO: Implement logistic regression prediction
        y_pred_proba = self.predict_proba(X)
        return np.where(y_pred_proba >= self.prob_threshold, 1, 0)

    def criterion(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate BCE loss.

        Args:
            y_true: True target values
            y_pred: Predicted values

        Returns:
            Loss value
        """
        # TODO: Implement loss function
        epsilon = 1e-15  # to prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    def F1_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate F1 score with handling of edge cases.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels

        Returns:
            F1 score (between 0 and 1), or 0.0 for edge cases
        """

        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        if tp + fp == 0 or tp + fn == 0:
            return 0.0

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        return 2 * precision * recall / (precision + recall + 1e-15)

    def label_binarize(self, y: np.ndarray) -> np.ndarray:
        """Binarize labels for binary classification.

        Args:
            y: Target vector

        Returns:
            Binarized labels
        """
        # TODO: Implement label binarization

    def get_auroc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate AUROC score.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted probabilities

        Returns:
            AUROC score (between 0 and 1)
        """
        # TODO: Implement AUROC calculation

        # Sort by predicted probability
        sorted_indices = np.argsort(-y_pred)
        y_true_sorted = y_true[sorted_indices]

        cum_pos = np.cumsum(y_true_sorted)
        total_pos = np.sum(y_true_sorted)
        total_neg = len(y_true_sorted) - total_pos

        tpr = cum_pos / (total_pos + 1e-15)
        fpr = np.cumsum(1 - y_true_sorted) / (total_neg + 1e-15)

        return np.trapezoid(tpr, fpr)

    def metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate AUROC.

        Args:
            y_true: True target values
            y_pred: Predicted values

        Returns:
            AUROC score
        """
        # TODO: Implement AUROC calculation

    def sigmoid(self, z: float) -> float:
        """Calculate the value of the sigmoid function.

        Args:
            z: Paramt W-Transpose*X

        Returns:
            float value of the sigmoid function
        """
        return 1 / (1 + np.exp(-z))

class ModelEvaluator:
    def __init__(self, n_splits: int = 5, random_state: int = 42):
        """Initialize evaluator with number of CV splits.

        Args:
            n_splits: Number of cross-validation folds
            random_state: Random state for reproducibility
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

    def cross_validation(self, model, X: np.ndarray, y: np.ndarray) -> List[float]:
        """Perform cross-validation

        Args:
            model: Model to be evaluated
            X: Feature matrix
            y: Target vector

        Returns:
            List of metric scores
        """
        # TODO: Implement cross-validation

def plot_iteration_loss(losses: list, plot_name: str) -> None:

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(losses)), losses, marker='o', markersize=2, linestyle='-')

    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title("Loss vs Iteration")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_name +".png", dpi=300, bbox_inches='tight')
    plt.show()

def tune_hyperparams_log_regr() -> None:

    dp = DataProcessor("./")
    (df_train, df_test) = dp.load_data()

    for dframe in (df_train, df_test):
        if dp.check_missing_values(dframe) > 0:
            dp.clean_data(df_train)

    # ========================================= #
    # Train and Test Data
    # ========================================= #
    (X_train, y_train) = dp.extract_features_labels(df_train)
    X_train_scaled = dp.normalize_ndarray(X_train)

    X_test = df_test.iloc[:, ].to_numpy()
    X_test_scaled = dp.normalize_ndarray(X_test)

    df_train_binary_label = dp.create_binary_labels(df_train)
    (X_train_binary_label, y_train_binary_label) = dp.extract_features_labels(df_train_binary_label)
    X_train_bin_scaled = dp.normalize_ndarray(X_train_binary_label)
    log_model = LogisticRegression(learning_rate=0.02, max_iter=50000, prob_threshold = 0.35)

    eval_list= []
    found = False
    for lr in [0.1, 0.08, 0.05, 0.02, 0.01, 0.005]:
        if found == False:
            for prob in [0.1, 0.2, 0.25, 0.3, 0.35, 0.4]:
                if found == False:
                    for iter in [10000, 25000, 50000]:
                        print(f"[{lr:6.4f}, {prob:6.4f}, {iter:7d}]", end=' -> ')
                        log_model.set_max_iter(iter)

                        log_model.set_prob_threshold(prob)
                        log_model.set_learning_rate(lr)

                        log_model.fit(X_train_bin_scaled, y_train_binary_label)

                        y_pred = log_model.predict(X_train_bin_scaled)
                        f1_score = log_model.F1_score(y_train_binary_label, y_pred)

                        y_pred_probs = log_model.predict_proba(X_train_bin_scaled)
                        auroc = log_model.get_auroc(y_train_binary_label, y_pred_probs)

                        print (f"f1_score = {f1_score:6.4f}", end=' -> ')
                        print (f"auroc= {auroc:6.4f}")

                        eval_list.append((lr, prob, iter, f1_score, auroc))

                        if f1_score > 0.90 and auroc > 0.90:
                            print(f"\t{eval_list[-1]}: BREAKING")

                            # Do not break, we want to see all combinations of hyperparams
                            # that give us desired auroc and f1-score
                            #found = True
                            break

                    print("")

def main():

    dp = DataProcessor("./")
    (df_train, df_test) = dp.load_data()

    for dframe in (df_train, df_test):
        if dp.check_missing_values(dframe) > 0:
            dp.clean_data(df_train)

    # ========================================= #
    # Train and Test Data
    # ========================================= #
    (X_train, y_train) = dp.extract_features_labels(df_train)
    X_train_scaled = dp.normalize_ndarray(X_train)

    X_test = df_test.iloc[:, ].to_numpy()
    X_test_scaled = dp.normalize_ndarray(X_test)

    '''
    # ========================================= #
    # EDA
    # ========================================= #
    dp.draw_histogram(pd.DataFrame(X_train))
    dp.draw_histogram(pd.DataFrame(X_train_scaled))

    dp.draw_scatter_plot(df_train, 2, 7)
    dp.get_pearson_corr(df_train)
    '''

    # ========================================= #
    # Linear Regression
    # ========================================= #
    '''
    linear_model = LinearRegression()
    linear_model.fit(X_train_scaled, y_train)

    y_pred = linear_model.predict(X_train_scaled)

    #print("Weights:", linear_model.weights)
    #print("Bias:", linear_model.bias)

    # Compute and print RMSE
    rmse = linear_model.metric(y_train, y_pred)
    print("RMSE:", rmse)

    plot_name =  "lin_regr_loss_"                 + \
                f"_iters_{linear_model.max_iter}" + \
                f"_LR_{linear_model.learning_rate}"

    plot_iteration_loss(linear_model.losses, plot_name)
    '''

    # ========================================= #
    # Logistic Regression
    # ========================================= #

    '''
    # The following function call was used to identify optimums values for
    # the hyperparameters. Once the values are noted, this function is not
    # being called anymore, but those hyperparam values are being used

    tune_hyperparams_log_regr()
    '''

    # ========================================= #
    # Train and Test Data
    # ========================================= #
    df_train_binary_label = dp.create_binary_labels(df_train)
    (X_train_binary_label, y_train_binary_label) = dp.extract_features_labels(df_train_binary_label)
    X_train_bin_scaled = dp.normalize_ndarray(X_train_binary_label)

    log_model = LogisticRegression(learning_rate=0.02, max_iter=50000, prob_threshold = 0.35)

    print(f"[{log_model.learning_rate:6.4f}, {log_model.prob_threshold:6.4f}, {log_model.max_iter:7d}]", end=' -> ')
    log_model.fit(X_train_bin_scaled, y_train_binary_label)

    y_pred = log_model.predict(X_train_bin_scaled)
    f1_score = log_model.F1_score(y_train_binary_label, y_pred)

    y_pred_probs = log_model.predict_proba(X_train_bin_scaled)
    auroc = log_model.get_auroc(y_train_binary_label, y_pred_probs)

    print (f"f1_score = {f1_score:6.4f}", end=' -> ')
    print (f"auroc= {auroc:6.4f}")

if __name__ == "__main__":
    main()
    print("Hello World!")

    '''
    print(f"Data dir: {dp.data_root}")
    print (df_train.head())
    print (df_test.head())
    print(f"#Missing Vals = {num_missing_data}")
    print(f"#Missing Vals = {num_missing_data}")

    num_ones = sum(1 for x in y_pred if x == 1)
    print (f"NUM_ONES = {num_ones}")
    num_zeros = sum(1 for x in y_pred if x == 0)
    print (f"NUM_ZEROS = {num_zeros}")
    '''
