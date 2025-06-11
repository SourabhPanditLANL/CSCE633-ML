import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from typing import Tuple, List

#from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
#from sklearn.metrics import mean_squared_error

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
        for col in data.columns:
            mask = data[col] != -200  # mask to exclude -200
            mean_val = data.loc[mask, col].mean()
            data.loc[data[col] == -200, col] = mean_val

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

    def train_val_split(self, X: np.ndarray, y: np.ndarray, val_ratio: float = 0.2,
                    seed: int = 1005) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Shuffle and split the dataset into training and validation sets.

        Args:
            X: Feature matrix
            y: label vector
            val_ratio: Proportion of data to use for validation
            seed: Random seed for reproducibility

        Returns: Tuple: (X_train, X_val, y_train, y_val)
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        np.random.seed(seed)
        np.random.shuffle(indices)
        split_index = int((1 - val_ratio) * n_samples)
        train_idx, val_idx = indices[:split_index], indices[split_index:]
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        return X_train, X_val, y_train, y_val


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
        #plt.show()

    def draw_histogram(self, data: pd.DataFrame, hist_name: str) -> None:
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
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Frequency')

        # Hide the last unused subplot
        if len(data.columns) < len(axes):
            axes[len(data.columns)].axis('off')

        plt.tight_layout()
        plt.savefig(hist_name, dpi=300, bbox_inches='tight')
        #plt.show()


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
        #plt.show()

    def normalize(self, X: np.ndarray,
                          min_vals: np.ndarray=None,
                          max_vals: np.ndarray=None) -> (np.ndarray, np.ndarray, np.ndarray):
        """
            Normalize each feature (column) in the NumPy array to the range [0, 1].

        Args:
            X: NumPy array of shape (n_samples, n_features)

        Returns:
            A new NumPy array with normalized values
        """
        X_normalized = X.copy().astype(float)  # Make a float copy for division

        if min_vals is None or max_vals is None:
            min_vals = np.min(X, axis=0)
            max_vals = np.max(X, axis=0)

        for i in range(X.shape[1]):
            if min_vals[i] != max_vals[i]:
                X_normalized[:, i] = (X[:, i] - min_vals[i]) / (max_vals[i] - min_vals[i])
            else:
                X_normalized[:, i] = 0.0

        return X_normalized, min_vals, max_vals

class LinearRegression:
    def __init__(self, learning_rate: float = 0.1, max_iter: int = 10000,
                 l1_lambda: float = 0.0,
                 l2_lambda: float = 0.0,
                 reg_flag:int = 0):
        """Initialize linear regression model.

        Args:
            learning_rate: Learning rate for gradient descent
            max_iter: Maximum number of iterations
            l2_lambda: L2 regularization strength
        """

        self.weights = None # We do not know #features yet.
        self.bias = 0.0
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.reg_flag = reg_flag
        self.losses = []
        self.scaled = False

    def fit(self, X: np.ndarray, y: np.ndarray, print_weights: bool = True) -> list[float]:
        """Train linear regression model.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            List of loss values
        """
        # TODO: Implement linear regression training

        n_samples, n_features = X.shape

        if self.scaled == False:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            self.scaled = True
        else:
            X_scaled = X

        if self.weights is None:
            self.weights = np.random.randn(n_features) * 0.01

        for _ in range(self.max_iter):
            y_pred = self.predict(X_scaled)
            error = y - y_pred

            if (self.reg_flag == 0):
                # No Regularization
                loss = self.criterion(y, y_pred)
                dw = -2  * X_scaled.T @ error / n_samples
            elif (self.reg_flag == 1):
                # Lasso loss = MSE + L1 penalty
                loss = self.criterion(y, y_pred) + self.l1_lambda * np.sum(np.abs(self.weights))
                dw = (-2 * X_scaled.T @ error / n_samples) + self.l1_lambda * np.sign(self.weights)
            else:
                # Ridge loss = MSE + L2 penalty
                loss = self.criterion(y, y_pred) + self.l2_lambda * np.sum(self.weights ** 2)
                dw = (-2 * X_scaled.T @ error / n_samples) + 2 * self.l2_lambda * self.weights

            np.clip(dw, -1e5, 1e5, out=dw)

            db = -2 * np.sum(error) / n_samples

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            self.losses.append(loss)

        if print_weights == True:
            print(f"(Linear Regression Weights: {self.weights}")

        return self.losses

    def predict(self, X: np.ndarray, scaled:bool = False) -> np.ndarray:
        """Make predictions with trained model.

        Args:
            X: Feature matrix

        Returns:
            Predicted values
        """
        # TODO: Implement linear regression prediction

        X_scaled = self.scaler.transform(X)
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
    def __init__(self, learning_rate: float = 0.05, max_iter: int = 1000, prob_threshold: float=0.2):
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
        self.losses = []
        self.prob_threshold = prob_threshold

    def set_prob_threshold(self, prob_thresh: float) -> None:
        self.prob_threshold = prob_thresh

    def set_learning_rate(self, learning_rate: float) -> None:
        self.learning_rate= learning_rate

    def set_max_iter(self, max_iter: float) -> None:
        self.max_iter = max_iter

    def fit(self, X: np.ndarray, y: np.ndarray, print_weights:bool = True) -> list[float]:
        """Train logistic regression model with normalization and L2 regularization.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            List of loss values
        """
        # TODO: Implement logistic regression training

        n_samples, n_features = X.shape
        if self.weights is None:
            self.weights = np.random.randn(n_features) * 0.01

        for _ in range(1, self.max_iter):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)
            loss = self.criterion(y, y_pred)
            self.losses.append(loss)

            dw = np.dot(X.T, (y_pred -y)) / n_samples
            db = np.sum(y_pred -y) / n_samples

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        if print_weights == True:
            print(f"(Logistic Regression Weights: {self.weights}")
        return self.losses

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
        epsilon = 1e-10  # to prevent log(0)
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
        return (y > 1000).astype(int)

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

        return np.clip(np.trapezoid(y=tpr, x=fpr), 0.0, 1.0)

    def metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate AUROC.

        Args:
            y_true: True target values
            y_pred: Predicted values

        Returns:
            AUROC score
        """
        # TODO: Implement AUROC calculation
        return self.get_auroc(y_true, y_pred)

    def sigmoid(self, z: float) -> float:
        """Calculate the value of the sigmoid function.

        Args:
            z: Paramt W-Transpose*X

        Returns:
            float value of the sigmoid function
        """
        z = np.clip(z, -500, 500)
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

        scores = []

        for train_index, val_index in self.kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            model.fit(X_train, y_train)

            if hasattr(model, "F1_score") and hasattr(model, "get_auroc"):
                #  This is our Logistic Regression
                y_pred_probs = model.predict_proba(X_val)
                y_pred_labels = model.predict(X_val)

                f1 = model.F1_score(y_val, y_pred_labels)
                auroc = model.get_auroc(y_val, y_pred_probs)
                scores.append((f1, auroc))

            else:
                # We have the good old linear regression
                y_pred = model.predict(X_val)
                rmse = model.metric(y_val, y_pred)
                scores.append(rmse)

        return scores

    def plot_roc_per_fold(self, model, X_train_bin: np.ndarray, y_train_bin: np.ndarray) -> None:
        """
        Plot ROC curves and compute AUROC for each fold using Logistic Regression.

        Args:
            model: Logistic Regression model for which ROC curve is desired
            X_train_bin: Feature matrix
            y_train_bin: Target vector

        Returns:
            None
        """
        plt.figure(figsize=(8, 6))

        for fold, (train_idx, val_idx) in enumerate(self.kf.split(X_train_bin), 1):
            X_train, X_val = X_train_bin[train_idx], X_train_bin[val_idx]
            y_train, y_val = y_train_bin[train_idx], y_train_bin[val_idx]

            model.fit(X_train, y_train, print_weights=False)
            y_probs = model.predict_proba(X_val)

            thresholds = np.sort(np.unique(y_probs))[::-1]
            tprs, fprs = [], []

            total_pos = np.sum(y_val)
            total_neg = len(y_val) - total_pos

            for thresh in thresholds:
                y_pred = (y_probs >= thresh).astype(int)
                tp = np.sum((y_val == 1) & (y_pred == 1))
                fp = np.sum((y_val == 0) & (y_pred == 1))

                tpr = tp / (total_pos + 1e-15)
                fpr = fp / (total_neg + 1e-15)

                tprs.append(tpr)
                fprs.append(fpr)

            tprs = np.array(tprs)
            fprs = np.array(fprs)
            auroc = np.trapezoid(tprs, fprs)

            plt.plot(fprs, tprs, label=f"Fold {fold} (AUROC = {auroc:.3f})")

        plt.plot([0, 1], [0, 1], '--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve per Fold (Logistic Regression)")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("ROC-Curve-Log_Regr.png", dpi=300, bbox_inches='tight')
        #plt.show()

