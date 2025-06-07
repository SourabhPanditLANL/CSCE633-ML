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
        return data

    def extract_features_labels(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features and labels from dataframe, convert to numpy arrays.

        Args:
            data: Input dataframe

        Returns:
            Tuple of feature matrix X and label vector y
        """
        # TODO: Implement feature/label extraction
        return (data.columns[:-1], data.columns[-1])


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
            print (f"FEATURE: {feature}")
            axes[i].hist(data[feature], bins=20, edgecolor='black')  # Drop NaNs for clean hist
            axes[i].set_title(f'Histogram of {feature}')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Frequency')

        plt.tight_layout()
        plt.savefig("features_histogram.png", dpi=300, bbox_inches='tight')
        plt.show()

    def draw_scatter_plot(self, data: pd.DataFrame, feature1_idx: int, feature2_idx: int) -> None:
        """Extract features and labels from dataframe, convert to numpy arrays.

        Args:
            data: Input dataframe
            features1_idx: index for features 1
            features2_idx: index for features 2

        Returns:
            returns Nothing
        """
        feature1 = data.columns[feature1_idx]
        y_feature = data.columns[feature2_idx]

        plt.figure(figsize=(8, 6))
        plt.scatter(data[feature1], data[y_feature], alpha=0.6, edgecolors='k')

        plt.xlabel(feature1)
        plt.ylabel(y_feature)
        plt.title(f'Scatter Plot: {feature1} vs {y_feature}')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("scatter_plot.png", dpi=300, bbox_inches='tight')
        plt.show()


class LinearRegression:
    def __init__(self):
        """Initialize linear regression model.

        Args:
            learning_rate: Learning rate for gradient descent
            max_iter: Maximum number of iterations
            l2_lambda: L2 regularization strength
        """
        self.weights = None
        self.bias = None
        self.learning_rate = None
        self.max_iter = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> list[float]:
        """Train linear regression model.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            List of loss values
        """
        # TODO: Implement linear regression training

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with trained model.

        Args:
            X: Feature matrix

        Returns:
            Predicted values
        """
        # TODO: Implement linear regression prediction

    def criterion(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate MSE loss.

        Args:
            y_true: True target values
            y_pred: Predicted values

        Returns:
            Loss value
        """
        # TODO: Implement loss function

    def metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate RMSE.

        Args:
            y_true: True target values
            y_pred: Predicted values

        Returns:
            Metric value
        """
        # TODO: Implement RMSE calculation

class LogisticRegression:
    def __init__(self):
        """Initialize logistic regression model.

        Args:
            learning_rate: Learning rate for gradient descent
            max_iter: Maximum number of iterations
        """
        self.weights = None
        self.bias = None
        self.learning_rate = None
        self.max_iter = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> list[float]:
        """Train logistic regression model with normalization and L2 regularization.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            List of loss values
        """
        # TODO: Implement logistic regression training

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Calculate prediction probabilities using normalized features.

        Args:
            X: Feature matrix

        Returns:
            Prediction probabilities
        """
        # TODO: Implement logistic regression prediction probabilities

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with trained model.

        Args:
            X: Feature matrix

        Returns:
            Predicted values
        """
        # TODO: Implement logistic regression prediction

    def criterion(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate BCE loss.

        Args:
            y_true: True target values
            y_pred: Predicted values

        Returns:
            Loss value
        """
        # TODO: Implement loss function

    def F1_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate F1 score with handling of edge cases.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels

        Returns:
            F1 score (between 0 and 1), or 0.0 for edge cases
        """
        # TODO: Implement F1 score calculation

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

    def metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate AUROC.

        Args:
            y_true: True target values
            y_pred: Predicted values

        Returns:
            AUROC score
        """
        # TODO: Implement AUROC calculation

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

def main():
    dp = DataProcessor("./data")

    (df_train, df_test) = dp.load_data()

    num_missing_data = dp.check_missing_values(df_train)

    if num_missing_data > 0:
        dp.clean_data(df_train)

    (features, label) = dp.extract_features_labels(df_train)

    dp.draw_histogram(df_train)
    dp.draw_scatter_plot(df_train, 2, 7)
    dp.get_pearson_corr(df_train)


    print("Hello World! ..... ")
    '''
    print(f"Data dir: {dp.data_root}")
    print (df_train.head())
    print (df_test.head())
    print(f"#Missing Vals = {num_missing_data}")
    print(f"#Missing Vals = {num_missing_data}")
    print(f"Features: {features}\t label: {label}")
    '''

if __name__ == "__main__":
    main()
    print("Hello World!")
