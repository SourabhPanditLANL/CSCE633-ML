import numpy as npy
import numpy as np
import pandas as pd
from sklearn.svm import SVC

#TODO: Check with Dr. Mortazavi if this is OK to use
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

use_new_approach: bool = True #TODO: Remove this after test/dev

'''
Problem: University Admission Classification using SVMs

Instructions:
1. Do not use any additional libraries. Your code will be tested in a pre-built environment with only
   the library specified in question instruction available. Importing additional libraries will result in
   compilation errors and you will lose marks.

2. Fill in the skeleton code precisely as provided. You may define additional
   default arguments or helper functions if necessary, but ensure the input/output format matches.
'''


class DataLoader:
    '''
    Put your call to class methods in the __init__ method. Autograder will call your __init__ method only.
    '''

    def __init__(self, data_path: str):
        """
        Initialize data processor with paths to train dataset.
        You need to have train and validation sets processed.

        Args:
            data_path: absolute path to your data file
        """

        print(f"\nEnter Dataloader(): ", flush=True)


        # TODO：complete your dataloader here!
        self.data = pd.read_csv(data_path)
        print(f"\tself.data.shape = {self.data.shape}")
        print(f"\tself.data.columns = {self.data.columns}")


        if self.data.isna().any(axis=1).sum() > 0:
            df = self.data;
            self.data = df.drop_na()

        self.features = self.data.columns

        old_label_data = self.data['label'].values
        self.data = self.create_binary_label(self.data)
        new_label_data = self.data['label'].values

        diff_label_data = (old_label_data - new_label_data)**2
        sum_label_data = diff_label_data.sum()
        print(f"\tDEBUG: DIFF SUM = {sum_label_data}")


        self.X_train, self.X_val, self.y_train, self.y_val = self.data_split(self.data)

        self.train_data = pd.DataFrame(self.X_train, columns=self.features)
        self.train_data['label'] = self.y_train

        self.val_data = pd.DataFrame(self.X_val, columns=self.features)
        self.val_data['lable'] = self.y_val

        print(f"\tself.train_data.shape = {self.train_data.shape} {type(self.train_data)}", flush=True)
        print(f"\tself.val_data.shape = {self.val_data.shape} {type(self.val_data)}", flush=True)
        print(f"\tself.y_train.shape = {self.y_train.shape}", flush=True)
        print(f"\tself.y_val.shape = {self.y_val.shape}", flush=True)
        print(f"\tExit Dataloader(): \n", flush=True)


    def create_binary_label(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Create a binary label for the training data.
        '''

        print(f"\nEnter create_binary_label()", flush=True)
        df['label'] = (df['Chance of Admit'] > df['Chance of Admit'].median()).astype(int)
        print(f"Exit create_binary_label()\n", flush=True)

        return df

    def data_split(self, df: pd.DataFrame):

        print(f"\nEnter data_split()", flush=True)
        print(f"\tDEBUG: data_split() df.columns = {df.columns}", flush=True)
        X = df[self.features]
        y = df['label']

        print(f"Exit data_split()\n", flush=True)
        return (train_test_split(X,
                                 y,
                                 test_size=0.2,         # 20% for validation
                                 random_state=42,       # Ensures reproducibility
                                 stratify=y             # preserve class distribution
                                 ))



class SVMTrainer:
    def __init__(self):

        print(f"\nEnter SVMTrainer:__init__()", flush=True)
        self.model =  None
        print(f"Exit SVMTrainer:__init__()\n", flush=True)

    #def train(self, self.train_data: np.ndarray, y_train: np.ndarray, kernel: str, **kwargs) -> SVC:
    def train(self, train_data: np.ndarray, y_train: np.ndarray, kernel: str, **kwargs) -> SVC:
        '''
        Train the SVM model with the given kernel and parameters.

        Parameters:
            self.train_data: Training features
            y_train: Training labels
            kernel: Kernel type
            **kwargs: Additional arguments you may use
        Returns:
            SVC: Trained sklearn.svm.SVC model
        '''

        print(f"\nEnter SVMTrainer:train()", flush=True)
        self.model = SVC(kernel = kernel, probability=True, random_state=42)
        self.model.fit(train_data, y_train)
        print(f"Exit SVMTrainer:train()\n", flush=True)
        return self.model


    def get_support_vectors(self, model: SVC) -> np.ndarray:
        '''
        Get the support vectors from the trained SVM model.
        '''

        print(f"\nEnter SVMTrainer:get_support_vectory()", flush=True)
        if self.model is not None and hasattr(self.model, "support_vectors_"):
            print(f"Exit SVMTrainer:get_support_vectory()\n", flush=True)
            return self.model.support_vectors_
        else:
            raise AttributeError("Model has not been trained yet or does not have support_vectors_.")

    def predict(self, y):
        print(f"\nEnter SVMTrainer::predict()", flush=True)
        print(f"Exit SVMTrainer::predict()\n", flush=True)
        return self.model.predict(y)

def plot_predictions(X, y_true, y_pred, feature_set, kernel_name, support_vectors=None):
    """
    Plots 2D scatter of data points, colored by predicted label.

    Args:
        X: numpy array of shape (n_samples, 2), the feature values
        y_true: array-like of shape (n_samples,), true labels
        y_pred: array-like of shape (n_samples,), predicted labels
        feature_set: list of two feature names (e.g., ['CGPA', 'SOP'])
        kernel_name: string name of the kernel
        support_vectors: Array of support vectors
    """
    plt.figure(figsize=(6, 5))

    scatter = plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='bwr', edgecolor='k', alpha=0.6, label='Predicted label')
    # plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='coolwarm', marker='x', alpha=0.3, label='True label')

    if support_vectors is not None:
        plt.scatter(
            support_vectors[:, 0], support_vectors[:, 1],
            s=100, linewidths=1, facecolors='none', edgecolors='yellow', marker='o', label='Support Vectors'
        )

    plt.xlabel(feature_set[0])
    plt.ylabel(feature_set[1])
    plt.title(f"SVM {kernel_name} Kernel\nFeatures: {feature_set[0]} vs {feature_set[1]}")
    plt.legend()
    plt.tight_layout()
    plt.show()


'''
Initialize my_best_model with the best model you found.
'''
my_best_model = SVC(kernel='linear',
                    probability=True,
                    random_state=42)

def main():
    # DataLoader and Data Preprocessing
    data_path = "./data-2.csv"
    dl = DataLoader(data_path)

    # eature combinations
    feature_combos = [
        ['CGPA', 'SOP'],
        ['CGPA', 'GRE Score'],
        ['SOP', 'LOR'],
        ['LOR', 'GRE Score']
    ]

    # SVM kernel configs
    svm_kernels = {
        'linear': {'kernel': 'linear', 'probability': True, 'random_state': 42},
        'rbf': {'kernel': 'rbf', 'probability': True, 'random_state': 42},
        'poly': {'kernel': 'poly', 'degree': 3, 'probability': True, 'random_state': 42}
    }

    print(f"DEBUG-100 svm_kernels['linear'] = {svm_kernels['linear']}")
    print(f"DEBUG-101 svm_kernels['rbf'] = {svm_kernels['rbf']}")
    print(f"DEBUG-102 svm_kernels['poly'] = {svm_kernels['poly']}")

    results = {}

    for kernel_name, svm_params in svm_kernels.items():
        results[kernel_name] = {}

        print(f"DEBUG-103 kernel_name = {kernel_name}, svm_params = {svm_params}", flush=True)
        for feature_set in feature_combos:

            # Instantiate and train SVM model
            if use_new_approach:
                trainer = SVMTrainer()
            else:
                model = SVC(**svm_params)

            X_train_combo = dl.train_data[feature_set].values
            X_val_combo = dl.val_data[feature_set].values

            # Scale features: fit scaler on training, transform on both
            scaler = StandardScaler()
            X_train_combo_scaled = scaler.fit_transform(X_train_combo)
            X_val_combo_scaled = scaler.transform(X_val_combo)


            if use_new_approach:
                trained_model = trainer.train(X_train_combo_scaled, dl.y_train, kernel_name, kwarg=svm_params)
            else:
                model.fit(X_train_combo_scaled, dl.y_train)

            # Store results
            if use_new_approach:
                results[kernel_name][tuple(feature_set)] = {
                    'model': trainer.model,
                    'support_vectors': trainer.get_support_vectors(trainer.model),
                    'train_pred': trainer.predict(X_train_combo_scaled),
                    'val_pred': trainer.predict(X_val_combo_scaled),
                    'scaler': scaler
                }
            else:
                results[kernel_name][tuple(feature_set)] = {
                    'model': model,
                    'support_vectors': model.support_vectors_,
                    'train_pred': model.predict(X_train_combo_scaled),
                    'val_pred': model.predict(X_val_combo_scaled),
                    'scaler': scaler
                }

    # Visualization: plot training predictions for each kernel/feature combo
    for kernel_name in results:
        #print(f"DEBUG: resutls = {results[kernel_name]}", flush=True)
        for feature_set in results[kernel_name]:
            print("", end='')
            #print(f"DEBUG-1: feature_set {feature_set}", flush=True)
            #print(f"DEBUG-2: {results[kernel_name][feature_set]['train_pred']}", flush=True)
            ##print(f"DEBUG-3: {dl.train_data[['CGPA', 'SOP']].values}", flush=True)
            #print(f"DEBUG-4: {dl.train_data[list(feature_set)]}", flush= None)

            '''
            plot_predictions(
                dl.train_data[list(feature_set)].values,
                dl.y_train.values,
                results[kernel_name][feature_set]['train_pred'],
                feature_set,
                kernel_name
            )
            '''

    # Evaluate on validation set to find best combo
    best_acc = 0
    best_combo = None
    for kernel_name in results:
        for feature_set in results[kernel_name]:
            acc = accuracy_score(
                dl.y_val.values,
                results[kernel_name][feature_set]['val_pred']
            )
            print(f"Validation accuracy for {kernel_name} with {feature_set}: {acc:.3f}")
            if acc > best_acc:
                best_acc = acc
                best_combo = (kernel_name, feature_set)

    print(f"Best model: {best_combo} with accuracy {best_acc:.3f}")

if __name__ == "__main__":
    main()

    print("Hello, World!")
