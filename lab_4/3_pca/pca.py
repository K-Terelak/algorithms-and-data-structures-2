from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def load_and_prepare_data(filename):
    df = pd.read_csv(filename)
    df.drop_duplicates(inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')

    zero_features = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in zero_features:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)

    df.dropna(inplace=True)
    return df


def knn_predict_all(X_train, y_train, X_test, k):
    y_pred = []
    for point in X_test:
        distances = np.linalg.norm(X_train - point, axis=1)
        k_indices = np.argsort(distances)[:k]
        k_labels = y_train[k_indices]
        most_common = Counter(k_labels).most_common(1)[0][0]
        y_pred.append(most_common)
    return np.array(y_pred)


def pca(X, n_components):
    if len(X[0]) < n_components:
        raise ValueError("Number of components can't be more than number of features")

    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean

    cov_matrix = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    selected_eigenvectors = eigenvectors[:, sorted_indices[:n_components]]

    X_reduced = np.dot(X_centered, selected_eigenvectors)
    explained_variance = eigenvalues[sorted_indices[:n_components]] / np.sum(eigenvalues)

    return X_reduced, explained_variance


if __name__ == '__main__':
    filename = "../diabetes.csv"
    df = load_and_prepare_data(filename)

    X = df.drop("Outcome", axis=1).values

    n_components = 2
    X_reduced, variance_explained = pca(X, n_components)

    print("Reduced data (first 5 rows):")
    print(X_reduced[:5])
    print("\nExplained variance by each component:")
    print(variance_explained)

    # KNN classification on reduced data
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)
    k = 5
    y_pred = knn_predict_all(X_train, y_train, X_test, k)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy of KNN on PCA-reduced data: {accuracy * 100:.2f}%")
