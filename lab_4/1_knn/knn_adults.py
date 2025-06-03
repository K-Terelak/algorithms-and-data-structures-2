from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_prepare_data(filename):
    df = pd.read_csv(filename)

    # Remove duplicate rows to ensure data quality
    df.drop_duplicates(inplace=True)

    # Drop rows with any NaN values
    df.dropna(inplace=True)

    # One-hot encoding dla kolumn kategorycznych
    cat_cols = ['workclass', 'education', 'marital-status', 'occupation',
                'relationship', 'race', 'gender', 'native-country']
    df = pd.get_dummies(df, columns=cat_cols)

    df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})

    return df


def knn_predict_single_point(X_train, y_train, query_point, k):
    distances = np.linalg.norm(X_train - query_point, axis=1)
    k_indices = np.argsort(distances)[:k]
    k_labels = y_train[k_indices]
    most_common = Counter(k_labels).most_common(1)[0][0]

    print("\nSelected test point (query):")
    print(f"{query_point}\n")
    print(f"{k} nearest neighbors:")
    print("-" * 50)
    for rank, neighbor_idx in enumerate(k_indices, start=1):
        print(
            f"{rank}. Index: {neighbor_idx:3d} | Class: {y_train[neighbor_idx]} | Distance: {distances[neighbor_idx]:.4f}")
        # print(f"   Data: {X_train[neighbor_idx]}")
    print("-" * 50)
    print(f"Predicted class for selected point: {most_common}")
    return most_common


def knn_predict_all(X_train, y_train, X_test, k):
    y_pred = []
    for idx, point in enumerate(X_test):
        distances = np.linalg.norm(X_train - point, axis=1)
        k_indices = np.argsort(distances)[:k]
        k_labels = y_train[k_indices]
        most_common = Counter(k_labels).most_common(1)[0][0]
        y_pred.append(most_common)
    return np.array(y_pred)


if __name__ == '__main__':
    filename = "../adult.csv"  # Provide the path to your CSV file here

    k = 5
    query_index = 0  # Index of the test point to inspect closely

    df = load_and_prepare_data(filename)

    # Assume the label column is 'Outcome'
    X = df.drop("income", axis=1).values
    y = df["income"].values

    # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Show detailed neighbors for the selected test point
    pred_class = knn_predict_single_point(X_train, y_train, X_test[query_index], k)
    print(f"\nActual class: {y_test[query_index]}")
    print(f"Prediction correct? {'YES' if pred_class == y_test[query_index] else 'NO'}")

    # For comparison, classify entire test set and show accuracy
    y_pred = knn_predict_all(X_train, y_train, X_test, k)
    print(f"\nClassification accuracy on test set: {accuracy_score(y_test, y_pred) * 100:.2f}%")
