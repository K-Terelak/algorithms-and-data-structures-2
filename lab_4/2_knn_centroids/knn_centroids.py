import numpy as np
import pandas as pd


def load_and_prepare_data(filename):
    df = pd.read_csv(filename)

    # Remove duplicate rows to ensure data quality
    df.drop_duplicates(inplace=True)

    # Convert all columns to numeric where possible; invalid parsing will become NaN
    df = df.apply(pd.to_numeric, errors='coerce')

    # Replace zero values with NaN in columns where zero is not a valid value
    zero_features = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in zero_features:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)

    # Drop all rows with any NaN values
    df.dropna(inplace=True)

    return df


def knn_cluster_from_csv(filename, k):
    df = load_and_prepare_data(filename)

    # Use only numeric columns for centroid calculation
    X = df.select_dtypes(include=[np.number]).values

    # Calculate centroid (mean vector)
    centroid = np.mean(X, axis=0)

    # Calculate distances from each point to centroid
    distances = np.linalg.norm(X - centroid, axis=1)

    # Get indices of k closest points to centroid
    indices = np.argsort(distances)[:k]

    # Create labels: 1 for closest k points, 0 otherwise
    labels = np.zeros(len(X), dtype=int)
    labels[indices] = 1

    return labels, distances, centroid


if __name__ == '__main__':
    filename = "../diabetes.csv"
    k = 5
    labels, distances, centroid = knn_cluster_from_csv(filename, k)

    print(f"Centroid:\n{centroid}\n")
    print(f"{k} points closest to centroid (labels=1):")
    for i, label in enumerate(labels):
        if label == 1:
            print(f"Index {i}, Distance: {distances[i]:.4f}")
