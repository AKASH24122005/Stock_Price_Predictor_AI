import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from utils.model_utils import save_scaler


def load_data(csv_file):
    """
    Load stock CSV file
    """
    df = pd.read_csv(csv_file)

    # Keep only Close price
    df = df[['Close']]

    return df


def preprocess_data(df, sequence_length=60):
    """
    Normalize data and create sequences
    """

    scaler = MinMaxScaler(feature_range=(0, 1))

    scaled_data = scaler.fit_transform(df)

    # Save scaler
    save_scaler(scaler)

    X = []
    y = []

    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i])

    X = np.array(X)
    y = np.array(y)

    return X, y, scaler


def split_data(X, y, train_ratio=0.8):
    """
    Split into training and testing datasets
    """

    train_size = int(len(X) * train_ratio)

    X_train = X[:train_size]
    X_test = X[train_size:]

    y_train = y[:train_size]
    y_test = y[train_size:]

    return X_train, X_test, y_train, y_test