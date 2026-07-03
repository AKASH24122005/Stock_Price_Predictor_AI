import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from utils.preprocessing import load_data, preprocess_data, split_data
from utils.model_utils import load_saved_model


def evaluate_model(csv_file):

    df = load_data(csv_file)

    X, y, scaler = preprocess_data(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = load_saved_model()

    predictions = model.predict(X_test, verbose=0)

    predictions = scaler.inverse_transform(predictions)
    actual = scaler.inverse_transform(y_test)

    rmse = np.sqrt(mean_squared_error(actual, predictions))
    mae = mean_absolute_error(actual, predictions)
    r2 = r2_score(actual, predictions)

    return actual, predictions, rmse, mae, r2