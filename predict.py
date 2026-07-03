import numpy as np

from utils.preprocessing import load_data
from utils.model_utils import load_saved_model
from utils.model_utils import load_scaler


def predict_next_day(csv_file):
    # Load model and scaler
    model = load_saved_model()
    scaler = load_scaler()

    # Load stock data
    df = load_data(csv_file)

    # Scale the close prices
    scaled_data = scaler.transform(df)

    # Get the last 60 days
    last_60_days = scaled_data[-60:]

    # Reshape for LSTM
    X_test = np.reshape(last_60_days, (1, 60, 1))

    # Predict
    prediction = model.predict(X_test, verbose=0)

    # Convert back to original price
    predicted_price = scaler.inverse_transform(prediction)

    return predicted_price[0][0]


if __name__ == "__main__":
    price = predict_next_day("dataset/AAPL.csv")
    print(f"Predicted Next Closing Price: ${price:.2f}")