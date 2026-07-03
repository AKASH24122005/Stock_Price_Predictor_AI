import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from utils.preprocessing import load_data, preprocess_data, split_data
from utils.model_utils import load_saved_model, load_scaler

# Load data
df = load_data("dataset/AAPL.csv")

X, y, scaler = preprocess_data(df)

X_train, X_test, y_train, y_test = split_data(X, y)

# Load trained model
model = load_saved_model()

# Predict
predictions = model.predict(X_test, verbose=0)

# Convert back to original prices
predictions = scaler.inverse_transform(predictions)
actual = scaler.inverse_transform(y_test)

# Metrics
rmse = np.sqrt(mean_squared_error(actual, predictions))
mae = mean_absolute_error(actual, predictions)
r2 = r2_score(actual, predictions)

print(f"RMSE : {rmse:.2f}")
print(f"MAE  : {mae:.2f}")
print(f"R²   : {r2:.4f}")

# Plot
plt.figure(figsize=(12, 6))
plt.plot(actual, label="Actual Price")
plt.plot(predictions, label="Predicted Price")
plt.title("Actual vs Predicted Stock Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()
