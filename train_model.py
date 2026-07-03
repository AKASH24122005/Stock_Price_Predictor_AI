import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from utils.preprocessing import load_data
from utils.preprocessing import preprocess_data
from utils.preprocessing import split_data
from utils.model_utils import save_model

# Create model directory
os.makedirs("model", exist_ok=True)

# Load data
df = load_data("dataset/AAPL.csv")

# Preprocess
X, y, scaler = preprocess_data(df)

# Split data
X_train, X_test, y_train, y_test = split_data(X, y)

print("Training Shape :", X_train.shape)
print("Testing Shape :", X_test.shape)

# Build LSTM Model
model = Sequential()

model.add(
    LSTM(
        units=64,
        return_sequences=True,
        input_shape=(X_train.shape[1], X_train.shape[2])
    )
)

model.add(Dropout(0.2))

model.add(
    LSTM(
        units=32
    )
)

model.add(Dropout(0.2))

model.add(Dense(16, activation="relu"))

model.add(Dense(1))

# Compile
model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)

# Early stopping
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

print("\nTraining Model...\n")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Save model
save_model(model)

print("\nModel Saved Successfully!")