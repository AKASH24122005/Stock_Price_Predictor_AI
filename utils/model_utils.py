from tensorflow.keras.models import load_model
import joblib
import os


os.makedirs("model", exist_ok=True)


def save_model(model):
    model.save("model/lstm_model.keras")


def load_saved_model():
    return load_model("model/lstm_model.keras")


def save_scaler(scaler):
    joblib.dump(scaler, "model/scaler.pkl")


def load_scaler():
    return joblib.load("model/scaler.pkl")