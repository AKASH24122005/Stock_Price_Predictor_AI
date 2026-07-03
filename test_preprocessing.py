from utils.preprocessing import load_data
from utils.preprocessing import preprocess_data
from utils.preprocessing import split_data

df = load_data("dataset/AAPL.csv")

print(df.head())

X, y, scaler = preprocess_data(df)

print("X Shape :", X.shape)
print("Y Shape :", y.shape)

X_train, X_test, y_train, y_test = split_data(X, y)

print()

print("Training Samples :", len(X_train))
print("Testing Samples :", len(X_test))