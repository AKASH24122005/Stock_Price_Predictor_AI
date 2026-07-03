import matplotlib.pyplot as plt


def plot_history(df):

    plt.figure(figsize=(12,5))

    plt.plot(
        df["Close"],
        label="Closing Price"
    )

    plt.title("Historical Closing Price")

    plt.xlabel("Days")

    plt.ylabel("Price")

    plt.legend()

    plt.grid()

    plt.show()


def plot_prediction(actual, predicted):

    plt.figure(figsize=(12,5))

    plt.plot(actual, label="Actual")

    plt.plot(predicted, label="Predicted")

    plt.legend()

    plt.grid()

    plt.show()