from utils.data_loader import download_stock_data

stocks = [
    "AAPL",
    "MSFT",
    "GOOG",
    "AMZN",
    "TSLA"
]

for stock in stocks:

    print(f"Downloading {stock}")

    download_stock_data(
        stock,
        "2018-01-01",
        "2025-12-31"
    )

print("Done")