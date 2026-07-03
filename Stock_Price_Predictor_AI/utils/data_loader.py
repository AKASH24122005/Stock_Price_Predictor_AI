import yfinance as yf
import os


def download_stock_data(symbol, start_date, end_date):
    """
    Download historical stock data from Yahoo Finance
    """

    # auto_adjust=False keeps standard OHLC columns
    df = yf.download(
        symbol,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        raise Exception("No data found.")

    # Flatten MultiIndex columns (new yfinance versions)
    if hasattr(df.columns, "levels"):
        df.columns = df.columns.get_level_values(0)

    os.makedirs("dataset", exist_ok=True)

    file_path = f"dataset/{symbol}.csv"

    df.to_csv(file_path, index=True)

    print(f"Saved to {file_path}")

    return df