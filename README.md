# 📈 AI Stock Price Predictor using LSTM

An AI-powered web application that predicts the next day's stock closing price using a Long Short-Term Memory (LSTM) neural network. The application downloads historical stock market data, preprocesses it, trains an LSTM model, evaluates its performance, and provides predictions through an interactive Streamlit dashboard.

---

## 🚀 Features

- 📊 Download historical stock data using Yahoo Finance
- 🧠 Train an LSTM Deep Learning model
- 📈 Predict the next day's stock closing price
- 📉 Visualize historical stock prices
- 📊 Compare Actual vs Predicted prices
- 📏 Display model evaluation metrics (RMSE, MAE, R² Score)
- 📋 Interactive Streamlit dashboard
- 💾 Download stock data as CSV
- 🎨 Clean and user-friendly interface

---

## 🛠️ Tech Stack

- **Python**
- **TensorFlow / Keras**
- **LSTM (Long Short-Term Memory)**
- **Streamlit**
- **Plotly**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Matplotlib**
- **Yahoo Finance (yfinance)**
- **Joblib**

---

## 📂 Project Structure

```
Stock_Price_Predictor_AI/
│
├── app.py
├── train_model.py
├── predict.py
├── evaluate_model.py
├── download_all.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
│
├── dataset/
│   ├── AAPL.csv
│   ├── MSFT.csv
│   ├── GOOG.csv
│   ├── AMZN.csv
│   └── TSLA.csv
│
├── model/
│   ├── lstm_model.keras
│   └── scaler.pkl
│
├── outputs/
│
├── images/
│   ├── dashboard.png
│   ├── historical_chart.png
│   ├── prediction_metrics.png
│   └── actual_vs_predicted.png
│
└── utils/
    ├── data_loader.py
    ├── preprocessing.py
    ├── model_utils.py
    ├── visualization.py
    └── evaluate.py
```

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/your-username/Stock_Price_Predictor_AI.git

cd Stock_Price_Predictor_AI
```

---

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### 1. Download historical stock data

```bash
python download_all.py
```

---

### 2. Train the LSTM model

```bash
python train_model.py
```

---

### 3. Evaluate the model

```bash
python evaluate_model.py
```

---

### 4. Run the Streamlit application

```bash
python -m streamlit run app.py
```

The application will open automatically in your browser.

---

## 📊 Model Evaluation Metrics

The model is evaluated using:

- **RMSE (Root Mean Squared Error)**
- **MAE (Mean Absolute Error)**
- **R² Score**

These metrics help measure the prediction accuracy of the trained LSTM model.

---

## 📸 Application Screenshots

### Dashboard

> Add a screenshot here

```
images/dashboard.png
```

---

### Historical Closing Price

> Add a screenshot here

```
images/historical_chart.png
```

---

### Actual vs Predicted

> Add a screenshot here

```
images/actual_vs_predicted.png
```

---

### Prediction Metrics

> Add a screenshot here

```
images/prediction_metrics.png
```

---

## 📈 Workflow

```
Download Stock Data
        │
        ▼
Data Preprocessing
        │
        ▼
Data Normalization
        │
        ▼
Sequence Creation
        │
        ▼
LSTM Model Training
        │
        ▼
Model Evaluation
        │
        ▼
Prediction
        │
        ▼
Streamlit Dashboard
```

---

## 🎯 Future Improvements

- Support real-time stock price prediction
- Multi-stock model training
- Multi-day forecasting (7, 15, 30 days)
- Hyperparameter tuning
- GRU and Transformer-based models
- News sentiment analysis integration
- Model deployment on Streamlit Cloud

---

## 📜 License

This project is licensed under the **MIT License**.

See the **LICENSE** file for more details.

---

## 👨‍💻 Author

**Akashtamilselvan**

GitHub: https://github.com/AKASH24122005/Stock_Price_Predictor_AI

---

## ⭐ Support

If you found this project useful:

⭐ Star this repository

🍴 Fork this repository

📢 Share it with others

---

## 🙏 Acknowledgements

- Yahoo Finance
- TensorFlow
- Streamlit
- Plotly
- Scikit-learn
- Keras
