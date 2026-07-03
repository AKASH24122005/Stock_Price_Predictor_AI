import streamlit as st
import plotly.graph_objects as go

from predict import predict_next_day
from utils.preprocessing import load_data
from utils.evaluate import evaluate_model

st.set_page_config(
    page_title="AI Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI Stock Price Predictor using LSTM")

st.markdown("---")

# Sidebar
st.sidebar.header("Settings")

stock = st.sidebar.selectbox(
    "Select Stock",
    [
        "AAPL",
        "MSFT",
        "GOOG",
        "AMZN",
        "TSLA"
    ]
)

csv_file = f"dataset/{stock}.csv"

try:
    df = load_data(csv_file)

except Exception:
    st.error(f"{stock}.csv not found.")
    st.stop()

# Current price
current_price = float(df["Close"].iloc[-1])

# Prediction
predicted_price = predict_next_day(csv_file)

# Model Evaluation
actual, predicted, rmse, mae, r2 = evaluate_model(csv_file)

# ==========================
# Metrics
# ==========================

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Current Price",
        f"${current_price:.2f}"
    )

with col2:
    st.metric(
        "Predicted Price",
        f"${predicted_price:.2f}",
        delta=f"{predicted_price-current_price:.2f}"
    )

with col3:
    st.metric(
        "RMSE",
        f"{rmse:.2f}"
    )

with col4:
    st.metric(
        "MAE",
        f"{mae:.2f}"
    )

with col5:
    st.metric(
        "R² Score",
        f"{r2:.3f}"
    )

st.markdown("---")

# ==========================
# Historical Chart
# ==========================

st.subheader("📈 Historical Closing Price")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["Close"],
        mode="lines",
        name="Close Price"
    )
)

fig.update_layout(
    xaxis_title="Days",
    yaxis_title="Price ($)"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.markdown("---")

# ==========================
# Actual vs Predicted
# ==========================

st.subheader("📊 Actual vs Predicted")

fig2 = go.Figure()

fig2.add_trace(
    go.Scatter(
        y=actual.flatten(),
        mode="lines",
        name="Actual Price"
    )
)

fig2.add_trace(
    go.Scatter(
        y=predicted.flatten(),
        mode="lines",
        name="Predicted Price"
    )
)

fig2.update_layout(
    xaxis_title="Days",
    yaxis_title="Price ($)"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

st.markdown("---")

# ==========================
# Latest Prices
# ==========================

st.subheader("📋 Latest 10 Closing Prices")

st.dataframe(
    df.tail(10),
    use_container_width=True
)

st.markdown("---")

# ==========================
# Download CSV
# ==========================

csv = df.to_csv(index=False)

st.download_button(
    "⬇ Download Stock Data",
    csv,
    file_name=f"{stock}.csv",
    mime="text/csv"
)