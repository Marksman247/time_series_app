# 📦 Import required libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.gofplots import qqplot
import pmdarima as pm
import warnings

# 📛 Suppress warnings
warnings.filterwarnings("ignore")

# 📄 Streamlit config
st.set_page_config(page_title="Time Series Forecasting App", page_icon="📊", layout="wide")

# ✅ Full-page custom style override — no white bar left anywhere
st.markdown("""
    <style>
    html, body, .stApp {
        background: linear-gradient(to bottom right, #ddeff9, #c6dbf0) !important;
        color: #2E3440 !important;
    }
    header {visibility: hidden;}
    .st-emotion-cache-18ni7ap {background: transparent !important;}
    .block-container {
        padding-top: 2rem;
    }
    h1, h2, h3, h4, h5, h6, p, label, span, div {
        color: #2E3440 !important;
    }
    .stTextInput>div>div>input, .stNumberInput>div>input, .stSelectbox>div>div>div {
        background-color: #ffffff;
        color: #2E3440;
        border: 1px solid #CBD5E1;
        border-radius: 6px;
        padding: 8px;
    }
    .stButton>button {
        background-color: #4C9F70;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #3F8F63;
        color: white;
    }
    .stSidebar {
        background: linear-gradient(to bottom right, #ddeff9, #c6dbf0) !important;
        color: #2E3440;
    }
    </style>
""", unsafe_allow_html=True)

# 📌 Title
st.title("📊 Time Series ARIMA Forecasting App")

# 📌 Sidebar
st.sidebar.header("⚙️ Settings")
forecast_period = st.sidebar.slider("Forecast Period", 5, 60, 20, 5)

st.sidebar.subheader("ARIMA Model")
auto_arima = st.sidebar.checkbox("Auto ARIMA", value=True)
if not auto_arima:
    p = st.sidebar.number_input("AR (p)", 0, 5, 1)
    q = st.sidebar.number_input("MA (q)", 0, 5, 1)
else:
    p = q = None

st.sidebar.markdown("---")
st.sidebar.caption("🔧 Built with ❤️ by Max")

# 📌 CSV upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("### 📋 Raw Data")
        st.dataframe(df.head())

        date_col = st.selectbox("Select Date column", df.columns)
        value_col = st.selectbox("Select Value column", df.columns)

        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        df = df.dropna(subset=[value_col])
        df = df.sort_values(date_col)
        ts = df.set_index(date_col)[value_col]

        st.write("### 📈 Time Series")
        st.line_chart(ts)

        adf_result = adfuller(ts)
        st.write("### 🧪 ADF Test Result")
        st.json({
            "ADF Statistic": adf_result[0],
            "p-value": adf_result[1],
            "Used Lag": adf_result[2],
            "Observations": adf_result[3]
        })

        stationary = adf_result[1] < 0.05
        if stationary:
            st.success("✅ Stationary")
            d = 0
        else:
            st.warning("⚠️ Non-stationary")
            d = st.slider("Differencing (d)", 0, 2, 1)
            if d > 0:
                ts = ts.diff(d).dropna()
                st.line_chart(ts)

        with st.spinner("Fitting model..."):
            if auto_arima:
                model = pm.auto_arima(ts, d=d, seasonal=False, stepwise=True)
                p, d, q = model.order
                st.success(f"Auto ARIMA: p={p}, d={d}, q={q}")
                resid = model.resid
            else:
                model = ARIMA(ts, order=(p, d, q)).fit()
                resid = model.resid

            st.write("### 📊 Model Summary")
            st.text(model.summary())

            if auto_arima:
                forecast, conf_int = model.predict(n_periods=forecast_period, return_conf_int=True)
                freq = pd.infer_freq(ts.index) or 'D'
                forecast_index = pd.date_range(start=ts.index[-1], periods=forecast_period+1, freq=freq)[1:]
                forecast_df = pd.DataFrame({'Forecast': forecast,
                                            'Lower': conf_int[:, 0],
                                            'Upper': conf_int[:, 1]},
                                           index=forecast_index)
            else:
                forecast_result = model.get_forecast(steps=forecast_period)
                forecast_df = forecast_result.summary_frame()

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ts.index, y=ts.values, mode='lines', name='Historical'))
            fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Forecast'], mode='lines', name='Forecast'))
            fig.add_trace(go.Scatter(
                x=list(forecast_df.index) + list(forecast_df.index[::-1]),
                y=list(forecast_df['Upper']) + list(forecast_df['Lower'][::-1]),
                fill='toself', fillcolor='rgba(76,159,112,0.2)', line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval'
            ))
            fig.update_layout(title="Forecast Plot", xaxis_title="Date", yaxis_title=value_col)
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("📄 Upload CSV to get started.")
