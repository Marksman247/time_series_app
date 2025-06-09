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

# 📛 Suppress warnings for a cleaner Streamlit interface
warnings.filterwarnings("ignore")

# 📄 Page configuration for Streamlit app
st.set_page_config(page_title="Time Series Forecasting App", page_icon="📈", layout="wide")

# 📌 App Title
st.title("📈 Advanced Time Series Analysis with ARIMA Forecasting")

# 📌 Sidebar controls — app settings and ARIMA options
st.sidebar.header("⚙️ App Settings")

# Forecast period control
forecast_period = st.sidebar.slider(
    "Select Forecast Period (steps ahead)",
    min_value=5,
    max_value=60,
    value=20,
    step=5
)

# ARIMA model option: Auto or Manual
st.sidebar.subheader("ARIMA Model Options")
auto_arima = st.sidebar.checkbox("Use Auto ARIMA (auto-select p, d, q)", value=True)

if not auto_arima:
    p = st.sidebar.number_input("AR order (p)", min_value=0, max_value=5, value=1)
    q = st.sidebar.number_input("MA order (q)", min_value=0, max_value=5, value=1)
else:
    p = q = None  # Auto-selected later

st.sidebar.markdown("---")
st.sidebar.caption("Built with ❤️ by Max")

# 📌 Upload CSV file for analysis
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Load dataset and preview
        df = pd.read_csv(uploaded_file)
        st.write("### 📋 Raw Data Preview")
        st.dataframe(df.head())

        # Select date and value columns
        date_col = st.selectbox("Select Date/Time column", df.columns)
        value_col = st.selectbox("Select Value column", df.columns)

        # Parse dates and clean data
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        df = df.sort_values(by=date_col)
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        df = df.dropna(subset=[value_col])

        if df.empty:
            st.error("No valid data after cleaning. Please check your file.")
            st.stop()

        ts = df.set_index(date_col)[value_col]

        # 📊 Plot raw time series
        st.write("### 📈 Time Series Plot")
        st.line_chart(ts)

        # 📊 Augmented Dickey-Fuller Stationarity Test
        adf_result = adfuller(ts)
        st.write("### 🧪 Augmented Dickey-Fuller Test Result")
        st.json({
            "ADF Statistic": adf_result[0],
            "p-value": adf_result[1],
            "Used Lag": adf_result[2],
            "Number of Observations": adf_result[3]
        })

        # Check stationarity based on p-value
        stationary = adf_result[1] < 0.05
        if stationary:
            st.success("✅ The time series is likely stationary (reject H₀).")
            d = 0
        else:
            st.warning("⚠️ The time series is likely non-stationary (fail to reject H₀).")
            d = st.slider("Select degree of differencing (d)", 0, 2, 1)
            if d > 0:
                ts = ts.diff(d).dropna()
                st.write(f"### Differenced Series (d={d})")
                st.line_chart(ts)

        # 📈 Fit ARIMA model and forecast
        with st.spinner("Fitting ARIMA model..."):
            try:
                if auto_arima:
                    # Auto ARIMA model selection
                    st.info("Running Auto ARIMA to find best parameters...")
                    auto_model = pm.auto_arima(ts, d=d, seasonal=False,
                                               stepwise=True, suppress_warnings=True,
                                               error_action='ignore', trace=False)
                    p, d, q = auto_model.order
                    st.success(f"Auto ARIMA selected order: p={p}, d={d}, q={q}")
                    model_fit = auto_model
                    resid = auto_model.resid

                else:
                    # Manual ARIMA model fitting
                    model = ARIMA(ts, order=(p, d, q))
                    model_fit = model.fit()
                    resid = model_fit.resid

                # Display model summary
                st.write("### 📊 ARIMA Model Summary")
                if auto_arima:
                    st.text(str(model_fit.summary()))
                else:
                    st.text(model_fit.summary().as_text())

                # Forecast future values
                if auto_arima:
                    forecast, conf_int = model_fit.predict(n_periods=forecast_period, return_conf_int=True)
                    last_date = ts.index[-1]
                    freq = pd.infer_freq(ts.index) or 'D'
                    forecast_index = pd.date_range(start=last_date, periods=forecast_period + 1, freq=freq)[1:]
                    forecast_df = pd.DataFrame({
                        'mean': forecast,
                        'mean_ci_lower': conf_int[:, 0],
                        'mean_ci_upper': conf_int[:, 1]
                    }, index=forecast_index)
                else:
                    forecast_result = model_fit.get_forecast(steps=forecast_period)
                    forecast_df = forecast_result.summary_frame()

                # 📈 Plotly interactive plot
                fig = go.Figure()

                # Add historical data
                fig.add_trace(go.Scatter(x=ts.index, y=ts.values,
                                         mode='lines', name='Historical'))

                # Add forecasted values
                fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['mean'],
                                         mode='lines', name='Forecast'))

                # Add confidence interval
                fig.add_trace(go.Scatter(
                    x=list(forecast_df.index) + list(forecast_df.index[::-1]),
                    y=list(forecast_df['mean_ci_upper']) + list(forecast_df['mean_ci_lower'][::-1]),
                    fill='toself',
                    fillcolor='rgba(255, 182, 193, 0.3)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=True,
                    name='Confidence Interval'
                ))

                fig.update_layout(title="📊 Time Series Forecast with Confidence Intervals",
                                  xaxis_title="Date",
                                  yaxis_title=value_col)

                st.plotly_chart(fig, use_container_width=True)

                # 📏 Forecast error metrics (only for manual model)
                if not auto_arima:
                    train_pred = model_fit.predict(start=ts.index[0], end=ts.index[-1])
                    mae = mean_absolute_error(ts, train_pred)
                    rmse = mean_squared_error(ts, train_pred) ** 0.5
                    st.write("### 📉 Forecast Error on Training Data")
                    st.write(f"**Mean Absolute Error (MAE)**: {mae:.3f}")
                    st.write(f"**Root Mean Squared Error (RMSE)**: {rmse:.3f}")

                # 📊 Diagnostic plots for residuals
                st.write("### 🔍 Model Diagnostics")

                fig_diag, axs = plt.subplots(2, 2, figsize=(12, 8))

                # Residuals over time
                axs[0, 0].plot(resid)
                axs[0, 0].set_title("Residuals Over Time")
                axs[0, 0].axhline(0, linestyle='--', color='gray')

                # ACF plot
                plot_acf(resid, ax=axs[0, 1], title="ACF of Residuals")

                # PACF plot
                plot_pacf(resid, ax=axs[1, 0], title="PACF of Residuals")

                # QQ plot
                qqplot(resid, line='s', ax=axs[1, 1])
                axs[1, 1].set_title("QQ Plot of Residuals")

                plt.tight_layout()
                st.pyplot(fig_diag)

            except Exception as e:
                st.error(f"Error fitting ARIMA model: {e}")

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.info("📄 Please upload a CSV file to get started.")
