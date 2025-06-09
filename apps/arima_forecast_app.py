# 📊 ARIMA Forecasting App built by MAX

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

# App Title
st.title("📈 ARIMA Time Series Forecasting App")
st.write("Built by **MAX**")

# Sidebar Config
st.sidebar.header("Upload & Configure Data")

# File Uploader
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.success("✅ Data uploaded successfully!")
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")

if st.session_state.df is not None:
    df = st.session_state.df
    st.write("### 📄 Raw Data Preview")
    st.dataframe(df.head())

    # Column selection
    date_col = st.sidebar.selectbox("Select Date Column", df.columns)
    value_col = st.sidebar.selectbox("Select Value Column to Forecast", df.columns)

    # Forecasting parameters
    forecast_period = st.sidebar.number_input("Forecast Period (future points)", min_value=1, max_value=1000, value=10)

    auto_arima_option = st.sidebar.checkbox("Use Auto ARIMA (recommended)", value=True)

    # Process and Forecast button
    if st.sidebar.button("Run Forecast"):
        try:
            # Convert date column to datetime
            df[date_col] = pd.to_datetime(df[date_col])

            # Set date as index and sort
            ts = df[[date_col, value_col]].dropna().set_index(date_col)
            ts = ts.sort_index()

            st.write("### 📊 Time Series Data")
            st.line_chart(ts)

            # Model fitting
            if auto_arima_option:
                st.write("### 🔍 Running Auto ARIMA...")
                model_fit = auto_arima(ts[value_col], seasonal=False, stepwise=True, suppress_warnings=True)
            else:
                st.write("### 🔍 Fitting ARIMA(1,1,1)...")
                model = ARIMA(ts[value_col], order=(1, 1, 1))
                model_fit = model.fit()

            # Show model summary
            st.write("### 📜 ARIMA Model Summary")
            if auto_arima_option:
                st.text(model_fit.summary())
            else:
                st.text(model_fit.summary().as_text())

            # Forecasting
            if auto_arima_option:
                forecast_values, conf_int = model_fit.predict(n_periods=forecast_period, return_conf_int=True)
                last_date = ts.index[-1]
                freq = pd.infer_freq(ts.index) or 'D'
                forecast_index = pd.date_range(start=last_date, periods=forecast_period + 1, freq=freq)[1:]

                forecast_df = pd.DataFrame({
                    'Forecast': forecast_values,
                    'Lower CI': conf_int[:, 0],
                    'Upper CI': conf_int[:, 1]
                }, index=forecast_index)

            else:
                forecast_result = model_fit.get_forecast(steps=forecast_period)
                forecast_df = forecast_result.summary_frame()
                forecast_df.index.name = ts.index.name

            # Display Forecast Table
            st.write("### 📊 Forecasted Values")
            st.dataframe(forecast_df)

            # Plot Forecast
            st.write("### 📈 Forecast Plot")
            fig, ax = plt.subplots(figsize=(10, 5))

            ts[value_col].plot(ax=ax, label='Actual', color='blue')

            if auto_arima_option:
                ax.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast', color='green')
                ax.fill_between(forecast_df.index, forecast_df['Lower CI'], forecast_df['Upper CI'], color='lightgreen', alpha=0.3)
            else:
                ax.plot(forecast_df.index, forecast_df['mean'], label='Forecast', color='green')
                ax.fill_between(forecast_df.index, forecast_df['mean_ci_lower'], forecast_df['mean_ci_upper'], color='lightgreen', alpha=0.3)

            ax.legend()
            ax.set_title("Forecast vs Actual")
            ax.set_xlabel("Date")
            ax.set_ylabel(value_col)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Error running forecast: {e}")

else:
    st.info("📤 Please upload a CSV file to begin.")

# End of App
