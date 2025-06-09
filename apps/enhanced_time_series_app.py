# Suppress warnings for a cleaner Streamlit output
import warnings
warnings.filterwarnings("ignore")

# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Streamlit App Title
st.title("Enhanced Time Series Explorer & ARIMA Forecasting")

# Upload CSV File Widget
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Proceed if a file is uploaded
if uploaded_file:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)
    st.write("Raw Data Preview:")
    st.dataframe(df.head())

    # User selects Date/Time and Value columns
    date_col = st.selectbox("Select Date/Time column", df.columns)
    value_col = st.selectbox("Select Value column", df.columns)

    try:
        # Convert selected date column to datetime
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])

        # Convert selected value column to numeric
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        df = df.dropna(subset=[value_col])

        # Check if any valid data remains after cleaning
        if df.empty:
            st.error("No valid data found after cleaning.")
            st.stop()

        # Set datetime index and sort chronologically
        df = df.set_index(date_col).sort_index()

        st.write("Cleaned Data Preview:")
        st.dataframe(df.head())

        # Extract the time series (target variable)
        ts = df[value_col]

        # Plot the original time series
        st.subheader("Original Time Series")
        fig, ax = plt.subplots()
        ts.plot(ax=ax)
        st.pyplot(fig)

        # Sidebar: ARIMA parameter inputs
        st.sidebar.subheader("ARIMA Model Parameters")
        p = st.sidebar.number_input("AR (p)", min_value=0, max_value=5, value=1)
        d = st.sidebar.number_input("I (d)", min_value=0, max_value=2, value=1)
        q = st.sidebar.number_input("MA (q)", min_value=0, max_value=5, value=1)

        # Sidebar: Forecast period input
        forecast_period = st.sidebar.number_input("Forecast Period (steps)", min_value=1, max_value=100, value=10)

        # Fit the ARIMA model
        try:
            model = ARIMA(ts, order=(p, d, q))
            model_fit = model.fit()

            # Forecast future values
            forecast = model_fit.forecast(steps=forecast_period)

            # Plot forecast alongside original series
            st.subheader("Forecasted Values")
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ts.plot(ax=ax2, label='Original')

            # Build future dates for the forecast
            last_date = ts.index[-1]
            inferred_freq = pd.infer_freq(ts.index) or 'D'  # fallback to 'D' if frequency missing
            forecast_index = pd.date_range(start=last_date, periods=forecast_period+1, freq=inferred_freq).shift(1)[1:]

            ax2.plot(forecast_index, forecast, color='orange', linestyle='--', label='Forecast')
            ax2.legend()
            st.pyplot(fig2)

            # Display forecast values in a table
            forecast_df = pd.DataFrame({
                'Forecast Date': forecast_index,
                'Forecasted Value': forecast.values
            })
            st.dataframe(forecast_df)

            # Calculate and display Mean Squared Error (MSE) on training data fit
            fitted_values = model_fit.fittedvalues
            mse = mean_squared_error(ts[d:], fitted_values[d:])
            st.write(f"Mean Squared Error (on training data): {mse:.4f}")

        except Exception as e:
            st.error(f"Error fitting ARIMA model: {e}")

    except Exception as e:
        st.error(f"Error processing data: {e}")

else:
    st.info("Please upload a CSV file to get started.")
