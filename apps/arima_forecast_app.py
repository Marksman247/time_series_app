# 📊 ARIMA Forecasting App built by MAX (pmdarima-free)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

# Ignore harmless warnings
warnings.filterwarnings("ignore")

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
    auto_arima_option = st.sidebar.checkbox("Use Auto ARIMA (AIC-based search)", value=True)

    # Process and Forecast button
    if st.sidebar.button("Run Forecast"):
        try:
            # Convert date column to datetime
            df[date_col] = pd.to_datetime(df[date_col])

            # Set date as index and sort
            ts = df[[date_col, value_col]].dropna().set_index(date_col).sort_index()

            st.write("### 📊 Time Series Data")
            st.line_chart(ts)

            # Model fitting
            if auto_arima_option:
                st.write("### 🔍 Running AIC-based ARIMA search...")
                best_aic = np.inf
                best_order = None
                best_model = None

                # Grid search on (p,d,q)
                for p in range(0, 3):
                    for d in range(0, 2):
                        for q in range(0, 3):
                            try:
                                model = SARIMAX(ts[value_col], order=(p, d, q), enforce_stationarity=False, enforce_invertibility=False)
                                results = model.fit(disp=False)
                                if results.aic < best_aic:
                                    best_aic = results.aic
                                    best_order = (p, d, q)
                                    best_model = results
                            except:
                                continue

                st.success(f"✅ Best ARIMA order: {best_order} (AIC: {best_aic:.2f})")

            else:
                st.write("### 🔍 Fitting ARIMA(1,1,1)...")
                model = SARIMAX(ts[value_col], order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False)
                best_model = model.fit(disp=False)
                best_order = (1, 1, 1)

            # Show model summary
            st.write("### 📜 ARIMA Model Summary")
            st.text(best_model.summary())

            # Forecasting
            forecast_result = best_model.get_forecast(steps=forecast_period)
            forecast_df = forecast_result.summary_frame()
            forecast_df.index.name = ts.index.name

            # Display Forecast Table
            st.write("### 📊 Forecasted Values")
            st.dataframe(forecast_df)

            # Plot Forecast
            st.write("### 📈 Forecast Plot")
            fig, ax = plt.subplots(figsize=(10, 5))
            ts[value_col].plot(ax=ax, label='Actual', color='blue')
            ax.plot(forecast_df.index, forecast_df['mean'], label='Forecast', color='green')
            ax.fill_between(forecast_df.index, forecast_df['mean_ci_lower'], forecast_df['mean_ci_upper'], color='lightgreen', alpha=0.3)
            ax.legend()
            ax.set_title(f"Forecast vs Actual (ARIMA{best_order})")
            ax.set_xlabel("Date")
            ax.set_ylabel(value_col)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Error running forecast: {e}")

else:
    st.info("📤 Please upload a CSV file to begin.")

# End of App
