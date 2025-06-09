import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("📈 Time Series Explorer & Simple Forecasting")

def clean_data(df, date_col, value_col):
    """
    Clean the dataframe:
    - Parse dates in date_col
    - Convert value_col to numeric, coerce errors
    - Drop rows with NaN in either column
    """
    try:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        df = df.dropna(subset=[date_col, value_col])
        return df
    except Exception as e:
        st.error(f"⚠️ Error cleaning data: {e}")
        return None

def plot_time_series(ts, window):
    """
    Plot original time series and rolling mean with the given window size.
    """
    rolling_mean = ts.rolling(window=window).mean()
    plt.figure(figsize=(10,5))
    plt.plot(ts.index, ts, label="Original")
    plt.plot(rolling_mean.index, rolling_mean, label=f"{window}-Day MA")
    plt.legend()
    plt.title("Time Series with Moving Average")
    plt.xlabel("Date")
    plt.ylabel("Value")
    st.pyplot(plt)

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    # Select columns for date and value
    date_col = st.selectbox("Select Date/Time column", options=df.columns)
    value_col = st.selectbox("Select Value column", options=df.columns)

    # Basic validation of columns
    if date_col == value_col:
        st.error("Date and Value columns must be different.")
    else:
        # Clean data
        df_clean = clean_data(df, date_col, value_col)

        if df_clean is not None:
            if df_clean.empty:
                st.error("⚠️ No data remaining after cleaning.")
            else:
                ts = df_clean.set_index(date_col)[value_col]

                st.subheader("Time Series Line Chart")
                st.line_chart(ts)

                # Moving average window slider
                window = st.sidebar.slider("Moving Average Window", 2, 30, 7)
                st.subheader(f"{window}-Day Moving Average Plot")
                plot_time_series(ts, window)
else:
    st.info("Please upload a CSV file to get started.")
