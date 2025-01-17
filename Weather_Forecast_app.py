
import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder

# Load the saved LSTM model
model_path = 'C:/Users/21. Technology/Downloads/Weather_Forecast_app/weather_forecast_model(lstm).h5'  # Replace with the actual path

saved_model = load_model(model_path)

# Load data and prepare encoders (replace with your actual data path)
data = pd.read_csv('C:/Users/21. Technology/Downloads/Weather_Forecast_app/preprocessed_weather_data.csv')

features_continuous = ['Temp_C', 'Dew Point Temp_C', 'Rel Hum_%', 'Wind Speed_km/h', 'Visibility_km', 'Press_kPa']
scaler = MinMaxScaler()
scaler.fit(data[features_continuous])
label_encoder = LabelEncoder()
data['Weather_Encoded'] = label_encoder.fit_transform(data['Weather'])
onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoder.fit(data['Weather_Encoded'].values.reshape(-1, 1))

# Function to prepare input sequence with the correct length
def prepare_input_sequence(date_time, sequence_length=72):
    target_datetime = datetime.strptime(date_time, "%Y-%m-%d %H:%M:%S")

    # Ensure that 'Date/Time' column in the data is datetime type
    data['Date/Time'] = pd.to_datetime(data['Date/Time'])

    # Identify the last available sequence of data before the given date_time
    data_subset = data[data['Date/Time'] < target_datetime].tail(sequence_length)
    if len(data_subset) < sequence_length:
        raise ValueError("Insufficient historical data to create the input sequence.")

    scaled_features = scaler.transform(data_subset[features_continuous])
    weather_labels = label_encoder.transform(data_subset['Weather'])
    weather_onehot = onehot_encoder.transform(weather_labels.reshape(-1, 1))
    input_sequence = np.hstack((scaled_features, weather_onehot))
    return input_sequence.reshape(1, sequence_length, -1)

# Function to decode weather
def decode_weather(encoded_weather):
    return label_encoder.inverse_transform(onehot_encoder.inverse_transform(encoded_weather).argmax(axis=1))

# Function to forecast specific date and time
def forecast_specific_date_time(date_time):
    sequence_length = 72  # Updated to match the model's expected input
    input_sequence = prepare_input_sequence(date_time, sequence_length)
    prediction = saved_model.predict(input_sequence)
    reversed_continuous = scaler.inverse_transform(prediction[:, :len(features_continuous)])
    predicted_weather_onehot = prediction[:, len(features_continuous):]
    predicted_weather = decode_weather(predicted_weather_onehot)
    forecast_data = pd.DataFrame(reversed_continuous, columns=features_continuous)
    forecast_data['Date/Time'] = date_time
    forecast_data['Weather'] = predicted_weather[0]
    return forecast_data[['Date/Time'] + features_continuous + ['Weather']]

# Function to simulate forecast for a specific date and hour
def simulate_forecast_specific_date_time(date_time, sequence_length):
    data = {
        'Date/Time': [date_time],
        'Temp_C': np.random.normal(15, 5, sequence_length),
        'Dew Point Temp_C': np.random.normal(10, 5, sequence_length),
        'Rel Hum_%': np.random.uniform(50, 100, sequence_length),
        'Wind Speed_km/h': np.random.uniform(5, 25, sequence_length),
        'Visibility_km': np.random.uniform(1, 20, sequence_length),
        'Press_kPa': np.random.uniform(98, 102, sequence_length),
        'Weather': np.random.choice(['Rain', 'Clear', 'Cloudy', 'Snow'], sequence_length)
    }
    return pd.DataFrame(data)

# Function to forecast next n days
def forecast_next_n_days(start_date, n_days, sequence_length):
    future_dates = [datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S") + timedelta(hours=hour) for hour in range(n_days * 24)]
    forecasts = [simulate_forecast_specific_date_time(date_time.strftime("%Y-%m-%d %H:%M:%S"), 1) for date_time in future_dates]
    return pd.concat(forecasts, ignore_index=True)

# Streamlit app
st.title("Weather Forecast App")
st.sidebar.header("Forecast Options")
forecast_type = st.sidebar.selectbox("Select Forecast Type", ["Specific Date and Time", "Specific Date", "Next 7 Days"])

if forecast_type == "Specific Date and Time":
    date_time = st.sidebar.text_input("Enter Date and Time (YYYY-MM-DD HH:MM:SS)")
    if st.sidebar.button("Forecast"):
        try:
            forecast_result = forecast_specific_date_time(date_time)
            st.write(forecast_result)
            st.download_button("Download CSV", forecast_result.to_csv(index=False), "forecast.csv", "text/csv")
        except Exception as e:
            st.error(f"Error: {e}")

elif forecast_type == "Specific Date":
    date = st.sidebar.date_input("Select Date")
    date_str = date.strftime("%Y-%m-%d")
    if st.sidebar.button("Forecast"):
        hourly_forecast = pd.concat([simulate_forecast_specific_date_time(f"{date_str} {hour:02d}:00:00", 1) for hour in range(24)], ignore_index=True)
        st.write(hourly_forecast)
        st.download_button("Download CSV", hourly_forecast.to_csv(index=False), "hourly_forecast.csv", "text/csv")

else:
    start_date = st.sidebar.text_input("Enter Start Date (YYYY-MM-DD HH:MM:SS)")
    if st.sidebar.button("Forecast"):
        forecast_7_days = forecast_next_n_days(start_date, 7, 1)
        st.write(forecast_7_days)
        st.download_button("Download CSV", forecast_7_days.to_csv(index=False), "forecast_7_days.csv", "text/csv")
