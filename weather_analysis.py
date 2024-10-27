import json
import requests
import pandas as pd
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import re
import logging

# Set up logging
logging.basicConfig(
    filename="weather_analysis.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load configuration
with open("config.json", "r") as f:
    config = json.load(f)

location = config['location']
period = config['period']
api_key = config['api_key']

# Fetch historical weather data
def get_weather_data(location, days, api_key):
    logging.info("Fetching historical weather data.")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    weather_data = []

    for day in range(days):
        date = (start_date + timedelta(days=day)).strftime('%Y-%m-%d')
        url = f"https://api.weatherapi.com/v1/history.json?key={api_key}&q={location}&dt={date}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            daily_data = {
                "date": date,
                "temp": data['forecast']['forecastday'][0]['day']['avgtemp_c'],
                "condition": data['forecast']['forecastday'][0]['day']['condition']['text']
            }
            weather_data.append(daily_data)
            logging.info(f"Data for {date} added successfully.")
        else:
            logging.error(f"Failed to fetch data for {date}. Status code: {response.status_code}")

    return weather_data

# Store data in Excel
weather_data = get_weather_data(location, period, api_key)
df = pd.DataFrame(weather_data)
df.to_excel("weather_data.xlsx", index=False)
logging.info("Weather data stored in weather_data.xlsx.")

# Predict the next day's weather
def predict_next_day(data):
    last_temp = data['temp'].iloc[-period:]
    next_temp = last_temp.mean()
    common_condition = data['condition'].mode()[0]
    logging.info("Prediction completed.")
    return {"predicted_temp": next_temp, "predicted_condition": common_condition}

# Fetch tomorrow's temperature from Google
def fetch_google_temperature(location):
    logging.info("Fetching temperature data from Google.")
    search_query = f"{location} temperature tomorrow"
    url = f"https://www.google.com/search?q={search_query}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36"
    }
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        temperature_element = soup.select_one('span#wob_tm.wob_t.q8U8x')
        if temperature_element:
            temperature_data = temperature_element.text.strip()
            logging.info("Google temperature data fetched successfully.")
            return temperature_data
    logging.warning("Could not retrieve temperature data from Google.")
    return None

# Evaluate probability of prediction accuracy
def evaluate_probability(predicted_temp, google_temp):
    predicted_temp_val = round(predicted_temp)
    google_temp_val = int(re.search(r'\d+', google_temp).group())

    if abs(predicted_temp_val - google_temp_val) <= 10:
        return "70-100% likely to happen"
    elif abs(predicted_temp_val - google_temp_val) <= 20:
        return "50-70% likely to happen"
    else:
        return "20-50% unlikely to happen"

# Enhanced output for prediction
prediction = predict_next_day(df)
print(f"Weather for tomorrow is likely to be {prediction['predicted_condition'].lower()} with an average temperature of approximately {prediction['predicted_temp']}°C.")

# Check Google for tomorrow's temperature and evaluate probability
google_temp = fetch_google_temperature(location)
if google_temp:
    print(f"Google shows tomorrow's temperature as: {google_temp}°C.")
    probability = evaluate_probability(prediction['predicted_temp'], google_temp)
    print(f"Probability of prediction accuracy: {probability}")
else:
    print("Note: Could not retrieve temperature data from Google. Please check Google for the most accurate answer.")

logging.info("Process completed.")
