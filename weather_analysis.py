import json
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import re
import logging
import os
import mysql.connector
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import time

# Set up logging
log_file = "weather_analysis.log"
if os.path.exists(log_file):
    os.remove(log_file)

logging.basicConfig(
    filename=log_file,
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load configuration
print("Loading configuration from config.json...")
with open("config.json", "r") as f:
    config = json.load(f)
print("Configuration loaded successfully!")

location = config['location']
period = config['period']
api_key = config['api_key']
excel_enabled = config.get('excel', False)
mysql_enabled = config.get('mysql', False)
excel_file = "weather_data.xlsx"
training_timeout = config.get('training_timeout_minutes', 5) * 60  # in seconds

# Remove existing Excel file if it exists
if excel_enabled and os.path.exists(excel_file):
    os.remove(excel_file)
    print(f"Removed existing {excel_file} to prepare for new data storage.")

# MySQL database connection setup
db_connection, cursor = None, None
if mysql_enabled:
    print("Attempting to connect to MySQL database...")
    try:
        db_connection = mysql.connector.connect(
            host=config['db_host'],
            user=config['db_user'],
            password=config['db_password'],
            database=config['db_name']
        )
        cursor = db_connection.cursor()
        logging.info("Connected to MySQL database successfully.")
        print("Connected to MySQL database successfully.")
        
        # Create table if not exists without clearing existing data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS weather_data (
                id INT AUTO_INCREMENT PRIMARY KEY,
                date DATE,
                temp FLOAT,
                humidity FLOAT,
                wind_speed FLOAT,
                `condition` VARCHAR(255)
            )
        ''')
        logging.info("Checked for table existence; table ready for data insertion.")
        print("Table checked and ready for data insertion.")
    except mysql.connector.Error as err:
        logging.error(f"Error connecting to MySQL: {err}")
        print(f"Error connecting to MySQL: {err}")

# Fetch historical weather data
def get_weather_data(location, days, api_key):
    print("Fetching historical weather data...")
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
                "humidity": data['forecast']['forecastday'][0]['day']['avghumidity'],
                "wind_speed": data['forecast']['forecastday'][0]['day']['maxwind_kph'],
                "condition": data['forecast']['forecastday'][0]['day']['condition']['text']
            }
            weather_data.append(daily_data)
            logging.info(f"Data for {date} added successfully.")
            print(f"Data for {date} fetched and added successfully.")
        else:
            logging.error(f"Failed to fetch data for {date}. Status code: {response.status_code}")
            print(f"Failed to fetch data for {date}. Status code: {response.status_code}")

    return weather_data

# Store data in Excel and/or MySQL
weather_data = get_weather_data(location, period, api_key)
df = pd.DataFrame(weather_data)

# Excel file storage
if excel_enabled:
    df.to_excel(excel_file, index=False)
    logging.info(f"Weather data stored in {excel_file}.")
    print(f"Weather data successfully stored in {excel_file}.")

# MySQL storage
if mysql_enabled and db_connection.is_connected():
    print("Inserting data into MySQL database...")
    for row in weather_data:
        try:
            cursor.execute(
                "INSERT INTO weather_data (date, temp, humidity, wind_speed, `condition`) VALUES (%s, %s, %s, %s, %s)",
                (row['date'], row['temp'], row['humidity'], row['wind_speed'], row['condition'])
            )
            db_connection.commit()
            logging.info(f"Data for {row['date']} inserted into the database successfully.")
            print(f"Data for {row['date']} inserted into MySQL database successfully.")
        except mysql.connector.Error as err:
            logging.error(f"Failed to insert data for {row['date']}: {err}")
            print(f"Failed to insert data for {row['date']}: {err}")

# Data Analysis and Visualization
def plot_weather_data(df):
    print("Generating weather data plots...")
    plt.figure(figsize=(10, 6))
    sns.lineplot(x="date", y="temp", data=df, label="Temperature")
    sns.lineplot(x="date", y="humidity", data=df, label="Humidity")
    plt.title("Temperature and Humidity Trends")
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()

plot_weather_data(df)

# Predictive Model with GridSearchCV and Cross-Validation
def train_predictive_model(df):
    print("Training predictive model with cross-validation and grid search...")
    df['day'] = range(len(df))
    X = df[['day']]
    y = df['temp']
    
    start_time = time.time()
    model = LinearRegression()
    scaler = StandardScaler()

    # Scale the input features
    X_scaled = scaler.fit_transform(X)

    # Grid search with cross-validation
    param_grid = {'fit_intercept': [True, False]}
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    
    try:
        grid_search.fit(X_scaled, y)
        logging.info("Predictive model trained successfully with GridSearchCV.")
    except Exception as e:
        logging.error(f"Error in model training: {e}")
        print("Error in training model:", e)
        return None

    best_model = grid_search.best_estimator_
    print(f"Best parameters found: {grid_search.best_params_}")
    logging.info(f"Best parameters from GridSearchCV: {grid_search.best_params_}")

    return best_model, scaler

model, scaler = train_predictive_model(df)

# Prediction and Evaluation
def predict_next_day(model, scaler, df):
    next_day = df['day'].iloc[-1] + 1
    next_day_scaled = scaler.transform([[next_day]])
    predicted_temp = model.predict(next_day_scaled)[0]
    logging.info("Next day's prediction completed.")
    return predicted_temp

predicted_temp = predict_next_day(model, scaler, df)
print(f"Predicted temperature for tomorrow: approx. {predicted_temp:.2f}°C")

# Model Evaluation
def evaluate_model(model, X_scaled, y):
    y_pred = model.predict(X_scaled)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    logging.info(f"Model Evaluation - MSE: {mse}, MAE: {mae}")
    print(f"Model Evaluation - Mean Squared Error: {mse:.2f}, Mean Absolute Error: {mae:.2f}")

evaluate_model(model, scaler.transform(df[['day']]), df['temp'])

# Fetch tomorrow's temperature from Google
def fetch_google_temperature(location):
    print("Fetching tomorrow's temperature from Google...")
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
            print(f"Google's predicted temperature for tomorrow is {temperature_data}°C.")
            return temperature_data
    logging.warning("Could not retrieve temperature data from Google.")
    print("Warning: Could not retrieve temperature data from Google.")
    return None

# Evaluate probability of prediction accuracy
def evaluate_probability(predicted_temp, google_temp):
    predicted_temp_val = round(predicted_temp)
    google_temp_val = int(re.search(r'\d+', google_temp).group())
    difference = abs(predicted_temp_val - google_temp_val)
    
    if difference <= 2:
        accuracy = "90-100% likely to happen"
    elif difference <= 5:
        accuracy = "70-90% likely to happen"
    elif difference <= 10:
        accuracy = "50-70% likely to happen"
    else:
        accuracy = "Below 50% accuracy - adjust prediction model"
    
    print(f"Evaluating prediction accuracy based on Google's data: {accuracy}")
    return accuracy

google_temp = fetch_google_temperature(location)
if google_temp:
    print(f"Google shows tomorrow's temperature as: {google_temp}°C.")
    probability = evaluate_probability(predicted_temp, google_temp)
    print(f"Probability of prediction accuracy: {probability}")
else:
    print("Note: Could not retrieve temperature data from Google. Please check Google for the most accurate answer.")

logging.info("Process completed.")
print("Process completed.")

# Close MySQL connection if open
if mysql_enabled and db_connection.is_connected():
    cursor.close()
    db_connection.close()
    logging.info("MySQL connection closed.")
    print("MySQL connection closed.")
