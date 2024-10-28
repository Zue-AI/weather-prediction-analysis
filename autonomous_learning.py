import subprocess
import logging
import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle

# Set up logging
log_file = "autonomous_learning.log"
if os.path.exists(log_file):
    os.remove(log_file)

logging.basicConfig(
    filename=log_file,
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Paths and file names
data_file = "weather_data.xlsx"
performance_log = "model_performance_log.csv"
model_file = "best_model.pkl"

# Run the initial weather analysis
def run_weather_analysis():
    """Runs the weather_analysis.py script."""
    logging.info("Running weather_analysis.py for data retrieval and storage.")
    try:
        subprocess.run(["python", "weather_analysis.py"], check=True)
        logging.info("weather_analysis.py completed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error in weather_analysis.py: {e}")

# Function to predict weather condition based on temperature
def predict_condition(temp):
    """Predicts the weather condition based on the temperature."""
    if temp > 30:
        return "Sunny"
    elif 20 <= temp <= 30:
        return "Partly Cloudy"
    elif 10 <= temp < 20:
        return "Cloudy"
    else:
        return "Rainy"

# Calculate model accuracy and adjust the model if needed
def track_and_improve_model():
    """Tracks prediction accuracy, expands features, and tunes model parameters."""
    
    # Load weather data
    if not os.path.exists(data_file):
        logging.error("Weather data file not found. Ensure weather_analysis.py is run first.")
        return
    
    df = pd.read_excel(data_file)
    df['day'] = range(len(df))  # Numeric day feature for prediction
    
    # Define features and labels
    X = df[['day']]
    y = df['temp']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Model loading or initialization
    if os.path.exists(model_file):
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        logging.info("Loaded existing model.")
    else:
        model = RandomForestRegressor()
        logging.info("No existing model found; initialized new RandomForestRegressor model.")
    
    # Train and tune the model using GridSearchCV
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_scaled, y)
    best_model = grid_search.best_estimator_
    
    # Save the best model
    with open(model_file, 'wb') as f:
        pickle.dump(best_model, f)
    logging.info("Saved the best model to disk.")
    
    # Prediction for the next day
    next_day = np.array([[df['day'].iloc[-1] + 1]])
    next_day_scaled = scaler.transform(next_day)
    predicted_temp = best_model.predict(next_day_scaled)[0]
    predicted_condition = predict_condition(predicted_temp)
    logging.info(f"Predicted temperature for next day: {predicted_temp:.2f}°C")
    logging.info(f"Predicted weather condition for next day: {predicted_condition}")
    
    # Save model performance
    y_pred = best_model.predict(X_scaled)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    performance = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'mse': mse,
        'mae': mae,
        'predicted_temp': predicted_temp,
        'predicted_condition': predicted_condition
    }
    
    # Log performance data
    performance_df = pd.DataFrame([performance])  # Wrap the new row as a DataFrame
    if os.path.exists(performance_log):
        previous_df = pd.read_csv(performance_log)
        performance_df = pd.concat([previous_df, performance_df], ignore_index=True)
    performance_df.to_csv(performance_log, index=False)
    logging.info(f"Model performance logged with MSE: {mse:.2f} and MAE: {mae:.2f}")
    
    # Adjust model complexity if error is high
    if mse > 10:  # Arbitrary threshold for demonstration; adjust based on performance
        logging.warning("High error detected. Considering alternative models.")

# Execute main functions without time constraints
run_weather_analysis()  # Initial data retrieval and storage
track_and_improve_model()  # Performance tracking and improvement
