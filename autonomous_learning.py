import subprocess
import time
import logging
import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

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

# Run the initial weather analysis
def run_weather_analysis():
    """Runs the weather_analysis.py script."""
    logging.info("Running weather_analysis.py for data retrieval and storage.")
    try:
        # Use 'python' instead of 'python3' for compatibility on all platforms
        subprocess.run(["python", "weather_analysis.py"], check=True)
        logging.info("weather_analysis.py completed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error in weather_analysis.py: {e}")

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

    # Train and tune the model using GridSearchCV
    model = LinearRegression()
    param_grid = {'fit_intercept': [True, False]}
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_scaled, y)
    best_model = grid_search.best_estimator_

    # Prediction for next day
    next_day = np.array([[df['day'].iloc[-1] + 1]])
    next_day_scaled = scaler.transform(next_day)  # No feature name issues here with transform only
    predicted_temp = best_model.predict(next_day_scaled)[0]
    logging.info(f"Predicted temperature for next day: {predicted_temp:.2f}°C")

    # Save model performance
    y_pred = best_model.predict(X_scaled)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    performance = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'mse': mse,
        'mae': mae,
        'predicted_temp': predicted_temp
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
        # Add other models or features here for experimentation if needed

# Run the initial analysis and set a loop to rerun every 24 hours
run_weather_analysis()  # Initial run

while True:
    track_and_improve_model()  # Daily performance tracking and improvement
    logging.info("Waiting 24 hours for the next run.")
    time.sleep(86400)  # Wait 24 hours before rerunning
    run_weather_analysis()  # Refresh data and retrain model with updated dataset
