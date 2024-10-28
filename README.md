
# Weather Prediction and Analysis

This project fetches historical weather data, builds a predictive model for future temperatures, autonomously improves this model, and includes a viewer to inspect the model’s saved parameters and performance.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [License](#license)

### Introduction
This Python project leverages historical weather data to make next-day temperature predictions, evaluates model accuracy against Google’s forecast, autonomously retrains the model based on performance, and provides visualization for analysis. This project is useful for those interested in time series prediction, data analysis, and model tracking for weather forecasting.

### Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Zue-AI/weather-prediction-analysis.git
   ```
2. **Install the Required Libraries**:
   ```bash
   pip install -r requirements.txt
   ```

### Usage
1. **Configure Settings**:
   - Populate `config.json` with your API key, location, and configuration options:
     - `api_key`: Your WeatherAPI key.
     - `location`: Target location for weather data.
     - `period`: Number of past days for data retrieval.
     - Options for saving data to Excel (`excel`) or MySQL (`mysql`).

2. **Run Scripts**:
   - **Fetch and Analyze Data**:
     ```bash
     python weather_analysis.py
     ```
   - **Autonomous Model Improvement**:
     - Run `autonomous_learning.py` to retrain and tune the model based on its latest performance, storing the best-performing model:
     ```bash
     python autonomous_learning.py
     ```
   - **Inspect Saved Model Parameters**:
     - Use `model_viewer.py` to view the saved model’s parameters, such as feature importances, accuracy, and other details:
     ```bash
     python model_viewer.py
     ```

### Project Structure
- `weather_analysis.py`: Core script for fetching data, making predictions, and evaluating results.
- `autonomous_learning.py`: Script to retrain and tune the model based on recent performance metrics, updating the model if accuracy improves.
- `model_viewer.py`: Tool for viewing the saved model’s parameters and performance metrics.
- `config.json`: Configuration file with API key, location, and other settings.
- `requirements.txt`: Required dependencies for the project.
- `weather_data.xlsx`: Excel output file (if enabled) for historical weather data.
- `weather_analysis.log`: Log file for tracing detailed execution and debugging.

### Technical Details
- **Data Sources**: Retrieves historical weather data from [WeatherAPI](https://www.weatherapi.com/) and Google’s forecasted data for accuracy validation.
- **Prediction Algorithm**: Uses linear regression to predict temperatures; `autonomous_learning.py` tunes parameters and selects the best model.
- **Model Tracking**: `model_viewer.py` allows users to inspect saved model parameters and performance.
- **Error Logging**: Logs all operations, including data retrieval, model training, and evaluation, for better traceability.
- **Performance Metrics**: Measures prediction accuracy using Mean Squared Error (MSE) and Mean Absolute Error (MAE).

### License
This project is licensed under the MIT License.
