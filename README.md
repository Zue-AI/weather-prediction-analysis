
# Weather Prediction and Analysis

This project fetches and analyzes historical weather data, makes weather predictions using a predictive model, and evaluates accuracy by comparing predictions to Google’s forecasted temperatures.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [License](#license)

### Introduction
This Python-based project leverages historical weather data to make next-day weather predictions, evaluates accuracy using Google’s temperature forecast, and provides data visualization and analysis insights. This tool is useful for anyone interested in time series prediction and data analysis, particularly in weather forecasting.

### Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/Zue-AI/weather-prediction-analysis.git
   ```
2. **Install the required libraries**:
   ```bash
   pip install -r requirements.txt
   ```

### Usage
1. **Configure Settings**:
   - Populate `config.json` with your configuration details, including:
     - `api_key`: Your API key from WeatherAPI
     - `location`: Desired location for weather data
     - `period`: Number of past days to retrieve data for
     - `excel` and `mysql` options for enabling Excel output or MySQL storage
   
2. **Run the Program**:
   ```bash
   python weather_analysis.py
   ```

### Project Structure
- `weather_analysis.py`: Main script for fetching data, prediction, and evaluation.
- `config.json`: Configuration file containing API key, location, and optional settings.
- `weather_data.xlsx`: Excel output file for historical weather data.
- `weather_analysis.log`: Log file for detailed execution tracing and debugging.
- `requirements.txt`: List of dependencies.

### Technical Details
- **Data Sources**: Fetches historical weather data from [WeatherAPI](https://www.weatherapi.com/) and optionally fetches Google’s temperature forecast.
- **Logging**: Logs data retrieval, data insertion into MySQL, prediction processes, and error handling for streamlined debugging.
- **Prediction Algorithm**: Uses linear regression from `scikit-learn` to model temperature trends based on recent historical data.
- **Model Evaluation**: Measures prediction accuracy using Mean Squared Error (MSE) and Mean Absolute Error (MAE) to gauge model performance.
- **Visualization**: Generates visual insights, including temperature and humidity trends, to enhance data exploration.

### License
This project is licensed under the MIT License.
