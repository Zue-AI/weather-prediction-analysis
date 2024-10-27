
# Weather Prediction and Analysis

This project analyzes historical weather data, makes weather predictions, and evaluates accuracy using Google’s temperature forecasts.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)

### Introduction
Using historical weather data, this Python-based project predicts the weather for the next day and evaluates its accuracy against Google forecasts.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Zue-AI/weather-prediction-analysis.git
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

### Usage
1. Populate `config.json` with your configuration details (e.g., API key, location, period).
2. Run the program:
   ```bash
   python weather_analysis.py
   ```

### Project Structure
- `weather_analysis.py`: Main script for data fetching, prediction, and evaluation.
- `config.json`: Configuration file with API key, location, and period.
- `weather_data.xlsx`: Excel output of the historical data.
- `weather_analysis.log`: Log file to trace program execution and debugging.

### Technical Details
- **Data Sources**: Uses [WeatherAPI](https://www.weatherapi.com/) for historical weather data.
- **Logging**: Captures data retrieval, prediction, and error handling.
- **Prediction Algorithm**: Averages recent temperatures for a simple predictive model.
- **Evaluation**: Compares prediction against Google temperature for accuracy.

## License
This project is licensed under the MIT License.
