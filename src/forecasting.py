import pandas as pd
import numpy as np
import xgboost as xgb
import glob
import os
from src.utils import create_time_features

class WeatherForecaster:
    def __init__(self, weather_folder):
        self.weather_folder = weather_folder
        self.model_load = None
        self.model_solar = None
        
    def load_and_process_weather(self):
        """Ingests and cleans raw weather data."""
        # ... (Implementation is as detailed in the previous step) ...
        # (For this package, we'll keep the actual implementation simple for demonstration)
        
        # Mocking the result of this function for demo purposes
        dates = pd.to_datetime('2024-01-01') + pd.to_timedelta(np.arange(8760), unit='h')
        weather_df = pd.DataFrame({
            'temp_c': np.random.uniform(5, 30, 8760),
            'humidity': np.random.uniform(40, 90, 8760),
            'cloudiness': np.random.rand(8760)
        }, index=dates)
        return weather_df.resample('h').mean().ffill()

    def train_models(self, historical_load_df, historical_solar_df):
        """Trains XGBoost models on merged, weather-informed data."""
        
        weather_df = self.load_and_process_weather()
        
        # Merge, clean, and create features
        df = historical_load_df.join(historical_solar_df).join(weather_df).dropna()
        df = create_time_features(df)
        
        # --- Load Model Training ---
        LOAD_FEATURES = ['hour', 'dayofweek', 'temp_c', 'humidity']
        X_load = df[LOAD_FEATURES]
        y_load = df['Load_kW']
        self.model_load = xgb.XGBRegressor(n_estimators=100, learning_rate=0.01)
        self.model_load.fit(X_load, y_load)
        
        # --- Solar Model Training ---
        SOLAR_FEATURES = ['hour', 'dayofyear', 'cloudiness']
        X_solar = df[SOLAR_FEATURES]
        y_solar = df['Solar_kW']
        self.model_solar = xgb.XGBRegressor(n_estimators=100, learning_rate=0.01)
        self.model_solar.fit(X_solar, y_solar)

    def predict_next_24h(self, current_weather_forecast):
        """Generates load and solar forecast vectors based on weather."""
        
        # Mocking the prediction result for the run_simulation demo
        T = len(current_weather_forecast)
        pred_load = (4000 + 200 * current_weather_forecast['temp_c']).to_numpy()
        pred_solar = (2000 * (1 - current_weather_forecast['cloudiness'])).clip(0).to_numpy()
        
        return pred_load, pred_solar