import json
import pandas as pd
import numpy as np
import os
from src.engine import MicrogridOptimizer
from src.ici_manager import ICIManager
from src.forecasting import WeatherForecaster

def mock_ieso_demand(t):
    # Simulates a system-wide Ontario peak during a heatwave (hour 12-18)
    base = 15000 + 5000 * np.sin((t % 24 - 8) / 4)
    if 12 <= t % 24 <= 18:
        base += np.random.normal(7000, 1000) # Big afternoon spike
    return max(0, base + np.random.normal(0, 500))

def run():
    print("🚀 Initializing Ontario Grid AI Commander...")
    config_path = 'config.json'
    with open(config_path) as f: config = json.load(f)
    
    # 1. Setup Engines
    optimizer = MicrogridOptimizer(config)
    ici_manager = ICIManager(config['optimization']['historical_peak_mw'])
    forecaster = WeatherForecaster(config['forecasting']['weather_folder'])
    
    # 2. Load and Train (Mocking data for demonstration)
    # In a real setup, load your demand/PV CSVs here and run:
    # hist_load, hist_solar = load_historical_data() 
    # forecaster.train_models(hist_load, hist_solar) 
    
    # --- DEMO SETUP: Create T=168 hours of simulated data ---
    T = 168 # Simulate one full week (168 hours)
    current_soc = config['current_specs']['battery_capacity_kwh'] * 0.5
    results_history = []
    
    # Mock Weather Forecast: A hot week
    mock_weather_data = pd.DataFrame({
        'temp_c': 20 + 10 * np.sin(np.linspace(0, 2 * np.pi * 7, T)) + np.linspace(0, 5, T),
        'humidity': 50 + 20 * np.sin(np.linspace(0, 2 * np.pi * 7, T)),
        'cloudiness': np.abs(np.sin(np.linspace(0, 2 * np.pi * 7, T/24))) # Day/Night cycle
    }, index=pd.to_datetime('2025-07-01 00:00:00') + pd.to_timedelta(np.arange(T), unit='h'))
    
    # 3. The MPC Loop (Receding Horizon Control)
    for t in range(T):
        
        # A. Current Hour Observation
        hour_of_day = t % 24
        
        # B. Get Forecast for next 24h (Simulates calling the XGBoost model)
        horizon_hours = 24
        weather_slice_24h = mock_weather_data.iloc[t : t + horizon_hours]
        
        # Use mock data in place of actual trained model predictions for this demo
        pred_load = (4000 + 2000 * np.sin((weather_slice_24h.index.hour - 6) / 4) + 200 * weather_slice_24h['temp_c']).to_numpy()
        pred_solar = (2000 * (1 - weather_slice_24h['cloudiness'])).clip(0).to_numpy()
        
        # C. Check for ICI Peak (The decision-maker override)
        ieso_demand_forecast = mock_ieso_demand(t)
        is_ici_peak = ici_manager.check_trigger(ieso_demand_forecast)
        
        # D. Optimize (The Brain)
        df_res, _, next_soc_opt = optimizer.solve(
            pred_load, pred_solar, current_soc, 
            horizon_hours=horizon_hours, 
            force_discharge=is_ici_peak
        )
        
        # E. Execute ONLY the first hour
        action = df_res.iloc[0].to_dict()
        action['Hour_Index'] = t
        action['is_ici_event'] = is_ici_peak
        action['System_Demand_MW'] = ieso_demand_forecast
        
        # Update State (The 'Real' world physics update)
        current_soc = next_soc_opt # Assuming the battery accurately tracks the model
        
        results_history.append(action)

    pd.DataFrame(results_history).to_csv("simulation_results.csv", index=False)
    print(f"✅ Simulation Complete. {T} hours processed. Data saved to 'simulation_results.csv'")

if __name__ == "__main__":
    run()