import pandas as pd
import numpy as np

def create_time_features(df):
    """Creates time-based features from the DataFrame index."""
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['dayofyear'] = df.index.dayofyear
    df['month'] = df.index.month
    return df

# Placeholder for functions from the original uploaded `utils.py`
def get_grid_price(h, grid_prices_config):
    """Looks up TOU price for a given hour (0-23)."""
    h = h % 24
    if 7 <= h < 11 or 17 <= h < 19: return grid_prices_config['mid_peak']
    if 11 <= h < 17: return grid_prices_config['on_peak']
    return grid_prices_config['off_peak']

def load_historical_data(demand_file, pv_file):
    """
    Mocks loading the real data and returning clean dataframes.
    In a complete version, this would contain the actual cleaning logic.
    """
    # Mock data to ensure the structure works
    dates = pd.to_datetime('2024-01-01') + pd.to_timedelta(np.arange(8760), unit='h')
    hist_load = pd.DataFrame({'Load_kW': np.random.rand(8760) * 1000}, index=dates)
    hist_solar = pd.DataFrame({'Solar_kW': np.random.rand(8760) * 500}, index=dates)
    return hist_load, hist_solar