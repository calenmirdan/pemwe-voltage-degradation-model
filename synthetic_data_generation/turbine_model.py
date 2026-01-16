"""
---------------------------------------------------------------------------------------------------------------
Script to generate `wind_data.csv` from raw 10-minute wind-speed measurements and a turbine power curve.

It loads a raw DWD-style wind file, filters a fixed date range, interpolates turbine power for each
wind speed sample, applies a minimum wind-speed threshold (velocityMin from `config.py`), and saves the
resulting time series (Time, Velocity [m/s], Power [kW]) for use in the synthetic data generation workflow.
---------------------------------------------------------------------------------------------------------------
"""
import numpy as np
import pandas as pd
from pathlib import Path
from config import *


# File paths
REPO_ROOT = Path(__file__).resolve().parents[1]
rawData = REPO_ROOT / 'synthetic_data_generation' / 'data' / 'produkt_zehn_min_ff_20000101_20081031_02522.txt'
wind_data = REPO_ROOT / 'synthetic_data_generation' / 'data' / 'wind_data.csv'
windTurbinePowerCurveData = REPO_ROOT / 'synthetic_data_generation' / 'data' /'wind_turbine_power_curve.csv'

# Load wind data
windRaw = pd.read_csv(rawData, sep=';')
windRaw.columns = ['STATIONS_ID', 'MESS_DATUM', 'QN', 'FF_10', 'DD_10', 'eor']
windRaw['MESS_DATUM'] = pd.to_datetime(windRaw['MESS_DATUM'], format='%Y%m%d%H%M', errors='coerce')
windRaw = windRaw[windRaw["FF_10"] > 0]  # Remove invalid wind speed data

# Filter by date range
start_date = pd.to_datetime('2005-01-01')
end_date = pd.to_datetime('2007-12-31')
filteredWind = windRaw[(windRaw['MESS_DATUM'] >= start_date) & (windRaw['MESS_DATUM'] <= end_date)]

# Select and rename relevant columns
wind = filteredWind[['MESS_DATUM', 'FF_10']].copy()
wind.rename(columns={'MESS_DATUM': "Time", 'FF_10': "Velocity [m/s]"}, inplace=True)

# Save processed wind data
wind.to_csv(wind_data, index=False)

# Load wind turbine power curve data
windTurbinePowerCurve = pd.read_csv(windTurbinePowerCurveData,sep=';')
windTurbinePowerCurve.columns = ['Velocity', 'Power']

# Interpolate power values for each wind velocity
wind['Power [kW]'] = np.interp(
    wind['Velocity [m/s]'],
    windTurbinePowerCurve['Velocity'],
    windTurbinePowerCurve['Power']
)

wind.loc[wind["Velocity [m/s]"] < velocityMin, "Power [kW]"] = 0

# Save the updated wind data with interpolated power values
wind.to_csv(wind_data, index=False)