"""
Example workflow for the voltage degradation model:
- load a synthetic PEMWE dataset,
- do data processing,
- section the data into quasi-steady polarization-curve segments,
- fit polarization curves with (R, B, i0) per segment,
- do a parameter regression of the time evolution of these parameters,
- evaluate the computed parameters and voltage values against ground truth.

The script is intended as a runnable reference showing how the functions in
`voltage_degradation_model/functions/electrolyzer_model_functions.py` are used together.
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
from scipy.optimize import fsolve
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

from config import *
from voltage_degradation_model.functions.voltage_degradation_model_functions import (
    data_sectioning,
    para_fit_n,
    fit_param,
    mc_EOL_estimate,
    model_funcs,
    voltage_model,
    rmse,
    compute_prognostic_horizon,
    rel_error,
)
from synthetic_data_generation.functions.electrolyzer_model_functions import (
    calc_E_nernst,
    calc_B,
    calc_i0,
    calc_R_cell,
    func_E_stack,
    cell_voltage_function,
)

# ----------------------------------------------------------------
# Load and prepare data
# ----------------------------------------------------------------
path_synthetic_data_pemwe = REPO_ROOT / 'synthetic_data_generation' / 'data' / 'synthetic_data_pemwe.csv'
pemweData = pd.read_csv(path_synthetic_data_pemwe)
pemweData['Time'] = pd.to_datetime(pemweData['Time'])

start_date = pemweData['Time'].min()
end_date = pd.to_datetime('2008-01-01')

# ----------------------------------------------------------------
# Data processing
# ----------------------------------------------------------------

pemweData = pemweData[pemweData[f'Current Density w. Noise_{errorRel_results} [A/cm2]'] >= 0.005]
pemweData = pemweData[pemweData[f'E_cell_deg_noise_{errorRel_results} [V]'] >= 1.25]

# ----------------------------------------------------------------
# Data sectioning
# ----------------------------------------------------------------

arr_help, deltaCD_bool = data_sectioning(pemweData[f'Current Density w. Noise_{errorRel_results} [A/cm2]'], deltaCD, n_min, n_max)
arr = [t for t, m in zip(arr_help, deltaCD_bool) if m]

# ----------------------------------------------------------------
# Polarization curve fit with visualization of fit
# ----------------------------------------------------------------

xdata = pemweData[f'Current Density w. Noise_{errorRel_results} [A/cm2]']
ydata = pemweData[f'E_cell_deg_noise_{errorRel_results} [V]']
fun = lambda currentDensity, R, B, i0: cell_voltage_function(currentDensity, R, B, i0, T, pK, pA)

para = para_fit_n(arr, fun, xdata, ydata, p0, 'linear', f_scale=0.1, N=number_B_fits)

# Create figure
fig, ax = plt.subplots(figsize=(2.17, 1.8), sharex=True)
fig.subplots_adjust(left=0.24, right=0.92, bottom=0.22, top=0.94)

N = 2  # Select last segment

# Plot measured data
ax.plot(pemweData[f'Current Density w. Noise_{errorRel_results} [A/cm2]'][arr[0][0]:arr[0][1]],
        pemweData[f'E_cell_deg_noise_{errorRel_results} [V]'][arr[0][0]:arr[0][1]],
        'o', markersize=2, alpha=0.5, markerfacecolor='none', markeredgecolor='#666666', label=r'Begin of Life',rasterized=True)

ax.plot(pemweData[f'Current Density w. Noise_{errorRel_results} [A/cm2]'][arr[len(arr) - N][0]:arr[len(arr) - N][1]],
        pemweData[f'E_cell_deg_noise_{errorRel_results} [V]'][arr[len(arr) - N][0]:arr[len(arr) - N][1]],
        '^', markersize=2, alpha=0.5, markerfacecolor='none', markeredgecolor='#666666', label=r'End of Test',rasterized=True)

# Plot fitted curves
for idx in [0, len(arr) - N]:
    R_fit, B_fit, i0_fit = para[idx]['parameters']
    x_fit = np.linspace(0.02, 3.5, 100)
    y_fit = cell_voltage_function(x_fit, R_fit, B_fit, i0_fit, T, pK, pA)
    label = "Fitted polarization curve" if idx == 0 else None
    ax.plot(x_fit, y_fit, '-', color="red", linewidth=1, label=label)

# Labels and limits
ax.set_xlabel(r'Current density in A cm$^{-2}$', fontsize=8)
if errorRel_results == 0:
    ax.set_ylabel(r'Cell voltage in V', fontsize=8)
    ax.legend(fontsize=5, loc="best", frameon=True)
else:
    ax.tick_params(axis='y', labelleft=False)
ax.set_xlim(0, 2)
ax.set_xticks(np.arange(0, 2.1, 0.5))
ax.set_ylim(1.4, 2.2)
ax.tick_params(axis='both', which='both', direction='in', labelsize=7)

plt.show()

# Extract time values (mean time per segment)
time_values = [pemweData['Time'][arr[i][0]:arr[i][1]].mean() for i in range(len(arr))]
time_values_h = [pemweData['time [h]'][arr[i][0]:arr[i][1]].mean() for i in range(len(arr))]

time_values_h_duration = [time_values_h[i] - time_values_h[i-1] for i in range(1,len(time_values_h))]

print(f'Data sectioning mean time per segment in h: {np.mean(time_values_h_duration)}')

if isinstance(time_values, pd.Series):
    time_val_np = time_values.to_numpy()
else:
    time_val_np = time_values


# ----------------------------------------------------------------
# Parameter Regression
# ----------------------------------------------------------------

# Build arrays for fitted parameters and their standard errors
para_array = np.array([d['parameters'] for d in para])
std_error_array = np.array([d['std_error'] for d in para])

# Fit the time evolution of each model parameter using the  model types defined in config.py (linear / exponential / constant)
popt_R, pcov_R = fit_param(time_val_np, para_array[:, 0], model_type_R, time_as_hours)
popt_B, pcov_B = fit_param(time_val_np, para_array[:, 1], model_type_B, time_as_hours)
popt_i0, pcov_i0 = fit_param(time_val_np, para_array[:, 2], model_type_i0, time_as_hours)

print("Second-stage fits:")
print("R(t) fit parameters:", popt_R)
print("B(t) fit parameters:", popt_B)
print("i0(t) fit parameters:", popt_i0)

# Visualization of parameter regression over time

# Define time axis for evaluating the fitted regression models
time_end = pemweData['time [h]'].max()
time_calc = np.arange(0, time_end + 1, 1)

# Store regression results in a structured dictionary
fit_params = {
    'R': {'type': model_type_R, 'params': popt_R},
    'B': {'type': model_type_B, 'params': popt_B},
    'i0': {'type': model_type_i0, 'params': popt_i0}
}

# Evaluate regression models for each parameter on the common time grid
R_interp = model_funcs[fit_params['R']['type']](time_calc, *fit_params['R']['params'])
B_interp = model_funcs[fit_params['B']['type']](time_calc, *fit_params['B']['params'])
i0_interp = model_funcs[fit_params['i0']['type']](time_calc, *fit_params['i0']['params'])

# Convert R to mΩ·cm² for readability
R_help = para_array[:, 0]*1e3
i0_help = para_array[:, 2]
para_array_help = np.vstack((R_help, i0_help)).T

fig2, ax2 = plt.subplots(2, 1, figsize=(2.17, 1.8), sharex=True)
fig2.subplots_adjust(left=0.24, right=0.92, bottom=0.22, top=0.94)

param_labels = [
    r"$R$ in m$\Omega$\,cm$^{2}$",
    r"$i_0$ in A\,cm$^{-2}$"
]
ylims = [(0, 300), (1e-9, 1e-5)]
interp_data = [R_interp*1e3, i0_interp]

for i in range(2):
    ax2[i].plot(time_values_h, para_array_help[:, i], '^', markersize=2,
                markerfacecolor='none', markeredgecolor='#666666', alpha=0.6,rasterized=True)
    ax2[i].plot(time_calc, interp_data[i], '-', linewidth=1, color='red', label='Regression')
    if errorRel_results == 0:
        ax2[i].set_ylabel(param_labels[i], fontsize=8)
    else:
        ax2[i].tick_params(axis='y', labelleft=False)
    ax2[i].set_ylim(ylims[i])
    ax2[i].grid(False)
    if i == 1:
        ax2[i].set_yscale('log')
        ax2[i].yaxis.set_major_locator(plt.LogLocator(base=10.0, numticks=10))
        ax2[i].yaxis.set_minor_locator(plt.LogLocator(base=10.0, subs='auto', numticks=10))
        ax2[i].yaxis.set_minor_formatter(plt.NullFormatter())
        ax2[i].tick_params(axis='y', which='minor', length=2,width=0.3)
    ax2[i].tick_params(axis='both', which='both', direction='in', labelsize=7)

ax2[-1].set_xlabel('Time in h', fontsize=8)
ax2[-1].set_xlim(0, time_end+100)

plt.show()

# ----------------------------------------------------------------
# Calculate Ground Truth Parameters and EoL Failure Times
# ----------------------------------------------------------------
# Calculate Ground Truth Parameters
incrementE = 1
max_iterations = 1000
iteration = 0
E_max_stack = calc_E_nernst(T, pK, pA) * n_cell
while abs(incrementE) > Delta and iteration < max_iterations:
    help1 = fsolve(func_E_stack, E_max_stack, args=(powerElectrolyzerMax * 1e3, T, pK, pA))
    incrementE = help1[0] - E_max_stack
    E_max_stack = help1[0]
    iteration += 1
i_max = powerElectrolyzerMax*1e3/(E_max_stack * A_cell)
print(f"i_max={i_max}")

# Calculate EOL operating conditions based on relativeVoltageIncreaseEOL
E_cell_max_EOL = E_max_stack / n_cell * (relativeVoltageIncreaseEOL + 1)
print(f"\nEOL Cell Voltage: {E_cell_max_EOL:.2f} V")

# Calculate ground truth parameters from synthetic data set
R_cell_true = calc_R_cell(T)
R_cell_increase_true = degradationRate / i_max
R_true = [R_cell_true, R_cell_increase_true]
B_true = calc_B(T)
i_0_true = calc_i0(T)

print("\nGround Truth Parameters:")
print("R(t):", R_true)
print("B(t):", B_true)
print("i0(t):", i_0_true)

# ----------------------------------------------------------------
# Evaluate computed parameter and voltage accuracy compared to ground truth values
# ----------------------------------------------------------------

# Calculate RMSE of the parameters
rmse_R = rmse(R_cell_true, popt_R[0])
rmse_R_increase = rmse(R_cell_increase_true, popt_R[1])
rmse_B = rmse(B_true, popt_B[0])
rmse_i0 = rmse(i_0_true, popt_i0[0])

print("\nParameter Fit RMSE:")
print("R: {:.4e}, R': {:.4e}, B: {:.4e}, i0: {:.4e}".format(rmse_R, rmse_R_increase,
                                                            rmse_B, rmse_i0))

# Calculate errors of the calculated voltage with the fit values compared to ground truth
cell_voltage_calc = []

for i in range(len(ydata)):
    volt_calc=voltage_model(pemweData['time [h]'].iloc[i], popt_R, model_type_R, popt_i0, model_type_i0, popt_B, model_type_B, xdata.iloc[i], T, pK, pA)
    cell_voltage_calc.append(volt_calc)

abs_error = np.abs((ydata - cell_voltage_calc)/ydata)
mae = np.mean(abs_error) * 100
rmse = np.sqrt(np.mean((ydata - cell_voltage_calc) ** 2))
mse  = mean_squared_error(ydata, cell_voltage_calc)
r2   = r2_score(ydata, cell_voltage_calc)
print(f"MAE  (mean absolute error): {mae:.4f} %")
print(f"RMSE calculated voltage vs. ground truth values: {rmse:.4f} V")
print(f"MSE  calculated voltage vs. ground truth values: {mse:.6e} V^2")
print(f"R^2  calculated voltage vs. ground truth values: {r2:.4f}")