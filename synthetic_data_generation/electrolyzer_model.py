"""
Generate a synthetic PEMWE operating dataset from wind power input.

The script reads `wind_data.csv`, converts wind power to stack operating points using an
iterative stack-voltage calculation, applies a simple irreversible degradation model, adds
configurable measurement noise, and writes the resulting dataset to `synthetic_data_pemwe.csv`.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from functions.electrolyzer_model_functions import *

REPO_ROOT = Path(__file__).resolve().parents[1]
path_wind_data = REPO_ROOT / 'synthetic_data_generation' / 'data' / 'wind_data.csv'
path_synthetic_data_pemwe = REPO_ROOT / 'synthetic_data_generation' / 'data' / 'synthetic_data_pemwe.csv'

# Load wind input data
wind_data = pd.read_csv(path_wind_data)
synthetic_data = wind_data.copy()
synthetic_data['Time'] = pd.to_datetime(synthetic_data['Time'])

# Define time axis
start_date = synthetic_data['Time'].min()
end_date = pd.to_datetime('2008-01-01')
synthetic_data = synthetic_data[synthetic_data['Time'] <= end_date].copy()
synthetic_data['time [h]'] = (synthetic_data['Time'] - start_date).dt.total_seconds() / 3600

# Pre-compute maximum stack voltage at maximum electrolyzer power (used for saturation)
incrementE = 1
max_iterations = 1000
iteration = 0
E_max = calc_E_nernst(T, pK, pA) * n_cell

while abs(incrementE) > Delta and iteration < max_iterations:
    sol = fsolve(func_E_stack, E_max, args=(powerElectrolyzerMax * 1e3, T, pK, pA))
    incrementE = sol[0] - E_max
    E_max = sol[0]
    iteration += 1

# Convert wind power to PEMWE stack operating points
E_stack = []
powerPEMWE = []

for power in synthetic_data['Power [kW]']:
    if power / 2 > 0 and power / N_stack <= powerElectrolyzerMax:
        incrementE = 1
        iteration = 0
        E_guess = calc_E_nernst(T, pK, pA) * n_cell

        while abs(incrementE) > Delta and iteration < max_iterations:
            sol = fsolve(func_E_stack, E_guess, args=(power * 1e3 / N_stack, T, pK, pA))
            incrementE = sol[0] - E_guess
            E_guess = sol[0]
            iteration += 1

        E_stack.append(E_guess)
        powerPEMWE.append(power / N_stack)

    elif power > powerElectrolyzerMax:
        E_stack.append(E_max)
        powerPEMWE.append(powerElectrolyzerMax)

    else:
        E_stack.append(0)
        powerPEMWE.append(0)

# Add derived PEMWE variables
synthetic_data['Stack Power [kW]'] = powerPEMWE
synthetic_data['E_stack [V]'] = E_stack
synthetic_data['E_cell [V]'] = synthetic_data['E_stack [V]'] / n_cell
synthetic_data['Current [A]'] = synthetic_data['Stack Power [kW]'] * 1e3 / synthetic_data['E_stack [V]']
synthetic_data['Current Density [A/cm2]'] = synthetic_data['Stack Power [kW]'] * 1e3 / (synthetic_data['E_stack [V]'] * A_cell)
synthetic_data['Current Density [A/cm2]'] = synthetic_data['Current Density [A/cm2]'].where(synthetic_data['Power [kW]'] > 0, 0)

# Irreversible degradation model (scaled with current density)
i_max = powerElectrolyzerMax * 1e3 / (E_max * A_cell)
helpDegradation = 0

E_cell_deg = [synthetic_data['E_cell [V]'].iloc[0]]
E_deg = [0]

for i in range(synthetic_data['E_cell [V]'].size - 1):
    timeDiff_h = (synthetic_data['Time'].iloc[i + 1] - synthetic_data['Time'].iloc[i]).total_seconds() / 3600
    helpDegradation += degradationRate * timeDiff_h
    deg_term = helpDegradation * (synthetic_data['Current Density [A/cm2]'].iloc[i + 1] / i_max)

    E_deg.append(deg_term)
    E_cell_deg.append(synthetic_data['E_cell [V]'].iloc[i + 1] + deg_term)

synthetic_data['E_deg [V]'] = E_deg
synthetic_data['E_cell_deg [V]'] = E_cell_deg

# Add measurement noise
for errorRel in errorRel_data:
    rng = np.random.default_rng(seed=42)

    E_max_noise = (E_max / n_cell) * errorRel
    i_max_noise = i_max * errorRel

    E_noise = rng.normal(0, E_max_noise, len(synthetic_data))
    i_noise = rng.normal(0, i_max_noise, len(synthetic_data))

    synthetic_data[f'E_noise_{errorRel} [V]'] = E_noise
    synthetic_data[f'i_noise_{errorRel} [A/cm2]'] = i_noise
    synthetic_data[f'E_cell_deg_noise_{errorRel} [V]'] = synthetic_data['E_cell_deg [V]'] + E_noise
    synthetic_data[f'Current Density w. Noise_{errorRel} [A/cm2]'] = synthetic_data['Current Density [A/cm2]'] + i_noise

# Save output dataset
synthetic_data.to_csv(path_synthetic_data_pemwe, index=False)

columns = [
    ("E_cell_deg_noise_0.01 [V]", "Cell voltage in V"),
    ("Current Density w. Noise_0.01 [A/cm2]", "Current density in A cm$^{-2}$")
]

fig, axes = plt.subplots(2, 1, figsize=(3.47, 2.5), sharex=True, constrained_layout=True)

for ax, (col, ylabel) in zip(axes, columns):
    ax.plot(synthetic_data["time [h]"], synthetic_data[col], linestyle='',marker='^', markersize=1,
            markerfacecolor='none', markeredgecolor='#666666', alpha=0.4,rasterized=True)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(False)
    ax.tick_params(axis='both', which='both', direction='in', labelsize=7)

axes[-1].set_xlabel('Time in h', fontsize=8)
axes[-1].set_xlim(0, synthetic_data["time [h]"].max()+100)

axes[0].set_ylim(0, 2.5)
axes[1].set_ylim(0, 2.1)
axes[1].set_yticks([0,0.5,1,1.5,2])

plt.show()