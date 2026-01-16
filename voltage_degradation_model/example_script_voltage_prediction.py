"""
Example script demonstrating the full prognostics workflow for a PEM water
electrolyzer using synthetic data. The script

- loads a synthetic PEMWE dataset,
- computes the ground-truth end-of-life (EOL) based on the underlying
  degradation model,
- estimates the prognostic horizon (PH) using repeated model re-identification
  and Monte Carlo uncertainty propagation,
- evaluates λ-metrics and relative accuracy (RA),
- and visualizes PH and RA over time.

Results are cached based on a hash of the configuration parameters to avoid
recomputing expensive Monte Carlo simulations.
"""


from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib as mpl
from scipy.optimize import fsolve
import time
import pandas as pd
import os
import pickle
import hashlib
import json
from config import*

from voltage_degradation_model.functions.voltage_degradation_model_functions import (
    compute_prognostic_horizon,
    compute_lambda_metrics,
    plot_prognostic_horizon,
    plot_relative_accuracy
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
# Create a hash of all relevant configuration parameters
# ----------------------------------------------------------------
config_params = {
    # Parameters Virtual Data
    "degradationRate": degradationRate,
    "errorRel": errorRel_results,
    # Fitting Parameters
    "deltaCD": deltaCD,
    "n_min": n_min,
    "n_max": n_max,
    "p0": p0,
    "number_B_fits": number_B_fits,
    "T": T,
    "pK": pK,
    "pA": pA,
    "model_type_R": str(model_type_R),
    "model_type_B": str(model_type_B),
    "model_type_i0": str(model_type_i0),
    # Parameters directly passed to computePH:
    "relativeVoltageIncreaseEOL": relativeVoltageIncreaseEOL,
    "lambda_value": lambda_p_value,
    "computation_frequency": str(computation_frequency),
    "alpha": alpha,
    "alpha_lambda": alpha_lambda,
    "beta": beta,
    "num_samples": num_samples,
    "lambda_array": lambda_array
}
config_hash = hashlib.md5(json.dumps(config_params, sort_keys=True).encode()).hexdigest()

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
# Calculate Ground Truth Parameters and EoL Failure Times
# ----------------------------------------------------------------
# Calculate Ground Truth Parameters
incrementE = 1
E_max_stack = calc_E_nernst(T, pK, pA) * n_cell

# Iterative solution for maximum stack voltage
while abs(incrementE) > Delta:
    help1 = fsolve(func_E_stack, E_max_stack, args=(powerElectrolyzerMax * 1e3, T, pK, pA))
    incrementE = help1[0] - E_max_stack
    E_max_stack = help1[0]

# Maximum operating current density
i_max = powerElectrolyzerMax * 1e3 / (E_max_stack * A_cell)

# EOL voltage criterion
E_cell_max_EOL = E_max_stack / n_cell * (relativeVoltageIncreaseEOL + 1)
print(f"\nEOL Cell Voltage: {E_cell_max_EOL:.2f} V")

# Ground-truth degradation parameters
R_cell_true = calc_R_cell(T)
R_cell_increase_true = degradationRate / i_max
R_true = [R_cell_true, R_cell_increase_true]
B_true = calc_B(T)
i_0_true = calc_i0(T)

# Solve for ground-truth EOL time
def f_true(t):
    return cell_voltage_function(
        i_max, R_true[0] + R_true[1] * t, B_true, i_0_true, T, pK, pA
    ) - E_cell_max_EOL

t_EOL_true = fsolve(f_true, 10000)[0]
ground_truth_EOL_timestamp = start_date + pd.Timedelta(hours=t_EOL_true)

print(f"\nGround Truth EOL (numeric): {t_EOL_true:.2f} hours")
print(f"Ground Truth EOL (timestamp): {ground_truth_EOL_timestamp}")

# ----------------------------------------------------------------
# Load cached results or compute prognostic metrics
# ----------------------------------------------------------------

results_filename = f"ph_results_Deg_Noise_{config_hash}.pkl"
start_time = time.time()

if os.path.exists(results_filename):
    # Load previously computed results
    with open(results_filename, "rb") as f:
        results_combined = pickle.load(f)

    PH_results = results_combined["PH_results"]
    lambda_results = results_combined["lambda_results"]

    print("Loaded cached PH_results and lambda_results.")
else:
    # Compute prognostic horizon (PH)
    PH_results = compute_prognostic_horizon(
        pemweData['Time'],
        pemweData[f'Current Density w. Noise_{errorRel_results} [A/cm2]'],
        pemweData[f'E_cell_deg_noise_{errorRel_results} [V]'],
        deltaCD, n_min, n_max,
        T, pK, pA,
        p0, number_B_fits,
        time_as_hours,
        i_max, E_cell_max_EOL, t_EOL_true,
        lambda_p_value=lambda_p_value,
        computation_frequency=computation_frequency,
        alpha=alpha,
        beta=beta,
        num_samples=num_samples,
        model_type_R=model_type_R,
        model_type_i0=model_type_i0
    )
    # Compute λ-metrics at predefined fractions of lifetime
    lambda_results = compute_lambda_metrics(
        pemweData['Time'],
        pemweData[f'Current Density w. Noise_{errorRel_results} [A/cm2]'],
        pemweData[f'E_cell_deg_noise_{errorRel_results} [V]'],
        deltaCD, n_min, n_max,
        T, pK, pA,
        p0, number_B_fits,
        time_as_hours,
        i_max, E_cell_max_EOL, t_EOL_true,
        lambda_array=lambda_array,
        alpha=alpha_lambda,
        beta=beta,
        num_samples=num_samples,
        model_type_R=model_type_R,
        model_type_i0=model_type_i0
    )
    results_combined = {
        "PH_results": PH_results,
        "lambda_results": lambda_results
    }
    # Cache results
    with open(results_filename, "wb") as f:
        pickle.dump(results_combined, f)
    print(f"\nComputed and saved new results.")

# Report execution time and key results
end_time = time.time()
execution_time = end_time - start_time
minutes, seconds = divmod(execution_time, 60)
print(f"\ncomputePH execution time: {int(minutes)} min {seconds:.1f} sec")

print(f"\nPrognostic Horizon (PH): {PH_results['PH']} hours")
print(f"\nPrognostic Horizon (PH) Time Stamp: {PH_results['PH_timestamp']}")
print(PH_results['beta-criterion'])
print(f"\nConvergence RA: {PH_results['convergence_RA']}")
print(f"\nRA:")
print(PH_results['RA'])

print(f'\nTime bounds: {lambda_results['bounds']}')
print(f"\nalpha-lambda criterion {lambda_array}: {lambda_results['beta_criterion']}")
mean_eol_values = [entry['mean_EOL'] for entry in lambda_results['all_results']]
print(f"\nEOL-values {lambda_array}: {mean_eol_values}")
prob_mass = [entry['prob_mass'] for entry in lambda_results['all_results']]
print(f"\nProbability mass {lambda_array}: {prob_mass}")
print(f"\nRA {lambda_array}: {lambda_results['RA']}")


# ----------------------------------------------------------------
# Visualization of PH and RA
# ----------------------------------------------------------------

plot_name_PH = f"ph_plot_virtual_data_{errorRel_results}.pdf"
plot_name_RA = f"ra_plot_virtual_data_{errorRel_results}.pdf"

hide_y_axis=False
show_legend=True

plot_prognostic_horizon(PH_results, t_EOL_true, alpha=alpha,save_fig=plot_name_PH, hide_y_axis=hide_y_axis,show_legend=show_legend)
plot_relative_accuracy(PH_results, save_fig=plot_name_RA, hide_y_axis=hide_y_axis, show_legend=show_legend)
