"""
Functions of the voltage degradation model :
- data sectioning of time-series by current-density variation,
- polarization curve fitting,
- regression of fitted parameters over time (R, B, i0),
- EOL/RUL estimation via root finding and Monte Carlo propagation,
- prognostic metrics (PH, λ-metrics, RA) and plotting utilities.

This file is intended to be imported by example scripts and other modules.
"""
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scipy.optimize import least_squares, brentq, curve_fit
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from synthetic_data_generation.functions.electrolyzer_model_functions import calc_E_nernst, cell_voltage_function
from config import *

#########################################
# Data Sectioning
#########################################

def data_sectioning(currentDensity, deltaCD, n_min, n_max):
    arr = []
    deltaCD_bool = []
    j = 0
    while j < len(currentDensity):
        l = j + n_min
        deltaCD_bool_help = 0
        while l < len(currentDensity) and (l - j) <= n_max:
            delta = np.max(currentDensity[j:l]) - np.min(currentDensity[j:l])
            if delta < deltaCD:
                l += 1
            else:
                deltaCD_bool_help = 1
                break
        arr.append([j, l - 1])
        deltaCD_bool.append(deltaCD_bool_help)
        j = l
    return arr, deltaCD_bool

#########################################
# Polarization Curve Fitting
#########################################

def residuals(p, fun, xdata, ydata):
    return fun(xdata, *p) - ydata

def para_fit_n(arr, fun, xdata, ydata, p0, loss, f_scale, N):
    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)
    B_values = []

    lower_bounds = [-np.inf, -np.inf, 1e-14]
    upper_bounds = [np.inf, np.inf, np.inf]
    for j in range(N):
        seg_x = xdata[arr[j][0]:arr[j][1]]
        seg_y = ydata[arr[j][0]:arr[j][1]]
        res = least_squares(
            residuals, p0,
            args=(fun, seg_x, seg_y),
            loss=loss, max_nfev=10000, ftol=1e-15, xtol=1e-15,
            bounds=(lower_bounds, upper_bounds),
            f_scale=f_scale, method='trf'
        )
        B_values.append(res.x[1])
        p0 = res.x

    B_fixed = np.mean(B_values)
    B_std = np.std(B_values, ddof=1) if N > 1 else 1e-12
    results = []
    lower_bounds_fixed = [-np.inf, 1e-14]
    upper_bounds_fixed = [np.inf, np.inf]

    p0_fixed = [p0[0], p0[2]]

    for j in range(len(arr)):
        seg_x = xdata[arr[j][0]:arr[j][1]]
        seg_y = ydata[arr[j][0]:arr[j][1]]

        def fixed_B_fun(current_density, R, i0):
            return fun(current_density, R, B_fixed, i0)

        def fixed_B_residuals(p, x, y):
            R, i0 = p
            return fixed_B_fun(x, R, i0) - y

        res = least_squares(
            fixed_B_residuals, p0_fixed,
            args=(seg_x, seg_y),
            loss=loss, max_nfev=10000, ftol=1e-15, xtol=1e-15,
            bounds=(lower_bounds_fixed, upper_bounds_fixed),
            f_scale=f_scale, method='trf'
        )

        params = [res.x[0], B_fixed, res.x[1]]

        if hasattr(res.jac, "toarray"):
            J = res.jac.toarray()
        else:
            J = res.jac
        m = len(res.fun)
        n = len(res.x)
        dof = m - n
        if dof <= 0:
            raise ValueError("Not enough degrees of freedom to estimate covariance.")
        s_sq = 2 * res.cost / dof
        JTJ = J.T.dot(J)
        cov_partial = s_sq * np.linalg.solve(JTJ, np.eye(n))
        std_err_partial = np.sqrt(np.diag(cov_partial))

        cov_full = np.zeros((3, 3))
        cov_full[0, 0] = cov_partial[0, 0]
        cov_full[1, 1] = B_std ** 2
        cov_full[2, 2] = cov_partial[1, 1]
        std_err_full = np.array([std_err_partial[0], B_std, std_err_partial[1]])

        result_dict = {
            'parameters': params,
            'covariance': cov_full,
            'std_error': std_err_full,
            'dof': dof,
            'cost': res.cost,
            'res': res
        }
        results.append(result_dict)
        p0_fixed = [res.x[0], res.x[1]]
    return results

#########################################
# Parameter Regression
#########################################

def linear_model(t, a, b):
    return a + b * t

def exp_model(t, a, b):
    return a * np.exp(b * t)

def constant_model(t, a):
    return a * np.ones_like(t)

model_funcs = {
    'linear': linear_model,
    'exponential': exp_model,
    'constant': constant_model
}


def fit_param(time_segments, values, model_type, time_as_hours=False):
    if time_as_hours:
        time_numeric = np.asarray(time_segments)
    else:
        time_segments = np.asarray(time_segments)
        values = np.asarray(values)
        time_numeric = np.array([(ts - time_segments[0]).total_seconds() / 3600.0 for ts in time_segments])

    if model_type == 'linear':
        p0 = [values[0], (values[-1] - values[0]) / (time_numeric[-1] - time_numeric[0])]
    elif model_type == 'exponential':
        p0 = [values[0], -1e-4]
    elif model_type == 'constant':
        p0 = [np.mean(values)]
    else:
        raise ValueError("Unknown model type: " + model_type)

    popt, pcov = curve_fit(model_funcs[model_type], time_numeric, values, p0=p0)

    return popt, pcov

def evaluateModel(model_type, popt, t):
    return model_funcs[model_type](t, *popt)


#########################################
# RUL Estimation
#########################################

def voltage_model(t, popt_R, model_type_R, popt_i0, model_type_i0, popt_B, model_type_B, currentDensity, T, pK, pA):
    E0 = calc_E_nernst(T, pK, pA)
    R_t = evaluateModel(model_type_R, popt_R, t)
    i0_t = evaluateModel(model_type_i0, popt_i0, t)
    B_t = evaluateModel(model_type_B, popt_B, t)
    if i0_t <= 0:
        return np.nan
    return E0 + R_t * currentDensity + B_t * np.log10(currentDensity / i0_t)

def find_EOL(popt_R, model_type_R, popt_i0, model_type_i0, popt_B, model_type_B, currentDensity, E_max, t_min=0, t_max=1e100):
    def f(t):
        val = voltage_model(t, popt_R, model_type_R, popt_i0, model_type_i0, popt_B, model_type_B, currentDensity, T, pK, pA)
        return val - E_max
    try:
        t_fail = brentq(f, t_min, t_max)
    except (ValueError, RuntimeError):
        t_fail = np.nan
    return t_fail

def mc_EOL_estimate(popt_R, pcov_R, model_type_R, popt_i0, pcov_i0, model_type_i0, popt_B, pcov_B, model_type_B, currentDensity, E_max, num_samples=1000):
    EOL_samples = []
    for _ in range(num_samples):
        if model_type_R == 'constant':
            sample_R = np.array([np.random.normal(popt_R[0], np.sqrt(pcov_R[0, 0]))])
        else:
            sample_R = np.random.multivariate_normal(popt_R, pcov_R)
        if model_type_B == 'constant':
            sample_B = np.array([np.random.normal(popt_B[0], np.sqrt(pcov_B[0, 0]))])
        else:
            sample_B = np.random.multivariate_normal(popt_B, pcov_B)
        if model_type_i0 == 'constant':
            sample_i0 = np.array([np.random.normal(popt_i0[0], np.sqrt(pcov_i0[0, 0]))])
        else:
            sample_i0 = np.random.multivariate_normal(popt_i0, pcov_i0)
        t_fail = find_EOL(sample_R, model_type_R, sample_i0, model_type_i0, sample_B, model_type_B, currentDensity, E_max, t_min=0, t_max=1e100)
        EOL_samples.append(t_fail)
    return np.array(EOL_samples)


def rmse(true, pred):
    return np.sqrt(np.mean((true - pred) ** 2))

def rel_error(pred, true):
    return (pred - true) / true * 100

def get_lambda_indexes(time_values, t_pred_end, lambda_array = [0.25,0.5,0.75]):
    time_values = np.asarray(time_values)
    is_datetime = np.issubdtype(time_values.dtype, np.datetime64)

    if is_datetime:
        time_numeric = (time_values - time_values[0]) / np.timedelta64(1, 'h')
        time_numeric = time_numeric.astype(float)
    else:
        time_numeric = time_values - time_values[0]

    lambda_array = np.array(lambda_array)
    lamda_array_time = lambda_array* (t_pred_end - time_numeric[0])

    lambda_indexes = []
    for time in lamda_array_time:
        lambda_indexes.append(np.where(time_numeric >= time)[0][0])

    return lambda_indexes, time_numeric

def get_propagated_indexes(time_values, lambda_p_value, t_pred_end, computation_frequency):
    time_values = np.asarray(time_values)
    is_datetime = np.issubdtype(time_values.dtype, np.datetime64)

    if is_datetime:
        time_numeric = (time_values - time_values[0]) / np.timedelta64(1, 'h')
        time_numeric = time_numeric.astype(float)
    else:
        time_numeric = time_values - time_values[0]

    threshold = lambda_p_value * (time_numeric[-1] - time_numeric[0])
    index_pred_start = np.where(time_numeric >= threshold)[0][0]
    index_pred_end = np.where(time_numeric < t_pred_end)[0][-1]

    offset_hours_map = {
        'hourly': 1,
        'daily': 24,
        'weekly': 24 * 7,
        'monthly': 24 * 30
    }
    offset_hours = offset_hours_map[computation_frequency]

    propagated_indexes = [index_pred_start]
    last_time = time_numeric[index_pred_start]

    for i in range(index_pred_start + 1, index_pred_end):
        if time_numeric[i] >= last_time + offset_hours:
            propagated_indexes.append(i)
            last_time = time_numeric[i]

    return propagated_indexes, time_numeric


def fit_model_at_time(cd, cell_voltage, time_snippet, deltaCD, n_min, n_max, T, pK, pA, p0,
                      number_B_fits, time_as_hours, model_type_R ='linear', model_type_i0='constant'):
    arr_help, deltaCD_bool = data_sectioning(cd, deltaCD, n_min, n_max)
    arr = [t for t, m in zip(arr_help, deltaCD_bool) if m]
    fun = lambda current_density, R, B, i0: cell_voltage_function(current_density, R, B, i0, T, pK, pA)
    para = para_fit_n(arr, fun, cd, cell_voltage, p0, 'linear', f_scale=0.1, N=number_B_fits)

    para_array = np.array([d['parameters'] for d in para])

    if np.issubdtype(time_snippet.dtype, np.datetime64):
        base_time = time_snippet.iloc[0] if isinstance(time_snippet, pd.Series) else time_snippet[0]
        time_in_hours = [(pd.to_timedelta(ts - base_time).total_seconds() / 3600) for ts in time_snippet]
    else:
        time_in_hours = time_snippet

    time_snippet_mean_segment = [
        np.mean(time_in_hours[arr[i][0]:arr[i][1]]) for i in range(len(arr))
    ]

    popt_R, pcov_R = fit_param(
        time_snippet_mean_segment, para_array[:, 0], model_type_R, time_as_hours=time_as_hours
    )
    popt_B = np.array([para_array[0, 1]])
    pcov_B = np.array([[0.0]])
    popt_i0, pcov_i0 = fit_param(
        time_snippet_mean_segment, para_array[:, 2], model_type_i0, time_as_hours=time_as_hours
    )

    return popt_R, pcov_R, popt_B, pcov_B, popt_i0, pcov_i0

def compute_probability_mass(valid_EOL, lower_bound, upper_bound):
    return np.mean((valid_EOL >= lower_bound) & (valid_EOL <= upper_bound))

def process_index(idx, time_val, time_numeric, current_density_data, cell_voltage_data, lower_bound, upper_bound,
                 deltaCD, n_min, n_max, T, pK, pA, p0, number_B_fits, time_as_hours, i_max,
                  E_cell_max_EOL, num_samples, beta, model_type_R='linear', model_type_i0='constant'):
    current_density_snippet = current_density_data[:idx]
    cell_voltage_snippet = cell_voltage_data[:idx]
    time_values_snippet = time_val[:idx]

    popt_R, pcov_R, popt_B, pcov_B, popt_i0, pcov_i0 = fit_model_at_time(
        current_density_snippet, cell_voltage_snippet, time_values_snippet, deltaCD, n_min, n_max, T, pK, pA, p0,
                      number_B_fits, time_as_hours, model_type_R=model_type_R, model_type_i0=model_type_i0)

    EOL_samples = mc_EOL_estimate(popt_R, pcov_R, model_type_R,
                                   popt_i0, pcov_i0, model_type_i0,
                                   popt_B, pcov_B, 'constant',
                                   i_max, E_cell_max_EOL, num_samples=num_samples)

    valid_EOL = EOL_samples[~np.isnan(EOL_samples)]
    if valid_EOL.size == 0:
        mean_EOL = np.nan
        prob_mass = 0
        beta_val = 0
    else:
        mean_EOL = np.mean(valid_EOL)
        prob_mass = compute_probability_mass(valid_EOL, lower_bound, upper_bound)
        beta_val = 1 if prob_mass >= beta else 0

    return {
        'time_index': idx,
        'time_numeric': time_numeric[idx],
        'time_stamp': time_val[idx],
        'EOL_samples': valid_EOL,
        'mean_EOL': mean_EOL,
        'prob_mass': prob_mass,
        'beta_criterion': beta_val,
        'popt_R': popt_R,
        'pcov_R': pcov_R,
        'popt_B': popt_B,
        'pcov_B': pcov_B,
        'popt_i0': popt_i0,
        'pcov_i0': pcov_i0,
    }

# ----------------------------------------------------------------
# Prognostic metrics (PH, λ-metrics, RA)
# ----------------------------------------------------------------

def calculate_convergence_RA(results, t_p, t_EOL_true):
    valid_entries = [r for r in results if
                     not np.isnan(r['mean_EOL']) and 'EOL_samples' in r and len(r['EOL_samples']) > 0]

    if not valid_entries:
        return np.nan, np.nan, np.nan

    t_vals = np.array([r['time_numeric'] for r in valid_entries])
    mean_RUL_preds = np.array([r['mean_EOL'] - r['time_numeric'] for r in valid_entries])
    true_RULs = t_EOL_true - t_vals

    RA = 1 - np.abs(true_RULs - mean_RUL_preds) / true_RULs

    delta_t = t_vals[1:] - t_vals[:-1]
    t_n_plus = t_vals[:1]
    t_n = t_vals[:-1]
    RA_n = RA[:-1]
    x_num = 0.5 * np.sum((t_n ** 2 - t_n_plus ** 2) * RA_n)
    y_num = 0.5 * np.sum(delta_t * RA_n ** 2)
    denom = np.sum(delta_t * RA_n)
    if denom == 0:
        return np.nan, np.nan, np.nan

    x_c = x_num / denom
    y_c = y_num / denom

    C_RA = np.sqrt((x_c-t_p) ** 2 + y_c ** 2)

    return RA, C_RA, x_c, y_c


def compute_prognostic_horizon(time_val, current_density_data, cell_voltage_data, deltaCD, n_min, n_max, T, pK, pA, p0,
                               number_B_fits, time_as_hours, i_max, E_cell_max_EOL, t_EOL_true, lambda_p_value=0.2, computation_frequency='monthly',
                               alpha=0.1, beta=0.95, num_samples=1000, model_type_R='linear',
                               model_type_i0='constant'):
    propagated_indexes, time_numeric = get_propagated_indexes(time_val, lambda_p_value=lambda_p_value, t_pred_end=t_EOL_true, computation_frequency=computation_frequency)
    upper_bound = t_EOL_true * (1 + alpha)
    lower_bound = t_EOL_true * (1 - alpha)

    if isinstance(time_val, pd.Series):
        time_val_np = time_val.to_numpy()
    else:
        time_val_np = time_val
    n_jobs = max(1, os.cpu_count() - 1)
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_index)(
            idx, time_val_np, time_numeric, current_density_data, cell_voltage_data,
            lower_bound, upper_bound, deltaCD, n_min, n_max, T, pK, pA, p0, number_B_fits, time_as_hours, i_max,
                  E_cell_max_EOL, num_samples, beta, model_type_R=model_type_R, model_type_i0=model_type_i0
        ) for idx in propagated_indexes
    )

    beta_criterion = [res['beta_criterion'] for res in results]
    all_EOL_samples = results

    if any(beta_criterion):
        first_valid = next((res for res in results if res['beta_criterion'] == 1), None)
        if first_valid is not None:
            horizon_index = first_valid['time_index']
            PH = t_EOL_true - time_numeric[horizon_index]
            PH_timestamp = pd.to_datetime(first_valid['time_stamp'])
        else:
            horizon_index = None
            PH = None
            PH_timestamp = None
    else:
        horizon_index = None
        PH = None
        PH_timestamp = None

    RA, convergence_RA, x_c, y_c = calculate_convergence_RA(all_EOL_samples, time_numeric[propagated_indexes[0]], t_EOL_true)

    PH_results = {
        'propagated_indexes': propagated_indexes,
        'all_EOL_samples': all_EOL_samples,
        'PH_index': horizon_index,
        'PH_timestamp': PH_timestamp,
        'PH': PH,
        'beta-criterion': beta_criterion,
        'RA': RA,
        'x_c': x_c,
        'y_c': y_c,
        'convergence_RA': convergence_RA
    }
    return PH_results

def compute_lambda_metrics(time_val, current_density_data, cell_voltage_data, deltaCD, n_min, n_max, T, pK, pA, p0,
                               number_B_fits, time_as_hours, i_max, E_cell_max_EOL, t_EOL_true, lambda_array=[0.25, 0.5, 0.75],
                               alpha=0.1, beta=0.95, num_samples=1000, model_type_R='linear',
                               model_type_i0='constant'):
    lambda_indexes, time_numeric = get_lambda_indexes(time_val, t_pred_end=t_EOL_true, lambda_array=lambda_array)
    if isinstance(time_val, pd.Series):
        time_val_np = time_val.to_numpy()
    else:
        time_val_np = time_val

    results = []
    bounds = []
    for lambda_idx in lambda_indexes:
        upper_bound = t_EOL_true + (t_EOL_true - time_numeric[lambda_idx]) * alpha
        lower_bound = t_EOL_true - (t_EOL_true - time_numeric[lambda_idx]) * alpha

        result = process_index(lambda_idx, time_val_np, time_numeric, current_density_data, cell_voltage_data,
            lower_bound, upper_bound, deltaCD, n_min, n_max, T, pK, pA, p0, number_B_fits, time_as_hours, i_max,
                  E_cell_max_EOL, num_samples, beta, model_type_R=model_type_R, model_type_i0=model_type_i0)
        results.append(result)
        bounds.append([lower_bound, upper_bound])

    beta_criterion = [res['beta_criterion'] for res in results]

    RA, convergence_RA, x_c, y_c = calculate_convergence_RA(results,time_numeric[lambda_indexes[0]], t_EOL_true)

    lambda_results = {
        'all_results': results,
        'bounds': bounds,
        'beta_criterion': beta_criterion,
        'RA': RA,
        'x_c': x_c,
        'y_c': y_c,
        'convergence_RA': convergence_RA
    }
    return lambda_results

# ----------------------------------------------------------------
# Plotting helpers
# ----------------------------------------------------------------

def plot_prognostic_horizon(PH_results, t_EOL_true, alpha=0.1, hide_y_axis = False,
                            show_legend = True):
    all_results = PH_results.get('all_EOL_samples', [])
    if not all_results:
        print("No EOL samples to plot.")
        return

    x_max = (1 + alpha) * t_EOL_true
    fig, ax = plt.subplots(figsize=(2.17, 1.8), sharex=True)
    fig.subplots_adjust(left=0.25, right=0.92, bottom=0.22, top=0.94)
    t_lin = np.linspace(0, x_max, 200)
    gt_rul = t_EOL_true - t_lin
    ax.plot(t_lin, gt_rul, 'k--', label='Ground Truth RUL', linewidth=1)

    lower_rul = (1 - alpha) * t_EOL_true - t_lin
    upper_rul = (1 + alpha) * t_EOL_true - t_lin
    ax.fill_between(t_lin, lower_rul, upper_rul, color='gray', alpha=0.2, label=r'$\alpha$-bounds')

    times_for_line = []
    means_for_line = []
    for i, res in enumerate(all_results):
        t_idx = res['time_numeric']
        eol_samples = res['EOL_samples']
        rul_samples = eol_samples - t_idx
        if len(rul_samples) == 0 or np.isnan(rul_samples).all():
            continue
        mean_rul = np.nanmean(rul_samples)
        low_ci, high_ci = np.percentile(rul_samples, [5, 95])
        lower_err = max(mean_rul - low_ci, 0)
        upper_err = max(high_ci - mean_rul, 0)
        ax.errorbar(
            t_idx, mean_rul,
            yerr=[[lower_err], [upper_err]],
            fmt='o',
            markersize=3,
            markerfacecolor='red',
            markeredgecolor='none',
            linewidth=0.5,
            ecolor='black',
            elinewidth=0.5,
            capsize=1,
            alpha=1,
        )
        times_for_line.append(t_idx)
        means_for_line.append(mean_rul)

    if mean_rul < low_ci or mean_rul > high_ci:
        print(f"Warning: mean RUL outside CI bounds at t = {t_idx:.1f}")

    ax.plot(times_for_line, means_for_line, 'r-', label='Predicted RUL',linewidth=1)

    ph_idx = PH_results.get('PH_index', None)
    if ph_idx is not None:
        ph_record = next((r for r in all_results if r['time_index'] == ph_idx), None)
        if ph_record is not None:
            horizon_time = ph_record['time_numeric']
            ax.axvline(horizon_time, color='red', linestyle='--', label='PH',linewidth=1)

    ax.set_xlim(0, 20000)
    ax.set_ylim(bottom=0, top=80000)
    if hide_y_axis:
        ax.tick_params(axis='y', labelleft=False)
    else:
        ax.set_ylabel('RUL in h', fontsize=8)
    ax.set_xlabel('Time in h', fontsize=8)
    ax.tick_params(axis='both', which='both', direction='in', labelsize=7)
    if show_legend:
        ax.legend(loc='best',fontsize=5)
    plt.show()

def plot_relative_accuracy(PH_results, hide_y_axis = False, show_legend = True):
    all_results = PH_results.get('all_EOL_samples', [])
    RA = PH_results.get('RA', [])

    if not all_results or len(RA) == 0:
        print("No RA values or EOL samples to plot.")
        return

    fig, ax = plt.subplots(figsize=(2.17, 1.8), sharex=True)
    fig.subplots_adjust(left=0.25, right=0.92, bottom=0.22, top=0.94)
    t_vals = [res['time_numeric'] for res in all_results]

    ax.plot(t_vals, RA, color='red', label='RA', linewidth=0.5, marker= 'o', markersize=3,
            markerfacecolor='red',markeredgecolor='none')
    ax.axhline(1, color='black', linestyle='--', linewidth=1, label='RA = 1')

    ax.set_xlim(0, 20000)
    ax.set_ylim(bottom=-3, top=1.1)
    ax.set_yticks([-3,-2, -1, 0, 1])
    if hide_y_axis:
        ax.tick_params(axis='y', labelleft=False)
    else:
        ax.set_ylabel('RA', fontsize=8)
    ax.set_xlabel('Time in h', fontsize=8)
    ax.tick_params(axis='both', which='both', direction='in', labelsize=7)
    if show_legend:
        ax.legend(loc='best',fontsize=5)
    plt.show()