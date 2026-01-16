"""
Global configuration for the PEMWE stack model, synthetic data generation, and PHM evaluation.

All parameter values and model assumptions follow the associated publication and its
Supplementary Information. Literature references are given inline for traceability.
"""

# ============================================================
# Physical constants
# ============================================================

R = 8.314   # Universal gas constant [J/(mol*K)]
F = 96485  # Faraday constant [C/mol]

# ============================================================
# Water vapour pressure (Antoine equation)
# Source: Sattler et al., 2001
# ============================================================

A_antoine = 8.19625   # Antoine coefficient A
B_antoine = 1730.63   # Antoine coefficient B
C_antoine = 233.426   # Antoine coefficient C

# ============================================================
# Wind turbine / wind-to-power conversion
# ============================================================

velocityMin = 3.0    # Cut-in wind speed [m/s]; generic onshore turbine assumption

# ============================================================
# Stack operating conditions
# Source: Hemauer et al., 2024 (SOA PEMWE stack)
# ============================================================

T = 328.15                 # Operating temperature [K]
pK = 200000               # Cathode pressure [Pa]
pA = 101325               # Anode pressure [Pa]
powerElectrolyzerMax = 700 # Maximum stack power [kW]
N_stack = 4                # Number of stacks [-]

# ============================================================
# Cell and stack parameters
# ============================================================

R_ele = 0.5e-3             # Additional electrical/contact resistance [Ohm*cm^2]
                           # (Bernt & Gasteiger, 2016)

conductivity_membrane = 2.29  # Membrane conductivity pre-factor [S/cm]
                             # (Kopitzke et al., 2000)

E_pro = 7829               # Activation energy for proton transport [J/mol]
                           # (Kopitzke et al., 2000)

alpha_cell = 0.5           # Charge transfer coefficient [-]
                           # (Garcia-Valverde et al., 2012)

i0_ref = 4.1e-6            # Exchange current density at reference conditions [A/cm^2]
                           # (Garcia-Valverde et al., 2012)

E_a = 53990.065           # Activation energy for charge transfer [J/mol]
                           # (Garcia-Valverde et al., 2012)

T_ref = 353.15             # Reference temperature [K]
                           # (Garcia-Valverde et al., 2012)

thicknes_membrane = 0.183   # Membrane thickness [mm] (assumed Nafion 117)

n_cell = 200               # Number of cells per stack [-]
                           # Chosen to reach ~2 A/cm^2 at rated power
                           # (Hemauer et al., 2024)

A_cell = 1000              # Active cell area [cm^2]
                           # (Hemauer et al., 2024)

# ============================================================
# Numerical settings (iterative stack-voltage calculation)
# ============================================================

Delta = 1e-4               # Convergence tolerance for iterative solver [-]
E_Stack_est = 180          # Initial guess for stack voltage [V]

# ============================================================
# Degradation settings
# ============================================================

degradationRate = 10e-6    # Voltage degradation rate [V/h]
                           # Assumed constant; calibrated to match SOA trends
                           # (see Supplementary Information)

# ============================================================
# Noise settings
# ============================================================

errorRel_data = [0.0, 0.01, 0.03]  # Relative measurement noise levels [-]
                                  # Used for synthetic data generation

errorRel_results = 0.03            # Relative noise on model results [-] (preset inputs: 0, 0.01, 0.03)

# ============================================================
# Data sectioning (analysis preprocessing)
# ============================================================

deltaCD = 1.6              # Current density window for sectioning [A/cm^2]
n_min = 300                # Minimum samples per section [-]
n_max = 2000               # Maximum samples per section [-]

# ============================================================
# Parameter fitting settings
# ============================================================

p0 = [0.1, 0.03, 1e-8]     # Initial guesses for parameter fitting
number_B_fits = 4          # Number of fits used for B estimation [-]

# ============================================================
# Time handling
# ============================================================

time_as_hours = True       # Interpret time axis in hours (True) or date time format (False)

# ============================================================
# Time-evolution model types for fitted parameters
# Options: 'linear', 'exponential', 'constant'
# ============================================================

model_type_R = "linear"    # Resistance evolution model
model_type_B = "constant"  # Tafel slope evolution model
model_type_i0 = "constant" # Exchange current density evolution model

# ============================================================
# End-of-life (EOL) definition
# ============================================================

relativeVoltageIncreaseEOL = 0.1  # Relative voltage increase defining EOL
                                 # (common PEMWE lifetime criterion)

# ============================================================
# PHM metric settings
# ============================================================

alpha = 0.08                       # PHM sensitivity / smoothing parameter [-]
alpha_lambda = 0.20                # Lambda weighting factor [-]
beta = 0.95                        # Confidence level for RUL estimation [-]
lambda_p_value = 0.019             # p-value used to compute t_p
computation_frequency = "monthly"  # PHM evaluation interval
num_samples = 1000                 # Monte Carlo sample size [-]
lambda_array = [0.25, 0.5, 0.75]   # Quantiles for uncertainty reporting [-]
