"""
Electrochemical functions for computing reversible and over potentials
and cell/stack voltage based on a polarization model.

"""
import math
import numpy as np
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from config import *

def antoine(T):
    """
    Water vapour pressure using the Antoine equation (Sattler et al., 2001).

    Parameters
    ----------
    T : float
        Temperature [K].

    Returns
    -------
    float
        Water vapour pressure [mbar].
    """
    return 10 ** (A_antoine - (B_antoine / ((T - 273.15) + C_antoine)))


def calc_E_rev(T):
    """
    Temperature-dependent standard reversible potential (Bratsch et al., 1989).

    Parameters
    ----------
    T : float
        Temperature [K].

    Returns
    -------
    float
        Standard reversible potential [V].
    """
    return 1.2291 + (T - 298.15) * (-8.456e-4)


def calc_E_nernst(T, pK, pA):
    """
    Nernst potential assuming ideal gases and liquids.

    Parameters
    ----------
    T : float
        Temperature [K].
    pK : float
        Cathode pressure [Pa].
    pA : float
        Anode pressure [Pa].

    Returns
    -------
    float
        Nernst potential [V].
    """
    pH2O = antoine(T) * 100  # mbar -> Pa
    E_rev = calc_E_rev(T)
    X = (pK - pH2O) / 101325 * ((pA - pH2O) / 101325) ** 0.5
    return E_rev + (R * T / (2 * F)) * math.log(X)


def calc_R_cell(T):
    """
    Temperature-dependent ohmic resistance (membrane + electrical).

    Parameters
    ----------
    T : float
        Temperature [K].

    Returns
    -------
    float
        Cell resistance [Ohm·cm²].
    """
    conductivity_T = conductivity_membrane * math.exp(-E_pro / (R * T))
    R_ohm = thicknes_membrane * 1e-1 / conductivity_T
    #R_ohm = thicknes_membrane / conductivity_membrane
    return R_ohm + R_ele


def calc_B(T):
    """
    Tafel slope coefficient.

    Parameters
    ----------
    T : float
        Temperature [K].

    Returns
    -------
    float
        Tafel coefficient B [V].
    """
    return 2.3026 * R * T / (2 * alpha_cell * F)


def calc_i0(T):
    """
    Exchange current density with Arrhenius temperature dependence.

    Parameters
    ----------
    T : float
        Temperature [K].

    Returns
    -------
    float
        Exchange current density i0 [A/cm²].
    """
    return i0_ref * math.exp(E_a / R * (1 / T_ref - 1 / T))


def cell_voltage_function(current_density, R, B, i0, T, pK, pA):
    """
    Cell voltage from Nernst, ohmic, and activation contributions.

    Parameters
    ----------
    current_density : float
        Current density [A/cm²].

    Returns
    -------
    float
        Cell voltage [V].
    """
    E0 = calc_E_nernst(T, pK, pA)
    return E0 + R * current_density + B * np.log10(current_density / i0)


def func_E_stack(E_stack, P_stack, T, pK, pA):
    """
    Residual function for iterative stack voltage calculation.

    Parameters
    ----------
    E_stack : float
        Stack voltage [V].
    P_stack : float
        Stack power [W].

    Returns
    -------
    float
        Residual of the stack voltage equation.
    """
    E_nernst = calc_E_nernst(T, pK, pA)
    R_cell = calc_R_cell(T)
    B = calc_B(T)
    i_0 = calc_i0(T)

    return (
        E_nernst
        + R_cell * (P_stack / (E_stack * A_cell))
        + B * np.log10(P_stack / (E_stack * A_cell * i_0))
        - E_stack/n_cell
    )
