"""
ODE solver for antibody-antigen binding kinetics.

All concentrations in MOLAR (M), all rate constants with proper units.
State variables are molecule COUNTS, not concentrations.
"""

from typing import Tuple
import numpy as np
from scipy.integrate import solve_ivp
from .parameters import SimulationParameters


def solve_binding_odes(
    params: SimulationParameters,
    t_span: Tuple[float, float],
    t_eval: np.ndarray
) -> dict:
    """
    Solve ODEs for independent A/B binding to antigens.
    
    System of ODEs (field OFF, independent binding):
    d[Free]/dt = -kon_a*C_A*[Free] - kon_b*C_B*[Free] 
                 + koff_a*[Bound_A] + koff_b*[Bound_B]
    
    d[Bound_A]/dt = kon_a*C_A*[Free] - koff_a*[Bound_A] 
                    - kon_b*C_B*[Bound_A] + koff_b*[Sandwich]
    
    d[Bound_B]/dt = kon_b*C_B*[Free] - koff_b*[Bound_B] 
                    - kon_a*C_A*[Bound_B] + koff_a*[Sandwich]
    
    d[Sandwich]/dt = kon_a*C_A*[Bound_B] + kon_b*C_B*[Bound_A] 
                     - koff_a*[Sandwich] - koff_b*[Sandwich]
    
    Args:
        params: SimulationParameters with:
            - kon_a, kon_b: M⁻¹s⁻¹ (association rate constants)
            - koff_a, koff_b: s⁻¹ (dissociation rate constants)
            - C_antibody_A, C_antibody_B: M (antibody concentrations)
            - N_antigen_sim: count (number of antigens)
        t_span: (t_start, t_end) in seconds
        t_eval: Array of time points to evaluate (seconds)
    
    Returns:
        Dictionary with:
            - 't': time points (s)
            - 'Free': Free antigen count vs time
            - 'Bound_A': Bound to A count vs time
            - 'Bound_B': Bound to B count vs time
            - 'Sandwich': Sandwich count vs time
            - 'params_used': Dictionary documenting all parameters with units
    
    Units verification:
        kon * C * [count] = (M⁻¹s⁻¹) × (M) × (count) = count/s ✓
        koff * [count] = (s⁻¹) × (count) = count/s ✓
    """
    # Document all parameters with units
    params_used = {
        'kon_a': f"{params.kon_a} M⁻¹s⁻¹",
        'koff_a': f"{params.koff_a} s⁻¹",
        'kon_b': f"{params.kon_b} M⁻¹s⁻¹",
        'koff_b': f"{params.koff_b} s⁻¹",
        'C_antibody_A': f"{params.C_antibody_A} M",
        'C_antibody_B': f"{params.C_antibody_B} M",
        'N_antigen_sim': f"{params.N_antigen_sim} count",
        'K_D_A': f"{params.koff_a/params.kon_a} M",
        'K_D_B': f"{params.koff_b/params.kon_b} M",
    }
    
    # Extract parameters
    kon_a = params.kon_a  # M⁻¹s⁻¹
    koff_a = params.koff_a  # s⁻¹
    kon_b = params.kon_b  # M⁻¹s⁻¹
    koff_b = params.koff_b  # s⁻¹
    C_A = params.C_antibody_A  # M
    C_B = params.C_antibody_B  # M
    
    # Initial conditions: all antigens free
    y0 = [params.N_antigen_sim, 0, 0, 0]  # [Free, Bound_A, Bound_B, Sandwich]
    
    def dydt(t, y):
        """RHS of ODEs. State y = [Free, Bound_A, Bound_B, Sandwich] in counts."""
        Free, Bound_A, Bound_B, Sandwich = y
        
        dFree = (-kon_a * C_A * Free 
                 - kon_b * C_B * Free 
                 + koff_a * Bound_A 
                 + koff_b * Bound_B)
        
        dBound_A = (kon_a * C_A * Free 
                    - koff_a * Bound_A 
                    - kon_b * C_B * Bound_A 
                    + koff_b * Sandwich)
        
        dBound_B = (kon_b * C_B * Free 
                    - koff_b * Bound_B 
                    - kon_a * C_A * Bound_B 
                    + koff_a * Sandwich)
        
        dSandwich = (kon_a * C_A * Bound_B 
                     + kon_b * C_B * Bound_A 
                     - koff_a * Sandwich 
                     - koff_b * Sandwich)
        
        return [dFree, dBound_A, dBound_B, dSandwich]
    
    # Solve ODEs
    sol = solve_ivp(dydt, t_span, y0, t_eval=t_eval, method='LSODA', rtol=1e-10, atol=1e-12)
    
    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")
    
    return {
        't': sol.t,
        'Free': sol.y[0],
        'Bound_A': sol.y[1],
        'Bound_B': sol.y[2],
        'Sandwich': sol.y[3],
        'params_used': params_used,
    }


def calculate_equilibrium_fractions(params: SimulationParameters) -> dict:
    """
    Calculate analytical equilibrium fractions for independent binding.
    
    For independent binding sites with equal affinities:
    K = kon/koff (M⁻¹)
    
    P(Free) = 1 / [(1 + K*C_A) * (1 + K*C_B)]
    P(Bound_A only) = K*C_A / [(1 + K*C_A) * (1 + K*C_B)]
    P(Bound_B only) = K*C_B / [(1 + K*C_A) * (1 + K*C_B)]
    P(Sandwich) = K*C_A * K*C_B / [(1 + K*C_A) * (1 + K*C_B)]
    
    Args:
        params: SimulationParameters
        
    Returns:
        Dictionary with equilibrium fractions and counts
    """
    K_A = params.kon_a / params.koff_a  # M⁻¹
    K_B = params.kon_b / params.koff_b  # M⁻¹
    C_A = params.C_antibody_A  # M
    C_B = params.C_antibody_B  # M
    
    # Binding factors
    factor_A = 1 + K_A * C_A
    factor_B = 1 + K_B * C_B
    Z = factor_A * factor_B  # Partition function
    
    # Fractions
    f_free = 1 / Z
    f_bound_A = (K_A * C_A) / Z
    f_bound_B = (K_B * C_B) / Z
    f_sandwich = (K_A * C_A * K_B * C_B) / Z
    
    # Convert to counts
    N = params.N_antigen_sim
    
    return {
        'fractions': {
            'Free': f_free,
            'Bound_A': f_bound_A,
            'Bound_B': f_bound_B,
            'Sandwich': f_sandwich,
        },
        'counts': {
            'Free': f_free * N,
            'Bound_A': f_bound_A * N,
            'Bound_B': f_bound_B * N,
            'Sandwich': f_sandwich * N,
        },
        'parameters': {
            'K_A': f"{K_A:.2e} M⁻¹",
            'K_B': f"{K_B:.2e} M⁻¹",
            'K_A * C_A': K_A * C_A,
            'K_B * C_B': K_B * C_B,
        }
    }
