"""
Event probability calculations for kinetic Monte Carlo simulation.
"""

import warnings


def calc_bind_probability_field_off(kon: float, C_antibody: float, dt: float) -> float:
    """
    Calculate standard binding probability when field is OFF.
    
    Args:
        kon: Association rate constant (M⁻¹s⁻¹)
        C_antibody: Antibody concentration (M)
        dt: Time step (seconds)
        
    Returns:
        Binding probability: kon × C_antibody × dt, capped at 1.0
        
    Raises:
        ValueError: If inputs are negative
        
    Warnings:
        Issues a warning if the first-order approximation breaks down (prob > 0.1)
    """
    if kon < 0 or C_antibody < 0 or dt < 0:
        raise ValueError(f"Negative parameters not allowed: kon={kon}, C_antibody={C_antibody}, dt={dt}")
    
    prob = kon * C_antibody * dt
    
    if prob > 1.0:
        warnings.warn(
            f"Binding probability exceeds 1.0 (kon * C_antibody * dt = {prob:.3f}). "
            f"Time step is too large for the given rate constants. Capping at 1.0.",
            RuntimeWarning
        )
        return 1.0
    
    if prob > 0.1:
        warnings.warn(
            f"Binding probability {prob:.3f} > 0.1. First-order approximation may be inaccurate. "
            f"Consider reducing the time step.",
            RuntimeWarning
        )
    
    return prob


def calc_bind_probability_field_on_enhanced(
    kon: float, C_enhancement: float, P_neighbor: float, dt: float
) -> float:
    """
    Calculate enhanced binding probability when field is ON.
    
    Args:
        kon: Association rate constant (M⁻¹s⁻¹)
        C_enhancement: Enhanced concentration (M)
        P_neighbor: Probability that neighbor is of given type
        dt: Time step (seconds)
        
    Returns:
        Enhanced binding probability: kon × C_enhancement × P_neighbor × dt, capped at 1.0
        
    Raises:
        ValueError: If inputs are negative
        
    Warnings:
        Issues a warning if the first-order approximation breaks down (prob > 0.1)
    """
    if kon < 0 or C_enhancement < 0 or P_neighbor < 0 or dt < 0:
        raise ValueError(
            f"Negative parameters not allowed: kon={kon}, C_enhancement={C_enhancement}, "
            f"P_neighbor={P_neighbor}, dt={dt}"
        )
    
    if P_neighbor > 1.0:
        raise ValueError(f"P_neighbor must be <= 1.0, got {P_neighbor}")
    
    prob = kon * C_enhancement * P_neighbor * dt
    
    if prob > 1.0:
        warnings.warn(
            f"Enhanced binding probability exceeds 1.0 (kon * C_enhancement * P_neighbor * dt = {prob:.3f}). "
            f"Time step is too large for the given rate constants. Capping at 1.0.",
            RuntimeWarning
        )
        return 1.0
    
    if prob > 0.1:
        warnings.warn(
            f"Enhanced binding probability {prob:.3f} > 0.1. First-order approximation may be inaccurate. "
            f"Consider reducing the time step.",
            RuntimeWarning
        )
    
    return prob


def calc_unbind_probability(koff: float, dt: float) -> float:
    """
    Calculate unbinding probability.
    
    Args:
        koff: Dissociation rate constant (s⁻¹)
        dt: Time step (seconds)
        
    Returns:
        Unbinding probability: koff × dt, capped at 1.0
        
    Raises:
        ValueError: If inputs are negative or probability exceeds 1.0
    """
    if koff < 0 or dt < 0:
        raise ValueError(f"Negative parameters not allowed: koff={koff}, dt={dt}")
    
    prob = koff * dt
    
    if prob > 1.0:
        raise ValueError(
            f"Unbinding probability exceeds 1.0 (koff * dt = {prob:.3f}). "
            f"The time step is too large for the given dissociation rate constant."
        )
    
    return prob


def calc_neighbor_probability(N_type: int, N_total: int) -> float:
    """
    Calculate probability that a neighbor is of a given type.
    
    Args:
        N_type: Number of particles of the given type
        N_total: Total number of particles
        
    Returns:
        Probability: N_type / N_total
    """
    if N_total == 0:
        return 0.0
    return N_type / N_total
