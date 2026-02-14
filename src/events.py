"""
Event probability calculations for kinetic Monte Carlo simulation.
"""


def calc_bind_probability_field_off(kon: float, C_antibody: float, dt: float) -> float:
    """
    Calculate standard binding probability when field is OFF.
    
    Args:
        kon: Association rate constant (M⁻¹s⁻¹)
        C_antibody: Antibody concentration (M)
        dt: Time step (seconds)
        
    Returns:
        Binding probability: kon × C_antibody × dt
    """
    return kon * C_antibody * dt


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
        Enhanced binding probability: kon × C_enhancement × P_neighbor × dt
    """
    return kon * C_enhancement * P_neighbor * dt


def calc_unbind_probability(koff: float, dt: float) -> float:
    """
    Calculate unbinding probability.
    
    Args:
        koff: Dissociation rate constant (s⁻¹)
        dt: Time step (seconds)
        
    Returns:
        Unbinding probability: koff × dt
    """
    return koff * dt


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
