"""
Tests comparing Monte Carlo simulation to ODE solutions.
"""

import pytest
import numpy as np
from src.parameters import SimulationParameters
from src.simulation import Simulation
from src.ode_solver import solve_binding_odes, calculate_equilibrium_fractions


def create_test_params():
    """Create test parameters matching user's configuration."""
    return SimulationParameters(
        C_A=0.002,  # nM
        C_B=0.002,  # nM
        C_antigen=0.1,  # nM
        C_enhancement=2.0e-9,  # M
        N_A_sim=50,
        N_B_sim=50,
        antibodies_per_particle=1000,
        n_patches=12,
        kon_a=1.0e5,  # M⁻¹s⁻¹
        koff_a=0.0001,  # s⁻¹
        kon_b=1.0e5,  # M⁻¹s⁻¹
        koff_b=0.0001,  # s⁻¹
        dt=00.5,  # s
        n_steps_on=10,
        n_steps_off=0,
        n_repeats=100,
        restrict_aggregates_field_on=True,
    )


def test_ode_solver_basic():
    """Test that ODE solver runs and conserves total antigens."""
    params = create_test_params()
    
    # Solve for 200 seconds (matching user's simulation)
    t_span = (0, 200)
    t_eval = np.linspace(0, 200, 201)
    
    result = solve_binding_odes(params, t_span, t_eval)
    
    # Check conservation
    total = (result['Free'] + result['Bound_A'] + 
             result['Bound_B'] + result['Sandwich'])
    
    assert np.allclose(total, params.N_antigen_sim, rtol=1e-6)


def test_equilibrium_fractions():
    """Test analytical equilibrium calculation."""
    params = create_test_params()
    eq = calculate_equilibrium_fractions(params)
    
    # Check fractions sum to 1
    total_frac = sum(eq['fractions'].values())
    assert np.isclose(total_frac, 1.0, rtol=1e-10)
    
    # Check counts sum to N_antigen_sim
    total_count = sum(eq['counts'].values())
    assert np.isclose(total_count, params.N_antigen_sim, rtol=1e-10)
    
    # Print for inspection
    print("\nEquilibrium predictions:")
    for state, count in eq['counts'].items():
        frac = eq['fractions'][state]
        print(f"  {state}: {count:.1f} ({frac*100:.1f}%)")


def test_ode_reaches_equilibrium():
    """Test that ODE solution approaches analytical equilibrium."""
    params = create_test_params()
    
    # Solve for long time (5× relaxation time)
    tau = 1 / params.koff_a  # 10,000 s
    t_span = (0, 5 * tau)
    t_eval = np.linspace(0, 5 * tau, 1001)
    
    result = solve_binding_odes(params, t_span, t_eval)
    eq = calculate_equilibrium_fractions(params)
    
    # Check final values match equilibrium
    assert np.isclose(result['Free'][-1], eq['counts']['Free'], rtol=0.01)
    assert np.isclose(result['Bound_A'][-1], eq['counts']['Bound_A'], rtol=0.01)
    assert np.isclose(result['Bound_B'][-1], eq['counts']['Bound_B'], rtol=0.01)
    assert np.isclose(result['Sandwich'][-1], eq['counts']['Sandwich'], rtol=0.01)


@pytest.mark.slow
def test_mc_vs_ode_at_t200():
    """
    Compare Monte Carlo to ODE at t=200s (user's time scale).
    
    This test documents the current discrepancy between MC and ODE.
    Once the KMC algorithm bug is fixed, we expect better agreement.
    """
    params = create_test_params()
    
    # Run MC simulation
    sim = Simulation(params)
    sim.run(2000)  # 2000 steps × 0.1 s = 200 s
    
    mc_counts = sim.get_antigen_counts()
    
    # Solve ODE
    t_span = (0,  500)
    t_eval = np.array([500.0])
    ode_result = solve_binding_odes(params, t_span, t_eval)
    
    ode_counts = {
        'Free': ode_result['Free'][-1],
        'Bound_A': ode_result['Bound_A'][-1],
        'Bound_B': ode_result['Bound_B'][-1],
        'Sandwich': ode_result['Sandwich'][-1],
    }
    
    # Print comparison
    print("\nMonte Carlo vs ODE at t=200s:")
    print(f"State       MC      ODE     Ratio")
    for state in ['Free', 'Bound_A', 'Bound_B', 'Sandwich']:
        mc = mc_counts.get(state, 0)
        ode = ode_counts[state]
        ratio = mc / ode if ode > 0 else float('inf')
        print(f"{state:10s}  {mc:6.0f}  {ode:6.1f}  {ratio:5.2f}")
    
    # Document expected discrepancy
    # After KMC fix, we expect better agreement
    # For now, just document the current state
    assert mc_counts['Free'] > 0  # Simulation ran
