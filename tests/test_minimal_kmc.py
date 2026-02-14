"""
Tests for minimal_kmc module.
"""

import pytest
import numpy as np
from src.minimal_kmc import (
    MinimalAntigen,
    run_minimal_kmc,
    run_multiple_replicates,
    calculate_statistics,
    compare_kmc_to_ode
)
from src.ode_solver import solve_binding_odes
from src.parameters import SimulationParameters


def test_minimal_antigen_initialization():
    """Test MinimalAntigen initialization."""
    ag = MinimalAntigen()
    assert ag.state == 0
    assert ag.is_free
    assert not ag.has_a
    assert not ag.has_b


def test_minimal_antigen_bind_a():
    """Test binding A to free antigen."""
    ag = MinimalAntigen()
    ag.bind_a()
    assert ag.state == 1
    assert not ag.is_free
    assert ag.has_a
    assert not ag.has_b


def test_minimal_antigen_bind_b():
    """Test binding B to free antigen."""
    ag = MinimalAntigen()
    ag.bind_b()
    assert ag.state == 2
    assert not ag.is_free
    assert not ag.has_a
    assert ag.has_b


def test_minimal_antigen_sandwich():
    """Test creating sandwich complex."""
    ag = MinimalAntigen()
    ag.bind_a()
    ag.bind_b()
    assert ag.state == 3
    assert ag.has_a
    assert ag.has_b


def test_minimal_antigen_unbind():
    """Test unbinding from sandwich."""
    ag = MinimalAntigen()
    ag.bind_a()
    ag.bind_b()
    ag.unbind_a()
    assert ag.state == 2
    assert not ag.has_a
    assert ag.has_b


def test_run_minimal_kmc_basic():
    """Test basic KMC simulation."""
    result = run_minimal_kmc(
        N_antigen=10,
        C_A=1e-6,
        C_B=1e-6,
        kon_a=1e5,
        koff_a=0.1,
        kon_b=1e5,
        koff_b=0.1,
        dt=0.01,
        n_steps=10,
        record_interval=1,
        seed=42
    )
    
    assert 't' in result
    assert 'Free' in result
    assert 'Bound_A' in result
    assert 'Bound_B' in result
    assert 'Sandwich' in result
    assert 'params' in result
    
    # Check conservation of antigens
    for i in range(len(result['t'])):
        total = (result['Free'][i] + result['Bound_A'][i] + 
                result['Bound_B'][i] + result['Sandwich'][i])
        assert total == 10


def test_run_minimal_kmc_probability_validation():
    """Test that probabilities exceeding 1 raise error."""
    with pytest.raises(ValueError, match="probabilities exceed"):
        run_minimal_kmc(
            N_antigen=10,
            C_A=1.0,  # Very high concentration
            C_B=1e-6,
            kon_a=1e5,
            koff_a=0.1,
            kon_b=1e5,
            koff_b=0.1,
            dt=1.0,  # Large time step
            n_steps=10,
            seed=42
        )


def test_run_multiple_replicates():
    """Test running multiple replicates."""
    n_replicates = 3
    replicates = run_multiple_replicates(
        n_replicates=n_replicates,
        N_antigen=10,
        C_A=1e-6,
        C_B=1e-6,
        kon_a=1e5,
        koff_a=0.1,
        kon_b=1e5,
        koff_b=0.1,
        dt=0.01,
        n_steps=10,
        record_interval=1
    )
    
    assert len(replicates) == n_replicates
    assert all('t' in rep for rep in replicates)
    assert all('Free' in rep for rep in replicates)


def test_calculate_statistics():
    """Test statistics calculation."""
    replicates = run_multiple_replicates(
        n_replicates=3,
        N_antigen=10,
        C_A=1e-6,
        C_B=1e-6,
        kon_a=1e5,
        koff_a=0.1,
        kon_b=1e5,
        koff_b=0.1,
        dt=0.01,
        n_steps=10,
        record_interval=1
    )
    
    stats = calculate_statistics(replicates)
    
    assert 't' in stats
    assert 'Free_mean' in stats
    assert 'Free_std' in stats
    assert 'Bound_A_mean' in stats
    assert 'Bound_A_std' in stats
    
    # Initial state should have zero variance (all start at Free=10)
    assert stats['Free_mean'][0] == 10.0
    assert stats['Free_std'][0] == 0.0


def test_compare_kmc_to_ode():
    """Test comparison with ODE."""
    # Create parameters
    params = SimulationParameters(
        C_A=0.002,
        C_B=0.002,
        C_antigen=0.1,
        C_enhancement=1.0e-6,
        N_A_sim=50,
        N_B_sim=50,
        antibodies_per_particle=1000,
        n_patches=12,
        kon_a=1.0e5,
        koff_a=0.0001,
        kon_b=1.0e5,
        koff_b=0.0001,
        dt=0.1,
        n_steps_on=10,
        n_steps_off=10,
        n_repeats=100,
    )
    
    # Run KMC
    kmc_result = run_minimal_kmc(
        N_antigen=params.N_antigen_sim,
        C_A=params.C_antibody_A,
        C_B=params.C_antibody_B,
        kon_a=params.kon_a,
        koff_a=params.koff_a,
        kon_b=params.kon_b,
        koff_b=params.koff_b,
        dt=params.dt,
        n_steps=10,
        record_interval=1,
        seed=42
    )
    
    # Solve ODE
    t_span = (0, 1)
    t_eval = np.linspace(0, 1, 11)
    ode_result = solve_binding_odes(params, t_span, t_eval)
    
    # Prepare KMC stats
    kmc_stats = {
        't': kmc_result['t'],
        'Free_mean': kmc_result['Free'],
        'Bound_A_mean': kmc_result['Bound_A'],
        'Bound_B_mean': kmc_result['Bound_B'],
        'Sandwich_mean': kmc_result['Sandwich']
    }
    
    # Compare
    comparison = compare_kmc_to_ode(kmc_stats, ode_result, params)
    
    assert 'Free' in comparison
    assert 'Bound_A' in comparison
    assert 'Bound_B' in comparison
    assert 'Sandwich' in comparison
    
    for state in ['Free', 'Bound_A', 'Bound_B', 'Sandwich']:
        assert 'kmc' in comparison[state]
        assert 'ode' in comparison[state]
        assert 'ratio' in comparison[state]
        assert 'abs_error' in comparison[state]
        assert 'rel_error' in comparison[state]


def test_minimal_kmc_conservation():
    """Test that total antigen count is conserved throughout simulation."""
    N_antigen = 100
    result = run_minimal_kmc(
        N_antigen=N_antigen,
        C_A=1e-6,
        C_B=1e-6,
        kon_a=1e5,
        koff_a=0.1,
        kon_b=1e5,
        koff_b=0.1,
        dt=0.01,
        n_steps=100,
        record_interval=10,
        seed=42
    )
    
    # Check conservation at every recorded time point
    for i in range(len(result['t'])):
        total = (result['Free'][i] + result['Bound_A'][i] + 
                result['Bound_B'][i] + result['Sandwich'][i])
        assert total == N_antigen, f"Conservation violated at step {i}: total={total}, expected={N_antigen}"


def test_minimal_kmc_independent_reactions():
    """Test that A and B binding are independent."""
    # With symmetric parameters, we should see symmetric binding
    N_antigen = 1000
    result = run_minimal_kmc(
        N_antigen=N_antigen,
        C_A=1e-6,
        C_B=1e-6,
        kon_a=1e5,
        koff_a=0.1,
        kon_b=1e5,  # Same as A
        koff_b=0.1,  # Same as A
        dt=0.01,
        n_steps=1000,
        record_interval=100,
        seed=42
    )
    
    # At equilibrium, Bound_A and Bound_B should be similar (within stochastic noise)
    # This is a rough check - not exact due to stochasticity
    final_a = result['Bound_A'][-1]
    final_b = result['Bound_B'][-1]
    
    # Allow for 50% difference due to stochasticity in a single run
    if final_a > 0 and final_b > 0:
        ratio = final_a / final_b
        assert 0.5 < ratio < 2.0, f"Asymmetry too large: Bound_A={final_a}, Bound_B={final_b}"
