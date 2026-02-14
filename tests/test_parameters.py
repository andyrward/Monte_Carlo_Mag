"""
Tests for SimulationParameters class.
"""

import pytest
from src.parameters import SimulationParameters


def test_parameters_initialization():
    """Test that parameters are initialized correctly."""
    params = SimulationParameters(
        C_A=10.0,
        C_B=10.0,
        C_antigen=1.0,
        C_enhancement=1.0e-6,
        N_A_sim=50,
        N_B_sim=50,
        antibodies_per_particle=1000,
        n_patches=12,
        kon_a=1.0e5,
        koff_a=0.1,
        kon_b=1.0e5,
        koff_b=0.1,
        dt=0.001,
        n_steps_on=1000,
        n_steps_off=1000,
        n_repeats=5,
    )
    
    assert params.C_A == 10.0
    assert params.C_B == 10.0
    assert params.N_A_sim == 50
    assert params.N_B_sim == 50


def test_v_box_calculation():
    """Test simulation volume calculation."""
    params = SimulationParameters(
        C_A=10.0,  # nM
        C_B=10.0,
        C_antigen=1.0,
        C_enhancement=1.0e-6,
        N_A_sim=50,
        N_B_sim=50,
        antibodies_per_particle=1000,
        n_patches=12,
        kon_a=1.0e5,
        koff_a=0.1,
        kon_b=1.0e5,
        koff_b=0.1,
        dt=0.001,
        n_steps_on=1000,
        n_steps_off=1000,
        n_repeats=5,
    )
    
    # V_box = N_A_sim / (C_A * 1e-9 * N_A)
    N_A = 6.022e23
    expected_v_box = 50 / (10.0 * 1e-9 * N_A)
    
    assert params.V_box is not None
    assert params.V_box == pytest.approx(expected_v_box)


def test_n_antigen_calculation():
    """Test antigen count calculation."""
    params = SimulationParameters(
        C_A=10.0,
        C_B=10.0,
        C_antigen=1.0,  # nM
        C_enhancement=1.0e-6,
        N_A_sim=50,
        N_B_sim=50,
        antibodies_per_particle=1000,
        n_patches=12,
        kon_a=1.0e5,
        koff_a=0.1,
        kon_b=1.0e5,
        koff_b=0.1,
        dt=0.001,
        n_steps_on=1000,
        n_steps_off=1000,
        n_repeats=5,
    )
    
    # N_antigen_sim = int(C_antigen * 1e-9 * N_A * V_box)
    N_A = 6.022e23
    expected_n_antigen = int(1.0 * 1e-9 * N_A * params.V_box)
    
    assert params.N_antigen_sim == expected_n_antigen


def test_c_antibody_a_calculation():
    """Test antibody A concentration calculation."""
    params = SimulationParameters(
        C_A=10.0,
        C_B=10.0,
        C_antigen=1.0,
        C_enhancement=1.0e-6,
        N_A_sim=50,
        N_B_sim=50,
        antibodies_per_particle=1000,
        n_patches=12,
        kon_a=1.0e5,
        koff_a=0.1,
        kon_b=1.0e5,
        koff_b=0.1,
        dt=0.001,
        n_steps_on=1000,
        n_steps_off=1000,
        n_repeats=5,
    )
    
    # C_antibody_A = (N_A_sim * antibodies_per_particle) / (N_A * V_box)
    N_A = 6.022e23
    expected_c_antibody_a = (50 * 1000) / (N_A * params.V_box)
    
    assert params.C_antibody_A is not None
    assert params.C_antibody_A == pytest.approx(expected_c_antibody_a)


def test_c_antibody_b_calculation():
    """Test antibody B concentration calculation."""
    params = SimulationParameters(
        C_A=10.0,
        C_B=10.0,
        C_antigen=1.0,
        C_enhancement=1.0e-6,
        N_A_sim=50,
        N_B_sim=50,
        antibodies_per_particle=1000,
        n_patches=12,
        kon_a=1.0e5,
        koff_a=0.1,
        kon_b=1.0e5,
        koff_b=0.1,
        dt=0.001,
        n_steps_on=1000,
        n_steps_off=1000,
        n_repeats=5,
    )
    
    # C_antibody_B = (N_B_sim * antibodies_per_particle) / (N_A * V_box)
    N_A = 6.022e23
    expected_c_antibody_b = (50 * 1000) / (N_A * params.V_box)
    
    assert params.C_antibody_B is not None
    assert params.C_antibody_B == pytest.approx(expected_c_antibody_b)


def test_different_particle_counts():
    """Test with different A and B particle counts."""
    params = SimulationParameters(
        C_A=10.0,
        C_B=5.0,
        C_antigen=1.0,
        C_enhancement=1.0e-6,
        N_A_sim=100,
        N_B_sim=25,
        antibodies_per_particle=500,
        n_patches=12,
        kon_a=1.0e5,
        koff_a=0.1,
        kon_b=1.0e5,
        koff_b=0.1,
        dt=0.001,
        n_steps_on=1000,
        n_steps_off=1000,
        n_repeats=5,
    )
    
    N_A = 6.022e23
    
    # Different antibody concentrations
    expected_c_antibody_a = (100 * 500) / (N_A * params.V_box)
    expected_c_antibody_b = (25 * 500) / (N_A * params.V_box)
    
    assert params.C_antibody_A == pytest.approx(expected_c_antibody_a)
    assert params.C_antibody_B == pytest.approx(expected_c_antibody_b)
