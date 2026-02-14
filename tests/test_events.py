"""
Tests for event probability calculations.
"""

import pytest
from src.events import (
    calc_bind_probability_field_off,
    calc_bind_probability_field_on_enhanced,
    calc_unbind_probability,
    calc_neighbor_probability,
)


def test_calc_bind_probability_field_off():
    """Test standard binding probability calculation."""
    kon = 1.0e5  # M^-1 s^-1
    C_antibody = 1.0e-6  # M
    dt = 0.001  # s
    
    prob = calc_bind_probability_field_off(kon, C_antibody, dt)
    expected = kon * C_antibody * dt
    
    assert prob == expected
    assert prob == pytest.approx(0.0001)


def test_calc_bind_probability_field_on_enhanced():
    """Test enhanced binding probability calculation."""
    kon = 1.0e5  # M^-1 s^-1
    C_enhancement = 1.0e-6  # M
    P_neighbor = 0.5
    dt = 0.001  # s
    
    prob = calc_bind_probability_field_on_enhanced(kon, C_enhancement, P_neighbor, dt)
    expected = kon * C_enhancement * P_neighbor * dt
    
    assert prob == expected
    assert prob == pytest.approx(0.00005)


def test_calc_unbind_probability():
    """Test unbinding probability calculation."""
    koff = 0.1  # s^-1
    dt = 0.001  # s
    
    prob = calc_unbind_probability(koff, dt)
    expected = koff * dt
    
    assert prob == expected
    assert prob == 0.0001


def test_calc_neighbor_probability():
    """Test neighbor probability calculation."""
    N_type = 50
    N_total = 100
    
    prob = calc_neighbor_probability(N_type, N_total)
    assert prob == 0.5


def test_calc_neighbor_probability_all_same_type():
    """Test neighbor probability when all particles are same type."""
    N_type = 100
    N_total = 100
    
    prob = calc_neighbor_probability(N_type, N_total)
    assert prob == 1.0


def test_calc_neighbor_probability_none_of_type():
    """Test neighbor probability when no particles of type."""
    N_type = 0
    N_total = 100
    
    prob = calc_neighbor_probability(N_type, N_total)
    assert prob == 0.0


def test_calc_neighbor_probability_zero_total():
    """Test neighbor probability with zero total particles."""
    N_type = 0
    N_total = 0
    
    prob = calc_neighbor_probability(N_type, N_total)
    assert prob == 0.0


def test_binding_probability_scales_with_time():
    """Test that binding probability scales linearly with dt."""
    kon = 1.0e5
    C_antibody = 1.0e-6
    
    prob1 = calc_bind_probability_field_off(kon, C_antibody, 0.001)
    prob2 = calc_bind_probability_field_off(kon, C_antibody, 0.002)
    
    assert prob2 == pytest.approx(2 * prob1)


def test_binding_probability_scales_with_concentration():
    """Test that binding probability scales linearly with concentration."""
    kon = 1.0e5
    dt = 0.001
    
    prob1 = calc_bind_probability_field_off(kon, 1.0e-6, dt)
    prob2 = calc_bind_probability_field_off(kon, 2.0e-6, dt)
    
    assert prob2 == pytest.approx(2 * prob1)
