"""
Tests for Antigen class.
"""

import pytest
from src.antigen import Antigen, AntigenState
from src.particle import Particle


def test_antigen_initialization():
    """Test that antigens are initialized correctly."""
    antigen = Antigen(antigen_id=1)
    
    assert antigen.antigen_id == 1
    assert antigen.state == AntigenState.FREE
    assert antigen.binding_a is None
    assert antigen.binding_b is None


def test_bind_to_a():
    """Test binding to type A particle."""
    antigen = Antigen(antigen_id=1)
    
    antigen.bind_to_a(particle_id=10, patch_id=5)
    assert antigen.binding_a == (10, 5)
    assert antigen.state == AntigenState.BOUND_A


def test_bind_to_b():
    """Test binding to type B particle."""
    antigen = Antigen(antigen_id=1)
    
    antigen.bind_to_b(particle_id=20, patch_id=3)
    assert antigen.binding_b == (20, 3)
    assert antigen.state == AntigenState.BOUND_B


def test_bind_to_both_creates_sandwich():
    """Test that binding to both A and B creates sandwich."""
    antigen = Antigen(antigen_id=1)
    
    antigen.bind_to_a(particle_id=10, patch_id=5)
    assert antigen.state == AntigenState.BOUND_A
    
    antigen.bind_to_b(particle_id=20, patch_id=3)
    assert antigen.state == AntigenState.SANDWICH
    assert antigen.binding_a == (10, 5)
    assert antigen.binding_b == (20, 3)


def test_bind_to_a_when_already_bound():
    """Test that binding to A when already bound raises error."""
    antigen = Antigen(antigen_id=1)
    antigen.bind_to_a(particle_id=10, patch_id=5)
    
    with pytest.raises(ValueError, match="already bound to A"):
        antigen.bind_to_a(particle_id=11, patch_id=6)


def test_bind_to_b_when_already_bound():
    """Test that binding to B when already bound raises error."""
    antigen = Antigen(antigen_id=1)
    antigen.bind_to_b(particle_id=20, patch_id=3)
    
    with pytest.raises(ValueError, match="already bound to B"):
        antigen.bind_to_b(particle_id=21, patch_id=4)


def test_unbind_from_a():
    """Test unbinding from type A particle."""
    antigen = Antigen(antigen_id=1)
    antigen.bind_to_a(particle_id=10, patch_id=5)
    
    binding = antigen.unbind_from_a()
    assert binding == (10, 5)
    assert antigen.binding_a is None
    assert antigen.state == AntigenState.FREE


def test_unbind_from_b():
    """Test unbinding from type B particle."""
    antigen = Antigen(antigen_id=1)
    antigen.bind_to_b(particle_id=20, patch_id=3)
    
    binding = antigen.unbind_from_b()
    assert binding == (20, 3)
    assert antigen.binding_b is None
    assert antigen.state == AntigenState.FREE


def test_unbind_from_a_in_sandwich():
    """Test that unbinding from A in sandwich transitions to BOUND_B."""
    antigen = Antigen(antigen_id=1)
    antigen.bind_to_a(particle_id=10, patch_id=5)
    antigen.bind_to_b(particle_id=20, patch_id=3)
    assert antigen.state == AntigenState.SANDWICH
    
    antigen.unbind_from_a()
    assert antigen.state == AntigenState.BOUND_B
    assert antigen.binding_a is None
    assert antigen.binding_b == (20, 3)


def test_unbind_from_b_in_sandwich():
    """Test that unbinding from B in sandwich transitions to BOUND_A."""
    antigen = Antigen(antigen_id=1)
    antigen.bind_to_a(particle_id=10, patch_id=5)
    antigen.bind_to_b(particle_id=20, patch_id=3)
    assert antigen.state == AntigenState.SANDWICH
    
    antigen.unbind_from_b()
    assert antigen.state == AntigenState.BOUND_A
    assert antigen.binding_a == (10, 5)
    assert antigen.binding_b is None


def test_update_state():
    """Test state updates."""
    antigen = Antigen(antigen_id=1)
    
    # Free
    antigen.update_state()
    assert antigen.state == AntigenState.FREE
    
    # Bound A
    antigen.binding_a = (10, 5)
    antigen.update_state()
    assert antigen.state == AntigenState.BOUND_A
    
    # Bound B
    antigen.binding_a = None
    antigen.binding_b = (20, 3)
    antigen.update_state()
    assert antigen.state == AntigenState.BOUND_B
    
    # Sandwich
    antigen.binding_a = (10, 5)
    antigen.update_state()
    assert antigen.state == AntigenState.SANDWICH


def test_is_on_north_or_south_patch():
    """Test checking if antigen is on North or South patch."""
    # Create particles
    particle_a = Particle(particle_id=10, particle_type='A', n_patches=10)
    particle_b = Particle(particle_id=20, particle_type='B', n_patches=10)
    particles = {10: particle_a, 20: particle_b}
    
    # Test with antigen on North patch
    antigen = Antigen(antigen_id=1)
    antigen.bind_to_a(particle_id=10, patch_id=0)  # North patch
    assert antigen.is_on_north_or_south_patch(particles) is True
    
    # Test with antigen on South patch
    antigen2 = Antigen(antigen_id=2)
    antigen2.bind_to_b(particle_id=20, patch_id=1)  # South patch
    assert antigen2.is_on_north_or_south_patch(particles) is True
    
    # Test with antigen on regular patch
    antigen3 = Antigen(antigen_id=3)
    antigen3.bind_to_a(particle_id=10, patch_id=5)  # Regular patch
    assert antigen3.is_on_north_or_south_patch(particles) is False
    
    # Test with sandwich on one North/South
    antigen4 = Antigen(antigen_id=4)
    antigen4.bind_to_a(particle_id=10, patch_id=0)  # North patch
    antigen4.bind_to_b(particle_id=20, patch_id=5)  # Regular patch
    assert antigen4.is_on_north_or_south_patch(particles) is True
