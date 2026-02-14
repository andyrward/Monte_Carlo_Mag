"""
Tests for Particle class.
"""

import pytest
from src.particle import Particle


def test_particle_initialization():
    """Test that particles are initialized correctly."""
    particle = Particle(particle_id=1, particle_type='A', n_patches=10)
    
    assert particle.particle_id == 1
    assert particle.particle_type == 'A'
    assert particle.n_patches == 10
    assert len(particle.patches) == 10
    assert all(v is None for v in particle.patches.values())
    assert len(particle.links) == 0


def test_particle_type_validation():
    """Test that invalid particle types raise errors."""
    with pytest.raises(ValueError, match="particle_type must be 'A' or 'B'"):
        Particle(particle_id=1, particle_type='C', n_patches=10)


def test_particle_patches_validation():
    """Test that invalid patch counts raise errors."""
    with pytest.raises(ValueError, match="n_patches must be between 2 and 30"):
        Particle(particle_id=1, particle_type='A', n_patches=31)
    
    with pytest.raises(ValueError, match="n_patches must be between 2 and 30"):
        Particle(particle_id=1, particle_type='A', n_patches=1)


def test_north_patch():
    """Test North patch detection."""
    particle = Particle(particle_id=1, particle_type='A', n_patches=10)
    
    assert particle.is_north_patch(0) is True
    assert particle.is_north_patch(1) is False
    assert particle.is_north_patch(5) is False


def test_south_patch():
    """Test South patch detection."""
    particle = Particle(particle_id=1, particle_type='A', n_patches=10)
    
    assert particle.is_south_patch(0) is False
    assert particle.is_south_patch(1) is True
    assert particle.is_south_patch(5) is False


def test_north_or_south_patch():
    """Test North or South patch detection."""
    particle = Particle(particle_id=1, particle_type='A', n_patches=10)
    
    assert particle.is_north_or_south_patch(0) is True
    assert particle.is_north_or_south_patch(1) is True
    assert particle.is_north_or_south_patch(2) is False
    assert particle.is_north_or_south_patch(5) is False


def test_bind_antigen():
    """Test antigen binding."""
    particle = Particle(particle_id=1, particle_type='A', n_patches=10)
    
    particle.bind_antigen(patch_id=5, antigen_id=100)
    assert particle.patches[5] == 100


def test_bind_antigen_to_occupied_patch():
    """Test that binding to occupied patch raises error."""
    particle = Particle(particle_id=1, particle_type='A', n_patches=10)
    particle.bind_antigen(patch_id=5, antigen_id=100)
    
    with pytest.raises(ValueError, match="already has antigen"):
        particle.bind_antigen(patch_id=5, antigen_id=101)


def test_bind_antigen_invalid_patch():
    """Test that binding to invalid patch raises error."""
    particle = Particle(particle_id=1, particle_type='A', n_patches=10)
    
    with pytest.raises(ValueError, match="Invalid patch_id"):
        particle.bind_antigen(patch_id=15, antigen_id=100)


def test_unbind_antigen():
    """Test antigen unbinding."""
    particle = Particle(particle_id=1, particle_type='A', n_patches=10)
    particle.bind_antigen(patch_id=5, antigen_id=100)
    
    antigen_id = particle.unbind_antigen(patch_id=5)
    assert antigen_id == 100
    assert particle.patches[5] is None


def test_unbind_antigen_from_empty_patch():
    """Test unbinding from empty patch."""
    particle = Particle(particle_id=1, particle_type='A', n_patches=10)
    
    antigen_id = particle.unbind_antigen(patch_id=5)
    assert antigen_id is None


def test_add_link():
    """Test adding particle-particle link."""
    particle = Particle(particle_id=1, particle_type='A', n_patches=10)
    
    particle.add_link(my_patch_id=5, other_particle_id=2, other_patch_id=3)
    assert particle.links[5] == (2, 3)


def test_add_link_to_occupied_patch():
    """Test that adding link to occupied patch raises error."""
    particle = Particle(particle_id=1, particle_type='A', n_patches=10)
    particle.add_link(my_patch_id=5, other_particle_id=2, other_patch_id=3)
    
    with pytest.raises(ValueError, match="already has a link"):
        particle.add_link(my_patch_id=5, other_particle_id=3, other_patch_id=4)


def test_remove_link():
    """Test removing particle-particle link."""
    particle = Particle(particle_id=1, particle_type='A', n_patches=10)
    particle.add_link(my_patch_id=5, other_particle_id=2, other_patch_id=3)
    
    link = particle.remove_link(my_patch_id=5)
    assert link == (2, 3)
    assert 5 not in particle.links


def test_remove_link_from_empty_patch():
    """Test removing link from empty patch."""
    particle = Particle(particle_id=1, particle_type='A', n_patches=10)
    
    link = particle.remove_link(my_patch_id=5)
    assert link is None
