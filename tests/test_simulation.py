"""
Tests for Simulation class.
"""

import pytest
from src.simulation import Simulation
from src.parameters import SimulationParameters
from src.antigen import AntigenState


def create_test_params(**kwargs):
    """Create test parameters with defaults."""
    defaults = {
        'C_A': 10.0,
        'C_B': 10.0,
        'C_antigen': 1.0,
        'C_enhancement': 1.0e-6,
        'N_A_sim': 10,
        'N_B_sim': 10,
        'antibodies_per_particle': 100,
        'n_patches': 5,
        'kon_a': 1.0e5,
        'koff_a': 0.1,
        'kon_b': 1.0e5,
        'koff_b': 0.1,
        'dt': 0.001,
        'n_steps_on': 10,
        'n_steps_off': 10,
        'n_repeats': 2,
    }
    defaults.update(kwargs)
    return SimulationParameters(**defaults)


def test_simulation_initialization():
    """Test that simulation is initialized correctly."""
    params = create_test_params()
    sim = Simulation(params)
    
    assert sim.params == params
    assert len(sim.particles_a) == params.N_A_sim
    assert len(sim.particles_b) == params.N_B_sim
    assert len(sim.antigens) == params.N_antigen_sim
    assert sim.current_step == 0
    assert sim.current_time == 0.0
    assert sim.field_on is False


def test_simulation_particle_ids_unique():
    """Test that all particle IDs are unique."""
    params = create_test_params()
    sim = Simulation(params)
    
    all_ids = [p.particle_id for p in sim.particles_a + sim.particles_b]
    assert len(all_ids) == len(set(all_ids))


def test_simulation_particle_types():
    """Test that particles have correct types."""
    params = create_test_params()
    sim = Simulation(params)
    
    assert all(p.particle_type == 'A' for p in sim.particles_a)
    assert all(p.particle_type == 'B' for p in sim.particles_b)


def test_is_field_on():
    """Test field state determination."""
    params = create_test_params(n_steps_on=10, n_steps_off=5)
    sim = Simulation(params)
    
    # First cycle - ON phase
    for step in range(10):
        sim.current_step = step
        assert sim.is_field_on() is True
    
    # First cycle - OFF phase
    for step in range(10, 15):
        sim.current_step = step
        assert sim.is_field_on() is False
    
    # Second cycle - ON phase
    for step in range(15, 25):
        sim.current_step = step
        assert sim.is_field_on() is True


def test_step_increments_time_and_step():
    """Test that step() increments time and step counter."""
    params = create_test_params()
    sim = Simulation(params)
    
    initial_time = sim.current_time
    initial_step = sim.current_step
    
    sim.step()
    
    assert sim.current_time == initial_time + params.dt
    assert sim.current_step == initial_step + 1


def test_step_records_history():
    """Test that step() records history."""
    params = create_test_params()
    sim = Simulation(params)
    
    initial_history_len = len(sim.history['time'])
    
    sim.step()
    
    assert len(sim.history['time']) == initial_history_len + 1
    assert len(sim.history['step']) == initial_history_len + 1
    assert len(sim.history['field_on']) == initial_history_len + 1


def test_run_multiple_steps():
    """Test running simulation for multiple steps."""
    params = create_test_params()
    sim = Simulation(params)
    
    n_steps = 10
    sim.run(n_steps)
    
    assert sim.current_step == n_steps
    assert sim.current_time == pytest.approx(n_steps * params.dt)
    assert len(sim.history['time']) == n_steps


def test_history_tracking():
    """Test that history tracks all required fields."""
    params = create_test_params()
    sim = Simulation(params)
    
    sim.run(5)
    
    assert 'time' in sim.history
    assert 'step' in sim.history
    assert 'field_on' in sim.history
    assert 'n_free' in sim.history
    assert 'n_bound_a' in sim.history
    assert 'n_bound_b' in sim.history
    assert 'n_sandwich' in sim.history
    
    # All should have same length
    lengths = [len(v) for v in sim.history.values()]
    assert all(l == lengths[0] for l in lengths)


def test_antigen_counts_sum_to_total():
    """Test that antigen state counts sum to total antigens."""
    params = create_test_params()
    sim = Simulation(params)
    
    sim.run(10)
    
    # Check at each timestep
    for i in range(len(sim.history['time'])):
        total = (
            sim.history['n_free'][i] +
            sim.history['n_bound_a'][i] +
            sim.history['n_bound_b'][i] +
            sim.history['n_sandwich'][i]
        )
        assert total == params.N_antigen_sim


def test_all_particles_lookup():
    """Test that all particles are in lookup dictionary."""
    params = create_test_params()
    sim = Simulation(params)
    
    assert len(sim._all_particles) == params.N_A_sim + params.N_B_sim
    
    for particle in sim.particles_a:
        assert particle.particle_id in sim._all_particles
        assert sim._all_particles[particle.particle_id] == particle
    
    for particle in sim.particles_b:
        assert particle.particle_id in sim._all_particles
        assert sim._all_particles[particle.particle_id] == particle


def test_simulation_with_no_antigens():
    """Test simulation with zero antigens."""
    params = create_test_params(C_antigen=0.0)
    sim = Simulation(params)
    
    assert len(sim.antigens) == 0
    
    # Should still run without errors
    sim.run(5)
    
    # All counts should be zero
    assert all(count == 0 for count in sim.history['n_free'])
    assert all(count == 0 for count in sim.history['n_bound_a'])
    assert all(count == 0 for count in sim.history['n_bound_b'])
    assert all(count == 0 for count in sim.history['n_sandwich'])


def test_sandwich_link_formation():
    """Test that sandwich formation creates reciprocal links."""
    from src.antigen import AntigenState
    
    params = create_test_params()
    sim = Simulation(params)
    
    # Manually create a sandwich to test link formation
    antigen = sim.antigens[0]
    particle_a = sim.particles_a[0]
    particle_b = sim.particles_b[0]
    
    # Bind antigen to both particles (using valid patch IDs: 0-4 for n_patches=5)
    particle_a.bind_antigen(patch_id=2, antigen_id=antigen.antigen_id)
    antigen.bind_to_a(particle_a.particle_id, patch_id=2)
    
    particle_b.bind_antigen(patch_id=3, antigen_id=antigen.antigen_id)
    antigen.bind_to_b(particle_b.particle_id, patch_id=3)
    
    # Verify sandwich state
    assert antigen.state == AntigenState.SANDWICH
    
    # Create links
    particle_a.add_link(2, particle_b.particle_id, 3)
    particle_b.add_link(3, particle_a.particle_id, 2)
    
    # Verify reciprocal links exist
    assert 2 in particle_a.links
    assert particle_a.links[2] == (particle_b.particle_id, 3)
    
    assert 3 in particle_b.links
    assert particle_b.links[3] == (particle_a.particle_id, 2)
    
    # Test link removal when sandwich breaks
    particle_a.remove_link(2)
    particle_b.remove_link(3)
    
    assert 2 not in particle_a.links
    assert 3 not in particle_b.links


def test_get_all_particles_method():
    """Test that get_all_particles() returns the correct dictionary."""
    params = create_test_params()
    sim = Simulation(params)
    
    all_particles = sim.get_all_particles()
    
    # Should have all particles
    assert len(all_particles) == params.N_A_sim + params.N_B_sim
    
    # Should contain both A and B particles
    for particle in sim.particles_a:
        assert particle.particle_id in all_particles
        assert all_particles[particle.particle_id] == particle
    
    for particle in sim.particles_b:
        assert particle.particle_id in all_particles
        assert all_particles[particle.particle_id] == particle

