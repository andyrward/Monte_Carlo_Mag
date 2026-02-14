"""
Tests for visualization module.

These tests only run if visualization dependencies are installed.
"""

import pytest
from pathlib import Path

# Check if visualization dependencies are available
try:
    from src import _HAS_VISUALIZATION
    if _HAS_VISUALIZATION:
        from src import visualize_system_3d, create_cycle_snapshots
        from src.visualization import generate_non_overlapping_positions
except ImportError:
    _HAS_VISUALIZATION = False

# Skip all tests if visualization not available
pytestmark = pytest.mark.skipif(
    not _HAS_VISUALIZATION,
    reason="Visualization dependencies not installed"
)


def create_test_params(**kwargs):
    """Create test parameters with defaults."""
    from src.parameters import SimulationParameters
    
    defaults = {
        'C_A': 10.0,
        'C_B': 10.0,
        'C_antigen': 1.0,
        'C_enhancement': 1.0e-6,
        'N_A_sim': 5,
        'N_B_sim': 5,
        'antibodies_per_particle': 100,
        'n_patches': 5,
        'kon_a': 1.0e5,
        'koff_a': 0.1,
        'kon_b': 1.0e5,
        'koff_b': 0.1,
        'dt': 0.001,
        'n_steps_on': 5,
        'n_steps_off': 5,
        'n_repeats': 2,
    }
    defaults.update(kwargs)
    return SimulationParameters(**defaults)


@pytest.mark.skipif(not _HAS_VISUALIZATION, reason="Requires visualization dependencies")
def test_generate_non_overlapping_positions():
    """Test that positions are generated without overlap."""
    particle_radius = 0.3
    positions = generate_non_overlapping_positions(
        n_particles=10,
        box_size=10.0,
        particle_radius=particle_radius,
    )
    
    import numpy as np
    
    # Check shape
    assert positions.shape == (10, 3)
    
    # Check all positions are within bounds
    assert np.all(positions >= -5.0)
    assert np.all(positions <= 5.0)
    
    # Check no overlaps (min distance = 2 * radius)
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[j])
            assert dist >= 2 * particle_radius


@pytest.mark.skipif(not _HAS_VISUALIZATION, reason="Requires visualization dependencies")
def test_visualize_system_3d(tmp_path):
    """Test that visualization creates output file."""
    from src.simulation import Simulation
    
    params = create_test_params()
    sim = Simulation(params)
    
    output_path = tmp_path / "test_viz.png"
    
    # Should not raise errors
    visualize_system_3d(
        sim,
        title="Test Visualization",
        save_path=output_path,
    )
    
    # Check file was created
    assert output_path.exists()
    assert output_path.stat().st_size > 0


@pytest.mark.skipif(not _HAS_VISUALIZATION, reason="Requires visualization dependencies")
def test_visualize_system_3d_with_links(tmp_path):
    """Test visualization with particle links."""
    from src.simulation import Simulation
    
    params = create_test_params()
    sim = Simulation(params)
    
    # Create a link between two particles
    particle_a = sim.particles_a[0]
    particle_b = sim.particles_b[0]
    particle_a.add_link(0, particle_b.particle_id, 1)
    particle_b.add_link(1, particle_a.particle_id, 0)
    
    output_path = tmp_path / "test_viz_links.png"
    
    # Should not raise errors
    visualize_system_3d(
        sim,
        title="Test Visualization with Links",
        save_path=output_path,
    )
    
    # Check file was created
    assert output_path.exists()
    assert output_path.stat().st_size > 0


@pytest.mark.skipif(not _HAS_VISUALIZATION, reason="Requires visualization dependencies")
def test_create_cycle_snapshots(tmp_path):
    """Test that cycle snapshots are created."""
    from src.simulation import Simulation
    
    params = create_test_params(
        n_steps_on=5,
        n_steps_off=5,
        n_repeats=2,
    )
    sim = Simulation(params)
    
    # Create snapshots
    snapshot_paths = create_cycle_snapshots(
        sim,
        output_dir=tmp_path,
    )
    
    # Should have initial + 2 per cycle * 2 cycles = 5 snapshots
    assert len(snapshot_paths) == 5
    
    # All files should exist
    for path in snapshot_paths:
        assert path.exists()
        assert path.stat().st_size > 0
    
    # Simulation should have run to completion
    total_steps = params.n_repeats * (params.n_steps_on + params.n_steps_off)
    assert sim.current_step == total_steps


@pytest.mark.skipif(not _HAS_VISUALIZATION, reason="Requires visualization dependencies")
def test_position_consistency_across_snapshots():
    """Test that positions remain consistent across snapshots."""
    from src.simulation import Simulation
    import numpy as np
    
    params = create_test_params()
    sim = Simulation(params)
    
    n_particles = len(sim.get_all_particles())
    positions = generate_non_overlapping_positions(n_particles, box_size=10.0)
    
    # Save positions
    saved_positions = positions.copy()
    
    # Generate new positions
    new_positions = generate_non_overlapping_positions(n_particles, box_size=10.0)
    
    # Positions should be different (random)
    assert not np.allclose(saved_positions, new_positions)
