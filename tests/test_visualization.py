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
        from src.visualization import generate_non_overlapping_positions, layout_chain_vertical
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


@pytest.mark.skipif(not _HAS_VISUALIZATION, reason="Requires visualization dependencies")
def test_get_patch_direction():
    """Test that patch directions are correct."""
    from src.visualization import get_patch_direction
    import numpy as np
    
    # Test North patch
    north = get_patch_direction(0, n_patches=12)
    assert np.allclose(north, [0.0, 0.0, 1.0])
    
    # Test South patch
    south = get_patch_direction(1, n_patches=12)
    assert np.allclose(south, [0.0, 0.0, -1.0])
    
    # Test regular patches are in XY plane
    for patch_id in range(2, 12):
        direction = get_patch_direction(patch_id, n_patches=12)
        # Z component should be 0
        assert abs(direction[2]) < 1e-10
        # Should be unit vector
        assert abs(np.linalg.norm(direction) - 1.0) < 1e-10


@pytest.mark.skipif(not _HAS_VISUALIZATION, reason="Requires visualization dependencies")
def test_calculate_linked_particle_position():
    """Test calculation of linked particle position."""
    from src.visualization import calculate_linked_particle_position
    import numpy as np
    
    particle_radius = 0.5
    particle1_center = np.array([0.0, 0.0, 0.0])
    
    # Link North patch (0) of particle 1 to South patch (1) of particle 2
    particle2_center = calculate_linked_particle_position(
        particle1_center, 0, 1, particle_radius, n_patches=12
    )
    
    # Particle 2 should be directly above particle 1
    # North of p1 at (0, 0, 0.5), South of p2 at particle2_center + (0, 0, -0.5)
    # These should be the same point
    expected_particle2_center = np.array([0.0, 0.0, 2 * particle_radius])
    assert np.allclose(particle2_center, expected_particle2_center)
    
    # Check distance between centers
    distance = np.linalg.norm(particle2_center - particle1_center)
    assert abs(distance - 2 * particle_radius) < 1e-10


@pytest.mark.skipif(not _HAS_VISUALIZATION, reason="Requires visualization dependencies")
def test_layout_cluster_geometric():
    """Test geometric layout of a cluster."""
    from src.visualization import layout_cluster_geometric
    from src.simulation import Simulation
    import numpy as np
    
    params = create_test_params(N_A_sim=3, N_B_sim=1)
    sim = Simulation(params)
    
    # Create a chain: particle 0 -> particle 1 -> particle 2
    particles = sim.particles_a
    particles[0].add_link(0, particles[1].particle_id, 1)
    particles[1].add_link(1, particles[0].particle_id, 0)
    particles[1].add_link(0, particles[2].particle_id, 1)
    particles[2].add_link(1, particles[1].particle_id, 0)
    
    all_particles = sim.get_all_particles()
    cluster = {particles[0].particle_id, particles[1].particle_id, particles[2].particle_id}
    
    positions = layout_cluster_geometric(
        cluster, all_particles, particle_radius=0.5, n_patches=sim.params.n_patches
    )
    
    # Should have positions for all 3 particles
    assert len(positions) == 3
    
    # First particle at origin
    assert np.allclose(positions[particles[0].particle_id], [0.0, 0.0, 0.0])
    
    # Particles should be spaced 2*radius apart (touching)
    dist_01 = np.linalg.norm(
        positions[particles[1].particle_id] - positions[particles[0].particle_id]
    )
    assert abs(dist_01 - 1.0) < 1e-10  # 2 * 0.5
    
    dist_12 = np.linalg.norm(
        positions[particles[2].particle_id] - positions[particles[1].particle_id]
    )
    assert abs(dist_12 - 1.0) < 1e-10


@pytest.mark.skipif(not _HAS_VISUALIZATION, reason="Requires visualization dependencies")
def test_layout_particles_geometric():
    """Test geometric layout of all particles."""
    from src.visualization import layout_particles_geometric
    from src.simulation import Simulation
    import numpy as np
    
    params = create_test_params(N_A_sim=2, N_B_sim=2)
    sim = Simulation(params)
    
    # Create a link between two particles
    sim.particles_a[0].add_link(0, sim.particles_b[0].particle_id, 1)
    sim.particles_b[0].add_link(1, sim.particles_a[0].particle_id, 0)
    
    positions = layout_particles_geometric(sim, particle_radius=0.5)
    
    # Should return array with correct shape
    n_particles = len(sim.get_all_particles())
    assert positions.shape == (n_particles, 3)
    
    # All positions should be finite
    assert np.all(np.isfinite(positions))


@pytest.mark.skipif(not _HAS_VISUALIZATION, reason="Requires visualization dependencies")
def test_visualize_with_geometric_layout(tmp_path):
    """Test visualization with geometric layout."""
    from src.simulation import Simulation
    
    params = create_test_params()
    sim = Simulation(params)
    
    # Create a link
    particle_a = sim.particles_a[0]
    particle_b = sim.particles_b[0]
    particle_a.add_link(0, particle_b.particle_id, 1)
    particle_b.add_link(1, particle_a.particle_id, 0)
    
    output_path = tmp_path / "test_geometric.png"
    
    # Use geometric layout
    visualize_system_3d(
        sim,
        title="Test Geometric Layout",
        save_path=output_path,
        use_geometric_layout=True,
        show_patch_directions=True,
    )
    
    # Check file was created
    assert output_path.exists()
    assert output_path.stat().st_size > 0


@pytest.mark.skipif(not _HAS_VISUALIZATION, reason="Requires visualization dependencies")
def test_visualize_without_geometric_layout(tmp_path):
    """Test visualization with random layout (backward compatibility)."""
    from src.simulation import Simulation
    
    params = create_test_params()
    sim = Simulation(params)
    
    output_path = tmp_path / "test_random.png"
    
    # Use random layout
    visualize_system_3d(
        sim,
        title="Test Random Layout",
        save_path=output_path,
        use_geometric_layout=False,
        show_patch_directions=False,
    )
    
    # Check file was created
    assert output_path.exists()
    assert output_path.stat().st_size > 0


@pytest.mark.skipif(not _HAS_VISUALIZATION, reason="Requires visualization dependencies")
def test_generate_non_overlapping_positions_in_region():
    """Test that positions are generated within a region without overlap."""
    from src.visualization import generate_non_overlapping_positions_in_region
    import numpy as np
    
    particle_radius = 0.3
    center = np.array([5.0, 5.0, 5.0])
    region_size = 3.0
    
    positions = generate_non_overlapping_positions_in_region(
        n_particles=10,
        center=center,
        region_size=region_size,
        particle_radius=particle_radius,
    )
    
    # Check shape
    assert positions.shape == (10, 3)
    
    # Check all positions are within region bounds
    min_bound = center - region_size / 2
    max_bound = center + region_size / 2
    assert np.all(positions >= min_bound)
    assert np.all(positions <= max_bound)
    
    # Check no overlaps (min distance = 2.2 * radius)
    min_distance = 2.2 * particle_radius
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[j])
            assert dist >= min_distance * 0.99  # Allow small numerical error


@pytest.mark.skipif(not _HAS_VISUALIZATION, reason="Requires visualization dependencies")
def test_clustered_layout_separation():
    """Test that clusters are separated in the new layout."""
    from src.visualization import layout_particles_geometric
    from src.simulation import Simulation
    from src.clusters import find_clusters
    import numpy as np
    
    params = create_test_params(N_A_sim=4, N_B_sim=4)
    sim = Simulation(params)
    
    # Create two separate chains
    # Chain 1: particles 0 and 1
    sim.particles_a[0].add_link(0, sim.particles_b[0].particle_id, 1)
    sim.particles_b[0].add_link(1, sim.particles_a[0].particle_id, 0)
    
    # Chain 2: particles 2 and 3
    sim.particles_a[1].add_link(0, sim.particles_b[1].particle_id, 1)
    sim.particles_b[1].add_link(1, sim.particles_a[1].particle_id, 0)
    
    # Get positions
    positions = layout_particles_geometric(sim, particle_radius=0.3)
    
    # Find clusters
    all_particles = sim.get_all_particles()
    clusters = find_clusters(all_particles)
    
    # Should have at least 2 multi-particle clusters
    multi_particle_clusters = [c for c in clusters if len(c) > 1]
    assert len(multi_particle_clusters) >= 2
    
    # Calculate centers of first two multi-particle clusters
    cluster1_ids = list(multi_particle_clusters[0])
    cluster2_ids = list(multi_particle_clusters[1])
    
    # Map particle IDs to positions
    all_particles_list = list(all_particles.values())
    particle_id_to_idx = {p.particle_id: i for i, p in enumerate(all_particles_list)}
    
    cluster1_positions = [positions[particle_id_to_idx[pid]] for pid in cluster1_ids]
    cluster2_positions = [positions[particle_id_to_idx[pid]] for pid in cluster2_ids]
    
    # Calculate cluster centers
    center1 = np.mean(cluster1_positions, axis=0)
    center2 = np.mean(cluster2_positions, axis=0)
    
    # Clusters should be separated by at least cluster_spacing (20.0)
    # Allow some variation due to random placement within regions (region_size=3.0)
    # Minimum distance = cluster_spacing - region_size = 20.0 - 3.0 = 17.0
    distance = np.linalg.norm(center1 - center2)
    assert distance >= 17.0  # Clusters on adjacent grid cells, accounting for region spread


@pytest.mark.skipif(not _HAS_VISUALIZATION, reason="Requires visualization dependencies")
def test_layout_chain_vertical():
    """Test vertical layout of chain particles."""
    from src.simulation import Simulation
    import numpy as np
    
    params = create_test_params(N_A_sim=3, N_B_sim=1)
    sim = Simulation(params)
    
    # Create a chain: particle 0 -> particle 1 -> particle 2
    particles = sim.particles_a
    particles[0].add_link(0, particles[1].particle_id, 1)
    particles[1].add_link(1, particles[0].particle_id, 0)
    particles[1].add_link(0, particles[2].particle_id, 1)
    particles[2].add_link(1, particles[1].particle_id, 0)
    
    all_particles = sim.get_all_particles()
    cluster = {particles[0].particle_id, particles[1].particle_id, particles[2].particle_id}
    
    center = np.array([10.0, 20.0, 30.0])
    particle_radius = 0.5
    
    positions = layout_chain_vertical(
        cluster=cluster,
        all_particles_dict=all_particles,
        center=center,
        particle_radius=particle_radius
    )
    
    # Should have 3 positions
    assert positions.shape == (3, 3)
    
    # All particles should have the same X and Y coordinates (aligned vertically)
    assert np.allclose(positions[:, 0], center[0])  # All X same
    assert np.allclose(positions[:, 1], center[1])  # All Y same
    
    # Z coordinates should be different and spaced properly
    z_coords = positions[:, 2]
    z_coords_sorted = sorted(z_coords)
    z_diffs = np.diff(z_coords_sorted)
    expected_spacing = 2.5 * particle_radius  # 1.25
    assert np.allclose(z_diffs, expected_spacing)
    
    # The chain should be centered around the center Z coordinate
    mean_z = np.mean(z_coords_sorted)
    assert abs(mean_z - center[2]) < 0.01  # Should be very close to center[2]


@pytest.mark.skipif(not _HAS_VISUALIZATION, reason="Requires visualization dependencies")
def test_chain_uses_vertical_layout():
    """Test that chains are laid out vertically in geometric layout."""
    from src.simulation import Simulation
    from src.visualization import layout_particles_geometric
    from src.clusters import find_clusters, classify_cluster
    import numpy as np
    
    params = create_test_params(N_A_sim=3, N_B_sim=1)
    sim = Simulation(params)
    
    # Create a chain: particle 0 -> particle 1 -> particle 2
    particles = sim.particles_a
    particles[0].add_link(0, particles[1].particle_id, 1)
    particles[1].add_link(1, particles[0].particle_id, 0)
    particles[1].add_link(0, particles[2].particle_id, 1)
    particles[2].add_link(1, particles[1].particle_id, 0)
    
    # Get positions
    positions = layout_particles_geometric(sim, particle_radius=0.5)
    
    # Verify there's a chain cluster
    all_particles = sim.get_all_particles()
    clusters = find_clusters(all_particles)
    
    # Find the chain cluster (the one with 3 particles)
    chain_cluster = None
    for cluster in clusters:
        if len(cluster) > 1:
            cluster_type = classify_cluster(cluster, all_particles)
            if cluster_type == 'Chain':
                chain_cluster = cluster
                break
    
    assert chain_cluster is not None, "No chain cluster found"
    assert len(chain_cluster) == 3, "Chain should have 3 particles"
    
    # Extract positions for the chain particles
    all_particles_list = list(all_particles.values())
    particle_id_to_idx = {p.particle_id: i for i, p in enumerate(all_particles_list)}
    
    chain_positions = []
    for particle_id in chain_cluster:
        idx = particle_id_to_idx[particle_id]
        chain_positions.append(positions[idx])
    
    chain_positions = np.array(chain_positions)
    
    # All X coordinates should be the same (vertical along Z)
    x_coords = chain_positions[:, 0]
    assert np.allclose(x_coords, x_coords[0])
    
    # All Y coordinates should be the same
    y_coords = chain_positions[:, 1]
    assert np.allclose(y_coords, y_coords[0])
    
    # Z coordinates should be different and evenly spaced
    z_coords = chain_positions[:, 2]
    assert len(np.unique(z_coords)) == 3  # All different
    z_diffs = np.diff(sorted(z_coords))
    assert np.allclose(z_diffs, z_diffs[0])  # Evenly spaced
