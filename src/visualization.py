"""
3D visualization tools for magnetic nanoparticle simulation.

Requires optional dependencies: numpy, matplotlib
Install with: pip install -e ".[visualization]"
"""

import random
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Any

try:
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Line3D
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False
    np = None  # type: ignore

if TYPE_CHECKING:
    from .simulation import Simulation
    from .particle import Particle
    import numpy as np

# Constants for geometric layout
CLUSTER_SPACING = 5.0  # Distance between clusters in geometric layout
MIN_BOX_SIZE = 10.0  # Minimum box size for visualization
BOX_PADDING_MULTIPLIER = 2.0  # Multiplier for box padding around structures
ARROW_LENGTH_MULTIPLIER = 1.5  # Multiplier for patch direction arrow length
ARROW_LENGTH_RATIO = 0.3  # Ratio of arrow head to total arrow length


def get_patch_direction(patch_id: int, n_patches: int = 12) -> Any:
    """
    Get the 3D direction vector for a patch on a particle.
    
    Patch layout:
    - Patch 0 (North): +Z direction
    - Patch 1 (South): -Z direction  
    - Patches 2-11: Distributed around equator in XY plane
    
    Args:
        patch_id: Patch index (0 to n_patches-1)
        n_patches: Total number of patches
        
    Returns:
        Unit vector (3D numpy array) pointing from particle center to patch position
        
    Raises:
        ImportError: If numpy is not available
    """
    if not HAS_DEPENDENCIES:
        raise ImportError("Visualization requires numpy and matplotlib. Install with: pip install -e '.[visualization]'")
    
    if patch_id == 0:  # North
        return np.array([0.0, 0.0, 1.0])
    elif patch_id == 1:  # South
        return np.array([0.0, 0.0, -1.0])
    else:
        # Distribute remaining patches around equator (in XY plane)
        n_regular = n_patches - 2
        angle_index = patch_id - 2
        theta = 2 * np.pi * angle_index / n_regular
        return np.array([np.cos(theta), np.sin(theta), 0.0])


def calculate_linked_particle_position(
    particle1_center: Any,
    patch1_id: int,
    patch2_id: int,
    particle_radius: float,
    n_patches: int
) -> Any:
    """
    Calculate position of particle 2 such that patch1 and patch2 are touching.
    
    Args:
        particle1_center: Position (3D array) of particle 1 center
        patch1_id: Which patch on particle 1
        patch2_id: Which patch on particle 2
        particle_radius: Radius of particles
        n_patches: Total patches per particle
        
    Returns:
        Position (3D array) of particle 2 center
        
    Raises:
        ImportError: If numpy is not available
    """
    if not HAS_DEPENDENCIES:
        raise ImportError("Visualization requires numpy and matplotlib. Install with: pip install -e '.[visualization]'")
    
    # Get patch directions
    patch1_direction = get_patch_direction(patch1_id, n_patches)
    patch2_direction = get_patch_direction(patch2_id, n_patches)
    
    # Patch 1 position on particle 1 surface
    patch1_position = particle1_center + particle_radius * patch1_direction
    
    # Particle 2 center positioned so patches touch:
    # patch1_position = particle2_center + particle_radius * patch2_direction
    # Therefore: particle2_center = patch1_position - particle_radius * patch2_direction
    particle2_center = patch1_position - particle_radius * patch2_direction
    
    return particle2_center


def layout_cluster_geometric(
    cluster: set[int],
    all_particles_dict: dict[int, 'Particle'],
    particle_radius: float,
    n_patches: int
) -> dict[int, Any]:
    """
    Position particles in a single cluster based on their patch connections.
    
    Uses BFS from a seed particle to build up the structure geometrically.
    
    Args:
        cluster: Set of particle IDs in this cluster
        all_particles_dict: Dictionary mapping particle_id to Particle objects
        particle_radius: Radius of particles
        n_patches: Number of patches per particle
        
    Returns:
        Dictionary mapping particle_id to position (3D array)
        
    Raises:
        ImportError: If numpy is not available
    """
    if not HAS_DEPENDENCIES:
        raise ImportError("Visualization requires numpy and matplotlib. Install with: pip install -e '.[visualization]'")
    
    from collections import deque
    
    if len(cluster) == 0:
        return {}
    
    positions = {}
    
    # Pick seed particle (place at origin of cluster)
    seed_particle_id = list(cluster)[0]
    positions[seed_particle_id] = np.array([0.0, 0.0, 0.0])
    
    # BFS to place connected particles
    queue = deque([seed_particle_id])
    visited = {seed_particle_id}
    
    while queue:
        current_particle_id = queue.popleft()
        current_particle = all_particles_dict[current_particle_id]
        current_position = positions[current_particle_id]
        
        # Process all links from this particle
        for my_patch_id, link in current_particle.links.items():
            other_particle_id, other_patch_id = link
            
            # Skip if already placed
            if other_particle_id in visited:
                continue
            
            # Calculate where to place the other particle
            other_position = calculate_linked_particle_position(
                current_position,
                my_patch_id,
                other_patch_id,
                particle_radius,
                n_patches
            )
            
            positions[other_particle_id] = other_position
            visited.add(other_particle_id)
            queue.append(other_particle_id)
    
    return positions


def generate_non_overlapping_positions_in_region(
    n_particles: int,
    center: Any,  # np.ndarray
    region_size: float,
    particle_radius: float,
    max_attempts: int = 5000
) -> Any:  # np.ndarray
    """
    Generate non-overlapping positions within a cubic region.
    
    Args:
        n_particles: Number of particles to place
        center: Center of the region (3D array)
        region_size: Size of cubic region (full width, particles placed in [-region_size/2, +region_size/2] around center)
        particle_radius: Radius of each particle
        max_attempts: Maximum attempts to place each particle
        
    Returns:
        Array of shape (n_particles, 3) with particle positions
        
    Raises:
        RuntimeError: If unable to place all particles
        ImportError: If numpy is not available
    """
    if not HAS_DEPENDENCIES:
        raise ImportError("Visualization requires numpy and matplotlib. Install with: pip install -e '.[visualization]'")
    
    positions = []
    min_distance = 2.2 * particle_radius  # Slightly more than touching
    
    for i in range(n_particles):
        placed = False
        
        for attempt in range(max_attempts):
            # Random position in region around center
            offset = np.random.uniform(-region_size/2, region_size/2, 3)
            pos = center + offset
            
            # Check distance to all existing particles
            if len(positions) == 0:
                positions.append(pos)
                placed = True
                break
            
            distances = np.linalg.norm(np.array(positions) - pos, axis=1)
            if np.all(distances >= min_distance):
                positions.append(pos)
                placed = True
                break
        
        if not placed:
            # If can't place with no overlap, place anyway with best effort
            if len(positions) > 0:
                # Find farthest point from existing particles
                best_pos = center
                best_min_dist = 0
                for _ in range(100):
                    test_pos = center + np.random.uniform(-region_size/2, region_size/2, 3)
                    distances = np.linalg.norm(np.array(positions) - test_pos, axis=1)
                    min_dist = np.min(distances)
                    if min_dist > best_min_dist:
                        best_pos = test_pos
                        best_min_dist = min_dist
                positions.append(best_pos)
            else:
                positions.append(center)
    
    return np.array(positions)


def layout_chain_vertical(
    cluster: set[int],
    all_particles_dict: dict[int, 'Particle'],
    center: Any,  # np.ndarray
    particle_radius: float
) -> Any:  # np.ndarray
    """
    Position chain particles vertically along Z-axis.
    
    Chains are linear structures - this places them stacked vertically
    with proper spacing.
    
    Args:
        cluster: Set of particle IDs in the chain
        all_particles_dict: Dictionary of all particles
        center: Center position for the chain
        particle_radius: Radius of particles
        
    Returns:
        Array of positions for particles in the chain
    """
    if not HAS_DEPENDENCIES:
        raise ImportError("Visualization requires numpy and matplotlib. Install with: pip install -e '.[visualization]'")
    
    from collections import deque
    
    cluster_list = list(cluster)
    n_particles = len(cluster_list)
    
    if n_particles == 0:
        return np.array([]).reshape(0, 3)
    
    # For a chain, find the endpoints (particles with only 1 link within the cluster)
    endpoints = []
    for particle_id in cluster_list:
        particle = all_particles_dict[particle_id]
        # Count links that connect to particles within this cluster
        cluster_links = [link for link in particle.links.values() if link[0] in cluster]
        if len(cluster_links) == 1:
            endpoints.append(particle_id)
    
    # Order particles from one end to the other using BFS
    if len(endpoints) >= 1:
        start_id = endpoints[0]
    else:
        # Shouldn't happen for a chain, but fallback
        start_id = cluster_list[0]
    
    # BFS to order particles
    ordered_ids = []
    visited = set()
    queue = deque([start_id])
    visited.add(start_id)
    
    while queue:
        current_id = queue.popleft()
        ordered_ids.append(current_id)
        current_particle = all_particles_dict[current_id]
        
        for patch_id, link in current_particle.links.items():
            other_id, other_patch = link
            if other_id not in visited and other_id in cluster:
                visited.add(other_id)
                queue.append(other_id)
    
    # Any particles not visited (shouldn't happen)
    for particle_id in cluster_list:
        if particle_id not in ordered_ids:
            ordered_ids.append(particle_id)
    
    # Place particles vertically along Z-axis
    spacing = 2.5 * particle_radius  # Slightly more than touching
    total_height = spacing * (n_particles - 1)
    start_z = center[2] - total_height / 2  # Center the chain vertically
    
    # Create mapping from particle_id to position
    position_map = {}
    for i, particle_id in enumerate(ordered_ids):
        position_map[particle_id] = np.array([
            center[0],  # Same X (centered)
            center[1],  # Same Y (centered)
            start_z + i * spacing  # Stacked along Z
        ])
    
    # Return positions in the same order as cluster_list (which equals cluster)
    positions = np.zeros((n_particles, 3))
    for i, particle_id in enumerate(cluster_list):
        positions[i] = position_map[particle_id]
    
    return positions


def layout_particles_geometric(simulation: 'Simulation', particle_radius: float = 0.3) -> Any:
    """
    Position particles with cluster-aware random layout.
    
    Algorithm:
    1. Find all clusters
    2. Assign each cluster to a region in 3D space (grid layout)
    3. Within each region, randomly place particles with no overlap
    4. Particles in same cluster stay together, clusters are separated
    
    Args:
        simulation: Simulation object
        particle_radius: Radius of particles
        
    Returns:
        Array of shape (n_particles, 3) with particle positions
        
    Raises:
        ImportError: If numpy is not available
    """
    if not HAS_DEPENDENCIES:
        raise ImportError("Visualization requires numpy and matplotlib. Install with: pip install -e '.[visualization]'")
    
    from .clusters import find_clusters
    
    all_particles_dict = simulation.get_all_particles()
    all_particles_list = list(all_particles_dict.values())
    n_particles = len(all_particles_list)
    
    if n_particles == 0:
        return np.array([]).reshape(0, 3)
    
    # Find all clusters
    clusters = find_clusters(all_particles_dict)
    n_clusters = len(clusters)
    
    # Calculate grid dimensions for clusters (3D grid)
    # Add 1 to provide extra space and prevent edge clusters from being too close
    n_per_side = int(np.ceil(n_clusters ** (1/3))) + 1
    cluster_spacing = 100.0  # 5x farther apart (was 20.0)
    cluster_region_size = 10.0  # More room within each cluster region
    
    # Map particle_id to position
    positions_dict = {}
    
    for cluster_idx, cluster in enumerate(clusters):
        # Calculate cluster center in 3D grid
        grid_x = (cluster_idx % n_per_side)
        grid_y = ((cluster_idx // n_per_side) % n_per_side)
        grid_z = (cluster_idx // (n_per_side * n_per_side))
        
        cluster_center = np.array([
            grid_x * cluster_spacing,
            grid_y * cluster_spacing,
            grid_z * cluster_spacing
        ])
        
        # Get particles in this cluster
        cluster_particle_ids = list(cluster)
        n_cluster_particles = len(cluster_particle_ids)
        
        # Determine cluster type
        if n_cluster_particles == 1:
            cluster_type = 'Single'
        else:
            from .clusters import classify_cluster
            cluster_type = classify_cluster(cluster, all_particles_dict)
        
        # Generate positions based on cluster type
        if n_cluster_particles == 1:
            # Single particle at cluster center
            cluster_positions = np.array([cluster_center])
        elif cluster_type == 'Chain':
            # Chains extend vertically along Z-axis
            cluster_positions = layout_chain_vertical(
                cluster=cluster,
                all_particles_dict=all_particles_dict,
                center=cluster_center,
                particle_radius=particle_radius
            )
        else:
            # Aggregates use random placement
            cluster_positions = generate_non_overlapping_positions_in_region(
                n_particles=n_cluster_particles,
                center=cluster_center,
                region_size=cluster_region_size,
                particle_radius=particle_radius,
                max_attempts=5000
            )
        
        # Assign positions
        for i, particle_id in enumerate(cluster_particle_ids):
            positions_dict[particle_id] = cluster_positions[i]
    
    # Convert to array in particle order
    particle_id_to_idx = {p.particle_id: i for i, p in enumerate(all_particles_list)}
    position_array = np.zeros((n_particles, 3))
    
    for particle in all_particles_list:
        idx = particle_id_to_idx[particle.particle_id]
        if particle.particle_id in positions_dict:
            position_array[idx] = positions_dict[particle.particle_id]
        else:
            # Fallback (shouldn't happen)
            position_array[idx] = np.random.randn(3) * 10
    
    return position_array


def add_patch_direction_markers(
    ax: Any,
    positions: Any,
    all_particles: list['Particle'],
    particle_radius: float,
    n_patches: int
) -> None:
    """
    Add visual markers showing North/South patch orientations.
    
    Draws colored arrows from particle centers pointing to North (red) and South (blue) patches.
    This makes it easy to see how particles are oriented in chains vs aggregates.
    
    Args:
        ax: Matplotlib 3D axis
        positions: Array of particle positions
        all_particles: List of Particle objects
        particle_radius: Radius of particles
        n_patches: Number of patches per particle
        
    Raises:
        ImportError: If numpy is not available
    """
    if not HAS_DEPENDENCIES:
        raise ImportError("Visualization requires numpy and matplotlib. Install with: pip install -e '.[visualization]'")
    
    particle_id_to_idx = {p.particle_id: i for i, p in enumerate(all_particles)}
    
    for particle in all_particles:
        idx = particle_id_to_idx[particle.particle_id]
        center = positions[idx]
        
        # Draw North patch (red arrow pointing in +Z initially)
        north_dir = get_patch_direction(0, n_patches)
        ax.quiver(
            center[0], center[1], center[2],
            north_dir[0], north_dir[1], north_dir[2],
            length=particle_radius * ARROW_LENGTH_MULTIPLIER,
            color='red',
            arrow_length_ratio=ARROW_LENGTH_RATIO,
            linewidth=2,
            alpha=0.6
        )
        
        # Draw South patch (blue arrow pointing in -Z initially)
        south_dir = get_patch_direction(1, n_patches)
        ax.quiver(
            center[0], center[1], center[2],
            south_dir[0], south_dir[1], south_dir[2],
            length=particle_radius * ARROW_LENGTH_MULTIPLIER,
            color='blue',
            arrow_length_ratio=ARROW_LENGTH_RATIO,
            linewidth=2,
            alpha=0.6
        )


def generate_non_overlapping_positions(
    n_particles: int,
    box_size: float = 10.0,
    particle_radius: float = 0.3,
    max_attempts: int = 10000
) -> Any:  # Would be np.ndarray if numpy available
    """
    Generate non-overlapping positions for particles in 3D space.
    
    Args:
        n_particles: Number of particles to place
        box_size: Size of the cubic box (centered at origin)
        particle_radius: Radius of each particle
        max_attempts: Maximum attempts to place each particle
        
    Returns:
        Array of shape (n_particles, 3) with particle positions
        
    Raises:
        RuntimeError: If unable to place all particles
        ImportError: If numpy is not available
    """
    if not HAS_DEPENDENCIES:
        raise ImportError("Visualization requires numpy and matplotlib. Install with: pip install -e '.[visualization]'")
    
    positions = []
    min_distance = 2 * particle_radius
    
    for i in range(n_particles):
        placed = False
        
        for attempt in range(max_attempts):
            # Random position in box
            pos = np.random.uniform(-box_size/2, box_size/2, 3)
            
            # Check distance to all existing particles
            if len(positions) == 0:
                positions.append(pos)
                placed = True
                break
            
            distances = np.linalg.norm(np.array(positions) - pos, axis=1)
            if np.all(distances >= min_distance):
                positions.append(pos)
                placed = True
                break
        
        if not placed:
            raise RuntimeError(
                f"Could not place particle {i+1}/{n_particles} after {max_attempts} attempts. "
                f"Try increasing box_size or reducing particle_radius."
            )
    
    return np.array(positions)


def visualize_system_3d(
    simulation: 'Simulation',
    positions: Optional[Any] = None,  # Would be np.ndarray if numpy available
    title: str = "Magnetic Nanoparticle System",
    box_size: float = 10.0,
    particle_radius: float = 0.3,
    save_path: Optional[Path] = None,
    show_stats: bool = True,
    show_patch_directions: bool = True,
    use_geometric_layout: bool = True,
) -> None:
    """
    Create 3D visualization of the particle system.
    
    Particles are color-coded:
    - Single particles: Brown spheres with red dots for North/South patches
    - Chains: Green spheres
    - Aggregates: Red spheres
    
    Args:
        simulation: Simulation object with current state
        positions: Array of particle positions (n_particles, 3). If None, generates positions.
        title: Plot title
        box_size: Size of the visualization box
        particle_radius: Radius for drawing particles
        save_path: If provided, saves figure to this path
        show_stats: If True, display statistics on the plot
        show_patch_directions: If True, show North/South patch direction markers
        use_geometric_layout: If True, use geometric layout based on patch connections
        
    Raises:
        ImportError: If visualization dependencies are not available
    """
    if not HAS_DEPENDENCIES:
        raise ImportError("Visualization requires numpy and matplotlib. Install with: pip install -e '.[visualization]'")
    
    from .clusters import find_clusters, classify_cluster
    
    # Get all particles
    all_particles = simulation.get_all_particles()
    n_particles = len(all_particles)
    
    # Generate positions if not provided
    if positions is None:
        if use_geometric_layout:
            positions = layout_particles_geometric(simulation, particle_radius)
            # Adjust box_size to fit geometric layout
            if len(positions) > 0:
                # Find bounding box of positions
                min_coords = np.min(positions, axis=0)
                max_coords = np.max(positions, axis=0)
                ranges = max_coords - min_coords
                # Add padding (50% on each side)
                box_size = max(MIN_BOX_SIZE, np.max(ranges) * BOX_PADDING_MULTIPLIER)
        else:
            # Fall back to random positions
            positions = generate_non_overlapping_positions(n_particles, box_size, particle_radius)
    
    # Find clusters and classify
    clusters = find_clusters(all_particles)
    
    # Create a mapping from particle_id to cluster type
    particle_cluster_type = {}
    for cluster in clusters:
        if len(cluster) == 1:
            cluster_type = 'Single'
        else:
            cluster_type = classify_cluster(cluster, all_particles)
        
        for particle_id in cluster:
            particle_cluster_type[particle_id] = cluster_type
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define colors
    color_map = {
        'Single': '#8B4513',  # Brown (keep as is)
        'Chain': '#228B22',   # Green (keep as is)
        'Aggregate': '#000000'  # Black (changed from red)
    }
    
    # Plot particles
    particle_ids = sorted(all_particles.keys())
    for i, particle_id in enumerate(particle_ids):
        particle = all_particles[particle_id]
        cluster_type = particle_cluster_type[particle_id]
        color = color_map[cluster_type]
        
        pos = positions[i]
        
        # Draw particle as sphere
        ax.scatter(pos[0], pos[1], pos[2], 
                  c=color, s=200, alpha=0.7, edgecolors='black', linewidth=1)
        
        # For single particles, add red dots for North/South patches
        if cluster_type == 'Single':
            # North patch (top)
            north_pos = pos + np.array([0, 0, particle_radius])
            ax.scatter(north_pos[0], north_pos[1], north_pos[2],
                      c='red', s=30, marker='o', alpha=1.0)
            
            # South patch (bottom)
            south_pos = pos + np.array([0, 0, -particle_radius])
            ax.scatter(south_pos[0], south_pos[1], south_pos[2],
                      c='red', s=30, marker='o', alpha=1.0)
    
    # Draw links between connected particles
    for i, particle_id in enumerate(particle_ids):
        particle = all_particles[particle_id]
        
        for patch_id, (other_particle_id, other_patch_id) in particle.links.items():
            # Only draw each link once (from lower ID to higher ID)
            if particle_id < other_particle_id:
                # Find position of other particle
                other_idx = particle_ids.index(other_particle_id)
                
                # Draw line
                line = Line3D(
                    [positions[i][0], positions[other_idx][0]],
                    [positions[i][1], positions[other_idx][1]],
                    [positions[i][2], positions[other_idx][2]],
                    color='black', linewidth=2, alpha=0.6
                )
                ax.add_line(line)
    
    # Add patch direction markers
    if show_patch_directions:
        all_particles_list = [all_particles[pid] for pid in particle_ids]
        add_patch_direction_markers(
            ax, positions, all_particles_list, 
            particle_radius, simulation.params.n_patches
        )
    
    # Set labels and title
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Set equal aspect ratio and limits
    ax.set_xlim(-box_size/2, box_size/2)
    ax.set_ylim(-box_size/2, box_size/2)
    ax.set_zlim(-box_size/2, box_size/2)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_map['Single'], edgecolor='black', label='Single'),
        Patch(facecolor=color_map['Chain'], edgecolor='black', label='Chain'),
        Patch(facecolor=color_map['Aggregate'], edgecolor='black', label='Aggregate'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Add statistics text
    if show_stats:
        from .antigen import AntigenState
        
        # Count cluster types
        n_single = sum(1 for t in particle_cluster_type.values() if t == 'Single')
        n_chains = len([c for c in clusters if len(c) > 1 and classify_cluster(c, all_particles) == 'Chain'])
        n_aggregates = len([c for c in clusters if len(c) > 1 and classify_cluster(c, all_particles) == 'Aggregate'])
        
        # Count antigens
        n_free = sum(1 for a in simulation.antigens if a.state == AntigenState.FREE)
        n_sandwich = sum(1 for a in simulation.antigens if a.state == AntigenState.SANDWICH)
        
        stats_text = (
            f"Time: {simulation.current_time:.2f} s\n"
            f"Step: {simulation.current_step}\n"
            f"Field: {'ON' if simulation.field_on else 'OFF'}\n"
            f"\n"
            f"Particles:\n"
            f"  Single: {n_single}\n"
            f"  Chains: {n_chains}\n"
            f"  Aggregates: {n_aggregates}\n"
            f"\n"
            f"Antigens:\n"
            f"  Free: {n_free}\n"
            f"  Sandwiches: {n_sandwich}"
        )
        
        # Add text box
        ax.text2D(0.02, 0.98, stats_text,
                 transform=ax.transAxes,
                 fontsize=9,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    plt.close(fig)


def create_cycle_snapshots(
    simulation: 'Simulation',
    output_dir: Path,
    box_size: float = 10.0,
    particle_radius: float = 0.3,
) -> list[Path]:
    """
    Create snapshots at the beginning and end of each field cycle.
    
    Args:
        simulation: Simulation object to run and visualize
        output_dir: Directory to save snapshot images
        box_size: Size of the visualization box
        particle_radius: Radius for drawing particles
        
    Returns:
        List of paths to saved snapshot images
        
    Raises:
        ImportError: If visualization dependencies are not available
    """
    if not HAS_DEPENDENCIES:
        raise ImportError("Visualization requires numpy and matplotlib. Install with: pip install -e '.[visualization]'")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate fixed positions for all particles (consistent across snapshots)
    n_particles = len(simulation.get_all_particles())
    positions = generate_non_overlapping_positions(n_particles, box_size, particle_radius)
    
    saved_paths = []
    
    # Calculate cycle boundaries
    cycle_length = simulation.params.n_steps_on + simulation.params.n_steps_off
    total_steps = simulation.params.n_repeats * cycle_length
    
    # Create initial snapshot
    snapshot_path = output_dir / f"snapshot_step_{simulation.current_step:06d}.png"
    visualize_system_3d(
        simulation,
        positions=positions,
        title=f"Initial State (Step {simulation.current_step})",
        box_size=box_size,
        particle_radius=particle_radius,
        save_path=snapshot_path,
        show_stats=True,
    )
    saved_paths.append(snapshot_path)
    
    # Run simulation and create snapshots at cycle boundaries
    for cycle in range(simulation.params.n_repeats):
        # End of ON phase
        on_end_step = (cycle * cycle_length) + simulation.params.n_steps_on
        steps_to_run = on_end_step - simulation.current_step
        
        if steps_to_run > 0:
            simulation.run(steps_to_run)
            
            snapshot_path = output_dir / f"snapshot_step_{simulation.current_step:06d}.png"
            visualize_system_3d(
                simulation,
                positions=positions,
                title=f"Cycle {cycle+1} - End of ON Phase (Step {simulation.current_step})",
                box_size=box_size,
                particle_radius=particle_radius,
                save_path=snapshot_path,
                show_stats=True,
            )
            saved_paths.append(snapshot_path)
        
        # End of OFF phase (end of cycle)
        off_end_step = (cycle + 1) * cycle_length
        steps_to_run = off_end_step - simulation.current_step
        
        if steps_to_run > 0:
            simulation.run(steps_to_run)
            
            snapshot_path = output_dir / f"snapshot_step_{simulation.current_step:06d}.png"
            visualize_system_3d(
                simulation,
                positions=positions,
                title=f"Cycle {cycle+1} - End of OFF Phase (Step {simulation.current_step})",
                box_size=box_size,
                particle_radius=particle_radius,
                save_path=snapshot_path,
                show_stats=True,
            )
            saved_paths.append(snapshot_path)
    
    print(f"\nCreated {len(saved_paths)} snapshots in: {output_dir}")
    return saved_paths


def create_animation(
    snapshot_paths: list[Path],
    output_path: Path,
    fps: int = 2,
) -> None:
    """
    Create an MP4 animation from snapshot images.
    
    Requires ffmpeg to be installed on the system.
    
    Args:
        snapshot_paths: List of paths to snapshot images (in order)
        output_path: Path to save the MP4 animation
        fps: Frames per second for the animation
        
    Raises:
        ImportError: If visualization dependencies are not available
        RuntimeError: If ffmpeg is not available
    """
    if not HAS_DEPENDENCIES:
        raise ImportError("Visualization requires numpy and matplotlib. Install with: pip install -e '.[visualization]'")
    
    import subprocess
    import shutil
    
    # Check if ffmpeg is available
    if not shutil.which('ffmpeg'):
        raise RuntimeError(
            "ffmpeg not found. Please install ffmpeg to create animations.\n"
            "On Ubuntu/Debian: sudo apt-get install ffmpeg\n"
            "On macOS: brew install ffmpeg"
        )
    
    # Create a temporary directory with symlinks to images in order
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create symlinks with sequential names
        for i, img_path in enumerate(snapshot_paths):
            link_path = tmpdir / f"frame_{i:04d}.png"
            link_path.symlink_to(img_path.absolute())
        
        # Run ffmpeg
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file
            '-framerate', str(fps),
            '-i', str(tmpdir / 'frame_%04d.png'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '23',
            str(output_path)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"\nCreated animation: {output_path}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg failed: {e.stderr.decode()}")
