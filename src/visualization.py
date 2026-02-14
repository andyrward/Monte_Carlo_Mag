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
) -> None:
    """
    Create 3D visualization of the particle system.
    
    Particles are color-coded:
    - Single particles: Brown spheres with red dots for North/South patches
    - Chains: Green spheres
    - Aggregates: Red spheres
    
    Args:
        simulation: Simulation object with current state
        positions: Array of particle positions (n_particles, 3). If None, generates random positions.
        title: Plot title
        box_size: Size of the visualization box
        particle_radius: Radius for drawing particles
        save_path: If provided, saves figure to this path
        show_stats: If True, display statistics on the plot
        
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
        'Single': '#8B4513',  # Brown
        'Chain': '#228B22',   # Green
        'Aggregate': '#DC143C'  # Red
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
