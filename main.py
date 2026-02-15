"""
Driver script for kinetic Monte Carlo simulation.
"""

import sys
import argparse
import yaml
from pathlib import Path

from src import (
    SimulationParameters, 
    Simulation, 
    find_clusters, 
    classify_cluster, 
    AntigenState,
    _HAS_VISUALIZATION
)

if _HAS_VISUALIZATION:
    from src import visualize_system_3d, create_cycle_snapshots, create_animation, plot_cluster_size_distributions


def load_parameters(config_file: str) -> SimulationParameters:
    """
    Load simulation parameters from YAML config file.
    
    Args:
        config_file: Path to YAML configuration file
        
    Returns:
        SimulationParameters object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_file}\n"
            f"Please ensure the config file exists or specify an alternative path."
        )
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return SimulationParameters(**config)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Kinetic Monte Carlo simulation of magnetic nanoparticles'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration YAML file (default: config/default_params.yaml)'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate 3D visualization of initial and final states'
    )
    
    parser.add_argument(
        '--snapshots',
        action='store_true',
        help='Create snapshots at each cycle boundary'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Directory for visualization outputs (default: output)'
    )
    
    parser.add_argument(
        '--animation',
        action='store_true',
        help='Create MP4 animation from snapshots (requires ffmpeg)'
    )
    
    return parser.parse_args()


def main():
    """Run the simulation and print statistics."""
    args = parse_args()
    
    # Check visualization dependencies if needed
    if (args.visualize or args.snapshots or args.animation) and not _HAS_VISUALIZATION:
        print("\nError: Visualization features require optional dependencies.", file=sys.stderr)
        print("Install with: pip install -e '.[visualization]'", file=sys.stderr)
        sys.exit(1)
    
    print("=" * 60)
    print("Kinetic Monte Carlo Simulation of Magnetic Nanoparticles")
    print("=" * 60)
    
    # Load parameters
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = Path(__file__).parent / "config" / "default_params.yaml"
    
    print(f"\nLoading parameters from: {config_path}")
    
    try:
        params = load_parameters(str(config_path))
    except FileNotFoundError as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nError loading parameters: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Print simulation setup
    print("\n" + "-" * 60)
    print("Simulation Setup:")
    print("-" * 60)
    print(f"Particle A count: {params.N_A_sim}")
    print(f"Particle B count: {params.N_B_sim}")
    print(f"Antigen count: {params.N_antigen_sim}")
    print(f"Patches per particle: {params.n_patches}")
    print(f"Simulation volume: {params.V_box:.2e} L")
    print(f"Time step: {params.dt} s")
    print(f"Steps ON: {params.n_steps_on}")
    print(f"Steps OFF: {params.n_steps_off}")
    print(f"Cycles: {params.n_repeats}")
    print(f"Field restrictions: {'Enabled' if params.restrict_aggregates_field_on else 'Disabled'}")
    
    # Initialize simulation
    print("\n" + "-" * 60)
    print("Initializing simulation...")
    print("-" * 60)
    sim = Simulation(params)
    
    # Create output directory if needed
    if args.visualize or args.snapshots:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initial visualization
    if args.visualize:
        print("\nGenerating initial state visualization...")
        initial_path = output_dir / "initial_state.png"
        visualize_system_3d(
            sim,
            title="Initial State",
            save_path=initial_path,
        )
    
    # Run with snapshots or regular run
    total_steps = params.n_repeats * (params.n_steps_on + params.n_steps_off)
    
    if args.snapshots:
        print(f"\nRunning {total_steps} steps with cycle snapshots...")
        snapshot_paths = create_cycle_snapshots(
            sim,
            output_dir=output_dir,
        )
        
        # Create animation if requested
        if args.animation:
            print("\nCreating animation...")
            animation_path = output_dir / "simulation_animation.mp4"
            try:
                create_animation(
                    snapshot_paths,
                    output_path=animation_path,
                    fps=2,
                )
            except RuntimeError as e:
                print(f"\nWarning: Could not create animation: {e}", file=sys.stderr)
    else:
        print(f"\nRunning {total_steps} steps...")
        sim.run(total_steps)
    
    # Final visualization
    if args.visualize:
        print("\nGenerating final state visualization...")
        final_path = output_dir / "final_state.png"
        visualize_system_3d(
            sim,
            title="Final State",
            save_path=final_path,
        )
        
        # Generate cluster size distribution plot
        distribution_path = output_dir / "cluster_size_distributions.png"
        print("Generating cluster size distribution plot...")
        plot_cluster_size_distributions(
            sim,
            save_path=distribution_path,
            title=f"Cluster Size Distributions (t={sim.current_time:.3f})"
        )
        print(f"Size distribution plot saved to {distribution_path}")
    
    # Print final statistics
    print("\n" + "=" * 60)
    print("Final Statistics:")
    print("=" * 60)
    
    # Count antigen states
    n_free = sum(1 for a in sim.antigens if a.state == AntigenState.FREE)
    n_bound_a = sum(1 for a in sim.antigens if a.state == AntigenState.BOUND_A)
    n_bound_b = sum(1 for a in sim.antigens if a.state == AntigenState.BOUND_B)
    n_sandwich = sum(1 for a in sim.antigens if a.state == AntigenState.SANDWICH)
    
    print(f"\nAntigen States:")
    print(f"  Free: {n_free}")
    print(f"  Bound to A: {n_bound_a}")
    print(f"  Bound to B: {n_bound_b}")
    print(f"  Sandwich: {n_sandwich}")
    
    # Find and classify clusters using public method
    all_particles = sim.get_all_particles()
    clusters = find_clusters(all_particles)
    
    # Count chains and aggregates
    n_chains = 0
    n_aggregates = 0
    
    for cluster in clusters:
        if len(cluster) > 1:  # Only count multi-particle clusters
            cluster_type = classify_cluster(cluster, all_particles)
            if cluster_type == 'Chain':
                n_chains += 1
            else:
                n_aggregates += 1
    
    print(f"\nClusters:")
    print(f"  Total clusters: {len(clusters)}")
    print(f"  Chains (North-South only): {n_chains}")
    print(f"  Aggregates (with regular patches): {n_aggregates}")
    print(f"  Single particles: {len([c for c in clusters if len(c) == 1])}")
    
    # Find largest cluster
    if clusters:
        largest_cluster = max(clusters, key=len)
        print(f"  Largest cluster size: {len(largest_cluster)} particles")
    
    print("\n" + "=" * 60)
    print("Simulation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
