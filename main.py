"""
Driver script for kinetic Monte Carlo simulation.
"""

import sys
import yaml
from pathlib import Path

from src import SimulationParameters, Simulation, find_clusters, classify_cluster, AntigenState


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


def main():
    """Run the simulation and print statistics."""
    print("=" * 60)
    print("Kinetic Monte Carlo Simulation of Magnetic Nanoparticles")
    print("=" * 60)
    
    # Load parameters
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
    
    # Initialize simulation
    print("\n" + "-" * 60)
    print("Initializing simulation...")
    print("-" * 60)
    sim = Simulation(params)
    
    # Run simulation
    total_steps = params.n_repeats * (params.n_steps_on + params.n_steps_off)
    print(f"Running {total_steps} steps...")
    sim.run(total_steps)
    
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
