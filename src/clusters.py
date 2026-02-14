"""
Cluster detection and classification for particle networks.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .particle import Particle


def find_clusters(particles: dict[int, 'Particle']) -> list[set[int]]:
    """
    Find connected components (clusters) in the particle network.
    
    Uses breadth-first search (BFS) to identify all connected particles
    through their links (sandwich complexes).
    
    Args:
        particles: Dictionary mapping particle_id to Particle objects
        
    Returns:
        List of clusters, where each cluster is a set of particle_ids
    """
    visited = set()
    clusters = []
    
    for particle_id in particles:
        if particle_id in visited:
            continue
        
        # Start a new cluster with BFS
        cluster = set()
        queue = [particle_id]
        
        while queue:
            current_id = queue.pop(0)
            if current_id in visited:
                continue
            
            visited.add(current_id)
            cluster.add(current_id)
            
            # Find all neighbors through links
            current_particle = particles[current_id]
            for patch_id, (other_particle_id, other_patch_id) in current_particle.links.items():
                if other_particle_id not in visited:
                    queue.append(other_particle_id)
        
        if cluster:
            clusters.append(cluster)
    
    return clusters


def classify_cluster(cluster: set[int], particles: dict[int, 'Particle']) -> str:
    """
    Classify a cluster as 'Chain' or 'Aggregate'.
    
    Chain: ALL links are North-South (patch 0 or 1 on both sides)
    Aggregate: At least ONE link is NOT North-South
    
    Args:
        cluster: Set of particle_ids in the cluster
        particles: Dictionary mapping particle_id to Particle objects
        
    Returns:
        'Chain' or 'Aggregate'
    """
    # Check all links in the cluster
    for particle_id in cluster:
        particle = particles[particle_id]
        
        for my_patch_id, (other_particle_id, other_patch_id) in particle.links.items():
            # Check if both patches are North or South
            if not particle.is_north_or_south_patch(my_patch_id):
                return 'Aggregate'
            
            # Also check the other particle's patch
            if other_particle_id in particles:
                other_particle = particles[other_particle_id]
                if not other_particle.is_north_or_south_patch(other_patch_id):
                    return 'Aggregate'
    
    return 'Chain'
