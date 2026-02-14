"""
Tests for cluster detection and classification.
"""

import pytest
from src.clusters import find_clusters, classify_cluster
from src.particle import Particle


def test_find_clusters_no_particles():
    """Test finding clusters with no particles."""
    particles = {}
    clusters = find_clusters(particles)
    assert clusters == []


def test_find_clusters_single_particle():
    """Test finding clusters with single particle."""
    particle = Particle(particle_id=1, particle_type='A', n_patches=10)
    particles = {1: particle}
    
    clusters = find_clusters(particles)
    assert len(clusters) == 1
    assert clusters[0] == {1}


def test_find_clusters_two_unconnected_particles():
    """Test finding clusters with two unconnected particles."""
    particle1 = Particle(particle_id=1, particle_type='A', n_patches=10)
    particle2 = Particle(particle_id=2, particle_type='B', n_patches=10)
    particles = {1: particle1, 2: particle2}
    
    clusters = find_clusters(particles)
    assert len(clusters) == 2
    assert {1} in clusters
    assert {2} in clusters


def test_find_clusters_two_connected_particles():
    """Test finding clusters with two connected particles."""
    particle1 = Particle(particle_id=1, particle_type='A', n_patches=10)
    particle2 = Particle(particle_id=2, particle_type='B', n_patches=10)
    
    # Create link between particles
    particle1.add_link(my_patch_id=5, other_particle_id=2, other_patch_id=3)
    particle2.add_link(my_patch_id=3, other_particle_id=1, other_patch_id=5)
    
    particles = {1: particle1, 2: particle2}
    
    clusters = find_clusters(particles)
    assert len(clusters) == 1
    assert clusters[0] == {1, 2}


def test_find_clusters_chain_of_particles():
    """Test finding clusters with chain of particles."""
    particle1 = Particle(particle_id=1, particle_type='A', n_patches=10)
    particle2 = Particle(particle_id=2, particle_type='B', n_patches=10)
    particle3 = Particle(particle_id=3, particle_type='A', n_patches=10)
    
    # Create chain: 1 - 2 - 3
    particle1.add_link(my_patch_id=0, other_particle_id=2, other_patch_id=1)
    particle2.add_link(my_patch_id=1, other_particle_id=1, other_patch_id=0)
    particle2.add_link(my_patch_id=0, other_particle_id=3, other_patch_id=1)
    particle3.add_link(my_patch_id=1, other_particle_id=2, other_patch_id=0)
    
    particles = {1: particle1, 2: particle2, 3: particle3}
    
    clusters = find_clusters(particles)
    assert len(clusters) == 1
    assert clusters[0] == {1, 2, 3}


def test_find_clusters_multiple_clusters():
    """Test finding multiple separate clusters."""
    # Cluster 1: particles 1-2
    particle1 = Particle(particle_id=1, particle_type='A', n_patches=10)
    particle2 = Particle(particle_id=2, particle_type='B', n_patches=10)
    particle1.add_link(my_patch_id=0, other_particle_id=2, other_patch_id=1)
    particle2.add_link(my_patch_id=1, other_particle_id=1, other_patch_id=0)
    
    # Cluster 2: particles 3-4
    particle3 = Particle(particle_id=3, particle_type='A', n_patches=10)
    particle4 = Particle(particle_id=4, particle_type='B', n_patches=10)
    particle3.add_link(my_patch_id=0, other_particle_id=4, other_patch_id=1)
    particle4.add_link(my_patch_id=1, other_particle_id=3, other_patch_id=0)
    
    # Single particle
    particle5 = Particle(particle_id=5, particle_type='A', n_patches=10)
    
    particles = {1: particle1, 2: particle2, 3: particle3, 4: particle4, 5: particle5}
    
    clusters = find_clusters(particles)
    assert len(clusters) == 3
    assert {1, 2} in clusters
    assert {3, 4} in clusters
    assert {5} in clusters


def test_classify_cluster_chain():
    """Test classifying a chain (all North-South links)."""
    particle1 = Particle(particle_id=1, particle_type='A', n_patches=10)
    particle2 = Particle(particle_id=2, particle_type='B', n_patches=10)
    
    # Link using North (0) and South (1) patches
    particle1.add_link(my_patch_id=0, other_particle_id=2, other_patch_id=1)
    particle2.add_link(my_patch_id=1, other_particle_id=1, other_patch_id=0)
    
    particles = {1: particle1, 2: particle2}
    cluster = {1, 2}
    
    classification = classify_cluster(cluster, particles)
    assert classification == 'Chain'


def test_classify_cluster_aggregate():
    """Test classifying an aggregate (contains regular patch links)."""
    particle1 = Particle(particle_id=1, particle_type='A', n_patches=10)
    particle2 = Particle(particle_id=2, particle_type='B', n_patches=10)
    
    # Link using regular patches (not 0 or 1)
    particle1.add_link(my_patch_id=5, other_particle_id=2, other_patch_id=3)
    particle2.add_link(my_patch_id=3, other_particle_id=1, other_patch_id=5)
    
    particles = {1: particle1, 2: particle2}
    cluster = {1, 2}
    
    classification = classify_cluster(cluster, particles)
    assert classification == 'Aggregate'


def test_classify_cluster_mixed_links():
    """Test classifying cluster with both chain and aggregate links."""
    particle1 = Particle(particle_id=1, particle_type='A', n_patches=10)
    particle2 = Particle(particle_id=2, particle_type='B', n_patches=10)
    particle3 = Particle(particle_id=3, particle_type='A', n_patches=10)
    
    # Link 1-2 using North-South
    particle1.add_link(my_patch_id=0, other_particle_id=2, other_patch_id=1)
    particle2.add_link(my_patch_id=1, other_particle_id=1, other_patch_id=0)
    
    # Link 2-3 using regular patches
    particle2.add_link(my_patch_id=5, other_particle_id=3, other_patch_id=3)
    particle3.add_link(my_patch_id=3, other_particle_id=2, other_patch_id=5)
    
    particles = {1: particle1, 2: particle2, 3: particle3}
    cluster = {1, 2, 3}
    
    # Should be classified as aggregate due to regular patch links
    classification = classify_cluster(cluster, particles)
    assert classification == 'Aggregate'


def test_classify_cluster_single_particle():
    """Test classifying single particle cluster."""
    particle1 = Particle(particle_id=1, particle_type='A', n_patches=10)
    particles = {1: particle1}
    cluster = {1}
    
    # Single particle has no links, so it's a chain by default
    classification = classify_cluster(cluster, particles)
    assert classification == 'Chain'
