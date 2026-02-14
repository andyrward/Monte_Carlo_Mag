"""
Particle class for magnetic nanoparticles with antibody patches.
"""

from typing import Optional


class Particle:
    """
    Represents a magnetic nanoparticle with antibody-coated patches.
    
    Attributes:
        particle_id: Unique integer identifier
        particle_type: 'A' or 'B'
        n_patches: Total number of patches (≤30)
        patches: Dict mapping patch_id to antigen_id (None if unbound)
        links: Dict mapping patch_id to tuple of (other_particle_id, other_patch_id)
        
    Constants:
        North patch is always index 0
        South patch is always index 1
        Regular patches are indices 2 to n_patches-1
    """
    
    def __init__(self, particle_id: int, particle_type: str, n_patches: int):
        """
        Initialize a Particle.
        
        Args:
            particle_id: Unique identifier
            particle_type: 'A' or 'B'
            n_patches: Total number of patches (must be ≤30)
        """
        if particle_type not in ['A', 'B']:
            raise ValueError(f"particle_type must be 'A' or 'B', got {particle_type}")
        if n_patches > 30 or n_patches < 2:
            raise ValueError(f"n_patches must be between 2 and 30, got {n_patches}")
        
        self.particle_id = particle_id
        self.particle_type = particle_type
        self.n_patches = n_patches
        self.patches: dict[int, Optional[int]] = {i: None for i in range(n_patches)}
        self.links: dict[int, tuple[int, int]] = {}
    
    def is_north_patch(self, patch_id: int) -> bool:
        """Check if patch is the North patch (index 0)."""
        return patch_id == 0
    
    def is_south_patch(self, patch_id: int) -> bool:
        """Check if patch is the South patch (index 1)."""
        return patch_id == 1
    
    def is_north_or_south_patch(self, patch_id: int) -> bool:
        """Check if patch is North or South (index 0 or 1)."""
        return patch_id == 0 or patch_id == 1
    
    def bind_antigen(self, patch_id: int, antigen_id: int) -> None:
        """
        Bind an antigen to a patch.
        
        Args:
            patch_id: Patch identifier
            antigen_id: Antigen identifier
        """
        if patch_id not in self.patches:
            raise ValueError(f"Invalid patch_id {patch_id} for particle with {self.n_patches} patches")
        if self.patches[patch_id] is not None:
            raise ValueError(f"Patch {patch_id} already has antigen {self.patches[patch_id]}")
        self.patches[patch_id] = antigen_id
    
    def unbind_antigen(self, patch_id: int) -> Optional[int]:
        """
        Remove antigen from a patch.
        
        Args:
            patch_id: Patch identifier
            
        Returns:
            The antigen_id that was removed, or None if patch was empty
        """
        if patch_id not in self.patches:
            raise ValueError(f"Invalid patch_id {patch_id} for particle with {self.n_patches} patches")
        antigen_id = self.patches[patch_id]
        self.patches[patch_id] = None
        return antigen_id
    
    def add_link(self, my_patch_id: int, other_particle_id: int, other_patch_id: int) -> None:
        """
        Add a particle-particle link through a sandwich complex.
        
        Args:
            my_patch_id: Patch on this particle
            other_particle_id: ID of the other particle
            other_patch_id: Patch on the other particle
        """
        if my_patch_id not in self.patches:
            raise ValueError(f"Invalid patch_id {my_patch_id} for particle with {self.n_patches} patches")
        if my_patch_id in self.links:
            raise ValueError(f"Patch {my_patch_id} already has a link")
        self.links[my_patch_id] = (other_particle_id, other_patch_id)
    
    def remove_link(self, my_patch_id: int) -> Optional[tuple[int, int]]:
        """
        Remove a link from this patch.
        
        Args:
            my_patch_id: Patch identifier
            
        Returns:
            The (other_particle_id, other_patch_id) tuple that was removed, or None
        """
        if my_patch_id not in self.patches:
            raise ValueError(f"Invalid patch_id {my_patch_id} for particle with {self.n_patches} patches")
        return self.links.pop(my_patch_id, None)
