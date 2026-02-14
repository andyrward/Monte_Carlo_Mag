"""
Antigen class for tracking binding states.
"""

from typing import Optional
from enum import Enum


class AntigenState(Enum):
    """Possible states for an antigen."""
    FREE = "Free"
    BOUND_A = "Bound_A"
    BOUND_B = "Bound_B"
    SANDWICH = "Sandwich"


class Antigen:
    """
    Represents an antigen molecule that can bind to type A and B particles.
    
    Attributes:
        antigen_id: Unique integer identifier
        state: Current binding state (Free, Bound_A, Bound_B, or Sandwich)
        binding_a: Tuple of (particle_id, patch_id) or None
        binding_b: Tuple of (particle_id, patch_id) or None
    """
    
    def __init__(self, antigen_id: int):
        """
        Initialize an Antigen.
        
        Args:
            antigen_id: Unique identifier
        """
        self.antigen_id = antigen_id
        self.state = AntigenState.FREE
        self.binding_a: Optional[tuple[int, int]] = None
        self.binding_b: Optional[tuple[int, int]] = None
    
    def bind_to_a(self, particle_id: int, patch_id: int) -> None:
        """
        Bind to a type A particle.
        
        Args:
            particle_id: ID of the type A particle
            patch_id: Patch ID on the particle
        """
        if self.binding_a is not None:
            raise ValueError(f"Antigen {self.antigen_id} already bound to A particle")
        self.binding_a = (particle_id, patch_id)
        self.update_state()
    
    def bind_to_b(self, particle_id: int, patch_id: int) -> None:
        """
        Bind to a type B particle.
        
        Args:
            particle_id: ID of the type B particle
            patch_id: Patch ID on the particle
        """
        if self.binding_b is not None:
            raise ValueError(f"Antigen {self.antigen_id} already bound to B particle")
        self.binding_b = (particle_id, patch_id)
        self.update_state()
    
    def unbind_from_a(self) -> Optional[tuple[int, int]]:
        """
        Unbind from type A particle.
        
        Returns:
            The (particle_id, patch_id) tuple that was removed, or None
        """
        binding = self.binding_a
        self.binding_a = None
        self.update_state()
        return binding
    
    def unbind_from_b(self) -> Optional[tuple[int, int]]:
        """
        Unbind from type B particle.
        
        Returns:
            The (particle_id, patch_id) tuple that was removed, or None
        """
        binding = self.binding_b
        self.binding_b = None
        self.update_state()
        return binding
    
    def update_state(self) -> None:
        """Update state based on current bindings."""
        if self.binding_a is not None and self.binding_b is not None:
            self.state = AntigenState.SANDWICH
        elif self.binding_a is not None:
            self.state = AntigenState.BOUND_A
        elif self.binding_b is not None:
            self.state = AntigenState.BOUND_B
        else:
            self.state = AntigenState.FREE
    
    def is_on_north_or_south_patch(self, particles: dict[int, 'Particle']) -> bool:
        """
        Check if antigen is bound to a North or South patch.
        
        Args:
            particles: Dictionary mapping particle_id to Particle objects
            
        Returns:
            True if bound to at least one North or South patch
        """
        if self.binding_a is not None:
            particle_id, patch_id = self.binding_a
            if particle_id in particles:
                if particles[particle_id].is_north_or_south_patch(patch_id):
                    return True
        
        if self.binding_b is not None:
            particle_id, patch_id = self.binding_b
            if particle_id in particles:
                if particles[particle_id].is_north_or_south_patch(patch_id):
                    return True
        
        return False
