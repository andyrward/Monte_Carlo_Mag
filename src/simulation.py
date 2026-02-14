"""
Main simulation engine for kinetic Monte Carlo simulation.
"""

import random
from typing import Optional

from .parameters import SimulationParameters
from .particle import Particle
from .antigen import Antigen, AntigenState
from .events import (
    calc_bind_probability_field_off,
    calc_bind_probability_field_on_enhanced,
    calc_unbind_probability,
    calc_neighbor_probability,
)


class Simulation:
    """
    Kinetic Monte Carlo simulation of magnetic nanoparticles with antibody-antigen binding.
    
    Attributes:
        params: SimulationParameters object
        particles_a: List of type A Particle objects
        particles_b: List of type B Particle objects
        antigens: List of Antigen objects
        current_step: Current simulation step
        current_time: Current simulation time (seconds)
        field_on: Current field state
        history: Dictionary storing time series data
    """
    
    def __init__(self, params: SimulationParameters):
        """
        Initialize the simulation.
        
        Args:
            params: SimulationParameters object
        """
        self.params = params
        self.current_step = 0
        self.current_time = 0.0
        self.field_on = False
        
        # Initialize particles
        self.particles_a = []
        for i in range(params.N_A_sim):
            particle = Particle(
                particle_id=i,
                particle_type='A',
                n_patches=params.n_patches
            )
            self.particles_a.append(particle)
        
        self.particles_b = []
        for i in range(params.N_B_sim):
            particle = Particle(
                particle_id=params.N_A_sim + i,  # Unique IDs
                particle_type='B',
                n_patches=params.n_patches
            )
            self.particles_b.append(particle)
        
        # Initialize antigens
        self.antigens = []
        for i in range(params.N_antigen_sim):
            antigen = Antigen(antigen_id=i)
            self.antigens.append(antigen)
        
        # Create lookup dictionary for all particles
        self._all_particles = {p.particle_id: p for p in self.particles_a + self.particles_b}
        
        # Initialize history
        self.history = {
            'time': [],
            'step': [],
            'field_on': [],
            'n_free': [],
            'n_bound_a': [],
            'n_bound_b': [],
            'n_sandwich': [],
        }
    
    def is_field_on(self) -> bool:
        """
        Determine if magnetic field is ON based on current step.
        
        Returns:
            True if field is ON, False otherwise
        """
        cycle_length = self.params.n_steps_on + self.params.n_steps_off
        step_in_cycle = self.current_step % cycle_length
        return step_in_cycle < self.params.n_steps_on
    
    def step(self) -> None:
        """Execute one kinetic Monte Carlo time step."""
        # Update field state
        self.field_on = self.is_field_on()
        
        # Process antigen binding/unbinding events
        self._process_antigen_events()
        
        # Update time and step
        self.current_time += self.params.dt
        self.current_step += 1
        
        # Record observables
        self._record_observables()
    
    def run(self, n_steps: int) -> None:
        """
        Run simulation for n_steps.
        
        Args:
            n_steps: Number of steps to run
        """
        for _ in range(n_steps):
            self.step()
    
    def get_all_particles(self) -> dict[int, Particle]:
        """
        Get dictionary of all particles in the simulation.
        
        Returns:
            Dictionary mapping particle_id to Particle objects
        """
        return self._all_particles
    
    def _process_antigen_events(self) -> None:
        """
        Process all possible binding/unbinding events for each antigen.
        
        Each antigen attempts ALL applicable reactions independently per timestep.
        This ensures proper KMC kinetics where events are independent. Reactions
        are determined based on the antigen's state at the start of the timestep,
        not mid-timestep state changes.
        """
        # Create list of indices and shuffle them
        indices = list(range(len(self.antigens)))
        random.shuffle(indices)
        
        for idx in indices:
            antigen = self.antigens[idx]
            
            # Capture initial state at start of timestep
            # All reaction decisions are based on this initial state
            initial_state = antigen.state
            
            # Try all applicable reactions independently based on initial state
            
            # Try binding to A (if not already bound to A at start)
            if initial_state == AntigenState.FREE or initial_state == AntigenState.BOUND_B:
                self._attempt_binding(antigen, 'A')
            
            # Try binding to B (if not already bound to B at start)
            if initial_state == AntigenState.FREE or initial_state == AntigenState.BOUND_A:
                self._attempt_binding(antigen, 'B')
            
            # Try unbinding from A (if bound to A at start)
            if initial_state == AntigenState.BOUND_A or initial_state == AntigenState.SANDWICH:
                self._attempt_unbinding(antigen, 'A')
            
            # Try unbinding from B (if bound to B at start)
            if initial_state == AntigenState.BOUND_B or initial_state == AntigenState.SANDWICH:
                self._attempt_unbinding(antigen, 'B')
    
    def _calculate_binding_probability(self, antigen: Antigen, bind_type: str) -> float:
        """
        Calculate binding probability for an antigen.
        
        Args:
            antigen: Antigen object
            bind_type: 'A' or 'B'
            
        Returns:
            Binding probability
        """
        if bind_type == 'A':
            kon = self.params.kon_a
            C_antibody = self.params.C_antibody_A
        else:
            kon = self.params.kon_b
            C_antibody = self.params.C_antibody_B
        
        if self.field_on:
            # Check if antigen is on North/South patch for enhanced binding
            if antigen.is_on_north_or_south_patch(self._all_particles):
                # Calculate neighbor probability
                if bind_type == 'A':
                    N_type = self.params.N_A_sim
                else:
                    N_type = self.params.N_B_sim
                N_total = self.params.N_A_sim + self.params.N_B_sim
                P_neighbor = calc_neighbor_probability(N_type, N_total)
                
                return calc_bind_probability_field_on_enhanced(
                    kon, self.params.C_enhancement, P_neighbor, self.params.dt
                )
        
        # Standard binding (field OFF or not on North/South patch)
        return calc_bind_probability_field_off(kon, C_antibody, self.params.dt)
    
    def _get_particle_cluster_type(self, particle_id: int) -> str:
        """
        Determine if a particle is 'Single', 'Chain', or 'Aggregate'.
        
        Args:
            particle_id: ID of the particle to check
            
        Returns:
            'Single', 'Chain', or 'Aggregate'
        """
        from .clusters import find_clusters, classify_cluster
        
        particle = self._all_particles[particle_id]
        
        # Check if particle has any links
        if not particle.links:
            return 'Single'
        
        # Find which cluster this particle belongs to
        clusters = find_clusters(self._all_particles)
        
        for cluster in clusters:
            if particle_id in cluster:
                # Classify the cluster
                return classify_cluster(cluster, self._all_particles)
        
        return 'Single'
    
    def _is_patch_allowed_for_binding(self, particle: Particle, patch_id: int) -> bool:
        """
        Determine if a patch is allowed for binding based on field state and particle cluster type.
        
        Rules when field is ON with restriction enabled:
        - Single particles: ONLY North/South patches (0, 1)
        - Chains: ONLY North/South patches (0, 1)  
        - Aggregates: ALL patches allowed (but won't create links)
        
        Args:
            particle: The particle to check
            patch_id: The patch ID to check
            
        Returns:
            True if binding is allowed on this patch
        """
        # If restriction is disabled, all patches are allowed
        if not self.params.restrict_aggregates_field_on:
            return True
        
        # If field is OFF, all patches are allowed
        if not self.field_on:
            return True
        
        # Field is ON and restriction is enabled
        # Determine particle's cluster type
        cluster_type = self._get_particle_cluster_type(particle.particle_id)
        
        # Aggregates can bind on any patch (they can still bind antigens)
        if cluster_type == 'Aggregate':
            return True
        
        # Single particles and Chains: ONLY North/South patches
        return particle.is_north_or_south_patch(patch_id)
    
    def _is_particle_allowed_for_sandwich(self, particle_id: int) -> bool:
        """
        Determine if a particle can form a new sandwich LINK.
        
        When field is ON and restriction is enabled:
        - Single particles: YES (can create links via N/S)
        - Chains: YES (can extend via N/S)
        - Aggregates: NO (frozen, timeout)
        
        Args:
            particle_id: ID of the particle to check
            
        Returns:
            True if particle can form sandwich links
        """
        # If restriction is disabled, all particles are allowed
        if not self.params.restrict_aggregates_field_on:
            return True
        
        # If field is OFF, all particles are allowed
        if not self.field_on:
            return True
        
        # Field is ON and restriction is enabled
        cluster_type = self._get_particle_cluster_type(particle_id)
        
        # Aggregates are NOT allowed to create new links
        if cluster_type == 'Aggregate':
            return False
        
        # Single particles and Chains are allowed
        return True
    
    def _attempt_binding(self, antigen: Antigen, bind_type: str) -> None:
        """
        Attempt to bind an antigen to a particle.
        
        Args:
            antigen: Antigen object
            bind_type: 'A' or 'B'
        """
        # Calculate binding probability
        prob = self._calculate_binding_probability(antigen, bind_type)
        
        # Roll dice
        if random.random() < prob:
            # Find an available patch
            if bind_type == 'A':
                particles = self.particles_a
            else:
                particles = self.particles_b
            
            # Collect all available patches with NEW filtering
            available_patches = []
            for particle in particles:
                for patch_id, antigen_id in particle.patches.items():
                    if antigen_id is None:
                        # NEW: Filter patches based on field state, particle type, and cluster state
                        if self._is_patch_allowed_for_binding(particle, patch_id):
                            available_patches.append((particle, patch_id))
            
            if available_patches:
                # Randomly select one
                particle, patch_id = random.choice(available_patches)
                
                # Bind antigen to particle
                particle.bind_antigen(patch_id, antigen.antigen_id)
                
                # Update antigen binding
                if bind_type == 'A':
                    antigen.bind_to_a(particle.particle_id, patch_id)
                else:
                    antigen.bind_to_b(particle.particle_id, patch_id)
                
                # If this creates a sandwich, check if both particles are allowed to link
                if antigen.state == AntigenState.SANDWICH:
                    particle_a_id, patch_a_id = antigen.binding_a
                    particle_b_id, patch_b_id = antigen.binding_b
                    
                    # NEW: Check if both particles are allowed to form sandwich link
                    if (self._is_particle_allowed_for_sandwich(particle_a_id) and 
                        self._is_particle_allowed_for_sandwich(particle_b_id)):
                        
                        particle_a = self._all_particles[particle_a_id]
                        particle_b = self._all_particles[particle_b_id]
                        
                        particle_a.add_link(patch_a_id, particle_b_id, patch_b_id)
                        particle_b.add_link(patch_b_id, particle_a_id, patch_a_id)
                    # If not allowed: antigen is in SANDWICH state but no link created
    
    def _attempt_unbinding(self, antigen: Antigen, unbind_type: str) -> None:
        """
        Attempt to unbind an antigen from a particle.
        
        Args:
            antigen: Antigen object
            unbind_type: 'A' or 'B'
        """
        # Check if antigen is bound to this type
        if unbind_type == 'A' and antigen.binding_a is None:
            return
        if unbind_type == 'B' and antigen.binding_b is None:
            return
        
        # Calculate unbinding probability
        if unbind_type == 'A':
            koff = self.params.koff_a
        else:
            koff = self.params.koff_b
        
        prob = calc_unbind_probability(koff, self.params.dt)
        
        # Roll dice
        if random.random() < prob:
            # Get binding info
            if unbind_type == 'A':
                particle_id, patch_id = antigen.binding_a
            else:
                particle_id, patch_id = antigen.binding_b
            
            particle = self._all_particles[particle_id]
            
            # If it's a sandwich, remove the link first
            if antigen.state == AntigenState.SANDWICH:
                particle_a_id, patch_a_id = antigen.binding_a
                particle_b_id, patch_b_id = antigen.binding_b
                
                particle_a = self._all_particles[particle_a_id]
                particle_b = self._all_particles[particle_b_id]
                
                particle_a.remove_link(patch_a_id)
                particle_b.remove_link(patch_b_id)
            
            # Unbind antigen from particle
            particle.unbind_antigen(patch_id)
            
            # Update antigen binding
            if unbind_type == 'A':
                antigen.unbind_from_a()
            else:
                antigen.unbind_from_b()
    
    def _record_observables(self) -> None:
        """Record current state to history."""
        # Count antigens in each state
        n_free = sum(1 for a in self.antigens if a.state == AntigenState.FREE)
        n_bound_a = sum(1 for a in self.antigens if a.state == AntigenState.BOUND_A)
        n_bound_b = sum(1 for a in self.antigens if a.state == AntigenState.BOUND_B)
        n_sandwich = sum(1 for a in self.antigens if a.state == AntigenState.SANDWICH)
        
        self.history['time'].append(self.current_time)
        self.history['step'].append(self.current_step)
        self.history['field_on'].append(self.field_on)
        self.history['n_free'].append(n_free)
        self.history['n_bound_a'].append(n_bound_a)
        self.history['n_bound_b'].append(n_bound_b)
        self.history['n_sandwich'].append(n_sandwich)
