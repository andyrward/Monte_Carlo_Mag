"""
Minimal Kinetic Monte Carlo for A/B binding validation.

No particles, no patches, no spatial constraints.
Pure probability-based binding of A and B to antigens in bulk solution.
"""

import random
from typing import Dict, List, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class MinimalAntigen:
    """
    Minimal antigen: just tracks binding state.
    
    States:
    - 0: Free
    - 1: Bound to A only
    - 2: Bound to B only
    - 3: Sandwich (bound to both A and B)
    """
    state: int = 0  # 0=Free, 1=Bound_A, 2=Bound_B, 3=Sandwich
    
    @property
    def is_free(self) -> bool:
        return self.state == 0
    
    @property
    def has_a(self) -> bool:
        return self.state in [1, 3]
    
    @property
    def has_b(self) -> bool:
        return self.state in [2, 3]
    
    def bind_a(self):
        """Add A binding."""
        if self.state == 0:
            self.state = 1
        elif self.state == 2:
            self.state = 3
    
    def bind_b(self):
        """Add B binding."""
        if self.state == 0:
            self.state = 2
        elif self.state == 1:
            self.state = 3
    
    def unbind_a(self):
        """Remove A binding."""
        if self.state == 1:
            self.state = 0
        elif self.state == 3:
            self.state = 2
    
    def unbind_b(self):
        """Remove B binding."""
        if self.state == 2:
            self.state = 0
        elif self.state == 3:
            self.state = 1


def run_minimal_kmc(
    N_antigen: int,
    C_A: float,  # M
    C_B: float,  # M
    kon_a: float,  # M⁻¹s⁻¹
    koff_a: float,  # s⁻¹
    kon_b: float,  # M⁻¹s⁻¹
    koff_b: float,  # s⁻¹
    dt: float,  # s
    n_steps: int,
    record_interval: int = 1,
    seed: int = None
) -> Dict:
    """
    Run minimal KMC simulation: antigens binding A/B independently in bulk solution.
    
    Algorithm per timestep:
    1. For each antigen, attempt ALL applicable reactions INDEPENDENTLY:
       - If not bound to A: attempt A binding with P = kon_a × C_A × dt
       - If not bound to B: attempt B binding with P = kon_b × C_B × dt
       - If bound to A: attempt A unbinding with P = koff_a × dt
       - If bound to B: attempt B unbinding with P = koff_b × dt
    
    Args:
        N_antigen: Number of antigens
        C_A: Concentration of A antibodies (M)
        C_B: Concentration of B antibodies (M)
        kon_a: A association rate constant (M⁻¹s⁻¹)
        koff_a: A dissociation rate constant (s⁻¹)
        kon_b: B association rate constant (M⁻¹s⁻¹)
        koff_b: B dissociation rate constant (s⁻¹)
        dt: Time step (s)
        n_steps: Number of steps to simulate
        record_interval: Record state every N steps
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with:
            - 't': Array of time points
            - 'Free': Array of Free antigen counts
            - 'Bound_A': Array of Bound to A counts
            - 'Bound_B': Array of Bound to B counts
            - 'Sandwich': Array of Sandwich counts
            - 'params': Dictionary of parameters used
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Initialize antigens (all free)
    antigens = [MinimalAntigen(state=0) for _ in range(N_antigen)]
    
    # Calculate probabilities
    P_bind_A = kon_a * C_A * dt
    P_bind_B = kon_b * C_B * dt
    P_unbind_A = koff_a * dt
    P_unbind_B = koff_b * dt
    
    # Validate probabilities
    if P_bind_A > 1.0 or P_bind_B > 1.0:
        raise ValueError(f"Binding probabilities exceed 1.0: P_A={P_bind_A:.3f}, P_B={P_bind_B:.3f}")
    if P_unbind_A > 1.0 or P_unbind_B > 1.0:
        raise ValueError(f"Unbinding probabilities exceed 1.0: P_A={P_unbind_A:.3f}, P_B={P_unbind_B:.3f}")
    
    # Storage
    times = []
    counts_free = []
    counts_bound_a = []
    counts_bound_b = []
    counts_sandwich = []
    
    # Record initial state
    def count_states():
        free = sum(1 for a in antigens if a.state == 0)
        bound_a = sum(1 for a in antigens if a.state == 1)
        bound_b = sum(1 for a in antigens if a.state == 2)
        sandwich = sum(1 for a in antigens if a.state == 3)
        return free, bound_a, bound_b, sandwich
    
    times.append(0.0)
    f, ba, bb, s = count_states()
    counts_free.append(f)
    counts_bound_a.append(ba)
    counts_bound_b.append(bb)
    counts_sandwich.append(s)
    
    # Simulate
    for step in range(1, n_steps + 1):
        current_time = step * dt
        
        # For each antigen, attempt all applicable reactions INDEPENDENTLY
        for antigen in antigens:
            # Store initial state for this antigen
            initial_state = antigen.state
            
            # Attempt A binding (if not already bound to A)
            if not antigen.has_a:
                if random.random() < P_bind_A:
                    antigen.bind_a()
            
            # Attempt B binding (if not already bound to B in INITIAL state)
            # Use initial_state to ensure independent reactions
            if initial_state not in [2, 3]:  # Was not bound to B initially
                if random.random() < P_bind_B:
                    antigen.bind_b()
            
            # Attempt A unbinding (if bound to A in INITIAL state)
            if initial_state in [1, 3]:  # Was bound to A initially
                if random.random() < P_unbind_A:
                    antigen.unbind_a()
            
            # Attempt B unbinding (if bound to B in INITIAL state)
            if initial_state in [2, 3]:  # Was bound to B initially
                if random.random() < P_unbind_B:
                    antigen.unbind_b()
        
        # Record state
        if step % record_interval == 0:
            times.append(current_time)
            f, ba, bb, s = count_states()
            counts_free.append(f)
            counts_bound_a.append(ba)
            counts_bound_b.append(bb)
            counts_sandwich.append(s)
    
    return {
        't': np.array(times),
        'Free': np.array(counts_free),
        'Bound_A': np.array(counts_bound_a),
        'Bound_B': np.array(counts_bound_b),
        'Sandwich': np.array(counts_sandwich),
        'params': {
            'N_antigen': N_antigen,
            'C_A': f"{C_A:.2e} M",
            'C_B': f"{C_B:.2e} M",
            'kon_a': f"{kon_a:.2e} M⁻¹s⁻¹",
            'koff_a': f"{koff_a:.2e} s⁻¹",
            'kon_b': f"{kon_b:.2e} M⁻¹s⁻¹",
            'koff_b': f"{koff_b:.2e} s⁻¹",
            'dt': f"{dt} s",
            'P_bind_A': P_bind_A,
            'P_bind_B': P_bind_B,
            'P_unbind_A': P_unbind_A,
            'P_unbind_B': P_unbind_B,
        }
    }


def run_multiple_replicates(n_replicates: int, **kmc_kwargs) -> List[Dict]:
    """
    Run multiple KMC replicates for statistical analysis.
    
    Args:
        n_replicates: Number of independent simulations
        **kmc_kwargs: Arguments passed to run_minimal_kmc
    
    Returns:
        List of result dictionaries
    """
    results = []
    for i in range(n_replicates):
        result = run_minimal_kmc(seed=i, **kmc_kwargs)
        results.append(result)
    return results


def calculate_statistics(replicates: List[Dict]) -> Dict:
    """
    Calculate mean and standard deviation across replicates.
    
    Args:
        replicates: List of result dictionaries from run_multiple_replicates
    
    Returns:
        Dictionary with mean and std for each state
    """
    # All replicates should have same time points
    t = replicates[0]['t']
    
    # Stack results
    free_all = np.array([r['Free'] for r in replicates])
    bound_a_all = np.array([r['Bound_A'] for r in replicates])
    bound_b_all = np.array([r['Bound_B'] for r in replicates])
    sandwich_all = np.array([r['Sandwich'] for r in replicates])
    
    return {
        't': t,
        'Free_mean': free_all.mean(axis=0),
        'Free_std': free_all.std(axis=0),
        'Bound_A_mean': bound_a_all.mean(axis=0),
        'Bound_A_std': bound_a_all.std(axis=0),
        'Bound_B_mean': bound_b_all.mean(axis=0),
        'Bound_B_std': bound_b_all.std(axis=0),
        'Sandwich_mean': sandwich_all.mean(axis=0),
        'Sandwich_std': sandwich_all.std(axis=0),
    }


def compare_kmc_to_ode(kmc_stats: Dict, ode_result: Dict, params: Dict) -> Dict:
    """
    Calculate comparison statistics between KMC and ODE.
    
    Args:
        kmc_stats: Statistics from calculate_statistics
        ode_result: Result from ode_solver.solve_binding_odes
        params: Parameter dictionary
    
    Returns:
        Dictionary with comparison metrics
    """
    # Get final values
    kmc_final = {
        'Free': kmc_stats['Free_mean'][-1],
        'Bound_A': kmc_stats['Bound_A_mean'][-1],
        'Bound_B': kmc_stats['Bound_B_mean'][-1],
        'Sandwich': kmc_stats['Sandwich_mean'][-1],
    }
    
    ode_final = {
        'Free': ode_result['Free'][-1],
        'Bound_A': ode_result['Bound_A'][-1],
        'Bound_B': ode_result['Bound_B'][-1],
        'Sandwich': ode_result['Sandwich'][-1],
    }
    
    # Calculate ratios and errors
    comparison = {}
    for state in ['Free', 'Bound_A', 'Bound_B', 'Sandwich']:
        kmc_val = kmc_final[state]
        ode_val = ode_final[state]
        ratio = kmc_val / ode_val if ode_val > 0 else float('inf')
        abs_error = kmc_val - ode_val
        rel_error = abs_error / ode_val if ode_val > 0 else float('inf')
        
        comparison[state] = {
            'kmc': kmc_val,
            'ode': ode_val,
            'ratio': ratio,
            'abs_error': abs_error,
            'rel_error': rel_error,
        }
    
    return comparison
