"""
Compare Monte Carlo simulation to ODE solution with plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.parameters import SimulationParameters
from src.simulation import Simulation
from src.ode_solver import solve_binding_odes, calculate_equilibrium_fractions


def main():
    # Use exact parameters from user's config
    params = SimulationParameters(
        C_A=0.002,
        C_B=0.002,
        C_antigen=0.1,
        C_enhancement=1.0e-6,
        N_A_sim=50,
        N_B_sim=50,
        antibodies_per_particle=1000,
        n_patches=12,
        kon_a=1.0e5,
        koff_a=0.0001,
        kon_b=1.0e5,
        koff_b=0.0001,
        dt=0.1,
        n_steps_on=10,
        n_steps_off=10,
        n_repeats=100,
        restrict_aggregates_field_on=True,
    )
    
    print("Running comparison with parameters:")
    print(f"  C_A = {params.C_A} nM (particles)")
    print(f"  C_antibody_A = {params.C_antibody_A:.2e} M (antibodies)")
    print(f"  kon = {params.kon_a:.2e} M⁻¹s⁻¹")
    print(f"  koff = {params.koff_a:.2e} s⁻¹")
    print(f"  K_D = {params.koff_a/params.kon_a:.2e} M")
    print(f"  N_antigen = {params.N_antigen_sim}")
    print()
    
    # Solve ODE
    print("Solving ODEs...")
    t_span = (0, 200)
    t_eval = np.linspace(0, 200, 201)
    ode_result = solve_binding_odes(params, t_span, t_eval)
    
    # Run MC simulation (multiple replicates for statistics)
    print("Running Monte Carlo simulation...")
    n_replicates = 5
    mc_time = []
    mc_free = []
    mc_bound_a = []
    mc_bound_b = []
    mc_sandwich = []
    
    for rep in range(n_replicates):
        print(f"  Replicate {rep+1}/{n_replicates}")
        sim = Simulation(params)
        
        # Record every 10 steps
        for step in range(0, 2001, 10):
            if step > 0:
                sim.run(10)
            
            counts = sim.get_antigen_counts()
            mc_time.append(sim.current_time)
            mc_free.append(counts.get('Free', 0))
            mc_bound_a.append(counts.get('Bound_A', 0))
            mc_bound_b.append(counts.get('Bound_B', 0))
            mc_sandwich.append(counts.get('Sandwich', 0))
    
    # Calculate equilibrium
    eq = calculate_equilibrium_fractions(params)
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Monte Carlo vs ODE Comparison (Field OFF)', fontsize=14, fontweight='bold')
    
    states = ['Free', 'Bound_A', 'Bound_B', 'Sandwich']
    ode_data = [ode_result['Free'], ode_result['Bound_A'], 
                ode_result['Bound_B'], ode_result['Sandwich']]
    mc_data = [mc_free, mc_bound_a, mc_bound_b, mc_sandwich]
    
    for ax, state, ode, mc in zip(axes.flat, states, ode_data, mc_data):
        # Plot ODE
        ax.plot(ode_result['t'], ode, 'b-', linewidth=2, label='ODE', alpha=0.8)
        
        # Plot MC (scatter with some transparency to see overlap)
        ax.scatter(mc_time, mc, c='red', s=10, alpha=0.3, label='MC (5 replicates)')
        
        # Plot equilibrium
        eq_count = eq['counts'][state]
        ax.axhline(eq_count, color='green', linestyle='--', linewidth=1.5, 
                  label=f'Equilibrium: {eq_count:.0f}', alpha=0.7)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Count')
        ax.set_title(state)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'mc_vs_ode_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to: {output_path}")
    
    # Print final comparison
    print("\nFinal counts at t=200s:")
    print(f"State       MC_avg  ODE     Equilibrium  MC/ODE")
    print("-" * 55)
    
    for state, ode, mc in zip(states, ode_data, mc_data):
        mc_avg = np.mean([mc[i] for i in range(len(mc_time)) if abs(mc_time[i] - 200) < 0.5])
        ode_final = ode[-1]
        eq_val = eq['counts'][state]
        ratio = mc_avg / ode_final if ode_final > 0 else float('inf')
        
        print(f"{state:10s}  {mc_avg:6.0f}  {ode_final:6.1f}  {eq_val:6.1f}       {ratio:.2f}")


if __name__ == '__main__':
    main()
