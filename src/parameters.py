"""
SimulationParameters class for storing and calculating simulation parameters.
"""

from dataclasses import dataclass, field


@dataclass
class SimulationParameters:
    """
    Parameters for the kinetic Monte Carlo simulation.
    
    Input parameters:
        C_A: Concentration of A particles (nM)
        C_B: Concentration of B particles (nM)
        C_antigen: Concentration of antigen (nM)
        C_enhancement: Enhanced concentration when field ON (M or μM)
        N_A_sim: Number of A particles in simulation
        N_B_sim: Number of B particles in simulation
        antibodies_per_particle: Number of antibodies per particle
        n_patches: Patches per particle (≤30)
        kon_a: Association rate constant for A (M⁻¹s⁻¹)
        koff_a: Dissociation rate constant for A (s⁻¹)
        kon_b: Association rate constant for B (M⁻¹s⁻¹)
        koff_b: Dissociation rate constant for B (s⁻¹)
        dt: Time step (seconds)
        n_steps_on: Steps with field ON
        n_steps_off: Steps with field OFF
        n_repeats: Number of ON/OFF cycles
    
    Derived quantities (calculated in __post_init__):
        V_box: Simulation box volume (liters)
        N_antigen_sim: Number of antigens in simulation
        C_antibody_A: Effective antibody A concentration (M)
        C_antibody_B: Effective antibody B concentration (M)
    """
    
    # Input parameters
    C_A: float  # nM
    C_B: float  # nM
    C_antigen: float  # nM
    C_enhancement: float  # M or μM
    N_A_sim: int
    N_B_sim: int
    antibodies_per_particle: int
    n_patches: int
    kon_a: float  # M⁻¹s⁻¹
    koff_a: float  # s⁻¹
    kon_b: float  # M⁻¹s⁻¹
    koff_b: float  # s⁻¹
    dt: float  # seconds
    n_steps_on: int
    n_steps_off: int
    n_repeats: int
    
    # Derived quantities (set in __post_init__)
    V_box: float = field(init=False)
    N_antigen_sim: int = field(init=False)
    C_antibody_A: float = field(init=False)
    C_antibody_B: float = field(init=False)
    
    def __post_init__(self):
        """Calculate derived quantities."""
        N_A = 6.022e23  # Avogadro's number
        
        # Calculate simulation box volume (liters)
        # V_box = N_A_sim / (C_A * 1e-9 * N_A)
        self.V_box = self.N_A_sim / (self.C_A * 1e-9 * N_A)
        
        # Calculate number of antigens in simulation
        # N_antigen_sim = int(C_antigen * 1e-9 * N_A * V_box)
        self.N_antigen_sim = int(self.C_antigen * 1e-9 * N_A * self.V_box)
        
        # Calculate effective antibody concentrations (M)
        # C_antibody_A = (N_A_sim * antibodies_per_particle) / (N_A * V_box)
        self.C_antibody_A = (self.N_A_sim * self.antibodies_per_particle) / (N_A * self.V_box)
        
        # C_antibody_B = (N_B_sim * antibodies_per_particle) / (N_A * V_box)
        self.C_antibody_B = (self.N_B_sim * self.antibodies_per_particle) / (N_A * self.V_box)
