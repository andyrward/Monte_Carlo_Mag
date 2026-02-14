# Monte Carlo Magnetic Nanoparticle Simulation

A kinetic Monte Carlo simulation of magnetic nanoparticles with antibody-antigen binding.

## Overview

This simulation models magnetic nanoparticles (types A and B) coated with antibodies that can bind antigens to form "sandwich" complexes. The simulation uses concentration-based kinetics with fixed time stepping and includes magnetic field ON/OFF phases.

## Installation

### Prerequisites

- Python 3.12 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/andyrward/Monte_Carlo_Mag.git
cd Monte_Carlo_Mag
```

2. Install the package:
```bash
pip install -e .
```

3. For development, install with dev dependencies:
```bash
pip install -e ".[dev]"
```

## Usage

### Running the Simulation

Run the simulation with default parameters:
```bash
python main.py
```

### Configuration

Edit `config/default_params.yaml` to customize simulation parameters:

```yaml
# Concentrations (nM)
C_A: 10.0              # Concentration of A particles
C_B: 10.0              # Concentration of B particles
C_antigen: 1.0         # Concentration of antigen

# Enhanced concentration (M)
C_enhancement: 0.000001  # Enhanced concentration when field ON

# Simulation size
N_A_sim: 50            # Number of A particles
N_B_sim: 50            # Number of B particles

# Particle properties
antibodies_per_particle: 1000
n_patches: 12          # Patches per particle (≤30)

# Kinetic rates
kon_a: 100000.0        # Association rate for A (M^-1 s^-1)
koff_a: 0.1            # Dissociation rate for A (s^-1)
kon_b: 100000.0        # Association rate for B (M^-1 s^-1)
koff_b: 0.1            # Dissociation rate for B (s^-1)

# Time stepping
dt: 0.001              # Time step (seconds)

# Field profile
n_steps_on: 1000       # Steps with field ON
n_steps_off: 1000      # Steps with field OFF
n_repeats: 5           # Number of ON/OFF cycles
```

## Architecture

### Core Components

#### Particle (`src/particle.py`)
Represents a magnetic nanoparticle with antibody-coated patches:
- North pole (patch 0) and South pole (patch 1)
- Regular patches (indices 2 to n_patches-1)
- Tracks antigen bindings and particle-particle links

#### Antigen (`src/antigen.py`)
Represents an antigen molecule that can bind to both A and B particles:
- States: Free, Bound_A, Bound_B, Sandwich
- Tracks bindings to both particle types

#### SimulationParameters (`src/parameters.py`)
Dataclass for simulation parameters with automatic calculation of derived quantities:
- Simulation box volume
- Number of antigens
- Effective antibody concentrations

#### Simulation (`src/simulation.py`)
Main KMC simulation engine:
- Time stepping with field ON/OFF phases
- Stochastic binding/unbinding events
- Observable tracking (antigen states, clusters)

### Event Calculations (`src/events.py`)
Probability calculations for:
- Standard binding (field OFF)
- Enhanced binding (field ON, North/South patches)
- Unbinding
- Neighbor probability

### Cluster Analysis (`src/clusters.py`)
- BFS-based cluster detection
- Chain vs. Aggregate classification
  - Chain: All links through North-South patches
  - Aggregate: Contains regular patch links

## Running Tests

Run all tests:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

## Output

The simulation outputs:
- Antigen state statistics (Free, Bound_A, Bound_B, Sandwich)
- Cluster statistics (total, chains, aggregates)
- Time series data in `simulation.history`

## Development Status

This is Phase 1 of the implementation. Current features:
- ✅ Core particle and antigen classes
- ✅ Kinetic Monte Carlo simulation engine
- ✅ Field ON/OFF phases
- ✅ Cluster detection and classification
- ✅ Comprehensive test suite (63 tests)

## License

See LICENSE file for details.

## Citation

If you use this code in your research, please cite the appropriate paper.