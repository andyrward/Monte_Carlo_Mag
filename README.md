# Monte Carlo Magnetic Nanoparticle Simulation

A kinetic Monte Carlo simulation of magnetic nanoparticles with antibody-antigen binding, featuring field-directed assembly restrictions and 3D visualization.

## Overview

This simulation models magnetic nanoparticles (types A and B) coated with antibodies that can bind antigens to form "sandwich" complexes. The simulation uses concentration-based kinetics with fixed time stepping and includes magnetic field ON/OFF phases with physically realistic assembly restrictions.

### Key Features

- **Field-Directed Assembly**: Realistic constraints on particle binding during magnetic field phases
- **3D Visualization**: Interactive visualization of simulation results with cluster classification
- **Cluster Analysis**: Automatic detection and classification of chains vs. aggregates
- **Comprehensive Testing**: 92+ unit tests ensuring code quality

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

2. Install the package (basic):
```bash
pip install -e .
```

3. Install with visualization support:
```bash
pip install -e ".[visualization]"
```

4. For development, install with dev dependencies:
```bash
pip install -e ".[dev,visualization]"
```

## Usage

### Basic Simulation

Run the simulation with default parameters:
```bash
python main.py
```

### Visualization Options

Generate initial and final state visualizations:
```bash
python main.py --visualize
```

Create snapshots at each cycle boundary:
```bash
python main.py --snapshots
```

Create an MP4 animation (requires ffmpeg):
```bash
python main.py --snapshots --animation
```

Specify custom output directory:
```bash
python main.py --visualize --output-dir my_results
```

Use custom configuration file:
```bash
python main.py --config my_config.yaml --visualize
```

### Command-Line Options

- `--config CONFIG`: Path to configuration YAML file (default: `config/default_params.yaml`)
- `--visualize`: Generate 3D visualization of initial and final states
- `--snapshots`: Create snapshots at each cycle boundary
- `--output-dir OUTPUT_DIR`: Directory for visualization outputs (default: `output`)
- `--animation`: Create MP4 animation from snapshots (requires ffmpeg)

## Configuration

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

# Magnetic field behavior
restrict_aggregates_field_on: true  # If true, aggregates cannot form new links when field is ON
```

### Field-Directed Assembly Restrictions

The `restrict_aggregates_field_on` parameter controls physically realistic binding behavior:

**When field is ON and restriction is enabled:**
- **Single particles**: Can ONLY bind via North/South patches (indices 0, 1)
- **Chain particles**: Can ONLY bind via North/South patches to extend the chain
- **Aggregate particles**: Can bind antigens on ANY patch, but CANNOT create new inter-particle links (topology frozen)

**When field is OFF:**
- All particles can bind via any patch
- All sandwich complexes create links normally
- Chains can crosslink into aggregates

This feature enables realistic modeling of field-directed assembly where:
- Single particles and chains align with the magnetic field and can only extend longitudinally
- Aggregates are kinetically frozen and cannot reorganize during field ON phases
- System can relax and form aggregates during field OFF phases

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
- Field restriction settings

#### Simulation (`src/simulation.py`)
Main KMC simulation engine:
- Time stepping with field ON/OFF phases
- Stochastic binding/unbinding events with field restrictions
- Observable tracking (antigen states, clusters)
- Helper methods for cluster type detection and patch filtering

### Event Calculations (`src/events.py`)
Probability calculations for:
- Standard binding (field OFF)
- Enhanced binding (field ON, North/South patches)
- Unbinding
- Neighbor probability

### Cluster Analysis (`src/clusters.py`)
- BFS-based cluster detection
- Chain vs. Aggregate classification
  - **Chain**: All links through North-South patches
  - **Aggregate**: Contains regular patch links

### Visualization (`src/visualization.py`)
3D visualization tools with:
- **Color coding**:
  - Single particles: Brown spheres with red dots for North/South patches
  - Chains: Green spheres
  - Aggregates: Red spheres
- **Features**:
  - Non-overlapping random particle placement
  - Black lines showing inter-particle links
  - Statistics overlay (time, step, field state, particle counts, antigen counts)
  - Snapshot generation at cycle boundaries
  - Animation support (requires ffmpeg)

## Visualization Examples

The visualization system generates 3D plots showing:

1. **Particle Classification**: Automatically detects and color-codes particles based on their cluster type
2. **North/South Markers**: Red dots on single particles indicate magnetic poles
3. **Connectivity**: Black lines show sandwich complex links between particles
4. **Statistics**: Overlay shows current simulation state and counts

Snapshots are created at key points:
- Initial state (step 0)
- End of each field ON phase
- End of each field OFF phase (end of cycle)

## Running Tests

Run all tests:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

Run specific test modules:
```bash
pytest tests/test_simulation.py -v
pytest tests/test_visualization.py -v
```

## Output

The simulation outputs:
- **Console statistics**: Antigen states, cluster counts, largest cluster size
- **Visualization files** (if enabled):
  - PNG images for snapshots
  - MP4 animation (if ffmpeg available)
- **Time series data**: Available in `simulation.history` for custom analysis

## Development Status

Current features:
- ✅ Core particle and antigen classes
- ✅ Kinetic Monte Carlo simulation engine
- ✅ Field ON/OFF phases
- ✅ Field-directed assembly restrictions
- ✅ Cluster detection and classification
- ✅ 3D visualization system
- ✅ Comprehensive test suite (92+ tests)

## License

See LICENSE file for details.

## Citation

If you use this code in your research, please cite the appropriate paper.