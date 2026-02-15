"""
Monte Carlo simulation of magnetic nanoparticles with antibody-antigen binding.
"""

from .particle import Particle
from .antigen import Antigen, AntigenState
from .parameters import SimulationParameters
from .simulation import Simulation
from .events import (
    calc_bind_probability_field_off,
    calc_bind_probability_field_on_enhanced,
    calc_unbind_probability,
    calc_neighbor_probability,
)
from .clusters import find_clusters, classify_cluster

__all__ = [
    'Particle',
    'Antigen',
    'AntigenState',
    'SimulationParameters',
    'Simulation',
    'calc_bind_probability_field_off',
    'calc_bind_probability_field_on_enhanced',
    'calc_unbind_probability',
    'calc_neighbor_probability',
    'find_clusters',
    'classify_cluster',
]

# Import visualization if dependencies are available
try:
    from .visualization import (
        visualize_system_3d,
        create_cycle_snapshots,
        create_animation,
        plot_cluster_size_distributions,
    )
    _HAS_VISUALIZATION = True
    __all__.extend([
        'visualize_system_3d',
        'create_cycle_snapshots',
        'create_animation',
        'plot_cluster_size_distributions',
    ])
except ImportError:
    _HAS_VISUALIZATION = False
