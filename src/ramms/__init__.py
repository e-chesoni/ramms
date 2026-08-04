# src/ramms/__init__.py

from .core import (
    RAMM_Node,
    RAMM_Strut,
    RAMM_Unit,
    RAMM_Chain,
    UnitType,
)

from .contact import (
    Gap,
    GapResult,
    get_node_segment_gap,
    get_segment_segment_distance,
)

from .symbolic import (
    SymbolicGap,
    get_active_gap_vector,
    get_candidate_gaps,
)

from .mobility import (
    get_gap_jacobian,
)

from .plotting import (
    plot_geometry,
    plot_gap,
)