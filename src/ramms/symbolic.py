"""
Symbolic gap calculations for RAMMs.
"""

# ============================================================================
# Imports
# ============================================================================
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

plt.style.use("seaborn-v0_8-whitegrid")
import plotly.graph_objects as go
import copy
import re
import pprint
import math
from enum import Enum
from plotly.subplots import make_subplots
from collections import Counter
from scipy.optimize import root
from IPython.display import display # for printing things in LaTex
from matplotlib.lines import Line2D
from dataclasses import dataclass
from scipy.optimize import root
from scipy.optimize import least_squares
from types import SimpleNamespace

from .exceptions import InvalidRAMMGeometryError
from .core import UnitType
from .contact import get_node_segment_gap, get_segment_segment_gap

@dataclass
class SymbolicGap:
    A: sp.Matrix
    B: sp.Matrix
    P: sp.Matrix
    u: sp.Matrix
    v: sp.Matrix
    t: sp.Expr
    Q: sp.Matrix
    gap_vector: sp.Matrix
    gap_magnitude: sp.Expr
    signed_gap: sp.Expr


# NOTE: We dont actually need this; just a check.
# The more useful method is working below
def get_symbolic_gap_general():

    Ay, Az = sp.symbols("Ay Az", real=True)
    By, Bz = sp.symbols("By Bz", real=True)
    Py, Pz = sp.symbols("Py Pz", real=True)

    A = sp.Matrix([Ay, Az])
    B = sp.Matrix([By, Bz])
    P = sp.Matrix([Py, Pz])

    u = B - A
    v = P - A

    u_squared_length = u.dot(u)
    t = v.dot(u) / u_squared_length

    Q = A + t * u
    gap_vector = P - Q
    gap_magnitude = gap_vector.norm()

    symbolic_gap = SymbolicGap(
        A=A,
        B=B,
        P=P,
        u=u,
        v=v,
        t=t,
        Q=Q,
        gap_vector=gap_vector,
        gap_magnitude=gap_magnitude,
        signed_gap=None
    )

    return symbolic_gap


def get_node_segment_symbolic_gap(node, segment, orientation):

    left_y, left_z = sp.symbols(
        f"A_{segment.node_1}_y "
        f"A_{segment.node_1}_z",
        real=True
    )

    right_y, right_z = sp.symbols(
        f"B_{segment.node_2}_y "
        f"B_{segment.node_2}_z",
        real=True
    )

    node_y, node_z = sp.symbols(
        f"P_{node}_y "
        f"P_{node}_z",
        real=True
    )

    A = sp.Matrix([left_y, left_z])
    B = sp.Matrix([right_y, right_z])
    P = sp.Matrix([node_y, node_z])

    u = B - A
    v = P - A

    u_squared_length = u.dot(u)
    t = v.dot(u) / u_squared_length

    Q = A + t * u
    gap_vector = P - Q
    gap_magnitude = gap_vector.norm()

    segment_length = sp.sqrt(u.dot(u))

    if orientation == "counterclockwise":
        normal = sp.Matrix([-u[1], u[0]]) / segment_length
    elif orientation == "clockwise":
        normal = sp.Matrix([u[1], -u[0]]) / segment_length
    else:
        raise ValueError(
            "orientation must be 'clockwise' or 'counterclockwise'"
        )

    signed_gap = sp.simplify(normal.dot(v))

    return SymbolicGap(
        A=A,
        B=B,
        P=P,
        u=u,
        v=v,
        t=t,
        Q=Q,
        gap_vector=gap_vector,
        gap_magnitude=gap_magnitude,
        signed_gap=signed_gap
    )


def get_segment_segment_symbolic_gap(
    segment_1,
    segment_2
):
    """
    Return the symbolic signed perpendicular gap between
    two parallel segment centerlines.

    Assumes the relevant finite segments are already known
    to be parallel and overlapping.
    """

    A_y, A_z = sp.symbols(
        f"{segment_1.node_1.descriptor}_y "
        f"{segment_1.node_1.descriptor}_z",
        real=True
    )

    B_y, B_z = sp.symbols(
        f"{segment_1.node_2.descriptor}_y "
        f"{segment_1.node_2.descriptor}_z",
        real=True
    )

    C_y, C_z = sp.symbols(
        f"{segment_2.node_1.descriptor}_y "
        f"{segment_2.node_1.descriptor}_z",
        real=True
    )

    A = sp.Matrix([A_y, A_z])
    B = sp.Matrix([B_y, B_z])
    C = sp.Matrix([C_y, C_z])

    u = B - A
    v = C - A

    segment_length = sp.sqrt(
        u.dot(u)
    )

    signed_gap = (
        u[0] * v[1]
        - u[1] * v[0]
    ) / segment_length

    return SymbolicGap(
        A=A,
        B=B,
        P=C,
        u=u,
        v=v,
        t=None,
        Q=None,
        gap_vector=None,
        gap_magnitude=None,
        signed_gap=sp.simplify(signed_gap)
    )


def get_shared_rail_node_node_symbolic_gap(
    lower_node,
    upper_node,
    railed_unit,
    node_diameter,
):
    """
    Symbolic clearance between two FREE nodes sharing the
    centerline of an interior RAILED unit.

    Example for RAILED Unit 2:

        lower node = 1T
        upper node = 3B

    Gap is measured along Unit 2's local rail direction.

        gap > 0 : separated
        gap = 0 : contact
        gap < 0 : crossed / overlapping
    """

    lower_y, lower_z = sp.symbols(
        f"{lower_node.descriptor}_y "
        f"{lower_node.descriptor}_z",
        real=True,
    )

    upper_y, upper_z = sp.symbols(
        f"{upper_node.descriptor}_y "
        f"{upper_node.descriptor}_z",
        real=True,
    )

    rail_B_y, rail_B_z = sp.symbols(
        f"{railed_unit.bottom_node.descriptor}_y "
        f"{railed_unit.bottom_node.descriptor}_z",
        real=True,
    )

    rail_T_y, rail_T_z = sp.symbols(
        f"{railed_unit.top_node.descriptor}_y "
        f"{railed_unit.top_node.descriptor}_z",
        real=True,
    )

    lower = sp.Matrix([
        lower_y,
        lower_z,
    ])

    upper = sp.Matrix([
        upper_y,
        upper_z,
    ])

    rail_B = sp.Matrix([
        rail_B_y,
        rail_B_z,
    ])

    rail_T = sp.Matrix([
        rail_T_y,
        rail_T_z,
    ])

    rail_vector = (
        rail_T - rail_B
    )

    rail_length = sp.sqrt(
        rail_vector.dot(
            rail_vector
        )
    )

    rail_direction = (
        rail_vector
        / rail_length
    )

    spacing = (
        upper - lower
    ).dot(
        rail_direction
    )

    return sp.simplify(
        spacing - node_diameter
    )


def get_candidate_gaps_old(chain, verbose=True):
    """
    Return candidate node-segment gaps for RAILED-FREE unit pairs.

    For each lower RAILED / upper FREE pair, this includes:

    1. Lower RAILED diamond nodes against upper FREE diamond struts.
    2. Upper FREE bottom node against the lower RAILED unit's
       left and right rails.

    Notes
    -----
    This currently assumes the alternating chain structure:

        RAILED, FREE, RAILED, FREE, ...

    and evaluates RAILED-FREE pairs:
        (0, 1), (2, 3), ...
    """

    gaps = []

    # Evaluate each RAILED-FREE pair.
    for base_unit_index in range(
        0,
        len(chain.units) - 1,
        2
    ):
        lower_unit_index = base_unit_index
        upper_unit_index = base_unit_index + 1

        skip_unit_index = base_unit_index + 2

        # ---------------------------------------------------------
        # Upper FREE unit diamond struts
        # ---------------------------------------------------------

        upper_unit_segment_BL = chain.get_segment_by_descriptor(
            f"{upper_unit_index}B{upper_unit_index}L"
        )

        upper_unit_segment_TR = chain.get_segment_by_descriptor(
            f"{upper_unit_index}T{upper_unit_index}R"
        )

        upper_unit_segment_RB = chain.get_segment_by_descriptor(
            f"{upper_unit_index}R{upper_unit_index}B"
        )

        upper_unit_segment_LT = chain.get_segment_by_descriptor(
            f"{upper_unit_index}L{upper_unit_index}T"
        )

        # ---------------------------------------------------------
        # Lower RAILED unit rails
        # ---------------------------------------------------------

        rail_0L = chain.get_rail(
            lower_unit_index,
            "left"
        )

        rail_0R = chain.get_rail(
            lower_unit_index,
            "right"
        )

        # ---------------------------------------------------------
        # Lower RAILED unit diamond nodes
        # ---------------------------------------------------------

        lower_node_T = chain.get_node_by_descriptor(
            f"{lower_unit_index}T"
        )

        lower_node_R = chain.get_node_by_descriptor(
            f"{lower_unit_index}R"
        )

        lower_node_B = chain.get_node_by_descriptor(
            f"{lower_unit_index}B"
        )

        lower_node_L = chain.get_node_by_descriptor(
            f"{lower_unit_index}L"
        )

        # ---------------------------------------------------------
        # Upper FREE unit node constrained by the rails
        # ---------------------------------------------------------

        upper_node_B = chain.get_node_by_descriptor(
            f"{upper_unit_index}B"
        )

        # ---------------------------------------------------------
        # Candidate gaps
        # ---------------------------------------------------------

        gaps.extend([
            # Lower top node vs upper FREE struts
            get_node_segment_gap(
                lower_node_T,
                upper_unit_segment_BL,
                "clockwise"
            ),

            get_node_segment_gap(
                lower_node_T,
                upper_unit_segment_TR,
                "clockwise"
            ),

            get_node_segment_gap(
                lower_node_T,
                upper_unit_segment_RB,
                "clockwise"
            ),

            get_node_segment_gap(
                lower_node_T,
                upper_unit_segment_LT,
                "clockwise"
            ),

            # Lower bottom node vs upper FREE bottom struts
            get_node_segment_gap(
                lower_node_B,
                upper_unit_segment_BL,
                "counterclockwise"
            ),

            get_node_segment_gap(
                lower_node_B,
                upper_unit_segment_RB,
                "counterclockwise"
            ),

            # Lower side nodes vs upper FREE lower struts
            get_node_segment_gap(
                lower_node_L,
                upper_unit_segment_BL,
                "counterclockwise"
            ),

            get_node_segment_gap(
                lower_node_R,
                upper_unit_segment_RB,
                "counterclockwise"
            ),

            # -----------------------------------------------------
            # Rail gaps
            #
            # Upper FREE bottom node vs lower RAILED rails
            # -----------------------------------------------------

            get_node_segment_gap(
                upper_node_B,
                rail_0L,
                "clockwise"
            ),

            get_node_segment_gap(
                upper_node_B,
                rail_0R,
                "counterclockwise"
            )
        ])

    # TODO: Evaluate skip level gaps
    for base_unit_index in range(
        0,
        len(chain.units) - 2,
        2
    ):
        if verbose:
            print("Evaluating skip-level gaps")

        lower_unit_index = base_unit_index
        upper_unit_index = base_unit_index + 2

        skip_unit_index = base_unit_index + 2

        # ---------------------------------------------------------
        # Lower railed unit struts
        # ---------------------------------------------------------

        lower_unit_segment_BL = chain.get_segment_by_descriptor(
            f"{lower_unit_index}B{lower_unit_index}L"
        )

        lower_unit_segment_TR = chain.get_segment_by_descriptor(
            f"{lower_unit_index}T{lower_unit_index}R"
        )

        lower_unit_segment_RB = chain.get_segment_by_descriptor(
            f"{lower_unit_index}R{lower_unit_index}B"
        )

        lower_unit_segment_LT = chain.get_segment_by_descriptor(
            f"{lower_unit_index}L{lower_unit_index}T"
        )
        if verbose:
            print(f"Skip level lower unit segments: {lower_unit_segment_BL}, {lower_unit_segment_TR}, {lower_unit_segment_RB}, {lower_unit_segment_LT}")

        # ---------------------------------------------------------
        # Upper railed unit struts
        # ---------------------------------------------------------

        upper_unit_segment_BL = chain.get_segment_by_descriptor(
            f"{upper_unit_index}B{upper_unit_index}L"
        )

        upper_unit_segment_TR = chain.get_segment_by_descriptor(
            f"{upper_unit_index}T{upper_unit_index}R"
        )

        upper_unit_segment_RB = chain.get_segment_by_descriptor(
            f"{upper_unit_index}R{upper_unit_index}B"
        )

        upper_unit_segment_LT = chain.get_segment_by_descriptor(
            f"{upper_unit_index}L{upper_unit_index}T"
        )
        if verbose:
            print(f"Skip level upper unit segments: {upper_unit_segment_BL}, {upper_unit_segment_TR}, {upper_unit_segment_RB}, {upper_unit_segment_LT}")
        
    try:
        skip_level_gap_right = get_segment_segment_gap(
            lower_unit_segment_TR,
            upper_unit_segment_BL,
            contact_offset=chain.segment_segment_contact_offset,
            verbose=verbose
        )

        if verbose:
            print(
                "Skip level gap with right rotation: "
                f"{skip_level_gap_right.length_mm}"
            )

        gaps.append(
            skip_level_gap_right
        )

    except InvalidRAMMGeometryError:
        # The finite segments do not currently overlap, so this
        # skip-level segment pair cannot form a contact yet.
        if verbose:
            print(
                f"Skipping {lower_unit_segment_TR} and "
                f"{upper_unit_segment_BL}: "
                "finite segments do not currently overlap."
            )

        """
        skip_level_gap_left = get_segment_segment_gap(
            lower_unit_segment_LT,
            upper_unit_segment_RB,
            contact_offset=chain.segment_segment_contact_offset,
            verbose=True
        )        
        """

    # =========================================================
    # Evaluate FREE-RAILED neighboring pairs:
    #
    #     1-2
    #     3-4
    #     5-6
    #     ...
    #
    # Here:
    #     lower unit = FREE
    #     upper unit = RAILED
    #
    # We evaluate selected upper-unit nodes against
    # lower-unit struts.
    # =========================================================

    for lower_unit_index in range(
        1,
        len(chain.units) - 1,
        2
    ):
        upper_unit_index = lower_unit_index + 1
        lower_node_T = chain.get_node_by_descriptor(
            f"{lower_unit_index}T"
        )
        
        rail_2L = chain.get_rail(
            upper_unit_index,
            "left"
        )

        rail_2R = chain.get_rail(
            upper_unit_index,
            "right"
            )
        # ---------------------------------------------------------
        # Lower FREE unit diamond struts
        # ---------------------------------------------------------

        lower_unit_segment_BL = chain.get_segment_by_descriptor(
            f"{lower_unit_index}B{lower_unit_index}L"
        )

        lower_unit_segment_TR = chain.get_segment_by_descriptor(
            f"{lower_unit_index}T{lower_unit_index}R"
        )

        lower_unit_segment_RB = chain.get_segment_by_descriptor(
            f"{lower_unit_index}R{lower_unit_index}B"
        )

        lower_unit_segment_LT = chain.get_segment_by_descriptor(
            f"{lower_unit_index}L{lower_unit_index}T"
        )

        # ---------------------------------------------------------
        # Upper RAILED unit nodes
        # ---------------------------------------------------------

        upper_node_R = chain.get_node_by_descriptor(
            f"{upper_unit_index}R"
        )

        upper_node_B = chain.get_node_by_descriptor(
            f"{upper_unit_index}B"
        )

        upper_node_L = chain.get_node_by_descriptor(
            f"{upper_unit_index}L"
        )

        # ---------------------------------------------------------
        # Candidate gaps
        # ---------------------------------------------------------

        gaps.extend([
            # Right-rotation candidates
            get_node_segment_gap(
                upper_node_L,
                lower_unit_segment_LT,
                "counterclockwise"  # determine orientation
            ),

            get_node_segment_gap(
                upper_node_B,
                lower_unit_segment_RB,
                "clockwise"  # determine orientation
            ),

            # Left-rotation candidates
            get_node_segment_gap(
                upper_node_R,
                lower_unit_segment_TR,
                "counterclockwise"  # determine orientation
            ),

            get_node_segment_gap(
                upper_node_B,
                lower_unit_segment_BL,
                "clockwise"  # determine orientation
            ),

            # TODO: get rail gaps for unit 1 top node constrained by unit 2 rails
            get_node_segment_gap(
                lower_node_T,
                rail_2L,
                "clockwise"
            ),

            get_node_segment_gap(
                lower_node_T,
                rail_2R,
                "counterclockwise"
            )
        ])
    
    return gaps


def get_candidate_gaps(chain, verbose=True):
    """
    Return candidate gaps for the chain.

    Includes
    --------
    1. RAILED-FREE neighboring node-segment / rail contacts:
           (0, 1), (2, 3), ...

    2. Skip-level RAILED-RAILED segment contacts:
           (0, 2), (2, 4), ...

    3. FREE-RAILED neighboring contacts:
           (1, 2), (3, 4), ...

    4. Shared-rail FREE node-node contacts:
           1T <-> 3B along Unit 2 rail
           3T <-> 5B along Unit 4 rail
           5T <-> 7B along Unit 6 rail
           ...

       The node ordering is evaluated ALONG the local rail
       direction, not in global z.
    """

    gaps = []

    # =========================================================
    # 1. RAILED-FREE neighboring pairs:
    #
    #     0-1
    #     2-3
    #     4-5
    #     ...
    # =========================================================

    for base_unit_index in range(
        0,
        len(chain.units) - 1,
        2
    ):
        lower_unit_index = base_unit_index
        upper_unit_index = base_unit_index + 1

        # ---------------------------------------------------------
        # Upper FREE unit diamond struts
        # ---------------------------------------------------------

        upper_unit_segment_BL = (
            chain.get_segment_by_descriptor(
                f"{upper_unit_index}B"
                f"{upper_unit_index}L"
            )
        )

        upper_unit_segment_TR = (
            chain.get_segment_by_descriptor(
                f"{upper_unit_index}T"
                f"{upper_unit_index}R"
            )
        )

        upper_unit_segment_RB = (
            chain.get_segment_by_descriptor(
                f"{upper_unit_index}R"
                f"{upper_unit_index}B"
            )
        )

        upper_unit_segment_LT = (
            chain.get_segment_by_descriptor(
                f"{upper_unit_index}L"
                f"{upper_unit_index}T"
            )
        )

        # ---------------------------------------------------------
        # Lower RAILED unit rails
        # ---------------------------------------------------------

        rail_0L = chain.get_rail(
            lower_unit_index,
            "left"
        )

        rail_0R = chain.get_rail(
            lower_unit_index,
            "right"
        )

        # ---------------------------------------------------------
        # Lower RAILED unit diamond nodes
        # ---------------------------------------------------------

        lower_node_T = (
            chain.get_node_by_descriptor(
                f"{lower_unit_index}T"
            )
        )

        lower_node_R = (
            chain.get_node_by_descriptor(
                f"{lower_unit_index}R"
            )
        )

        lower_node_B = (
            chain.get_node_by_descriptor(
                f"{lower_unit_index}B"
            )
        )

        lower_node_L = (
            chain.get_node_by_descriptor(
                f"{lower_unit_index}L"
            )
        )

        # ---------------------------------------------------------
        # Upper FREE node constrained by lower RAILED rails
        # ---------------------------------------------------------

        upper_node_B = (
            chain.get_node_by_descriptor(
                f"{upper_unit_index}B"
            )
        )

        # ---------------------------------------------------------
        # Candidate gaps
        # ---------------------------------------------------------

        gaps.extend([
            # Lower top node vs upper FREE struts
            get_node_segment_gap(
                lower_node_T,
                upper_unit_segment_BL,
                "clockwise"
            ),

            get_node_segment_gap(
                lower_node_T,
                upper_unit_segment_TR,
                "clockwise"
            ),

            get_node_segment_gap(
                lower_node_T,
                upper_unit_segment_RB,
                "clockwise"
            ),

            get_node_segment_gap(
                lower_node_T,
                upper_unit_segment_LT,
                "clockwise"
            ),

            # Lower bottom node vs upper FREE lower struts
            get_node_segment_gap(
                lower_node_B,
                upper_unit_segment_BL,
                "counterclockwise"
            ),

            get_node_segment_gap(
                lower_node_B,
                upper_unit_segment_RB,
                "counterclockwise"
            ),

            # Lower side nodes vs upper FREE lower struts
            get_node_segment_gap(
                lower_node_L,
                upper_unit_segment_BL,
                "counterclockwise"
            ),

            get_node_segment_gap(
                lower_node_R,
                upper_unit_segment_RB,
                "counterclockwise"
            ),

            # Upper FREE bottom node vs lower RAILED rails
            get_node_segment_gap(
                upper_node_B,
                rail_0L,
                "clockwise"
            ),

            get_node_segment_gap(
                upper_node_B,
                rail_0R,
                "counterclockwise"
            ),
        ])

    # =========================================================
    # 2. Skip-level RAILED-RAILED pairs:
    #
    #     0-2
    #     2-4
    #     4-6
    #     ...
    # =========================================================

    for base_unit_index in range(
        0,
        len(chain.units) - 2,
        2
    ):
        lower_unit_index = base_unit_index
        upper_unit_index = base_unit_index + 2

        # ---------------------------------------------------------
        # Lower RAILED unit struts
        # ---------------------------------------------------------

        lower_unit_segment_BL = (
            chain.get_segment_by_descriptor(
                f"{lower_unit_index}B"
                f"{lower_unit_index}L"
            )
        )

        lower_unit_segment_TR = (
            chain.get_segment_by_descriptor(
                f"{lower_unit_index}T"
                f"{lower_unit_index}R"
            )
        )

        lower_unit_segment_RB = (
            chain.get_segment_by_descriptor(
                f"{lower_unit_index}R"
                f"{lower_unit_index}B"
            )
        )

        lower_unit_segment_LT = (
            chain.get_segment_by_descriptor(
                f"{lower_unit_index}L"
                f"{lower_unit_index}T"
            )
        )

        if verbose:
            print(
                "Evaluating skip-level gaps"
            )
            print(
                "Skip level lower unit segments: "
                f"{lower_unit_segment_BL}, "
                f"{lower_unit_segment_TR}, "
                f"{lower_unit_segment_RB}, "
                f"{lower_unit_segment_LT}"
            )

        # ---------------------------------------------------------
        # Upper RAILED unit struts
        # ---------------------------------------------------------

        upper_unit_segment_BL = (
            chain.get_segment_by_descriptor(
                f"{upper_unit_index}B"
                f"{upper_unit_index}L"
            )
        )

        upper_unit_segment_TR = (
            chain.get_segment_by_descriptor(
                f"{upper_unit_index}T"
                f"{upper_unit_index}R"
            )
        )

        upper_unit_segment_RB = (
            chain.get_segment_by_descriptor(
                f"{upper_unit_index}R"
                f"{upper_unit_index}B"
            )
        )

        upper_unit_segment_LT = (
            chain.get_segment_by_descriptor(
                f"{upper_unit_index}L"
                f"{upper_unit_index}T"
            )
        )

        if verbose:
            print(
                "Skip level upper unit segments: "
                f"{upper_unit_segment_BL}, "
                f"{upper_unit_segment_TR}, "
                f"{upper_unit_segment_RB}, "
                f"{upper_unit_segment_LT}"
            )

        # ---------------------------------------------------------
        # Right-rotation skip-level contact
        # ---------------------------------------------------------

        try:
            skip_level_gap_right = (
                get_segment_segment_gap(
                    lower_unit_segment_TR,
                    upper_unit_segment_BL,
                    contact_offset=(
                        chain.segment_segment_contact_offset
                    ),
                    verbose=verbose
                )
            )

            if verbose:
                print(
                    "Skip level gap with right rotation: "
                    f"{skip_level_gap_right.length_mm}"
                )

            gaps.append(
                skip_level_gap_right
            )

        except InvalidRAMMGeometryError:
            if verbose:
                print(
                    f"Skipping "
                    f"{lower_unit_segment_TR} and "
                    f"{upper_unit_segment_BL}: "
                    "finite segments do not currently overlap."
                )

    # =========================================================
    # 3. FREE-RAILED neighboring pairs:
    #
    #     1-2
    #     3-4
    #     5-6
    #     ...
    # =========================================================

    for lower_unit_index in range(
        1,
        len(chain.units) - 1,
        2
    ):
        upper_unit_index = (
            lower_unit_index + 1
        )

        lower_node_T = (
            chain.get_node_by_descriptor(
                f"{lower_unit_index}T"
            )
        )

        rail_2L = chain.get_rail(
            upper_unit_index,
            "left"
        )

        rail_2R = chain.get_rail(
            upper_unit_index,
            "right"
        )

        # ---------------------------------------------------------
        # Lower FREE unit diamond struts
        # ---------------------------------------------------------

        lower_unit_segment_BL = (
            chain.get_segment_by_descriptor(
                f"{lower_unit_index}B"
                f"{lower_unit_index}L"
            )
        )

        lower_unit_segment_TR = (
            chain.get_segment_by_descriptor(
                f"{lower_unit_index}T"
                f"{lower_unit_index}R"
            )
        )

        lower_unit_segment_RB = (
            chain.get_segment_by_descriptor(
                f"{lower_unit_index}R"
                f"{lower_unit_index}B"
            )
        )

        lower_unit_segment_LT = (
            chain.get_segment_by_descriptor(
                f"{lower_unit_index}L"
                f"{lower_unit_index}T"
            )
        )

        # ---------------------------------------------------------
        # Upper RAILED unit nodes
        # ---------------------------------------------------------

        upper_node_R = (
            chain.get_node_by_descriptor(
                f"{upper_unit_index}R"
            )
        )

        upper_node_B = (
            chain.get_node_by_descriptor(
                f"{upper_unit_index}B"
            )
        )

        upper_node_L = (
            chain.get_node_by_descriptor(
                f"{upper_unit_index}L"
            )
        )

        # ---------------------------------------------------------
        # Candidate gaps
        # ---------------------------------------------------------

        gaps.extend([
            # Right-rotation candidates
            get_node_segment_gap(
                upper_node_L,
                lower_unit_segment_LT,
                "counterclockwise"
            ),

            get_node_segment_gap(
                upper_node_B,
                lower_unit_segment_RB,
                "clockwise"
            ),

            # Left-rotation candidates
            get_node_segment_gap(
                upper_node_R,
                lower_unit_segment_TR,
                "counterclockwise"
            ),

            get_node_segment_gap(
                upper_node_B,
                lower_unit_segment_BL,
                "clockwise"
            ),

            # Lower FREE top node constrained by upper RAILED rails
            get_node_segment_gap(
                lower_node_T,
                rail_2L,
                "clockwise"
            ),

            get_node_segment_gap(
                lower_node_T,
                rail_2R,
                "counterclockwise"
            ),
        ])

    # =========================================================
    # 4. Shared-rail FREE node-node gaps
    #
    # Interior RAILED units:
    #
    #     Unit 2 rail:  1T <-> 3B
    #     Unit 4 rail:  3T <-> 5B
    #     Unit 6 rail:  5T <-> 7B
    #     ...
    #
    # The separation is measured ALONG the local rail direction.
    # =========================================================

    for railed_unit_index in range(
        2,
        len(chain.units) - 1,
        2
    ):
        lower_free_index = (
            railed_unit_index - 1
        )

        upper_free_index = (
            railed_unit_index + 1
        )

        lower_node = (
            chain.get_node_by_descriptor(
                f"{lower_free_index}T"
            )
        )

        upper_node = (
            chain.get_node_by_descriptor(
                f"{upper_free_index}B"
            )
        )

        railed_unit = (
            chain.units[
                railed_unit_index
            ]
        )

        rail_bottom = np.asarray(
            railed_unit.bottom_node.coordinates,
            dtype=float,
        )

        rail_top = np.asarray(
            railed_unit.top_node.coordinates,
            dtype=float,
        )

        rail_vector = (
            rail_top - rail_bottom
        )

        rail_length = np.linalg.norm(
            rail_vector
        )

        if rail_length <= 1e-12:
            raise InvalidRAMMGeometryError(
                f"RAILED Unit "
                f"{railed_unit_index} "
                "has zero-length centerline."
            )

        rail_direction = (
            rail_vector / rail_length
        )

        lower_position = np.asarray(
            lower_node.coordinates,
            dtype=float,
        )

        upper_position = np.asarray(
            upper_node.coordinates,
            dtype=float,
        )

        # Center-to-center spacing along the rail.
        #
        # IMPORTANT:
        # Do NOT subtract node diameter here.
        #
        # We store the centerline spacing in length_mm just like
        # the other gap candidates store centerline separation.
        # get_active_gap_vector() will use the candidate-specific
        # contact offset below.
        spacing_along_rail = float(
            np.dot(
                upper_position - lower_position,
                rail_direction,
            )
        )

        shared_rail_gap = SimpleNamespace(
            name=(
                f"gap_{lower_free_index}T_"
                f"{upper_free_index}B_"
                f"along_{railed_unit_index}_rail"
            ),

            # Identify this special candidate type.
            gap_type="shared_rail_node_node",

            # Existing candidate fields.
            node=None,
            segment_1=None,
            segment_2=None,
            orientation=None,

            # Shared-rail-specific fields.
            lower_node=lower_node,
            upper_node=upper_node,
            railed_unit=railed_unit,

            # Physical contact occurs at one node diameter.
            contact_offset=chain.node_diameter,

            # Preserve the existing candidate.result.length_mm API.
            result=SimpleNamespace(
                length_mm=spacing_along_rail
            ),
        )

        gaps.append(
            shared_rail_gap
        )

    return gaps


def get_active_gap_vector(
    candidate_gaps,
    contact_offset=0.0,
    contact_tolerance=1e-6,
    print_gaps=True,
    print_active_gap_vector=True
):
    """
    Build the symbolic active-gap vector.

    Supports:
        1. node-segment gaps
        2. segment-segment gaps
        3. shared-rail node-node gaps

    Shared-rail node-node examples:
        1T <-> 3B along Unit 2 rail
        3T <-> 5B along Unit 4 rail
        ...
    """

    active_gap_expressions = []

    print(
        f"Evaluating gaps with default contact offset: "
        f"{contact_offset} mm\n"
        f"Numerical contact tolerance: "
        f"{contact_tolerance} mm"
    )

    for candidate in candidate_gaps:

        gap_length_mm = (
            candidate.result.length_mm
        )

        # ---------------------------------------------------------
        # Candidate-specific contact offset
        #
        # Existing candidates fall back to the supplied default.
        #
        # Shared-rail node-node candidates use node diameter.
        # ---------------------------------------------------------

        candidate_contact_offset = getattr(
            candidate,
            "contact_offset",
            contact_offset,
        )

        clearance = (
            gap_length_mm
            - candidate_contact_offset
        )

        gap_type = getattr(
            candidate,
            "gap_type",
            None,
        )

        # ---------------------------------------------------------
        # Shared-rail node-node candidate
        # ---------------------------------------------------------

        is_shared_rail_node_node = (
            gap_type
            == "shared_rail_node_node"
        )

        # ---------------------------------------------------------
        # Determine display objects
        # ---------------------------------------------------------

        if is_shared_rail_node_node:

            contact_object_1 = (
                candidate.lower_node
            )

            contact_object_2 = (
                candidate.upper_node
            )

        elif candidate.node is not None:

            contact_object_1 = (
                candidate.segment_1
            )

            contact_object_2 = (
                candidate.node
            )

        elif candidate.segment_2 is not None:

            contact_object_1 = (
                candidate.segment_1
            )

            contact_object_2 = (
                candidate.segment_2
            )

        else:
            raise ValueError(
                f"Could not identify gap type for "
                f"{candidate.name}."
            )

        # ---------------------------------------------------------
        # Print numerical gap state
        # ---------------------------------------------------------

        if print_gaps:

            if clearance < -contact_tolerance:

                print(
                    f"❌ PENETRATING! "
                    f"{contact_object_1} and "
                    f"{contact_object_2}\n"
                    f"Gap: {clearance:.6f} mm"
                )

            elif math.isclose(
                clearance,
                0.0,
                abs_tol=contact_tolerance
            ):

                display_clearance = 0.0

                print(
                    f"✅ CONTACT: "
                    f"{contact_object_1} and "
                    f"{contact_object_2}\n"
                    f"Gap: "
                    f"{display_clearance:.6f} mm"
                )

            else:

                print(
                    f"Gap between "
                    f"{contact_object_1} and "
                    f"{contact_object_2} is open.\n"
                    f"Gap: {clearance:.6f} mm"
                )

        # ---------------------------------------------------------
        # Add ONLY active contact constraints
        # ---------------------------------------------------------

        if not math.isclose(
            clearance,
            0.0,
            abs_tol=contact_tolerance
        ):
            continue

        # ---------------------------------------------------------
        # Shared-rail node-node active gap
        # ---------------------------------------------------------

        if is_shared_rail_node_node:

            symbolic_gap = (
                get_shared_rail_node_node_symbolic_gap(
                    lower_node=(
                        candidate.lower_node
                    ),
                    upper_node=(
                        candidate.upper_node
                    ),
                    railed_unit=(
                        candidate.railed_unit
                    ),
                    node_diameter=(
                        candidate.contact_offset
                    ),
                )
            )

            # Your helper already represents:
            #
            #     spacing_along_rail - node_diameter
            #
            # so DO NOT subtract the contact offset again here.
            active_gap_expressions.append(
                symbolic_gap
            )

        # ---------------------------------------------------------
        # Node-segment active gap
        # ---------------------------------------------------------

        elif candidate.node is not None:

            symbolic_gap = (
                get_node_segment_symbolic_gap(
                    candidate.node,
                    candidate.segment_1,
                    candidate.orientation
                )
            )

            active_gap_expressions.append(
                symbolic_gap.signed_gap
                - candidate_contact_offset
            )

        # ---------------------------------------------------------
        # Segment-segment active gap
        # ---------------------------------------------------------

        elif candidate.segment_2 is not None:

            symbolic_gap = (
                get_segment_segment_symbolic_gap(
                    candidate.segment_1,
                    candidate.segment_2
                )
            )

            active_gap_expressions.append(
                symbolic_gap.signed_gap
                - candidate_contact_offset
            )

        else:

            raise ValueError(
                f"Gap {candidate.name} is neither "
                "node-segment, segment-segment, "
                "nor shared-rail node-node."
            )

    active_gap_vector = sp.Matrix(
        active_gap_expressions
    )

    if print_active_gap_vector:
        sp.pprint(
            active_gap_vector
        )

    return active_gap_vector


def parse_gap_coordinate_symbol(symbol):
    """
    Parse a symbolic coordinate used in a gap expression.

    Examples
    --------
    A_1B_y   -> (1, "B", "y")
    B_1L_z   -> (1, "L", "z")
    P_0T_y   -> (0, "T", "y")
    A_0RLL_y -> (0, "RLL", "y")

    0T_y     -> (0, "T", "y")
    2B_z     -> (2, "B", "z")

    The optional leading A/B/P describes the point's role in the
    gap construction and is not part of its physical identity.
    """

    # Node--segment style:
    # A_1B_y, B_1L_z, P_0T_y, ...
    match = re.fullmatch(
        r"[ABP]_(\d+)([A-Za-z]+)_([yz])",
        symbol.name,
    )

    if match is None:
        # Segment--segment style:
        # 0T_y, 2B_z, ...
        match = re.fullmatch(
            r"(\d+)([A-Za-z]+)_([yz])",
            symbol.name,
        )

    if match is None:
        return None

    unit_index = int(match.group(1))
    point_name = match.group(2)
    axis = match.group(3)

    return unit_index, point_name, axis


def get_symbolic_point_position_2unit(
    point_default,
    pivot_default,
    dz,
    theta,
):
    """
    Express a point on Unit 1 in terms of the two-unit
    generalized coordinates dz and theta.
    """

    y0, z0 = point_default
    py, pz = pivot_default

    ry = y0 - py
    rz = z0 - pz

    y = (
        py
        + ry * sp.cos(theta)
        - rz * sp.sin(theta)
    )

    z = (
        pz
        + dz
        + ry * sp.sin(theta)
        + rz * sp.cos(theta)
    )

    return {
        "y": sp.simplify(y),
        "z": sp.simplify(z),
    }


def get_symbolic_point_position(
    point_default,
    pivot_default,
    dz,
    theta,
):
    """
    Express a point position symbolically in terms of
    vertical translation and rotation only.
    """

    y0, z0 = point_default
    py, pz = pivot_default

    ry = y0 - py
    rz = z0 - pz

    y = (
        py
        + ry * sp.cos(theta)
        - rz * sp.sin(theta)
    )

    z = (
        pz
        + dz
        + ry * sp.sin(theta)
        + rz * sp.cos(theta)
    )

    return {
        "y": sp.simplify(y),
        "z": sp.simplify(z),
    }

def get_generalized_coordinates(chain):
    """
    Build the ordered generalized-coordinate vector for the chain.
    """

    q = []
    unit_coordinates = {}

    for unit_index, unit in enumerate(chain.units):

        # Unit 0 is fixed
        if unit_index == 0:
            unit_coordinates[unit_index] = {
                "dy": sp.Integer(0),
                "dz": sp.Integer(0),
                "theta": sp.Integer(0),
            }
            continue

        # FREE unit: z translation + rotation
        if unit.unit_type == UnitType.FREE:
            dz, theta = sp.symbols(
                f"z_{unit_index} theta_{unit_index}",
                real=True,
            )

            unit_coordinates[unit_index] = {
                "dy": sp.Integer(0),
                "dz": dz,
                "theta": theta,
            }

            q.extend([dz, theta])

        # Moving RAILED unit: y + z translation + rotation
        elif unit.unit_type == UnitType.RAILED:
            dz, theta = sp.symbols(
                f"z_{unit_index} theta_{unit_index}",
                real=True,
            )

            unit_coordinates[unit_index] = {
                "dz": dz,
                "theta": theta,
            }

            q.extend([dz, theta])

        else:
            raise ValueError(
                f"Unsupported unit type for Unit {unit_index}"
            )

    return sp.Matrix(q), unit_coordinates


def express_gaps_in_generalized_coordinates(
    chain,
    active_gap_vector,
    point_positions,
):
    """
    Convert an active gap vector from Cartesian coordinate symbols
    into the generalized coordinates of an n-unit chain.
    """

    q, unit_coordinates = get_generalized_coordinates(chain)

    substitutions = {}

    symbols = set()

    for gap in active_gap_vector:
        symbols.update(gap.free_symbols)

    for symbol in symbols:

        parsed = parse_gap_coordinate_symbol(symbol)

        if parsed is None:
            continue

        unit_index, point_name, axis = parsed
        point_key = f"{unit_index}{point_name}"

        if point_key not in point_positions:
            raise KeyError(
                f"No physical point found for {symbol}. "
                f"Expected point_positions['{point_key}']."
            )

        point_default = point_positions[point_key]

        # Unit 0 is fixed
        if unit_index == 0:
            axis_index = 0 if axis == "y" else 1

            substitutions[symbol] = sp.Float(
                point_default[axis_index]
            )

            continue

        # Moving units rotate about their bottom node
        pivot_default = point_positions[
            f"{unit_index}B"
        ]

        coords = unit_coordinates[unit_index]

        symbolic_position = get_symbolic_point_position(
            point_default=point_default,
            pivot_default=pivot_default,
            #dy=coords["dy"],
            dz=coords["dz"],
            theta=coords["theta"],
        )

        substitutions[symbol] = symbolic_position[axis]

    generalized_gap_vector = sp.Matrix([
        sp.simplify(gap.subs(substitutions))
        for gap in active_gap_vector
    ])

    return generalized_gap_vector, q