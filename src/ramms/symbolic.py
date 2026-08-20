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


def get_candidate_gaps(chain, verbose=True):
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

        skip_level_gap_right = get_segment_segment_gap(
            lower_unit_segment_TR,
            upper_unit_segment_BL,
            contact_offset=chain.segment_segment_contact_offset,
            verbose=True
        )
        """
        skip_level_gap_left = get_segment_segment_gap(
            lower_unit_segment_LT,
            upper_unit_segment_RB,
            contact_offset=chain.segment_segment_contact_offset,
            verbose=True
        )        
        """
        if verbose:
            print(f"Skip level gap with right rotation: {skip_level_gap_right.length_mm}")
            #print(f"Skip level gap with right rotation: {skip_level_gap_left.length_mm}")

        gaps.extend([
            # TODO: write method to get segment-segment gaps
            skip_level_gap_right
        ])

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


def get_active_gap_vector(
    candidate_gaps,
    contact_offset=0.0,
    contact_tolerance=1e-6,
    print_gaps=True,
    print_active_gap_vector=True
):
    active_gap_expressions = []

    print(
        f"Evaluating gaps with contact offset: "
        f"{contact_offset} mm\n"
        f"Numerical contact tolerance: "
        f"{contact_tolerance} mm"
    )

    for candidate in candidate_gaps:

        gap_length_mm = (
            candidate.result.length_mm
        )

        # Physical clearance between the bodies.
        clearance = (
            gap_length_mm
            - contact_offset
        )

        # ---------------------------------------------------------
        # Determine what segment_1 is contacting
        # ---------------------------------------------------------

        if candidate.node is not None:
            contact_object = candidate.node

        elif candidate.segment_2 is not None:
            contact_object = candidate.segment_2

        else:
            contact_object = "UNKNOWN"

        if print_gaps:
            gap_distance = gap_length_mm - contact_offset

            if gap_distance < -contact_tolerance:

                print(
                    f"❌ PENETRATING! "
                    f"{candidate.segment_1} and "
                    f"{contact_object}\n"
                    f"Gap: {gap_distance:.6f} mm"
                )

            elif math.isclose(
                gap_distance,
                0.0,
                abs_tol=contact_tolerance
            ):
                gap_distance = 0.0 # clamp close to zero values for display
                print(
                    f"✅ CONTACT: "
                    f"{candidate.segment_1} and "
                    f"{contact_object}\n"
                    f"Gap: {gap_distance:.6f} mm"
                )

            else:

                print(
                    f"Gap between "
                    f"{candidate.segment_1} and "
                    f"{contact_object} is open.\n"
                    f"Gap: {gap_distance:.6f} mm"
                )

        # Add only actual contact constraints.
        if math.isclose(
            clearance,
            0.0,
            abs_tol=contact_tolerance
        ):
            # ---------------------------------------------
            # Node-segment active gap
            # ---------------------------------------------
            if candidate.node is not None:
                symbolic_gap = get_node_segment_symbolic_gap(
                    candidate.node,
                    candidate.segment_1,
                    candidate.orientation
                )

                active_gap_expressions.append(
                    symbolic_gap.signed_gap - contact_offset
                )

            # ---------------------------------------------
            # Segment-segment active gap
            # ---------------------------------------------
            elif candidate.segment_2 is not None:
                symbolic_gap = get_segment_segment_symbolic_gap(
                    candidate.segment_1,
                    candidate.segment_2
                )

                active_gap_expressions.append(
                    symbolic_gap.signed_gap - contact_offset
                )

            else:
                raise ValueError(
                    f"Gap {candidate.name} is neither "
                    "node-segment nor segment-segment."
                )

    active_gap_vector = sp.Matrix(
        active_gap_expressions
    )

    if print_active_gap_vector:
        sp.pprint(active_gap_vector)

    return active_gap_vector

def rigid_point_2d(
    point_default,
    pivot_default,
    dz,
    theta,
):
    """
    Symbolic position of a point undergoing:
        1. translation in z by dz
        2. rotation by theta about pivot_default

    Coordinates use the RAMMS convention (y, z).
    """

    y0, z0 = point_default
    py, pz = pivot_default

    # Point coordinates relative to the rotation pivot
    ry = y0 - py
    rz = z0 - pz

    # Rigid rotation + translation
    y = py + ry * sp.cos(theta) - rz * sp.sin(theta)

    z = (
        pz
        + dz
        + ry * sp.sin(theta)
        + rz * sp.cos(theta)
    )

    return sp.Matrix([y, z])


def parse_gap_coordinate_symbol(symbol):
    """
    Parse a symbolic coordinate used in a gap expression.

    Examples
    --------
    A_1B_y  -> (1, "B", "y")
    B_1L_z  -> (1, "L", "z")
    P_0T_y  -> (0, "T", "y")
    A_0RLL_y -> (0, "RLL", "y")

    The leading A/B/P describes the point's role in the gap
    construction and is not part of its physical identity.
    """

    match = re.fullmatch(
        r"[ABP]_(\d+)([A-Za-z]+)_([yz])",
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


def express_gaps_in_generalized_coordinates(
    active_gap_vector,
    point_positions,
):
    """
    Convert a two-unit active gap vector from Cartesian
    coordinate symbols into generalized coordinates.

    Two-unit generalized coordinates:
        q = [z_1, theta_1]

    Parameters
    ----------
    active_gap_vector : sympy.Matrix
        Existing active gap vector written using symbols such as
        A_1B_y, B_1L_z, P_0T_y, etc.

    point_positions : dict
        Default/reference coordinates of every physical point that
        can appear in the active gaps.

        Example:
        {
            "0T": (0.0, 20.0),
            "0L": (-8.0, 10.0),
            "1B": (0.0, 10.0),
            "1L": (-8.0, 20.0),
            ...
        }

        Rail endpoints such as "0RLL" should also be included.

    Returns
    -------
    generalized_gap_vector : sympy.Matrix
        Same active gaps, now written in terms of z_1 and theta_1.

    q : sympy.Matrix
        Ordered generalized-coordinate vector.
    """

    z_1, theta_1 = sp.symbols(
        "z_1 theta_1",
        real=True,
    )

    q = sp.Matrix([
        z_1,
        theta_1,
    ])

    # Unit 1 rotates about its bottom node.
    pivot_default = point_positions["1B"]

    substitutions = {}

    # Find every Cartesian symbol actually used by the active gaps.
    symbols = set()

    for gap in active_gap_vector:
        symbols.update(gap.free_symbols)

    for symbol in symbols:

        parsed = parse_gap_coordinate_symbol(symbol)

        # Ignore symbols that are not Cartesian gap coordinates.
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

        # ---------------------------------------------
        # Unit 0 is fixed
        # ---------------------------------------------
        if unit_index == 0:
            axis_index = 0 if axis == "y" else 1

            substitutions[symbol] = sp.Float(
                point_default[axis_index]
            )

        # ---------------------------------------------
        # Unit 1 moves with z_1 and theta_1
        # ---------------------------------------------
        elif unit_index == 1:

            symbolic_position = (
                get_symbolic_point_position_2unit(
                    point_default=point_default,
                    pivot_default=pivot_default,
                    dz=z_1,
                    theta=theta_1,
                )
            )

            substitutions[symbol] = (
                symbolic_position[axis]
            )

        else:
            raise NotImplementedError(
                "Current implementation supports "
                "two-unit chains only."
            )

    generalized_gap_vector = sp.Matrix([
        sp.simplify(gap.subs(substitutions))
        for gap in active_gap_vector
    ])

    return generalized_gap_vector, q

