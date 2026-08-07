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


from .contact import get_node_segment_gap

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


def get_candidate_gaps(chain):
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
    for lower_unit_index in range(
        0,
        len(chain.units) - 1,
        2
    ):
        upper_unit_index = lower_unit_index + 1

        # ---------------------------------------------------------
        # Upper FREE unit diamond struts
        # ---------------------------------------------------------

        segment_BL = chain.get_segment_by_descriptor(
            f"{upper_unit_index}B{upper_unit_index}L"
        )

        segment_TR = chain.get_segment_by_descriptor(
            f"{upper_unit_index}T{upper_unit_index}R"
        )

        segment_RB = chain.get_segment_by_descriptor(
            f"{upper_unit_index}R{upper_unit_index}B"
        )

        segment_LT = chain.get_segment_by_descriptor(
            f"{upper_unit_index}L{upper_unit_index}T"
        )

        # ---------------------------------------------------------
        # Lower RAILED unit rails
        # ---------------------------------------------------------

        rail_L = chain.get_rail(
            lower_unit_index,
            "left"
        )

        rail_R = chain.get_rail(
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
                segment_BL,
                "clockwise"
            ),

            get_node_segment_gap(
                lower_node_T,
                segment_TR,
                "clockwise"
            ),

            get_node_segment_gap(
                lower_node_T,
                segment_RB,
                "clockwise"
            ),

            get_node_segment_gap(
                lower_node_T,
                segment_LT,
                "clockwise"
            ),

            # Lower bottom node vs upper FREE bottom struts
            get_node_segment_gap(
                lower_node_B,
                segment_BL,
                "counterclockwise"
            ),

            get_node_segment_gap(
                lower_node_B,
                segment_RB,
                "counterclockwise"
            ),

            # Lower side nodes vs upper FREE lower struts
            get_node_segment_gap(
                lower_node_L,
                segment_BL,
                "counterclockwise"
            ),

            get_node_segment_gap(
                lower_node_R,
                segment_RB,
                "counterclockwise"
            ),

            # -----------------------------------------------------
            # Rail gaps
            #
            # Upper FREE bottom node vs lower RAILED rails
            # -----------------------------------------------------

            get_node_segment_gap(
                upper_node_B,
                rail_L,
                "clockwise"
            ),

            get_node_segment_gap(
                upper_node_B,
                rail_R,
                "counterclockwise"
            )
        ])

        # ---------------------------------------------------------
        # Temporary debugging
        # ---------------------------------------------------------

        left_rail_gap = get_node_segment_gap(
            upper_node_B,
            rail_L,
            "counterclockwise"
        )

        right_rail_gap = get_node_segment_gap(
            upper_node_B,
            rail_R,
            "clockwise"
        )

        print(
            f"\nRail gaps between "
            f"{upper_node_B.descriptor} and "
            f"Unit {lower_unit_index} rails:"
        )

        print(
            f"  {rail_L.descriptor} -> "
            f"{upper_node_B.descriptor}: "
            f"{left_rail_gap.length_mm:.6f} mm"
        )

        print(
            f"  {rail_R.descriptor} -> "
            f"{upper_node_B.descriptor}: "
            f"{right_rail_gap.length_mm:.6f} mm"
        )

    return gaps

def get_active_gap_vector(candidate_gaps, tolerance=1e-6, verbose=False):
    active_gap_expressions = []

    for candidate in candidate_gaps:
        gap_length_mm = candidate.result.length_mm

        if verbose:
            if gap_length_mm < -tolerance:
                print(
                    f"❌ PENETRATING! {candidate.segment_1} and {candidate.node}\n"
                    f"Gap: {gap_length_mm:.6f} mm"
                )

            elif gap_length_mm <= tolerance:
                print(
                    f"✅ CONTACT: {candidate.segment_1} and {candidate.node}\n"
                    f"Gap: {gap_length_mm:.6f} mm"
                )

            else:
                print(
                    f"Gap between {candidate.segment_1} and {candidate.node} is greater than 0.\n"
                    f"Gap: {gap_length_mm:.6f} mm"
                )

        if math.isclose(gap_length_mm, 0.0, abs_tol=tolerance):
            symbolic_gap = get_node_segment_symbolic_gap(
                candidate.node,
                candidate.segment_1,
                candidate.orientation
            )
            active_gap_expressions.append(symbolic_gap.signed_gap)
    
    active_gap_vector = sp.Matrix(active_gap_expressions)

    if verbose:
        sp.pprint(active_gap_vector)

    return active_gap_vector
