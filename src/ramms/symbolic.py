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
    gaps = []

    for lower_unit in range(0, len(chain.units) - 1, 2):
        upper_unit = lower_unit + 1

        segment_BL = chain.get_segment_by_descriptor(
            f"{upper_unit}B{upper_unit}L"
        )
        segment_TR = chain.get_segment_by_descriptor(
            f"{upper_unit}T{upper_unit}R"
        )
        segment_RB = chain.get_segment_by_descriptor(
            f"{upper_unit}R{upper_unit}B"
        )
        segment_LT = chain.get_segment_by_descriptor(
            f"{upper_unit}L{upper_unit}T"
        )

        node_T = chain.get_node_by_descriptor(f"{lower_unit}T")
        node_R = chain.get_node_by_descriptor(f"{lower_unit}R")
        node_B = chain.get_node_by_descriptor(f"{lower_unit}B")
        node_L = chain.get_node_by_descriptor(f"{lower_unit}L")

        gaps.extend([
            get_node_segment_gap(node_T, segment_BL, "clockwise"),
            get_node_segment_gap(node_T, segment_TR, "clockwise"),
            get_node_segment_gap(node_T, segment_RB, "clockwise"),
            get_node_segment_gap(node_T, segment_LT, "clockwise"),

            get_node_segment_gap(node_B, segment_BL, "counterclockwise"),
            get_node_segment_gap(node_B, segment_RB, "counterclockwise"),

            get_node_segment_gap(node_L, segment_BL, "counterclockwise"),
            get_node_segment_gap(node_R, segment_RB, "counterclockwise"),
        ])

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
        display(active_gap_vector)

    return active_gap_vector
