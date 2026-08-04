"""
Workspace functions used to examine RAMMs.
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


# ============================================================================
# Helpers
# ============================================================================
def set_two_unit_configuration(
    chain,
    theta_deg,
    z_shift,
    default_offset=10
):
    """
    Recreate the old transform_top_unit_from_reference():

    - reset to the default configuration
    - rotate unit 1 around node 1B
    - translate the entire unit vertically by z_shift
    """
    chain.reset(offset=default_offset)

    unit_1 = chain.units[1]
    pivot = unit_1.bottom_node

    unit_1.transform(
        point=pivot,
        degrees=theta_deg
    )

    unit_1.translate(
        dy=0.0,
        dz=z_shift
    )


def get_right_limit_gaps(chain):
    """
    Maximum-right-rotation contacts from the old notebook:

    1. segment 1R1B contacts node 0R
    2. segment 1B1L contacts node 0T
    """
    segment_1R1B = chain.get_segment_by_descriptor("1R1B")
    segment_1B1L = chain.get_segment_by_descriptor("1B1L")

    node_0R = chain.get_node_by_descriptor("0R")
    node_0T = chain.get_node_by_descriptor("0T")

    gap_right = get_node_segment_gap(
        node=node_0R,
        segment=segment_1R1B,
        orientation="counterclockwise"
    )

    gap_top = get_node_segment_gap(
        node=node_0T,
        segment=segment_1B1L,
        orientation="clockwise"
    )

    return gap_right, gap_top

def get_reference_side(
    chain,
    node_descriptor,
    segment_descriptor,
    orientation,
    default_offset=10
):
    """
    Return +1 or -1 so that the gap is positive on the valid side
    occupied in the default configuration.
    """
    set_two_unit_configuration(
        chain,
        theta_deg=0.0,
        z_shift=0.0,
        default_offset=default_offset
    )

    node = chain.get_node_by_descriptor(node_descriptor)
    segment = chain.get_segment_by_descriptor(segment_descriptor)

    gap = get_node_segment_gap(
        node,
        segment,
        orientation
    ).result.length_mm

    if math.isclose(gap, 0.0, abs_tol=1e-12):
        raise ValueError(
            f"Cannot determine reference side for "
            f"{segment_descriptor} and {node_descriptor}: "
            "the default gap is zero."
        )

    return 1.0 if gap > 0 else -1.0


# ============================================================================
# Public API
# ============================================================================
def find_right_max_rotation(
    chain,
    theta_guess=-38.6,
    z_guess=-1.8,
    theta_bounds=(-60.0, 0.0),
    z_bounds=(-8.0, 5.0),
    default_offset=10,
    contact_offset=0.0,
    contact_tolerance=1e-6,
    verbose=True
):
    """
    Find the maximum-right-rotation double-contact configuration.

    Target contacts:
        segment 1R1B with node 0R
        segment 1B1L with node 0T

    Parameters
    ----------
    contact_offset : float, optional
        Required centerline distance at physical contact.

        For zero-thickness geometry:
            contact_offset = 0

        For a circular node and finite-width strut:
            contact_offset = node_radius + strut_half_width
    """

    if contact_offset < 0:
        raise ValueError("contact_offset must be nonnegative.")

    # Determine which side of each segment is valid in the
    # default, nonpenetrating configuration.
    right_side = get_reference_side(
        chain=chain,
        node_descriptor="0R",
        segment_descriptor="1R1B",
        orientation="counterclockwise",
        default_offset=default_offset
    )

    top_side = get_reference_side(
        chain=chain,
        node_descriptor="0T",
        segment_descriptor="1B1L",
        orientation="clockwise",
        default_offset=default_offset
    )

    def evaluate_target_gaps(theta_deg, z_shift):
        set_two_unit_configuration(
            chain=chain,
            theta_deg=theta_deg,
            z_shift=z_shift,
            default_offset=default_offset
        )

        segment_RB = chain.get_segment_by_descriptor("1R1B")
        segment_BL = chain.get_segment_by_descriptor("1B1L")

        node_0R = chain.get_node_by_descriptor("0R")
        node_0T = chain.get_node_by_descriptor("0T")

        gap_right = get_node_segment_gap(
            node_0R,
            segment_RB,
            "counterclockwise"
        )

        gap_top = get_node_segment_gap(
            node_0T,
            segment_BL,
            "clockwise"
        )

        return (
            gap_right.result.length_mm,
            gap_top.result.length_mm
        )

    def residuals(x):
        theta_deg, z_shift = x

        signed_gap_right, signed_gap_top = evaluate_target_gaps(
            theta_deg,
            z_shift
        )

        # Reorient each signed gap so it is positive on the
        # valid side of the segment.
        separation_right = right_side * signed_gap_right
        separation_top = top_side * signed_gap_top

        # Physical contact occurs when:
        # oriented centerline separation = contact_offset
        clearance_right = separation_right - contact_offset
        clearance_top = separation_top - contact_offset

        return np.array([
            clearance_right,
            clearance_top
        ])

    solution = least_squares(
        residuals,
        x0=np.array([theta_guess, z_guess], dtype=float),
        bounds=(
            [theta_bounds[0], z_bounds[0]],
            [theta_bounds[1], z_bounds[1]]
        ),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12
    )

    theta_deg, z_shift = solution.x

    # Leave the chain in the solved configuration.
    signed_gap_right, signed_gap_top = evaluate_target_gaps(
        theta_deg,
        z_shift
    )

    separation_right = right_side * signed_gap_right
    separation_top = top_side * signed_gap_top

    clearance_right = separation_right - contact_offset
    clearance_top = separation_top - contact_offset

    residual_norm = np.linalg.norm([
        clearance_right,
        clearance_top
    ])

    valid = (
        solution.success
        and abs(clearance_right) <= contact_tolerance
        and abs(clearance_top) <= contact_tolerance
    )

    if verbose:
        print(f"Solver success: {solution.success}")
        print(f"Physically valid: {valid}")
        print(f"θ: {theta_deg:.6f}°")
        print(f"z shift: {z_shift:.6f} mm")
        print(f"Contact offset: {contact_offset:.6f} mm")
        print(f"Reference side 1R1B–0R: {right_side:+.0f}")
        print(f"Reference side 1B1L–0T: {top_side:+.0f}")

        print(
            f"Signed gap 1R1B–0R: "
            f"{signed_gap_right:.9f} mm"
        )
        print(
            f"Oriented separation 1R1B–0R: "
            f"{separation_right:.9f} mm"
        )
        print(
            f"Clearance 1R1B–0R: "
            f"{clearance_right:.9f} mm"
        )

        print(
            f"Signed gap 1B1L–0T: "
            f"{signed_gap_top:.9f} mm"
        )
        print(
            f"Oriented separation 1B1L–0T: "
            f"{separation_top:.9f} mm"
        )
        print(
            f"Clearance 1B1L–0T: "
            f"{clearance_top:.9f} mm"
        )

        print(f"Residual norm: {residual_norm:.3e}")

    return {
        "success": valid,
        "solver_success": solution.success,
        "theta_deg": theta_deg,
        "z_shift": z_shift,
        "contact_offset": contact_offset,
        "right_side": right_side,
        "top_side": top_side,
        "signed_gap_right": signed_gap_right,
        "signed_gap_top": signed_gap_top,
        "separation_right": separation_right,
        "separation_top": separation_top,
        "clearance_right": clearance_right,
        "clearance_top": clearance_top,
        "residual_norm": residual_norm,
        "solution": solution
    }


def find_left_max_rotation(
    chain,
    theta_guess=38.6,
    z_guess=-1.8,
    theta_bounds=(0.0, 60.0),
    z_bounds=(-8.0, 5.0),
    default_offset=10,
    contact_offset=0.0,
    contact_tolerance=1e-6,
    verbose=True
):
    """
    Find the maximum-left-rotation double-contact configuration.

    Target contacts:
        segment 1B1L with node 0L
        segment 1R1B with node 0T
    """

    if contact_offset < 0:
        raise ValueError("contact_offset must be nonnegative.")

    # Determine the valid side of each segment from the default configuration.
    left_side = get_reference_side(
        chain=chain,
        node_descriptor="0L",
        segment_descriptor="1B1L",
        orientation="clockwise",
        default_offset=default_offset
    )

    top_side = get_reference_side(
        chain=chain,
        node_descriptor="0T",
        segment_descriptor="1R1B",
        orientation="counterclockwise",
        default_offset=default_offset
    )

    def evaluate_target_gaps(theta_deg, z_shift):
        set_two_unit_configuration(
            chain=chain,
            theta_deg=theta_deg,
            z_shift=z_shift,
            default_offset=default_offset
        )

        segment_BL = chain.get_segment_by_descriptor("1B1L")
        segment_RB = chain.get_segment_by_descriptor("1R1B")

        node_0L = chain.get_node_by_descriptor("0L")
        node_0T = chain.get_node_by_descriptor("0T")

        gap_left = get_node_segment_gap(
            node_0L,
            segment_BL,
            "clockwise"
        )

        gap_top = get_node_segment_gap(
            node_0T,
            segment_RB,
            "counterclockwise"
        )

        return (
            gap_left.result.length_mm,
            gap_top.result.length_mm
        )

    def residuals(x):
        theta_deg, z_shift = x

        signed_gap_left, signed_gap_top = evaluate_target_gaps(
            theta_deg,
            z_shift
        )

        # Make separation positive on the valid side.
        separation_left = left_side * signed_gap_left
        separation_top = top_side * signed_gap_top

        return np.array([
            separation_left - contact_offset,
            separation_top - contact_offset
        ])

    solution = least_squares(
        residuals,
        x0=np.array([theta_guess, z_guess], dtype=float),
        bounds=(
            [theta_bounds[0], z_bounds[0]],
            [theta_bounds[1], z_bounds[1]]
        ),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12
    )

    theta_deg, z_shift = solution.x

    # Re-evaluate and leave the chain in the solved configuration.
    signed_gap_left, signed_gap_top = evaluate_target_gaps(
        theta_deg,
        z_shift
    )

    separation_left = left_side * signed_gap_left
    separation_top = top_side * signed_gap_top

    clearance_left = separation_left - contact_offset
    clearance_top = separation_top - contact_offset

    residual_norm = np.linalg.norm([
        clearance_left,
        clearance_top
    ])

    valid = (
        solution.success
        and abs(clearance_left) <= contact_tolerance
        and abs(clearance_top) <= contact_tolerance
    )

    if verbose:
        print(f"Solver success: {solution.success}")
        print(f"Physically valid: {valid}")
        print(f"θ: {theta_deg:.6f}°")
        print(f"z shift: {z_shift:.6f} mm")
        print(f"Contact offset: {contact_offset:.6f} mm")
        print(f"Reference side 1B1L–0L: {left_side:+.0f}")
        print(f"Reference side 1R1B–0T: {top_side:+.0f}")

        print(
            f"Signed gap 1B1L–0L: "
            f"{signed_gap_left:.9f} mm"
        )
        print(
            f"Oriented separation 1B1L–0L: "
            f"{separation_left:.9f} mm"
        )
        print(
            f"Clearance 1B1L–0L: "
            f"{clearance_left:.9f} mm"
        )

        print(
            f"Signed gap 1R1B–0T: "
            f"{signed_gap_top:.9f} mm"
        )
        print(
            f"Oriented separation 1R1B–0T: "
            f"{separation_top:.9f} mm"
        )
        print(
            f"Clearance 1R1B–0T: "
            f"{clearance_top:.9f} mm"
        )

        print(f"Residual norm: {residual_norm:.3e}")

    return {
        "success": valid,
        "solver_success": solution.success,
        "theta_deg": theta_deg,
        "z_shift": z_shift,
        "contact_offset": contact_offset,
        "left_side": left_side,
        "top_side": top_side,
        "signed_gap_left": signed_gap_left,
        "signed_gap_top": signed_gap_top,
        "separation_left": separation_left,
        "separation_top": separation_top,
        "clearance_left": clearance_left,
        "clearance_top": clearance_top,
        "residual_norm": residual_norm,
        "solution": solution
    }


def find_vertical_limit(
    chain,
    direction,
    default_offset=10,
    contact_offset=0.0,
    z_search=20,
    n_scan=200,
    gap_tolerance=1e-8,
    z_tolerance=1e-8,
    verbose=True
):
    """
    Find the upper or lower vertical limit of unit 1 at theta = 0.

    direction="up":
        node 0T approaches segments 1B1L and 1R1B

    direction="down":
        node 0B approaches segments 1B1L and 1R1B

    contact_offset:
        Required centerline separation at physical contact.
        Defaults to 0 for zero-thickness geometry.
    """

    if contact_offset < 0:
        raise ValueError("contact_offset must be nonnegative.")

    direction = direction.lower()

    if direction == "up":
        fixed_node_descriptor = "0T"
        z_start = 0.0
        z_end = z_search

    elif direction == "down":
        fixed_node_descriptor = "0B"
        z_start = 0.0
        z_end = -z_search

    else:
        raise ValueError("direction must be 'up' or 'down'")

    # Determine the valid side of each strut from the default configuration.
    left_side = get_reference_side(
        chain=chain,
        node_descriptor=fixed_node_descriptor,
        segment_descriptor="1B1L",
        orientation="clockwise",
        default_offset=default_offset
    )

    right_side = get_reference_side(
        chain=chain,
        node_descriptor=fixed_node_descriptor,
        segment_descriptor="1R1B",
        orientation="counterclockwise",
        default_offset=default_offset
    )

    def evaluate(z_shift):
        set_two_unit_configuration(
            chain=chain,
            theta_deg=0.0,
            z_shift=z_shift,
            default_offset=default_offset
        )

        node = chain.get_node_by_descriptor(fixed_node_descriptor)

        segment_left = chain.get_segment_by_descriptor("1B1L")
        segment_right = chain.get_segment_by_descriptor("1R1B")

        gap_left = get_node_segment_gap(
            node,
            segment_left,
            "clockwise"
        ).result.length_mm

        gap_right = get_node_segment_gap(
            node,
            segment_right,
            "counterclockwise"
        ).result.length_mm

        separation_left = left_side * gap_left
        separation_right = right_side * gap_right

        clearance_left = separation_left - contact_offset
        clearance_right = separation_right - contact_offset

        return {
            "signed_gap_left": gap_left,
            "signed_gap_right": gap_right,
            "separation_left": separation_left,
            "separation_right": separation_right,
            "clearance_left": clearance_left,
            "clearance_right": clearance_right
        }

    def contact_value(z_shift):
        result = evaluate(z_shift)

        # First of the two struts to reach physical contact.
        return min(
            result["clearance_left"],
            result["clearance_right"]
        )

    # Confirm that the default configuration is not already penetrating.
    initial_result = evaluate(0.0)

    if (
        initial_result["clearance_left"] < -gap_tolerance
        or initial_result["clearance_right"] < -gap_tolerance
    ):
        raise ValueError(
            "The default configuration is already penetrating "
            "for the specified contact_offset."
        )

    # Scan outward from the default configuration until first contact.
    z_values = np.linspace(z_start, z_end, n_scan + 1)

    previous_z = z_values[0]
    previous_clearance = contact_value(previous_z)

    bracket = None

    for current_z in z_values[1:]:
        current_clearance = contact_value(current_z)

        if (
            abs(current_clearance) <= gap_tolerance
            or previous_clearance * current_clearance < 0
        ):
            bracket = (previous_z, current_z)
            break

        previous_z = current_z
        previous_clearance = current_clearance

    if bracket is None:
        raise ValueError(
            f"No {direction} contact found within z shift "
            f"[{z_start}, {z_end}]."
        )

    z_a, z_b = bracket
    clearance_a = contact_value(z_a)

    # Refine the first-contact location using bisection.
    while abs(z_b - z_a) > z_tolerance:
        z_mid = 0.5 * (z_a + z_b)
        clearance_mid = contact_value(z_mid)

        if abs(clearance_mid) <= gap_tolerance:
            z_a = z_b = z_mid
            break

        if clearance_a * clearance_mid <= 0:
            z_b = z_mid
        else:
            z_a = z_mid
            clearance_a = clearance_mid

    z_shift = 0.5 * (z_a + z_b)

    # Re-evaluate and leave the chain at the solved position.
    result = evaluate(z_shift)

    left_in_contact = (
        abs(result["clearance_left"]) <= gap_tolerance
    )
    right_in_contact = (
        abs(result["clearance_right"]) <= gap_tolerance
    )

    no_target_penetration = (
        result["clearance_left"] >= -gap_tolerance
        and result["clearance_right"] >= -gap_tolerance
    )

    valid = (
        no_target_penetration
        and (left_in_contact or right_in_contact)
    )

    if verbose:
        print(f"Direction: {direction}")
        print(f"Physically valid: {valid}")
        print("θ: 0.000000°")
        print(f"z shift: {z_shift:.9f} mm")
        print(f"Contact offset: {contact_offset:.9f} mm")
        print(
            f"Reference side 1B1L–{fixed_node_descriptor}: "
            f"{left_side:+.0f}"
        )
        print(
            f"Reference side 1R1B–{fixed_node_descriptor}: "
            f"{right_side:+.0f}"
        )

        print(
            f"Oriented separation "
            f"1B1L–{fixed_node_descriptor}: "
            f"{result['separation_left']:.9f} mm"
        )
        print(
            f"Clearance 1B1L–{fixed_node_descriptor}: "
            f"{result['clearance_left']:.9f} mm"
        )

        print(
            f"Oriented separation "
            f"1R1B–{fixed_node_descriptor}: "
            f"{result['separation_right']:.9f} mm"
        )
        print(
            f"Clearance 1R1B–{fixed_node_descriptor}: "
            f"{result['clearance_right']:.9f} mm"
        )

    return {
        "success": valid,
        "direction": direction,
        "theta_deg": 0.0,
        "z_shift": z_shift,
        "contact_offset": contact_offset,
        "fixed_node": fixed_node_descriptor,
        "left_side": left_side,
        "right_side": right_side,
        **result
    }


