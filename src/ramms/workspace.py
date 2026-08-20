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

from .contact import get_node_segment_gap, get_segment_segment_distance
from .exceptions import InvalidRAMMGeometryError
from .core import RAMM_Chain
from .kinematics import rotate_with_cascade
from .mobility import configuration_is_valid


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
    chain.reset()

    unit_1 = chain.units[1]
    pivot = unit_1.bottom_node

    unit_1.rotate(
        pivot=pivot,
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
    contact_offset=0.0, # node radius
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


def find_right_segment_segment_contact(
    chain,
    moving_unit_index=2,
    dz_guess=-8.0,
    dz_bounds=(-30.0, 0.0),
    contact_tolerance=1e-6,
    verbose=True
):
    """
    Move the top RAILED unit downward until the skip-level
    segment pair reaches physical contact.

    Assumptions
    -----------
    The chain is already in the max-right configuration for
    Units 0 and 1.

    Target contact
    --------------
        Unit 0 segment: 0T0R
        Unit 2 segment: 2B2L

    The segments must already be parallel. This method only
    translates Unit 2 vertically; it does not rotate it.

    Physical segment contact occurs when:

        perpendicular centerline distance
            = chain.segment_segment_contact_offset
    """

    # ---------------------------------------------------------
    # 1. Get the two target segments
    # ---------------------------------------------------------

    fixed_segment = chain.get_segment_by_descriptor(
        "0T0R"
    )

    moving_segment = chain.get_segment_by_descriptor(
        "2B2L"
    )

    contact_offset = (
        chain.segment_segment_contact_offset
    )

    # ---------------------------------------------------------
    # 2. Save the current max-right configuration
    #
    # Every trial translation will start from this exact state.
    # ---------------------------------------------------------

    starting_coordinates = chain._save_coordinates()

    # ---------------------------------------------------------
    # 3. Confirm the segments are already parallel
    # ---------------------------------------------------------

    initial_result = get_segment_segment_distance(
        fixed_segment,
        moving_segment
    )

    if not initial_result["parallel"]:
        raise InvalidRAMMGeometryError(
            "Segments 0T0R and 2B2L are not parallel. "
            "Vertical translation alone cannot create the "
            "requested contact."
        )

    # ---------------------------------------------------------
    # 4. Evaluate segment clearance for a trial dz
    # ---------------------------------------------------------

    def evaluate_clearance(dz):

        # Always start from the original max-right configuration.
        chain._restore_coordinates(
            starting_coordinates
        )

        # Move only Unit 2.
        chain.translate(
            unit_index=moving_unit_index,
            dy=0.0,
            dz=dz,
            verbose=False
        )

        fixed_segment = chain.get_segment_by_descriptor(
            "0T0R"
        )

        moving_segment = chain.get_segment_by_descriptor(
            "2B2L"
        )

        result = get_segment_segment_distance(
            fixed_segment,
            moving_segment
        )

        if not result["parallel"]:
            raise InvalidRAMMGeometryError(
                "Segments unexpectedly became nonparallel "
                "during vertical translation."
            )

        perpendicular_distance = (
            result["perpendicular_distance"]
        )

        clearance = (
            perpendicular_distance
            - contact_offset
        )

        return clearance, result

    # ---------------------------------------------------------
    # 5. Residual for least-squares solver
    # ---------------------------------------------------------

    def residual(x):

        dz = x[0]

        clearance, _ = evaluate_clearance(
            dz
        )

        return np.array([
            clearance
        ])

    # ---------------------------------------------------------
    # 6. Solve for vertical translation
    # ---------------------------------------------------------

    solution = least_squares(
        residual,
        x0=np.array(
            [dz_guess],
            dtype=float
        ),
        bounds=(
            [dz_bounds[0]],
            [dz_bounds[1]]
        ),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12
    )

    dz = solution.x[0]

    # ---------------------------------------------------------
    # 7. Leave the chain at the solved contact configuration
    # ---------------------------------------------------------

    clearance, final_result = evaluate_clearance(
        dz
    )

    valid = (
        solution.success
        and abs(clearance) <= contact_tolerance
        and final_result["overlap"] is not None
        and final_result["overlap"] >= -contact_tolerance
    )

    # ---------------------------------------------------------
    # 8. Print result
    # ---------------------------------------------------------

    if verbose:
        print(
            f"Segment contact solver success: "
            f"{solution.success}"
        )

        print(
            f"Physically valid contact: "
            f"{valid}"
        )

        print(
            f"Vertical translation of Unit "
            f"{moving_unit_index}: "
            f"{dz:.6f} mm"
        )

        print(
            f"Perpendicular distance: "
            f"{final_result['perpendicular_distance']:.9f} mm"
        )

        print(
            f"Contact offset: "
            f"{contact_offset:.9f} mm"
        )

        print(
            f"Clearance: "
            f"{clearance:.9f} mm"
        )

        print(
            f"Projected overlap: "
            f"{final_result['overlap']:.9f} mm"
        )

    return {
        "success": valid,
        "solver_success": solution.success,
        "dz": dz,
        "contact_offset": contact_offset,
        "perpendicular_distance": (
            final_result["perpendicular_distance"]
        ),
        "clearance": clearance,
        "overlap": final_result["overlap"],
        "solution": solution
    }


def find_two_unit_jamming_candidate(
    chain,
    direction="right",
    verbose=True,
):
    """
    Move a 2-unit chain to its limiting rotational configuration.

    Parameters
    ----------
    chain : RAMM_Chain
        Two-unit chain to configure.

    direction : {"left", "right"}
        Direction Unit 1 rotates toward its limiting configuration.

    verbose : bool
        Print solver information.

    Returns
    -------
    dict
        Solver result for the limiting configuration.

    Notes
    -----
    The input chain is left in the solved configuration.
    """

    if len(chain.units) != 2:
        raise ValueError(
            "find_two_unit_jamming_candidate requires a 2-unit chain."
        )

    direction = direction.lower()

    contact_offset = chain.node_strut_contact_offset

    if direction == "right":
        result = find_right_max_rotation(
            chain=chain,
            contact_offset=contact_offset,
            verbose=verbose,
        )

    elif direction == "left":
        result = find_left_max_rotation(
            chain=chain,
            contact_offset=contact_offset,
            verbose=verbose,
        )

    else:
        raise ValueError(
            "direction must be 'left' or 'right'."
        )

    if not result["success"]:
        raise InvalidRAMMGeometryError(
            f"Could not find the two-unit {direction} "
            "limiting configuration."
        )

    return {
        "success": True,
        "direction": direction,
        "theta_deg": result["theta_deg"],
        "z_shift": result["z_shift"],
        "solver_result": result,
    }


def find_three_unit_jamming_candidate(
    chain,
    direction="right",
    verbose=True
):
    """
    Construct the first-stage candidate jamming configuration
    for a 3-unit chain.

    Workflow
    --------
    1. get max angle for units 1 and 2 (its 38.66 with no offset)
    2. Solve the corresponding 2-unit max-rotation configuration.
    3. Extract the solved angle and z shift.
    4. Apply that motion to Unit 1 of the 3-unit chain using
       cascading motion.
    5. Verify the selected Unit 0 / Unit 2 segments are parallel.
    6. Translate Unit 2 vertically until those segments contact.
    """

    direction = direction.lower()
    print(f"direction: {direction}")

    # ---------------------------------------------------------
    # 1. Determine the actual Unit 0 -> Unit 1 offset
    # ---------------------------------------------------------

    coords_0B = chain.units[0].bottom_node.coordinates
    coords_1B = chain.units[1].bottom_node.coordinates
    coords_2B = chain.units[2].bottom_node.coordinates

    if verbose:
        print("Initial bottom node coordinates for each unit:\n"
            f"unit_0: {coords_0B}\n"
            f"unit_1: {coords_1B}\n"
            f"unit_2: {coords_2B}\n"
        )
    # ---------------------------------------------------------
    # 5. Apply the solved rotation WITH cascade
    #
    # Unit 1 rotates.
    # Unit 2 moves in response to the upper rail constraint.
    # ---------------------------------------------------------
    # TODO: rotation angle should depend on node_diameter
    # Contact offsets come directly from the chain geometry.
    node_strut_contact_offset = (
        chain.node_strut_contact_offset
    )

    segment_segment_contact_offset = (
        chain.segment_segment_contact_offset
    )

    if verbose:
        print(
            "Contact offsets:\n"
            f"node-strut: {node_strut_contact_offset}\n"
            f"segment-segment: {segment_segment_contact_offset}"
        )
    #theta_deg = 38.66
    #theta_deg = 38.65980825409009
    # TODO: run find max rotation to get theta_deg
    # ---------------------------------------------------------
    # Find the Unit 0-1 limiting configuration on a TEMPORARY
    # 2-unit chain.
    #
    # Do not run find_left/right_max_rotation() directly on the
    # 3-unit chain because that solver leaves its input chain in
    # the solved configuration.
    # ---------------------------------------------------------

    lower_offset = (
        coords_1B[0] - coords_0B[0],
        coords_1B[1] - coords_0B[1]
    )

    two_unit_chain = RAMM_Chain.generate(
        n_units=2,
        start_position=coords_0B,
        offsets=lower_offset,
        node_diameter=chain.node_diameter,
        strut_width=chain.strut_width
    )

    if direction == "right":
        result = find_right_max_rotation(
            two_unit_chain,
            contact_offset=node_strut_contact_offset,
            verbose=False
        )

    elif direction == "left":
        result = find_left_max_rotation(
            two_unit_chain,
            contact_offset=node_strut_contact_offset,
            verbose=False
        )

    else:
        raise ValueError(
            "direction must be 'left' or 'right'."
        )

    if not result["success"]:
        raise InvalidRAMMGeometryError(
            "Could not find the Unit 0-1 limiting configuration."
        )

    theta_deg = result["theta_deg"]
    z_shift = result["z_shift"]

    if verbose:
        print("\nTwo-unit limiting solution:")
        print(f"  theta = {theta_deg:.6f}°")
        print(f"  z shift = {z_shift:.6f} mm")


    # ---------------------------------------------------------
    # Apply the vertical shift from the 2-unit solution.
    # ---------------------------------------------------------

    chain.translate(
        unit_index=1,
        dy=0.0,
        dz=z_shift,
        verbose=verbose
    )

    pivot = chain.units[1].bottom_node

    cascade_result = rotate_with_cascade(
        chain=chain,
        unit_index=1,
        pivot=pivot,
        degrees=theta_deg,
        constraint_validator=configuration_is_valid, # why are all the rail gaps flipped?
        verbose=False
    )

    if verbose:
        print(
            "\nApplied two-unit solution to "
            "three-unit chain with cascade."
        )
        print(
            f"  requested rotation = "
            f"{theta_deg:.6f} deg"
        )
        print(
            f"  actual rotation = "
            f"{cascade_result['actual_degrees']:.6f} deg"
        )

    # TODO: move unit 2 down until segments 0T0R and 2B2L are in contact
    segment_contact_result = (
        find_right_segment_segment_contact(
            chain=chain,
            verbose=verbose
        )
    )
    # ---------------------------------------------------------
    # 8. Return the solved information
    # ---------------------------------------------------------

    return {
        "success": True,
        "direction": direction,
        #"lower_offset": lower_offset,
        #"two_unit_solution": lower_limit,
        "theta_deg": theta_deg,
        #"z_shift": z_shift,
        "cascade_result": cascade_result,
        "segment_contact": segment_contact_result
    }
