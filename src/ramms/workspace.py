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
from .core import (
    UnitType,
    RAMM_Chain,
)
from .kinematics import (
    _translate_units_from,
    find_max_valid_translation,
    set_adjacent_unit_configuration,
    enforce_shared_rail_node_spacing,
    propagate_free_unit_rotation,
)
from .mobility import configuration_is_valid


# ============================================================================
# Helpers
# ============================================================================
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
):
    """
    Return +1 or -1 so that the gap is positive on the valid side
    occupied in the current reference configuration.
    """

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
            "the reference gap is zero."
        )

    return 1.0 if gap > 0 else -1.0

# ============================================================================
# Contact Finders
# ============================================================================
def find_right_segment_segment_contact(
    chain,
    lower_unit_index,
    dz_guess=-8.0,
    dz_bounds=(-30.0, 0.0),
    contact_tolerance=1e-6,
    constraint_validator=None,
    verbose=True
):
    """
    Move the upper skip-level RAILED unit downward toward the
    selected segment-segment contact.

    Assumptions
    -----------
    The chain is already in the max-right configuration for
    the relevant lower subset.

    Target contact
    --------------
        Lower RAILED unit segment:  T -> R
        Upper RAILED unit segment:  B -> L

    The segments must already be parallel. This method only
    translates the upper skip-level unit vertically; it does not
    rotate it.

    Physical segment contact occurs when:

        perpendicular centerline distance
            = chain.segment_segment_contact_offset

    Motion limiting contacts
    ------------------------
    During the downward translation, the neighboring FREE unit may
    contact the moving RAILED unit before the desired skip-level
    segment-segment contact is reached.

    For right rotation, specifically monitor:

        moving Unit L node
            against
        intermediate FREE Unit L-T segment

    and

        moving Unit B node
            against
        intermediate FREE Unit R-B segment

    If either contact becomes limiting first, stop at the last
    nonpenetrating configuration and return that result.
    """

    moving_unit_index = (
        lower_unit_index + 2
    )

    intermediate_unit_index = (
        lower_unit_index + 1
    )

    if moving_unit_index >= len(chain.units):
        raise ValueError(
            "No skip-level unit exists above "
            f"Unit {lower_unit_index}."
        )

    # ---------------------------------------------------------
    # 1. Get the target skip-level segments
    # ---------------------------------------------------------

    fixed_segment = chain.get_segment_by_descriptor(
        f"{lower_unit_index}T{lower_unit_index}R"
    )

    moving_segment = chain.get_segment_by_descriptor(
        f"{moving_unit_index}B{moving_unit_index}L"
    )

    contact_offset = (
        chain.segment_segment_contact_offset
    )

    # ---------------------------------------------------------
    # 2. Get the two node-segment pairs that may limit the
    #    downward motion first.
    #
    # Right-rotation case:
    #
    #     Unit 2L <-> Unit 1 L-T
    #     Unit 2B <-> Unit 1 R-B
    #
    # Generalized here using the supplied lower_unit_index.
    # ---------------------------------------------------------

    moving_left_node = (
        chain.get_node_by_descriptor(
            f"{moving_unit_index}L"
        )
    )

    moving_bottom_node = (
        chain.get_node_by_descriptor(
            f"{moving_unit_index}B"
        )
    )

    intermediate_left_segment = (
        chain.get_segment_by_descriptor(
            f"{intermediate_unit_index}L"
            f"{intermediate_unit_index}T"
        )
    )

    intermediate_right_segment = (
        chain.get_segment_by_descriptor(
            f"{intermediate_unit_index}R"
            f"{intermediate_unit_index}B"
        )
    )

    node_strut_contact_offset = (
        chain.node_strut_contact_offset
    )

    # ---------------------------------------------------------
    # 3. Save the current max-right configuration
    #
    # Every trial translation starts from this exact state.
    # ---------------------------------------------------------

    starting_coordinates = (
        chain._save_coordinates()
    )

    # ---------------------------------------------------------
    # 4. Confirm the target segments are already parallel
    # ---------------------------------------------------------

    initial_result = (
        get_segment_segment_distance(
            fixed_segment,
            moving_segment
        )
    )

    if not initial_result["parallel"]:
        raise InvalidRAMMGeometryError(
            f"Segments "
            f"{lower_unit_index}T{lower_unit_index}R and "
            f"{moving_unit_index}B{moving_unit_index}L "
            "are not parallel. Vertical translation alone "
            "cannot create the requested contact."
        )

    # ---------------------------------------------------------
    # 5. Evaluate the geometry at a trial dz
    # ---------------------------------------------------------

    def evaluate_configuration(dz):

        # Always start from the original max-right configuration.
        chain._restore_coordinates(
            starting_coordinates
        )

        # Move the upper skip-level unit and everything above it.
        _translate_units_from(
            chain,
            start_unit_index=moving_unit_index,
            dy=0.0,
            dz=dz
        )

        # -----------------------------------------------------
        # A. Target skip-level segment-segment clearance
        # -----------------------------------------------------

        fixed_segment_current = (
            chain.get_segment_by_descriptor(
                f"{lower_unit_index}T"
                f"{lower_unit_index}R"
            )
        )

        moving_segment_current = (
            chain.get_segment_by_descriptor(
                f"{moving_unit_index}B"
                f"{moving_unit_index}L"
            )
        )

        segment_result = (
            get_segment_segment_distance(
                fixed_segment_current,
                moving_segment_current
            )
        )

        if not segment_result["parallel"]:
            raise InvalidRAMMGeometryError(
                "Segments unexpectedly became nonparallel "
                "during vertical translation."
            )

        segment_clearance = (
            segment_result["perpendicular_distance"]
            - contact_offset
        )

        # -----------------------------------------------------
        # B. Moving L node against intermediate L-T segment
        # -----------------------------------------------------

        left_gap = get_node_segment_gap(
            moving_left_node,
            intermediate_left_segment,
            "counterclockwise",
        )

        left_clearance = (
            left_gap.result.length_mm
            - node_strut_contact_offset
        )

        # -----------------------------------------------------
        # C. Moving B node against intermediate R-B segment
        # -----------------------------------------------------

        bottom_gap = get_node_segment_gap(
            moving_bottom_node,
            intermediate_right_segment,
            "clockwise",
        )

        bottom_clearance = (
            bottom_gap.result.length_mm
            - node_strut_contact_offset
        )

        return {
            "segment_clearance": (
                segment_clearance
            ),
            "segment_result": (
                segment_result
            ),
            "left_clearance": (
                left_clearance
            ),
            "bottom_clearance": (
                bottom_clearance
            ),
        }

    # ---------------------------------------------------------
    # 6. Residual for desired segment-segment contact
    # ---------------------------------------------------------

    def residual(x):

        result = evaluate_configuration(
            x[0]
        )

        return np.array([
            result[
                "segment_clearance"
            ]
        ])

    # ---------------------------------------------------------
    # 7. Solve for the dz that WOULD produce the desired
    #    skip-level segment-segment contact.
    #
    # We have not yet committed to moving that far.
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

    target_dz = float(
        solution.x[0]
    )

    # ---------------------------------------------------------
    # 8. Define validity of the downward translation itself
    #
    # The two nearby node-segment contacts are allowed to reach
    # contact, but they may NOT penetrate.
    #
    # An optional external validator can still reject additional
    # chain-wide penetrations.
    # ---------------------------------------------------------

    def trial_configuration_is_valid():

        result = evaluate_configuration(
            current_trial_dz[0]
        )

        node_contacts_valid = (
            result["left_clearance"]
            >= -contact_tolerance
            and result["bottom_clearance"]
            >= -contact_tolerance
        )

        if not node_contacts_valid:
            return False

        if constraint_validator is not None:
            return constraint_validator(
                chain
            )

        return True

    # ---------------------------------------------------------
    # 9. Search from dz = 0 toward the desired segment-contact
    #    dz and stop at the first intervening constraint.
    #
    # find_max_valid_translation() expects:
    #
    #     apply_translation(dz)
    #     constraint_validator()
    #
    # so keep track of which dz it most recently applied.
    # ---------------------------------------------------------

    current_trial_dz = [0.0]

    def apply_trial_translation(
        trial_dz
    ):

        current_trial_dz[0] = (
            trial_dz
        )

        evaluate_configuration(
            trial_dz
        )

    motion_limit_result = (
        find_max_valid_translation(
            target_translation=target_dz,
            apply_translation=(
                apply_trial_translation
            ),
            constraint_validator=(
                trial_configuration_is_valid
            ),
        )
    )

    dz = float(
        motion_limit_result[
            "translation"
        ]
    )

    # ---------------------------------------------------------
    # 10. Leave chain at the final reachable configuration
    # ---------------------------------------------------------

    final_result = (
        evaluate_configuration(
            dz
        )
    )

    segment_clearance = (
        final_result[
            "segment_clearance"
        ]
    )

    left_clearance = (
        final_result[
            "left_clearance"
        ]
    )

    bottom_clearance = (
        final_result[
            "bottom_clearance"
        ]
    )

    segment_result = (
        final_result[
            "segment_result"
        ]
    )

    # ---------------------------------------------------------
    # 11. Determine WHY we stopped
    # ---------------------------------------------------------

    segment_contact_reached = (
        solution.success
        and abs(segment_clearance)
        <= contact_tolerance
        and segment_result["overlap"]
        is not None
        and segment_result["overlap"]
        >= -contact_tolerance
    )

    left_contact_reached = (
        abs(left_clearance)
        <= contact_tolerance
    )

    bottom_contact_reached = (
        abs(bottom_clearance)
        <= contact_tolerance
    )

    constraint_limited = (
        motion_limit_result[
            "constraint_limited"
        ]
    )

    limiting_contacts = []

    if left_contact_reached:
        limiting_contacts.append(
            f"{moving_unit_index}L-"
            f"{intermediate_unit_index}L"
            f"{intermediate_unit_index}T"
        )

    if bottom_contact_reached:
        limiting_contacts.append(
            f"{moving_unit_index}B-"
            f"{intermediate_unit_index}R"
            f"{intermediate_unit_index}B"
        )

    # Finding either the requested segment contact OR an earlier
    # physical motion limit is a successful finder operation.
    success = (
        segment_contact_reached
        or constraint_limited
    )

    # ---------------------------------------------------------
    # 12. Print result
    # ---------------------------------------------------------

    if verbose:

        print(
            f"Segment contact solver success: "
            f"{solution.success}"
        )

        print(
            f"Desired segment-contact dz: "
            f"{target_dz:.6f} mm"
        )

        print(
            f"Actual reachable dz: "
            f"{dz:.6f} mm"
        )

        print(
            f"Segment-segment clearance: "
            f"{segment_clearance:.9f} mm"
        )

        print(
            f"{moving_unit_index}L-"
            f"{intermediate_unit_index}L"
            f"{intermediate_unit_index}T clearance: "
            f"{left_clearance:.9f} mm"
        )

        print(
            f"{moving_unit_index}B-"
            f"{intermediate_unit_index}R"
            f"{intermediate_unit_index}B clearance: "
            f"{bottom_clearance:.9f} mm"
        )

        if segment_contact_reached:

            print(
                "\nDesired skip-level "
                "segment-segment contact reached."
            )

        elif constraint_limited:

            print(
                "\nDesired segment-segment contact "
                "could not be reached by vertical "
                "translation alone."
            )

            print(
                "Stopped at the closest physically "
                "valid configuration."
            )

            if limiting_contacts:
                print(
                    "Limiting contact(s): "
                    + ", ".join(
                        limiting_contacts
                    )
                )

        print(
            f"Projected overlap: "
            f"{segment_result['overlap']:.9f} mm"
        )

    # ---------------------------------------------------------
    # 13. Return
    # ---------------------------------------------------------

    return {
        "success": success,
        "solver_success": (
            solution.success
        ),
        "lower_unit_index": (
            lower_unit_index
        ),
        "moving_unit_index": (
            moving_unit_index
        ),
        "target_dz": (
            target_dz
        ),
        "dz": (
            dz
        ),
        "segment_contact_reached": (
            segment_contact_reached
        ),
        "constraint_limited": (
            constraint_limited
        ),
        "limiting_contacts": (
            limiting_contacts
        ),
        "contact_offset": (
            contact_offset
        ),
        "perpendicular_distance": (
            segment_result[
                "perpendicular_distance"
            ]
        ),
        "clearance": (
            segment_clearance
        ),
        "left_clearance": (
            left_clearance
        ),
        "bottom_clearance": (
            bottom_clearance
        ),
        "overlap": (
            segment_result[
                "overlap"
            ]
        ),
        "solution": (
            solution
        ),
    }

# ============================================================================
# Motion Limiting Finders
# ============================================================================
def find_right_limiting_configuration_two_unit(
    chain,
    lower_unit_index=0,
    moving_unit_index=1,
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
        segment moving_unit R-B with lower_unit R
        segment moving_unit B-L with lower_unit T

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

    if not 0 <= lower_unit_index < len(chain.units):
        raise IndexError(
            f"Invalid lower unit index: {lower_unit_index}"
        )

    if not 0 <= moving_unit_index < len(chain.units):
        raise IndexError(
            f"Invalid moving unit index: {moving_unit_index}"
        )

    # Save the configuration at the beginning of the solve.
    # Every trial configuration is evaluated relative to this state.
    starting_coordinates = chain._save_coordinates()

    node_R_descriptor = f"{lower_unit_index}R"
    node_T_descriptor = f"{lower_unit_index}T"

    segment_RB_descriptor = (
        f"{moving_unit_index}R{moving_unit_index}B"
    )

    segment_BL_descriptor = (
        f"{moving_unit_index}B{moving_unit_index}L"
    )

    # Determine which side of each segment is valid in the
    # default, nonpenetrating configuration.
    right_side = get_reference_side(
        chain=chain,
        node_descriptor="0R",
        segment_descriptor="1R1B",
        orientation="counterclockwise",
    )

    top_side = get_reference_side(
        chain=chain,
        node_descriptor="0T",
        segment_descriptor="1B1L",
        orientation="clockwise",
    )

    def evaluate_target_gaps(theta_deg, z_shift):
        set_adjacent_unit_configuration(
            chain=chain,
            moving_unit_index=moving_unit_index,
            starting_coordinates=starting_coordinates,
            theta_deg=theta_deg,
            z_shift=z_shift,
        )

        segment_RB = chain.get_segment_by_descriptor(
            segment_RB_descriptor
        )

        segment_BL = chain.get_segment_by_descriptor(
            segment_BL_descriptor
        )

        node_R = chain.get_node_by_descriptor(
            node_R_descriptor
        )

        node_T = chain.get_node_by_descriptor(
            node_T_descriptor
        )

        gap_right = get_node_segment_gap(
            node_R,
            segment_RB,
            "counterclockwise"
        )

        gap_top = get_node_segment_gap(
            node_T,
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
        print(
            f"Reference side "
            f"{segment_RB_descriptor}–{node_R_descriptor}: "
            f"{right_side:+.0f}"
        )
        print(
            f"Reference side "
            f"{segment_BL_descriptor}–{node_T_descriptor}: "
            f"{top_side:+.0f}"
        )

        print(
            f"Signed gap "
            f"{segment_RB_descriptor}–{node_R_descriptor}: "
            f"{signed_gap_right:.9f} mm"
        )
        print(
            f"Oriented separation "
            f"{segment_RB_descriptor}–{node_R_descriptor}: "
            f"{separation_right:.9f} mm"
        )
        print(
            f"Clearance "
            f"{segment_RB_descriptor}–{node_R_descriptor}: "
            f"{clearance_right:.9f} mm"
        )

        print(
            f"Signed gap "
            f"{segment_BL_descriptor}–{node_T_descriptor}: "
            f"{signed_gap_top:.9f} mm"
        )
        print(
            f"Oriented separation "
            f"{segment_BL_descriptor}–{node_T_descriptor}: "
            f"{separation_top:.9f} mm"
        )
        print(
            f"Clearance "
            f"{segment_BL_descriptor}–{node_T_descriptor}: "
            f"{clearance_top:.9f} mm"
        )

        print(f"Residual norm: {residual_norm:.3e}")

    return {
        "success": valid,
        "solver_success": solution.success,
        "lower_unit_index": lower_unit_index,
        "moving_unit_index": moving_unit_index,
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


def find_left_limiting_configuration_two_unit(
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
        set_adjacent_unit_configuration(
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
        set_adjacent_unit_configuration(
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


def find_limiting_configuration_two_unit(
    chain,
    direction,
    verbose=True,
):
    """
    Move a 2-unit chain to its limiting rotational configuration.
    Rotates right by default.

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
        result = find_right_limiting_configuration_two_unit(
            chain=chain,
            contact_offset=contact_offset,
            verbose=verbose,
        )

    elif direction == "left":
        result = find_left_limiting_configuration_two_unit(
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


def _try_limiting_configuration_three_unit(
    chain,
    start_unit_index=0,
    direction="right",
    theta_deg_override=None,
    contact_tolerance=1e-6,
    verbose=True,
):
    """
    Attempt to construct a 3-unit limiting configuration for one
    prescribed Unit-1 rotation angle.

    This contains the original three-unit construction logic.

    The normal 2-unit limiting configuration is still solved first
    to obtain the reference z shift and maximum rotation.

    If theta_deg_override is supplied, that rotation is used instead
    of the full 2-unit limiting rotation.

    The attempt is considered to have reached the desired 3-unit
    configuration only when the skip-level segment-segment contact
    is actually reached.

    If an intervening node-segment contact limits the downward
    translation first, that information is returned to the caller
    so a smaller Unit-1 rotation can be attempted.
    """

    direction = direction.lower()

    if verbose:
        print(
            f"direction: {direction}"
        )

    # Record the sequence of motions actually applied to the chain.
    motions = []

    # Indices of the three-unit subset.
    unit_0_index = start_unit_index
    unit_1_index = start_unit_index + 1
    unit_2_index = start_unit_index + 2

    if unit_2_index >= len(chain.units):
        raise ValueError(
            "The requested three-unit subset extends "
            "beyond the end of the chain."
        )

    # ---------------------------------------------------------
    # 1. Determine the actual Unit 0 -> Unit 1 offset
    # ---------------------------------------------------------

    coords_0B = (
        chain.units[
            unit_0_index
        ].bottom_node.coordinates
    )

    coords_1B = (
        chain.units[
            unit_1_index
        ].bottom_node.coordinates
    )

    coords_2B = (
        chain.units[
            unit_2_index
        ].bottom_node.coordinates
    )

    if verbose:
        print(
            "Initial bottom node coordinates for each unit:\n"
            f"unit_{unit_0_index}: {coords_0B}\n"
            f"unit_{unit_1_index}: {coords_1B}\n"
            f"unit_{unit_2_index}: {coords_2B}\n"
        )

    # ---------------------------------------------------------
    # 2. Contact offsets come directly from chain geometry
    # ---------------------------------------------------------

    node_strut_contact_offset = (
        chain.node_strut_contact_offset
    )

    segment_segment_contact_offset = (
        chain.segment_segment_contact_offset
    )

    if verbose:
        print(
            "Contact offsets:\n"
            f"node-strut: "
            f"{node_strut_contact_offset}\n"
            f"segment-segment: "
            f"{segment_segment_contact_offset}"
        )

    # ---------------------------------------------------------
    # 3. Find the Unit 0-1 limiting configuration on a TEMPORARY
    #    2-unit chain.
    #
    # Do not run find_left/right_max_rotation() directly on the
    # 3-unit chain because that solver leaves its input chain in
    # the solved configuration.
    # ---------------------------------------------------------

    lower_offset = (
        coords_1B[0] - coords_0B[0],
        coords_1B[1] - coords_0B[1],
    )

    two_unit_chain = RAMM_Chain.generate(
        n_units=2,
        start_position=coords_0B,
        offsets=lower_offset,
        node_diameter=chain.node_diameter,
        strut_width=chain.strut_width,
    )

    if direction == "right":

        result = (
            find_right_limiting_configuration_two_unit(
                two_unit_chain,
                contact_offset=(
                    node_strut_contact_offset
                ),
                verbose=False,
            )
        )

    elif direction == "left":

        result = (
            find_left_limiting_configuration_two_unit(
                two_unit_chain,
                contact_offset=(
                    node_strut_contact_offset
                ),
                verbose=False,
            )
        )

    else:
        raise ValueError(
            "direction must be 'left' or 'right'."
        )

    if not result["success"]:
        raise InvalidRAMMGeometryError(
            f"Could not find the Unit "
            f"{unit_0_index}-{unit_1_index} "
            "limiting configuration."
        )

    # ---------------------------------------------------------
    # 4. Use either:
    #
    #     - the full 2-unit limiting angle, or
    #     - the angle requested by the outer stepper.
    # ---------------------------------------------------------

    max_theta_deg = (
        result["theta_deg"]
    )

    if theta_deg_override is None:

        theta_deg = (
            max_theta_deg
        )

    else:

        theta_deg = float(
            theta_deg_override
        )

    # For this first implementation, retain the z shift obtained
    # from the corresponding 2-unit limiting solve.
    z_shift = (
        result["z_shift"]
    )

    if verbose:
        print(
            "\nTwo-unit limiting solution:"
        )

        print(
            f"  max theta = "
            f"{max_theta_deg:.6f}°"
        )

        if theta_deg_override is not None:
            print(
                f"  trial theta = "
                f"{theta_deg:.6f}°"
            )

        print(
            f"  z shift = "
            f"{z_shift:.6f} mm"
        )

    # ---------------------------------------------------------
    # 5. Apply the vertical shift from the 2-unit solution
    # ---------------------------------------------------------

    chain.translate(
        unit_index=unit_1_index,
        dy=0.0,
        dz=z_shift,
        verbose=verbose,
    )

    motions.append({
        "type": "translate",
        "unit_index": unit_1_index,
        "dy": 0.0,
        "dz": z_shift,
    })

    # ---------------------------------------------------------
    # 6. Apply the trial rotation WITH cascade
    #
    # Unit 1 rotates.
    # Unit 2 moves in response to the upper rail constraint.
    # ---------------------------------------------------------

    pivot = (
        chain.units[
            unit_1_index
        ].bottom_node
    )

    cascade_result = (
        propagate_free_unit_rotation(
            chain=chain,
            unit_index=unit_1_index,
            pivot=pivot,
            degrees=theta_deg,
            constraint_validator=lambda chain: (
                configuration_is_valid(
                    chain,
                    contact_tolerance=(
                        contact_tolerance
                    ),
                    verbose=False,
                )
            ),
            verbose=False,
        )
    )

    motions.append({
        "type": "propagate_free_unit_rotation",
        "unit_index": unit_1_index,
        "degrees": (
            cascade_result[
                "actual_degrees"
            ]
        ),
    })

    if verbose:
        print(
            "\nApplied trial two-unit solution to "
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

    # ---------------------------------------------------------
    # 7. Attempt to move Unit 2 downward toward the desired
    #    skip-level segment-segment contact.
    #
    # The updated finder will stop early if either relevant
    # node-segment contact becomes limiting first.
    # ---------------------------------------------------------

    if direction != "right":
        raise NotImplementedError(
            "Angle stepping is currently implemented "
            "for right rotation only."
        )

    segment_contact_result = (
        find_right_segment_segment_contact(
            chain=chain,
            lower_unit_index=(
                unit_0_index
            ),
            contact_tolerance=(
                contact_tolerance
            ),
            verbose=verbose,
        )
    )

    motions.append({
        "type": "translate_from",
        "start_unit_index": (
            unit_2_index
        ),
        "dy": 0.0,
        "dz": (
            segment_contact_result[
                "dz"
            ]
        ),
    })

    # ---------------------------------------------------------
    # 8. Did this particular angle actually allow the desired
    #    skip-level segment-segment contact?
    #
    # IMPORTANT:
    # Do not use segment_contact_result["success"] here.
    #
    # That is also True when the translation successfully found
    # an earlier physical constraint.
    # ---------------------------------------------------------

    segment_contact_reached = (
        segment_contact_result[
            "segment_contact_reached"
        ]
    )

    # ---------------------------------------------------------
    # 9. Return this trial
    # ---------------------------------------------------------

    return {
        "success": (
            segment_contact_reached
        ),
        "direction": direction,
        "start_unit_index": (
            start_unit_index
        ),
        "trial_theta_deg": (
            theta_deg
        ),
        "max_theta_deg": (
            max_theta_deg
        ),
        "z_shift": (
            z_shift
        ),
        "motions": motions,
        "two_unit_solution": (
            result
        ),
        "cascade_result": (
            cascade_result
        ),
        "segment_contact_result": (
            segment_contact_result
        ),
    }

def find_limiting_configuration_three_unit(
    chain,
    start_unit_index=0,
    direction="right",
    angle_step_deg=0.25,
    min_angle_deg=0.0,
    contact_tolerance=1e-6,
    verbose=True,
):
    """
    Find a 3-unit limiting configuration by progressively reducing
    the Unit-1 rotation from the corresponding 2-unit maximum.

    Workflow
    --------
    1. Determine the full 2-unit limiting rotation.
    2. Try that angle on the 3-unit chain.
    3. Attempt to move Unit 2 downward toward the desired
       skip-level segment-segment contact.
    4. If an intervening node-segment contact limits that motion
       first, restore the original 3-unit configuration.
    5. Reduce the Unit-1 rotation by angle_step_deg.
    6. Repeat until the desired skip-level segment-segment contact
       becomes reachable.
    7. Leave the chain at the successful configuration.

    Notes
    -----
    Every trial begins from exactly the same starting coordinates.

    For right rotation, the 2-unit limiting angle is negative.
    The search therefore reduces its magnitude toward zero.
    """

    direction = (
        direction.lower()
    )

    if direction != "right":
        raise NotImplementedError(
            "Three-unit angle stepping is currently "
            "implemented for right rotation only."
        )

    if angle_step_deg <= 0.0:
        raise ValueError(
            "angle_step_deg must be positive."
        )

    if min_angle_deg < 0.0:
        raise ValueError(
            "min_angle_deg must be nonnegative."
        )

    # ---------------------------------------------------------
    # 1. Save the untouched 3-unit starting configuration
    #
    # EVERY trial will return here first.
    # ---------------------------------------------------------

    starting_coordinates = (
        chain._save_coordinates()
    )

    # ---------------------------------------------------------
    # 2. First determine the normal maximum 2-unit angle.
    #
    # We can use the internal trial routine once, but we do not
    # want to keep the geometry it produces yet.
    # ---------------------------------------------------------

    chain._restore_coordinates(
        starting_coordinates
    )

    initial_trial = (
        _try_limiting_configuration_three_unit(
            chain=chain,
            start_unit_index=(
                start_unit_index
            ),
            direction=direction,
            theta_deg_override=None,
            contact_tolerance=(
                contact_tolerance
            ),
            verbose=False,
        )
    )

    max_theta_deg = float(
        initial_trial[
            "max_theta_deg"
        ]
    )

    # Restore because the initial trial moved the actual chain.
    chain._restore_coordinates(
        starting_coordinates
    )

    # Right rotation currently gives a negative angle.
    angle_sign = (
        -1.0
        if max_theta_deg < 0.0
        else 1.0
    )

    max_angle_magnitude = abs(
        max_theta_deg
    )

    # ---------------------------------------------------------
    # 3. Search from maximum rotation toward zero
    # ---------------------------------------------------------

    trial_index = 0
    trial_angle_magnitude = (
        max_angle_magnitude
    )

    best_result = None
    best_clearance = math.inf
    best_coordinates = None

    while (
        trial_angle_magnitude
        >= min_angle_deg
        - 1e-12
    ):

        trial_theta_deg = (
            angle_sign
            * trial_angle_magnitude
        )

        # Every trial starts from exactly the same chain state.
        chain._restore_coordinates(
            starting_coordinates
        )

        if verbose:
            print(
                "\n"
                "========================================"
            )

            print(
                f"Three-unit angle trial "
                f"{trial_index}"
            )

            print(
                f"  trial theta: "
                f"{trial_theta_deg:.6f}°"
            )

            print(
                "========================================"
            )

        trial_result = (
            _try_limiting_configuration_three_unit(
                chain=chain,
                start_unit_index=(
                    start_unit_index
                ),
                direction=direction,
                theta_deg_override=(
                    trial_theta_deg
                ),
                contact_tolerance=(
                    contact_tolerance
                ),
                verbose=verbose,
            )
        )

        segment_result = (
            trial_result[
                "segment_contact_result"
            ]
        )

        segment_clearance = abs(
            segment_result[
                "clearance"
            ]
        )

        # -----------------------------------------------------
        # Keep track of the closest attempt in case no exact
        # solution is found within the stepped range.
        # -----------------------------------------------------

        if (
            segment_clearance
            < best_clearance
        ):

            best_clearance = (
                segment_clearance
            )

            best_result = (
                trial_result
            )

            best_coordinates = (
                chain._save_coordinates()
            )

        # -----------------------------------------------------
        # SUCCESS:
        # desired skip-level segment contact was actually reached
        # before an intervening contact blocked the translation.
        # -----------------------------------------------------

        if (
            segment_result[
                "segment_contact_reached"
            ]
        ):

            if verbose:
                print(
                    "\nFound three-unit limiting "
                    "configuration."
                )

                print(
                    f"  successful theta: "
                    f"{trial_theta_deg:.6f}°"
                )

                print(
                    f"  segment clearance: "
                    f"{segment_result['clearance']:.9f} mm"
                )

            return {
                "success": True,
                "direction": direction,
                "start_unit_index": (
                    start_unit_index
                ),
                "max_theta_deg": (
                    max_theta_deg
                ),
                "successful_theta_deg": (
                    trial_theta_deg
                ),
                "angle_step_deg": (
                    angle_step_deg
                ),
                "n_trials": (
                    trial_index + 1
                ),
                "trial_result": (
                    trial_result
                ),
                "two_unit_solution": (
                    trial_result[
                        "two_unit_solution"
                    ]
                ),
                "cascade_result": (
                    trial_result[
                        "cascade_result"
                    ]
                ),
                "segment_contact_result": (
                    segment_result
                ),
                "motions": (
                    trial_result[
                        "motions"
                    ]
                ),
            }

        # -----------------------------------------------------
        # This angle was blocked before segment contact.
        #
        # Report WHY, then try a slightly smaller rotation.
        # -----------------------------------------------------

        if verbose:
            print(
                "\nDesired skip-level contact "
                "was not reachable at this angle."
            )

            if (
                segment_result[
                    "limiting_contacts"
                ]
            ):
                print(
                    "  limiting contact(s): "
                    + ", ".join(
                        segment_result[
                            "limiting_contacts"
                        ]
                    )
                )

            print(
                f"  remaining segment clearance: "
                f"{segment_result['clearance']:.6f} mm"
            )

            print(
                f"  trying "
                f"{angle_step_deg:.6f}° "
                "less rotation..."
            )

        trial_index += 1

        trial_angle_magnitude -= (
            angle_step_deg
        )

    # ---------------------------------------------------------
    # 4. No stepped angle reached the desired contact
    #
    # Leave the chain at the closest configuration found rather
    # than at an arbitrary final failed trial.
    # ---------------------------------------------------------

    if best_coordinates is not None:

        chain._restore_coordinates(
            best_coordinates
        )

    else:

        chain._restore_coordinates(
            starting_coordinates
        )

    if verbose:
        print(
            "\nCould not reach the desired "
            "skip-level segment-segment contact "
            "within the tested angle range."
        )

        if best_result is not None:
            print(
                f"Closest tested angle: "
                f"{best_result['trial_theta_deg']:.6f}°"
            )

            print(
                f"Closest segment clearance: "
                f"{best_clearance:.9f} mm"
            )

    return {
        "success": False,
        "direction": direction,
        "start_unit_index": (
            start_unit_index
        ),
        "max_theta_deg": (
            max_theta_deg
        ),
        "successful_theta_deg": None,
        "angle_step_deg": (
            angle_step_deg
        ),
        "n_trials": (
            trial_index
        ),
        "best_result": (
            best_result
        ),
        "best_clearance": (
            best_clearance
            if best_result is not None
            else None
        ),
    }

def find_limiting_configuration_three_unit_old(
    chain,
    start_unit_index=0,
    direction="right",
    contact_tolerance=1e-6,
    verbose=True
):
    """
    Construct the first-stage candidate jamming configuration
    for a 3-unit subset of a chain.

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

    # Record the sequence of motions actually applied to the chain.
    motions = []

    # Indices of the three-unit subset.
    unit_0_index = start_unit_index
    unit_1_index = start_unit_index + 1
    unit_2_index = start_unit_index + 2

    if unit_2_index >= len(chain.units):
        raise ValueError(
            "The requested three-unit subset extends "
            "beyond the end of the chain."
        )

    # ---------------------------------------------------------
    # 1. Determine the actual Unit 0 -> Unit 1 offset
    # ---------------------------------------------------------

    coords_0B = chain.units[unit_0_index].bottom_node.coordinates
    coords_1B = chain.units[unit_1_index].bottom_node.coordinates
    coords_2B = chain.units[unit_2_index].bottom_node.coordinates

    if verbose:
        print("Initial bottom node coordinates for each unit:\n"
            f"unit_{unit_0_index}: {coords_0B}\n"
            f"unit_{unit_1_index}: {coords_1B}\n"
            f"unit_{unit_2_index}: {coords_2B}\n"
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
        result = find_right_limiting_configuration_two_unit(
            two_unit_chain,
            contact_offset=node_strut_contact_offset,
            verbose=False
        )

    elif direction == "left":
        result = find_left_limiting_configuration_two_unit(
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
            f"Could not find the Unit "
            f"{unit_0_index}-{unit_1_index} "
            "limiting configuration."
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
        unit_index=unit_1_index,
        dy=0.0,
        dz=z_shift,
        verbose=verbose
    )

    motions.append({
        "type": "translate",
        "unit_index": unit_1_index,
        "dy": 0.0,
        "dz": z_shift,
    })

    pivot = chain.units[unit_1_index].bottom_node

    cascade_result = propagate_free_unit_rotation(
        chain=chain,
        unit_index=unit_1_index,
        pivot=pivot,
        degrees=theta_deg,
        constraint_validator=lambda chain: configuration_is_valid(
            chain,
            contact_tolerance=contact_tolerance,
            verbose=False,
        ),
        verbose=False
    )

    motions.append({
        "type": "propagate_free_unit_rotation",
        "unit_index": unit_1_index,
        "degrees": cascade_result["actual_degrees"],
    })

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
            lower_unit_index=unit_0_index,
            verbose=verbose
        )
    )

    motions.append({
        "type": "translate_from",
        "start_unit_index": unit_2_index,
        "dy": 0.0,
        "dz": segment_contact_result["dz"],
    })

    # ---------------------------------------------------------
    # 8. Return the solved information
    # ---------------------------------------------------------

    return {
        "success": True, # TODO: placeholder for now; should be updating this
        "direction": direction,
        "start_unit_index": start_unit_index,
        "motions": motions,
        "two_unit_solution": result,
        "cascade_result": cascade_result,
        "segment_contact_result": segment_contact_result,
    }

"""
def find_limiting_configuration_four_unit(
    chain,
    start_index=0,
    direction="right",
    contact_tolerance=1e-6,
    verbose=True,
):

    if direction != "right":
        raise NotImplementedError(
            "Only right rotation is currently implemented."
        )

    three_unit_result = find_limiting_configuration_three_unit(
        chain=chain,
        start_unit_index=start_index,
        direction="right",
        contact_tolerance=contact_tolerance,
    )

    if verbose:
        print(
            "\nValid before shared-rail correction:",
            configuration_is_valid(
                chain,
                contact_tolerance=contact_tolerance,
            )
        )

    correction = enforce_shared_rail_node_spacing(
        chain,
        railed_unit_index=start_index+2,
    )
    # The "first unit" is unit 0; Python is zero indexed
    subset_unit_1_idx = start_index + 1
    subset_unit_2_idx = start_index + 2
    subset_unit_3_idx = start_index + 3

    if verbose:
        print(
            "\nShared-rail correction:",
            correction
        )
        # unit 1 top node coordinates
        print(
            f"\n{subset_unit_1_idx}T:",
            chain.units[subset_unit_1_idx].top_node.coordinates
        )
        # unit 3 bottom node coordinates
        print(
            f"{subset_unit_3_idx}B:",
            chain.units[subset_unit_3_idx].bottom_node.coordinates
        )

        print(
            "\nValid after shared-rail correction:",
            configuration_is_valid(
                chain,
                contact_tolerance=contact_tolerance,
            )
        )

    last_two_unit_subset_result = find_right_limiting_configuration_two_unit(
        chain=chain,
        lower_unit_index=subset_unit_2_idx,
        moving_unit_index=subset_unit_3_idx,
        contact_offset=chain.node_strut_contact_offset,
        contact_tolerance=contact_tolerance,
        verbose=True,
    )

    if verbose:
        print(
            "\nUnit 2-3 limiting solution:"
            f"\n  theta = {last_two_unit_subset_result['theta_deg']:.6f}°"
            f"\n  z shift = {last_two_unit_subset_result['z_shift']:.6f} mm"
        )

        print(
            "\nValid after Unit 3 limiting rotation:",
            configuration_is_valid(
                chain,
                contact_tolerance=contact_tolerance,
            )
        )

    return {
        "success": True, # TODO: placeholder for now; should be updating this
        "direction": direction,
        "three_unit_solution": three_unit_result,
    }
"""

def find_limiting_configuration_four_unit(
    chain,
    start_unit_index=0,
    direction="right",
    contact_tolerance=1e-6,
    verbose=True,
):
    """
    Build a four-unit rotational motion-limiting candidate
    from the corresponding three-unit candidate.
    """

    direction = direction.lower()

    # Indices of the four-unit subset.
    subset_unit_0_idx = start_unit_index
    subset_unit_1_idx = start_unit_index + 1
    subset_unit_2_idx = start_unit_index + 2
    subset_unit_3_idx = start_unit_index + 3

    if subset_unit_3_idx >= len(chain.units):
        raise ValueError(
            "The requested four-unit subset extends "
            "beyond the end of the chain."
        )

    if direction != "right":
        raise NotImplementedError(
            "Only right rotation is currently implemented."
        )

    three_unit_result = find_limiting_configuration_three_unit(
        chain=chain,
        start_unit_index=start_unit_index,
        direction=direction,
        contact_tolerance=contact_tolerance,
        verbose=verbose,
    )

    if verbose:
        print(
            "\nValid before shared-rail correction:",
            configuration_is_valid(
                chain,
                contact_tolerance=contact_tolerance,
            )
        )

    correction = enforce_shared_rail_node_spacing(
        chain,
        railed_unit_index=subset_unit_2_idx,
    )

    if verbose:
        print(
            "\nShared-rail correction:",
            correction
        )

        # unit 1 top node coordinates
        print(
            f"\n{subset_unit_1_idx}T:",
            chain.units[
                subset_unit_1_idx
            ].top_node.coordinates
        )

        # unit 3 bottom node coordinates
        print(
            f"{subset_unit_3_idx}B:",
            chain.units[
                subset_unit_3_idx
            ].bottom_node.coordinates
        )

        print(
            "\nValid after shared-rail correction:",
            configuration_is_valid(
                chain,
                contact_tolerance=contact_tolerance,
            )
        )

    last_two_unit_subset_result = (
        find_right_limiting_configuration_two_unit(
            chain=chain,
            lower_unit_index=subset_unit_2_idx,
            moving_unit_index=subset_unit_3_idx,
            contact_offset=chain.node_strut_contact_offset,
            contact_tolerance=contact_tolerance,
            verbose=verbose,
        )
    )

    if verbose:
        print(
            f"\nUnit {subset_unit_2_idx}-"
            f"{subset_unit_3_idx} limiting solution:"
            f"\n  theta = "
            f"{last_two_unit_subset_result['theta_deg']:.6f}°"
            f"\n  z shift = "
            f"{last_two_unit_subset_result['z_shift']:.6f} mm"
        )

        print(
            f"\nValid after Unit "
            f"{subset_unit_3_idx} limiting rotation:",
            configuration_is_valid(
                chain,
                contact_tolerance=contact_tolerance,
            )
        )

    return {
        "success": True, # TODO: placeholder for now; should be updating this
        "direction": direction,
        "start_unit_index": start_unit_index,
        "three_unit_solution": three_unit_result,
        "last_two_unit_subset_solution": (
            last_two_unit_subset_result
        ),
        "shared_rail_correction": correction,
    }