"""
Workspace functions used to examine RAMMs.
"""

# ============================================================================
# Imports
# ============================================================================
import numpy as np
import sympy as sp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from ramms.plotting import plot_geometry
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

#TODO: temp for testing; remove later# visual aids to mimic physical geometry
NODE_DIAMETER_PLOT = 1200
SEG_LINE_WIDTH = 30
RAIL_VISUAL_OFFSET = 1.2
RAIL_VISUAL_SHORTTEN = 2

# graph dimentions to match physical geometry
TWO_UNIT_YLIM = (-5, 30)
THREE_UNIT_YLIM = (-5, 50)
FOUR_UNIT_YLIM = (-5, 65)
FIVE_UNIT_YLIM = (-5, 80)
SIX_UNIT_YLIM = (-5, 95)

XLIM = (-15, 15)
XLIM_WIDE = (-25, 25)

TWO_UNIT_ROTATED_RIGHT_XLIM=(-15, 25)
TWO_UNIT_ROTATED_LEFT_XLIM = (-25, 15)
THREE_UNIT_ROTATED_RIGHT_XLIM=(-15, 25)
THREE_UNIT_ROTATED_LEFT_XLIM = (-25, 15)
FIVE_UNIT_ROTATED_RIGHT_XLIM=(-15, 35) # TODO: will need to reverse this for left rotation
SIX_UNIT_ROTATED_RIGHT_XLIM=(-15, 45)

ROTATED_THREE_UNIT_YLIM=(-5, 40)
ROTATED_FOUR_UNIT_YLIM=(-5, 45)
ROTATED_FIVE_UNIT_YLIM=(-5, 55)
ROTATED_SIX_UNIT_YLIM=(-5, 65)

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
            result["left_clearance"] >= -contact_tolerance
            and result["bottom_clearance"] >= -contact_tolerance
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
def find_right_limiting_configuration_two_unit_old(
    chain,
    lower_unit_index=0,
    moving_unit_index=1,
    theta_guess=-38.6,
    z_guess=-1.8,
    theta_bounds=(-60.0, 0.0),
    z_bounds=(-8.0, 5.0),
    default_offset=10,
    contact_offset=0.0,  # node radius
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
        raise ValueError(
            "contact_offset must be nonnegative."
        )

    if not 0 <= lower_unit_index < len(chain.units):
        raise IndexError(
            f"Invalid lower unit index: "
            f"{lower_unit_index}"
        )

    if not 0 <= moving_unit_index < len(chain.units):
        raise IndexError(
            f"Invalid moving unit index: "
            f"{moving_unit_index}"
        )

    # ---------------------------------------------------------
    # Save configuration at beginning of solve.
    #
    # Every trial is evaluated from this state.
    # ---------------------------------------------------------

    starting_coordinates = (
        chain._save_coordinates()
    )

    # ---------------------------------------------------------
    # Target geometry descriptors
    # ---------------------------------------------------------

    node_R_descriptor = (
        f"{lower_unit_index}R"
    )

    node_T_descriptor = (
        f"{lower_unit_index}T"
    )

    segment_RB_descriptor = (
        f"{moving_unit_index}R"
        f"{moving_unit_index}B"
    )

    segment_BL_descriptor = (
        f"{moving_unit_index}B"
        f"{moving_unit_index}L"
    )

    # ---------------------------------------------------------
    # Determine which side of each segment is valid in the
    # starting nonpenetrating configuration.
    #
    # IMPORTANT:
    # Use the actual requested unit indices rather than
    # hard-coded Unit 0 / Unit 1 descriptors.
    # ---------------------------------------------------------

    right_side = get_reference_side(
        chain=chain,
        node_descriptor=node_R_descriptor,
        segment_descriptor=segment_RB_descriptor,
        orientation="counterclockwise",
    )

    top_side = get_reference_side(
        chain=chain,
        node_descriptor=node_T_descriptor,
        segment_descriptor=segment_BL_descriptor,
        orientation="clockwise",
    )

    # ---------------------------------------------------------
    # Evaluate the two target gaps for a candidate theta/z pair
    # ---------------------------------------------------------

    def evaluate_target_gaps(
        theta_deg,
        z_shift
    ):

        set_adjacent_unit_configuration(
            chain=chain,
            moving_unit_index=moving_unit_index,
            starting_coordinates=starting_coordinates,
            theta_deg=theta_deg,
            z_shift=z_shift,
        )

        segment_RB = (
            chain.get_segment_by_descriptor(
                segment_RB_descriptor
            )
        )

        segment_BL = (
            chain.get_segment_by_descriptor(
                segment_BL_descriptor
            )
        )

        node_R = (
            chain.get_node_by_descriptor(
                node_R_descriptor
            )
        )

        node_T = (
            chain.get_node_by_descriptor(
                node_T_descriptor
            )
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

    # ---------------------------------------------------------
    # Residual vector
    #
    # We want both physical clearances to become zero.
    # ---------------------------------------------------------

    def residuals(x):

        theta_deg, z_shift = x

        (
            signed_gap_right,
            signed_gap_top
        ) = evaluate_target_gaps(
            theta_deg,
            z_shift
        )

        # Reorient each signed gap so it is positive on the
        # valid side of the segment.
        separation_right = (
            right_side
            * signed_gap_right
        )

        separation_top = (
            top_side
            * signed_gap_top
        )

        # Physical contact occurs when:
        #
        # oriented centerline separation
        #     =
        # contact offset
        clearance_right = (
            separation_right
            - contact_offset
        )

        clearance_top = (
            separation_top
            - contact_offset
        )

        return np.array([
            clearance_right,
            clearance_top
        ])

    # ---------------------------------------------------------
    # Solve for theta and vertical shift
    # ---------------------------------------------------------

    solution = least_squares(
        residuals,
        x0=np.array(
            [
                theta_guess,
                z_guess
            ],
            dtype=float
        ),
        bounds=(
            [
                theta_bounds[0],
                z_bounds[0]
            ],
            [
                theta_bounds[1],
                z_bounds[1]
            ]
        ),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12
    )

    theta_deg, z_shift = (
        solution.x
    )

    # ---------------------------------------------------------
    # Leave the chain in the solved configuration
    # ---------------------------------------------------------

    (
        signed_gap_right,
        signed_gap_top
    ) = evaluate_target_gaps(
        theta_deg,
        z_shift
    )

    separation_right = (
        right_side
        * signed_gap_right
    )

    separation_top = (
        top_side
        * signed_gap_top
    )

    clearance_right = (
        separation_right
        - contact_offset
    )

    clearance_top = (
        separation_top
        - contact_offset
    )

    residual_norm = np.linalg.norm([
        clearance_right,
        clearance_top
    ])

    # ---------------------------------------------------------
    # Validate target double-contact configuration
    # ---------------------------------------------------------

    valid = (
        solution.success
        and abs(
            clearance_right
        ) <= contact_tolerance
        and abs(
            clearance_top
        ) <= contact_tolerance
    )

    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------

    if verbose:

        print(
            f"Solver success: "
            f"{solution.success}"
        )

        print(
            f"Physically valid: "
            f"{valid}"
        )

        print(
            f"θ: "
            f"{theta_deg:.6f}°"
        )

        print(
            f"z shift: "
            f"{z_shift:.6f} mm"
        )

        print(
            f"Contact offset: "
            f"{contact_offset:.6f} mm"
        )

        print(
            f"Reference side "
            f"{segment_RB_descriptor}–"
            f"{node_R_descriptor}: "
            f"{right_side:+.0f}"
        )

        print(
            f"Reference side "
            f"{segment_BL_descriptor}–"
            f"{node_T_descriptor}: "
            f"{top_side:+.0f}"
        )

        print(
            f"Signed gap "
            f"{segment_RB_descriptor}–"
            f"{node_R_descriptor}: "
            f"{signed_gap_right:.9f} mm"
        )

        print(
            f"Oriented separation "
            f"{segment_RB_descriptor}–"
            f"{node_R_descriptor}: "
            f"{separation_right:.9f} mm"
        )

        print(
            f"Clearance "
            f"{segment_RB_descriptor}–"
            f"{node_R_descriptor}: "
            f"{clearance_right:.9f} mm"
        )

        print(
            f"Signed gap "
            f"{segment_BL_descriptor}–"
            f"{node_T_descriptor}: "
            f"{signed_gap_top:.9f} mm"
        )

        print(
            f"Oriented separation "
            f"{segment_BL_descriptor}–"
            f"{node_T_descriptor}: "
            f"{separation_top:.9f} mm"
        )

        print(
            f"Clearance "
            f"{segment_BL_descriptor}–"
            f"{node_T_descriptor}: "
            f"{clearance_top:.9f} mm"
        )

        print(
            f"Residual norm: "
            f"{residual_norm:.3e}"
        )

    # ---------------------------------------------------------
    # Return
    # ---------------------------------------------------------

    return {
        "success": valid,
        "solver_success": (
            solution.success
        ),
        "lower_unit_index": (
            lower_unit_index
        ),
        "moving_unit_index": (
            moving_unit_index
        ),
        "theta_deg": (
            theta_deg
        ),
        "z_shift": (
            z_shift
        ),
        "contact_offset": (
            contact_offset
        ),
        "right_side": (
            right_side
        ),
        "top_side": (
            top_side
        ),
        "signed_gap_right": (
            signed_gap_right
        ),
        "signed_gap_top": (
            signed_gap_top
        ),
        "separation_right": (
            separation_right
        ),
        "separation_top": (
            separation_top
        ),
        "clearance_right": (
            clearance_right
        ),
        "clearance_top": (
            clearance_top
        ),
        "residual_norm": (
            residual_norm
        ),
        "solution": (
            solution
        )
    }


def resolve_shared_rail_crossing_old(
    chain,
    railed_unit_index,
    moving_unit_index,
    angle_step_deg=0.1,
    contact_tolerance=1e-6,
    max_steps=500,
    verbose=True,
):
    """
    Back the upper FREE unit out of an invalid shared-rail
    configuration by reducing its right rotation.

    Example for Units 2-3:

        shared rail:
            1T and 3B inside Unit 2

        required:
            3B must remain above 1T

        additional requirement:
            3B3L must remain on the valid side of 2T.
    """

    lower_free_index = (
        railed_unit_index - 1
    )

    # Save the state produced by the 2-unit solve.
    starting_coordinates = (
        chain._save_coordinates()
    )

    lower_shared_node = (
        chain.units[
            lower_free_index
        ].top_node
    )

    moving_bottom_node = (
        chain.units[
            moving_unit_index
        ].bottom_node
    )

    moving_BL_segment = (
        chain.get_segment_by_descriptor(
            f"{moving_unit_index}B"
            f"{moving_unit_index}L"
        )
    )

    railed_top_node = (
        chain.units[
            railed_unit_index
        ].top_node
    )

    # Determine the valid side of 2T relative to 3B3L
    # from the configuration we are starting from.
    valid_side = get_reference_side(
        chain=chain,
        node_descriptor=(
            f"{railed_unit_index}T"
        ),
        segment_descriptor=(
            f"{moving_unit_index}B"
            f"{moving_unit_index}L"
        ),
        orientation="clockwise",
    )

    for step in range(max_steps):

        # -----------------------------------------------------
        # 1. Check shared-rail ordering
        # -----------------------------------------------------

        railed_unit = chain.units[
            railed_unit_index
        ]

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

        rail_direction = (
            rail_vector / rail_length
        )

        lower_free_index = (
            railed_unit_index - 1
        )

        lower_node = np.asarray(
            chain.units[
                lower_free_index
            ].top_node.coordinates,
            dtype=float,
        )

        upper_node = np.asarray(
            chain.units[
                moving_unit_index
            ].bottom_node.coordinates,
            dtype=float,
        )

        current_spacing = np.dot(
            upper_node - lower_node,
            rail_direction,
        )

        shared_clearance = (
            current_spacing
            - chain.node_diameter
        )

        # -----------------------------------------------------
        # 2. Check 3B3L / 2T side
        # -----------------------------------------------------

        gap = get_node_segment_gap(
            railed_top_node,
            moving_BL_segment,
            "clockwise",
        )

        separation = (
            valid_side
            * gap.result.length_mm
        )

        contact_clearance = (
            separation
            - chain.node_strut_contact_offset
        )

        # -----------------------------------------------------
        # Done when both conditions are physically valid.
        # -----------------------------------------------------

        if (
            shared_clearance
            >= -contact_tolerance
            and contact_clearance
            >= -contact_tolerance
        ):

            if verbose:
                print(
                    "\nShared-rail crossing resolved:"
                    f"\n  steps: {step}"
                    f"\n  shared clearance: "
                    f"{shared_clearance:.6f} mm"
                    f"\n  3B3L-2T clearance: "
                    f"{contact_clearance:.6f} mm"
                )

            return {
                "success": True,
                "steps": step,
                "shared_clearance": (
                    shared_clearance
                ),
                "contact_clearance": (
                    contact_clearance
                ),
            }

        # -----------------------------------------------------
        # Reduce the right rotation.
        #
        # Right rotation is negative, so rotating LEFT means
        # applying a positive incremental angle.
        # -----------------------------------------------------

        pivot = (
            chain.units[
                moving_unit_index
            ].bottom_node
        )

        chain.units[
            moving_unit_index
        ].rotate(
            pivot=pivot,
            degrees=angle_step_deg,
        )

        # -----------------------------------------------------
        # Restore shared-rail spacing by moving the upper unit
        # along the rail.
        #
        # This is now done AFTER reducing rotation, rather than
        # using translation alone to fix an impossible state.
        # -----------------------------------------------------

        enforce_shared_rail_node_spacing(
            chain,
            railed_unit_index=(
                railed_unit_index
            ),
            tolerance=(
                contact_tolerance
            ),
        )

    chain._restore_coordinates(
        starting_coordinates
    )

    raise InvalidRAMMGeometryError(
        "Could not resolve shared-rail crossing "
        "within the allowed rotation range."
    )


def resolve_shared_rail_crossing(
    chain,
    railed_unit_index,
    moving_unit_index,
    angle_step_deg=0.1,
    contact_tolerance=1e-6,
    max_steps=500,
    verbose=True,
):
    """
    Back the upper FREE unit out of an invalid shared-rail
    configuration by reducing its right rotation.

    Example for Units 2-3:

        shared rail:
            1T and 3B inside Unit 2

        required:
            3B must remain above 1T

        additional requirement:
            3B3L must remain on the valid side of 2T.

    Any rotational or translational correction applied to the
    moving unit is propagated upward through the rest of the chain.
    """

    lower_free_index = (
        railed_unit_index - 1
    )

    # Save the state produced by the 2-unit solve.
    starting_coordinates = (
        chain._save_coordinates()
    )

    lower_shared_node = (
        chain.units[
            lower_free_index
        ].top_node
    )

    moving_bottom_node = (
        chain.units[
            moving_unit_index
        ].bottom_node
    )

    moving_BL_segment = (
        chain.get_segment_by_descriptor(
            f"{moving_unit_index}B"
            f"{moving_unit_index}L"
        )
    )

    railed_top_node = (
        chain.units[
            railed_unit_index
        ].top_node
    )

    # Determine the valid side of the railed top node
    # relative to the moving B-L segment.
    valid_side = get_reference_side(
        chain=chain,
        node_descriptor=(
            f"{railed_unit_index}T"
        ),
        segment_descriptor=(
            f"{moving_unit_index}B"
            f"{moving_unit_index}L"
        ),
        orientation="clockwise",
    )

    for step in range(max_steps):

        # -----------------------------------------------------
        # 1. Check shared-rail ordering
        # -----------------------------------------------------

        railed_unit = chain.units[
            railed_unit_index
        ]

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

        lower_node = np.asarray(
            chain.units[
                lower_free_index
            ].top_node.coordinates,
            dtype=float,
        )

        upper_node = np.asarray(
            chain.units[
                moving_unit_index
            ].bottom_node.coordinates,
            dtype=float,
        )

        current_spacing = np.dot(
            upper_node - lower_node,
            rail_direction,
        )

        shared_clearance = (
            current_spacing
            - chain.node_diameter
        )

        # -----------------------------------------------------
        # 2. Check moving B-L segment / railed top-node side
        # -----------------------------------------------------

        # Refresh these objects from the current chain state,
        # since the moving unit changes every iteration.
        moving_BL_segment = (
            chain.get_segment_by_descriptor(
                f"{moving_unit_index}B"
                f"{moving_unit_index}L"
            )
        )

        railed_top_node = (
            chain.units[
                railed_unit_index
            ].top_node
        )

        gap = get_node_segment_gap(
            railed_top_node,
            moving_BL_segment,
            "clockwise",
        )

        separation = (
            valid_side
            * gap.result.length_mm
        )

        contact_clearance = (
            separation
            - chain.node_strut_contact_offset
        )

        # -----------------------------------------------------
        # 3. Done when both conditions are physically valid
        # -----------------------------------------------------

        if (
            shared_clearance
            >= -contact_tolerance
            and contact_clearance
            >= -contact_tolerance
        ):

            if verbose:
                print(
                    "\nShared-rail crossing resolved:"
                    f"\n  steps: {step}"
                    f"\n  shared clearance: "
                    f"{shared_clearance:.6f} mm"
                    f"\n  "
                    f"{moving_unit_index}B"
                    f"{moving_unit_index}L-"
                    f"{railed_unit_index}T clearance: "
                    f"{contact_clearance:.6f} mm"
                )

            return {
                "success": True,
                "steps": step,
                "shared_clearance": (
                    shared_clearance
                ),
                "contact_clearance": (
                    contact_clearance
                ),
            }

        # -----------------------------------------------------
        # 4. Reduce the right rotation
        #
        # Right rotation is negative, so rotating LEFT means
        # applying a positive incremental angle.
        #
        # IMPORTANT:
        # Use propagation here so every unit above the moving
        # FREE unit follows this correction.
        # -----------------------------------------------------

        pivot = (
            chain.units[
                moving_unit_index
            ].bottom_node
        )

        propagate_free_unit_rotation(
            chain=chain,
            unit_index=moving_unit_index,
            pivot=pivot,
            degrees=angle_step_deg,
            constraint_validator=lambda chain: True,
            verbose=False,
        )

        # -----------------------------------------------------
        # 5. Restore shared-rail spacing
        #
        # enforce_shared_rail_node_spacing() already propagates
        # its translation upward through the chain.
        # -----------------------------------------------------

        enforce_shared_rail_node_spacing(
            chain,
            railed_unit_index=(
                railed_unit_index
            ),
            tolerance=(
                contact_tolerance
            ),
        )

    # ---------------------------------------------------------
    # Could not resolve the crossing
    # ---------------------------------------------------------

    chain._restore_coordinates(
        starting_coordinates
    )

    raise InvalidRAMMGeometryError(
        "Could not resolve shared-rail crossing "
        "within the allowed rotation range."
    )


def find_right_limiting_configuration_two_unit(
    chain,
    lower_unit_index=0,
    moving_unit_index=1,
    theta_guess=-38.6,
    z_guess=-1.8,
    theta_bounds=(-60.0, 0.0),
    z_bounds=(-8.0, 5.0),
    contact_offset=0.0,
    contact_tolerance=1e-6,
    verbose=True
):
    """
    Find the maximum-right-rotation double-contact configuration.

    Target contacts:
        moving_unit R-B segment with lower_unit R
        moving_unit B-L segment with lower_unit T

    During the solve, the moving B-L segment is not allowed
    to cross through the lower unit's T node.

    For the Unit 2-3 solve, this means:

        3B3L may approach 2T,
        but it may not pass through 2T.

    Parameters
    ----------
    contact_offset : float, optional
        Required centerline distance at physical contact.

        For zero-thickness geometry:
            contact_offset = 0

        For a circular node and finite-width strut:
            contact_offset = node radius + strut half-width
    """

    if contact_offset < 0:
        raise ValueError(
            "contact_offset must be nonnegative."
        )

    if not 0 <= lower_unit_index < len(chain.units):
        raise IndexError(
            f"Invalid lower unit index: "
            f"{lower_unit_index}"
        )

    if not 0 <= moving_unit_index < len(chain.units):
        raise IndexError(
            f"Invalid moving unit index: "
            f"{moving_unit_index}"
        )

    # ---------------------------------------------------------
    # 1. Save starting configuration
    # ---------------------------------------------------------

    starting_coordinates = (
        chain._save_coordinates()
    )

    # ---------------------------------------------------------
    # 2. Target descriptors
    # ---------------------------------------------------------

    node_R_descriptor = (
        f"{lower_unit_index}R"
    )

    node_T_descriptor = (
        f"{lower_unit_index}T"
    )

    segment_RB_descriptor = (
        f"{moving_unit_index}R"
        f"{moving_unit_index}B"
    )

    segment_BL_descriptor = (
        f"{moving_unit_index}B"
        f"{moving_unit_index}L"
    )

    # ---------------------------------------------------------
    # 3. Determine valid sides in the starting configuration
    # ---------------------------------------------------------

    right_side = get_reference_side(
        chain=chain,
        node_descriptor=node_R_descriptor,
        segment_descriptor=segment_RB_descriptor,
        orientation="counterclockwise",
    )

    top_side = get_reference_side(
        chain=chain,
        node_descriptor=node_T_descriptor,
        segment_descriptor=segment_BL_descriptor,
        orientation="clockwise",
    )

    # ---------------------------------------------------------
    # 4. Apply a trial theta / z configuration and evaluate
    #    the two target gaps
    # ---------------------------------------------------------

    def evaluate_target_gaps(
        theta_deg,
        z_shift
    ):

        set_adjacent_unit_configuration(
            chain=chain,
            moving_unit_index=moving_unit_index,
            starting_coordinates=starting_coordinates,
            theta_deg=theta_deg,
            z_shift=z_shift,
        )

        segment_RB = (
            chain.get_segment_by_descriptor(
                segment_RB_descriptor
            )
        )

        segment_BL = (
            chain.get_segment_by_descriptor(
                segment_BL_descriptor
            )
        )

        node_R = (
            chain.get_node_by_descriptor(
                node_R_descriptor
            )
        )

        node_T = (
            chain.get_node_by_descriptor(
                node_T_descriptor
            )
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

    # ---------------------------------------------------------
    # 5. Convert signed gaps into physical clearances
    # ---------------------------------------------------------

    def evaluate_clearances(x):

        theta_deg, z_shift = x

        (
            signed_gap_right,
            signed_gap_top
        ) = evaluate_target_gaps(
            theta_deg,
            z_shift
        )

        separation_right = (
            right_side
            * signed_gap_right
        )

        separation_top = (
            top_side
            * signed_gap_top
        )

        clearance_right = (
            separation_right
            - contact_offset
        )

        clearance_top = (
            separation_top
            - contact_offset
        )

        return (
            clearance_right,
            clearance_top
        )

    # ---------------------------------------------------------
    # 6. Objective
    #
    # We still want BOTH target clearances to become zero.
    # ---------------------------------------------------------

    def objective(x):

        (
            clearance_right,
            clearance_top
        ) = evaluate_clearances(x)

        return float(
            clearance_right**2
            + clearance_top**2
        )

    # ---------------------------------------------------------
    # 7. Nonpenetration constraint
    #
    # Do not allow:
    #
    #     moving_unit B-L
    #
    # to pass through:
    #
    #     lower_unit T
    #
    # For Units 2-3:
    #
    #     3B3L cannot cross 2T.
    #
    # SLSQP inequality constraints must satisfy:
    #
    #     constraint(x) >= 0
    # ---------------------------------------------------------

    def top_contact_nonpenetration_constraint(x):

        _, clearance_top = (
            evaluate_clearances(x)
        )

        return clearance_top

    # ---------------------------------------------------------
    # 8. Solve
    # ---------------------------------------------------------

    solution = minimize(
        objective,
        x0=np.array(
            [
                theta_guess,
                z_guess
            ],
            dtype=float
        ),
        method="SLSQP",
        bounds=[
            theta_bounds,
            z_bounds,
        ],
        constraints=[
            {
                "type": "ineq",
                "fun": (
                    top_contact_nonpenetration_constraint
                ),
            }
        ],
        options={
            "ftol": 1e-12,
            "maxiter": 500,
            "disp": False,
        },
    )

    theta_deg, z_shift = (
        solution.x
    )

    # ---------------------------------------------------------
    # 9. Leave chain at the final solved configuration
    # ---------------------------------------------------------

    (
        signed_gap_right,
        signed_gap_top
    ) = evaluate_target_gaps(
        theta_deg,
        z_shift
    )

    separation_right = (
        right_side
        * signed_gap_right
    )

    separation_top = (
        top_side
        * signed_gap_top
    )

    clearance_right = (
        separation_right
        - contact_offset
    )

    clearance_top = (
        separation_top
        - contact_offset
    )

    residual_norm = np.linalg.norm([
        clearance_right,
        clearance_top
    ])

    # ---------------------------------------------------------
    # 10. Validate
    # ---------------------------------------------------------

    top_contact_nonpenetrating = (
        clearance_top
        >= -contact_tolerance
    )

    valid = (
        solution.success
        and top_contact_nonpenetrating
        and abs(
            clearance_right
        ) <= contact_tolerance
        and abs(
            clearance_top
        ) <= contact_tolerance
    )

    # ---------------------------------------------------------
    # 11. Print
    # ---------------------------------------------------------

    if verbose:

        print(
            f"Solver success: "
            f"{solution.success}"
        )

        print(
            f"Physically valid: "
            f"{valid}"
        )

        print(
            f"θ: "
            f"{theta_deg:.6f}°"
        )

        print(
            f"z shift: "
            f"{z_shift:.6f} mm"
        )

        print(
            f"Contact offset: "
            f"{contact_offset:.6f} mm"
        )

        print(
            f"Reference side "
            f"{segment_RB_descriptor}–"
            f"{node_R_descriptor}: "
            f"{right_side:+.0f}"
        )

        print(
            f"Reference side "
            f"{segment_BL_descriptor}–"
            f"{node_T_descriptor}: "
            f"{top_side:+.0f}"
        )

        print(
            f"Signed gap "
            f"{segment_RB_descriptor}–"
            f"{node_R_descriptor}: "
            f"{signed_gap_right:.9f} mm"
        )

        print(
            f"Oriented separation "
            f"{segment_RB_descriptor}–"
            f"{node_R_descriptor}: "
            f"{separation_right:.9f} mm"
        )

        print(
            f"Clearance "
            f"{segment_RB_descriptor}–"
            f"{node_R_descriptor}: "
            f"{clearance_right:.9f} mm"
        )

        print(
            f"Signed gap "
            f"{segment_BL_descriptor}–"
            f"{node_T_descriptor}: "
            f"{signed_gap_top:.9f} mm"
        )

        print(
            f"Oriented separation "
            f"{segment_BL_descriptor}–"
            f"{node_T_descriptor}: "
            f"{separation_top:.9f} mm"
        )

        print(
            f"Clearance "
            f"{segment_BL_descriptor}–"
            f"{node_T_descriptor}: "
            f"{clearance_top:.9f} mm"
        )

        print(
            f"{segment_BL_descriptor}–"
            f"{node_T_descriptor} "
            f"nonpenetrating: "
            f"{top_contact_nonpenetrating}"
        )

        print(
            f"Residual norm: "
            f"{residual_norm:.3e}"
        )

    # ---------------------------------------------------------
    # 12. Return
    # ---------------------------------------------------------

    return {
        "success": valid,
        "solver_success": (
            solution.success
        ),
        "lower_unit_index": (
            lower_unit_index
        ),
        "moving_unit_index": (
            moving_unit_index
        ),
        "theta_deg": (
            theta_deg
        ),
        "z_shift": (
            z_shift
        ),
        "contact_offset": (
            contact_offset
        ),
        "right_side": (
            right_side
        ),
        "top_side": (
            top_side
        ),
        "signed_gap_right": (
            signed_gap_right
        ),
        "signed_gap_top": (
            signed_gap_top
        ),
        "separation_right": (
            separation_right
        ),
        "separation_top": (
            separation_top
        ),
        "clearance_right": (
            clearance_right
        ),
        "clearance_top": (
            clearance_top
        ),
        "top_contact_nonpenetrating": (
            top_contact_nonpenetrating
        ),
        "residual_norm": (
            residual_norm
        ),
        "solution": (
            solution
        )
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
    contact_tolerance=1e-6, # not yet used in this solver
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
            "find_limiting_configuration_two_unit requires a 2-unit chain."
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


def find_right_lower_contact_shift(
    chain,
    lower_unit_index,
    dz_guess=-1.0,
    dz_bounds=(-20.0, 5.0),
    contact_tolerance=1e-6,
    verbose=True,
):
    """
    Translate the intermediate FREE unit and everything above it
    vertically toward contact between its R-B segment and the
    lower RAILED unit's R node.

    For a 3-unit subset starting at Unit 0:

        target contact:
            Unit 1 R-B segment with Unit 0 R node

    While moving downward, also monitor:

        Unit 1 L-T segment with Unit 0 T node

    If that second contact becomes limiting first, stop at the
    closest nonpenetrating configuration instead of allowing
    penetration.
    """

    intermediate_unit_index = (
        lower_unit_index + 1
    )

    if intermediate_unit_index >= len(chain.units):
        raise ValueError(
            "No intermediate unit exists above "
            f"Unit {lower_unit_index}."
        )

    contact_offset = (
        chain.node_strut_contact_offset
    )

    # ---------------------------------------------------------
    # 1. Save starting configuration
    #
    # Every trial translation starts from this exact state.
    # ---------------------------------------------------------

    starting_coordinates = (
        chain._save_coordinates()
    )

    # ---------------------------------------------------------
    # 2. Evaluate both relevant lower contacts at a trial dz
    # ---------------------------------------------------------

    def evaluate_contacts(dz):

        chain._restore_coordinates(
            starting_coordinates
        )

        # Move Unit 1 and everything above it together.
        _translate_units_from(
            chain,
            start_unit_index=intermediate_unit_index,
            dy=0.0,
            dz=dz,
        )

        # -----------------------------------------------------
        # Target contact:
        #
        #     Unit 1 R-B segment
        #         with
        #     Unit 0 R node
        # -----------------------------------------------------

        lower_right_node = (
            chain.get_node_by_descriptor(
                f"{lower_unit_index}R"
            )
        )

        intermediate_right_segment = (
            chain.get_segment_by_descriptor(
                f"{intermediate_unit_index}R"
                f"{intermediate_unit_index}B"
            )
        )

        right_gap = get_node_segment_gap(
            lower_right_node,
            intermediate_right_segment,
            "counterclockwise",
        )

        right_clearance = (
            right_gap.result.length_mm
            - contact_offset
        )

        # -----------------------------------------------------
        # Competing contact:
        #
        #     Unit 1 L-T segment
        #         with
        #     Unit 0 T node
        #
        # This is the penetration we explicitly want to prevent.
        # -----------------------------------------------------

        lower_top_node = (
            chain.get_node_by_descriptor(
                f"{lower_unit_index}T"
            )
        )

        intermediate_left_segment = (
            chain.get_segment_by_descriptor(
                f"{intermediate_unit_index}L"
                f"{intermediate_unit_index}T"
            )
        )

        left_gap = get_node_segment_gap(
            lower_top_node,
            intermediate_left_segment,
            "clockwise",
        )

        left_clearance = (
            abs(left_gap.result.length_mm)
            - contact_offset
        )

        return {
            "right_clearance": float(
                right_clearance
            ),
            "left_clearance": float(
                left_clearance
            ),
        }

    # ---------------------------------------------------------
    # 3. Find the dz that WOULD make the target contact zero
    # ---------------------------------------------------------

    def residual(x):

        result = evaluate_contacts(
            x[0]
        )

        return np.array([
            result[
                "right_clearance"
            ]
        ])

    solution = least_squares(
        residual,
        x0=np.array(
            [dz_guess],
            dtype=float,
        ),
        bounds=(
            [dz_bounds[0]],
            [dz_bounds[1]],
        ),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )

    target_dz = float(
        solution.x[0]
    )

    # ---------------------------------------------------------
    # 4. Check whether Unit 1L-T / Unit 0T becomes limiting
    #    before we reach the desired right-side contact.
    #
    # We know dz = 0 is the starting configuration.
    # target_dz is the full desired downward motion.
    #
    # If target_dz is already valid, use it directly.
    # Otherwise bisect between zero and target_dz to find the
    # largest valid downward shift.
    # ---------------------------------------------------------

    target_result = (
        evaluate_contacts(
            target_dz
        )
    )

    target_is_valid = (
        target_result[
            "left_clearance"
        ]
        >= -contact_tolerance
    )

    constraint_limited = False

    if target_is_valid:

        dz = (
            target_dz
        )

    else:

        constraint_limited = True

        # -----------------------------------------------------
        # The requested target shift penetrates 1L1T / 0T.
        #
        # Find the last nonpenetrating shift between:
        #
        #     dz = 0
        #     dz = target_dz
        # -----------------------------------------------------

        valid_alpha = 0.0
        invalid_alpha = 1.0

        dz_tolerance = 1e-9

        while (
            abs(
                invalid_alpha
                - valid_alpha
            )
            * abs(target_dz)
            > dz_tolerance
        ):

            trial_alpha = (
                valid_alpha
                + invalid_alpha
            ) / 2.0

            trial_dz = (
                trial_alpha
                * target_dz
            )

            trial_result = (
                evaluate_contacts(
                    trial_dz
                )
            )

            trial_valid = (
                trial_result[
                    "left_clearance"
                ]
                >= -contact_tolerance
            )

            if trial_valid:

                valid_alpha = (
                    trial_alpha
                )

            else:

                invalid_alpha = (
                    trial_alpha
                )

        dz = (
            valid_alpha
            * target_dz
        )

    # ---------------------------------------------------------
    # 5. Leave chain at final reachable configuration
    # ---------------------------------------------------------

    final_result = (
        evaluate_contacts(
            dz
        )
    )

    right_clearance = (
        final_result[
            "right_clearance"
        ]
    )

    left_clearance = (
        final_result[
            "left_clearance"
        ]
    )

    # ---------------------------------------------------------
    # 6. Determine what happened
    # ---------------------------------------------------------

    target_contact_reached = (
        solution.success
        and abs(
            right_clearance
        )
        <= contact_tolerance
    )

    competing_contact_reached = (
        abs(
            left_clearance
        )
        <= contact_tolerance
    )

    success = (
        target_contact_reached
        or constraint_limited
    )

    # ---------------------------------------------------------
    # 7. Report
    # ---------------------------------------------------------

    if verbose:

        print(
            "\nLower contact shift:"
        )

        print(
            f"  desired dz: "
            f"{target_dz:.6f} mm"
        )

        print(
            f"  actual reachable dz: "
            f"{dz:.6f} mm"
        )

        print(
            f"  1R1B-0R clearance: "
            f"{right_clearance:.9f} mm"
        )

        print(
            f"  1L1T-0T clearance: "
            f"{left_clearance:.9f} mm"
        )

        print(
            f"  target contact reached: "
            f"{target_contact_reached}"
        )

        print(
            f"  constraint limited: "
            f"{constraint_limited}"
        )

        if constraint_limited:

            print(
                "  1L1T-0T became limiting before "
                "1R1B-0R contact could be reached."
            )

        elif target_contact_reached:

            print(
                "  desired 1R1B-0R contact reached."
            )

    # ---------------------------------------------------------
    # 8. Return
    # ---------------------------------------------------------

    return {
        "success": success,
        "solver_success": (
            solution.success
        ),
        "lower_unit_index": (
            lower_unit_index
        ),
        "intermediate_unit_index": (
            intermediate_unit_index
        ),
        "target_dz": (
            target_dz
        ),
        "dz": (
            dz
        ),
        "right_clearance": (
            right_clearance
        ),
        "left_clearance": (
            left_clearance
        ),
        "contact_reached": (
            target_contact_reached
        ),
        "constraint_limited": (
            constraint_limited
        ),
        "limiting_contact": (
            (
                f"{intermediate_unit_index}L"
                f"{intermediate_unit_index}T-"
                f"{lower_unit_index}T"
            )
            if constraint_limited
            else None
        ),
        "solution": (
            solution
        ),
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

    lower_contact_result = find_right_lower_contact_shift(
        chain=chain,
        lower_unit_index=unit_0_index,
        contact_tolerance=contact_tolerance,
        verbose=verbose,
    )

    motions.append({
        "type": "translate_from",
        "start_unit_index": unit_1_index,
        "dy": 0.0,
        "dz": lower_contact_result["dz"],
    })

    if not lower_contact_result["success"]:
        return {
            "success": False,
            "direction": direction,
            "start_unit_index": start_unit_index,
            "trial_theta_deg": theta_deg,
            "max_theta_deg": max_theta_deg,
            "z_shift": z_shift,
            "motions": motions,
            "two_unit_solution": result,
            "cascade_result": cascade_result,
            "lower_contact_result": lower_contact_result,
            "segment_contact_result": None,
        }

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
    verbose=False,
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
    if isinstance(verbose, dict):
        verbose = verbose.get(
            "find_limiting_configuration_three_unit",
            False
        )

    if verbose:
        print("Finding limiting configuration...")

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

    if isinstance(verbose, dict):
        print_try_limiting_configuration_three_unit = verbose.get(
            "_try_limiting_configuration_three_unit",
            False
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
            verbose=_try_limiting_configuration_three_unit,
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

    plot_geometry(
        chain,
        plot_title="Intermediate plot",
        xlim=XLIM,
        ylim=FOUR_UNIT_YLIM,
        node_diameter=NODE_DIAMETER_PLOT,
        segment_line_width=SEG_LINE_WIDTH,
        rail_visual_offset=RAIL_VISUAL_OFFSET,
        rail_visual_shorten=RAIL_VISUAL_SHORTTEN,
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

    shared_rail_resolution = resolve_shared_rail_crossing(
        chain=chain,
        railed_unit_index=subset_unit_2_idx,
        moving_unit_index=subset_unit_3_idx,
        contact_tolerance=contact_tolerance,
        verbose=verbose,
    )

    # Almost worked; causes penetration at 3B3B and 2T
    """
    final_shared_rail_correction = enforce_shared_rail_node_spacing(
        chain,
        railed_unit_index=subset_unit_2_idx,
        tolerance=contact_tolerance,
    )

    if verbose:
        print(
            "\nFinal shared-rail correction:",
            final_shared_rail_correction,
        )
    """
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


def find_limiting_configuration_five_unit(
    chain,
    start_unit_index=0,
    direction="right",
    contact_tolerance=1e-6,
    verbose=True,
):
    """
    Build a five-unit rotational motion-limiting candidate
    from the corresponding four-unit candidate.
    """

    direction = direction.lower()

    # Indices of the four-unit subset.
    subset_unit_2_idx = start_unit_index + 2

    if direction != "right":
        raise NotImplementedError(
            "Only right rotation is currently implemented."
        )

    four_unit_result = find_limiting_configuration_four_unit(
        chain=chain,
        start_unit_index=start_unit_index,
        direction=direction,
        contact_tolerance=contact_tolerance,
        verbose=verbose,
    )

    _ = find_right_segment_segment_contact(
        chain=chain,
        lower_unit_index=subset_unit_2_idx,
        contact_tolerance=contact_tolerance,
        constraint_validator=lambda chain: configuration_is_valid(
            chain,
            contact_tolerance=contact_tolerance,
            verbose=False,
        ),
        verbose=True,
    )
    
    return {
        "success": True, # TODO: placeholder for now; should be updating this
        "direction": direction,
        "start_unit_index": start_unit_index,
        "four_unit_solution": four_unit_result,
    }


def find_limiting_configuration_six_unit(
    chain,
    start_index=0,
    direction="right",
    contact_tolerance=1e-6,
    verbose=True,
):
    _ = find_limiting_configuration_five_unit(
        chain,
        direction=direction,
        contact_tolerance=contact_tolerance,
        verbose=verbose,
    )
    
    subset_unit_4_idx = start_index + 4
    subset_unit_5_idx = start_index + 5
    
    correction = enforce_shared_rail_node_spacing(
        chain,
        railed_unit_index=subset_unit_4_idx,
    )

    last_two_unit_subset_result = (
        find_right_limiting_configuration_two_unit(
            chain=chain,
            lower_unit_index=subset_unit_4_idx,
            moving_unit_index=subset_unit_5_idx,
            contact_offset=chain.node_strut_contact_offset,
            contact_tolerance=contact_tolerance,
            verbose=verbose,
        )
    )


def find_limiting_configuration_n_unit(
    chain:RAMM_Chain,
    start_index=0,
    direction="right",
    contact_tolerance=1e-6,
    angle_step_deg=0.25,
    verbose=True,
):
    chain_length = len(chain.units)
    if chain_length < 2:
        print(f"Chains require at least 2 units. Chain length: {chain_length}")

    direction = direction.lower()
    if chain_length == 2:
        two_unit_result = find_limiting_configuration_two_unit(
            chain=chain,
            direction=direction,
            contact_tolerance=contact_tolerance,
            verbose=verbose,
        )
        return two_unit_result

    # 3 unit solution is used in 3-unit + solver
    # get 3 unit solution
    three_unit_result = find_limiting_configuration_three_unit(
        chain=chain,
        start_unit_index=start_index,
        direction=direction,
        contact_tolerance=contact_tolerance,
        verbose=verbose,
    )

    if chain_length == 3:
        return three_unit_result
    
    # SIX UNIT CHAIN WALKTHROUGH
    for unit_idx in range(chain_length - 3): # runs twice
        sovling_chain_length = 3 + 1 + unit_idx # TODO: obviously 4; clean this up later
        if (sovling_chain_length % 2) == 0:
            print(f"solving chain length {sovling_chain_length}")
            subset_railed_unit_index = sovling_chain_length - 2
            subset_free_moving_unit_idx = subset_railed_unit_index + 1
            correction = enforce_shared_rail_node_spacing(
                chain,
                railed_unit_index=subset_railed_unit_index, # 2 for four unit solver, 4 for six unit solver...etc TODO: check!
            )

            last_two_unit_subset_result = (
                find_right_limiting_configuration_two_unit(
                    chain=chain,
                    lower_unit_index=subset_railed_unit_index,
                    moving_unit_index=subset_free_moving_unit_idx,
                    contact_offset=chain.node_strut_contact_offset,
                    contact_tolerance=contact_tolerance,
                    verbose=verbose,
                )
            )

            shared_rail_resolution = resolve_shared_rail_crossing(
                chain=chain,
                railed_unit_index=subset_railed_unit_index,
                moving_unit_index=subset_free_moving_unit_idx,
                contact_tolerance=contact_tolerance,
                verbose=verbose,
            )
        else:
            lower_unit_idx = unit_idx + 1 # 2
            # usually run 4 unit solver here for 5 units; but should have that already if we're here

            _ = find_right_segment_segment_contact(
                    chain=chain,
                    lower_unit_index=lower_unit_idx, # for a five unit chain, this is 2
                    contact_tolerance=contact_tolerance,
                    constraint_validator=lambda chain: configuration_is_valid(
                        chain,
                        contact_tolerance=contact_tolerance,
                        verbose=False,
                    ),
                    verbose=True,
                )
    return
