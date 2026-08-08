import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-whitegrid")
import math

from .core import *


def get_fractional_centerline_position(
        chain,
        free_unit_index,
        railed_unit_index,
        tolerance=1e-6
    ):
        """
        Calculate the fractional position of the FREE unit's top node
        along the neighboring RAILED unit's bottom-to-top centerline.

        s = 0:
            FREE top coincides with the RAILED bottom node.

        s = 1:
            FREE top coincides with the RAILED top node.

        0 < s < 1:
            FREE top lies between the RAILED bottom and top nodes.
        """
        free_unit = chain.units[free_unit_index]
        railed_unit = chain.units[railed_unit_index]

        if free_unit.unit_type != UnitType.FREE:
            raise ValueError(
                f"Unit {free_unit_index} must be FREE."
            )

        if railed_unit.unit_type != UnitType.RAILED:
            raise ValueError(
                f"Unit {railed_unit_index} must be RAILED."
            )

        free_top = np.asarray(
            free_unit.top_node.coordinates,
            dtype=float
        )

        railed_bottom = np.asarray(
            railed_unit.bottom_node.coordinates,
            dtype=float
        )

        railed_top = np.asarray(
            railed_unit.top_node.coordinates,
            dtype=float
        )

        centerline = railed_top - railed_bottom
        centerline_length_squared = np.dot(
            centerline,
            centerline
        )

        if centerline_length_squared <= tolerance:
            raise InvalidRAMMGeometryError(
                f"RAILED Unit {railed_unit_index} has a "
                "zero-length centerline."
            )

        s = np.dot(
            free_top - railed_bottom,
            centerline
        ) / centerline_length_squared

        # Perpendicular error between the FREE top node and
        # the RAILED unit centerline.
        projection = railed_bottom + s * centerline

        centerline_error = np.linalg.norm(
            free_top - projection
        )

        if centerline_error > tolerance:
            raise InvalidRAMMGeometryError(
                f"FREE Unit {free_unit_index}'s top node is not "
                f"on RAILED Unit {railed_unit_index}'s centerline. "
                f"Error: {centerline_error:.6e}"
            )

        if s < -tolerance or s > 1 + tolerance:
            """
            raise InvalidRAMMGeometryError(
                f"FREE Unit {free_unit_index}'s top node is not "
                f"between RAILED Unit {railed_unit_index}'s "
                f"bottom and top nodes. s = {s:.6f}"
            )
            """
            print(f"WARNING: FREE Unit 1's top node is not between RAILED Unit 2's bottom and top nodes. s = {s}")
            # TODO: instead of warning me, fix it! (translate the top unit)
        return float(s)


def translate_units_from(
        chain,
        start_unit_index,
        dy,
        dz
    ):
        """
        Translate start_unit_index and every unit above it.

        This preserves the relative geometry of the upper portion
        of the chain.
        """
        if not 0 <= start_unit_index < len(chain.units):
            raise IndexError(
                f"Invalid start unit index: {start_unit_index}"
            )

        for unit in chain.units[start_unit_index:]:
            unit.translate(
                dy=dy,
                dz=dz
            )


def reposition_railed_unit_from_free_top(
        chain,
        free_unit_index,
        railed_unit_index,
        fractional_position
    ):
        """
        Translate the neighboring RAILED unit and all units above it
        so the FREE top node remains at the specified fractional
        position along the RAILED unit's centerline.

        The RAILED unit's orientation is preserved in this first model.
        """
        free_unit = chain.units[free_unit_index]
        railed_unit = chain.units[railed_unit_index]

        free_top = np.asarray(
            free_unit.top_node.coordinates,
            dtype=float
        )

        current_bottom = np.asarray(
            railed_unit.bottom_node.coordinates,
            dtype=float
        )

        current_top = np.asarray(
            railed_unit.top_node.coordinates,
            dtype=float
        )

        centerline = current_top - current_bottom

        # We want:
        #
        # free_top = new_bottom + s * centerline
        #
        # Therefore:
        #
        # new_bottom = free_top - s * centerline
        target_bottom = (
            free_top
            - fractional_position * centerline
        )

        translation = target_bottom - current_bottom
        dy, dz = translation

        translate_units_from(
            chain,
            start_unit_index=railed_unit_index,
            dy=dy,
            dz=dz
        )

        return float(dy), float(dz)


def align_railed_unit_to_free_top(
    chain,
    free_unit_index,
    railed_unit_index
):
    """
    Translate the RAILED unit and everything above it only in the
    direction perpendicular to its centerline.

    This forces the FREE top node onto the RAILED centerline while
    allowing the FREE node to slide along the rail direction.
    """
    free_unit = chain.units[free_unit_index]
    railed_unit = chain.units[railed_unit_index]

    free_top = np.asarray(
        free_unit.top_node.coordinates,
        dtype=float
    )

    railed_bottom = np.asarray(
        railed_unit.bottom_node.coordinates,
        dtype=float
    )

    railed_top = np.asarray(
        railed_unit.top_node.coordinates,
        dtype=float
    )

    centerline = railed_top - railed_bottom

    length_squared = np.dot(
        centerline,
        centerline
    )

    if length_squared <= 1e-12:
        raise InvalidRAMMGeometryError(
            "RAILED unit has zero-length centerline."
        )

    # Where does the FREE top project onto the current centerline?
    s = np.dot(
        free_top - railed_bottom,
        centerline
    ) / length_squared

    projection = (
        railed_bottom
        + s * centerline
    )

    # Vector from the centerline to the FREE top node.
    #
    # Importantly, this vector is perpendicular to the centerline.
    perpendicular_error = (
        free_top - projection
    )

    dy, dz = perpendicular_error

    # Shift the RAILED unit and everything above it sideways
    # just enough to put the FREE top on its centerline.
    translate_units_from(
        chain,
        start_unit_index=railed_unit_index,
        dy=dy,
        dz=dz
    )

    return {
        "dy": float(dy),
        "dz": float(dz),
        "fractional_position": float(s)
    }


def validate_free_railed_centerline_constraint(
        chain,
        free_unit_index,
        railed_unit_index,
        expected_fractional_position=None,
        tolerance=1e-8
    ):
        """
        Confirm that the FREE top node lies on and between the
        neighboring RAILED unit's bottom-to-top centerline.
        """
        actual_s = get_fractional_centerline_position(
            chain,
            free_unit_index=free_unit_index,
            railed_unit_index=railed_unit_index,
            tolerance=tolerance
        )

        if expected_fractional_position is not None:
            if not math.isclose(
                actual_s,
                expected_fractional_position,
                abs_tol=tolerance
            ):
                raise InvalidRAMMGeometryError(
                    "The fractional rail position changed during "
                    "the cascading transformation. "
                    f"Expected {expected_fractional_position:.9f}, "
                    f"received {actual_s:.9f}."
                )

        return actual_s


def _apply_cascading_rotation(
    chain,
    unit_index,
    pivot,
    degrees,
    tolerance=1e-8
):
    """
    Apply exactly the requested rotation and cascade it upward.

    This function does NOT search for a contact limit.
    """

    free_unit = chain.units[unit_index]

    free_unit.rotate(
        pivot=pivot,
        degrees=degrees
    )

    railed_unit_index = unit_index + 1

    has_railed_unit_above = (
        railed_unit_index < len(chain.units)
        and chain.units[railed_unit_index].unit_type
        == UnitType.RAILED
    )

    translation = (0.0, 0.0)
    actual_s = None

    if has_railed_unit_above:

        alignment = align_railed_unit_to_free_top(
            chain,
            free_unit_index=unit_index,
            railed_unit_index=railed_unit_index
        )

        translation = (
            alignment["dy"],
            alignment["dz"]
        )

        actual_s = get_fractional_centerline_position(
            chain,
            free_unit_index=unit_index,
            railed_unit_index=railed_unit_index,
            tolerance=tolerance
        )

    return {
        "translation": translation,
        "fractional_position": actual_s,
        "railed_unit_index": (
            railed_unit_index
            if has_railed_unit_above
            else None
        )
    }


def rotate_with_cascade(
    chain,
    unit_index,
    pivot,
    degrees,
    constraint_validator,
    angle_tolerance=1e-6,
    tolerance=1e-6,
    verbose=True
):
    """
    Description
    ------------
        Attempt a cascading rotation about an arbitrary pivot.

        The chain follows the requested motion until the first physical
        or geometric constraint prevents further motion.

        If the full requested rotation is valid, the full motion is applied.

        If the full requested rotation is invalid, the method uses bisection
        to find the largest valid rotation between 0 and the requested angle.

    Parameters
    ------------
        chain:
            RAMM_Chain being transformed.

        unit_index:
            Index of the FREE unit being rotated.

        pivot:
            RAMM_Node or (y, z) coordinate pair about which the FREE unit
            attempts to rotate.

        degrees:
            Requested counterclockwise rotation in degrees.

        constraint_validator:
            Callable with signature:

                constraint_validator(chain) -> bool

            Returns True when the resulting configuration is physically valid
            and False when penetration or another forbidden condition occurs.

        angle_tolerance:
            Resolution used when searching for the contact-limited rotation.

        tolerance:
            Numerical tolerance used by the cascading geometry helpers.

        verbose:
            Print information about the resulting motion.
    """

    # ---------------------------------------------------------
    # Validate requested unit
    # ---------------------------------------------------------

    if not 0 <= unit_index < len(chain.units):
        raise IndexError(
            f"Invalid unit index: {unit_index}"
        )

    free_unit = chain.units[unit_index]

    if free_unit.unit_type != UnitType.FREE:
        raise ValueError(
            "rotate_with_cascade() must be called on a FREE unit."
        )

    if degrees == 0:
        return {
            "success": True,
            "requested_degrees": 0.0,
            "actual_degrees": 0.0,
            "contact_limited": False,
            "translation": (0.0, 0.0),
            "fractional_position": None
        }

    # Save the state at the beginning of THIS motion.
    starting_coordinates = chain._save_coordinates()

    # ---------------------------------------------------------
    # 1. Try the full requested motion
    # ---------------------------------------------------------

    try:
        full_result = _apply_cascading_rotation(
            chain,
            unit_index=unit_index,
            pivot=pivot,
            degrees=degrees,
            tolerance=tolerance
        )

        full_motion_valid = constraint_validator(
            chain
        )

    except InvalidRAMMGeometryError:
        # Examples:
        # - FREE node moves beyond the finite rail span
        # - centerline constraint becomes geometrically impossible
        #
        # This does not mean the function should crash.
        # It means the requested angle is beyond the allowed motion.
        full_motion_valid = False
        full_result = None

    # ---------------------------------------------------------
    # 2. Full motion is valid
    # ---------------------------------------------------------

    if full_motion_valid:

        if verbose:
            print(
                f"Full requested rotation is valid."
            )
            print(
                f"Requested rotation: {degrees:.6f}°"
            )
            print(
                f"Actual rotation:    {degrees:.6f}°"
            )

        return {
            "success": True,
            "requested_degrees": degrees,
            "actual_degrees": degrees,
            "contact_limited": False,
            "translation": full_result["translation"],
            "fractional_position": (
                full_result["fractional_position"]
            )
        }

    # ---------------------------------------------------------
    # 3. Full motion is invalid
    #
    # Restore the chain to the state before this attempted motion.
    # ---------------------------------------------------------

    chain._restore_coordinates(
        starting_coordinates
    )

    # alpha represents the fraction of the requested motion:
    #
    # alpha = 0 -> starting configuration
    # alpha = 1 -> full requested rotation
    #
    # We know:
    # alpha = 0 is valid
    # alpha = 1 is invalid
    valid_alpha = 0.0
    invalid_alpha = 1.0

    # ---------------------------------------------------------
    # 4. Binary search for the last valid configuration
    # ---------------------------------------------------------

    while (
        abs(invalid_alpha - valid_alpha)
        * abs(degrees)
        > angle_tolerance
    ):

        trial_alpha = (
            valid_alpha + invalid_alpha
        ) / 2.0

        trial_degrees = (
            trial_alpha * degrees
        )

        # Every trial must begin from the exact same starting state.
        chain._restore_coordinates(
            starting_coordinates
        )

        try:
            _apply_cascading_rotation(
                chain,
                unit_index=unit_index,
                pivot=pivot,
                degrees=trial_degrees,
                tolerance=tolerance
            )

            trial_valid = constraint_validator(
                chain
            )

        except InvalidRAMMGeometryError:
            trial_valid = False

        if trial_valid:
            valid_alpha = trial_alpha

        else:
            invalid_alpha = trial_alpha

    # ---------------------------------------------------------
    # 5. Apply the final valid rotation
    # ---------------------------------------------------------

    actual_degrees = (
        valid_alpha * degrees
    )

    chain._restore_coordinates(
        starting_coordinates
    )

    final_result = _apply_cascading_rotation(
        chain,
        unit_index=unit_index,
        pivot=pivot,
        degrees=actual_degrees,
        tolerance=tolerance
    )

    # Sanity check the final state.
    final_valid = constraint_validator(
        chain
    )

    if not final_valid:
        chain._restore_coordinates(
            starting_coordinates
        )

        raise InvalidRAMMGeometryError(
            "Could not find a valid contact-limited "
            "cascading rotation."
        )

    # ---------------------------------------------------------
    # 6. Report result
    # ---------------------------------------------------------

    if verbose:
        print(
            f"Requested rotation: "
            f"{degrees:.6f}°"
        )
        print(
            f"Rotation limited by contact/constraint: "
            f"{actual_degrees:.6f}°"
        )
        print(
            f"Motion fraction completed: "
            f"{valid_alpha:.6f}"
        )

    return {
        "success": True,
        "requested_degrees": degrees,
        "actual_degrees": actual_degrees,
        "contact_limited": True,
        "translation": (
            final_result["translation"]
        ),
        "fractional_position": (
            final_result["fractional_position"]
        )
    }

def set_two_unit_configuration(chain, theta, z):
    pass