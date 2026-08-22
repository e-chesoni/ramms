import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-whitegrid")
import math

from .core import *
from .logger import *


def _get_free_node_position_along_rail(
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


def _translate_units_from(
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


def _reposition_railed_unit_from_free_top(
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

        _translate_units_from(
            chain,
            start_unit_index=railed_unit_index,
            dy=dy,
            dz=dz
        )

        return float(dy), float(dz)


def _align_railed_unit_to_free_top(
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
    _translate_units_from(
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


def _validate_free_railed_centerline_constraint(
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
        actual_s = _get_free_node_position_along_rail(
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


def enforce_shared_rail_node_spacing(
    chain,
    railed_unit_index,
    tolerance=1e-6,
):
    """
    Ensure the FREE nodes on either side of an interior RAILED unit
    do not overlap along the rail centerline.
    """

    lower_free_index = railed_unit_index - 1
    upper_free_index = railed_unit_index + 1

    if upper_free_index >= len(chain.units):
        return (0.0, 0.0)

    railed_unit = chain.units[railed_unit_index]

    rail_bottom = np.asarray(
        railed_unit.bottom_node.coordinates,
        dtype=float,
    )

    rail_top = np.asarray(
        railed_unit.top_node.coordinates,
        dtype=float,
    )

    rail_vector = rail_top - rail_bottom
    rail_length = np.linalg.norm(rail_vector)

    if rail_length <= tolerance:
        raise InvalidRAMMGeometryError(
            f"RAILED Unit {railed_unit_index} has zero-length centerline."
        )

    rail_direction = rail_vector / rail_length

    lower_node = np.asarray(
        chain.units[lower_free_index].top_node.coordinates,
        dtype=float,
    )

    upper_node = np.asarray(
        chain.units[upper_free_index].bottom_node.coordinates,
        dtype=float,
    )

    # Positions measured along the RAILED unit's local centerline.
    lower_position = np.dot(
        lower_node - rail_bottom,
        rail_direction,
    )

    upper_position = np.dot(
        upper_node - rail_bottom,
        rail_direction,
    )

    current_spacing = upper_position - lower_position
    required_spacing = chain.node_diameter

    if current_spacing >= required_spacing - tolerance:
        return (0.0, 0.0)

    correction = required_spacing - current_spacing

    translation = correction * rail_direction
    dy, dz = translation

    _translate_units_from(
        chain,
        start_unit_index=upper_free_index,
        dy=dy,
        dz=dz,
    )

    return float(dy), float(dz)


def _apply_propagating_rotation(
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

        alignment = _align_railed_unit_to_free_top(
            chain,
            free_unit_index=unit_index,
            railed_unit_index=railed_unit_index
        )

        translation = (
            alignment["dy"],
            alignment["dz"]
        )

        actual_s = _get_free_node_position_along_rail(
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


def set_adjacent_unit_configuration(
    chain,
    moving_unit_index,
    starting_coordinates,
    theta_deg,
    z_shift,
):
    """
    Set a moving FREE unit to a trial rotation and translation
    from its starting configuration.

    If a RAILED unit exists immediately above the moving FREE
    unit, preserve the FREE top node's position along that rail.
    """

    # Restore the configuration from the beginning of the solve.
    chain._restore_coordinates(starting_coordinates)

    moving_unit = chain.units[moving_unit_index]

    railed_unit_index = moving_unit_index + 1

    has_railed_unit_above = (
        railed_unit_index < len(chain.units)
        and chain.units[railed_unit_index].unit_type
        == UnitType.RAILED
    )

    # Save where the FREE top node currently lies along the
    # RAILED unit above it before applying the trial motion.
    fractional_position = None

    if has_railed_unit_above:
        fractional_position = (
            _get_free_node_position_along_rail(
                chain,
                free_unit_index=moving_unit_index,
                railed_unit_index=railed_unit_index,
            )
        )

    pivot = moving_unit.bottom_node

    moving_unit.rotate(
        pivot=pivot,
        degrees=theta_deg
    )

    moving_unit.translate(
        dy=0.0,
        dz=z_shift
    )

    # Move the RAILED unit and everything above it so the FREE
    # top node remains at the same position along its rails.
    if has_railed_unit_above:
        _reposition_railed_unit_from_free_top(
            chain,
            free_unit_index=moving_unit_index,
            railed_unit_index=railed_unit_index,
            fractional_position=fractional_position,
        )


def get_free_bottom_to_preceding_rail_top_distance(
    chain,
    free_unit_index,
):
    """
    Return the signed distance along the preceding RAILED unit's
    centerline from its top node to the FREE unit's bottom node.

    Positive:
        FREE bottom lies beyond the RAILED top in the local
        bottom-to-top rail direction.

    Zero:
        Nodes coincide along the rail direction.

    Negative:
        FREE bottom lies below the RAILED top along the local
        rail direction.
    """

    if free_unit_index <= 0:
        raise ValueError(
            "FREE unit must have a preceding unit."
        )

    free_unit = chain.units[free_unit_index]
    railed_unit = chain.units[free_unit_index - 1]

    if free_unit.unit_type != UnitType.FREE:
        raise ValueError(
            f"Unit {free_unit_index} must be FREE."
        )

    if railed_unit.unit_type != UnitType.RAILED:
        raise ValueError(
            f"Unit {free_unit_index - 1} must be RAILED."
        )

    rail_bottom = np.asarray(
        railed_unit.bottom_node.coordinates,
        dtype=float,
    )

    rail_top = np.asarray(
        railed_unit.top_node.coordinates,
        dtype=float,
    )

    free_bottom = np.asarray(
        free_unit.bottom_node.coordinates,
        dtype=float,
    )

    rail_vector = rail_top - rail_bottom
    rail_length = np.linalg.norm(rail_vector)

    if rail_length <= 1e-12:
        raise InvalidRAMMGeometryError(
            f"RAILED Unit {free_unit_index - 1} "
            "has zero-length centerline."
        )

    rail_direction = rail_vector / rail_length

    return float(
        np.dot(
            free_bottom - rail_top,
            rail_direction,
        )
    )


def get_free_bottom_rail_top_distances(chain):
    """
    Return local rail-direction distances for every FREE unit.
    """

    distances = {}

    for unit_index, unit in enumerate(chain.units):

        if unit.unit_type != UnitType.FREE:
            continue

        distances[unit_index] = (
            get_free_bottom_to_preceding_rail_top_distance(
                chain,
                free_unit_index=unit_index,
            )
        )

    return distances


def find_max_valid_translation(
    target_translation,
    apply_translation,
    constraint_validator,
    translation_tolerance=1e-6,
    n_initial_steps=100,
):
    """
    Find the furthest valid translation toward a target translation.

    Parameters
    ----------
    target_translation:
        Desired scalar translation.

    apply_translation:
        Callable that places the chain at a specified trial
        translation from the saved starting configuration.

    constraint_validator:
        Callable that returns True if the resulting chain
        configuration is physically valid.

    translation_tolerance:
        Resolution used to refine the limiting translation.

    n_initial_steps:
        Number of coarse steps used to locate the first
        invalid configuration.

    Returns
    -------
    dict
        Whether the target was reached and, if not, the last
        valid translation before another constraint intervened.
    """

    last_valid_translation = 0.0
    first_invalid_translation = None

    trial_translations = np.linspace(
        0.0,
        target_translation,
        n_initial_steps,
    )

    # Find the first invalid point along the requested motion.
    for trial_translation in trial_translations[1:]:

        apply_translation(trial_translation)

        if constraint_validator():
            last_valid_translation = trial_translation
        else:
            first_invalid_translation = trial_translation
            break

    # The entire requested translation is valid.
    if first_invalid_translation is None:
        apply_translation(target_translation)

        return {
            "target_reached": True,
            "translation": target_translation,
            "constraint_limited": False,
        }

    # Refine the transition from valid to invalid.
    valid_translation = last_valid_translation
    invalid_translation = first_invalid_translation

    while (
        abs(invalid_translation - valid_translation)
        > translation_tolerance
    ):
        trial_translation = (
            valid_translation + invalid_translation
        ) / 2.0

        apply_translation(trial_translation)

        if constraint_validator():
            valid_translation = trial_translation
        else:
            invalid_translation = trial_translation

    # Leave the chain at the last valid configuration.
    apply_translation(valid_translation)

    return {
        "target_reached": False,
        "translation": valid_translation,
        "constraint_limited": True,
    }


@log_call
def propagate_free_unit_rotation(
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
        full_result = _apply_propagating_rotation(
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
            _apply_propagating_rotation(
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

    final_result = _apply_propagating_rotation(
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
