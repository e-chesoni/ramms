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

from .core import *


def get_fractional_centerline_position(
        chain,
        free_unit_index,
        railed_unit_index,
        tolerance=1e-9
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
            raise InvalidRAMMGeometryError(
                f"FREE Unit {free_unit_index}'s top node is not "
                f"between RAILED Unit {railed_unit_index}'s "
                f"bottom and top nodes. s = {s:.6f}"
            )

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

def rotate_with_cascade(
        chain,
        unit_index,
        pivot,
        degrees,
        penetration_validator=None,
        tolerance=1e-8,
        verbose=True
    ):
        """
        Rotate a FREE unit and propagate the resulting motion upward.

        First-version kinematic rule
        ----------------------------
        If the FREE unit has a RAILED unit immediately above it:

        1. Record the FREE top node's fractional position along the
        RAILED unit's centerline.
        2. Rotate the FREE unit.
        3. Preserve the RAILED unit's orientation.
        4. Translate the RAILED unit and every unit above it so the
        moved FREE top node retains the same fractional position.
        5. Validate the centerline constraint.
        6. Optionally validate nonpenetration.

        Parameters
        ----------
        unit_index:
            Index of the FREE unit being rotated.

        pivot:
            RAMM_Node or coordinate pair about which the FREE unit
            rotates.

        degrees:
            Counterclockwise rotation in degrees.

        penetration_validator:
            Optional callable:

                penetration_validator(chain) -> bool

            It should return True when the resulting chain is valid
            and False when penetration occurs.

            If validation fails, the entire move is undone.

        tolerance:
            Numerical tolerance for centerline validation.

        verbose:
            Print information about the cascading motion.
        """
        if not 0 <= unit_index < len(chain.units):
            raise IndexError(
                f"Invalid unit index: {unit_index}"
            )

        free_unit = chain.units[unit_index]

        if free_unit.unit_type != UnitType.FREE:
            raise ValueError(
                "rotate_with_cascade() must initially be called "
                "on a FREE unit."
            )

        saved_coordinates = chain._save_coordinates()

        railed_unit_index = unit_index + 1

        has_railed_unit_above = (
            railed_unit_index < len(chain.units)
            and chain.units[railed_unit_index].unit_type
            == UnitType.RAILED
        )

        try:
            if has_railed_unit_above:
                fractional_position = (
                    get_fractional_centerline_position(
                        chain,
                        free_unit_index=unit_index,
                        railed_unit_index=railed_unit_index,
                        tolerance=tolerance
                    )
                )
            else:
                fractional_position = None

            # Rotate only the requested FREE unit first.
            free_unit.rotate(
                pivot=pivot,
                degrees=degrees
            )

            translation = (0.0, 0.0)

            if has_railed_unit_above:
                translation = (
                        reposition_railed_unit_from_free_top(
                        chain,
                        free_unit_index=unit_index,
                        railed_unit_index=railed_unit_index,
                        fractional_position=fractional_position
                    )
                )

                actual_s = (
                    validate_free_railed_centerline_constraint(
                        chain,
                        free_unit_index=unit_index,
                        railed_unit_index=railed_unit_index,
                        expected_fractional_position=(
                            fractional_position
                        ),
                        tolerance=tolerance
                    )
                )
            else:
                actual_s = None

            # Let the existing contact framework decide whether
            # the new configuration penetrates.
            if penetration_validator is not None:
                is_valid = penetration_validator(chain)

                if not is_valid:
                    raise InvalidRAMMGeometryError(
                        "Cascading rotation caused penetration."
                    )

        except Exception:
            chain._restore_coordinates(saved_coordinates)
            raise

        if verbose:
            print(
                f"Rotated FREE Unit {unit_index} by "
                f"{degrees:.6f}°."
            )

            if has_railed_unit_above:
                dy, dz = translation

                print(
                    f"Translated RAILED Unit "
                    f"{railed_unit_index} and all units above it:"
                )
                print(f"  dy = {dy:.6f} mm")
                print(f"  dz = {dz:.6f} mm")
                print(
                    f"Preserved fractional centerline position: "
                    f"s = {actual_s:.6f}"
                )
            else:
                print(
                    "No RAILED unit exists immediately above; "
                    "no upward cascade was required."
                )

        return {
            "success": True,
            "rotated_unit_index": unit_index,
            "degrees": degrees,
            "railed_unit_index": (
                railed_unit_index
                if has_railed_unit_above
                else None
            ),
            "fractional_position": actual_s,
            "translation": translation
        }

def set_two_unit_configuration(chain, theta, z):
    pass