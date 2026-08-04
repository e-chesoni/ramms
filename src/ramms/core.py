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

from .exceptions import InvalidRAMMGeometryError

class UnitType(Enum):
    RAILED = "railed"
    FREE = "free"


class RAMM_Node:
    """
    A point belonging to a RAMM unit.

    Standard node examples:
        top, right, bottom, left

    Virtual-node examples:
        left rail lower, left rail upper
    """

    def __init__(
        self,
        long_name,
        coordinates,
        short_label=None,
        is_virtual=False
    ):
        self.long_name = long_name

        # Coordinates are stored as (y, z):
        # y = horizontal position
        # z = vertical position
        self.coordinates = coordinates

        # Standard nodes default to the first letter:
        # top -> T, right -> R, etc.
        #
        # Virtual rail nodes receive explicit labels such as:
        # RLL = rail left lower
        self.short_label = (
            short_label
            if short_label is not None
            else long_name[0].upper()
        )

        self.is_virtual = is_virtual
        self.unit_index = None

    @property
    def get_details(self):
        return f"{self.long_name}: {self.coordinates}"

    @property
    def descriptor(self):
        """
        Return a shortened descriptor.

        Examples
        --------
        Standard node:
            top node of Unit 0 -> 0T

        Virtual rail node:
            left rail lower node of Unit 0 -> 0RLL
        """
        return f"{self.unit_index}{self.short_label}"

    def __str__(self):
        return self.descriptor


class RAMM_Strut:
    """
    A segment connecting two RAMM_Node objects.

    Diamond nodes should be passed in clockwise order.

    Virtual rails are directed from their lower endpoint to their
    upper endpoint.
    """

    def __init__(
        self,
        node_1,
        node_2,
        is_virtual=False,
        segment_type="strut"
    ):
        self.node_1 = node_1
        self.node_2 = node_2

        self.is_virtual = is_virtual
        self.segment_type = segment_type

    @property
    def long_name(self):
        return (
            f"{self.node_1.long_name} -> "
            f"{self.node_2.long_name}"
        )

    @property
    def descriptor(self):
        """
        Return a shortened descriptor.

        Examples
        --------
        Diamond strut:
            Unit 1 bottom -> left = 1B1L

        Virtual left rail:
            Unit 0 rail-left-lower -> rail-left-upper
            = 0RLL0RLU
        """
        return (
            f"{self.node_1.descriptor}"
            f"{self.node_2.descriptor}"
        )

    def __str__(self):
        return self.descriptor


class RAMM_Unit:
    """
    A four-sided RAMM diamond unit.

    Nodes must be passed clockwise, beginning with the top node.

    RAILED units additionally contain two virtual rail segments.
    FREE units do not contain virtual rails.
    """

    def __init__(
        self,
        top_node,
        right_node,
        bottom_node,
        left_node,
        unit_type,
        node_diameter=2.0,
        rail_nodes=None,
        rails=None
    ):
        self.unit_type = unit_type
        self.node_diameter = float(node_diameter)

        if self.node_diameter <= 0:
            raise ValueError(
                "node_diameter must be greater than zero."
            )

        # ---------------------------------------------------------
        # Physical diamond nodes
        # ---------------------------------------------------------

        self.top_node = top_node
        self.right_node = right_node
        self.bottom_node = bottom_node
        self.left_node = left_node

        self.nodes = [
            top_node,
            right_node,
            bottom_node,
            left_node
        ]

        # Validate only the four diamond nodes.
        self.validate_geometry()

        # ---------------------------------------------------------
        # Physical diamond struts
        # ---------------------------------------------------------

        self.struts = [
            RAMM_Strut(
                top_node,
                right_node,
                is_virtual=False,
                segment_type="diamond_strut"
            ),
            RAMM_Strut(
                right_node,
                bottom_node,
                is_virtual=False,
                segment_type="diamond_strut"
            ),
            RAMM_Strut(
                bottom_node,
                left_node,
                is_virtual=False,
                segment_type="diamond_strut"
            ),
            RAMM_Strut(
                left_node,
                top_node,
                is_virtual=False,
                segment_type="diamond_strut"
            )
        ]

        # ---------------------------------------------------------
        # Virtual rail geometry
        # ---------------------------------------------------------

        self.rail_nodes = (
            list(rail_nodes)
            if rail_nodes is not None
            else []
        )

        self.rails = (
            list(rails)
            if rails is not None
            else []
        )

        if self.unit_type == UnitType.FREE:
            if self.rail_nodes or self.rails:
                raise InvalidRAMMGeometryError(
                    "FREE units cannot contain virtual rails."
                )

        if self.unit_type == UnitType.RAILED:
            if len(self.rail_nodes) != 4:
                raise InvalidRAMMGeometryError(
                    "A RAILED unit must contain four virtual "
                    "rail endpoint nodes."
                )

            if len(self.rails) != 2:
                raise InvalidRAMMGeometryError(
                    "A RAILED unit must contain two virtual rails."
                )

        # Every point that moves rigidly with the unit.
        self.all_nodes = self.nodes + self.rail_nodes

        # Every segment associated with the unit.
        self.all_segments = self.struts + self.rails

        self.rail_length = self._get_rail_length()

    def _get_rail_length(self):
        """
        Return the rail length if this is a RAILED unit.
        """
        if not self.rails:
            return 0.0

        rail = self.rails[0]

        y1, z1 = rail.node_1.coordinates
        y2, z2 = rail.node_2.coordinates

        return math.hypot(
            y2 - y1,
            z2 - z1
        )

    def validate_geometry(self):
        """
        Confirm that all four diamond struts have equal length.

        Virtual rails are intentionally excluded because their length
        is derived separately from the diamond geometry.
        """
        distances = []

        for i in range(4):
            node_1 = self.nodes[i]
            node_2 = self.nodes[(i + 1) % 4]

            y1, z1 = node_1.coordinates
            y2, z2 = node_2.coordinates

            distance = math.hypot(
                y2 - y1,
                z2 - z1
            )

            distances.append(distance)

        if not all(
            math.isclose(
                distance,
                distances[0],
                rel_tol=1e-9,
                abs_tol=1e-9
            )
            for distance in distances
        ):
            raise InvalidRAMMGeometryError(
                "RAMM unit diamond struts must have equal length."
            )

    @staticmethod
    def transform_point(point, pivot, degrees):
        """
        Rotate one (y, z) point counterclockwise around a pivot.
        """
        y, z = point
        pivot_y, pivot_z = pivot

        theta = math.radians(degrees)

        relative_y = y - pivot_y
        relative_z = z - pivot_z

        rotated_y = (
            relative_y * math.cos(theta)
            - relative_z * math.sin(theta)
        )

        rotated_z = (
            relative_y * math.sin(theta)
            + relative_z * math.cos(theta)
        )

        return (
            pivot_y + rotated_y,
            pivot_z + rotated_z
        )

    def transform(self, point, degrees): # TODO: Rename this rotate when you leave notebook
        """
        Rotate the entire rigid unit around a node or arbitrary point.
        """
        if isinstance(point, RAMM_Node):
            pivot = point.coordinates
        else:
            pivot = point

        for node in self.all_nodes:
            node.coordinates = self.transform_point(
                point=node.coordinates,
                pivot=pivot,
                degrees=degrees
            )

    def translate(self, dy=0.0, dz=0.0):
        """
        Translate the entire rigid unit.
        """
        for node in self.all_nodes:
            y, z = node.coordinates

            node.coordinates = (
                y + dy,
                z + dz
            )
    
    def get_rail(self, side):
        """
        Return the requested virtual rail.

        Parameters
        ----------
        side:
            "left" or "right"
        """
        if self.unit_type != UnitType.RAILED:
            raise ValueError(
                "FREE units do not contain rails."
            )

        side = side.lower()

        if side == "left":
            return self.left_rail

        if side == "right":
            return self.right_rail

        raise ValueError(
            "side must be 'left' or 'right'."
        )

    def get_struts_clockwise(self):
        """
        Return the four diamond struts in clockwise order,
        beginning with the top-right strut.

        Order:
            TR, RB, BL, LT
        """
        return list(self.struts)

    def get_strut_by_descriptor(self, descriptor):
        """
        Return the requested diamond strut.

        Example
        -------
        chain.get_strut_by_descriptor("1R1B")
        """
        for strut in self.get_struts_bottom_to_top_clockwise():
            if strut.descriptor == descriptor:
                return strut

        raise ValueError(
            f"No strut with descriptor '{descriptor}' found."
        )

    @classmethod
    def generate(
        cls,
        bottom_position,
        unit_type,
        rotation_deg=0,
        node_diameter=2.0
    ):
        """
        Generate a RAMM unit.

        Parameters
        ----------
        bottom_position:
            Bottom-node position given as (y, z).

        unit_type:
            UnitType.FREE or UnitType.RAILED.

        rotation_deg:
            Counterclockwise rotation in degrees about the bottom node.

        node_diameter:
            Diameter of the node constrained by the rails.

            For RAILED units, the two rail centerlines are positioned
            at +/- node_diameter / 2 from the unit center. Therefore,
            the rail centerlines are separated by node_diameter.
        """

        if node_diameter <= 0:
            raise ValueError(
                "node_diameter must be greater than zero."
            )

        # ---------------------------------------------------------
        # Define all base geometry in one place
        # ---------------------------------------------------------

        local_nodes = {
            "bottom": (0.0, 0.0),
            "right":  (8.0, 10.0),
            "top":    (0.0, 20.0),
            "left":   (-8.0, 10.0)
        }

        bottom_local_y, bottom_local_z = (
            local_nodes["bottom"]
        )

        right_local_y, right_local_z = (
            local_nodes["right"]
        )

        top_local_y, top_local_z = (
            local_nodes["top"]
        )

        left_local_y, left_local_z = (
            local_nodes["left"]
        )

        # The current unit is symmetric about its vertical centerline.
        unit_center_y = (
            bottom_local_y + top_local_y
        ) / 2

        unit_center_z = (
            bottom_local_z + top_local_z
        ) / 2

        diamond_half_width = max(
            abs(right_local_y - unit_center_y),
            abs(left_local_y - unit_center_y)
        )

        diamond_half_height = (
            top_local_z - bottom_local_z
        ) / 2

        # ---------------------------------------------------------
        # Local-to-global coordinate conversion
        # ---------------------------------------------------------

        def create_global_position(local_position):
            """
            Rotate around the local bottom node and then translate to
            the requested global bottom-node position.
            """
            rotated_y, rotated_z = cls.transform_point(
                point=local_position,
                pivot=(0.0, 0.0),
                degrees=rotation_deg
            )

            bottom_y, bottom_z = bottom_position

            return (
                bottom_y + rotated_y,
                bottom_z + rotated_z
            )

        # ---------------------------------------------------------
        # Create the four diamond nodes
        # ---------------------------------------------------------

        bottom_node = RAMM_Node(
            long_name="bottom",
            coordinates=create_global_position(
                local_nodes["bottom"]
            )
        )

        right_node = RAMM_Node(
            long_name="right",
            coordinates=create_global_position(
                local_nodes["right"]
            )
        )

        top_node = RAMM_Node(
            long_name="top",
            coordinates=create_global_position(
                local_nodes["top"]
            )
        )

        left_node = RAMM_Node(
            long_name="left",
            coordinates=create_global_position(
                local_nodes["left"]
            )
        )

        # Defaults for a FREE unit.
        rail_nodes = []
        rails = []

        # ---------------------------------------------------------
        # Generate rails only for a RAILED unit
        # ---------------------------------------------------------

        if unit_type == UnitType.RAILED:
            rail_half_spacing = node_diameter / 2

            if rail_half_spacing >= diamond_half_width:
                raise InvalidRAMMGeometryError(
                    "The requested node diameter is too large "
                    "for the rails to fit inside the diamond."
                )

            # The diamond edge narrows linearly from its widest point
            # toward the top and bottom.
            #
            # At horizontal offset rail_half_spacing:
            #
            # rail_half_length / diamond_half_height
            #     = 1 - rail_half_spacing / diamond_half_width
            rail_half_length = (
                diamond_half_height
                * (
                    1
                    - rail_half_spacing / diamond_half_width
                )
            )

            left_rail_y = (
                unit_center_y - rail_half_spacing
            )

            right_rail_y = (
                unit_center_y + rail_half_spacing
            )

            rail_lower_z = (
                unit_center_z - rail_half_length
            )

            rail_upper_z = (
                unit_center_z + rail_half_length
            )

            local_rail_positions = {
                "left_lower": (
                    left_rail_y,
                    rail_lower_z
                ),
                "left_upper": (
                    left_rail_y,
                    rail_upper_z
                ),
                "right_lower": (
                    right_rail_y,
                    rail_lower_z
                ),
                "right_upper": (
                    right_rail_y,
                    rail_upper_z
                )
            }

            # RLL = Rail Left Lower
            left_rail_lower = RAMM_Node(
                long_name="left rail lower",
                coordinates=create_global_position(
                    local_rail_positions["left_lower"]
                ),
                short_label="RLL",
                is_virtual=True
            )

            # RLU = Rail Left Upper
            left_rail_upper = RAMM_Node(
                long_name="left rail upper",
                coordinates=create_global_position(
                    local_rail_positions["left_upper"]
                ),
                short_label="RLU",
                is_virtual=True
            )

            # RRL = Rail Right Lower
            right_rail_lower = RAMM_Node(
                long_name="right rail lower",
                coordinates=create_global_position(
                    local_rail_positions["right_lower"]
                ),
                short_label="RRL",
                is_virtual=True
            )

            # RRU = Rail Right Upper
            right_rail_upper = RAMM_Node(
                long_name="right rail upper",
                coordinates=create_global_position(
                    local_rail_positions["right_upper"]
                ),
                short_label="RRU",
                is_virtual=True
            )

            rail_nodes = [
                left_rail_lower,
                left_rail_upper,
                right_rail_lower,
                right_rail_upper
            ]

            left_rail = RAMM_Strut(
                node_1=left_rail_lower,
                node_2=left_rail_upper,
                is_virtual=True,
                segment_type="rail"
            )

            right_rail = RAMM_Strut(
                node_1=right_rail_lower,
                node_2=right_rail_upper,
                is_virtual=True,
                segment_type="rail"
            )

            rails = [
                left_rail,
                right_rail
            ]

        unit = cls(
            top_node=top_node,
            right_node=right_node,
            bottom_node=bottom_node,
            left_node=left_node,
            unit_type=unit_type,
            node_diameter=node_diameter,
            rail_nodes=rail_nodes,
            rails=rails
        )

        # Convenient direct references.
        if unit_type == UnitType.RAILED:
            unit.left_rail = unit.rails[0]
            unit.right_rail = unit.rails[1]
        else:
            unit.left_rail = None
            unit.right_rail = None

        return unit


class RAMM_Chain:
    def __init__(
        self,
        node_diameter=2.0
    ):
        self.units = []
        self.node_diameter = float(node_diameter)
    
    def get_struts_bottom_to_top_clockwise(self):
        """
        Return all diamond struts in the chain.

        Order:
            Unit 0: TR, RB, BL, LT
            Unit 1: TR, RB, BL, LT
            ...
        """
        all_struts = []

        for unit in self.units:
            all_struts.extend(
                unit.get_struts_clockwise()
            )

        return all_struts

    @staticmethod
    def _get_unit_index_from_descriptor(descriptor):
        """
        Extract the leading integer from a descriptor.

        Examples
        --------
        0T       -> 0
        1B1L     -> 1
        10RLL    -> 10
        """
        match = re.match(r"^\d+", descriptor)

        if match is None:
            raise ValueError(
                f"Descriptor '{descriptor}' does not begin "
                "with a unit index."
            )

        return int(match.group())

    def validate_chain_geometry_upon_creation(self):
        for i, unit in enumerate(self.units):

            # Unit 0 must always be railed.
            if (
                i == 0
                and unit.unit_type == UnitType.FREE
            ):
                raise InvalidRAMMGeometryError(
                    "Unit 0 must be a RAILED unit."
                )

            # A FREE unit must initially have zero relative
            # horizontal offset from the preceding RAILED unit.
            if unit.unit_type == UnitType.FREE:
                previous_unit = self.units[i - 1]

                free_y = (
                    unit.bottom_node.coordinates[0]
                )

                neighbor_y = (
                    previous_unit.bottom_node.coordinates[0]
                )

                if not math.isclose(
                    free_y,
                    neighbor_y,
                    abs_tol=1e-9
                ):
                    raise InvalidRAMMGeometryError(
                        "A FREE unit must have zero relative "
                        "horizontal offset from its neighboring "
                        "RAILED unit upon chain creation."
                    )

    def get_node_by_descriptor(
        self,
        node_descriptor,
        include_virtual=True
    ):
        unit_index = self._get_unit_index_from_descriptor(
            node_descriptor
        )

        unit = self.units[unit_index]

        nodes_to_search = (
            unit.all_nodes
            if include_virtual
            else unit.nodes
        )

        for node in nodes_to_search:
            if node.descriptor == node_descriptor:
                return node

        raise ValueError(
            f"Unit {unit_index} has no node "
            f"'{node_descriptor}'."
        )

    def get_segment_by_descriptor(
        self,
        segment_descriptor,
        include_virtual=True
    ):
        unit_index = self._get_unit_index_from_descriptor(
            segment_descriptor
        )

        unit = self.units[unit_index]

        segments_to_search = (
            unit.all_segments
            if include_virtual
            else unit.struts
        )

        for segment in segments_to_search:
            if segment.descriptor == segment_descriptor:
                return segment

        raise ValueError(
            f"Unit {unit_index} has no segment "
            f"'{segment_descriptor}'."
        )

    def get_node_by_long_name(
        self,
        unit_index,
        node_long_name,
        include_virtual=True
    ):
        unit = self.units[unit_index]

        nodes_to_search = (
            unit.all_nodes
            if include_virtual
            else unit.nodes
        )

        for node in nodes_to_search:
            if node.long_name == node_long_name:
                return node

        raise ValueError(
            f"Unit {unit_index} has no node labeled "
            f"'{node_long_name}'."
        )

    def get_segment_by_long_name(
        self,
        unit_index,
        segment_long_name,
        include_virtual=True
    ):
        unit = self.units[unit_index]

        segments_to_search = (
            unit.all_segments
            if include_virtual
            else unit.struts
        )

        for segment in segments_to_search:
            if segment.long_name == segment_long_name:
                return segment

        raise ValueError(
            f"Unit {unit_index} has no segment "
            f"'{segment_long_name}'."
        )

    def get_rail(self, unit_index, side):
        """
        Return the left or right rail from a RAILED unit.

        Examples
        --------
        chain.get_rail(0, "left")
        chain.get_rail(2, "right")
        """
        unit = self.units[unit_index]

        return unit.get_rail(side)

    def get_unit_above(self, unit_index):
        """
        Return the unit immediately above the requested unit.

        Returns None when unit_index refers to the top unit.
        """
        above_index = unit_index + 1

        if above_index >= len(self.units):
            return None

        return self.units[above_index]

    def get_unit_below(self, unit_index):
        """
        Return the unit immediately below the requested unit.

        Returns None when unit_index refers to Unit 0.
        """
        below_index = unit_index - 1

        if below_index < 0:
            return None

        return self.units[below_index]
    
    def _get_fractional_centerline_position(
        self,
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
        free_unit = self.units[free_unit_index]
        railed_unit = self.units[railed_unit_index]

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

    def _save_coordinates(self):
        """
        Save all physical-node and rail-endpoint coordinates.

        The result can be passed to _restore_coordinates().
        """
        return [
            [
                tuple(node.coordinates)
                for node in unit.all_nodes
            ]
            for unit in self.units
        ]

    def _restore_coordinates(self, saved_coordinates):
        """
        Restore coordinates created by _save_coordinates().
        """
        if len(saved_coordinates) != len(self.units):
            raise ValueError(
                "Saved geometry does not match the current chain."
            )

        for unit, unit_coordinates in zip(
            self.units,
            saved_coordinates
        ):
            if len(unit_coordinates) != len(unit.all_nodes):
                raise ValueError(
                    "Saved unit geometry does not match the "
                    "current unit."
                )

            for node, coordinates in zip(
                unit.all_nodes,
                unit_coordinates
            ):
                node.coordinates = tuple(coordinates)

    def _translate_units_from(
        self,
        start_unit_index,
        dy,
        dz
    ):
        """
        Translate start_unit_index and every unit above it.

        This preserves the relative geometry of the upper portion
        of the chain.
        """
        if not 0 <= start_unit_index < len(self.units):
            raise IndexError(
                f"Invalid start unit index: {start_unit_index}"
            )

        for unit in self.units[start_unit_index:]:
            unit.translate(
                dy=dy,
                dz=dz
            )

    def _reposition_railed_unit_from_free_top(
        self,
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
        free_unit = self.units[free_unit_index]
        railed_unit = self.units[railed_unit_index]

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

        self._translate_units_from(
            start_unit_index=railed_unit_index,
            dy=dy,
            dz=dz
        )

        return float(dy), float(dz)

    def _validate_free_railed_centerline_constraint(
        self,
        free_unit_index,
        railed_unit_index,
        expected_fractional_position=None,
        tolerance=1e-8
    ):
        """
        Confirm that the FREE top node lies on and between the
        neighboring RAILED unit's bottom-to-top centerline.
        """
        actual_s = self._get_fractional_centerline_position(
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
        self,
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
        if not 0 <= unit_index < len(self.units):
            raise IndexError(
                f"Invalid unit index: {unit_index}"
            )

        free_unit = self.units[unit_index]

        if free_unit.unit_type != UnitType.FREE:
            raise ValueError(
                "rotate_with_cascade() must initially be called "
                "on a FREE unit."
            )

        saved_coordinates = self._save_coordinates()

        railed_unit_index = unit_index + 1

        has_railed_unit_above = (
            railed_unit_index < len(self.units)
            and self.units[railed_unit_index].unit_type
            == UnitType.RAILED
        )

        try:
            if has_railed_unit_above:
                fractional_position = (
                    self._get_fractional_centerline_position(
                        free_unit_index=unit_index,
                        railed_unit_index=railed_unit_index,
                        tolerance=tolerance
                    )
                )
            else:
                fractional_position = None

            # Rotate only the requested FREE unit first.
            free_unit.transform(
                point=pivot,
                degrees=degrees
            )

            translation = (0.0, 0.0)

            if has_railed_unit_above:
                translation = (
                    self._reposition_railed_unit_from_free_top(
                        free_unit_index=unit_index,
                        railed_unit_index=railed_unit_index,
                        fractional_position=fractional_position
                    )
                )

                actual_s = (
                    self._validate_free_railed_centerline_constraint(
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
                is_valid = penetration_validator(self)

                if not is_valid:
                    raise InvalidRAMMGeometryError(
                        "Cascading rotation caused penetration."
                    )

        except Exception:
            self._restore_coordinates(saved_coordinates)
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

    @classmethod
    def generate(
        cls,
        n_units,
        start_position,
        offsets,
        rotations=None,
        node_diameter=2.0
    ):
        """
        Generate a RAMM chain.

        Unit types automatically alternate:

            Unit 0: RAILED
            Unit 1: FREE
            Unit 2: RAILED
            Unit 3: FREE
            ...

        Parameters
        ----------
        n_units:
            Desired number of units.

        start_position:
            Bottom-node position (y, z) of Unit 0.

        offsets:
            One offset, two alternating offsets, or exactly
            n_units - 1 offsets.

        rotations:
            None, one repeated rotation, two alternating rotations,
            or exactly n_units rotations.

        node_diameter:
            Diameter used to define the spacing of the virtual rails.
        """

        if n_units < 1:
            raise ValueError(
                "A chain must contain at least one unit."
            )

        if node_diameter <= 0:
            raise ValueError(
                "node_diameter must be greater than zero."
            )

        required_offset_count = n_units - 1

        # ---------------------------------------------------------
        # Expand offsets
        # ---------------------------------------------------------

        if required_offset_count == 0:
            expanded_offsets = []

        elif (
            isinstance(offsets, tuple)
            and len(offsets) == 2
            and all(
                isinstance(value, (int, float))
                for value in offsets
            )
        ):
            expanded_offsets = [
                offsets
                for _ in range(required_offset_count)
            ]

        elif isinstance(offsets, (list, tuple)):
            if len(offsets) == 0:
                raise ValueError(
                    "At least one offset must be provided "
                    "for a chain containing multiple units."
                )

            if len(offsets) == 1:
                expanded_offsets = [
                    offsets[0]
                    for _ in range(required_offset_count)
                ]

            elif len(offsets) == 2:
                expanded_offsets = [
                    offsets[i % 2]
                    for i in range(required_offset_count)
                ]

            elif len(offsets) == required_offset_count:
                expanded_offsets = list(offsets)

            else:
                raise ValueError(
                    "offsets must contain one offset, "
                    "two alternating offsets, or exactly "
                    f"{required_offset_count} offsets."
                )

        else:
            raise TypeError(
                "offsets must be an offset tuple or a sequence "
                "of offset tuples."
            )

        for offset in expanded_offsets:
            if (
                not isinstance(offset, (list, tuple))
                or len(offset) != 2
                or not all(
                    isinstance(value, (int, float))
                    for value in offset
                )
            ):
                raise TypeError(
                    "Each offset must be a numeric "
                    "(delta_y, delta_z) pair."
                )

        # ---------------------------------------------------------
        # Expand rotations
        # ---------------------------------------------------------

        if rotations is None:
            expanded_rotations = [0.0] * n_units

        elif isinstance(rotations, (int, float)):
            expanded_rotations = [
                float(rotations)
                for _ in range(n_units)
            ]

        elif isinstance(rotations, (list, tuple)):
            if len(rotations) == 0:
                raise ValueError(
                    "rotations cannot be an empty sequence."
                )

            if not all(
                isinstance(rotation, (int, float))
                for rotation in rotations
            ):
                raise TypeError(
                    "Every rotation must be numeric."
                )

            if len(rotations) == 1:
                expanded_rotations = [
                    rotations[0]
                    for _ in range(n_units)
                ]

            elif len(rotations) == 2:
                expanded_rotations = [
                    rotations[i % 2]
                    for i in range(n_units)
                ]

            elif len(rotations) == n_units:
                expanded_rotations = list(rotations)

            else:
                raise ValueError(
                    "rotations must contain one rotation, "
                    "two alternating rotations, or exactly "
                    f"{n_units} rotations."
                )

        else:
            raise TypeError(
                "rotations must be numeric, a sequence of "
                "numeric rotations, or None."
            )

        # ---------------------------------------------------------
        # Construct the chain
        # ---------------------------------------------------------

        chain = cls(
            node_diameter=node_diameter
        )

        current_position = start_position

        for i in range(n_units):
            unit_type = (
                UnitType.RAILED
                if i % 2 == 0
                else UnitType.FREE
            )

            unit = RAMM_Unit.generate(
                bottom_position=current_position,
                unit_type=unit_type,
                rotation_deg=expanded_rotations[i],
                node_diameter=node_diameter
            )

            chain.units.append(unit)

            if i < required_offset_count:
                delta_y, delta_z = expanded_offsets[i]
                current_y, current_z = current_position

                current_position = (
                    current_y + delta_y,
                    current_z + delta_z
                )

        # Assign every physical and virtual node to its unit.
        for i, unit in enumerate(chain.units):
            for node in unit.all_nodes:
                node.unit_index = i

        chain.validate_chain_geometry_upon_creation()

        return chain

    def rotate(self, unit_index, pivot, degrees):
        print(
            f"Rotating unit {unit_index} around "
            f"{pivot} by {degrees} degrees..."
        )

        self.units[unit_index].transform(
            point=pivot,
            degrees=degrees
        )

    def translate(
        self,
        unit_index,
        dy=0.0,
        dz=0.0,
        verbose=True
    ):
        """
        Translate a single unit.

        Parameters
        ----------
        unit_index:
            Index of the unit to translate.

        dy:
            Horizontal translation.

        dz:
            Vertical translation.
        """
        if not 0 <= unit_index < len(self.units):
            raise IndexError(
                f"Invalid unit index: {unit_index}"
            )

        if verbose:
            print(
                f"Translating Unit {unit_index}: "
                f"dy = {dy:.6f}, dz = {dz:.6f}"
            )

        self.units[unit_index].translate(
            dy=dy,
            dz=dz
        )

    def reset(self, offset=10):
        """
        Reset the chain to its default configuration while preserving
        its node diameter.
        """
        default_chain = type(self).generate(
            n_units=len(self.units),
            start_position=(0, 0),
            offsets=(0, offset),
            rotations=0,
            node_diameter=self.node_diameter
        )

        self.units = default_chain.units