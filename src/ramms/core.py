import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-whitegrid")
import copy
import re
import math
from enum import Enum

from .exceptions import InvalidRAMMGeometryError


class UnitType(Enum):
    RAILED = "railed"
    FREE = "free"


class RAMM_Node:
    """
    A point belonging to a RAMM unit.

    Standard node examples:
        top, right, bottom, left

    Rail-node examples:
        left rail lower, left rail upper
    """

    def __init__(
        self,
        long_name,
        coordinates,
        short_label=None,
        is_rail_node=False
    ):
        self.long_name = long_name

        # Coordinates are stored as (y, z):
        # y = horizontal position
        # z = vertical position
        self.coordinates = coordinates

        # Standard nodes default to the first letter:
        # top -> T, right -> R, etc.
        #
        # Rail rail nodes receive explicit labels such as:
        # RLL = rail left lower
        self.short_label = (
            short_label
            if short_label is not None
            else long_name[0].upper()
        )

        self.is_rail_node = is_rail_node
        self.unit_index = None

    @property
    def get_details(self):
        return f"{self.long_name} node coordinates: {self.coordinates}"

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

    Nodes should be passed in clockwise order.

    Rail nodes are directed from their lower endpoint to their
    upper endpoint.
    """

    def __init__(
        self,
        node_1,
        node_2,
        is_rail_node=False,
        segment_type="strut"
    ):
        self.node_1 = node_1
        self.node_2 = node_2

        self.is_rail_node = is_rail_node
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
        RAMM strut:
            Unit 1 bottom -> left = 1B1L

        Left rail:
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
    A four-sided RAMM unit (forms a rigid diamond).

    Nodes must be passed clockwise, beginning with the top node.

    RAILED units additionally contain two rail segments.
    FREE units do not contain rails.
    """

    def __init__(
        self,
        top_node,
        right_node,
        bottom_node,
        left_node,
        unit_type,
        node_diameter=2.0,
        strut_width=2.0,
        rail_nodes=None,
        rails=None
    ):
        self.unit_type = unit_type
        self.node_diameter = float(node_diameter)
        self.strut_width = float(strut_width)
        if self.node_diameter < 0:
            raise ValueError(
                "node_diameter must be nonnegative."
            )

        if self.strut_width < 0:
            raise ValueError(
                "strut_width must be nonnegative."
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
                is_rail_node=False,
                segment_type="diamond_strut"
            ),
            RAMM_Strut(
                right_node,
                bottom_node,
                is_rail_node=False,
                segment_type="diamond_strut"
            ),
            RAMM_Strut(
                bottom_node,
                left_node,
                is_rail_node=False,
                segment_type="diamond_strut"
            ),
            RAMM_Strut(
                left_node,
                top_node,
                is_rail_node=False,
                segment_type="diamond_strut"
            )
        ]

        # ---------------------------------------------------------
        # Rail geometry
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
                    "FREE units cannot contain rails."
                )

        if self.unit_type == UnitType.RAILED:
            if len(self.rail_nodes) != 4:
                raise InvalidRAMMGeometryError(
                    "A RAILED unit must contain four "
                    "rail endpoint nodes."
                )

            if len(self.rails) != 2:
                raise InvalidRAMMGeometryError(
                    "A RAILED unit must contain two rails."
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

        return math.hypot( # get the euclidian diststace between 2 points
            y2 - y1,
            z2 - z1
        )

    def validate_geometry(self):
        """
        Confirm that all four diamond struts have equal length.

        Rails are intentionally excluded because their length
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
    def rotate_point(point, pivot, degrees):
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

    def rotate(self, pivot, degrees): # for point, you can pass either a point or a ramm node
        """
        Rotate the entire rigid unit around a node or arbitrary point.

        Parameters
        ----------
        point:
            coordinates in tuple format: (y, z)

        degrees:
            amount to rotate in degrees
        """
        # check to make sure we got the right thing
        if isinstance(pivot, RAMM_Node):
            pivot = pivot.coordinates
        elif not isinstance(pivot, tuple):
            raise ValueError(
                "Pivot must be a RAMM_Node or coordinate tuple."
            )
        elif len(pivot) != 2:
                raise ValueError(
                    "Coordinate tuple must contain exactly two values."
                )
        
        for node in self.all_nodes:
            node.coordinates = self.rotate_point(
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
        Return the requested rail.

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
        Return the four RAMM struts in clockwise order,
        beginning with the top-right strut.

        Order:
            TR, RB, BL, LT
        """
        return list(self.struts)

    def get_strut_by_descriptor(self, descriptor):
        """
        Return the requested RAMM strut.

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
        node_diameter=2.0,
        strut_width=2.0
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

        if node_diameter < 0:
            raise ValueError(
                "node_diameter cannot be negative."
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
            rotated_y, rotated_z = cls.rotate_point(
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
            rail_half_spacing = (
                node_diameter + strut_width
            ) / 2

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
                diamond_half_height * ( 1 - rail_half_spacing / diamond_half_width)
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
                is_rail_node=True
            )

            # RLU = Rail Left Upper
            left_rail_upper = RAMM_Node(
                long_name="left rail upper",
                coordinates=create_global_position(
                    local_rail_positions["left_upper"]
                ),
                short_label="RLU",
                is_rail_node=True
            )

            # RRL = Rail Right Lower
            right_rail_lower = RAMM_Node(
                long_name="right rail lower",
                coordinates=create_global_position(
                    local_rail_positions["right_lower"]
                ),
                short_label="RRL",
                is_rail_node=True
            )

            # RRU = Rail Right Upper
            right_rail_upper = RAMM_Node(
                long_name="right rail upper",
                coordinates=create_global_position(
                    local_rail_positions["right_upper"]
                ),
                short_label="RRU",
                is_rail_node=True
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
                is_rail_node=True,
                segment_type="rail"
            )

            right_rail = RAMM_Strut(
                node_1=right_rail_lower,
                node_2=right_rail_upper,
                is_rail_node=True,
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
        node_diameter=2.0,
        strut_width=2.0
    ):
        self.units = []
        self.node_diameter = float(node_diameter)
        self.strut_width = float(strut_width)
        self.default_coordinates = None
    @property
    def node_strut_contact_offset(self):
        return (
            self.node_diameter / 2
            + self.strut_width / 2
        )

    @property
    def segment_segment_contact_offset(self):
        return self.strut_width

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
        include_rails=True # may help reduce search time in the long run, since terms are already divided
    ):
        unit_index = self._get_unit_index_from_descriptor(
            node_descriptor
        )

        unit = self.units[unit_index]

        nodes_to_search = (
            unit.all_nodes
            if include_rails
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
        include_rails=True # may help reduce search time in the long run, since terms are already divided
    ):
        unit_index = self._get_unit_index_from_descriptor(
            segment_descriptor
        )

        unit = self.units[unit_index]

        segments_to_search = (
            unit.all_segments
            if include_rails
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
        include_virtual=True # may help reduce search time in the long run, since terms are already divided
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
        include_virtual=True # may help reduce search time in the long run, since terms are already divided
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

    @classmethod
    def generate(
        cls,
        n_units,
        start_position,
        offsets,
        rotations=None,
        node_diameter=2.0,
        strut_width=2.0
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

        if node_diameter < 0:
            raise ValueError(
                "node_diameter cannot be negative."
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
            node_diameter=node_diameter,
            strut_width=strut_width
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
                node_diameter=node_diameter,
                strut_width=strut_width
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

        # ensure default coordinates cannot be overwritten
        chain.default_coordinates = copy.deepcopy(
            chain._save_coordinates()
        )

        return chain

    def rotate(self, unit_index, pivot, degrees):
        print(
            f"Rotating unit {unit_index} around "
            f"{pivot} by {degrees} degrees..."
        )

        self.units[unit_index].rotate(
            pivot=pivot,
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

    def reset(self):
        """
        Restore the chain to the configuration in which it was generated.
        """
        if self.default_coordinates is None:
            raise ValueError(
                "No default geometry has been saved."
            )

        self._restore_coordinates(
            self.default_coordinates
        )
