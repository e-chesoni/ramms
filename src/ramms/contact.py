"""
Contact detection and gap calculations for RAMMs.
"""

# ============================================================================
# Imports
# ============================================================================
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-whitegrid")
import math
from dataclasses import dataclass

from .core import *


# ============================================================================
# Data Structures
# ============================================================================
@dataclass
class GapResult:
    length_mm: float
    projection_coordinate: tuple[float, float]
    node_coordinate: tuple[float, float]


class Gap:
    def __init__(
        self,
        name,
        segment_1,
        node,
        orientation,
        result,
        segment_2=None
    ):
        self.name = name
        self.segment_1 = segment_1
        self.segment_2 = segment_2
        self.node = node
        self.orientation = orientation
        self.result = result

    @property
    def length_mm(self):
        return self.result.length_mm

# ============================================================================
# Internal Helpers
# ============================================================================
# helper to return nodes top to bottom
def get_nodes_top_to_bottom(segment):
    y_1, z_1 = segment.node_1.coordinates
    y_2, z_2 = segment.node_2.coordinates
    
    if z_1 > z_2:
        print(f"For segment {segment}, node 1 (appears first clockwise) is above node 2")
        return (y_1, z_1), (y_2, z_2)
    else:
        print(f"For segment {segment}, node 2 (appears first clockwise) is above node 1")
        return (y_2, z_2), (y_1, z_1)

    return


def get_nodes_left_to_right(segment, tolerance=1e-9):
    y1, _ = segment.node_1.coordinates
    y2, _ = segment.node_2.coordinates

    if math.isclose(y1, y2, abs_tol=tolerance):
        raise ValueError(
            f"Segment {segment} is vertical, so left/right ordering is undefined."
        )

    if y1 < y2:
        return segment.node_1, segment.node_2

    return segment.node_2, segment.node_1

# for segment-segment gap method
def _parallel_segments_overlap(
    A,
    B,
    C,
    D,
    tolerance=1e-6
):
    """
    Check whether two parallel finite segments overlap
    along their common tangent direction.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    D = np.asarray(D, dtype=float)

    u = B - A
    length = np.linalg.norm(u)

    if length <= tolerance:
        raise ValueError(
            "Segment A-B has zero length."
        )

    tangent = u / length

    # Project all four endpoints onto the common tangent axis.
    a_proj = np.dot(A, tangent)
    b_proj = np.dot(B, tangent)
    c_proj = np.dot(C, tangent)
    d_proj = np.dot(D, tangent)

    segment_1_min, segment_1_max = sorted(
        (a_proj, b_proj)
    )

    segment_2_min, segment_2_max = sorted(
        (c_proj, d_proj)
    )

    overlap = (
        min(segment_1_max, segment_2_max)
        - max(segment_1_min, segment_2_min)
    )

    return overlap >= -tolerance, overlap


def cross_2d(a, b): # TODO: make this private (with a leading _)
    return a[0] * b[1] - a[1] * b[0]

# ============================================================================
# Public API
# ============================================================================
def get_node_segment_distance(node, segment, orientation):
    """
    Calculate the signed gap between a node and a finite segment.

    A negative gap indicates penetration only when the node's perpendicular
    projection lies on the finite segment. Distances to segment endpoints
    remain nonnegative.
    """

    ay, az = segment.node_1.coordinates
    by, bz = segment.node_2.coordinates
    py, pz = node.coordinates

    # Segment direction: u = B - A
    uy = by - ay
    uz = bz - az

    # Vector from A to node: v = P - A
    vy = py - ay
    vz = pz - az

    u_length_squared = uy**2 + uz**2

    if math.isclose(u_length_squared, 0.0):
        raise InvalidRAMMGeometryError(
            "Cannot calculate a gap from a zero-length segment."
        )

    # Projection onto the infinite supporting line
    raw_t = (vy * uy + vz * uz) / u_length_squared

    # Projection onto the finite segment
    t = max(0.0, min(1.0, raw_t))

    qy = ay + t * uy
    qz = az + t * uz

    unsigned_gap = math.hypot(
        py - qy,
        pz - qz
    )

    orientation = orientation.lower()

    if orientation == "counterclockwise":
        expected_side = 1
    elif orientation == "clockwise":
        expected_side = -1
    else:
        raise ValueError(
            "orientation must be 'clockwise' or 'counterclockwise'"
        )

    # Only use the line-side test when the perpendicular projection
    # actually lies on the finite segment.
    projection_is_on_segment = 0.0 <= raw_t <= 1.0

    if not projection_is_on_segment:
        # The closest point is an endpoint. The side of the infinite
        # supporting line does not indicate penetration.
        signed_gap = unsigned_gap

    else:
        cross_product = uy * vz - uz * vy

        if math.isclose(cross_product, 0.0, abs_tol=1e-12):
            signed_gap = 0.0
        else:
            actual_side = 1 if cross_product > 0 else -1

            signed_gap = (
                unsigned_gap
                if actual_side == expected_side
                else -unsigned_gap
            )

    return GapResult(
        length_mm=signed_gap,
        projection_coordinate=(qy, qz),
        node_coordinate=(py, pz),
    )


def get_node_segment_gap(node, segment, orientation, verbose=False):

    gap_result = get_node_segment_distance(
        node=node,
        segment=segment,
        orientation=orientation,
    )

    gap_name = (
        f"gap_{segment.descriptor}_"
        f"{node.descriptor}"
    )

    gap = Gap(
        name=gap_name,
        segment_1=segment,
        node=node,
        orientation=orientation,
        result=gap_result,
    )

    if verbose:
        if gap.length_mm < 0:
            print(
                f"Invalid configuration\n"
                f"Gap: {gap.length_mm:.6f} mm"
            )

        elif math.isclose(
            gap.length_mm,
            0.0,
            abs_tol=1e-6
        ):
            print("Contact")

        else:
            print(
                f"Segment {segment.descriptor} and "
                f"node {node.descriptor} "
                f"are not in contact.\n"
                f"Gap: {gap.length_mm:.6f} mm"
            )

    return gap


def get_segment_segment_distance(
    bottom_unit_segment: RAMM_Strut,
    top_unit_segment: RAMM_Strut,
    tolerance=1e-6
):
    """
    Determine whether two finite segments are parallel and,
    if so, whether they are touching.

    Motion-blocking contact requires:
        1. segments are parallel
        2. supporting-line distance is approximately zero
        3. finite segment projections overlap
    """

    A = np.asarray(
        bottom_unit_segment.node_1.coordinates,
        dtype=float
    )
    B = np.asarray(
        bottom_unit_segment.node_2.coordinates,
        dtype=float
    )
    C = np.asarray(
        top_unit_segment.node_1.coordinates,
        dtype=float
    )
    D = np.asarray(
        top_unit_segment.node_2.coordinates,
        dtype=float
    )

    u = B - A
    v = D - C

    length_u = np.linalg.norm(u)
    length_v = np.linalg.norm(v)

    if length_u <= tolerance or length_v <= tolerance:
        raise ValueError(
            "Cannot evaluate a zero-length segment."
        )

    # ---------------------------------------------------------
    # 1. Are the segments parallel?
    # ---------------------------------------------------------

    parallel_measure = abs(
        cross_2d(u, v)
    ) / (length_u * length_v)

    parallel = parallel_measure <= tolerance

    if not parallel:
        return {
            "parallel": False,
            "perpendicular_distance": None,
            "overlap": None,
            "touching": False
        }

    # ---------------------------------------------------------
    # 2. How far apart are the parallel supporting lines?
    # ---------------------------------------------------------

    perpendicular_distance = abs(
        cross_2d(u, C - A)
    ) / length_u

    # ---------------------------------------------------------
    # 3. Do the finite segments overlap?
    # ---------------------------------------------------------

    overlapping, overlap = _parallel_segments_overlap(
        A,
        B,
        C,
        D,
        tolerance=tolerance
    )

    # ---------------------------------------------------------
    # 4. Are they actually touching?
    # ---------------------------------------------------------

    touching = (
        perpendicular_distance <= tolerance
        and overlapping
    )

    return {
        "parallel": parallel,
        "perpendicular_distance": perpendicular_distance,
        "overlap": overlap,
        "touching": touching
    }
