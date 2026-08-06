"""Manual numerical gap checks derived from the notebook."""

from ramms.contact import (
    get_node_segment_gap,
    get_segment_segment_distance,
)
from ramms.plotting import plot_gap, plot_geometry
from ramms.kinematics import rotate_with_cascade

from _fixtures import make_three_unit_chain, make_two_unit_chain


def check_node_segment_gaps() -> None:
    chain = make_two_unit_chain()
    top_unit_index = 1

    segment_1B1L = chain.get_segment_by_long_name(top_unit_index, "bottom -> left")
    segment_1T1R = chain.get_segment_by_long_name(top_unit_index, "top -> right")
    segment_1R1B = chain.get_segment_by_long_name(top_unit_index, "right -> bottom")
    segment_1L1T = chain.get_segment_by_long_name(top_unit_index, "left -> top")

    node_0T = chain.units[0].top_node
    node_0R = chain.units[0].right_node
    node_0B = chain.units[0].bottom_node
    node_0L = chain.units[0].left_node

    cases = [
        (node_0T, segment_1B1L, "clockwise"),
        (node_0T, segment_1R1B, "clockwise"),
        (node_0T, segment_1B1L, "clockwise"),
        (node_0T, segment_1T1R, "clockwise"),
        (node_0B, segment_1B1L, "counterclockwise"),
        (node_0B, segment_1R1B, "counterclockwise"),
        (node_0L, segment_1B1L, "counterclockwise"),
        (node_0R, segment_1R1B, "counterclockwise"),
    ]

    for node, segment, orientation in cases:
        gap = get_node_segment_gap(node, segment, orientation, verbose=True)
        plot_gap(chain, gap)

    # Retained from the notebook as an additional segment-segment sanity check.
    print(get_segment_segment_distance(segment_1B1L, segment_1R1B))


def check_three_unit_segment_gap(rotation_degrees: float) -> None:
    chain = make_three_unit_chain(
        node_diameter=6.0,
        offsets=((0, 8.2), (0, 16)),
    )

    plot_geometry(
        chain,
        plot_title="Before Cascading Rotation",
        xlim=(-15, 15),
        ylim=(-5, 55),
        node_diameter=1200,
        segment_line_width=20,
    )

    rotate_with_cascade(
        chain,
        unit_index=1,
        pivot=chain.units[1].bottom_node,
        degrees=rotation_degrees,
        verbose=True,
    )
    chain.translate(2, dy=0, dz=-8.75, verbose=True)

    segment_0T0R = chain.get_segment_by_descriptor("0T0R")
    segment_2B2L = chain.get_segment_by_descriptor("2B2L")
    result = get_segment_segment_distance(segment_0T0R, segment_2B2L)
    print("Segment-segment result:", result)

    print("\nStruts bottom-to-top, clockwise:")
    for strut in chain.get_struts_bottom_to_top_clockwise():
        print(strut)

    print("Pivot descriptor:", chain.get_node_by_descriptor("1T").descriptor)

    if rotation_degrees > 0:
        xlim_min = -25
        xlim_max = 15
    else:
        xlim_min = -15
        xlim_max = 25

    plot_geometry(
        chain,
        plot_title="After Rotation and Translation",
        xlim=(xlim_min, xlim_max),
        ylim=(-5, 45),
        node_diameter=1200,
        segment_line_width=30,
    )


def main() -> None:
    #check_node_segment_gaps()
    check_three_unit_segment_gap(rotation_degrees=-28.39)
    check_three_unit_segment_gap(rotation_degrees=28.39)


if __name__ == "__main__":
    main()
