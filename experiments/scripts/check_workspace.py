"""Visual/manual checks for the two-unit workspace limit searches."""

from ramms.plotting import plot_geometry
from ramms.workspace import (
    find_left_max_rotation,
    find_right_max_rotation,
    find_vertical_limit,
)

from _fixtures import make_two_unit_chain
from ramms.core import RAMM_Chain
from ramms.workspace import find_three_unit_jamming_candidate


# offsets are given in mm
NODE_DIAMETER = 2.0
STRUT_WIDTH = 2.0
VERTICAL_CONTACT_OFFSET = 2.0

NODE_RADIUS = NODE_DIAMETER / 2
STRUT_HALF_WIDTH = STRUT_WIDTH / 2


def show_rotation_result(chain, result, title: str) -> None:
    plot_geometry(
        chain,
        plot_title=(
            f"{title}\n"
            f"theta = {result['theta_deg']:.2f} deg, "
            f"z = {result['z_shift']:.2f} mm"
        ),
    )


def check_rotation_limits() -> None:
    chain = make_two_unit_chain()

    right_zero = find_right_max_rotation(chain, contact_offset=0.0)
    show_rotation_result(chain, right_zero, "Maximum Right Rotation — No Offset")

    NODE_STRUT_CONTACT_OFFSET = (
        NODE_RADIUS
        + STRUT_HALF_WIDTH
    )
    right_thick = find_right_max_rotation(chain, contact_offset=NODE_STRUT_CONTACT_OFFSET)
    show_rotation_result(chain, right_thick, "Maximum Right Rotation with Offset to Match Physical System")

    left_zero = find_left_max_rotation(chain, contact_offset=0.0)
    show_rotation_result(chain, left_zero, "Maximum Left Rotation — No Offset")

    left_thick = find_left_max_rotation(chain, contact_offset=NODE_STRUT_CONTACT_OFFSET)
    show_rotation_result(chain, left_thick, "Maximum Left Rotation with Offset to Match Physical System")


def check_vertical_limits() -> None:
    chain = make_two_unit_chain()

    for direction, label in (("up", "Maximum"), ("down", "Minimum")):
        for contact_offset in (0.0, VERTICAL_CONTACT_OFFSET):
            result = find_vertical_limit(
                chain,
                direction=direction,
                contact_offset=contact_offset,
            )
            plot_geometry(
                chain,
                plot_title=(
                    f"{label} Vertical Position — offset {contact_offset:.1f} mm\n"
                    f"z shift = {result['z_shift']:.2f} mm"
                ),
            )


def test_three_unit_jamming() -> None:
    three_unit_chain = RAMM_Chain.generate(
        n_units=3,
        start_position=(0, 0),
        offsets=[
            (0, 8.2),
            (0, 16)
        ],
        node_diameter=2.0
    )

    result = find_three_unit_jamming_candidate(
        chain=three_unit_chain,
        direction="right",
        node_strut_contact_offset=2.0,
        segment_segment_contact_offset=2.0
    )

    plot_geometry(
        three_unit_chain,
        plot_title="3-Unit Candidate Jamming Configuration"
    )


def main() -> None:
    #check_rotation_limits()
    #check_vertical_limits()
    test_three_unit_jamming()

if __name__ == "__main__":
    main()
