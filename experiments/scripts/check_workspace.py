"""Visual/manual checks for the two-unit workspace limit searches."""

from ramms.logger import log_call
from ramms.core import RAMM_Chain
from _fixtures import (
    make_two_unit_chain,
    make_three_unit_chain,
    NODE_DIAMETER_PLOT,
    SEG_LINE_WIDTH,
    XLIM,
    TWO_UNIT_YLIM,
    THREE_UNIT_YLIM,
    FOUR_UNIT_YLIM,
    TWO_UNIT_ROTATED_RIGHT_XLIM,
    TWO_UNIT_ROTATED_LEFT_XLIM,
    THREE_UNIT_ROTATED_RIGHT_XLIM,
    THREE_UNIT_ROTATED_LEFT_XLIM,
    ROTATED_THREE_UNIT_YLIM,
)
from ramms.plotting import plot_geometry
from ramms.workspace import (
    find_left_max_rotation,
    find_right_max_rotation,
    find_vertical_limit,
    find_three_unit_jamming_candidate
)


@log_call
def test_find_max_rotation(chain_units:int) -> None:
    if chain_units == 2:
        chain = make_two_unit_chain()
    elif chain_units == 3:
        chain = make_three_unit_chain()
    else:
        print(f"Requested units in chain must be 2 or 3.\n"
              f"Number of units requested: {chain_units}"
        )

    result = find_right_max_rotation(chain, verbose=False)
    theta_deg = result["theta_deg"]

    print(f"max rotation: {theta_deg}")


@log_call
def test_find_max_rotation_unit_chain() -> None:
    chain = make_two_unit_chain()
    result = find_right_max_rotation(chain, verbose=False)
    theta_deg = result["theta_deg"]
    print(f"max rotation: {theta_deg}")


@log_call
def show_rotation_result(chain, result, title: str) -> None:
    plot_geometry(
        chain,
        plot_title=(
            f"{title}\n"
            f"theta = {result['theta_deg']:.2f} deg, "
            f"z = {result['z_shift']:.2f} mm"
        ),
        node_diameter=NODE_DIAMETER_PLOT,
        segment_line_width=SEG_LINE_WIDTH,
    )


@log_call
def check_rotation_limits() -> None:
    chain = make_two_unit_chain()

    right_zero = find_right_max_rotation(chain, contact_offset=0.0)
    show_rotation_result(chain, right_zero, "Maximum Right Rotation — No Offset")

    rotation_contact_offset = ((chain.node_diameter/2) + (chain.strut_width/2))
    right_thick = find_right_max_rotation(chain, contact_offset=rotation_contact_offset)
    show_rotation_result(chain, right_thick, "Maximum Right Rotation with Offset to Match Physical System")

    left_zero = find_left_max_rotation(chain, contact_offset=0.0)
    show_rotation_result(chain, left_zero, "Maximum Left Rotation — No Offset")

    left_thick = find_left_max_rotation(chain, contact_offset=rotation_contact_offset)
    show_rotation_result(chain, left_thick, "Maximum Left Rotation with Offset to Match Physical System")


@log_call
def check_vertical_limits() -> None:
    chain = make_two_unit_chain()
    vertical_contact_offset = (chain.node_diameter/2)
    for direction, label in (("up", "Maximum"), ("down", "Minimum")):
        for contact_offset in (0.0, vertical_contact_offset):
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
                node_diameter=NODE_DIAMETER_PLOT,
                segment_line_width=SEG_LINE_WIDTH,
            )


@log_call
def test_three_unit_jamming() -> None:
    three_unit_chain = RAMM_Chain.generate(
        n_units=3,
        start_position=(0, 0),
        offsets=[
            (0, 8.2),
            (0, 16)
        ],
        node_diameter=2.0,
        strut_width=2.0
    )

    result = find_three_unit_jamming_candidate(
        chain=three_unit_chain,
        direction="right",
    )

    plot_geometry(
        three_unit_chain,
        plot_title="3-Unit Candidate Jamming Configuration",
        node_diameter=NODE_DIAMETER_PLOT,
        segment_line_width=SEG_LINE_WIDTH,
    )


def run_workspace_tests() -> None:
    #test_find_max_rotation(3)
    #check_rotation_limits()
    #check_vertical_limits()
    test_three_unit_jamming()


if __name__ == "__main__":
    run_workspace_tests()
