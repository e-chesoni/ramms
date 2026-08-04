"""Visual/manual checks for the two-unit workspace limit searches."""

from ramms.plotting import plot_geometry
from ramms.workspace import (
    find_left_max_rotation,
    find_right_max_rotation,
    find_vertical_limit,
)

from _fixtures import make_two_unit_chain


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
    show_rotation_result(chain, right_zero, "Maximum Right Rotation — Zero Thickness")

    contact_offset = 2.0 / 2 + 2.0 / 2
    right_thick = find_right_max_rotation(chain, contact_offset=contact_offset)
    show_rotation_result(chain, right_thick, "Maximum Right Rotation — Finite Features")

    left_zero = find_left_max_rotation(chain, contact_offset=0.0)
    show_rotation_result(chain, left_zero, "Maximum Left Rotation — Zero Thickness")

    contact_offset = 3.0 / 2 + 3.0 / 2
    left_thick = find_left_max_rotation(chain, contact_offset=contact_offset)
    show_rotation_result(chain, left_thick, "Maximum Left Rotation — Finite Features")


def check_vertical_limits() -> None:
    chain = make_two_unit_chain()

    for direction, label in (("up", "Maximum"), ("down", "Minimum")):
        for contact_offset in (0.0, 3.0):
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


def main() -> None:
    check_rotation_limits()
    check_vertical_limits()


if __name__ == "__main__":
    main()
