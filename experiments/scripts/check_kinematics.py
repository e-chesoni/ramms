"""Manual checks for rotation, translation, and cascading motion."""

import numpy as np

from ramms.plotting import plot_geometry
from ramms.kinematics import get_fractional_centerline_position, rotate_with_cascade

from _fixtures import make_three_unit_chain, make_two_unit_chain


def check_two_unit_rotation() -> None:
    chain = make_two_unit_chain()
    chain.reset()
    pivot = chain.units[1].bottom_node
    chain.rotate(unit_index=1, pivot=pivot, degrees=38.7)
    plot_geometry(chain, plot_title="Two-unit rotation")


def check_cascading_rotation() -> None:
    chain = make_three_unit_chain(offsets=((0, 8.2), (0, 14)))

    plot_geometry(
        chain,
        plot_title="Before Cascading Rotation",
        xlim=(-15, 15),
        ylim=(-5, 60),
    )

    unit_1_bottom_before = np.asarray(
        chain.units[1].bottom_node.coordinates,
        dtype=float,
    )
    unit_2_bottom_before = np.asarray(
        chain.units[2].bottom_node.coordinates,
        dtype=float,
    )

    rotate_with_cascade(
        chain,
        unit_index=1,
        pivot=chain.units[1].bottom_node,
        degrees=-25,
        verbose=True,
    )

    plot_geometry(
        chain,
        plot_title="After Cascading Rotation",
        xlim=(-20, 20),
        ylim=(-5, 60),
    )

    print("Unit 1 bottom before:", unit_1_bottom_before)
    print("Unit 2 bottom before:", unit_2_bottom_before)
    print(
        "Fractional position after motion:",
        get_fractional_centerline_position(
            chain,
            free_unit_index=1,
            railed_unit_index=2,
        ),
    )


def main() -> None:
    check_two_unit_rotation()
    check_cascading_rotation()


if __name__ == "__main__":
    main()
