"""Manual checks for rotation, translation, and cascading motion."""

import numpy as np

from ramms.plotting import plot_geometry
from ramms.kinematics import get_fractional_centerline_position, rotate_with_cascade
from ramms.mobility import configuration_is_valid

from _fixtures import make_three_unit_chain, make_two_unit_chain


def check_two_unit_rotation() -> None:
    chain = make_two_unit_chain()
    
    ROTATE_UNIT = 1
    #PIVOT = chain.units[1].bottom_node
    PIVOT = (10,10)
    ROTATION_DEG = 38.66

    chain.reset()
    plot_geometry(chain, plot_title="Before Rotation")

    chain.rotate(unit_index=ROTATE_UNIT, pivot=PIVOT, degrees=ROTATION_DEG)

    plot_geometry(chain, plot_title=f"After {ROTATION_DEG} Degree Rotation about {PIVOT}")


def check_cascading_rotation() -> None:
    DEFAULT_ROTATION_DEG = 38.66 # tested 50 deg to make sure rotation stops after rail contact occurs
    chain = make_three_unit_chain(offsets=((0, 8.2), (0, 14)))

    # plot three unit chain before rotating
    plot_geometry(
        chain,
        plot_title="Before Cascading Rotation",
        xlim=(-15, 15),
        ylim=(-5, 60),
    )

    # get bottom node coordinates to print later
    unit_1_bottom_before = np.asarray(
        chain.units[1].bottom_node.coordinates,
        dtype=float,
    )
    unit_2_bottom_before = np.asarray(
        chain.units[2].bottom_node.coordinates,
        dtype=float,
    )

    # Set test pivot to rotate the top unit about (this can be an arbitrary point--
    # motion will be halted if rail contact is detected)
    # TODO: consider incoporating sliding along the rail depending on the direction of the force
    # ...like after quals or something...
    
    #PIVOT = chain.units[1].bottom_node # coordinates are usually (0, z) (in this case (0, 8.2)) -- 7.Aug.2026 
    PIVOT = (1, 8.2) # change the pivot to just right of the node's center point (so we can get rail contact)

    rotate_with_cascade(
        chain,
        unit_index=1,
        pivot=PIVOT,
        degrees=-DEFAULT_ROTATION_DEG, # make negative to rotate to the right
        constraint_validator=configuration_is_valid
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
    #check_two_unit_rotation()
    check_cascading_rotation()


if __name__ == "__main__":
    main()
