"""Manual checks for rotation, translation, and cascading motion."""

import numpy as np

from ramms.logger import log_call
from ramms.plotting import plot_geometry
from ramms.kinematics import _get_free_node_position_along_rail, propagate_free_unit_rotation
from ramms.mobility import configuration_is_valid

from _fixtures import (
    make_two_unit_chain,
    make_three_unit_chain,
    NODE_DIAMETER_PLOT,
    SEG_LINE_WIDTH,
    RAIL_VISUAL_OFFSET,
    RAIL_VISUAL_SHORTTEN,
    XLIM,
    TWO_UNIT_YLIM,
    THREE_UNIT_YLIM,
    TWO_UNIT_ROTATED_RIGHT_XLIM,
    TWO_UNIT_ROTATED_LEFT_XLIM,
    THREE_UNIT_ROTATED_RIGHT_XLIM,
    THREE_UNIT_ROTATED_LEFT_XLIM,
    ROTATED_THREE_UNIT_YLIM,
)

@log_call
def check_two_unit_rotation() -> None:
    chain = make_two_unit_chain()
    
    ROTATE_UNIT = 1
    #PIVOT = chain.units[1].bottom_node
    PIVOT = (10,10)
    ROTATION_DEG = 38.66

    chain.reset()
    #plot_geometry(chain, plot_title="Before Rotation")
    plot_geometry(
        chain,
        plot_title="Before Rotation",
        xlim=XLIM,
        ylim=TWO_UNIT_YLIM,
        node_diameter=NODE_DIAMETER_PLOT,
        segment_line_width=SEG_LINE_WIDTH,
    )

    chain.rotate(unit_index=ROTATE_UNIT, pivot=PIVOT, degrees=ROTATION_DEG)

    plot_geometry(chain, plot_title=f"After {ROTATION_DEG} Degree Rotation about {PIVOT}")


@log_call
def check_cascading_rotation() -> None:
    DEFAULT_ROTATION_DEG = 38.66 # tested 50 deg to make sure rotation stops after rail contact occurs
    chain = make_three_unit_chain(offsets=((0, 10), (0, 14)))

    # plot three unit chain before rotating
    plot_geometry(
        chain,
        plot_title="Before Cascading Rotation",
        xlim=XLIM,
        ylim=THREE_UNIT_YLIM,
        node_diameter=NODE_DIAMETER_PLOT,
        segment_line_width=SEG_LINE_WIDTH,
        rail_visual_offset=RAIL_VISUAL_OFFSET,
        rail_visual_shorten=RAIL_VISUAL_SHORTTEN,
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
    
    PIVOT = chain.units[1].bottom_node # coordinates are usually (0, z) (in this case (0, 8.2)) -- 7.Aug.2026 
    #PIVOT = (1, 8.2) # change the pivot to just right of the node's center point (so we can get rail contact)

    propagate_free_unit_rotation(
        chain,
        unit_index=1,
        pivot=PIVOT,
        degrees=-DEFAULT_ROTATION_DEG, # make negative to rotate to the right
        constraint_validator=configuration_is_valid
    )

    plot_geometry(
        chain,
        plot_title="After Cascading Rotation",
        xlim=THREE_UNIT_ROTATED_RIGHT_XLIM,
        ylim=THREE_UNIT_YLIM,
        node_diameter=NODE_DIAMETER_PLOT,
        segment_line_width=SEG_LINE_WIDTH,
        rail_visual_offset=RAIL_VISUAL_OFFSET,
        rail_visual_shorten=RAIL_VISUAL_SHORTTEN,
    )

    print("Unit 1 bottom before:", unit_1_bottom_before)
    print("Unit 2 bottom before:", unit_2_bottom_before)
    print(
        "Fractional position after motion:",
        _get_free_node_position_along_rail(
            chain,
            free_unit_index=1,
            railed_unit_index=2,
        ),
    )
    

if __name__ == "__main__":
    #check_two_unit_rotation()
    check_cascading_rotation()
