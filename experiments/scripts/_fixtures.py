"""Shared chain constructors for the notebook-derived check scripts."""

from ramms.core import RAMM_Chain

# visual aids to mimic physical geometry
NODE_DIAMETER_PLOT = 1200
SEG_LINE_WIDTH = 30
RAIL_VISUAL_OFFSET = 1.2
RAIL_VISUAL_SHORTTEN = 2

# graph dimentions to match physical geometry
THREE_UNIT_YLIM = (-5, 50)
FOUR_UNIT_YLIM = (-5, 65)
FIVE_UNIT_YLIM = (-5, 80)
XLIM = (-15, 15)
XLIM_WIDE = (-25, 25)
TWO_UNIT_ROTATED_RIGHT_XLIM=(-15, 25) # TODO: will need to reverse this for left rotation
TWO_UNIT_ROTATED_LEFT_XLIM = (-25, 15)
THREE_UNIT_ROTATED_RIGHT_XLIM=(-15, 25) # TODO: will need to reverse this for left rotation
THREE_UNIT_ROTATED_LEFT_XLIM = (-25, 15)
TWO_UNIT_YLIM = (-5, 30)
ROTATED_THREE_UNIT_YLIM=(-5, 45)

def make_two_unit_chain(node_diameter: float = 2.0) -> RAMM_Chain:
    total_units = 2
    starting_pos_0B = (0,0)
    unit_1_bottom_unit_offset = (0,10)

    print(f"Making {total_units} unit chain:\n"
        f"Total units: {total_units}\n"
        f"0B starting position: {starting_pos_0B}\n"
        f"1B offset (starting position): {unit_1_bottom_unit_offset}\n"
        f"Node diameter: {node_diameter}\n"
    )
    
    return RAMM_Chain.generate(
        n_units=total_units,
        start_position=starting_pos_0B,
        offsets=unit_1_bottom_unit_offset,
        node_diameter=node_diameter,
    )


def make_three_unit_chain(node_diameter: float = 2.0, offsets=((0, 8.2), (0, 14))) -> RAMM_Chain:
    total_units = 3
    starting_pos_0B = (0,0)

    print(f"Making {total_units} unit chain:\n"
        f"Total units: {total_units}\n"
        f"0B starting position: {starting_pos_0B}\n"
        f"1B starting position: {offsets[0]}\n"
        f"2B starting position: {offsets[1]}\n"
        f"Node diameter: {node_diameter}\n"
    )

    return RAMM_Chain.generate(
        n_units=total_units,
        start_position=(0, 0),
        offsets=list(offsets),
        node_diameter=node_diameter,
    )


def make_four_unit_chain(node_diameter: float = 2.0, offsets=((0, 12), (0, 13), (0, 13))) -> RAMM_Chain:
    total_units = 4
    starting_pos_0B = (0,0)
    
    print(f"Making {total_units} unit chain:\n"
        f"Total units: {total_units}\n"
        f"0B starting position: {starting_pos_0B}\n"
        f"1B starting position: {offsets[0]}\n"
        f"2B starting position: {offsets[1]}\n"
        f"3B starting position: {offsets[2]}\n"
        f"Node diameter: {node_diameter}\n"
    )
    
    return RAMM_Chain.generate(
        n_units=total_units,
        start_position=starting_pos_0B,
        offsets=list(offsets),
        node_diameter=node_diameter,
    )

def make_five_unit_chain(node_diameter: float = 2.0, offsets=((0, 12), (0, 13), (0, 13), (0, 13))) -> RAMM_Chain:
    total_units = 5
    starting_pos_0B = (0,0)
    
    print(f"Making {total_units} unit chain:\n"
        f"Total units: {total_units}\n"
        f"0B starting position: {starting_pos_0B}\n"
        f"1B starting position: {offsets[0]}\n"
        f"2B starting position: {offsets[1]}\n"
        f"3B starting position: {offsets[2]}\n"
        f"4B starting position: {offsets[3]}\n"
        f"Node diameter: {node_diameter}\n"
    )
    
    return RAMM_Chain.generate(
        n_units=total_units,
        start_position=starting_pos_0B,
        offsets=list(offsets),
        node_diameter=node_diameter,
    )

def make_chain(
    n_units: int,
    offsets,
    node_diameter: float = 2.0,
    start_position=(0, 0),
) -> RAMM_Chain:
    """
    Create an n-unit RAMM chain from prescribed bottom-node offsets.

    Parameters
    ----------
    n_units:
        Total number of units in the chain.

    offsets:
        Positions/offsets used to place Units 1 through n-1.
        Must contain exactly n_units - 1 entries.

    node_diameter:
        Physical node diameter.

    start_position:
        Starting position of Unit 0 bottom node.
    """

    offsets = list(offsets)

    if len(offsets) != n_units - 1:
        raise ValueError(
            f"{n_units}-unit chain requires "
            f"{n_units - 1} offsets; got {len(offsets)}."
        )

    print(
        f"Making {n_units} unit chain:\n"
        f"0B starting position: {start_position}\n"
        f"Node diameter: {node_diameter}"
    )

    for unit_index, position in enumerate(
        offsets,
        start=1,
    ):
        print(
            f"{unit_index}B starting position: "
            f"{position}"
        )

    print()

    return RAMM_Chain.generate(
        n_units=n_units,
        start_position=start_position,
        offsets=offsets,
        node_diameter=node_diameter,
    )