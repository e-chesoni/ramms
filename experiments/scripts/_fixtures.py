"""Shared chain constructors for the notebook-derived check scripts."""

from ramms.core import RAMM_Chain


def make_two_unit_chain(node_diameter: float = 2.0) -> RAMM_Chain:
    total_units = 2
    starting_pos_0B = (0,0)
    unit_1_bottom_unit_offset = (0,10)
    node_diameter = node_diameter

    print(f"Making 2 unit chain:\n"
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
    node_diameter = node_diameter

    print(f"Making 2 unit chain:\n"
        f"Total units: {total_units}\n"
        f"0B starting position: {starting_pos_0B}\n"
        f"1B starting position: {offsets[0]}\n"
        f"2B starting position: {offsets[1]}\n"
        f"Node diameter: {node_diameter}\n"
    )    
    return RAMM_Chain.generate(
        n_units=3,
        start_position=(0, 0),
        offsets=list(offsets),
        node_diameter=node_diameter,
    )


def make_four_unit_chain(node_diameter: float = 2.0) -> RAMM_Chain:
    return RAMM_Chain.generate(
        n_units=4,
        start_position=(0, 0),
        offsets=[(0, 10), (0, 14), (0, 10)],
        node_diameter=node_diameter,
    )
