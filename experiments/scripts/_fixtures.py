"""Shared chain constructors for the notebook-derived check scripts."""

from ramms.core import RAMM_Chain


def make_two_unit_chain(node_diameter: float = 2.0) -> RAMM_Chain:
    return RAMM_Chain.generate(
        n_units=2,
        start_position=(0, 0),
        offsets=(0, 10),
        node_diameter=node_diameter,
    )


def make_three_unit_chain(
    node_diameter: float = 2.0,
    offsets=((0, 8.2), (0, 14)),
) -> RAMM_Chain:
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
