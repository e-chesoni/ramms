"""Manual symbolic-gap checks derived from the notebook."""

import sympy as sp

from ramms.symbolic import (
    get_node_segment_symbolic_gap,
    get_symbolic_gap_general,
)

from _fixtures import make_two_unit_chain


def show(label: str, expression) -> None:
    print(f"\n{label}:")
    sp.pprint(expression)


def main() -> None:
    symbolic_gap = get_symbolic_gap_general()
    show("Symbolic gap A", symbolic_gap.A)
    show("Symbolic gap t", symbolic_gap.t)
    show("Symbolic gap vector", symbolic_gap.gap_vector)
    show("Symbolic gap magnitude", symbolic_gap.gap_magnitude)

    chain = make_two_unit_chain()
    node_0T = chain.units[0].top_node
    segment_1B1L = chain.get_segment_by_long_name(1, "bottom -> left")

    specific_gap = get_node_segment_symbolic_gap(
        node_0T,
        segment_1B1L,
        "clockwise",
    )
    show("Specific symbolic gap magnitude", specific_gap.gap_magnitude)

    print("ℹ️: this gap is purely symbolic, therefore we do not calculate the signed gap for it.")
if __name__ == "__main__":
    main()
