"""End-to-end check from a configured chain to active gaps and Jacobian."""

import sympy as sp

from ramms.mobility import get_gap_jacobian
from ramms.symbolic import get_candidate_gaps, get_active_gap_vector

from _fixtures import make_two_unit_chain


def main() -> None:
    chain = make_two_unit_chain()
    chain.reset()
    chain.rotate(
        unit_index=1,
        pivot=chain.units[1].bottom_node,
        degrees=38.7,
    )

    candidate_gaps = get_candidate_gaps(chain)
    active_gap_vector = get_active_gap_vector(
        candidate_gaps,
        tolerance=0.01,
        verbose=True,
    )

    jacobian, coordinates = get_gap_jacobian(
        active_gap_vector,
        verbose=True,
    )

    print("\nGeneralized coordinates:")
    sp.pprint(coordinates)
    print("\nGap Jacobian:")
    sp.pprint(jacobian)


if __name__ == "__main__":
    main()
