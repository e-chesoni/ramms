"""End-to-end check from a configured chain to active gaps and Jacobian."""

import sympy as sp
from IPython.display import display

from ramms.mobility import get_gap_jacobian
from ramms.symbolic import get_candidate_gaps, get_active_gap_vector

from _fixtures import make_two_unit_chain
from ramms.plotting import plot_geometry

def main() -> None:
    NODE_DIAMETER_PLOT = 1200
    SEG_LINE_WIDTH = 30
    
    chain = make_two_unit_chain()
    chain.reset() # do we neeed this if you're making a new chain?
    chain.translate(1,dz=-1.8,verbose=True)
    chain.rotate(
        unit_index=1,
        pivot=chain.units[1].bottom_node,
        degrees=38.66,
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

    print("\nClose the plot to exit out of this run.")

    plot_geometry(chain, 
                 xlim=(-15, 15), 
                 ylim=(-5, 60)
    )
    
    """
    print("\nGeneralized coordinates:")
    sp.pprint(coordinates)
    print("\nGap Jacobian:")
    sp.pprint(jacobian)
    """

if __name__ == "__main__":
    main()
