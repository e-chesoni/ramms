"""End-to-end check from a configured chain to active gaps and Jacobian."""

import sympy as sp
from IPython.display import display

from _fixtures import (
    make_two_unit_chain,
    make_three_unit_chain
)
from ramms.plotting import plot_geometry
from ramms.symbolic import (
    get_candidate_gaps, 
    get_active_gap_vector
)
from ramms.mobility import get_gap_jacobian

from ramms.workspace import (
    find_three_unit_jamming_candidate,
)

def find_active_gaps_2unit_chain() -> None:
    NODE_DIAMETER_PLOT = 1200
    SEG_LINE_WIDTH = 30
    ROTATION_DEG = 38.66 # 38.66 = max; change to 50 and uncomment below to test finding rail contact

    chain = make_two_unit_chain()
    PIVOT = chain.units[1].bottom_node
    #PIVOT = (5,10) # uncomment to test finding rail contact
    print(f"Rotation pivot coordinates: {PIVOT}")

    #chain.reset() # do we neeed this if you're making a new chain?
    chain.translate(1,dz=-1.8,verbose=True)
    chain.rotate(
        unit_index=1,
        pivot=PIVOT,
        degrees=ROTATION_DEG,
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

def check_three_unit_mobility() -> None:
    """
    End-to-end mobility check for a candidate 3-unit
    jamming configuration.
    """

    NODE_DIAMETER_PLOT = 1200
    SEG_LINE_WIDTH = 30

    # ---------------------------------------------------------
    # 1. Create 3-unit chain
    # ---------------------------------------------------------

    chain = make_three_unit_chain(
        offsets=(
            (0, 8.2),
            (0, 14)
        )
    )

    # ---------------------------------------------------------
    # 2. Move chain to candidate jamming configuration
    # ---------------------------------------------------------

    result = find_three_unit_jamming_candidate(
        chain=chain,
        direction="right",
        node_strut_contact_offset=2.0,
        segment_segment_contact_offset=2.0,
        verbose=True
    )

    # ---------------------------------------------------------
    # 3. Find candidate / active gaps
    # ---------------------------------------------------------

    candidate_gaps = get_candidate_gaps(
        chain
    )

    active_gap_vector = get_active_gap_vector(
        candidate_gaps,
        tolerance=0.01,
        verbose=True
    )

    # ---------------------------------------------------------
    # 4. Calculate Jacobian
    # ---------------------------------------------------------

    jacobian, coordinates = get_gap_jacobian(
        active_gap_vector,
        verbose=True
    )

    # ---------------------------------------------------------
    # 5. Plot final configuration
    # ---------------------------------------------------------

    print(
        "\nClose the plot to exit out of this run."
    )

    plot_geometry(
        chain,
        plot_title="3-Unit Candidate Jamming Configuration",
        xlim=(-15, 15),
        ylim=(-5, 45),
        node_diameter=NODE_DIAMETER_PLOT,
        segment_line_width=SEG_LINE_WIDTH
    )

if __name__ == "__main__":
    #find_active_gaps_2unit_chain()
    check_three_unit_mobility()
