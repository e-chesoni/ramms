"""End-to-end check from a configured chain to active gaps and Jacobian."""

import sympy as sp
from IPython.display import display

from ramms.logger import log_call
from _fixtures import (
    make_two_unit_chain,
    make_three_unit_chain
)
from ramms.core import RAMM_Chain
from ramms.plotting import plot_geometry
from ramms.symbolic import (
    get_candidate_gaps, 
    get_active_gap_vector
)
from ramms.mobility import get_gap_jacobian

from ramms.workspace import (
    find_three_unit_jamming_candidate,
    find_two_unit_jamming_candidate
)

NODE_DIAMETER_PLOT = 1200
SEG_LINE_WIDTH = 30

@log_call
def test(print_find_candidate_results=False):
    two_unit_chain = make_two_unit_chain()
    PIVOT = two_unit_chain.units[1].bottom_node

    result = find_two_unit_jamming_candidate(
        two_unit_chain,
        direction="right",
        verbose=print_find_candidate_results
    )

    plot_geometry(
        two_unit_chain,
        plot_title="2-Unit Candidate Jamming Configuration",
        xlim=(-15, 20), # TODO: will need to reverse this for left rotation
        ylim=(-5, 45),
        node_diameter=NODE_DIAMETER_PLOT,
        segment_line_width=SEG_LINE_WIDTH
    )       


@log_call
def check_two_unit_mobility(print_find_candidate_results=False, print_gen_coords=False, print_gaps=False, print_active_gap_vec=False, print_active_gap_details=False) -> None:
    two_unit_chain = make_two_unit_chain()
    PIVOT = two_unit_chain.units[1].bottom_node
    
    result = find_two_unit_jamming_candidate(
        two_unit_chain,
        direction="right",
        verbose=print_find_candidate_results
    )

    tol = two_unit_chain.node_strut_contact_offset
    print(f"TOLERANCE:{tol}")
    candidate_gaps = get_candidate_gaps(two_unit_chain)
    active_gap_vector = get_active_gap_vector(
        candidate_gaps,
        contact_offset=tol,
        print_gaps=print_gaps,
        print_active_gap_vector=print_active_gap_vec,
    )

    jacobian, coordinates = get_gap_jacobian(
        active_gap_vector,
        print_active_gap=print_active_gap_vec,
        print_active_gap_details=print_active_gap_details,
    )

    print("\nClose the plot to exit out of this run.")

    plot_geometry(
            two_unit_chain,
            plot_title="2-Unit Candidate Jamming Configuration",
            xlim=(-15, 20), # TODO: will need to reverse this for left rotation
            ylim=(-5, 45),
            node_diameter=NODE_DIAMETER_PLOT,
            segment_line_width=SEG_LINE_WIDTH
        )  


@log_call
def check_three_unit_mobility() -> None:
    """
    End-to-end mobility check for a candidate 3-unit
    jamming configuration.
    """

    # ---------------------------------------------------------
    # 1. Create 3-unit chain
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # 2. Move chain to candidate jamming configuration
    # ---------------------------------------------------------

    three_unit_chain = RAMM_Chain.generate(
        n_units=3,
        start_position=(0, 0),
        offsets=[
            (0, 8.2),
            (0, 16)
        ],
        node_diameter=2.0,
        strut_width=2.0
    )

    result = find_three_unit_jamming_candidate(
        chain=three_unit_chain,
        direction="right",
    )

    # ---------------------------------------------------------
    # 3. Find candidate / active gaps
    # ---------------------------------------------------------
    # TODO: this needs to evaluate gaps taking tolerance into account 
        # so if the node diameter is 2mm, gaps should be triggered if 
        # length < [(node_diameter/2) + (strut diameter/2)]
        # these parameters should be taken from the chain
    candidate_gaps = get_candidate_gaps(
        three_unit_chain
    )
    # TODO: tolerance needs to be [(node_diameter/2) + (strut diameter/2)]
    # tolerance is half the node diameter plus half the strut diameter
    # (to resemble the contact location IRL)

    active_gap_vector = get_active_gap_vector(
        candidate_gaps,
        contact_offset=(
            three_unit_chain.node_strut_contact_offset
        ),
        contact_tolerance=3.6, # 1e-6 # NOTE: we tweak this to accomidate imperfect geometry
        print_active_gap_vector=True
    )

    # ---------------------------------------------------------
    # 4. Calculate Jacobian
    # ---------------------------------------------------------

    jacobian, coordinates = get_gap_jacobian(
        active_gap_vector,
        print_active_gap=True,
        print_active_gap_details=False # if uncommented, the jacobian is so large, you cant really see the rest of the output (10,46)
    )

    # ---------------------------------------------------------
    # 5. Plot final configuration
    # ---------------------------------------------------------

    print(
        "\nClose the plot to exit out of this run."
    )

    plot_geometry(
        three_unit_chain,
        plot_title="3-Unit Candidate Jamming Configuration",
        xlim=(-15, 20), # TODO: will need to reverse this for left rotation
        ylim=(-5, 45),
        node_diameter=NODE_DIAMETER_PLOT,
        segment_line_width=SEG_LINE_WIDTH
    )

if __name__ == "__main__":
    #test()
    check_two_unit_mobility(print_gaps=True)
    #check_three_unit_mobility()
