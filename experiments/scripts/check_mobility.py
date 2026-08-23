"""End-to-end check from a configured chain to active gaps and Jacobian."""

import numpy as np
import sympy as sp
from IPython.display import display

from ramms.logger import log_call
from _fixtures import (
    make_two_unit_chain,
    make_three_unit_chain,
    make_four_unit_chain,
    make_five_unit_chain,
    make_chain,
    NODE_DIAMETER_PLOT,
    SEG_LINE_WIDTH,
    RAIL_VISUAL_OFFSET,
    RAIL_VISUAL_SHORTTEN,
    XLIM,
    XLIM_WIDE,
    TWO_UNIT_YLIM,
    THREE_UNIT_YLIM,
    FOUR_UNIT_YLIM,
    FIVE_UNIT_YLIM,
    TWO_UNIT_ROTATED_RIGHT_XLIM,
    TWO_UNIT_ROTATED_LEFT_XLIM,
    THREE_UNIT_ROTATED_RIGHT_XLIM,
    THREE_UNIT_ROTATED_LEFT_XLIM,
    FIVE_UNIT_ROTATED_RIGHT_XLIM,
    ROTATED_THREE_UNIT_YLIM,
)
from ramms.core import RAMM_Chain
from ramms.plotting import plot_geometry
from ramms.kinematics import (
    enforce_shared_rail_node_spacing,
    get_free_bottom_rail_top_distances,
    propagate_free_unit_rotation,
)
from ramms.symbolic import (
    get_candidate_gaps, 
    get_active_gap_vector,
    express_gaps_in_generalized_coordinates,
)
from ramms.mobility import (
    configuration_is_valid,
    get_gap_jacobian,
    get_generalized_gap_vector,
    get_generalized_jacobian,
    evaluate_candidate_gaps,
    analyze_generalized_jacobian,
    analyze_remaining_motion,
    anayze_motion_limiting_candidate,
)

from ramms.workspace import (
    find_limiting_configuration_two_unit,
    find_right_limiting_configuration_two_unit,
    find_limiting_configuration_three_unit,
    find_limiting_configuration_four_unit,
    find_right_segment_segment_contact,
)

@log_call
def check_two_unit_mobility(direction="right", print_find_candidate_results=False, print_gen_coords=False, print_gaps=False, print_active_gap_vec=False, print_active_gap_details=False) -> None:
    two_unit_chain = make_two_unit_chain()
    PIVOT = two_unit_chain.units[1].bottom_node

    print(f"Rotating chain {direction}")
    # TODO: you should make a version of this that returns rotation and tranlation
    # this rotates the chain (so we don't need the result)
    # TODO: should probably change the name then...
    _ = find_limiting_configuration_two_unit(
        two_unit_chain,
        direction=direction,
        verbose=print_find_candidate_results
    )

    tol = two_unit_chain.node_strut_contact_offset
    print(f"TOLERANCE:{tol}")
    candidate_gaps = get_candidate_gaps(two_unit_chain) # TODO: fix me: problems now for 2 unit solution following changes madde for 4 and 5 unit solutions...
    active_gap_vector = get_active_gap_vector(
        candidate_gaps,
        contact_offset=tol,
        print_gaps=print_gaps,
        print_active_gap_vector=print_active_gap_vec,
    )

    # Get point coordinates at the candidate configuration
    point_positions = two_unit_chain.get_geometric_point_positions()

    # Express active gaps in generalized coordinates
    generalized_gap_vector, q = get_generalized_gap_vector(
        two_unit_chain,
        point_positions,
        active_gap_vector
    )

    # Differentiate gaps to obtain the constraint Jacobian
    generalized_jacobian = get_generalized_jacobian(
        generalized_gap_vector,
        q
    )

    # Verify active gaps are zero at the candidate state
    evaluate_candidate_gaps(generalized_gap_vector, q)

    print("\n")

    # Compute SVD, rank, and nullity
    analysis = analyze_generalized_jacobian(
        generalized_jacobian,
        q,
    )

    # Test admissible motion in each generalized-coordinate direction
    analyze_remaining_motion(analysis, q)

    print("\nClose the plot to exit out of this run.")

    plot_geometry(
        two_unit_chain,
        plot_title="2-Unit Candidate Jamming Configuration",
        xlim=TWO_UNIT_ROTATED_RIGHT_XLIM,
        ylim=TWO_UNIT_YLIM,
        node_diameter=NODE_DIAMETER_PLOT,
        segment_line_width=SEG_LINE_WIDTH,
        rail_visual_offset=RAIL_VISUAL_OFFSET,
        rail_visual_shorten=RAIL_VISUAL_SHORTTEN,
    )  

@log_call
def check_three_unit_mobility() -> None:
    """
    End-to-end mobility check for a candidate 3-unit
    jamming configuration.
    """

    # generate chain
    three_unit_chain = make_three_unit_chain()
    CONTACT_TOLERANCE = 0.085 # TODO: dont force 3.6; try new dz,dy solver

    # find right rotated motion limited candidate config

    _ = find_limiting_configuration_three_unit(
        chain=three_unit_chain,
        direction="right",
        angle_step_deg=0.25,
        contact_tolerance=CONTACT_TOLERANCE,
    )
    """
    # Find candidate / active gaps
    candidate_gaps = get_candidate_gaps(
        three_unit_chain
    )

    active_gap_vector = get_active_gap_vector(
        candidate_gaps,
        contact_offset=(
            three_unit_chain.node_strut_contact_offset
        ),
        contact_tolerance=3.6, # 1e-6 # NOTE: we tweak this to accomidate imperfect geometry
        print_active_gap_vector=True
    )

    # Get point coordinates at the candidate configuration
    point_positions = three_unit_chain.get_geometric_point_positions()

    # Express active gaps in generalized coordinates
    generalized_gap_vector, q = get_generalized_gap_vector(
        three_unit_chain,
        point_positions,
        active_gap_vector
    )

    # Differentiate gaps to obtain the constraint Jacobian
    generalized_jacobian = get_generalized_jacobian(
        generalized_gap_vector,
        q
    )

    # Verify active gaps are zero at the candidate state
    evaluate_candidate_gaps(generalized_gap_vector, q)

    print("\n")

    # Compute SVD, rank, and nullity
    analysis = analyze_generalized_jacobian(
        generalized_jacobian,
        q,
    )

    # Test admissible motion in each generalized-coordinate direction
    analyze_remaining_motion(analysis, q)   
    """

    anayze_motion_limiting_candidate(three_unit_chain, CONTACT_TOLERANCE)

    # Plot final configuration
    print(
        "\nClose the plot to exit out of this run."
    )

    plot_geometry(
        three_unit_chain,
        plot_title="3-Unit Candidate Jamming Configuration",
        xlim=THREE_UNIT_ROTATED_RIGHT_XLIM,
        ylim=ROTATED_THREE_UNIT_YLIM,
        node_diameter=NODE_DIAMETER_PLOT,
        segment_line_width=SEG_LINE_WIDTH,
        rail_visual_offset=RAIL_VISUAL_OFFSET,
        rail_visual_shorten=RAIL_VISUAL_SHORTTEN,
    )

@log_call
def check_four_unit_mobility():
    four_unit_chain = make_four_unit_chain()
    PIVOT = four_unit_chain.units[1].bottom_node
    CONTACT_TOLERANCE = 0.085

    plot_geometry(
        four_unit_chain,
        plot_title="Before Cascading Rotation",
        xlim=XLIM,
        ylim=FOUR_UNIT_YLIM,
        node_diameter=NODE_DIAMETER_PLOT,
        segment_line_width=SEG_LINE_WIDTH,
        rail_visual_offset=RAIL_VISUAL_OFFSET,
        rail_visual_shorten=RAIL_VISUAL_SHORTTEN,
    )

    # find right rotated motion limited candidate config
    _ = find_limiting_configuration_four_unit(
        four_unit_chain,
        start_unit_index=0,
        direction="right",
        contact_tolerance=CONTACT_TOLERANCE,
        verbose=True,
    )

    anayze_motion_limiting_candidate(four_unit_chain, CONTACT_TOLERANCE)

    # Plot final configuration
    print(
        "\nClose the plot to exit."
    )

    plot_geometry(
        four_unit_chain,
        plot_title="After Second Cascading Rotation About Unit 2's Bottom Node",
        xlim=THREE_UNIT_ROTATED_RIGHT_XLIM,
        ylim=FOUR_UNIT_YLIM,
        node_diameter=NODE_DIAMETER_PLOT,
        segment_line_width=SEG_LINE_WIDTH,
        rail_visual_offset=RAIL_VISUAL_OFFSET,
        rail_visual_shorten=RAIL_VISUAL_SHORTTEN,
    )

@log_call
def check_five_unit_mobility():
    chain = make_five_unit_chain()
    PIVOT = chain.units[1].bottom_node
    CONTACT_TOLERANCE = 1e-6

    plot_geometry(
        chain,
        plot_title="Before Cascading Rotation",
        xlim=XLIM,
        ylim=FIVE_UNIT_YLIM,
        node_diameter=NODE_DIAMETER_PLOT,
        segment_line_width=SEG_LINE_WIDTH,
        rail_visual_offset=RAIL_VISUAL_OFFSET,
        rail_visual_shorten=RAIL_VISUAL_SHORTTEN,
    )

    _ = find_limiting_configuration_four_unit(
        chain=chain,
        start_unit_index=0,
        direction="right",
        contact_tolerance=CONTACT_TOLERANCE,
    )

    plot_geometry(
        chain,
        plot_title="After finding four unit subset motion limiting candidate",
        xlim=FIVE_UNIT_ROTATED_RIGHT_XLIM,
        ylim=FIVE_UNIT_YLIM,
        node_diameter=NODE_DIAMETER_PLOT,
        segment_line_width=SEG_LINE_WIDTH,
        rail_visual_offset=RAIL_VISUAL_OFFSET,
        rail_visual_shorten=RAIL_VISUAL_SHORTTEN,
    )
    
    segment_contact_result = find_right_segment_segment_contact(
        chain=chain,
        lower_unit_index=2,
        contact_tolerance=CONTACT_TOLERANCE,
        constraint_validator=lambda chain: configuration_is_valid(
            chain,
            contact_tolerance=CONTACT_TOLERANCE,
            verbose=False,
        ),
        verbose=True,
    )
    
    distances = get_free_bottom_rail_top_distances(chain)

    print(f"Distances between RAILED top node and FREE bottom node (encapsulated in rail)")
    for unit_index, distance in distances.items():
        print(
            f"Unit {unit_index}B to "
            f"Unit {unit_index - 1}T: "
            f"{distance:.6f} mm"
        )

    anayze_motion_limiting_candidate(chain, CONTACT_TOLERANCE)

    plot_geometry(
        chain,
        plot_title="After finding four unit subset motion limiting candidate",
        xlim=FIVE_UNIT_ROTATED_RIGHT_XLIM,
        ylim=FIVE_UNIT_YLIM,
        node_diameter=NODE_DIAMETER_PLOT,
        segment_line_width=SEG_LINE_WIDTH,
        rail_visual_offset=RAIL_VISUAL_OFFSET,
        rail_visual_shorten=RAIL_VISUAL_SHORTTEN,
    )
    """
    
    # find candidate gaps and active gap vector
    candidate_gaps = get_candidate_gaps(
        chain
    )
    
    active_gap_vector = get_active_gap_vector(
        candidate_gaps,
        contact_offset=(
            chain.node_strut_contact_offset
        ),
        contact_tolerance=3.6, # 1e-6 # NOTE: we tweak this to accomidate imperfect geometry
        print_active_gap_vector=True
    )
    # Get point coordinates at the candidate configuration
    point_positions = chain.get_geometric_point_positions()

    # Express active gaps in generalized coordinates
    generalized_gap_vector, q = get_generalized_gap_vector(
        chain,
        point_positions,
        active_gap_vector
    )

    # Differentiate gaps to obtain the constraint Jacobian
    generalized_jacobian = get_generalized_jacobian(
        generalized_gap_vector,
        q
    )

    # Verify active gaps are zero at the candidate state
    evaluate_candidate_gaps(generalized_gap_vector, q)

    print("\n")

    # Compute SVD, rank, and nullity
    analysis = analyze_generalized_jacobian(
        generalized_jacobian,
        q,
    )

    # Test admissible motion in each generalized-coordinate direction
    analyze_remaining_motion(analysis, q)
    """

if __name__ == "__main__":
    #check_two_unit_mobility(direction="right", print_gaps=True)
    #check_three_unit_mobility()
    check_four_unit_mobility()
    #check_five_unit_mobility()