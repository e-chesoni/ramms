"""End-to-end check from a configured chain to active gaps and Jacobian."""

import numpy as np
import sympy as sp
from IPython.display import display

from ramms.logger import log_call
from _fixtures import (
    make_two_unit_chain,
    make_three_unit_chain,
    make_four_unit_chain,
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
    TWO_UNIT_ROTATED_RIGHT_XLIM,
    TWO_UNIT_ROTATED_LEFT_XLIM,
    THREE_UNIT_ROTATED_RIGHT_XLIM,
    THREE_UNIT_ROTATED_LEFT_XLIM,
    ROTATED_THREE_UNIT_YLIM,
)
from ramms.core import RAMM_Chain
from ramms.plotting import plot_geometry
from ramms.kinematics import (
    enforce_shared_rail_node_spacing,
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
)

from ramms.workspace import (
    find_limiting_configuration_three_unit,
    find_limiting_configuration_two_unit,
    find_right_limiting_configuration_two_unit,
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
    candidate_gaps = get_candidate_gaps(two_unit_chain)
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

    _ = find_limiting_configuration_three_unit(
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
        xlim=THREE_UNIT_ROTATED_RIGHT_XLIM,
        ylim=ROTATED_THREE_UNIT_YLIM,
        node_diameter=NODE_DIAMETER_PLOT,
        segment_line_width=SEG_LINE_WIDTH,
        rail_visual_offset=RAIL_VISUAL_OFFSET,
        rail_visual_shorten=RAIL_VISUAL_SHORTTEN,
    )

def check_four_unit_mobility():
    four_unit_chain = make_four_unit_chain()
    PIVOT = four_unit_chain.units[1].bottom_node
    CONTACT_TOLERANCE = 3.6

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


    _ = find_limiting_configuration_three_unit(
        chain=four_unit_chain,
        start_unit_index=0,
        direction="right",
        contact_tolerance=CONTACT_TOLERANCE,
    )

    print(
        "\nValid before shared-rail correction:",
        configuration_is_valid(
            four_unit_chain,
            contact_tolerance=CONTACT_TOLERANCE,
        )
    )

    correction = enforce_shared_rail_node_spacing(
        four_unit_chain,
        railed_unit_index=2,
    )

    print(
        "\nShared-rail correction:",
        correction
    )

    print(
        "\n1T:",
        four_unit_chain.units[1].top_node.coordinates
    )

    print(
        "3B:",
        four_unit_chain.units[3].bottom_node.coordinates
    )

    print(
        "\nValid after shared-rail correction:",
        configuration_is_valid(
            four_unit_chain,
            contact_tolerance=CONTACT_TOLERANCE,
        )
    )

    plot_geometry(
        four_unit_chain,
        plot_title="After First Cascading Rotation About Unit 1's Bottom Node",
        xlim=THREE_UNIT_ROTATED_RIGHT_XLIM,
        ylim=FOUR_UNIT_YLIM,
        node_diameter=NODE_DIAMETER_PLOT,
        segment_line_width=SEG_LINE_WIDTH,
        rail_visual_offset=RAIL_VISUAL_OFFSET,
        rail_visual_shorten=RAIL_VISUAL_SHORTTEN,
    )

    result = find_right_limiting_configuration_two_unit(
        chain=four_unit_chain,
        lower_unit_index=2,
        moving_unit_index=3,
        contact_offset=four_unit_chain.node_strut_contact_offset,
        contact_tolerance=3.6,
        verbose=True,
    )

    print(
        "\nUnit 2-3 limiting solution:"
        f"\n  theta = {result['theta_deg']:.6f}°"
        f"\n  z shift = {result['z_shift']:.6f} mm"
    )

    print(
        "\nValid after Unit 3 limiting rotation:",
        configuration_is_valid(
            four_unit_chain,
            contact_tolerance=3.6,
        )
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

    candidate_gaps = get_candidate_gaps(
        four_unit_chain
    )
    
    active_gap_vector = get_active_gap_vector(
        candidate_gaps,
        contact_offset=(
            four_unit_chain.node_strut_contact_offset
        ),
        contact_tolerance=3.6, # 1e-6 # NOTE: we tweak this to accomidate imperfect geometry
        print_active_gap_vector=True
    )

    # Get point coordinates at the candidate configuration
    point_positions = four_unit_chain.get_geometric_point_positions()

    # Express active gaps in generalized coordinates
    generalized_gap_vector, q = get_generalized_gap_vector(
        four_unit_chain,
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


if __name__ == "__main__":
    #check_two_unit_mobility(direction="right", print_gaps=True)
    #check_three_unit_mobility()
    check_four_unit_mobility()