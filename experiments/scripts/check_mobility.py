"""End-to-end check from a configured chain to active gaps and Jacobian."""

import numpy as np
import sympy as sp
from IPython.display import display

from ramms.logger import log_call
from _fixtures import (
    get_rotated_xlim,
    get_rotated_ylim,
    get_ylim,
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
    SIX_UNIT_YLIM,
    TWO_UNIT_ROTATED_RIGHT_XLIM,
    TWO_UNIT_ROTATED_LEFT_XLIM,
    THREE_UNIT_ROTATED_RIGHT_XLIM,
    THREE_UNIT_ROTATED_LEFT_XLIM,
    FIVE_UNIT_ROTATED_RIGHT_XLIM,
    SIX_UNIT_ROTATED_RIGHT_XLIM,
    ROTATED_THREE_UNIT_YLIM,
    ROTATED_FOUR_UNIT_YLIM,
    ROTATED_FIVE_UNIT_YLIM,
    ROTATED_SIX_UNIT_YLIM,
)
from ramms.core import RAMM_Chain, UnitType
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
    find_limiting_configuration_five_unit,
    find_limiting_configuration_n_unit,
    find_limiting_configuration_six_unit,
    find_limiting_configuration_two_unit,
    find_right_limiting_configuration_two_unit,
    find_limiting_configuration_three_unit,
    find_limiting_configuration_four_unit,
    find_right_segment_segment_contact,
)

CONTACT_TOLERANCE = 0.085

@log_call
def check_neighboring_free_node_distances(chain):
    distances = get_free_bottom_rail_top_distances(chain)

    print(f"Distances between RAILED top node and FREE bottom node (encapsulated in rail)")
    for unit_index, distance in distances.items():
        print(
            f"Unit {unit_index}B to "
            f"Unit {unit_index - 1}T: "
            f"{distance:.6f} mm"
        )

@log_call
def check_top_to_bottom_node_distances(chain):
    """
    Get the distance from each free unit's top node to the bottom node
    of the next free unit.

    Example
    -------
    Free units: 1, 3, 5, 7

    Returns:
        {
            "1T-3B": distance,
            "3T-5B": distance,
            "5T-7B": distance,
        }
    """

    free_units = [
        unit for unit in chain.units
        if unit.unit_type == UnitType.FREE
    ]

    distances = {}

    for current_unit, next_unit in zip(free_units, free_units[1:]):
        top_node = current_unit.top_node
        bottom_node = next_unit.bottom_node

        distance = np.linalg.norm(
            np.array(top_node.coordinates)
            - np.array(bottom_node.coordinates)
        ) - current_unit.node_diameter # accounts for radii of both (all nodes in a chain have the same diameter)

        node_pair = f"{top_node.descriptor}-{bottom_node.descriptor}"
        distances[node_pair] = distance

    return distances

@log_call
def check_neighboring_top_node_distances(chain):
    """
    Get the distance between the top nodes of each free unit and the
    railed unit immediately following it.

    Example
    -------
    Units: 0R, 1F, 2R, 3F, 4R, 5F, 6R

    Checks:
        1T -> 2T
        3T -> 4T
        5T -> 6T

    Returns:
        {
            "1T-2T": distance,
            "3T-4T": distance,
            "5T-6T": distance,
        }
    """

    distances = {}

    for current_unit, next_unit in zip(chain.units, chain.units[1:]):
        if current_unit.unit_type != UnitType.FREE:
            continue

        current_top = current_unit.top_node
        next_top = next_unit.top_node

        # get the center distance and subtract node radii
        distance = np.linalg.norm(
            np.array(current_top.coordinates)
            - np.array(next_top.coordinates)
        ) - (chain.node_diameter)

        node_pair = f"{current_top.descriptor}-{next_top.descriptor}"
        distances[node_pair] = distance

    return distances

@log_call
def check_two_unit_mobility(direction="right", print_find_candidate_results=False, print_gen_coords=False, print_gaps=False, print_active_gap_vec=False, print_active_gap_details=False) -> None:
    two_unit_chain = make_two_unit_chain()
    PIVOT = two_unit_chain.units[1].bottom_node

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

    _ = find_limiting_configuration_three_unit(
        chain=three_unit_chain,
        direction="right",
        angle_step_deg=0.25,
        contact_tolerance=CONTACT_TOLERANCE,
    )

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
        ylim=ROTATED_FOUR_UNIT_YLIM,
        node_diameter=NODE_DIAMETER_PLOT,
        segment_line_width=SEG_LINE_WIDTH,
        rail_visual_offset=RAIL_VISUAL_OFFSET,
        rail_visual_shorten=RAIL_VISUAL_SHORTTEN,
    )

@log_call
def check_five_unit_mobility():
    chain = make_five_unit_chain()
    PIVOT = chain.units[1].bottom_node

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

    _ = find_limiting_configuration_five_unit(
        chain,
        direction="right",
        contact_tolerance=CONTACT_TOLERANCE,
        verbose=True,
    )

    anayze_motion_limiting_candidate(chain, CONTACT_TOLERANCE)

    plot_geometry(
        chain,
        plot_title="After finding four unit subset motion limiting candidate",
        xlim=FIVE_UNIT_ROTATED_RIGHT_XLIM,
        ylim=ROTATED_FIVE_UNIT_YLIM,
        node_diameter=NODE_DIAMETER_PLOT,
        segment_line_width=SEG_LINE_WIDTH,
        rail_visual_offset=RAIL_VISUAL_OFFSET,
        rail_visual_shorten=RAIL_VISUAL_SHORTTEN,
    )

    check_neighboring_free_node_distances(chain)

@log_call
def check_six_unit_mobility():
    n = 6
    offsets = ((0, 12), (0, 13), (0, 13), (0, 13), (0, 13))
    chain = make_chain(n_units=n, offsets=offsets)
    PIVOT = chain.units[1].bottom_node

    plot_geometry(
        chain,
        plot_title=f"{n}-unit Chain Before Finding Right Rotated Contact Limiting Candidate",
        xlim=XLIM,
        ylim=SIX_UNIT_YLIM,
        node_diameter=NODE_DIAMETER_PLOT,
        segment_line_width=SEG_LINE_WIDTH,
        rail_visual_offset=RAIL_VISUAL_OFFSET,
        rail_visual_shorten=RAIL_VISUAL_SHORTTEN,
    )

    _ = find_limiting_configuration_six_unit(
        chain,
        direction="right",
        contact_tolerance=CONTACT_TOLERANCE,
        verbose=True,
    )

    anayze_motion_limiting_candidate(chain, CONTACT_TOLERANCE)

    plot_geometry(
        chain,
        plot_title=f"{n}-unit Chain After Finding Right Rotated Contact Limiting Candidate",
        xlim=SIX_UNIT_ROTATED_RIGHT_XLIM,
        ylim=ROTATED_SIX_UNIT_YLIM,
        node_diameter=NODE_DIAMETER_PLOT,
        segment_line_width=SEG_LINE_WIDTH,
        rail_visual_offset=RAIL_VISUAL_OFFSET,
        rail_visual_shorten=RAIL_VISUAL_SHORTTEN,
    )

@log_call
def check_n_unit_mobility(n_units, direction):
    chain = make_chain(n_units=n_units)
    PIVOT = chain.units[1].bottom_node
    
    y_lim = get_ylim(n_units)
    rotated_x_lim = get_rotated_xlim(n_units, direction)
    rotated_y_lim = get_rotated_ylim(n_units, direction)
    
    plot_geometry(
        chain,
        plot_title="Before Cascading Rotation",
        xlim=XLIM,
        ylim=y_lim,
        node_diameter=NODE_DIAMETER_PLOT,
        segment_line_width=SEG_LINE_WIDTH,
        rail_visual_offset=RAIL_VISUAL_OFFSET,
        rail_visual_shorten=RAIL_VISUAL_SHORTTEN,
    )

    # find right rotated motion limited candidate config
    _ = find_limiting_configuration_n_unit(
        chain,
        start_index=0,
        direction=direction,
        contact_tolerance=CONTACT_TOLERANCE,
        verbose=True,
    )

    anayze_motion_limiting_candidate(chain, CONTACT_TOLERANCE)

    # Plot final configuration
    print(
        "\nClose the plot to exit."
    )

    plot_geometry(
        chain,
        plot_title="After Second Cascading Rotation About Unit 2's Bottom Node",
        xlim=rotated_x_lim,
        ylim=rotated_y_lim,
        node_diameter=NODE_DIAMETER_PLOT,
        segment_line_width=SEG_LINE_WIDTH,
        rail_visual_offset=RAIL_VISUAL_OFFSET,
        rail_visual_shorten=RAIL_VISUAL_SHORTTEN,
    )

    neighboring_free_unit_distances = check_top_to_bottom_node_distances(chain)
    top_node_distances = check_neighboring_top_node_distances(chain)
    print(f"neighboring_free_unit_distances: {neighboring_free_unit_distances}")
    print(f"top_node_distances: {top_node_distances}")

def make_and_display_chain(n_units, direction="right"):
    #chain = make_chain(n_units=n_units)
    chain = RAMM_Chain.generate(
        n_units=n_units,
        start_position=(0, 0),
        offsets=(0,0),
        node_diameter=2.0,
    )
    y_lim = get_ylim(n_units+1)
    
    plot_geometry(
        chain,
        plot_title="Chain",
        xlim=XLIM,
        ylim=(-5,25),
        node_diameter=NODE_DIAMETER_PLOT,
        segment_line_width=SEG_LINE_WIDTH,
        rail_visual_offset=RAIL_VISUAL_OFFSET,
        rail_visual_shorten=RAIL_VISUAL_SHORTTEN,
    )       

if __name__ == "__main__":
    #chain = make_and_display_chain(n_units=1)
    check_two_unit_mobility(direction="right", print_gaps=True)
    #check_three_unit_mobility()
    #check_four_unit_mobility()
    #check_five_unit_mobility()
    #check_six_unit_mobility()
    #check_n_unit_mobility(8, "right")