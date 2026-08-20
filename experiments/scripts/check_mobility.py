"""End-to-end check from a configured chain to active gaps and Jacobian."""

import numpy as np
import sympy as sp
from IPython.display import display

from ramms.logger import log_call
from _fixtures import (
    make_two_unit_chain,
    make_three_unit_chain,
    NODE_DIAMETER_PLOT,
    SEG_LINE_WIDTH,
    XLIM,
    TWO_UNIT_YLIM,
    THREE_UNIT_YLIM,
    TWO_UNIT_ROTATED_RIGHT_XLIM,
    TWO_UNIT_ROTATED_LEFT_XLIM,
    THREE_UNIT_ROTATED_RIGHT_XLIM,
    THREE_UNIT_ROTATED_LEFT_XLIM,
    ROTATED_THREE_UNIT_YLIM,
)
from ramms.core import RAMM_Chain
from ramms.plotting import plot_geometry
from ramms.symbolic import (
    get_candidate_gaps, 
    get_active_gap_vector,
    express_gaps_in_generalized_coordinates,
)
from ramms.mobility import get_gap_jacobian

from ramms.workspace import (
    find_three_unit_jamming_candidate,
    find_two_unit_jamming_candidate
)

@log_call
def test():
    two_unit_chain = make_two_unit_chain()
    node = two_unit_chain.get_node_by_descriptor("1B")
    print(node)
    print(vars(node))

@log_call
def get_geometric_point_positions(chain):
    point_positions = {}

    # Ordinary diamond nodes for all units
    for unit_index in range(len(chain.units)):
        for node_name in ["B", "L", "R", "T"]:
            node = chain.get_node_by_descriptor(
                f"{unit_index}{node_name}"
            )

            point_positions[node.descriptor] = (
                float(node.coordinates[0]),
                float(node.coordinates[1]),
            )

    # Rail endpoints for railed units
    for unit_index in range(0, len(chain.units), 2):
        for side in ["left", "right"]:
            rail = chain.get_rail(unit_index, side)

            for rail_node in [rail.node_1, rail.node_2]:
                point_positions[rail_node.descriptor] = (
                    float(rail_node.coordinates[0]),
                    float(rail_node.coordinates[1]),
                )

    return point_positions


def classify_gap_change(value, tol=1e-9):
    if value < -tol:
        return "PENETRATION"
    elif value > tol:
        return "BREAKING"
    else:
        return "MAINTAINED"


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

    point_positions = get_geometric_point_positions(
        two_unit_chain
    )

    generalized_gap_vector, q = (
        express_gaps_in_generalized_coordinates(
            active_gap_vector,
            point_positions,
        )
    )

    generalized_jacobian = generalized_gap_vector.jacobian(q)

    print("\nGeneralized coordinates:")
    sp.pprint(q)

    print("\nFree symbols:")
    print(generalized_gap_vector.free_symbols)

    print("\nGeneralized gap vector:")
    sp.pprint(generalized_gap_vector)

    print("\nGeneralized Jacobian:")
    sp.pprint(generalized_jacobian)

    print("\nGeneralized Jacobian dimensions:")
    print(generalized_jacobian.shape)

    # Sanity check: evaluate the generalized gap vector
    # at the candidate configuration
    candidate_state = {
        q[0]: 0,
        q[1]: 0,
    }

    print("\nGeneralized gaps at candidate state:")
    sp.pprint(
        generalized_gap_vector
        .subs(candidate_state)
        .evalf()
    )

    J_numeric = generalized_jacobian.subs(
        {
            q[0]: 0,
            q[1]: 0,
        }
    ).evalf()

    print("\nNumerical generalized Jacobian:")
    sp.pprint(J_numeric)

    print("\nRank:")
    print(J_numeric.rank())


    J_np = np.array(
        J_numeric.tolist(),
        dtype=float,
    )

    U, singular_values, Vt = np.linalg.svd(
        J_np,
        full_matrices=True,
    )

    print("\nSingular values:")
    print(singular_values)

    rank = np.linalg.matrix_rank(J_np)

    nullity = J_np.shape[1] - rank

    print("\nNumerical rank:")
    print(rank)

    print("\nNullity:")
    print(nullity)

    print("\nRight singular vectors:")
    print(Vt)

    direction_tests = {
        "+z_1": np.array([1.0, 0.0]),
        "-z_1": np.array([-1.0, 0.0]),
        "+theta_1": np.array([0.0, 1.0]),
        "-theta_1": np.array([0.0, -1.0]),
    }

    print("\nDirectional gap-change tests:")

    for name, dq in direction_tests.items():
        delta_g = J_np @ dq

        print(f"\n{name}")

        for i, value in enumerate(delta_g):
            status = classify_gap_change(value)

            print(
                f"  gap {i}: "
                f"{value:+.6f}  -> {status}"
            )

    for name, dq in direction_tests.items():
        delta_g = J_np @ dq

        admissible = np.all(delta_g >= -1e-9)

        print(
            f"{name}: "
            f"{'ADMISSIBLE' if admissible else 'BLOCKED'}"
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
            xlim=TWO_UNIT_ROTATED_RIGHT_XLIM,
            ylim=TWO_UNIT_YLIM,
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
        xlim=THREE_UNIT_ROTATED_RIGHT_XLIM,
        ylim=ROTATED_THREE_UNIT_YLIM,
        node_diameter=NODE_DIAMETER_PLOT,
        segment_line_width=SEG_LINE_WIDTH
    )

if __name__ == "__main__":
    test()
    check_two_unit_mobility(print_gaps=True)
    #check_three_unit_mobility()
