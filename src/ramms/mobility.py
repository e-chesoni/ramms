"""
Assess mobility for any RAMMs.
"""

# ============================================================================
# Imports
# ============================================================================
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
plt.style.use("seaborn-v0_8-whitegrid")
import plotly.graph_objects as go
import copy
import re
import pprint
import math
from enum import Enum
from collections import Counter
from scipy.optimize import root
from IPython.display import display # for printing things in LaTex
from matplotlib.lines import Line2D
from dataclasses import dataclass
from scipy.optimize import root
from scipy.optimize import least_squares

from .logger import *
from .symbolic import (
    get_candidate_gaps,
    get_active_gap_vector,
    express_gaps_in_generalized_coordinates,
)

from dataclasses import dataclass

@dataclass
class MobilityAnalysis:
    jacobian_numeric: np.ndarray
    singular_values: np.ndarray
    rank: int
    nullity: int
    right_singular_vectors: np.ndarray

def _classify_gap_change(value, tol=1e-9):
    if value < -tol:
        return "PENETRATION"
    elif value > tol:
        return "BREAKING"
    else:
        return "MAINTAINED"


@log_call
def configuration_is_valid(
    chain,
    contact_tolerance=1e-6,
    verbose=True
):
    """
    Return False if any physical member penetrates another.

    Contact is allowed.
    Penetration is not.
    """

    candidate_gaps = get_candidate_gaps(
        chain,
        verbose=verbose,
    )

    for gap in candidate_gaps:

        if gap.result.length_mm < -contact_tolerance:

            if verbose:
                print(
                    "\nINVALID CONFIGURATION:"
                    f"\n  gap: {gap.name}"
                    f"\n  value: {gap.result.length_mm:.6f} mm"
                )

            return False

    return True


@log_call
def get_gap_jacobian(active_gap_vector, q=None, print_active_gap=False, print_active_gap_details=False):
    """
    Compute the Jacobian of the active gap vector.

    Parameters
    ----------
    active_gap_vector : sympy.Matrix
        One scalar gap expression per row.

    q : sympy.Matrix, optional
        Generalized coordinates. If omitted, all symbols appearing
        in the active gap vector are used.

    Returns
    -------
    J : sympy.Matrix
        Constraint Jacobian.

    q : sympy.Matrix
        Coordinates used to compute the Jacobian.
    """
    active_gap_vector = sp.Matrix(active_gap_vector)

    if q is None:
        q = sp.Matrix(
            sorted(
                active_gap_vector.free_symbols,
                key=lambda s: s.name
            )
        )

    # ---------------------------------------------------------
    # No active constraints
    # ---------------------------------------------------------
    if active_gap_vector.rows == 0:
        J = sp.zeros(0, len(q))

        if print_active_gap:
            print("No active gaps.")
            print("\nGeneralized coordinates:")
            sp.pprint(q)
            print("\nActive gap Jacobian:")
            sp.pprint(J)
            print(f"\nJacobian dimensions: {J.shape}")

        return J, q

    J = active_gap_vector.jacobian(q)

    if print_active_gap_details:
        print(f"Active gap vector:")
        sp.pprint(active_gap_vector)
        print("\n")
        print(f"Generalized coordinates:")
        sp.pprint(q)
        print("\n")
        print(f"Active gap vector Jacobian:")
        sp.pprint(J)
        print("\n")
        print("Jacobian dimensions: ")
        print(J.shape)
    
    return active_gap_vector.jacobian(q), q


@log_call
def get_generalized_gap_vector(chain, point_positions, active_gap_vector, verbose=True):
    generalized_gap_vector, q = (
        express_gaps_in_generalized_coordinates(
            chain,
            active_gap_vector,
            point_positions,
        )
    )
    if verbose:
        print("\nGeneralized coordinates:")
        sp.pprint(q)

        print("\nSanity Check: Generalized gap vector symbols (should match gen. coords.):")
        print(generalized_gap_vector.free_symbols)

        print("\nGeneralized gap vector shape:")
        sp.pprint(generalized_gap_vector.shape)

        print("\nGeneralized gap vector:")
        sp.pprint(generalized_gap_vector)

    return generalized_gap_vector, q


@log_call
def get_generalized_jacobian(generalized_gap_vector, q, verbose=True):
    generalized_jacobian = generalized_gap_vector.jacobian(q)

    if verbose:
        print("\nGeneralized Jacobian dimensions:")
        print(generalized_jacobian.shape)

        print("\nGeneralized Jacobian:")
        sp.pprint(generalized_jacobian)

    return generalized_jacobian


@log_call
def evaluate_candidate_gaps(generalized_gap_vector, q, verbose=True):
    candidate_state = {
        coordinate: 0
        for coordinate in q
    }

    evaluated_gaps = (
        generalized_gap_vector
        .subs(candidate_state)
        .evalf()
    )

    if verbose:
        print("\nGeneralized gaps at candidate state (ideally, these are all aprox. 0):")
        sp.pprint(evaluated_gaps)

    return evaluated_gaps


@log_call
def analyze_generalized_jacobian(generalized_jacobian, q, verbose=True):
    # Candidate configuration corresponds to q = 0
    candidate_state = {
        coordinate: 0
        for coordinate in q
    }

    # Evaluate symbolic Jacobian at candidate configuration
    J_numeric = (
        generalized_jacobian
        .subs(candidate_state)
        .evalf()
    )

    # Convert to NumPy for numerical linear algebra
    J_np = np.array(
        J_numeric.tolist(),
        dtype=float,
    )

    # Singular value decomposition
    U, singular_values, Vt = np.linalg.svd(
        J_np,
        full_matrices=True,
    )

    # Rank and nullity
    rank = np.linalg.matrix_rank(J_np)
    nullity = J_np.shape[1] - rank

    if verbose:
        print("\nNumerical generalized Jacobian:")
        sp.pprint(J_numeric)

        print("\nSingular values:")
        print(singular_values)

        print("\nRank:")
        print(rank)

        print("\nNullity:")
        print(nullity)

        print("\nRight singular vectors:")
        print(Vt)

    return MobilityAnalysis(
        jacobian_numeric=J_numeric,
        singular_values=singular_values,
        rank=rank,
        nullity=nullity,
        right_singular_vectors=Vt,
    )

@log_call
# for each DoF in q, ask can I move in this direction without penetrating any active contact?
# if one gap is negative, the whole proposed direction is blocked.
def analyze_remaining_motion(analysis, q, tol=1e-9, verbose=True):
    results = {}

   # test positive and negative motions of every variable in q
    for i, coordinate in enumerate(q):
        for sign, magnitude in [("+", 1.0), ("-", -1.0)]:

            # vec of zeros corresponding dim to q
            dq = np.zeros(len(q))
            # replace one zero with motion (+/- z, y, theta) we want to test
            dq[i] = magnitude

            # what happens? (first-order change for every active gap)
            delta_g = analysis.jacobian_numeric @ dq

            # Direction is admissible if no gap penetrates
            admissible = np.all(delta_g >= -tol)

            name = f"{sign}{coordinate}"

            results[name] = {
                "dq": dq,
                "delta_g": delta_g,
                "admissible": admissible,
            }

            if verbose:
                print(f"\n{name}")

                for gap_index, value in enumerate(delta_g):
                    status = _classify_gap_change(
                        value,
                        tol=tol,
                    )

                    print(
                        f"  gap {gap_index}: "
                        f"{value:+.6f} -> {status}"
                    )

                print(
                    "  result: "
                    f"{'ADMISSIBLE' if admissible else 'BLOCKED'}"
                )

    return results

def anayze_motion_limiting_candidate(chain):
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