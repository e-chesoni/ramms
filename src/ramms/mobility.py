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
from plotly.subplots import make_subplots
from collections import Counter
from scipy.optimize import root
from IPython.display import display # for printing things in LaTex
from matplotlib.lines import Line2D
from dataclasses import dataclass
from scipy.optimize import root
from scipy.optimize import least_squares

from ramms.symbolic import get_candidate_gaps


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

    candidate_gaps = get_candidate_gaps(chain, verbose)

    for gap in candidate_gaps:

        if gap.result.length_mm < -contact_tolerance:
            return False

    return True


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