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


def get_gap_jacobian(active_gap_vector, q=None, verbose=False):
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

    if q is None: # get all the variables in the active_gap_vector (these are our generalized coordinates)
        q = sp.Matrix(
            sorted(active_gap_vector.free_symbols, key=lambda s: s.name)
        )

    J = active_gap_vector.jacobian(q)
    
    if verbose:
        print(f"Active gap vector:")
        display(active_gap_vector)
        print("\n")
        print(f"Generalized coordinates:")
        display(q)
        print("\n")
        print(f"Active gap vector Jacobian:")
        display(J)
        print("\n")
        print("Jacobian dimensions: ")
        print(J.shape)
    
    return active_gap_vector.jacobian(q), q