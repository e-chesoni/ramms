"""
Drake core sanity check:
- MultibodyPlant creation
- Basic optimization solve
"""

from pydrake.multibody.plant import MultibodyPlant
from pydrake.solvers import MathematicalProgram, Solve


def test_multibody():
    plant = MultibodyPlant(time_step=0.0)
    print(f"[OK] MultibodyPlant created with {plant.num_bodies()} body/bodies")


def test_optimization():
    prog = MathematicalProgram()
    x = prog.NewContinuousVariables(1, "x")
    prog.AddQuadraticCost(x[0] ** 2)

    result = Solve(prog)

    if not result.is_success():
        raise RuntimeError("Optimization failed")

    print("[OK] Optimization solved successfully")
    print("     x* =", result.GetSolution(x))


if __name__ == "__main__":
    print("Running Drake core tests...\n")
    test_multibody()
    test_optimization()
    print("\nAll Drake core tests passed ✅")
