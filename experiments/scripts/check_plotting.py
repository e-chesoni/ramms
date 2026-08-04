"""Visual smoke checks for chain plotting."""

from ramms.plotting import plot_geometry

from _fixtures import make_four_unit_chain, make_two_unit_chain


def main() -> None:
    plot_geometry(make_two_unit_chain(), xlim=(-15, 15), ylim=(-5, 60))
    plot_geometry(make_four_unit_chain(), xlim=(-15, 15), ylim=(-5, 60))


if __name__ == "__main__":
    main()
