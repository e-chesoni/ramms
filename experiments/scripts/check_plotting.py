"""Visual smoke checks for chain plotting."""

from ramms.logger import log_call
from _fixtures import (
    make_two_unit_chain,
    make_three_unit_chain,
    make_four_unit_chain,
    NODE_DIAMETER_PLOT,
    SEG_LINE_WIDTH,
    XLIM,
    TWO_UNIT_YLIM,
    THREE_UNIT_YLIM,
    FOUR_UNIT_YLIM,
    TWO_UNIT_ROTATED_RIGHT_XLIM,
    TWO_UNIT_ROTATED_LEFT_XLIM,
    THREE_UNIT_ROTATED_RIGHT_XLIM,
    THREE_UNIT_ROTATED_LEFT_XLIM,
    ROTATED_THREE_UNIT_YLIM,
)
from ramms.plotting import plot_geometry

@log_call
def test_plots() -> None:
    plot_geometry(
        make_two_unit_chain(),
        plot_title="Two-Unit F.O.C.C.",
        xlim=XLIM,
        ylim=THREE_UNIT_YLIM,
        node_diameter=NODE_DIAMETER_PLOT,
        segment_line_width=SEG_LINE_WIDTH,
    )
    plot_geometry(
        make_four_unit_chain(),
        plot_title="Four-Unit F.O.C.C.",
        xlim=XLIM,
        ylim=FOUR_UNIT_YLIM,
        node_diameter=NODE_DIAMETER_PLOT,
        segment_line_width=SEG_LINE_WIDTH,
    )    


if __name__ == "__main__":
    test_plots()
