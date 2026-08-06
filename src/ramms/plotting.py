import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-whitegrid")
from plotly.subplots import make_subplots

from .core import UnitType


def plot_geometry(
    chain,
    ax=None,
    show=True,
    plot_title=None,
    xlim=None,
    ylim=None,
    node_diameter=None,
    segment_line_width=None
):
    """
    Plot a RAMM chain.

    Coordinates are interpreted as:
        y = horizontal axis
        z = vertical axis
    """

    if node_diameter is None:
        node_diameter = 300

    if segment_line_width is None:
        segment_line_width = 5

    segment_outline_width = segment_line_width + 2

    outline_color = "white"

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 10))

    # Draw FREE units first so RAILED rails appear on top
    units_to_draw = (
        [u for u in chain.units if u.unit_type == UnitType.FREE]
        +
        [u for u in chain.units if u.unit_type == UnitType.RAILED]
    )

    # ---------------------------------------------------------
    # Draw ALL segments first
    # ---------------------------------------------------------

    for unit in units_to_draw:

        if unit.unit_type == UnitType.RAILED:
            unit_color = "#1f3f6d"
        else:
            unit_color = "#6699cc"

        for segment in unit.all_segments:

            y1, z1 = segment.node_1.coordinates
            y2, z2 = segment.node_2.coordinates

            # Outline
            ax.plot(
                [y1, y2],
                [z1, z2],
                color=outline_color,
                linewidth=segment_outline_width,
                solid_capstyle="round",
                solid_joinstyle="round",
                zorder=1,
            )

            # Interior
            ax.plot(
                [y1, y2],
                [z1, z2],
                color=unit_color,
                linewidth=segment_line_width,
                solid_capstyle="round",
                solid_joinstyle="round",
                zorder=2,
            )

    # ---------------------------------------------------------
    # Draw ALL nodes second
    # ---------------------------------------------------------

    for unit in units_to_draw:

        if unit.unit_type == UnitType.RAILED:
            unit_color = "#1f3f6d"
        else:
            unit_color = "#6699cc"

        for node in unit.nodes:

            y, z = node.coordinates

            ax.scatter(
                y,
                z,
                s=node_diameter,
                facecolor="lightgray",
                edgecolor="lightgray",
                linewidth=4,
                zorder=10,
            )

            ax.annotate(
                node.descriptor,
                xy=(y, z),
                ha="center",
                va="center",
                fontsize=11,
                color=unit_color,
                zorder=11,
            )

    if plot_title is None:
        ax.set_title("RAMMs Configuration Geometry")
    else:
        ax.set_title(plot_title)

    ax.set_xlabel("y")
    ax.set_ylabel("z")

    ax.set_aspect("equal", adjustable="box")

    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        ax.margins(x=0.1)

    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ax.margins(y=0.1)

    if show:
        plt.show()

    return ax


def plot_gap(chain, gap):
    """
    Plot a RAMM chain and overlay a calculated gap in red.
    """

    plot_title=f"Gap Between Segment {gap.segment_1} and Node {gap.node}"

    _, ax = plt.subplots(figsize=(7, 10))

    # Plot the chain without showing it yet.
    plot_geometry(
        chain=chain,
        ax=ax,
        show=False,
        plot_title=plot_title,
    )

    qy, qz = gap.result.projection_coordinate
    py, pz = gap.result.node_coordinate

    # Draw the gap from closest point Q to node P.
    ax.plot(
        [qy, py],
        [qz, pz],
        color="red",
        linestyle="--",
        linewidth=3,
        zorder=10,
        label=f"g = {gap.length_mm:.3f}",
    )

    # Mark closest point Q.
    ax.scatter(
        qy,
        qz,
        color="red",
        marker="x",
        s=100,
        linewidths=3,
        zorder=11,
    )

    ax.annotate(
        "Q",
        xy=(qy, qz),
        xytext=(6, 6),
        textcoords="offset points",
        color="red",
        fontsize=11,
        zorder=12,
    )

    midpoint_y = (qy + py) / 2
    midpoint_z = (qz + pz) / 2

    ax.annotate(
        f"g = {gap.length_mm:.3f}",
        xy=(midpoint_y, midpoint_z),
        xytext=(6, 6),
        textcoords="offset points",
        color="red",
        fontsize=11,
        zorder=12,
    )

    ax.legend()

    plt.show()

    