import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-whitegrid")
from plotly.subplots import make_subplots

from .logger import *
from .core import UnitType


@log_call
def plot_geometry(
    chain,
    ax=None,
    show=True,
    plot_title=None,
    xlim=None,
    ylim=None,
    node_diameter=None,
    segment_line_width=None,
    rail_visual_offset=0.0,
    rail_visual_shorten=0.0,
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

            # visual aid
            if segment.segment_type == "rail":

                # Direction along the rail
                dy = y2 - y1
                dz = z2 - z1
                length = np.sqrt(dy**2 + dz**2)

                uy = dy / length
                uz = dz / length

                # Unit normal perpendicular to the rail
                ny = -uz
                nz = uy

                # ---------------------------------------------------------
                # Determine which direction is outward from the unit
                # ---------------------------------------------------------

                unit_index = segment.node_1.unit_index
                unit = chain.units[unit_index]

                # Use the unit's bottom/top centerline as its approximate center
                center_y = (
                    unit.bottom_node.coordinates[0]
                    + unit.top_node.coordinates[0]
                ) / 2

                center_z = (
                    unit.bottom_node.coordinates[1]
                    + unit.top_node.coordinates[1]
                ) / 2

                rail_mid_y = (y1 + y2) / 2
                rail_mid_z = (z1 + z2) / 2

                # Vector from unit center toward rail
                outward_y = rail_mid_y - center_y
                outward_z = rail_mid_z - center_z

                # Make normal point outward
                if ny * outward_y + nz * outward_z < 0:
                    ny *= -1
                    nz *= -1

                # ---------------------------------------------------------
                # Widen rails visually
                # ---------------------------------------------------------

                y1 += rail_visual_offset * ny
                z1 += rail_visual_offset * nz

                y2 += rail_visual_offset * ny
                z2 += rail_visual_offset * nz

                # ---------------------------------------------------------
                # Shorten rails visually along their own direction
                # ---------------------------------------------------------

                y1 += rail_visual_shorten * uy
                z1 += rail_visual_shorten * uz

                y2 -= rail_visual_shorten * uy
                z2 -= rail_visual_shorten * uz

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


@log_call
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

    