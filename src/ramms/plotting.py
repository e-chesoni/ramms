import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-whitegrid")
import matplotlib.patches as patches
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


@log_call
def overlay_cartesian_motion_cone(
    ax,
    point,
    gap_jacobian,
    point_jacobian,
    scale=4.0,
    n_directions=720,
    tol=1e-10,
):
    """
    Overlay the first-order admissible motion cone at a physical point.

    gap_jacobian:
        J_g such that J_g @ qdot >= 0

    point_jacobian:
        J_p such that pdot = J_p @ qdot

    point:
        physical [y, z] location where cone is drawn
    """

    Jg = np.asarray(gap_jacobian, dtype=float)
    Jp = np.asarray(point_jacobian, dtype=float)
    point = np.asarray(point, dtype=float)

    # Remove zero rows
    Jg = Jg[np.linalg.norm(Jg, axis=1) > tol]

    # ---------------------------------------------------------
    # Sample generalized velocity directions
    # ---------------------------------------------------------

    angles = np.linspace(0, 2*np.pi, n_directions)

    qdots = np.vstack([
        np.cos(angles),
        np.sin(angles),
    ])

    # Keep only directions satisfying all contact inequalities
    feasible = np.all(Jg @ qdots >= -tol, axis=0)

    qdots_feasible = qdots[:, feasible]

    # ---------------------------------------------------------
    # Map generalized velocity -> physical point velocity
    # ---------------------------------------------------------

    pdots = Jp @ qdots_feasible

    # Normalize only for visualization
    norms = np.linalg.norm(pdots, axis=0)
    good = norms > tol

    pdots = pdots[:, good]
    pdots /= np.linalg.norm(pdots, axis=0)

    # ---------------------------------------------------------
    # Draw rays / shaded cone
    # ---------------------------------------------------------

    y0, z0 = point

    ys = y0 + scale * pdots[0, :]
    zs = z0 + scale * pdots[1, :]

    # Shaded physical motion region
    ax.fill(
        np.concatenate([[y0], ys, [y0]]),
        np.concatenate([[z0], zs, [z0]]),
        alpha=0.2,
        zorder=20,
    )

    # Boundary / representative rays
    for idx in [0, -1]:
        ax.plot(
            [y0, ys[idx]],
            [z0, zs[idx]],
            linewidth=2,
            zorder=21,
        )

@log_call
def overlay_rotation_cone(
    ax,
    center,
    radius_inner=2.2,
    radius_outer=3.4,
    theta_start=20,
    theta_end=120,
    label=r"$+\dot{\theta}_1$ admissible",
):
    y0, z0 = center

    # Curved wedge / annular sector
    wedge = patches.Wedge(
        center=(y0, z0),
        r=radius_outer,
        theta1=theta_start,
        theta2=theta_end,
        width=radius_outer - radius_inner,
        alpha=0.2,
        zorder=29,
    )
    ax.add_patch(wedge)

    # Curved arrow along middle of sector
    radius_mid = 0.5 * (radius_inner + radius_outer)

    theta_arrow_start = theta_start + 10
    theta_arrow_end = theta_end - 10

    t_start = np.deg2rad(theta_arrow_start)
    t_end = np.deg2rad(theta_arrow_end)

    p_start = (
        y0 + radius_mid * np.cos(t_start),
        z0 + radius_mid * np.sin(t_start),
    )

    p_end = (
        y0 + radius_mid * np.cos(t_end),
        z0 + radius_mid * np.sin(t_end),
    )

    ax.annotate(
        "",
        xy=p_end,
        xytext=p_start,
        arrowprops=dict(
            arrowstyle="->",
            linewidth=2,
            connectionstyle="arc3,rad=0.35",
        ),
        zorder=31,
    )

    # Label
    theta_mid = np.deg2rad((theta_start + theta_end) / 2)

    p_label = (
        y0 + 1.25 * radius_outer * np.cos(theta_mid),
        z0 + 1.25 * radius_outer * np.sin(theta_mid),
    )

    ax.text(
        p_label[0],
        p_label[1],
        label,
        ha="center",
        va="center",
        fontsize=10,
        zorder=32,
    )

def overlay_rotation_sector(
    ax,
    center,
    radius,
    theta_start,
    theta_end,
    label=r"$+\dot{\theta}_1$ admissible",
):
    y0, z0 = center

    wedge = patches.Wedge(
        center=(y0, z0),
        r=radius,
        theta1=theta_start,
        theta2=theta_end,
        alpha=0.12,
        zorder=5,
    )
    ax.add_patch(wedge)

    # curved arrow near outer edge
    r_arrow = 0.85 * radius

    t1 = np.deg2rad(theta_start + 5)
    t2 = np.deg2rad(theta_end - 5)

    p_start = (
        y0 + r_arrow * np.cos(t1),
        z0 + r_arrow * np.sin(t1),
    )

    p_end = (
        y0 + r_arrow * np.cos(t2),
        z0 + r_arrow * np.sin(t2),
    )

    ax.annotate(
        "",
        xy=p_end,
        xytext=p_start,
        arrowprops=dict(
            arrowstyle="->",
            linewidth=2,
            connectionstyle="arc3,rad=0.3",
        ),
        zorder=20,
    )

@log_call
def overlay_contact_normal(
    ax,
    gap,
    scale=4.0,
    linewidth=3,
):
    q = np.asarray(
        gap.result.projection_coordinate,
        dtype=float
    )

    p = np.asarray(
        gap.result.node_coordinate,
        dtype=float
    )

    n = q - p
    n_hat = n / np.linalg.norm(n)

    end = q + scale * n_hat

    ax.annotate(
        "",
        xy=end,
        xytext=q,
        arrowprops=dict(
            arrowstyle="->",
            color="red",
            linewidth=linewidth,
        ),
        zorder=30,
    )

@log_call
def overlay_contact_velocity(
    ax,
    gap,
    pivot,
    qdot,
    scale=3.0,
    linewidth=3,
):
    """
    Overlay Cartesian velocity of the moving contact point Q
    produced by a specified generalized velocity qdot.

    Assumes:
        q = [z_1, theta_1]
        pivot = 1B
        Q lies on the moving unit.
    """

    # Ignore non-node-segment contacts
    if gap.node is None:
        return

    # Ignore rail contacts
    if getattr(gap.segment_1, "segment_type", None) == "rail":
        return

    q = np.asarray(
        gap.result.projection_coordinate,
        dtype=float,
    )

    pivot = np.asarray(
        pivot,
        dtype=float,
    )

    qdot = np.asarray(
        qdot,
        dtype=float,
    )

    # Position of contact point relative to pivot
    r = q - pivot

    # Point Jacobian:
    #
    # [ ydot ]   [ 0  -r_z ] [ zdot     ]
    # [ zdot ] = [ 1   r_y ] [ thetadot ]
    #
    J_q = np.array([
        [0.0, -r[1]],
        [1.0,  r[0]],
    ])

    velocity = J_q @ qdot

    speed = np.linalg.norm(velocity)

    if speed < 1e-12:
        return

    # Normalize only for visualization
    v_hat = velocity / speed

    end = q + scale * v_hat

    ax.annotate(
        "",
        xy=end,
        xytext=q,
        arrowprops=dict(
            arrowstyle="->",
            color="black",
            linewidth=linewidth,
        ),
        zorder=31,
    )

@log_call
def overlay_gap_velocity(
    ax,
    gap,
    pivot,
    qdot,
    text_offset=(6, 6),
):
    # Ignore non-node-segment contacts
    if gap.node is None:
        return

    # Ignore rail contacts
    if getattr(gap.segment_1, "segment_type", None) == "rail":
        return

    q = np.asarray(
        gap.result.projection_coordinate,
        dtype=float,
    )

    p = np.asarray(
        gap.result.node_coordinate,
        dtype=float,
    )

    pivot = np.asarray(
        pivot,
        dtype=float,
    )

    qdot = np.asarray(
        qdot,
        dtype=float,
    )

    # Signed normal direction
    n = q - p
    n_hat = n / np.linalg.norm(n)

    # Position of contact point Q relative to pivot
    r = q - pivot

    # Cartesian point Jacobian
    J_q = np.array([
        [0.0, -r[1]],
        [1.0,  r[0]],
    ])

    # Cartesian velocity of Q
    v = J_q @ qdot

    # Normal component of velocity
    gdot = np.dot(n_hat, v)

    ax.annotate(
        rf"$\dot{{g}}={gdot:.2f}$",
        xy=q,
        xytext=text_offset,
        textcoords="offset points",
        color="black",
        fontsize=11,
        zorder=32,
    )

    print(
        f"{gap.name}: "
        f"v = {v}, "
        f"gdot = {gdot:.6f}"
    )