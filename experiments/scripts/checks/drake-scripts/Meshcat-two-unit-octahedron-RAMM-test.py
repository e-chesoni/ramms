import time
import numpy as np

import pydrake.multibody.plant as mbp
import pydrake.systems.framework as framework
import pydrake.geometry as geometry

from pydrake.geometry import Meshcat, MeshcatVisualizer
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.tree import SpatialInertia, UnitInertia
from pydrake.multibody.plant import CoulombFriction
from pydrake.systems.analysis import Simulator  # <-- ADD THIS


def obj_bbox_center(path: str) -> np.ndarray:
    mins = np.array([np.inf, np.inf, np.inf])
    maxs = np.array([-np.inf, -np.inf, -np.inf])

    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    v = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                    mins = np.minimum(mins, v)
                    maxs = np.maximum(maxs, v)

    if not np.isfinite(mins).all():
        raise RuntimeError(f"No vertices found in OBJ: {path}")

    return 0.5 * (mins + maxs)


def main():
    builder = framework.DiagramBuilder()
    plant, scene_graph = mbp.AddMultibodyPlantSceneGraph(builder, time_step=1e-3)

    obj_path = "octahedron.obj"

    mass = 0.05
    inertia = SpatialInertia(
        mass=mass,
        p_PScm_E=np.zeros(3),
        G_SP_E=UnitInertia.SolidBox(0.05, 0.05, 0.05)
    )

    body1 = plant.AddRigidBody("octa_1", inertia)
    body2 = plant.AddRigidBody("octa_2", inertia)

    # Center mesh on body frame
    p_Gcenter_G = obj_bbox_center(obj_path)
    X_BG = RigidTransform(-p_Gcenter_G)

    # Desired *initial* poses
    X_WB1 = RigidTransform(RollPitchYaw(0.0, 0.0, 0.0), np.array([0.0, -1, 1]))
    X_WB2 = RigidTransform(RollPitchYaw(0.0, 0.0, np.pi/4), np.array([0.0, -1, 2]))

    # Bottom fixed
    plant.WeldFrames(plant.world_frame(), body1.body_frame(), X_WB1)
    # Top is free: DO NOT weld

    # Visuals
    octahedron_bottom_color = np.array([0.0078, 0.1882, 0.2784, 1.0])  # #023047
    octahedron_top_color    = np.array([0.5569, 0.7922, 0.9020, 1.0])  # #8ecae6

    plant.RegisterVisualGeometry(body1, X_BG, geometry.Mesh(obj_path), "octa1_vis", octahedron_bottom_color)
    plant.RegisterVisualGeometry(body2, X_BG, geometry.Mesh(obj_path), "octa2_vis", octahedron_top_color)

    # Collision
    friction = CoulombFriction(static_friction=0.8, dynamic_friction=0.6)
    plant.RegisterCollisionGeometry(body1, X_BG, geometry.Convex(obj_path), "octa1_col", friction)
    plant.RegisterCollisionGeometry(body2, X_BG, geometry.Convex(obj_path), "octa2_col", friction)

    plant.RegisterCollisionGeometry(plant.world_body(), RigidTransform(), geometry.HalfSpace(), "ground_col", friction)
    plant.RegisterVisualGeometry(plant.world_body(), RigidTransform(), geometry.HalfSpace(), "ground_vis",
                                 np.array([0.9, 0.9, 0.9, 1.0]))

    plant.Finalize()

    # Meshcat
    meshcat = Meshcat()
    meshcat.Delete()
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()

    # --- IMPORTANT: set initial pose of the free body (body2) ---
    plant_context = plant.GetMyContextFromRoot(context)
    plant.SetFreeBodyPose(plant_context, body2, X_WB2)

    diagram.ForcedPublish(context)

    print(meshcat.web_url())
    
    # No Sim
    print("Running without simulation... Ctrl+C to quit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting cleanly.")
    
    # Sim
    '''
    print("Simulating... Ctrl+C to quit.")

    simulator = Simulator(diagram, context)
    simulator.set_target_realtime_rate(1.0)
    simulator.Initialize()

    try:
        while True:
            simulator.AdvanceTo(simulator.get_context().get_time() + 1.0)
    except KeyboardInterrupt:
        print("\nExiting cleanly.")
    '''

    

if __name__ == "__main__":
    main()
