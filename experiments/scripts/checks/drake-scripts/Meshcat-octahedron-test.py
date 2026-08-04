"""
Meshcat sanity check:
- Launch Meshcat
- Display octahedron.obj via SceneGraph
- Prove translation + rotation work
"""

import time
import numpy as np
import pydrake.multibody.plant as mbp
import pydrake.systems.framework as framework
import pydrake.geometry as geometry
from pydrake.geometry import Meshcat, MeshcatVisualizer
from pydrake.math import RigidTransform, RollPitchYaw


def main():
    builder = framework.DiagramBuilder()
    plant, scene_graph = mbp.AddMultibodyPlantSceneGraph(builder, time_step=0.0)

    obj_path = "octahedron.obj"                                                     # use absolute path if needed

    # Compose rotation + translation into one transform 
    rpy = RollPitchYaw(0.0, np.pi, 0.0)                                             # 180 deg about Y (example)
    p_WG = np.array([0.0, 0.0, 1])                                                  # offset 1m (remember: we switch to meters to help drake run better, 
                                                                                    # even thouhg the chain was made in mm)
    X_WG = RigidTransform(rpy, p_WG)

    # Register visual geometry on world body
    plant.RegisterVisualGeometry(
        plant.world_body(),
        X_WG,
        geometry.Mesh(obj_path),
        "octahedron_visual",
        np.array([0.2, 0.6, 1.0, 1.0])
    )

    plant.Finalize()

    meshcat = Meshcat()

    # Clear old geometry so updated position is displayed
    meshcat.Delete()

    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    diagram.ForcedPublish(context)

    print(meshcat.web_url())
    print("Press Ctrl+C to quit.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting cleanly.")


if __name__ == "__main__":
    main()
