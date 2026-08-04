"""
Meshcat sanity check:
- Launch Meshcat
- Display a visible test object
- Keep running until user quits with Ctrl+C
"""

import time
from pydrake.geometry import Meshcat, Sphere, Rgba


def main():
    meshcat = Meshcat()

    # Add a visible object
    meshcat.SetObject(
        "/test_sphere",
        Sphere(0.2),
        Rgba(1.0, 0.0, 0.0, 1.0)
    )

    print("[OK] Meshcat is running")
    print("Open this URL in your browser:")
    print(meshcat.web_url())
    print("\nA red sphere should be visible at the origin.")
    print("Press Ctrl+C in this terminal to quit.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nMeshcat test stopped by user. Exiting cleanly.")


if __name__ == "__main__":
    main()
