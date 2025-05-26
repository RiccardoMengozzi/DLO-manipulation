import argparse, torch
import numpy as np
import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.gpu)

    ########################## create a scene ##########################

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=5e-3,
            substeps=15,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3, 0.0, 1.5),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=30,
            max_FPS=60,
        ),
        show_viewer=args.vis,
        vis_options=gs.options.VisOptions(
            visualize_mpm_boundary=True,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-1.0,-1.0, -0.05),
            upper_bound=(1.0, 1.0, 1.0),
            grid_density=128,
        ),

    )

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    cube = scene.add_entity(
        material=gs.materials.MPM.Elastic(),
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.05 + 0.8),
            euler=(0, 0, 0),
            size=(0.1, 0.1, 0.1),
        ),
        surface=gs.surfaces.Default(
            color=(0.8, 0.2, 0.2),
            vis_mode="particle",    
        ),
    )

    table = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="models/SimpleTable/SimpleTable.obj",
            pos=(0.0, 0.0, 0.0),
            euler=(0, 0, 0),
            scale=1,
        ),
        material=gs.materials.Rigid(),
        surface=gs.surfaces.Default(
        ),
    )

    wind_field = scene.add_force_field(
        gs.force_fields.Wind(
            direction=(1.0, 0.0, 0.0),
            strength=1000.0,
            radius=1.0,
            center=(0.0, 0.0, 0.7),
        ),
    )
    wind_field.activate()

    scene.build()
    

    while True:
        scene.step()


if __name__ == "__main__":
    main()