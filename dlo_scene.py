import genesis as gs
import numpy as np
import argparse, sys

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.cpu if args.cpu else gs.gpu)

    ########################## create a scene ##########################
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
        ),
        mpm_options=gs.options.MPMOptions(
                    lower_bound=(-0.5, -0.5, 0.7),
                    upper_bound=(0.5, 0.5, 1.5),
                    grid_density=128,
                ),
        rigid_options=gs.options.RigidOptions(
            box_box_detection=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            res=(1080, 720),
            refresh_rate=30,
            camera_pos=(0.7, 0.0, 1.5),
            camera_lookat=(0.3, 0.0, 0.75),
            camera_up=(0.0, 0.0, 1.0),
            camera_fov=100,
        ),
        vis_options=gs.options.VisOptions(
            visualize_mpm_boundary=True,
            
        ),
        show_viewer=args.vis,
        show_FPS=False,
    )
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    table = scene.add_entity(
        morph=gs.morphs.MJCF(
            file="models/table/table.xml",
            pos=(0.0, 0.0, 0.0),
            euler=(0, 0, 0),
        ),

    )
    rope = scene.add_entity(
        material=gs.materials.MPM.Elastic(
            E   = 3.0e3,          # Young’s modulus (Pa)
            model="neohooken",
        ),
        morph=gs.morphs.Cylinder(
            height=0.4,
            radius=0.005,
            pos=(0.0, 0.0, 0.77),
            euler=(90, 90, 0),
        ),
        surface=gs.surfaces.Default(
            vis_mode="visual",
        ),
    )
    franka_pos = (-0.4, 0.0, 0.76)
    franka = scene.add_entity(
        morph=gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
            pos=franka_pos,
            euler=(0, 0, 0),
        ),
        material=gs.materials.Rigid(gravity_compensation=1.0),
    )

    ########################## build ##########################
    scene.build()

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    qpos = np.array([0, -0.2, 0, -0.2, 0, 1.571, 0.785, 0.0, 0.0])
    franka.set_qpos(qpos)
    scene.step()

    ########################## run the scene ##########################
    sim_time = 1 #sec
    horizon = int(sim_time / scene.sim_options.dt)
    for i in range(horizon):
        # compute current time in seconds
        t = (i / horizon) * sim_time
        # print with carriage return, formatted to 2 decimal places
        sys.stdout.write(f"\rWaiting [Time elapsed: {t:.2f}s]")
        sys.stdout.flush()
        scene.step()


    end_effector = franka.get_link("hand")
    target_pos = np.array([0.0, 0.0, 0.4 + 0.76])
    target_quat = np.array([0, 1, 0, 0])

    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=target_pos,
        quat=target_quat,
    )

    franka.control_dofs_position(qpos[:-2], motors_dof)

    # hold
    sim_time = 1 #sec
    horizon = int(sim_time / scene.sim_options.dt)
    for i in range(horizon):
        # compute current time in seconds
        t = (i / horizon) * sim_time
        # print with carriage return, formatted to 2 decimal places
        sys.stdout.write(f"\rReaching above target [Time elapsed: {t:.2f}s]")
        sys.stdout.flush()
        scene.step()

    target_pos = np.array([0.0, 0.0, 0.135 + 0.76])
    target_quat = np.array([0, 1, 0, 0])

    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=target_pos - franka_pos,
        quat=target_quat,
    )

    franka.control_dofs_position(qpos[:-2], motors_dof)

    # hold
    sim_time = 1 #sec
    horizon = int(sim_time / scene.sim_options.dt)
    for i in range(horizon):
        # compute current time in seconds
        t = (i / horizon) * sim_time
        # print with carriage return, formatted to 2 decimal places
        sys.stdout.write(f"\rReaching target [Time elapsed: {t:.2f}s]")
        sys.stdout.flush()
        scene.step()


    # grasp
    finder_pos = -0.0
    sim_time = 1 #sec
    horizon = int(sim_time / scene.sim_options.dt)
    for i in range(horizon):
        # compute current time in seconds
        t = (i / horizon) * sim_time
        # print with carriage return, formatted to 2 decimal places
        sys.stdout.write(f"\rGrasping [Time elapsed: {t:.2f}s]")
        sys.stdout.flush()
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_position(np.array([finder_pos, finder_pos]), fingers_dof)
        scene.step()

    # lift
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.3]),
        quat=np.array([0, 1, 0, 0]),
    )
    sim_time = 1 #sec
    horizon = int(sim_time / scene.sim_options.dt)
    for i in range(horizon):
        # compute current time in seconds
        t = (i / horizon) * sim_time
        # print with carriage return, formatted to 2 decimal places
        sys.stdout.write(f"\rLifting [Time elapsed: {t:.2f}s]")
        sys.stdout.flush()
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_position(np.array([finder_pos, finder_pos]), fingers_dof)
        scene.step()


if __name__ == "__main__":
    main()