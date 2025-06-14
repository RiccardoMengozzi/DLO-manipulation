import argparse
import numpy as np
import genesis as gs
from genesis.engine.entities import RigidEntity, MPMEntity
from genesis.engine.entities.rigid_entity import RigidLink
from numpy.typing import NDArray
from typing import Tuple
from scipy.spatial.transform import Rotation as R
import time
from tqdm import tqdm


np.set_printoptions(
    precision=4,  # number of digits after the decimal
    suppress=True,  # disable scientific notation for small numbers
    linewidth=120,  # max characters per line before wrapping
    threshold=1000,  # max total elements to print before summarizing
    edgeitems=3,  # how many items at array edges to show
    formatter={  # custom formatting per dtype
        float: "{: .4f}".format,
        int: "{: d}".format,
    },
)

NUMBER_OF_EPISODES = 3
ACTION_TIME = 2.0 # seconds (for 256 batch size)
DT = 1e-2  # simulation time step

TABLE_HEIGHT = 0.7005
HEIGHT_OFFSET = TABLE_HEIGHT
EE_OFFSET = 0.122
EE_QUAT_ROTATION = np.array([0, 0, -1, 0])





def step(
    scene: gs.Scene,
    cam,
    track: bool=False,
    link=None,
    track_offset=np.array([0.0, 0.0, 0.0]),
    gui: bool = False,
    render_interval: int = 1,
    current_step: int = 1,
    show_real_time_factor: bool = False,
):
    """
    Step the scene and update the camera.
    """
    start_time = time.time()

    render = False
    if current_step % render_interval == 0:
        render = True 

    scene.step(update_visualizer=render)
    if gui:
        if render: 
            cam.render()
    if track:
        assert link is not None, "Link must be provided to track the camera."
        ee_pos = link.get_pos().cpu().numpy() + track_offset
        cam.set_pose(pos=ee_pos, lookat=[ee_pos[0], ee_pos[1], 0.0])

    end_time = time.time()
    if show_real_time_factor:
        real_time_factor = DT / (end_time - start_time)
        print(f"Real-time factor: {real_time_factor:.4f}x")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-g", "--gui", action="store_true", default=False)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    parser.add_argument("-f", "--show_fps", action="store_true", default=False)
    parser.add_argument(
        "-n", "--n_episodes", type=int, default=NUMBER_OF_EPISODES,
        help="Number of episodes to run. Default is 3."
    )
    parser.add_argument("-l", "--episode_length", type=int, default=10)
    
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.cpu if args.cpu else gs.gpu,
            logging_level="error",
            )

    ########################## create a scene ##########################
    scene: gs.Scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
        ),
        viewer_options=gs.options.ViewerOptions(
            res=(1080, 720),
            camera_pos=(0.5, 0.0, 1.4),
            camera_lookat=(0.5, 0.0, 0.0),
            camera_fov=80,
            refresh_rate=30,
            max_FPS=240,
        ),
        vis_options=gs.options.VisOptions(
            visualize_mpm_boundary=True,
            show_world_frame=True,
        ),
        show_FPS=args.show_fps,
        show_viewer=args.vis,
    )

    cam = scene.add_camera(
        res=(1080, 720),
        pos=(0.5, -0.5, TABLE_HEIGHT + .6),
        lookat=(0.5, 0.0, TABLE_HEIGHT),
        fov=50,
        GUI=args.gui,
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )

    table = scene.add_entity(
        morph=gs.morphs.URDF(
            file="models/SimpleTable/SimpleTable.urdf",
            pos=(0.0, 0.0, 0.0),
            euler=(0, 0, 90),
            scale=1,
            fixed=True,
        ),
        material=gs.materials.Rigid(),
        surface=gs.surfaces.Default(),
    )

    franka: RigidEntity = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
            pos=(0.0, 0.0, HEIGHT_OFFSET),
        ),
        material=gs.materials.Rigid(
            friction=2.0,
            needs_coup=True,
            coup_friction=2.0,
            sdf_cell_size=0.005,
        ),
    )

    ########################## build ##########################
    scene.build()

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)

    # Optional: set control gains
    franka.set_dofs_kp(
        np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
    )
    franka.set_dofs_kv(
        np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
    )
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
    )


    observations = []
    actions = []
    episode_ends = [] # One past the last step of the episode
    steps_counter = 0
    for i in tqdm(range(args.n_episodes), desc="Running episodes"):
        circle_center = np.array([0.5 + np.random.uniform(-0.1, 0.1), np.random.uniform(-0.05, 0.1), TABLE_HEIGHT + EE_OFFSET + np.random.uniform(0.0, 0.2)])
        circle_radius = 0.15

        end_effector : RigidLink = franka.get_link("hand")

        target_quat = EE_QUAT_ROTATION
        target_pos = circle_center + np.array([1.0, 0.0, 0]) * circle_radius
        qpos = franka.inverse_kinematics(
            link=end_effector,
            pos=target_pos,
            quat=target_quat,
        )
        franka.set_dofs_position(qpos, [*motors_dof, *fingers_dof])
        step(scene, cam, link=end_effector)

        circle_sampling = 16
        scene.clear_debug_objects()
        for j in range(1, int(720//circle_sampling) + 1):
            steps_counter += 1
            target_pos = circle_center + np.array([np.cos(circle_sampling*j/360*np.pi), np.sin(circle_sampling*j/360*np.pi), 0]) * circle_radius
            scene.draw_debug_sphere(
                pos=target_pos - np.array([0, 0, EE_OFFSET]),
                radius=0.0025,
                color=(1, 0, 0, 1),
            )
            # move to pre action position
            qpos = franka.inverse_kinematics(
                link=end_effector,
                pos=target_pos,
                quat=target_quat,
            )
            qpos[-2:] = 0.0  # fingers closed
            observations.append(end_effector.get_pos().cpu().numpy())
            franka.control_dofs_position(qpos, [*motors_dof, *fingers_dof])
            action_steps = 0
            while np.linalg.norm(end_effector.get_pos().cpu().numpy() - target_pos) > 0.0075 and action_steps < int(0.2 // DT):
                step(scene, cam, link=end_effector)
                action_steps += 1
            actions.append(end_effector.get_pos().cpu().numpy())
        episode_ends.append(steps_counter)


    # Save observations and actions
    observations = np.array(observations)
    actions = np.array(actions)
    np.savez("circle_dataset.npz", observations=observations, actions=actions, episode_ends=episode_ends)
    print(f"Saved dataset with {len(observations)} observations and actions and {len(episode_ends)} episodes.")

if __name__ == "__main__":
    main()
