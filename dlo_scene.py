import argparse
import numpy as np
import genesis as gs
from genesis.engine.entities import RigidEntity, MPMEntity
from numpy.typing import NDArray
from typing import Tuple
from scipy.spatial.transform import Rotation as R


np.set_printoptions(
    precision=4,        # number of digits after the decimal
    suppress=True,      # disable scientific notation for small numbers
    linewidth=120,      # max characters per line before wrapping
    threshold=1000,     # max total elements to print before summarizing
    edgeitems=3,        # how many items at array edges to show
    formatter={         # custom formatting per dtype
        float: '{: .4f}'.format,
        int:   '{: d}'.format
    }
)

MPM_GRID_DENSITY = 256
SUBSTEPS = 20
TABLE_HEIGHT = 0.7005
HEIGHT_OFFSET = TABLE_HEIGHT
EE_OFFSET = 0.12  
EE_QUAT_ROTATION = np.array([0, 0, -1, 0])
ROPE_RADIUS = 0.003


def compute_pose_from_paticle_index(particles: NDArray, particle_index: int, scene : gs.Scene) -> Tuple[NDArray, NDArray]:
    vectors = np.diff(particles, axis=0)  # Compute vectors between consecutive particles
    reference_axis = np.array([0.0, 0.0, 1.0])  # Z-axis as reference
    perpendicular_vectors = -np.cross(vectors, reference_axis)  # Compute perpendicular vectors
    reference_axiss = np.tile(reference_axis, (vectors.shape[0], 1))

    print(f"Vector: {vectors[particle_index]}")
    print(f"Perpendicular Vector: {perpendicular_vectors[particle_index]}")
    print(f"Reference Axis: {reference_axiss[particle_index]}")



    # DRAW FRAMES
    scene.clear_debug_objects()
    [scene.draw_debug_arrow(particle, vector, ROPE_RADIUS, color=(1.0, 0.0, 0.0, 1.0)) for particle, vector in zip(particles, vectors)]
    [scene.draw_debug_arrow(particle, vector, ROPE_RADIUS, color=(0.0, 1.0, 0.0, 1.0)) for particle, vector in zip(particles, perpendicular_vectors)]
    [scene.draw_debug_arrow(particle, vector, ROPE_RADIUS, color=(0.0, 0.0, 1.0, 1.0)) for particle, vector in zip(particles, reference_axiss * np.linalg.norm(vectors[0]))]

    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)  # Normalize vectors
    perpendicular_vectors = perpendicular_vectors / np.linalg.norm(perpendicular_vectors, axis=1, keepdims=True)
    particle_frames = np.stack((vectors, perpendicular_vectors, reference_axiss), axis=2)


    print(f"Vector: {vectors[particle_index]}")
    print(f"Perpendicular Vector: {perpendicular_vectors[particle_index]}")
    print(f"Reference Axis: {reference_axiss[particle_index]}")

    for i, particle_frame in enumerate(particle_frames):
        # SVD della singola 3×3
        U, _, Vt = np.linalg.svd(particle_frame)
        # calcola determinante del proiettato U@Vt
        det_uv = np.linalg.det(U @ Vt)
        # costruisci D = diag(1,1,sign(det(UVt)))
        D = np.diag([1.0, 1.0, det_uv])
        # rettifica in SO(3)
        particle_frames[i] = U @ D @ Vt

    
    R_offset = gs.quat_to_R(EE_QUAT_ROTATION)
    print(f"R_offset: {R_offset}")
    print(f"particle_frames[particle_index]: {particle_frames[particle_index]}")
    print(f"quaternion: {gs.R_to_quat(particle_frames[particle_index])}")
    quaternion = gs.R_to_quat(particle_frames[particle_index] @ R_offset)
    pos = particles[particle_index] + np.array([0.0, 0.0, EE_OFFSET])
    return pos, quaternion





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.cpu if args.cpu else gs.gpu)

    ########################## create a scene ##########################

    scene : gs.Scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
            substeps=SUBSTEPS,
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
            visualize_mpm_boundary=False,
            show_world_frame=True,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(0.3, -0.2, HEIGHT_OFFSET - 0.05),
            upper_bound=(0.9, 0.6, HEIGHT_OFFSET + 0.3),
            grid_density=MPM_GRID_DENSITY,
        ),
        show_FPS=False,
        show_viewer=args.vis,
    )

    cam = scene.add_camera(
        res=(1080, 720),
        pos=(1.5, 0.0, 0.7),
        lookat=(0.5, 0.0, 0.0),
        fov=50,
        GUI=False
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
        surface=gs.surfaces.Default(
        ),
    )


    rope : MPMEntity = scene.add_entity(
        material=gs.materials.MPM.Elastic(
            E=5e4, # Determines the squishiness of the rope (very low values act as a sponge)
            nu=0.45,
            rho=2000,
            sampler="pbs"
        ),
        morph=gs.morphs.Cylinder(
            height=0.2,
            radius=ROPE_RADIUS,
            pos=(0.5, 0.0, 0.003 + HEIGHT_OFFSET),
            euler=(90, 0, 0),
        ),
        surface=gs.surfaces.Default(
            roughness=2,
            vis_mode="particle"
        ),
    )
    franka : RigidEntity = scene.add_entity(
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

    qpos = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04])
    franka.set_qpos(qpos)
    scene.step()


    while True:

        particles = rope.get_particles()
        # only get every n-th particle, otherwise vectors dont follow rope shape
        print(f"Number of particles before downsampling: {len(particles)}")
        particles = particles[1::30]
        print(f"Number of particles: {len(particles)}")
        idx = np.random.randint(2, len(particles) - 2)

        target_pos, target_quat = compute_pose_from_paticle_index(particles, idx, scene)
        print(f"Idx: {idx}, Target position: {target_pos}, Target quaternion: {target_quat}")

        end_effector = franka.get_link("hand")

        # move to pre-grasp pose
        qpos = franka.inverse_kinematics(
            link=end_effector,
            pos=target_pos,
            quat=target_quat,
        )
        qpos[-2:] = 0.04

        path = franka.plan_path(
            qpos_goal=qpos,
            num_waypoints=100,
        )

        for waypoint in path:
            franka.control_dofs_position(waypoint, [*motors_dof, *fingers_dof])


        while np.linalg.norm(franka.get_qpos().cpu().numpy() - qpos.cpu().numpy()) > 0.02:
            scene.step()


        ### GRASP ###
        qpos[-2:] = 0.0
        target_force = np.array([-0.5, -0.5])
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_force(target_force, fingers_dof)
        
        for i in range(200):
            scene.step()


        ### MOVE ###
        # target_pos[1] += 0.2
        rotation = np.random.randint(-70,70)
        translation = np.array([np.random.uniform(-0.03, 0.03), np.random.uniform(-0.03, 0.03), 0.0])
        print(f"Rotation: {rotation}, Translation: {translation}")
        target_quat = (R.from_euler("xyz", [rotation, 0, 0], degrees=True) * R.from_quat(target_quat)).as_quat()
        target_pos += translation  # Move up a bit

        qpos = franka.inverse_kinematics(
            link=end_effector,
            pos=target_pos,
            quat=target_quat,
        )
        qpos[-2:] = 0.0
        franka.control_dofs_position(qpos, [*motors_dof, *fingers_dof])

        while np.linalg.norm(franka.get_qpos().cpu().numpy() - qpos.cpu().numpy()) > 0.02:
            scene.step()

        ### RELEASE ###
        qpos[-2:] = 0.2
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_position(qpos[-2:], fingers_dof)
        
        for i in range(200):
            scene.step()


        ### LIFT ###
        target_pos[2] += 0.4
        qpos = franka.inverse_kinematics(
            link=end_effector,
            pos=target_pos,
            quat=target_quat,
        )
        qpos[-2:] = 0.0
        franka.control_dofs_position(qpos, [*motors_dof, *fingers_dof])

        while np.linalg.norm(franka.get_qpos().cpu().numpy() - qpos.cpu().numpy()) > 0.02:
            scene.step()



if __name__ == "__main__":
    main()