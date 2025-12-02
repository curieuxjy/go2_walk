"""
Go2 Inspection code based on Joint Monkey
------------------------------------------------------------------------------------
- Animates degree-of-freedom ranges for a given asset.
- Demonstrates usage of DOF properties and states.
- Demonstrates line drawing utilities to visualize DOF frames (origin and axis).
------------------------------------------------------------------------------------
- egocentric camera
- lidar ray visualization
- terrain
"""

import math
import numpy as np
import random
from isaacgym import gymapi, gymutil
from isaacgym.terrain_utils import *
import matplotlib.pyplot as plt
from isaacgym import gymtorch
import time
import os
from PIL import Image as im

# Resources
RESOURCE_ROOT = "../dreamwaq/legged_gym/resources"
BASIC_ROOT = "../gym_basic"
# Simulation Constants
DT = 1.0 / 60.0

# Environment Bounds
ENV_LOWER = gymapi.Vec3(-2.0, -2.0, 0.0)
ENV_UPPER = gymapi.Vec3(2.0, 2.0, 4.0)

# Robot Starting Position
ROBOT_POS = gymapi.Vec3(0.0, 0.0, 0.34)

# Default Joint Angles
DEFAULT_JOINT_ANGLES = {
    "FL_hip_joint": 0.1,
    "FL_thigh_joint": 0.8,
    "FL_calf_joint": -1.5,
    "FR_hip_joint": -0.1,
    "FR_thigh_joint": 0.8,
    "FR_calf_joint": -1.5,
    "RL_hip_joint": 0.1,
    "RL_thigh_joint": 1.0,
    "RL_calf_joint": -1.5,
    "RR_hip_joint": -0.1,
    "RR_thigh_joint": 1.0,
    "RR_calf_joint": -1.5,
}
DEFAULTS = list(DEFAULT_JOINT_ANGLES.values())

# Viewer Configuration
VIEWER_POS = gymapi.Vec3(1.0, 1.0, 1.0)
TARGET_POS = gymapi.Vec3(0, 0, 0)

# Terrain variables
TERRAIN_WIDTH = 600  # unit
TERRAIN_LENGTH = 600  # unit
TERRAIN_TRANS_X = -2  # [m]
TERRAIN_TRANS_Y = -2  # [m]
TERRAIN_TRANS_Z = 1  # [m]
STEP_WIDTH = 0.30  # [m]
STEP_HEIGHT = -0.17  # [m]

# smaller unit
HORIZONTAL_SCALE = 0.01  # [m] -> think in cm units
VERTICAL_SCALE = 0.01  # [m] -> UNIT

# joint animation states
ANIM_SEEK_LOWER = 1
ANIM_SEEK_UPPER = 2
ANIM_SEEK_DEFAULT = 3
ANIM_FINISHED = 4

# marks for camera
AXES_GEOM = gymutil.AxesGeometry(0.5)
# a wireframe sphere
sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
sphere_pose = gymapi.Transform(r=sphere_rot)
SPHERE_GEOM = gymutil.WireframeSphereGeometry(
    0.3, 12, 12, sphere_pose, color=(1, 1, 0)
)


def create_egocentric_camera(env, actor_handle):
    """
    Creates an egocentric camera attached to a specified actor within the Isaac Gym environment.

    Parameters:
    - env (Env): The environment handle.
    - sim (Sim): The simulation handle.
    - viewer: The viewer handle for rendering (optional, used for debugging).
    - actor_handle (int): The handle of the actor to attach the camera to.

    Returns:
    - egocentric_cam_handle (int): The handle of the created camera sensor.
    - egocentric_cam_transform (gymapi.Transform): The transform of the camera relative to the actor.
    - cam_optical_frame (gymapi.Transform): The absolute transform of the camera in the simulation.
    """

    # Define the index of the camera sensor's rigid body within the actor
    CAMERA_SENSOR_INDEX = 28

    # Retrieve the rigid body states of the actor
    rigid_body_states = gym.get_actor_rigid_body_states(env, actor_handle, gymapi.STATE_POS)
    poses = rigid_body_states["pose"]

    # Ensure the CAMERA_SENSOR_INDEX is within bounds
    if CAMERA_SENSOR_INDEX >= len(poses):
        raise IndexError(f"CAMERA_SENSOR_INDEX {CAMERA_SENSOR_INDEX} is out of bounds for the actor's rigid bodies.")

    # Get the pose of the camera sensor's rigid body
    cam_rigid_body_pose = gymapi.Transform.from_buffer(poses[CAMERA_SENSOR_INDEX])

    # Define the local position offset for the camera sensor relative to its rigid body
    camera_sensor_local_pos = gymapi.Vec3(0.0, 0.0, 0.0)
    camera_sensor_world_pos = cam_rigid_body_pose.transform_point(camera_sensor_local_pos)

    # Define the camera's transform relative to the actor's rigid body
    egocentric_cam_transform = gymapi.Transform(camera_sensor_world_pos, cam_rigid_body_pose.r)

    # Retrieve the handle for the camera's rigid body
    cam_box_handle = gym.get_actor_rigid_body_handle(env, actor_handle, CAMERA_SENSOR_INDEX)

    # Define camera properties
    camera_props = gymapi.CameraProperties()
    camera_props.width = 1280
    camera_props.height = 720
    # Uncomment and set the desired horizontal field of view if needed
    # camera_props.horizontal_fov = 1.047197551  # 60 degrees

    # Create the camera sensor
    egocentric_cam_handle = gym.create_camera_sensor(env, camera_props)

    # # ---------------------------------------------- OPTIONAL ----------------------------------------------
    # # Define the camera's offset relative to the rigid body
    # camera_offset = gymapi.Vec3(0.28945, 0, -0.046825)  # Adjust based on your specific requirements
    #
    # # Define the camera's rotation using Euler angles (ZYX convention)
    # # For example, to rotate -90 degrees around the Y-axis (pitch)
    # roll = 0
    # pitch = -math.pi / 2  # -90 degrees
    # yaw = 0
    # camera_rotation = gymapi.Quat.from_euler_zyx(yaw, pitch, roll)
    #
    # print(f"Camera Offset: {camera_offset}, Camera Rotation: {camera_rotation}")
    #
    # # Define the optical frame transform for the camera
    # cam_optical_frame = gymapi.Transform(camera_offset, camera_rotation)
    #
    # # Optional: Draw axes for debugging purposes
    # # Ensure AXES_GEOM is defined appropriately in your environment
    # # gymutil.draw_lines(AXES_GEOM, gym, viewer, env, cam_optical_frame)

    # Attach the camera to the actor's rigid body with the defined transform
    gym.attach_camera_to_body(
        egocentric_cam_handle,
        env,
        cam_box_handle,
        egocentric_cam_transform,
        gymapi.FOLLOW_TRANSFORM
    )

    # Retrieve the camera's absolute transform in the simulation
    # cam_absolute_transform = gym.get_camera_transform(sim, env, egocentric_cam_handle)

    # Retrieve the camera's view and projection matrices for verification
    # camera_view_matrix = gym.get_camera_view_matrix(sim, env, egocentric_cam_handle)
    # camera_proj_matrix = gym.get_camera_proj_matrix(sim, env, egocentric_cam_handle)
    #
    # # Print matrices for debugging purposes
    # print("Camera View Matrix:\n", camera_view_matrix)
    # print("Camera Projection Matrix:\n", camera_proj_matrix)

    return egocentric_cam_handle, egocentric_cam_transform # , cam_absolute_transform

def create_ball(sim, env, fixed=False):
    ball_urdf = os.path.join("ball.urdf")
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = fixed
    ball_asset = gym.load_asset(sim, BASIC_ROOT, ball_urdf, asset_options)
    ball_pose = gymapi.Transform()
    ball_pose.p = gymapi.Vec3(0.3, 0, 0.1)
    ball_pose.r = gymapi.Quat(0, 0, 0, 1)
    asset_name = "ball"
    # create_actor -> group: int = - 1, filter: int = - 1, segmentationId: int = 0
    ball_handle = gym.create_actor(env, ball_asset, ball_pose, asset_name, 0, 0)
    return ball_handle


def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)


def print_asset_info(asset, name):
    print("======== Asset info %s: ========" % (name))
    num_bodies = gym.get_asset_rigid_body_count(asset)
    num_joints = gym.get_asset_joint_count(asset)
    num_dofs = gym.get_asset_dof_count(asset)
    print("Got %d bodies, %d joints, and %d DOFs" % (num_bodies, num_joints, num_dofs))

    # Iterate through bodies
    print("Bodies:")
    for i in range(num_bodies):
        name = gym.get_asset_rigid_body_name(asset, i)
        print(" %2d: '%s'" % (i, name))
    return


def new_sub_terrain(vertical_scale, horizontal_scale):
    return SubTerrain(
        width=TERRAIN_WIDTH,
        length=TERRAIN_LENGTH,
        vertical_scale=vertical_scale,
        horizontal_scale=horizontal_scale,
    )


def create_plane(sim):
    # add ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    # print("plane_params", plane_params)
    gym.add_ground(sim, plane_params)


def create_stairs(horizontal_scale, vertical_scale, step_width, step_height):
    stair = stairs_terrain(
        new_sub_terrain(vertical_scale, horizontal_scale),
        step_width=step_width,
        step_height=step_height,
    )

    heightfield = stair.height_field_raw

    vertices, triangles = convert_heightfield_to_trimesh(
        heightfield,
        horizontal_scale=horizontal_scale,
        vertical_scale=vertical_scale,
        slope_threshold=0,
    )
    tm_params = gymapi.TriangleMeshParams()
    tm_params.nb_vertices = vertices.shape[0]
    tm_params.nb_triangles = triangles.shape[0]
    tm_params.transform.p.x = TERRAIN_TRANS_X
    tm_params.transform.p.y = TERRAIN_TRANS_Y
    tm_params.transform.p.z = TERRAIN_TRANS_Z
    gym.add_triangle_mesh(sim, vertices.flatten(), triangles.flatten(), tm_params)
    # hf_params = gymapi.HeightFieldParams()
    # hf_params.transform.p.x = 0
    # hf_params.transform.p.y = 0
    # gym.add_heightfield(sim, heightfield, hf_params)
    return


def create_viewer(sim):
    # create viewer
    viewer_properties = gymapi.CameraProperties()
    viewer_properties.use_collision_geometry = True
    viewer = gym.create_viewer(sim, viewer_properties)
    if viewer is None:
        print("*** Failed to create viewer")
        quit()
    # position the camera
    gym.viewer_camera_look_at(viewer, None, VIEWER_POS, TARGET_POS)
    return viewer


def create_robot_actor(sim, env, fixed=True, dof_print=False):
    # asset_root = "dreamwaq/legged_gym/resources/"
    asset_file = asset_descriptors[0].file_name
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = fixed
    asset_options.flip_visual_attachments = asset_descriptors[0].flip_visual_attachments
    # asset_options.use_mesh_materials = True  # False
    # asset_options.use_physx_armature = True
    # asset_options.override_com = True
    # asset_options.override_inertia = True
    # asset_options.vhacd_enabled = False

    # ASSET
    print("Loading asset '%s' from '%s'" % (asset_file, RESOURCE_ROOT))
    robot_asset = gym.load_asset(sim, RESOURCE_ROOT, asset_file, asset_options)
    print_asset_info(robot_asset, "robot")

    # DOF PROPERTIES
    # get array of DOF names
    dof_names = gym.get_asset_dof_names(robot_asset)
    print("DOF names: %s" % dof_names)

    # get array of DOF properties
    dof_props = gym.get_asset_dof_properties(robot_asset)
    # print("DOF properties: %s" % dof_props)
    num_dofs = gym.get_asset_dof_count(robot_asset)
    # create an array of DOF states that will be used to update the actors
    dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)

    # get list of DOF types
    dof_types = [gym.get_asset_dof_type(robot_asset, i) for i in range(num_dofs)]

    # get the position slice of the DOF state array
    dof_positions = dof_states["pos"]

    # get the limit-related slices of the DOF properties array
    stiffnesses = dof_props["stiffness"]
    dampings = dof_props["damping"]
    armatures = dof_props["armature"]  # armature
    has_limits = dof_props["hasLimits"]
    lower_limits = dof_props["lower"]
    upper_limits = dof_props["upper"]

    # initialize default positions, limits, and speeds
    # (make sure they are in reasonable ranges)
    defaults = DEFAULTS  # DEFAULTS
    speeds = np.zeros(num_dofs)

    for i in range(num_dofs):
        if has_limits[i]:
            if dof_types[i] == gymapi.DOF_ROTATION:
                lower_limits[i] = clamp(lower_limits[i], -math.pi, math.pi)
                upper_limits[i] = clamp(upper_limits[i], -math.pi, math.pi)
        else:
            # set reasonable animation limits for unlimited joints
            if dof_types[i] == gymapi.DOF_ROTATION:
                # unlimited revolute joint
                lower_limits[i] = -math.pi
                upper_limits[i] = math.pi
            elif dof_types[i] == gymapi.DOF_TRANSLATION:
                # unlimited prismatic joint
                lower_limits[i] = -1.0
                upper_limits[i] = 1.0
        # set DOF position to default
        dof_positions[i] = defaults[i]
        # set speed depending on DOF type and range of motion
        if dof_types[i] == gymapi.DOF_ROTATION:
            # speed_scale = 1.0
            speeds[i] = 1.0 * clamp(
                2 * (upper_limits[i] - lower_limits[i]), 0.25 * math.pi, 3.0 * math.pi
            )
        else:
            speeds[i] = 1.0 * clamp(2 * (upper_limits[i] - lower_limits[i]), 0.1, 7.0)

    if dof_print:
        for i in range(num_dofs):
            print("DOF %d" % i)
            print("  Name:     '%s'" % dof_names[i])
            print("  Type:     %s" % gym.get_dof_type_string(dof_types[i]))
            print("  Stiffness:  %r" % stiffnesses[i])
            print("  Damping:  %r" % dampings[i])
            print("  Armature:  %r" % armatures[i])
            print("  Limited?  %r" % has_limits[i])
            print("  Default:  %r" % defaults[i])
            print("  Speed:    %r" % speeds[i])
            if has_limits[i]:
                print("    Lower   %f" % lower_limits[i])
                print("    Upper   %f" % upper_limits[i])

    # ACTOR
    pose = gymapi.Transform()
    pose.p = ROBOT_POS
    # from_euler_zyx(x-roll, y, z)
    # random_rad = random.uniform(-1.5, 1.5) # 90deg == 1.57rad
    pose.r = gymapi.Quat.from_euler_zyx(0, 0, 0)  # (random_rad, 0, 0)
    # pose.r = gymapi.Quat(0, 0, 0, 1)

    # necessary when loading an asset that is defined using z-up convention
    # into a simulation that uses y-up convention.
    robot_actor = gym.create_actor(env, robot_asset, pose, "actor", 0, 0)
    gym.set_actor_dof_states(env, robot_actor, dof_states, gymapi.STATE_ALL)

    # Egocentric camera
    egocentric_cam_handle, egocentric_cam_transform = create_egocentric_camera(
        env, robot_actor
    )

    return (
        robot_actor,
        dof_props,
        dof_states,
        dof_positions,
        speeds,
        egocentric_cam_handle,
        egocentric_cam_transform,
    )


def update_state(sim):
    root_state = gym.acquire_actor_root_state_tensor(sim)
    dof_state_tensor = gym.acquire_dof_state_tensor(sim)
    net_contact_forces = gym.acquire_net_contact_force_tensor(sim)
    rigid_body_states = gym.acquire_rigid_body_state_tensor(sim)
    # Functions starting with get_ are the older version of functions starting with acquire_

    gym.refresh_dof_state_tensor(sim)
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_net_contact_force_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim)

    root_states = gymtorch.wrap_tensor(root_state)
    dof_state = gymtorch.wrap_tensor(dof_state_tensor)
    net_contact_forces = gymtorch.wrap_tensor(net_contact_forces)
    rigid_body_states = gymtorch.wrap_tensor(
        rigid_body_states
    )  # shape (num_rigid_bodies, 13)
    # print(rigid_body_states.shape) # torch.Size([25, 13])

    dof_pos = dof_state.view(12, 2)[..., 0]
    dof_vel = dof_state.view(12, 2)[..., 1]
    # position([0:3]),
    # rotation([3:7]),
    # linear velocity([7:10]),
    # and angular velocity([10:13])
    base_pos = root_states[0][0:3]
    base_quat = root_states[0][3:7]

    ###############################################################
    # print("base pose: ", base_pos)
    # print("base quat: ", base_quat)
    # print("dof pos: ", dof_pos)

def create_sim():
    # initialize gym
    gym = gymapi.acquire_gym()

    # configure sim
    sim_params = gymapi.SimParams()
    sim_params.dt = DT
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

    args = gymutil.parse_arguments(description="Go2 test environment")
    if args.physics_engine == gymapi.SIM_FLEX:
        pass
    elif args.physics_engine == gymapi.SIM_PHYSX:
        # sim_params.physx.contact_collection = gymapi.CC_LAST_SUBSTEP  # Collect contacts for last substep only (value = 1)
        # sim_params.physx.contact_offset = 0.0
        sim_params.physx.solver_type = 1  # better but expensive
        # 0 : PGS (Iterative sequential impulse solver
        # 1 : TGS (Non-linear iterative solver, more robust but slightly more expensive
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.num_threads = args.num_threads
        sim_params.physx.use_gpu = args.use_gpu

    sim_params.use_gpu_pipeline = False
    if args.use_gpu_pipeline:
        print("WARNING: Forcing CPU pipeline.")

    sim = gym.create_sim(
        args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params
    )
    if sim is None:
        print("*** Failed to create sim")
        quit()
    return gym, sim


# simple asset descriptor for selecting from a list
class AssetDesc:
    def __init__(self, file_name, flip_visual_attachments=False):
        self.file_name = file_name
        self.flip_visual_attachments = flip_visual_attachments


if __name__ == "__main__":

    asset_descriptors = [AssetDesc("robots/go2/urdf/go2.urdf", True)]

    # Load simulation objects
    gym, sim = create_sim()

    # Load terrain
    # TERRAIN
    create_plane(sim)
    # horizontal_scale = HORIZONTAL_SCALE
    # vertical_scale = VERTICAL_SCALE
    # step_width = STEP_WIDTH
    # step_height = STEP_HEIGHT
    # create_stairs(horizontal_scale, vertical_scale, step_width, step_height)
    # create_pyramid_stairs(horizontal_scale, vertical_scale, step_width, step_height)

    # VIEWER
    viewer = create_viewer(sim)

    # Create environment (multiple envs can be created within a single sim)
    num_per_row = 1
    env = gym.create_env(sim, ENV_LOWER, ENV_UPPER, num_per_row)

    # Create actor (assign to env after loading asset)
    ball = create_ball(sim, env)
    (
        robot_actor,
        dof_props,
        dof_states,
        dof_positions,
        speeds,
        egocentric_cam_handle,
        egocentric_cam_transform,
    ) = create_robot_actor(sim, env, dof_print=False, fixed=True)

    lower_limits = dof_props["lower"]
    upper_limits = dof_props["upper"]
    # initialize animation state
    anim_state = ANIM_SEEK_LOWER
    current_dof = 0
    gym.set_actor_dof_states(env, robot_actor, dof_states, gymapi.STATE_POS)

    if not os.path.exists("egocentric_cam"):
        os.mkdir("egocentric_cam")

    frame_count = 0
    time = 0
    while not gym.query_viewer_has_closed(viewer):

        gym.simulate(sim)
        time += 1
        gym.fetch_results(sim, True)

        speed = speeds[current_dof]

        if anim_state == ANIM_SEEK_LOWER:
            dof_positions[current_dof] -= speed * DT
            if dof_positions[current_dof] <= lower_limits[current_dof]:
                dof_positions[current_dof] = lower_limits[current_dof]
                anim_state = ANIM_SEEK_UPPER
        elif anim_state == ANIM_SEEK_UPPER:
            dof_positions[current_dof] += speed * DT
            if dof_positions[current_dof] >= upper_limits[current_dof]:
                dof_positions[current_dof] = upper_limits[current_dof]
                anim_state = ANIM_SEEK_DEFAULT
        if anim_state == ANIM_SEEK_DEFAULT:
            dof_positions[current_dof] -= speed * DT
            if (
                dof_positions[current_dof] <= DEFAULTS[current_dof]
            ):
                dof_positions[current_dof] = DEFAULTS[
                    current_dof
                ]
                anim_state = ANIM_FINISHED
        elif anim_state == ANIM_FINISHED:
            dof_positions[current_dof] = DEFAULTS[current_dof]
            current_dof = (current_dof + 1) % 12  # num_dofs
            anim_state = ANIM_SEEK_LOWER

        gym.clear_lines(viewer)
        gym.set_actor_dof_states(env, robot_actor, dof_states, gymapi.STATE_POS)

        # get the DOF frame (origin and axis)
        dof_handle = gym.get_actor_dof_handle(env, robot_actor, current_dof)
        frame = gym.get_dof_frame(env, dof_handle)
        # draw a line from DOF origin along the DOF axis
        p1 = frame.origin
        p2 = frame.origin + frame.axis * 0.5
        color = gymapi.Vec3(1.0, 0.0, 0.0)
        gymutil.draw_line(p1, p2, color, gym, viewer, env)

        # Display camera position
        gymutil.draw_lines(AXES_GEOM, gym, viewer, env, egocentric_cam_transform)
        gymutil.draw_lines(SPHERE_GEOM, gym, viewer, env, egocentric_cam_transform)

        gym.step_graphics(sim)

        update_state(sim)

        # Execute camera sensor
        gym.render_all_camera_sensors(sim)

        # Save image at certain time steps
        if np.mod(frame_count, 100) == 0 and frame_count < 500:
            print("captured!", frame_count)
            # The gym utility to write images to disk is recommended only for RGB images.
            rgb_filename = "egocentric_cam/rgb_frame%d.png" % (frame_count)
            gym.write_camera_image_to_file(
                sim, env, egocentric_cam_handle, gymapi.IMAGE_COLOR, rgb_filename
            )
            # gym.write_camera_image_to_file(sim, env, env_cam, gymapi.IMAGE_COLOR, rgb_filename)

            # Retrieve image data directly. Use this for Depth, Segmentation, and Optical Flow images
            # Here we retrieve a depth image, normalize it to be visible in an
            # output image and then write it to disk using Pillow
            depth_image = gym.get_camera_image(
                sim, env, egocentric_cam_handle, gymapi.IMAGE_DEPTH
            )
            # depth_image = gym.get_camera_image(sim, env, env_cam, gymapi.IMAGE_DEPTH)

            # -inf implies no depth value, set it to zero. output will be black.
            depth_image[depth_image == -np.inf] = 0
            # clamp depth image to 10 meters to make output image human friendly
            depth_image[depth_image < -10] = -10

            # flip the direction so near-objects are light and far objects are dark
            normalized_depth = -255.0 * (depth_image / np.min(depth_image + 1e-4))

            # Convert to a pillow image and write it to disk
            normalized_depth_image = im.fromarray(
                normalized_depth.astype(np.uint8), mode="L"
            )
            normalized_depth_image.save(
                "egocentric_cam/depth_frame%d.jpg" % (frame_count)
            )

        gym.draw_viewer(viewer, sim, True)
        # cam_trans = gym.get_viewer_camera_transform(viewer, env)
        # print(cam_trans.p) # viewer position

        # Wait for dt to elapse in real time.
        # This synchronizes the physics simulation with the rendering rate.
        gym.sync_frame_time(sim)
        frame_count += 1

        # if time%60==0:
        #     print("time(sec): ",time//60)

    print("Done")

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
