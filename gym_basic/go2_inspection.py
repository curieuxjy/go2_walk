"""
Go2 Inspection code based on Joint Monkey
------------------------------------------------------------------------------------
Got 29 bodies, 28 joints, and 12 DOFs
Bodies:
  0: 'base'
  1: 'FL_hip'
  2: 'FL_thigh'
  3: 'FL_calf'
  4: 'FL_calflower'
  5: 'FL_calflower1'
  6: 'FL_foot'
  7: 'FR_hip'
  8: 'FR_thigh'
  9: 'FR_calf'
 10: 'FR_calflower'
 11: 'FR_calflower1'
 12: 'FR_foot'
 13: 'Head_upper'
 14: 'Head_lower'
 15: 'RL_hip'
 16: 'RL_thigh'
 17: 'RL_calf'
 18: 'RL_calflower'
 19: 'RL_calflower1'
 20: 'RL_foot'
 21: 'RR_hip'
 22: 'RR_thigh'
 23: 'RR_calf'
 24: 'RR_calflower'
 25: 'RR_calflower1'
 26: 'RR_foot'
 27: 'imu'
 28: 'radar'
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
# Simulation Constants
DT = 1.0 / 60.0

# Environment Bounds
ENV_LOWER = gymapi.Vec3(-2.0, -2.0, 0.0)
ENV_UPPER = gymapi.Vec3(2.0, 2.0, 4.0)

# Robot Starting Position
ROBOT_POS = gymapi.Vec3(0.0, 0.0, 0.42)

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

# joint animation states
ANIM_SEEK_LOWER = 1
ANIM_SEEK_UPPER = 2
ANIM_SEEK_DEFAULT = 3
ANIM_FINISHED = 4

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


def create_plane(sim):
    # add ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    # print("plane_params", plane_params)
    gym.add_ground(sim, plane_params)


def create_viewer(sim):
    # create viewer
    viewer_properties = gymapi.CameraProperties()
    viewer_properties.use_collision_geometry = True
    viewer = gym.create_viewer(sim, viewer_properties)
    if viewer is None:
        print("*** Failed to create viewer")
        quit()
    # Viewer Configuration
    VIEWER_POS = gymapi.Vec3(1.0, 1.0, 1.0)
    TARGET_POS = gymapi.Vec3(0, 0, 0)
    gym.viewer_camera_look_at(viewer, None, VIEWER_POS, TARGET_POS)
    return viewer


def create_actor(sim, env, fixed=True, dof_print=False):

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

    speeds = np.zeros(num_dofs)
    speed_scale = 1.0

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
        dof_positions[i] = DEFAULTS[i]

        # set speed depending on DOF type and range of motion
        if dof_types[i] == gymapi.DOF_ROTATION:
            speeds[i] = speed_scale * clamp(
                2 * (upper_limits[i] - lower_limits[i]), 0.25 * math.pi, 3.0 * math.pi
            )
        else:
            speeds[i] = speed_scale * clamp(
                2 * (upper_limits[i] - lower_limits[i]), 0.1, 7.0
            )

    if dof_print:
        for i in range(num_dofs):
            print("DOF %d" % i)
            print("  Name:     '%s'" % dof_names[i])
            print("  Type:     %s" % gym.get_dof_type_string(dof_types[i]))
            print("  Stiffness:  %r" % stiffnesses[i])
            print("  Damping:  %r" % dampings[i])
            print("  Armature:  %r" % armatures[i])
            print("  Limited?  %r" % has_limits[i])
            print("  Default:  %r" % DEFAULTS[i])
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

    # necessary when loading an asset that is defined using z-up convention
    # into a simulation that uses y-up convention.
    robot_actor = gym.create_actor(env, robot_asset, pose, "actor", 0, 0)
    gym.set_actor_dof_states(env, robot_actor, dof_states, gymapi.STATE_ALL)

    return (
        robot_asset,
        robot_actor,
        dof_props,
        dof_states,
        dof_positions,
        speeds,
    )


def print_any_state(sim):
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
    rigid_body_states = gymtorch.wrap_tensor(rigid_body_states)  # shape (num_rigid_bodies, 13)

    dof_pos = dof_state.view(12, 2)[..., 0]
    dof_vel = dof_state.view(12, 2)[..., 1]
    # position([0:3]),
    # rotation([3:7]),
    # linear velocity([7:10]),
    # and angular velocity([10:13])
    base_pos = root_states[0][0:3]
    base_quat = root_states[0][3:7]

    ###############################################################
    print("base pose: ", base_pos)
    print("base quat: ", base_quat)
    print("dof pos: ", dof_pos)

    # order: (same to viewer ordering)
    # 'FL_hip', 'FL_thigh', 'FL_calf'
    # 'FR_hip', 'FR_thigh', 'FR_calf'
    # 'RL_hip', 'RL_thigh', 'RL_calf'
    # 'RR_hip', 'RR_thigh', 'RR_calf'

    # foot 6, 12, 20, 26
    print(">>> FL foot ", net_contact_forces[6])


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

def animate_dof(anim_state, current_dof, speed):
    global dof_positions

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
        if (dof_positions[current_dof] <= DEFAULTS[current_dof]):  # DEFAULTS[current_dof]:
            dof_positions[current_dof] = DEFAULTS[current_dof]  # DEFAULTS[current_dof]
            anim_state = ANIM_FINISHED

    elif anim_state == ANIM_FINISHED:
        dof_positions[current_dof] = DEFAULTS[current_dof]
        current_dof = (current_dof + 1) % 12  # num_dofs
        anim_state = ANIM_SEEK_LOWER

    return anim_state, current_dof

# simple asset descriptor for selecting from a list
class AssetDesc:
    def __init__(self, file_name, flip_visual_attachments=False):
        self.file_name = file_name
        self.flip_visual_attachments = flip_visual_attachments


if __name__ == "__main__":

    asset_descriptors = [AssetDesc("robots/go2/urdf/go2.urdf", True)]

    gym, sim = create_sim()

    create_plane(sim)

    # VIEWER
    viewer = create_viewer(sim)

    num_per_row = 1
    env = gym.create_env(sim, ENV_LOWER, ENV_UPPER, num_per_row)

    robot_asset, robot_actor, dof_props, dof_states, dof_positions, speeds = (
        create_actor(sim, env, dof_print=False, fixed=True)
    )

    gym.set_actor_dof_states(env, robot_actor, dof_states, gymapi.STATE_POS)

    lower_limits = dof_props["lower"]
    upper_limits = dof_props["upper"]
    # initialize animation state
    anim_state = ANIM_SEEK_LOWER
    current_dof = 0
    speed = speeds[current_dof]

    while not gym.query_viewer_has_closed(viewer):
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        anim_state, current_dof = animate_dof(anim_state, current_dof, speed)

        gym.clear_lines(viewer)
        gym.set_actor_dof_states(env, robot_actor, dof_states, gymapi.STATE_POS)

        # DOF visualization
        dof_handle = gym.get_actor_dof_handle(env, robot_actor, current_dof)
        frame = gym.get_dof_frame(env, dof_handle)
        # draw a line from DOF origin along the DOF axis
        p1 = frame.origin
        p2 = frame.origin + frame.axis * 0.5
        color = gymapi.Vec3(1.0, 0.0, 0.0)
        gymutil.draw_line(p1, p2, color, gym, viewer, env)

        # Execute simulation step
        gym.step_graphics(sim)

        print_any_state(sim)

        # Update viewer
        gym.draw_viewer(viewer, sim, True)
        # viewer_trans = gym.get_viewer_camera_transform(viewer, env)
        # print(viewer_trans.p) # viewer position

        # Wait for dt to elapse in real time.
        # This synchronizes the physics simulation with the rendering rate.
        gym.sync_frame_time(sim)

    print("Done")

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
