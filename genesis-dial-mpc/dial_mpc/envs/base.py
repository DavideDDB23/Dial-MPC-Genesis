import abc
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import functools

from flax import struct
import jax
import numpy as np
import jax.numpy as jp
import dial_mpc.utils.math as math
from jax import vmap
import torch

@struct.dataclass
class Transform():
  """Transforms the position and rotation of a coordinate frame.

  Attributes:
    pos: (3,) position transform of the coordinate frame
    rot: (4,) quaternion rotation the coordinate frame
  """

  pos: jax.Array
  rot: jax.Array

  def do(self, o):
    """Apply the transform."""
    return _transform_do(o, self)

  def inv_do(self, o):
    """Apply the inverse of the transform."""
    return _transform_inv_do(o, self)

  def to_local(self, t: 'Transform') -> 'Transform':
    """Move transform into basis of t."""
    pos = math.rotate(self.pos - t.pos, math.quat_inv(t.rot))
    rot = math.quat_mul(math.quat_inv(t.rot), self.rot)
    return Transform(pos=pos, rot=rot)

  @classmethod
  def create(
      cls, pos: Optional[jax.Array] = None, rot: Optional[jax.Array] = None
  ) -> 'Transform':
    """Creates a transform with either pos, rot, or both."""
    if pos is None and rot is None:
      raise ValueError('must specify either pos or rot')
    elif pos is None and rot is not None:
      pos = jp.zeros(rot.shape[:-1] + (3,))
    elif rot is None and pos is not None:
      rot = jp.tile(jp.array([1.0, 0.0, 0.0, 0.0]), pos.shape[:-1] + (1,))
    return Transform(pos=pos, rot=rot)

  @classmethod
  def zero(cls, shape=()) -> 'Transform':
    """Returns a zero transform with a batch shape."""
    pos = jp.zeros(shape + (3,))
    rot = jp.tile(jp.array([1.0, 0.0, 0.0, 0.0]), shape + (1,))
    return Transform(pos, rot)

  def vmap(self, in_axes=0, out_axes=0):
      """Returns an object that vmaps each follow-on instance method call."""

      # TODO: i think this is kinda handy, but maybe too clever?

      outer_self = self

      class VmapField:
        """Returns instance method calls as vmapped."""

        def __init__(self, in_axes, out_axes):
          self.in_axes = [in_axes]
          self.out_axes = [out_axes]

        def vmap(self, in_axes=0, out_axes=0):
          self.in_axes.append(in_axes)
          self.out_axes.append(out_axes)
          return self

        def __getattr__(self, attr):
          fun = getattr(outer_self.__class__, attr)
          # load the stack from the bottom up
          vmap_order = reversed(list(zip(self.in_axes, self.out_axes)))
          for in_axes, out_axes in vmap_order:
            fun = vmap(fun, in_axes=in_axes, out_axes=out_axes)
          fun = functools.partial(fun, outer_self)
          return fun

      return VmapField(in_axes, out_axes)

@struct.dataclass
class Motion():
  """Spatial motion vector describing linear and angular velocity.

  More on spatial vectors: http://royfeatherstone.org/spatial/v2/index.html

  Attributes:
    ang: (3,) angular velocity about a normal
    vel: (3,) linear velocity in the direction of the normal
  """

  ang: jax.Array
  vel: jax.Array

  def cross(self, other):
    return _motion_cross(other, self)

  def dot(self, m: Union['Motion', 'Force']) -> jax.Array:
    return jp.dot(self.vel, m.vel) + jp.dot(self.ang, m.ang)

  def matrix(self) -> jax.Array:
    return jp.concatenate([self.ang, self.vel], axis=-1)

  @classmethod
  def create(
      cls, ang: Optional[jax.Array] = None, vel: Optional[jax.Array] = None
  ) -> 'Motion':
    if ang is None and vel is None:
      raise ValueError('must specify either ang or vel')
    ang = jp.zeros_like(vel) if ang is None else ang
    vel = jp.zeros_like(ang) if vel is None else vel

    return Motion(ang=ang, vel=vel)

  @classmethod
  def zero(cls, shape=()) -> 'Motion':
    ang = jp.zeros(shape + (3,))
    vel = jp.zeros(shape + (3,))
    return Motion(ang, vel)

@struct.dataclass
class Force():
  """Spatial force vector describing linear and angular (torque) force.

  Attributes:
    ang: (3,) angular velocity about a normal
    vel: (3,) linear velocity in the direction of the normal
  """

  ang: jax.Array
  vel: jax.Array

  @classmethod
  def create(
      cls, ang: Optional[jax.Array] = None, vel: Optional[jax.Array] = None
  ) -> 'Force':
    if ang is None and vel is None:
      raise ValueError('must specify either ang or vel')
    ang = jp.zeros_like(vel) if ang is None else ang
    vel = jp.zeros_like(ang) if vel is None else vel

    return Force(ang=ang, vel=vel)

@functools.singledispatch
def _transform_do(other, self: Transform):
  del other, self
  return NotImplemented


@functools.singledispatch
def _transform_inv_do(other, self: Transform):
  del other, self
  return NotImplemented

@_transform_do.register(Transform)
def _(t: Transform, self: Transform) -> Transform:
  pos = self.pos + math.rotate(t.pos, self.rot)
  rot = math.quat_mul(self.rot, t.rot)
  return Transform(pos, rot)


@_transform_do.register(Motion)
def _(m: Motion, self: Transform) -> Motion:
  rot_t = math.quat_inv(self.rot)
  ang = math.rotate(m.ang, rot_t)
  vel = math.rotate(m.vel - jp.cross(self.pos, m.ang), rot_t)
  return Motion(ang, vel)


@_transform_inv_do.register(Motion)
def _(m: Motion, self: Transform) -> Motion:
  rot_t = self.rot
  ang = math.rotate(m.ang, rot_t)
  vel = math.rotate(m.vel, rot_t) + jp.cross(self.pos, ang)
  return Motion(ang, vel)


@_transform_do.register(Force)
def _(f: Force, self: Transform) -> Force:
  vel = math.rotate(f.vel, self.rot)
  ang = math.rotate(f.ang, self.rot) + jp.cross(self.pos, vel)
  return Force(ang, vel)

@functools.singledispatch
def _motion_cross(other, self: Motion):
  del other, self
  return NotImplemented


@_motion_cross.register(Motion)
def _(m: Motion, self: Motion) -> Motion:
  vel = jp.cross(self.ang, m.vel) + jp.cross(self.vel, m.ang)
  ang = jp.cross(self.ang, m.ang)
  return Motion(ang, vel)


@_motion_cross.register(Force)
def _(f: Force, self: Motion) -> Force:
  vel = jp.cross(self.ang, f.vel)
  ang = jp.cross(self.ang, f.ang) + jp.cross(self.vel, f.vel)
  return Force(ang, vel)


@struct.dataclass
class BaseState:
  """Dynamic state that changes after every pipeline step.

  Attributes:
    q: (q_size,) joint position vector
    qd: (qd_size,) joint velocity vector
    x: (num_links,) link position in world frame
    xd: (num_links,) link velocity in world frame
    ctrl: (motor_dofs len,) control command 
    site_xpos: (num_site,) feet positions
  """

  q: jax.Array
  qd: jax.Array
  x: Transform
  xd: Motion
  ctrl: jax.Array
  site_xpos: jax.Array


Observation = Union[jax.Array, Mapping[str, jax.Array]]
ObservationSize = Union[int, Mapping[str, Union[Tuple[int, ...], int]]]

@struct.dataclass
class State():
  """Environment state for training and inference."""

  pipeline_state: Optional[BaseState]
  obs: Observation
  reward: jax.Array
  done: jax.Array
  metrics: Dict[str, jax.Array] = struct.field(default_factory=dict)
  info: Dict[str, Any] = struct.field(default_factory=dict)


class PipelineEnv():
  
  def _to_jax(self, t):
    """Convert torch tensors to jax arrays."""
    return jp.array(t)
  
  def _reorder_genesis_to_brax(self, q_genesis, qd_genesis):
    """Reorder genesis state to match Brax ordering.
    
    Args:
      q_genesis: Genesis generalized coordinates
      qd_genesis: Genesis generalized velocities
      
    Returns:
      Tuple of reordered (q, qd)
    """
    # First 7 values (base pos and quat) stay the same
    q_base = q_genesis[:7]
      
    # Get the joint positions from q_genesis - starts at index 7
    q_joints = q_genesis[7:]
      
    # Genesis order is: [hip_FR, hip_FL, hip_RR, hip_RL, thigh_FR, thigh_FL, thigh_RR, thigh_RL, calf_FR, calf_FL, calf_RR, calf_RL]
    # Brax order is: [hip_FR, thigh_FR, calf_FR, hip_FL, thigh_FL, calf_FL, hip_RR, thigh_RR, calf_RR, hip_RL, thigh_RL, calf_RL]
      
    # Extract by joint type
    hips = q_joints[0:4]   # [FR, FL, RR, RL]
    thighs = q_joints[4:8]  # [FR, FL, RR, RL]
    calves = q_joints[8:12] # [FR, FL, RR, RL]
      
    # Reorder by leg
    q_joints_reordered = jp.concatenate([
          jp.array([hips[0], thighs[0], calves[0]]),     # FR leg
          jp.array([hips[1], thighs[1], calves[1]]),     # FL leg
          jp.array([hips[2], thighs[2], calves[2]]),     # RR leg
          jp.array([hips[3], thighs[3], calves[3]])      # RL leg
      ])
      
    # Combine base and reordered joints
    q_out = jp.concatenate([q_base, q_joints_reordered])
      
    # Reorder qd to match Brax
    # First 6 values (base lin/ang vel) stay the same
    qd_base = qd_genesis[:6]
    
    # The joint velocities are grouped by joint type in the same way
    qd_joints = qd_genesis[6:]
      
    # Apply the same reordering as for positions
    joint_vels_hips = qd_joints[0:4]
    joint_vels_thighs = qd_joints[4:8]
    joint_vels_calves = qd_joints[8:12]
      
    # Reorder by leg
    qd_joints_reordered = jp.concatenate([
          jp.array([joint_vels_hips[0], joint_vels_thighs[0], joint_vels_calves[0]]),  # FR leg
          jp.array([joint_vels_hips[1], joint_vels_thighs[1], joint_vels_calves[1]]),  # FL leg
          jp.array([joint_vels_hips[2], joint_vels_thighs[2], joint_vels_calves[2]]),  # RR leg
          jp.array([joint_vels_hips[3], joint_vels_thighs[3], joint_vels_calves[3]])   # RL leg
    ])
      
    # Combine base and reordered joint velocities
    qd_out = jp.concatenate([qd_base, qd_joints_reordered])

    return q_out, qd_out
  
  def _create_link_transform_and_motion(self, link_pos_raw, link_quat_raw, link_lin_vel_raw, link_ang_vel_raw):
    """Create Transform and Motion objects with reordered link data.
    
    Args:
      link_pos_raw: Raw link positions
      link_quat_raw: Raw link quaternions
      link_lin_vel_raw: Raw link linear velocities
      link_ang_vel_raw: Raw link angular velocities
      
    Returns:
      Tuple of (Transform, Motion, ordered link indices)
    """
    # Create reordering indices for links
    # Base stays at index 0
    base_idx = 0
      
    # FR leg: hip, thigh, calf
    fr_hip_idx = 1
    fr_thigh_idx = 5
    fr_calf_idx = 9
      
    # FL leg: hip, thigh, calf
    fl_hip_idx = 2
    fl_thigh_idx = 6
    fl_calf_idx = 10
    
    # RR leg: hip, thigh, calf
    rr_hip_idx = 3
    rr_thigh_idx = 7
    rr_calf_idx = 11
    
    # RL leg: hip, thigh, calf
    rl_hip_idx = 4
    rl_thigh_idx = 8
    rl_calf_idx = 12
    
    # List of ordered indices for each link
    ordered_indices = [
        base_idx,
        fr_hip_idx, fr_thigh_idx, fr_calf_idx,
        fl_hip_idx, fl_thigh_idx, fl_calf_idx,
        rr_hip_idx, rr_thigh_idx, rr_calf_idx,
        rl_hip_idx, rl_thigh_idx, rl_calf_idx
    ]
    
    # Create new arrays with reordered indices
    link_pos_ordered = jp.array([link_pos_raw[idx] for idx in ordered_indices])
    link_quat_ordered = jp.array([link_quat_raw[idx] for idx in ordered_indices])
    link_lin_vel_ordered = jp.array([link_lin_vel_raw[idx] for idx in ordered_indices])
    link_ang_vel_ordered = jp.array([link_ang_vel_raw[idx] for idx in ordered_indices])
    
    # Create Transform with reordered positions and orientations
    x = Transform(pos=link_pos_ordered, rot=link_quat_ordered)

    # Create Motion with reordered velocities
    cvel = Motion(vel=link_lin_vel_ordered, ang=link_ang_vel_ordered)
    
    return x, cvel, ordered_indices
  
  def _calculate_xd(self, cvel: Motion, inertial_positions_local: jax.Array, ordered_indices: list, link_orientations_world: jax.Array):
    """Calculate link velocities in the center of mass frame.
    
    Args:
      cvel: Link velocities (at link origin, in world frame)
      inertial_positions_local: Inertial positions (CoM offset from link origin, in local link frame)
      ordered_indices: Ordered indices for reordering links
      link_orientations_world: Orientations of the links in the world frame (x.rot)
      
    Returns:
      Motion object with velocities transformed to COM frame
    """
    # inertial_positions_local is _cached_inertial_positions, which is r_OC_L (Origin to CoM, local)
    # r_OC_L is the position of CoM of the link measured w.r.to the link origin, local to the link frame
    r_OC_L_ordered = inertial_positions_local[jp.array(ordered_indices)] # (num_links, 3)

    # Rotate r_OC_L to r_OC_W (Origin to CoM, world)
    # link_orientations_world is x.rot (num_links, 4)
    r_OC_W_ordered = jax.vmap(math.rotate)(r_OC_L_ordered, link_orientations_world) # (num_links, 3)

    # Calculate CoM velocity: v_C_W = v_O_W + ω_W × r_OC_W
    # cvel.vel is v_O_W (num_links, 3)
    # cvel.ang is ω_W (num_links, 3)
    vel_xd = cvel.vel + jax.vmap(jp.cross)(cvel.ang, r_OC_W_ordered)
    ang_xd = cvel.ang # Angular velocity of the CoM is the same as the link

    return Motion(vel=vel_xd, ang=ang_xd)
        
  # Function to transform the offset from local to world frame using the link's orientation
  def transform_offset_to_world(self, offset, quat):
    # Convert quaternion to rotation matrix
    w, x, y, z = quat
    xx, yy, zz = x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    
    rot_mat = jp.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)]
    ])
    
    # Apply rotation matrix to offset
    return jp.matmul(rot_mat, offset)
        
  
  def _calculate_site_positions(self, link_pos_ordered, link_quat_ordered):
    """Calculate site positions (feet and IMU).
    
    Args:
      link_pos_ordered: Reordered link positions
      link_quat_ordered: Reordered link quaternions
      
    Returns:
      Array of site positions in order [IMU, FR, FL, RR, RL]
    """
    # Calculate IMU position
    imu_offset = jp.array([-0.02557, 0.0, 0.04232])
    imu_site_pos = link_pos_ordered[0] + self.transform_offset_to_world(imu_offset, link_quat_ordered[0])  # base link
    
    # Get foot positions from the solver
    solver = self.scene.sim.solvers[self._rigid_solver_idx]
    feet_pos = self._to_jax(solver.get_geoms_pos(self._feet_site_id_brax))
  
    # Combine IMU and reordered feet positions
    site_positions = jp.concatenate([jp.expand_dims(imu_site_pos, 0), feet_pos])
    
    return site_positions

  def pipeline_init(
    self,
    q: jax.Array,
    qd: jax.Array,
    act: Optional[jax.Array] = None,
    ctrl: Optional[jax.Array] = None,
  ) -> BaseState:
    """Initialize the robot to the given state.
    
    Args:
      q: Joint positions (in MuJoCo order)
      qd: Joint velocities (in MuJoCo order)
      act: Optional action
      ctrl: Optional control command
      
    Returns:
      BaseState object
    """
    # Set initial base pose 
    self.robot.set_pos(q[:3], zero_velocity=True)
    self.robot.set_quat(q[3:7], zero_velocity=True)
      
    # Extract joint values from q (skipping base pos and quat)
    joint_values = q[7:]
      
    # Set joint positions according to DOF indices
    self.robot.set_dofs_position(
        position=joint_values,
        dofs_idx_local=self.motor_dofs,
        zero_velocity=True,
    )
      
    # Zero velocities
    self.robot.zero_all_dofs_velocity()
    
    # Get generalized coordinates and velocities from Genesis
    q_genesis = self._to_jax(self.robot.get_qpos())
    qd_genesis = self._to_jax(self.robot.get_dofs_velocity())
    
    # Reorder to match Brax/MuJoCo convention
    q_out, qd_out = self._reorder_genesis_to_brax(q_genesis, qd_genesis)
    
    # Create a ctrl array filled with zeros, no action to do
    ctrl = jp.zeros(len(self.motor_dofs))
    
    # Get link positions, orientations, and velocities
    link_pos_raw = self._to_jax(self.robot.get_links_pos())
    link_quat_raw = self._to_jax(self.robot.get_links_quat())
    link_lin_vel_raw = self._to_jax(self.robot.get_links_vel())
    link_ang_vel_raw = self._to_jax(self.robot.get_links_ang())
    
    # Create transform and motion for links
    x, cvel, ordered_indices = self._create_link_transform_and_motion(
        link_pos_raw, link_quat_raw, link_lin_vel_raw, link_ang_vel_raw
    )
    
    # Calculate link velocities
    cached_inertial_positions_jax = jp.asarray(self._cached_inertial_positions)
    xd = self._calculate_xd(cvel, cached_inertial_positions_jax, ordered_indices, x.rot)
    
    # Calculate site positions
    site_xpos = self._calculate_site_positions(x.pos, x.rot)

    # Create and return the base state
    return BaseState(q=q_out, qd=qd_out, x=x, xd=xd, ctrl=ctrl, site_xpos=site_xpos)  
    
  def pipeline_step(self, pipeline_state: BaseState, action: jax.Array) -> BaseState:
    """Step the physics simulation using the provided action.
    
    Args:
      pipeline_state: Current state
      action: Action to apply
      
    Returns:
      Updated BaseState
    """
    # Apply action to robot based on control mode
    if self._config.leg_control == "position":
        self.robot.control_dofs_position(position=action, dofs_idx_local=self.motor_dofs)
    else:  # torque (force) control
        self.robot.control_dofs_force(force=action, dofs_idx_local=self.motor_dofs)
    
    # Step the physics engine
    self.scene.step()

    # Get generalized coordinates and velocities from Genesis
    q_genesis = self._to_jax(self.robot.get_qpos())
    qd_genesis = self._to_jax(self.robot.get_dofs_velocity())
    
    # Reorder to match Brax
    q_out, qd_out = self._reorder_genesis_to_brax(q_genesis, qd_genesis)

    # Get link positions, orientations, and velocities
    link_pos_raw = self._to_jax(self.robot.get_links_pos())
    link_quat_raw = self._to_jax(self.robot.get_links_quat())
    link_lin_vel_raw = self._to_jax(self.robot.get_links_vel())
    link_ang_vel_raw = self._to_jax(self.robot.get_links_ang())
    
    # Create transform and motion for links
    x, cvel, ordered_indices = self._create_link_transform_and_motion(
        link_pos_raw, link_quat_raw, link_lin_vel_raw, link_ang_vel_raw
    )
    
    # Calculate link velocities
    cached_inertial_positions_jax = jp.asarray(self._cached_inertial_positions)
    xd = self._calculate_xd(cvel, cached_inertial_positions_jax, ordered_indices, x.rot)
    
    # Calculate site positions
    site_xpos = self._calculate_site_positions(x.pos, x.rot)

    # Create and return the base state
    return BaseState(q=q_out, qd=qd_out, x=x, xd=xd, ctrl=action, site_xpos=site_xpos)  