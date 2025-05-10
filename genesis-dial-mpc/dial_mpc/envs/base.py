import abc
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import functools

from flax import struct
import jax
import numpy as np
import jax.numpy as jp
import dial_mpc.utils.math as math
from jax import vmap

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
  """

  q: jax.Array
  qd: jax.Array
  x: Transform
  xd: Motion
  ctrl: jax.Array


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
  
  def pipeline_init(
      self,
      q: jax.Array,
      qd: jax.Array,
      act: Optional[jax.Array] = None,
      ctrl: Optional[jax.Array] = None,
  ) -> BaseState:
      # The Genesis DOF order is different from the MuJoCo keyframe order
      # MuJoCo keyframe order from 'home':
      # [0, 0, 0.27, 1, 0, 0, 0, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8]
      # This means: [base_pos(3), base_quat(4), 
      #              FR_hip, FR_thigh, FR_calf, 
      #              FL_hip, FL_thigh, FL_calf, 
      #              RR_hip, RR_thigh, RR_calf, 
      #              RL_hip, RL_thigh, RL_calf]
      
      # Genesis joint DOF indices: 
      # Motor DOFs: [6, 10, 14, 7, 11, 15, 8, 12, 16, 9, 13, 17]
      # This means: [FR_hip, FR_thigh, FR_calf, 
      #              FL_hip, FL_thigh, FL_calf, 
      #              RR_hip, RR_thigh, RR_calf, 
      #              RL_hip, RL_thigh, RL_calf]
      
      # Set initial base pose 
      self.robot.set_pos(q[:3], zero_velocity=True)
      self.robot.set_quat(q[3:7], zero_velocity=True)
      
      # We need to map the MuJoCo keyframe joint order to Genesis DOF indices
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

      # Get current state from Genesis
      # Convert torch tensors to jax arrays where needed
      def to_jax(t): return jp.array(t.numpy()) if hasattr(t, 'numpy') else jp.array(t)

      # Get generalized coordinates and velocities
      q_genesis = to_jax(self.robot.get_qpos())
      qd_genesis = to_jax(self.robot.get_dofs_velocity())
      
      # Now reorder q_genesis to match Brax/MuJoCo ordering
      # First 7 values (base pos and quat) stay the same
      q_base = q_genesis[:7]
      
      # The joint values are grouped by joint type (all hips, all thighs, all calves)
      # We need to regroup them by leg (FR_hip/thigh/calf, FL_hip/thigh/calf, etc.)
      
      # Get the joint positions from q_genesis - starts at index 7
      q_joints = q_genesis[7:]
      
      # Genesis order is (from the debug output):
      # [hip_FR, hip_FL, hip_RR, hip_RL, thigh_FR, thigh_FL, thigh_RR, thigh_RL, calf_FR, calf_FL, calf_RR, calf_RL]
      
      # We need to reorder to:
      # [hip_FR, thigh_FR, calf_FR, hip_FL, thigh_FL, calf_FL, hip_RR, thigh_RR, calf_RR, hip_RL, thigh_RL, calf_RL]
      
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
      
      # Similarly, we should reorder qd to match Brax/MuJoCo
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

      # Get link positions and orientations
      link_pos_raw = to_jax(self.robot.get_links_pos())
      link_quat_raw = to_jax(self.robot.get_links_quat())
      
      # Based on the debug output, we need to reorder the link transforms
      # The expected ordering is:
      # [base, 
      #  FR_hip, FR_thigh, FR_calf, 
      #  FL_hip, FL_thigh, FL_calf, 
      #  RR_hip, RR_thigh, RR_calf, 
      #  RL_hip, RL_thigh, RL_calf]
      
      # Looking at current ordering:
      # Index 0: base
      # Indices 1-4: Four hip links [FR, FL, RR, RL]
      # Indices 5-8: Four thigh links [FR, FL, RR, RL]
      # Indices 9-12: Four calf links [FR, FL, RR, RL]
      
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
      
      # Create new arrays with reordered indices
      link_pos_ordered = jp.array([
          link_pos_raw[base_idx],
          link_pos_raw[fr_hip_idx],
          link_pos_raw[fr_thigh_idx],
          link_pos_raw[fr_calf_idx],
          link_pos_raw[fl_hip_idx],
          link_pos_raw[fl_thigh_idx],
          link_pos_raw[fl_calf_idx],
          link_pos_raw[rr_hip_idx],
          link_pos_raw[rr_thigh_idx],
          link_pos_raw[rr_calf_idx],
          link_pos_raw[rl_hip_idx],
          link_pos_raw[rl_thigh_idx],
          link_pos_raw[rl_calf_idx]
      ])
      
      link_quat_ordered = jp.array([
          link_quat_raw[base_idx],
          link_quat_raw[fr_hip_idx],
          link_quat_raw[fr_thigh_idx],
          link_quat_raw[fr_calf_idx],
          link_quat_raw[fl_hip_idx],
          link_quat_raw[fl_thigh_idx],
          link_quat_raw[fl_calf_idx],
          link_quat_raw[rr_hip_idx],
          link_quat_raw[rr_thigh_idx],
          link_quat_raw[rr_calf_idx],
          link_quat_raw[rl_hip_idx],
          link_quat_raw[rl_thigh_idx],
          link_quat_raw[rl_calf_idx]
      ])
      
      # Create Transform with reordered positions and orientations
      x = Transform(pos=link_pos_ordered, rot=link_quat_ordered)

      # Get link velocities in the same reordered way
      link_lin_vel_raw = to_jax(self.robot.get_links_vel())
      link_ang_vel_raw = to_jax(self.robot.get_links_ang())
      
      # Reorder velocities the same way as positions
      link_lin_vel_ordered = jp.array([
          link_lin_vel_raw[base_idx],
          link_lin_vel_raw[fr_hip_idx],
          link_lin_vel_raw[fr_thigh_idx],
          link_lin_vel_raw[fr_calf_idx],
          link_lin_vel_raw[fl_hip_idx],
          link_lin_vel_raw[fl_thigh_idx],
          link_lin_vel_raw[fl_calf_idx],
          link_lin_vel_raw[rr_hip_idx],
          link_lin_vel_raw[rr_thigh_idx],
          link_lin_vel_raw[rr_calf_idx],
          link_lin_vel_raw[rl_hip_idx],
          link_lin_vel_raw[rl_thigh_idx],
          link_lin_vel_raw[rl_calf_idx]
      ])
      
      link_ang_vel_ordered = jp.array([
          link_ang_vel_raw[base_idx],
          link_ang_vel_raw[fr_hip_idx],
          link_ang_vel_raw[fr_thigh_idx],
          link_ang_vel_raw[fr_calf_idx],
          link_ang_vel_raw[fl_hip_idx],
          link_ang_vel_raw[fl_thigh_idx],
          link_ang_vel_raw[fl_calf_idx],
          link_ang_vel_raw[rr_hip_idx],
          link_ang_vel_raw[rr_thigh_idx],
          link_ang_vel_raw[rr_calf_idx],
          link_ang_vel_raw[rl_hip_idx],
          link_ang_vel_raw[rl_thigh_idx],
          link_ang_vel_raw[rl_calf_idx]
      ])

      # Get inertial properties and reorder them the same way
      inertial_positions = []
      for i in range(len(self.robot.links)):
          link = self.robot.links[i]
          inertial_pos_local = link.inertial_pos
          inertial_pos_array = jp.array(inertial_pos_local) if not hasattr(inertial_pos_local, 'numpy') else jp.array(inertial_pos_local.numpy())
          inertial_positions.append(inertial_pos_array)
      
      # Reorder inertial positions the same way as links
      inertial_positions_ordered = [
          inertial_positions[base_idx],
          inertial_positions[fr_hip_idx],
          inertial_positions[fr_thigh_idx],
          inertial_positions[fr_calf_idx],
          inertial_positions[fl_hip_idx],
          inertial_positions[fl_thigh_idx],
          inertial_positions[fl_calf_idx],
          inertial_positions[rr_hip_idx],
          inertial_positions[rr_thigh_idx],
          inertial_positions[rr_calf_idx],
          inertial_positions[rl_hip_idx],
          inertial_positions[rl_thigh_idx],
          inertial_positions[rl_calf_idx]
      ]
      
      # Stack the reordered inertial positions
      link_com_local = jp.stack(inertial_positions_ordered)
      
      # Compute the offset - in local frame the offset is negative of inertial pos
      offset_local = -link_com_local
      
      # Transform the offset to world frame
      offset = Transform.create(pos=offset_local)
      
      # Create spatial velocity with reordered velocities
      cvel = Motion(vel=link_lin_vel_ordered, ang=link_ang_vel_ordered)

      # Transform velocities by offset
      xd = offset.vmap().do(cvel)

      ctrl = jp.zeros(len(self.motor_dofs))

      return BaseState(q=q_out, qd=qd_out, x=x, xd=xd, ctrl=ctrl)
        