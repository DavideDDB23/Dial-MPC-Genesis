import numpy as np
import torch


def wrap_to_pi(angles):
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


def gs_inv_quat(quat: torch.Tensor) -> torch.Tensor:
    """Returns the conjugate (inverse) of unit quaternions, supports arbitrary leading batch dims."""
    q = torch.as_tensor(quat)
    # scalar part unchanged; vector part negated
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def gs_transform_by_quat(pos, quat):
    qw, qx, qy, qz = quat.unbind(-1)

    rot_matrix = torch.stack(
        [
            1.0 - 2 * qy**2 - 2 * qz**2,
            2 * qx * qy - 2 * qz * qw,
            2 * qx * qz + 2 * qy * qw,
            2 * qx * qy + 2 * qz * qw,
            1 - 2 * qx**2 - 2 * qz**2,
            2 * qy * qz - 2 * qx * qw,
            2 * qx * qz - 2 * qy * qw,
            2 * qy * qz + 2 * qx * qw,
            1 - 2 * qx**2 - 2 * qy**2,
        ],
        dim=-1,
    ).reshape(*quat.shape[:-1], 3, 3)
    rotated_pos = torch.matmul(rot_matrix, pos.unsqueeze(-1)).squeeze(-1)

    return rotated_pos


def gs_quat2euler(quat):  # xyz
    # Extract quaternion components
    qw, qx, qy, qz = quat.unbind(-1)

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    pitch = torch.where(
        torch.abs(sinp) >= 1,
        torch.sign(sinp) * torch.tensor(torch.pi / 2),
        torch.asin(sinp),
    )

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack([roll, pitch, yaw], dim=-1)


def gs_euler2quat(xyz):  # xyz

    roll, pitch, yaw = xyz.unbind(-1)

    cosr = (roll * 0.5).cos()
    sinr = (roll * 0.5).sin()
    cosp = (pitch * 0.5).cos()
    sinp = (pitch * 0.5).sin()
    cosy = (yaw * 0.5).cos()
    siny = (yaw * 0.5).sin()

    qw = cosr * cosp * cosy + sinr * sinp * siny
    qx = sinr * cosp * cosy - cosr * sinp * siny
    qy = cosr * sinp * cosy + sinr * cosp * siny
    qz = cosr * cosp * siny - sinr * sinp * cosy

    return torch.stack([qw, qx, qy, qz], dim=-1)


def gs_quat_from_angle_axis(angle, axis):
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    return normalize(torch.cat([w, xyz], dim=-1))


def normalize(x, eps: float = 1e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


def gs_quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([w, x, y, z], dim=-1).view(shape)

    return quat


def gs_quat_apply(a, b):
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, 1:]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, :1] * t + xyz.cross(t, dim=-1)).view(shape)


def gs_quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, 1:3] = 0.
    quat_yaw = normalize(quat_yaw)
    return gs_quat_apply(quat_yaw, vec)


def gs_quat_conjugate(a):
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((a[:, :1], -a[:, 1:], ), dim=-1).view(shape)

def gs_rotate(vec: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """Rotate batched vectors by batched quaternions."""
    # vec: (...,3), quat: (...,4)
    w = quat[..., 0:1]  # (...,1)
    u = quat[..., 1:]   # (...,3)
    # dot(u, v) and cross(u, v)
    dot_u_v = (u * vec).sum(dim=-1, keepdim=True)  # (...,1)
    cross_u_v = torch.cross(u, vec, dim=-1)        # (...,3)
    # quaternion rotation formula: (w^2 - ||u||^2)*v + 2*(u·v)*u + 2*w*(u×v)
    part1 = (w * w - u.mul(u).sum(dim=-1, keepdim=True)) * vec
    part2 = 2 * dot_u_v * u
    part3 = 2 * w * cross_u_v
    return part1 + part2 + part3

def quat_to_3x3(q: torch.Tensor) -> torch.Tensor:
    """Converts batched quaternion to 3x3 rotation matrices."""
    # q: (...,4)
    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]
    # compute scale factor
    d = (q * q).sum(dim=-1)         # (...)
    s = 2.0 / d                     # (...)
    xs = x * s
    ys = y * s
    zs = z * s
    wx = w * xs
    wy = w * ys
    wz = w * zs
    xx = x * xs
    xy = x * ys
    xz = x * zs
    yy = y * ys
    yz = y * zs
    zz = z * zs
    # assemble rotation matrix rows
    row0 = torch.stack([1 - (yy + zz), xy - wz, xz + wy], dim=-1)
    row1 = torch.stack([xy + wz, 1 - (xx + zz), yz - wx], dim=-1)
    row2 = torch.stack([xz - wy, yz + wx, 1 - (xx + yy)], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)

def gs_inv_rotate(vec: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """Rotate batched vectors by the inverse (conjugate) of batched quaternions directly."""
    # vec: (...,3), quat: (...,4)
    w = quat[..., :1]  # (...,1)
    u = quat[..., 1:]  # (...,3)
    # inverse (conjugate) quaternion = [w, -u]
    inv_q = torch.cat([w, -u], dim=-1)
    return gs_rotate(vec, inv_q)


def global_to_body_velocity(v, q):
    """Transforms global-frame velocity to body-frame using the inverse quaternion rotation."""
    # Simply rotate by the inverse quaternion for batched inputs
    return gs_inv_rotate(v, q)


def body_to_global_velocity(v, q):
    """Transforms body-frame velocity to global frame using quaternion rotation."""
    # simply rotate by quaternion
    return gs_rotate(v, q)


def get_foot_step(duty_ratio, cadence, amplitude, phases, time):
    """
    Compute the foot step height.
    Args:
        duty_ratio: The duty ratio of the step (% on the ground)
        cadence: The cadence of the step (per second)
        amplitude: The height of the step
        phases: The phase of the step. Wraps around 1. (N-dim where N is the number of legs)
        time: The time of the step
    Returns:
        The height of the foot steps
    """
    
    def step_height(t, footphase, duty_ratio):
        angle = (t + torch.pi - footphase) % (2 * torch.pi) - torch.pi
        angle = torch.where(duty_ratio < 1, angle * 0.5 / (1 - duty_ratio), angle)
        clipped_angle = torch.clip(angle, -torch.pi / 2, torch.pi / 2)
        value = torch.where(duty_ratio < 1, torch.cos(clipped_angle), torch.tensor(0.0, device=t.device))
        final_value = torch.where(torch.abs(value) >= 1e-6, torch.abs(value), torch.tensor(0.0, device=t.device))
        return final_value
    
    h_steps = amplitude * torch.func.vmap(step_height, in_dims=(None, 0, None))(
        time * 2 * torch.pi * cadence + torch.pi,
        2 * torch.pi * phases,
        duty_ratio,
    )
    return h_steps