import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from dial_mpc.utils.utils import (
    gs_inv_quat,
    gs_rotate,
    gs_inv_rotate,
    gs_quat_to_3x3,
    global_to_body_velocity,
    body_to_global_velocity,
)

tol = 1e-5

def assert_allclose(a, b, msg):
    if not torch.allclose(a, b, atol=tol, rtol=0):
        raise AssertionError(f"{msg} failed: {a} vs {b}")

# 90° rotation around Z axis
q_z90 = torch.tensor([0.70710678, 0.0, 0.0, 0.70710678])
v_x = torch.tensor([1.0, 0.0, 0.0])
v_y = torch.tensor([0.0, 1.0, 0.0])

# expected results
exp_rot = v_y
exp_inv_rot = v_x

# tests

def test_gs_inv_quat():
    inv_q = gs_inv_quat(q_z90)
    expected = torch.tensor([0.70710678, -0.0, -0.0, -0.70710678])
    assert_allclose(inv_q, expected, 'gs_inv_quat')


def test_gs_rotate():
    v_rot = gs_rotate(v_x, q_z90)
    assert_allclose(v_rot, exp_rot, 'gs_rotate')


def test_gs_inv_rotate():
    v_back = gs_inv_rotate(exp_rot, q_z90)
    assert_allclose(v_back, exp_inv_rot, 'gs_inv_rotate')


def test_quat_to_3x3():
    R = gs_quat_to_3x3(q_z90)
    expected_R = torch.tensor([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0],
    ])
    assert_allclose(R, expected_R, 'quat_to_3x3')


def test_global_to_body_velocity():
    vb = global_to_body_velocity(v_x, q_z90)
    # should be rotated by inverse => -90° around Z
    expected_vb = torch.tensor([0.0, -1.0, 0.0])
    assert_allclose(vb, expected_vb, 'global_to_body_velocity')


def test_body_to_global_velocity():
    # expected body-frame vector after global_to_body_velocity is [0, -1, 0]
    expected_vb = torch.tensor([0.0, -1.0, 0.0])
    vg = body_to_global_velocity(expected_vb, q_z90)
    assert_allclose(vg, v_x, 'body_to_global_velocity')


def run_all():
    test_gs_inv_quat()
    test_gs_rotate()
    test_gs_inv_rotate()
    test_quat_to_3x3()
    test_global_to_body_velocity()
    test_body_to_global_velocity()
    print('All utils tests passed!')

if __name__ == '__main__':
    run_all() 