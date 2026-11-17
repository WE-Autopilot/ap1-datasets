"""Unit tests for the KITTI adapter's transform helpers. No ROS needed."""
import numpy as np
import pytest

from ap1_datasets.kitti import geometry as geo


def random_quat(rng):
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


def test_quat_matrix_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = random_quat(rng)
        back = geo.matrix_to_quat(geo.quat_to_matrix(q))
        assert np.allclose(back, q if q[3] >= 0 else -q, atol=1e-9)


def test_quat_from_rpy_matches_yaw_quat():
    assert np.allclose(geo.quat_from_rpy(0, 0, 0.7), geo.yaw_quat(0.7))


def test_invert():
    rng = np.random.default_rng(1)
    t = geo.make_transform(rng.normal(size=3), random_quat(rng))
    assert np.allclose(t @ geo.invert(t), np.eye(4))


def test_optical_in_body_axes():
    """Optical z (viewing direction) is body x; optical x is body -y; optical y is body -z."""
    r = geo.quat_to_matrix(geo.OPTICAL_IN_BODY)
    assert np.allclose(r @ [0, 0, 1], [1, 0, 0])
    assert np.allclose(r @ [1, 0, 0], [0, -1, 0])
    assert np.allclose(r @ [0, 1, 0], [0, 0, -1])


def test_body_from_optical_undoes_the_optical_rotation():
    rng = np.random.default_rng(2)
    t_base_body = geo.make_transform(rng.normal(size=3), random_quat(rng))
    t_base_optical = t_base_body @ geo.make_transform(np.zeros(3), geo.OPTICAL_IN_BODY)
    assert np.allclose(geo.body_from_optical(t_base_optical), t_base_body)


def test_body_twist_straight_and_turning():
    t0 = geo.make_transform([0, 0, 0], geo.yaw_quat(np.pi / 2))   # facing +y
    t1 = geo.make_transform([0, 1, 0], geo.yaw_quat(np.pi / 2))
    v, w = geo.body_twist(t0, t1, 0.1)
    assert np.allclose(v, [10, 0, 0])      # forward in the body frame
    assert np.allclose(w, 0)

    t2 = geo.make_transform([0, 1, 0], geo.yaw_quat(np.pi / 2 + 0.05))
    _, w = geo.body_twist(t1, t2, 0.1)
    assert w == pytest.approx([0, 0, 0.5])
