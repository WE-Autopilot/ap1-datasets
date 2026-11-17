"""Rigid-transform helpers. Quaternions are (x, y, z, w), as in ROS messages."""
import numpy as np


def quat_to_matrix(q):
    """3x3 rotation matrix of a quaternion (x, y, z, w); normalised first."""
    x, y, z, w = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(m):
    """Unit quaternion (x, y, z, w) of a 3x3 rotation matrix, with w >= 0."""
    m = np.asarray(m, dtype=float)
    tr = np.trace(m)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = np.array([(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s,
                      (m[1, 0] - m[0, 1]) / s, 0.25 * s])
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k]) * 2
        q = np.zeros(4)
        q[i] = 0.25 * s
        q[j] = (m[j, i] + m[i, j]) / s
        q[k] = (m[k, i] + m[i, k]) / s
        q[3] = (m[k, j] - m[j, k]) / s
    q /= np.linalg.norm(q)
    return q if q[3] >= 0 else -q


def quat_from_rpy(roll, pitch, yaw):
    """Quaternion (x, y, z, w) of fixed-axis roll, pitch, yaw in radians (REP 103)."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ])


def make_transform(translation, quat):
    """4x4 homogeneous transform from a translation and a quaternion."""
    t = np.eye(4)
    t[:3, :3] = quat_to_matrix(quat)
    t[:3, 3] = translation
    return t


def split_transform(t):
    """(translation, quaternion) of a 4x4 homogeneous transform."""
    return t[:3, 3].copy(), matrix_to_quat(t[:3, :3])


def invert(t):
    """Inverse of a 4x4 rigid transform."""
    inv = np.eye(4)
    inv[:3, :3] = t[:3, :3].T
    inv[:3, 3] = -t[:3, :3].T @ t[:3, 3]
    return inv


# Rotation of a camera optical frame (z forward, x right, y down) in its body
# frame (x forward, y left, z up): the usual <camera>_link -> <camera>_optical_frame.
OPTICAL_IN_BODY = quat_from_rpy(-np.pi / 2, 0.0, -np.pi / 2)


def body_from_optical(t_parent_optical):
    """parent -> camera body transform, given parent -> camera optical frame."""
    return t_parent_optical @ invert(make_transform(np.zeros(3), OPTICAL_IN_BODY))


def yaw_quat(yaw):
    """Quaternion of a rotation by yaw (radians) about z."""
    return np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)])


def body_twist(t_prev, t_curr, dt):
    """Linear and angular velocity of the body frame, expressed in the body frame.

    t_prev and t_curr are world -> body 4x4 transforms dt seconds apart
    (finite difference, so the result belongs to the middle of the interval).
    """
    r = t_curr[:3, :3]
    v_world = (t_curr[:3, 3] - t_prev[:3, 3]) / dt
    d = t_prev[:3, :3].T @ r                       # rotation over the interval, body frame
    angle = np.arccos(np.clip((np.trace(d) - 1) / 2, -1.0, 1.0))
    if angle < 1e-12:
        w = np.zeros(3)
    else:
        axis = np.array([d[2, 1] - d[1, 2], d[0, 2] - d[2, 0], d[1, 0] - d[0, 1]])
        w = axis / (2 * np.sin(angle)) * angle / dt
    return r.T @ v_world, w
