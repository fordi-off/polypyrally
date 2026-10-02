"""Small 3D math helpers (numpy, row-vector-free: matrices act on column vectors, passed to GL transposed)."""
import math
import numpy as np


def perspective(fov_y_deg, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fov_y_deg) / 2)
    m = np.zeros((4, 4))
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def ortho(l, r, b, t, n, f):
    m = np.eye(4)
    m[0, 0] = 2 / (r - l)
    m[1, 1] = 2 / (t - b)
    m[2, 2] = -2 / (f - n)
    m[0, 3] = -(r + l) / (r - l)
    m[1, 3] = -(t + b) / (t - b)
    m[2, 3] = -(f + n) / (f - n)
    return m


def look_at(eye, target, up=(0.0, 1.0, 0.0)):
    eye, target, up = np.asarray(eye, float), np.asarray(target, float), np.asarray(up, float)
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    n = np.linalg.norm(r)
    if n < 1e-6:
        r = np.cross(f, np.array([1.0, 0.0, 0.0]))
        n = np.linalg.norm(r)
    r /= n
    u = np.cross(r, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = r, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def transform(R, t):
    """4x4 from a 3x3 rotation (nested lists / array) and translation."""
    m = np.eye(4)
    m[:3, :3] = R
    m[:3, 3] = t
    return m


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def gl_bytes(m):
    return np.asarray(m, dtype=np.float32).T.tobytes()
