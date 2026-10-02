"""Vectorised hash / value-noise helpers (numpy)."""
import numpy as np


def hash2(ix, iz, seed=0):
    """Deterministic float in [0,1) per integer lattice point."""
    with np.errstate(over="ignore"):
        return _hash2(ix, iz, seed)


def _hash2(ix, iz, seed):
    h = (np.asarray(ix).astype(np.uint32) * np.uint32(374761393)) ^ (np.asarray(iz).astype(np.uint32) * np.uint32(668265263)) ^ np.uint32((seed * 2246822519) & 0xFFFFFFFF)
    h = (h ^ (h >> np.uint32(13))) * np.uint32(1274126177)
    h = h ^ (h >> np.uint32(16))
    return (h & np.uint32(0xFFFFFF)).astype(np.float64) / float(0x1000000)


def vnoise(x, z, seed=0):
    """Smooth value noise in [-1, 1]."""
    x = np.asarray(x, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    ix, iz = np.floor(x), np.floor(z)
    fx, fz = x - ix, z - iz
    ux, uz = fx * fx * (3 - 2 * fx), fz * fz * (3 - 2 * fz)
    a, b = hash2(ix, iz, seed), hash2(ix + 1, iz, seed)
    c, d = hash2(ix, iz + 1, seed), hash2(ix + 1, iz + 1, seed)
    return (a + (b - a) * ux + (c - a) * uz + (a - b - c + d) * ux * uz) * 2 - 1


def fbm(x, z, octaves=5, lac=2.0, gain=0.5, seed=0):
    amp, f, tot, norm = 1.0, 1.0, 0.0, 0.0
    for o in range(octaves):
        tot = tot + vnoise(x * f, z * f, seed + o * 31) * amp
        norm += amp
        amp *= gain
        f *= lac
    return tot / norm


def smoothstep(a, b, x):
    t = np.clip((np.asarray(x, dtype=np.float64) - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def box_blur(a, r, passes=3):
    """Separable box blur with edge clamping; `passes` boxes approximate a gaussian."""
    out = a.astype(np.float64)
    for _ in range(passes):
        for axis in (0, 1):
            pad = [(0, 0), (0, 0)]
            pad[axis] = (r + 1, r)
            p = np.pad(out, pad, mode='edge')
            cs = np.cumsum(p, axis=axis)
            hi = np.take(cs, np.arange(2 * r + 1, p.shape[axis]), axis=axis)
            lo = np.take(cs, np.arange(0, p.shape[axis] - 2 * r - 1), axis=axis)
            out = (hi - lo) / (2 * r + 1)
    return out
