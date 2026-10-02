"""Dust, gravel spray and tyre smoke as camera-facing puffs (numpy-simulated, drawn in one batch)."""
import math
import numpy as np

MAX = 2200


class Particles:
    def __init__(self):
        self.n = 0
        self.pos = np.zeros((MAX, 3))
        self.vel = np.zeros((MAX, 3))
        self.life = np.zeros(MAX)
        self.max_life = np.ones(MAX)
        self.size = np.zeros(MAX)
        self.grow = np.zeros(MAX)
        self.col = np.zeros((MAX, 3))
        self.alpha = np.zeros(MAX)
        self.grav = np.zeros(MAX)
        self.drag = np.ones(MAX) * 1.6
        self.rng = np.random.RandomState(5)

    def emit(self, pos, vel, count, life, size, grow, col, alpha, spread=0.6, grav=0.0, drag=1.6):
        if count <= 0:
            return
        k = min(count, MAX - self.n)
        if k <= 0:
            return
        i = self.n
        r = self.rng
        self.pos[i:i + k] = np.array(pos) + r.randn(k, 3) * 0.12
        self.vel[i:i + k] = np.array(vel) + r.randn(k, 3) * spread
        self.life[i:i + k] = life * (0.7 + 0.6 * r.rand(k))
        self.max_life[i:i + k] = self.life[i:i + k]
        self.size[i:i + k] = size * (0.7 + 0.6 * r.rand(k))
        self.grow[i:i + k] = grow
        self.col[i:i + k] = col
        self.alpha[i:i + k] = alpha
        self.grav[i:i + k] = grav
        self.drag[i:i + k] = drag
        self.n += k

    def update(self, dt):
        n = self.n
        if n == 0:
            return
        self.life[:n] -= dt
        alive = self.life[:n] > 0
        if not alive.all():
            idx = np.nonzero(alive)[0]
            m = len(idx)
            for a in (self.pos, self.vel, self.life, self.max_life, self.size, self.grow, self.col, self.alpha, self.grav, self.drag):
                a[:m] = a[idx]
            self.n = n = m
        if n == 0:
            return
        self.vel[:n] *= (1.0 - self.drag[:n] * dt)[:, None]
        self.vel[:n, 1] += np.where(self.grav[:n] > 0, -self.grav[:n], 0.55) * dt     # dust drifts up, chunks fall
        self.pos[:n] += self.vel[:n] * dt
        self.size[:n] += self.grow[:n] * dt

    def vertices(self, right, up):
        """Camera-facing hexagon puffs as float32 (N*12*3... ) [pos3, uv2, rgba4] per vertex."""
        n = self.n
        if n == 0:
            return None
        f = (self.life[:n] / self.max_life[:n])
        a = self.alpha[:n] * np.clip(f * 1.6, 0, 1) * np.clip((1 - f) * 8.0, 0, 1)
        sz = self.size[:n]
        c = self.pos[:n]
        # hexagon as 6 triangles around the centre
        verts = np.zeros((n, 6, 3, 9), dtype=np.float32)
        ang = np.arange(7) * math.tau / 6 + 0.3
        for k in range(6):
            for j, (u0, v0) in enumerate(((0.0, 0.0), (math.cos(ang[k]), math.sin(ang[k])), (math.cos(ang[k + 1]), math.sin(ang[k + 1])))):
                p = c + (right[None, :] * (u0 * sz)[:, None]) + (up[None, :] * (v0 * sz)[:, None])
                verts[:, k, j, 0:3] = p
                verts[:, k, j, 3] = u0
                verts[:, k, j, 4] = v0
                verts[:, k, j, 5:8] = self.col[:n]
                verts[:, k, j, 8] = a
        return verts.reshape(-1, 9)
