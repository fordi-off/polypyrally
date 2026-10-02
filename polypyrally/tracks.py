"""Tyre marks left on the ground by spinning or sliding wheels: a ring buffer of thin quads, drawn with the same
alpha pipeline as the dust."""
import numpy as np

CAP = 7000                      # quads kept


class Tracks:
    def __init__(self):
        self.verts = np.zeros((CAP * 6, 9), dtype=np.float32)
        self.head = 0           # next quad slot
        self.count = 0
        self.last = {}          # wheel index -> last (x, y, z, nx, nz) point
        self.dirty = None       # (first, last) quad range written since the last upload

    def feed(self, car, stage):
        for i, wl in enumerate(car.wheels):
            slip = max(abs(wl.kappa) * 4.5, wl.slip)
            if wl.contact and slip > 1.15 and wl.load > 300:
                x, z = wl.cx, wl.cz
                y = stage.height(x, z) + 0.045
                prev = self.last.get(i)
                self.last[i] = (x, y, z)
                if prev is None:
                    continue
                dx, dz = x - prev[0], z - prev[2]
                ln = (dx * dx + dz * dz) ** 0.5
                if ln < 0.05 or ln > 1.6:
                    continue
                nx, nz = -dz / ln * 0.085, dx / ln * 0.085
                a = min(0.55, 0.16 + 0.12 * (slip - 1.1))
                col = (0.07, 0.055, 0.045) if wl.mu > 0.72 else (0.04, 0.05, 0.03)
                q = self.head
                v = self.verts[q * 6:q * 6 + 6]
                pts = ((prev[0] + nx, prev[1], prev[2] + nz), (prev[0] - nx, prev[1], prev[2] - nz), (x - nx, y, z - nz),
                       (prev[0] + nx, prev[1], prev[2] + nz), (x - nx, y, z - nz), (x + nx, y, z + nz))
                for k, p in enumerate(pts):
                    v[k, 0:3] = p
                    v[k, 3:5] = 0.0
                    v[k, 5:8] = col
                    v[k, 8] = a
                self.head = (q + 1) % CAP
                self.count = min(CAP, self.count + 1)
                self.dirty = (q, q) if self.dirty is None else (min(self.dirty[0], q), max(self.dirty[1], q))
            else:
                self.last.pop(i, None)

    def clear(self):
        self.verts[:] = 0
        self.head = self.count = 0
        self.last.clear()
        self.dirty = None
