"""Tiny flat-shaded mesh builder. Every face carries its own normal and colour, which gives the faceted
low-poly look. Vertex layout (11 floats): position(3) normal(3) linear-colour(3) ao(1) emissive(1)."""
import math
import numpy as np

FLOATS = 11


def lin(c):
    """sRGB 0-255 tuple (or 0-1 floats > 1 not allowed) -> linear float rgb."""
    return tuple((v / 255.0) ** 2.2 for v in c[:3])


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _norm(a):
    ln = math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]) or 1.0
    return (a[0] / ln, a[1] / ln, a[2] / ln)


class MeshBuilder:
    def __init__(self):
        self.rows = []
        self.cx = self.cy = self.cz = 0.0          # running transform offset applied to pushed geometry

    # ------------------------------------------------------------ primitives (CCW seen from outside)
    def tri(self, a, b, c, color, ao=1.0, em=0.0):
        n = _norm(_cross(_sub(b, a), _sub(c, a)))
        col = lin(color)
        row = self.rows.append
        for p in (a, b, c):
            row((p[0], p[1], p[2], n[0], n[1], n[2], col[0], col[1], col[2], ao, em))

    def quad(self, a, b, c, d, color, ao=1.0, em=0.0):
        self.tri(a, b, c, color, ao, em)
        self.tri(a, c, d, color, ao, em)

    def poly(self, pts, color, ao=1.0, em=0.0):
        for i in range(1, len(pts) - 1):
            self.tri(pts[0], pts[i], pts[i + 1], color, ao, em)

    def box(self, cx, cy, cz, sx, sy, sz, color, em=0.0):
        x0, x1, y0, y1, z0, z1 = cx - sx / 2, cx + sx / 2, cy - sy / 2, cy + sy / 2, cz - sz / 2, cz + sz / 2
        P = lambda x, y, z: (x, y, z)
        q = self.quad
        q(P(x0, y0, z1), P(x1, y0, z1), P(x1, y1, z1), P(x0, y1, z1), color, 1.0, em)
        q(P(x1, y0, z0), P(x0, y0, z0), P(x0, y1, z0), P(x1, y1, z0), color, 1.0, em)
        q(P(x1, y0, z1), P(x1, y0, z0), P(x1, y1, z0), P(x1, y1, z1), color, 1.0, em)
        q(P(x0, y0, z0), P(x0, y0, z1), P(x0, y1, z1), P(x0, y1, z0), color, 1.0, em)
        q(P(x0, y1, z1), P(x1, y1, z1), P(x1, y1, z0), P(x0, y1, z0), color, 1.0, em)
        q(P(x0, y0, z0), P(x1, y0, z0), P(x1, y0, z1), P(x0, y0, z1), color, 1.0, em)

    def cone(self, base, radius, height, color, n=7, color2=None, rot=0.0, ao_base=0.75):
        """Cone with apex above `base` (x, y, z). Alternating facet shades via color2."""
        bx, by, bz = base
        apex = (bx, by + height, bz)
        ring = [(bx + math.cos(rot + i * math.tau / n) * radius, by, bz + math.sin(rot + i * math.tau / n) * radius) for i in range(n)]
        for i in range(n):
            col = color if (color2 is None or i % 2 == 0) else color2
            self.tri(ring[(i + 1) % n], ring[i], apex, col, 1.0)
        self.poly(list(reversed(ring)), color, ao_base)

    def prism(self, base, r0, r1, height, color, n=6, color2=None, rot=0.0, cap=True):
        """Frustum along +y: bottom radius r0, top radius r1."""
        bx, by, bz = base
        lo = [(bx + math.cos(rot + i * math.tau / n) * r0, by, bz + math.sin(rot + i * math.tau / n) * r0) for i in range(n)]
        hi = [(bx + math.cos(rot + i * math.tau / n) * r1, by + height, bz + math.sin(rot + i * math.tau / n) * r1) for i in range(n)]
        for i in range(n):
            j = (i + 1) % n
            col = color if (color2 is None or i % 2 == 0) else color2
            self.quad(lo[j], lo[i], hi[i], hi[j], col)
        if cap and r1 > 0:
            self.poly(hi, color)
        if cap:
            self.poly(list(reversed(lo)), color, 0.7)

    def blob(self, center, radius, color, detail=1, squash=1.0, jitter=0.0, seed=0, color2=None):
        """Low-poly icosphere. jitter makes it lumpy (bushes, rocks)."""
        verts, faces = _icosphere(detail)
        rnd = np.random.RandomState(seed)
        vs = []
        for v in verts:
            k = 1.0 + (rnd.rand() - 0.5) * 2 * jitter
            vs.append((center[0] + v[0] * radius * k, center[1] + v[1] * radius * k * squash, center[2] + v[2] * radius * k))
        for fi, (a, b, c) in enumerate(faces):
            col = color if (color2 is None or fi % 3) else color2
            self.tri(vs[a], vs[b], vs[c], col)

    # ------------------------------------------------------------ output
    def extend(self, other, dx=0.0, dy=0.0, dz=0.0):
        a = np.array(other.rows, dtype=np.float32)
        if len(a):
            a[:, 0] += dx
            a[:, 1] += dy
            a[:, 2] += dz
            self.rows.extend(map(tuple, a.tolist()))

    def build(self):
        if not self.rows:
            return np.zeros((0, FLOATS), np.float32)
        return np.array(self.rows, dtype=np.float32)


_ICO = {}


def _icosphere(detail):
    if detail in _ICO:
        return _ICO[detail]
    t = (1 + 5 ** 0.5) / 2
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
         (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    v = [_norm(p) for p in v]
    for _ in range(detail):
        cache = {}
        nf = []

        def mid(i, j):
            key = (min(i, j), max(i, j))
            if key not in cache:
                m = _norm(tuple((v[i][k] + v[j][k]) / 2 for k in range(3)))
                v.append(m)
                cache[key] = len(v) - 1
            return cache[key]
        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = nf
    _ICO[detail] = (v, f)
    return v, f
