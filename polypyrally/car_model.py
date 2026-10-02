"""The rally coupe, lofted from a handful of cross-sections into flat-shaded facets."""
import math
from .meshkit import MeshBuilder

# x, y_bottom, y_belt, y_roof, w_bottom, w_belt, w_roof
STATIONS = [
    (2.18, -0.22, 0.02, 0.05, 0.68, 0.82, 0.72),
    (1.86, -0.30, 0.13, 0.16, 0.82, 0.92, 0.84),
    (1.30, -0.33, 0.27, 0.30, 0.86, 0.95, 0.88),
    (0.92, -0.33, 0.33, 0.36, 0.86, 0.95, 0.88),
    (0.12, -0.33, 0.35, 0.82, 0.86, 0.95, 0.72),
    (-0.72, -0.33, 0.37, 0.80, 0.86, 0.96, 0.70),
    (-1.50, -0.32, 0.36, 0.54, 0.86, 0.97, 0.76),
    (-1.95, -0.30, 0.33, 0.38, 0.84, 0.94, 0.82),
    (-2.18, -0.26, 0.28, 0.32, 0.78, 0.88, 0.76),
]
GLASS = (22, 30, 40)
DARK = (22, 22, 26)


REF = (0.0, 0.2, 0.0)


def _out(b, pts, col, em=0.0, ao=1.0):
    """Add a convex polygon, wound so that its normal points away from the car's interior."""
    cx = sum(p[0] for p in pts) / len(pts) - REF[0]
    cy = sum(p[1] for p in pts) / len(pts) - REF[1]
    cz = sum(p[2] for p in pts) / len(pts) - REF[2]
    a, b_, c = pts[0], pts[1], pts[2]
    ux, uy, uz = b_[0] - a[0], b_[1] - a[1], b_[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    if nx * cx + ny * cy + nz * cz < 0:
        pts = pts[::-1]
    b.poly(pts, col, ao, em)


def body(paint, stripe=(236, 236, 240)):
    b = MeshBuilder()
    S = STATIONS
    dark_paint = tuple(int(c * 0.82) for c in paint)
    for i in range(len(S) - 1):
        x0, yb0, ye0, yr0, wb0, we0, wr0 = S[i]
        x1, yb1, ye1, yr1, wb1, we1, wr1 = S[i + 1]
        for sgn in (1, -1):
            _out(b, [(x0, yb0, sgn * wb0), (x1, yb1, sgn * wb1), (x1, ye1, sgn * we1), (x0, ye0, sgn * we0)], paint)
            glass = (yr0 - ye0 > 0.12) and (yr1 - ye1 > 0.12)
            _out(b, [(x0, ye0, sgn * we0), (x1, ye1, sgn * we1), (x1, yr1, sgn * wr1), (x0, yr0, sgn * wr0)], GLASS if glass else paint)
        _out(b, [(x0, yb0, wb0), (x0, yb0, -wb0), (x1, yb1, -wb1), (x1, yb1, wb1)], DARK)
        mid = (x0 + x1) / 2
        glass_top = (0.12 < mid < 0.92) or (-1.5 < mid < -0.72)
        for k, (lo, hi) in enumerate(((0.0, 0.3), (0.3, 0.72), (0.72, 1.0))):
            col = GLASS if glass_top else (stripe if k == 1 else paint)
            for sgn in (1, -1):
                _out(b, [(x0, yr0, sgn * lo * wr0), (x0, yr0, sgn * hi * wr0), (x1, yr1, sgn * hi * wr1), (x1, yr1, sgn * lo * wr1)], col)
    for idx in (0, -1):
        x, yb, ye, yr, wb, we, wr = S[idx]
        _out(b, [(x, yb, wb), (x, ye, we), (x, yr, wr), (x, yr, -wr), (x, ye, -we), (x, yb, -wb)], dark_paint)
    b.box(2.17, -0.18, 0.0, 0.20, 0.16, 1.5, DARK)
    b.box(2.12, -0.30, 0.0, 0.36, 0.05, 1.6, (30, 30, 34))
    b.box(-2.17, -0.18, 0.0, 0.2, 0.16, 1.5, DARK)
    for s in (1, -1):
        b.box(2.14, 0.10, s * 0.62, 0.10, 0.12, 0.30, (255, 238, 190), em=0.55)
        b.box(2.17, -0.05, s * 0.55, 0.08, 0.08, 0.22, (255, 150, 40), em=0.2)
        b.box(-2.19, 0.18, s * 0.66, 0.06, 0.12, 0.36, (255, 30, 24), em=0.45)
        for ax in (1.28, -1.32):
            _arch(b, ax, s)
        b.box(0.62, 0.42, s * 1.04, 0.16, 0.10, 0.12, DARK)
        b.box(0.0, -0.30, s * 0.97, 1.5, 0.06, 0.06, (34, 34, 38))
        _disc(b, (-0.2, 0.16, s * 0.965), s, 0.20, (244, 244, 248))
        b.box(-1.98, 0.40, s * 0.62, 0.1, 0.22, 0.06, (28, 28, 32))
    b.box(-0.3, 0.84, 0.0, 0.7, 0.07, 0.3, DARK)
    b.box(-2.02, 0.50, 0.0, 0.34, 0.04, 1.5, (28, 28, 32))
    _disc(b, (1.30, 0.285, 0.0), 0, 0.26, (244, 244, 248), top=True)
    return b.build()


def _disc(b, c, s, r, col, n=10, top=False):
    pts = []
    for i in range(n):
        a = i * math.tau / n
        if top:
            pts.append((c[0] + math.cos(a) * r, c[1] + 0.012, c[2] + math.sin(a) * r))
        else:
            pts.append((c[0] + math.cos(a) * r, c[1] + math.sin(a) * r, c[2]))
    _out(b, pts, col)


def _arch(b, ax, s):
    n = 12
    pts = []
    for i in range(n + 1):
        a = math.pi * i / n
        pts.append((ax + math.cos(a) * 0.46, -0.20 + math.sin(a) * 0.46 * 0.95, s * 0.96))
    cx = (ax, -0.20, s * 0.96)
    for i in range(n):
        _out(b, [cx, pts[i], pts[i + 1]], (14, 14, 18))


def wheel():
    """Tyre + rim, axis along z, centred at the origin."""
    b = MeshBuilder()
    R, W, n = 0.33, 0.14, 14
    for i in range(n):
        a0, a1 = i * math.tau / n, (i + 1) * math.tau / n
        c0, s0, c1, s1 = math.cos(a0), math.sin(a0), math.cos(a1), math.sin(a1)
        tone = (30, 30, 34) if i % 2 == 0 else (24, 24, 28)
        b.quad((c0 * R, s0 * R, W), (c1 * R, s1 * R, W), (c1 * R, s1 * R, -W), (c0 * R, s0 * R, -W), tone)
        r2 = R * 0.62
        for sgn in (1, -1):
            zz = sgn * W
            ring = [(c0 * R, s0 * R, zz), (c1 * R, s1 * R, zz), (c1 * r2, s1 * r2, zz * 1.04), (c0 * r2, s0 * r2, zz * 1.04)]
            b.quad(*(ring if sgn > 0 else ring[::-1]), (18, 18, 22))
            rim = [(c0 * r2, s0 * r2, zz * 1.04), (c1 * r2, s1 * r2, zz * 1.04), (0, 0, zz * 1.04)]
            col = (196, 198, 206) if i % 2 == 0 else (150, 152, 162)
            b.tri(*(rim if sgn > 0 else rim[::-1]), col)
    return b.build()
