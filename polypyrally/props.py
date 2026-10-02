"""Start / finish arches and roadside spectators, merged into one static mesh."""
import math
import numpy as np
from .meshkit import MeshBuilder
from .noise import hash2
from .stage import ROAD_HALF


def _frame(st, s):
    i = int(max(1, min(st.n - 2, s / st.ds)))
    x, z, th, y = st.rx[i], st.rz[i], st.rth[i], st.rh[i]
    fx, fz = math.cos(th), math.sin(th)
    return x, y, z, fx, fz, -fz, fx            # position + forward + right (x,z)


def arch(b, st, s, banner, checker=False):
    x, y, z, fx, fz, rx, rz = _frame(st, s)
    half = ROAD_HALF + 1.6

    def P(side, lift, along=0.0):
        return (x + rx * side * half + fx * along, y + lift, z + rz * side * half + fz * along)
    h = 6.0
    for side in (-1, 1):
        base = P(side, 0.0)
        post = MeshBuilder()
        post.box(0, h / 2, 0, 0.5, h, 0.5, (210, 210, 218))
        b.extend(post, base[0], base[1], base[2])
    # beam + banner built in world space, rotated along the road
    def W(u, v, w):                          # u along road, v up, w lateral
        return (x + fx * u + rx * w, y + v, z + fz * u + rz * w)
    wid = half
    # beam
    for (u0, u1, v0, v1, col) in ((-0.3, 0.3, h - 1.3, h, banner),):
        b.quad(W(u0, v0, -wid), W(u0, v1, -wid), W(u0, v1, wid), W(u0, v0, wid), col)
        b.quad(W(u1, v0, wid), W(u1, v1, wid), W(u1, v1, -wid), W(u1, v0, -wid), col)
        b.quad(W(u0, v1, -wid), W(u1, v1, -wid), W(u1, v1, wid), W(u0, v1, wid), (240, 240, 245))
    if checker:
        n = 14
        for k in range(n):
            for r in range(2):
                col = (24, 24, 28) if (k + r) % 2 == 0 else (244, 244, 248)
                w0, w1 = -wid + 2 * wid * k / n, -wid + 2 * wid * (k + 1) / n
                v0, v1 = h - 1.3 + r * 0.65, h - 1.3 + (r + 1) * 0.65
                b.quad(W(-0.31, v0, w0), W(-0.31, v1, w0), W(-0.31, v1, w1), W(-0.31, v0, w1), col)
                b.quad(W(0.31, v0, w1), W(0.31, v1, w1), W(0.31, v1, w0), W(0.31, v0, w0), col)
    else:
        for u in (-0.31, 0.31):
            lo, hi = (-wid, wid)
            sgn = 1 if u > 0 else -1
            b.quad(W(u, h - 0.95, lo * 0.7 * sgn), W(u, h - 0.35, lo * 0.7 * sgn), W(u, h - 0.35, hi * 0.7 * sgn), W(u, h - 0.95, hi * 0.7 * sgn), (250, 250, 252), 0.0, 0.0)


def spectators(b, st, seed=2):
    """Little knots of people, mostly on the outside of the bends."""
    rng = np.random.RandomState(seed)
    cols = [(220, 60, 52), (60, 120, 220), (240, 200, 60), (80, 180, 110), (240, 240, 244), (200, 90, 200), (255, 140, 50)]
    k = np.gradient(np.unwrap(st.rth), st.ds)
    s = 140.0
    while s < st.length - 80:
        i = int(s / st.ds)
        x, y, z, fx, fz, rx, rz = _frame(st, s)
        side = -1 if k[i] > 0 else 1                       # outside of the bend
        if abs(k[i]) < 0.004:
            side = 1 if rng.rand() > 0.5 else -1
        off = ROAD_HALF + 4.6 + rng.rand() * 2.0
        cx, cz = x + rx * side * off, z + rz * side * off
        h = st.height(cx, cz)
        for j in range(rng.randint(4, 9)):
            px = cx + fx * (rng.rand() - 0.5) * 9 + rx * (rng.rand() - 0.5) * 3
            pz = cz + fz * (rng.rand() - 0.5) * 9 + rz * (rng.rand() - 0.5) * 3
            py = st.height(px, pz)
            c = cols[rng.randint(len(cols))]
            body = MeshBuilder()
            body.prism((0, 0, 0), 0.26, 0.2, 1.05, c, n=5)
            body.blob((0, 1.3, 0), 0.17, (232, 190, 150), detail=0)
            body.cone((0, 1.38, 0), 0.19, 0.2, cols[rng.randint(len(cols))], n=5)
            b.extend(body, px, py, pz)
        s += 70 + rng.rand() * 110


def build(st):
    b = MeshBuilder()
    arch(b, st, 12.0, (214, 44, 40))
    arch(b, st, st.length - 30.0, (28, 28, 34), checker=True)
    spectators(b, st)
    return b.build()
