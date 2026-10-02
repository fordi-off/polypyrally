"""A rally stage: a winding road with real elevation, a heightfield built around it and fast height queries
for the physics. The drawn terrain mesh and the physics use the very same triangulated grid."""
import math
import numpy as np
from .noise import fbm, vnoise, smoothstep, box_blur, hash2

GRID = 2.0                 # metres between height samples
ROAD_HALF = 3.4            # drivable gravel half-width
PAD = 340.0                # terrain margin around the road's bounding box


def _road_curvature(rng, n, ds):
    """Piecewise constant curvature (straights, sweepers, hairpins) smoothed into clothoid-like transitions."""
    k = np.zeros(n)
    i = 0
    side = 1.0 if rng.rand() > 0.5 else -1.0
    while i < n:
        r = rng.rand()
        if r < 0.28:                                   # straight
            ln = rng.uniform(50, 170)
            kk = 0.0
        else:
            if r < 0.55:                               # fast sweeper
                radius, ang = rng.uniform(110, 260), rng.uniform(0.3, 0.9)
            elif r < 0.88:                             # medium corner
                radius, ang = rng.uniform(45, 100), rng.uniform(0.6, 1.5)
            else:                                      # hairpin
                radius, ang = rng.uniform(16, 32), rng.uniform(2.3, 3.1)
            ln = radius * ang
            kk = side / radius
            side = -side if rng.rand() < 0.7 else side
        j = min(n, i + max(2, int(ln / ds)))
        k[i:j] = kk
        i = j
    w = max(3, int(26.0 / ds))
    ker = np.ones(w) / w
    for _ in range(2):
        k = np.convolve(np.pad(k, (w, w), mode='edge'), ker, mode='same')[w:-w]
    return k


class Stage:
    def __init__(self, seed=1, length=2800.0):
        self.seed = seed
        self.length = length
        self._make_road()
        self._make_terrain()

    # ------------------------------------------------------------------ road
    def _make_road(self):
        ds = 2.0
        for attempt in range(80):
            rng = np.random.RandomState(self.seed * 1000 + attempt)
            n = int(self.length / ds)
            k = _road_curvature(rng, n, ds)
            th = np.cumsum(k * ds) + rng.uniform(-0.3, 0.3)
            th -= th[0]
            x = np.concatenate([[0.0], np.cumsum(np.cos(th[:-1]) * ds)])
            z = np.concatenate([[0.0], np.cumsum(np.sin(th[:-1]) * ds)])
            # reject self-overlapping roads (samples far apart along the road but close in space)
            step = 4
            px, pz = x[::step], z[::step]
            idx = np.arange(len(px))
            dd = np.hypot(px[:, None] - px[None, :], pz[:, None] - pz[None, :])
            far = np.abs(idx[:, None] - idx[None, :]) * step * ds > 140.0
            if not np.any(dd[far] < 60.0):
                break
        self.n = n
        self.ds = ds
        self.rx, self.rz, self.rth = x, z, th
        s = np.arange(n) * ds
        self.rs = s
        # elevation profile: grade from smooth noise, plus a few crests to jump over
        grade = 0.055 * vnoise(s / 260.0, 0.5, self.seed) + 0.035 * vnoise(s / 90.0, 3.5, self.seed + 5) + 0.012
        grade -= np.mean(grade) * 0.6
        h = np.cumsum(grade * ds)
        for c in np.arange(180.0, self.length - 140.0, 230.0):
            c += 60.0 * vnoise(c, 9.0, self.seed + 7)
            h += 1.5 * np.exp(-((s - c) / 11.0) ** 2) * (0.7 + 0.5 * hash2(int(c), 5, self.seed))
        h += 40.0
        self.rh = h
        self.pitch = np.arctan2(np.gradient(h, ds), 1.0)

    # ------------------------------------------------------------------ terrain
    def _make_terrain(self):
        g = GRID
        self.ox = math.floor((self.rx.min() - PAD) / g) * g
        self.oz = math.floor((self.rz.min() - PAD) / g) * g
        self.NX = int((self.rx.max() + PAD - self.ox) / g) + 2
        self.NZ = int((self.rz.max() + PAD - self.oz) / g) + 2
        NX, NZ = self.NX, self.NZ
        X = self.ox + np.arange(NX) * g
        Z = self.oz + np.arange(NZ) * g
        XX, ZZ = np.meshgrid(X, Z)
        RW = 78.0                                    # distance-field reach
        D = np.full((NZ, NX), RW, dtype=np.float64)
        HR = np.zeros((NZ, NX))
        SN = np.zeros((NZ, NX))
        rad = int(RW / g) + 1
        for i in range(0, self.n):
            cx, cz = self.rx[i], self.rz[i]
            ci, cj = int((cx - self.ox) / g), int((cz - self.oz) / g)
            i0, i1 = max(0, ci - rad), min(NX, ci + rad + 1)
            j0, j1 = max(0, cj - rad), min(NZ, cj + rad + 1)
            dd = np.hypot(XX[j0:j1, i0:i1] - cx, ZZ[j0:j1, i0:i1] - cz)
            m = dd < D[j0:j1, i0:i1]
            D[j0:j1, i0:i1] = np.where(m, dd, D[j0:j1, i0:i1])
            HR[j0:j1, i0:i1] = np.where(m, self.rh[i], HR[j0:j1, i0:i1])
            SN[j0:j1, i0:i1] = np.where(m, self.rs[i], SN[j0:j1, i0:i1])
        self.D = D
        # smooth "road height" field so hairpin interiors don't cliff
        wt = 1.0 - smoothstep(10.0, RW, D)
        r = int(14 / g)
        num = box_blur(HR * wt, r) + 1e-3 * float(self.rh.mean())
        den = box_blur(wt, r) + 1e-3
        Rb = num / den
        # hills: large rolling terrain + detail, calmer near the road
        big = fbm(XX / 420.0, ZZ / 420.0, 4, seed=self.seed + 11) * 46.0
        mid = fbm(XX / 130.0, ZZ / 130.0, 4, seed=self.seed + 23) * 11.0
        small = fbm(XX / 28.0, ZZ / 28.0, 3, seed=self.seed + 37) * 1.5
        far = smoothstep(14.0, 70.0, D)
        rel = (big * 0.55 + mid) * far + small * (0.25 + 0.75 * far)
        base = Rb + rel * np.where(D >= RW - 1e-6, 1.0, 1.0)
        # beyond the field's reach the road no longer pins the height: ease onto the broader landscape
        free = smoothstep(RW * 0.5, RW, D)
        landscape = float(self.rh.mean()) + big + mid * 0.8
        base = base * (1 - free) + landscape * free
        # the road bed itself: exact elevation + a crown, and gentle verge shaping
        w_road = 1.0 - smoothstep(ROAD_HALF + 1.5, ROAD_HALF + 9.0, D)
        crown = -0.05 * np.clip(D / ROAD_HALF, 0, 1.4) ** 2
        ditch = -0.5 * np.exp(-((D - ROAD_HALF - 3.2) / 1.6) ** 2)
        road_h = HR + crown + ditch * (1 - w_road * 0.0)
        H = base * (1 - w_road) + road_h * w_road
        self.Hn = H.astype(np.float64)
        self.Dn = D
        self.SNn = SN
        # python lists for the scalar physics queries
        self.H = self.Hn.ravel().tolist()
        self.Dl = self.Dn.ravel().tolist()

    # ------------------------------------------------------------------ queries (pure python, hot path)
    def height_normal(self, x, z):
        g = GRID
        fx = (x - self.ox) / g
        fz = (z - self.oz) / g
        i, j = int(fx), int(fz)
        NX = self.NX
        if i < 0 or j < 0 or i >= NX - 1 or j >= self.NZ - 1:
            return 0.0, 0.0, 1.0, 0.0
        u, v = fx - i, fz - j
        H = self.H
        k = j * NX + i
        h00, h10, h01, h11 = H[k], H[k + 1], H[k + NX], H[k + NX + 1]
        if (i + j) & 1 == 0:
            if u >= v:
                h = h00 + (h10 - h00) * u + (h11 - h10) * v
                gx, gz = h10 - h00, h11 - h10
            else:
                h = h00 + (h11 - h01) * u + (h01 - h00) * v
                gx, gz = h11 - h01, h01 - h00
        else:
            if u + v <= 1.0:
                h = h00 + (h10 - h00) * u + (h01 - h00) * v
                gx, gz = h10 - h00, h01 - h00
            else:
                h = h11 + (h01 - h11) * (1 - u) + (h10 - h11) * (1 - v)
                gx, gz = h11 - h01, h11 - h10
        gx /= g
        gz /= g
        il = 1.0 / math.sqrt(gx * gx + 1.0 + gz * gz)
        return h, -gx * il, il, -gz * il

    def height(self, x, z):
        return self.height_normal(x, z)[0]

    def road_distance(self, x, z):
        g = GRID
        i = int((x - self.ox) / g + 0.5)
        j = int((z - self.oz) / g + 0.5)
        if i < 0 or j < 0 or i >= self.NX or j >= self.NZ:
            return 99.0
        return self.Dl[j * self.NX + i]

    def nearest(self, x, z, hint=None, window=60):
        """Index of the road sample closest to (x, z), searched near `hint` when given."""
        if hint is None:
            lo, hi = 0, self.n
        else:
            lo, hi = max(0, hint - window), min(self.n, hint + window)
        dx = self.rx[lo:hi] - x
        dz = self.rz[lo:hi] - z
        return lo + int(np.argmin(dx * dx + dz * dz))

    def heights_np(self, x, z):
        """Vectorised bilinear height lookup (scenery placement)."""
        g = GRID
        fx = np.clip((np.asarray(x) - self.ox) / g, 0, self.NX - 1.001)
        fz = np.clip((np.asarray(z) - self.oz) / g, 0, self.NZ - 1.001)
        i, j = fx.astype(int), fz.astype(int)
        u, v = fx - i, fz - j
        H = self.Hn
        return (H[j, i] * (1 - u) * (1 - v) + H[j, i + 1] * u * (1 - v) + H[j + 1, i] * (1 - u) * v + H[j + 1, i + 1] * u * v)

    def start_pose(self, s=14.0):
        i = max(1, min(self.n - 2, int(s / self.ds)))
        return self.rx[i], self.rh[i], self.rz[i], self.rth[i]
