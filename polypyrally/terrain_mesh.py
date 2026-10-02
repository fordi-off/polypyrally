"""Builds the flat-shaded terrain mesh chunk by chunk (numpy), with three levels of detail and skirts that hide
the cracks between them. Full-detail triangles use exactly the physics grid's triangulation."""
import numpy as np
from .noise import hash2, fbm, smoothstep, box_blur
from .stage import GRID, ROAD_HALF

CH = 32                                    # cells per chunk side (64 m)
CHUNK_M = CH * GRID


def srgb(c):
    return np.array(c, dtype=np.float64) / 255.0


# palette (sRGB): golden-hour friendly, slightly desaturated greens and warm dirt
GRAVEL = srgb((150, 130, 104))
GRAVEL_TRACK = srgb((124, 106, 86))
SHOULDER = srgb((134, 106, 74))
GRASS_A = srgb((82, 112, 48))
GRASS_B = srgb((114, 126, 54))
DRY = srgb((158, 142, 84))
ROCK = srgb((126, 120, 116))
ROCK_WARM = srgb((148, 122, 98))
SOIL = srgb((110, 84, 58))


class TerrainMesher:
    def __init__(self, stage):
        self.st = stage
        H = stage.Hn
        # concavity -> ambient occlusion in valleys / cuttings
        conc = box_blur(H, 3) - H
        self.ao = np.clip(1.0 - np.clip(conc * 0.11, 0.0, 0.45), 0.5, 1.0).astype(np.float32)
        self.slopen = None

    def chunk_range(self):
        st = self.st
        return (st.NX - 1) // CH + 1, (st.NZ - 1) // CH + 1

    def build(self, ci, cj, step):
        """Returns float32 (N, 11) vertices for chunk (ci, cj) at LOD `step` (1, 2 or 4)."""
        st = self.st
        s = step
        m = CH // s
        i0, j0 = ci * CH, cj * CH
        NX, NZ = st.NX, st.NZ
        ii = i0 + s * np.arange(m)
        jj = j0 + s * np.arange(m)
        ii = ii[ii + s < NX]
        jj = jj[jj + s < NZ]
        if len(ii) == 0 or len(jj) == 0:
            return None
        I, J = np.meshgrid(ii, jj)
        H, D, AO = st.Hn, st.Dn, self.ao
        g = GRID

        def corner(di, dj):
            return (st.ox + (I + di * s) * g, H[J + dj * s, I + di * s], st.oz + (J + dj * s) * g, D[J + dj * s, I + di * s], AO[J + dj * s, I + di * s])
        c00, c10, c01, c11 = corner(0, 0), corner(1, 0), corner(0, 1), corner(1, 1)
        par = ((I // s + J // s) & 1) == 0
        # triangle corner tuples per cell: T1, T2 (each three corners)
        T1 = [tuple(np.where(par, a, b) for a, b in zip(c0, c1)) for c0, c1 in ((c00, c00), (c10, c10), (c11, c11))]
        # even: T1=(00,10,11) T2=(00,11,01)      odd: T1=(00,10,01) T2=(10,11,01)
        T1 = [c00, c10, tuple(np.where(par, a, b) for a, b in zip(c11, c01))]
        T2 = [tuple(np.where(par, a, b) for a, b in zip(c00, c10)), tuple(np.where(par, a, b) for a, b in zip(c11, c11)), c01]
        tris = []
        for k, T in enumerate((T1, T2)):
            P = [np.stack([c[0], c[1], c[2]], axis=-1) for c in T]               # (m, m, 3)
            Dd = (T[0][3] + T[1][3] + T[2][3]) / 3.0
            ao = (T[0][4] + T[1][4] + T[2][4]) / 3.0
            n = np.cross(P[1] - P[0], P[2] - P[0])
            flip = n[..., 1] < 0
            n = np.where(flip[..., None], -n, n)
            nn = n / np.linalg.norm(n, axis=-1, keepdims=True)
            cx = (P[0][..., 0] + P[1][..., 0] + P[2][..., 0]) / 3.0
            cz = (P[0][..., 2] + P[1][..., 2] + P[2][..., 2]) / 3.0
            cy = (P[0][..., 1] + P[1][..., 1] + P[2][..., 1]) / 3.0
            col = self._colour(cx, cy, cz, Dd, nn[..., 1], I * 2 + k, J)
            # order vertices so the normal faces up
            a = P[0]
            b = np.where(flip[..., None], P[2], P[1])
            c = np.where(flip[..., None], P[1], P[2])
            tris.append((a, b, c, nn, col, ao))
        out = []
        for a, b, c, nn, col, ao in tris:
            n_ = a.shape[0] * a.shape[1]
            v = np.zeros((n_, 3, 11), dtype=np.float32)
            for t, p in enumerate((a, b, c)):
                v[:, t, 0:3] = p.reshape(-1, 3)
                v[:, t, 3:6] = nn.reshape(-1, 3)
                v[:, t, 6:9] = col.reshape(-1, 3)
                v[:, t, 9] = ao.reshape(-1)
            out.append(v.reshape(-1, 11))
        verts = np.concatenate(out)
        # skirts: vertical fringes on the four chunk borders
        sk = self._skirts(I, J, s, tris, ii, jj, par)
        if sk is not None:
            verts = np.concatenate([verts, sk])
        return verts

    def _colour(self, x, y, z, d, ny, hx, hz):
        shape = x.shape
        jit = 0.90 + 0.2 * hash2(hx, hz, 5)
        # base landscape colour
        nz = fbm(x / 55.0, z / 55.0, 3, seed=3)
        nz2 = fbm(x / 17.0, z / 17.0, 2, seed=8)
        t = np.clip(0.5 + 0.5 * nz * 1.6, 0, 1)[..., None]
        grass = GRASS_A * (1 - t) + GRASS_B * t
        dry = np.clip((nz2 * 1.8 - 0.1), 0, 1)[..., None] * 0.55
        grass = grass * (1 - dry) + DRY * dry
        altitude = np.clip((y - 60.0) / 60.0, 0, 1)[..., None]
        grass = grass * (1 - altitude * 0.5) + ROCK * altitude * 0.5
        steep = smoothstep(0.86, 0.66, ny)[..., None]
        rock = ROCK * 0.6 + ROCK_WARM * 0.4
        col = grass * (1 - steep) + rock * steep
        # verge soil near the road, then shoulder, then the gravel itself
        wsoil = (1 - smoothstep(ROAD_HALF + 1.5, ROAD_HALF + 7.0, d))[..., None]
        col = col * (1 - wsoil * 0.8) + SOIL * wsoil * 0.8
        wsh = (1 - smoothstep(ROAD_HALF - 0.3, ROAD_HALF + 2.2, d))[..., None]
        col = col * (1 - wsh) + SHOULDER * wsh
        wrd = (1 - smoothstep(ROAD_HALF - 1.6, ROAD_HALF - 0.1, d))[..., None]
        track = (np.exp(-((d - 1.15) / 0.5) ** 2))[..., None]
        road = GRAVEL * (1 - track * 0.45) + GRAVEL_TRACK * track * 0.45
        col = col * (1 - wrd) + road * wrd
        col = np.clip(col * jit[..., None], 0, 1)
        return (col ** 2.2).reshape(shape + (3,))

    def _skirts(self, I, J, s, tris, ii, jj, par):
        st = self.st
        drop = 5.0 * s
        rows = []
        (a1, b1, c1, n1, col1, ao1), (a2, b2, c2, n2, col2, ao2) = tris
        m, n = I.shape

        def add(p0, p1, normal, col, ao):
            lo0 = p0 - np.array([0, drop, 0])
            lo1 = p1 - np.array([0, drop, 0])
            v = np.zeros((6, 11), dtype=np.float32)
            for t, p in enumerate((p0, lo0, lo1, p0, lo1, p1)):
                v[t, 0:3] = p
                v[t, 3:6] = normal
                v[t, 6:9] = col
                v[t, 9] = ao
            rows.append(v)

        def vert(i, j):
            return np.array([st.ox + i * GRID, st.Hn[j, i], st.oz + j * GRID])
        # border vertex strips (with the colour of the neighbouring face)
        col_of = lambda r, c: col1[r, c]
        for c in range(n):                                  # north edge (z min)
            add(vert(ii[c], jj[0]), vert(ii[c] + s, jj[0]), (0, 0, -1), col_of(0, c), 0.6)
        for c in range(n):                                  # south edge (z max)
            add(vert(ii[c] + s, jj[-1] + s), vert(ii[c], jj[-1] + s), (0, 0, 1), col_of(m - 1, c), 0.6)
        for r in range(m):                                  # west edge
            add(vert(ii[0], jj[r] + s), vert(ii[0], jj[r]), (-1, 0, 0), col_of(r, 0), 0.6)
        for r in range(m):                                  # east edge
            add(vert(ii[-1] + s, jj[r]), vert(ii[-1] + s, jj[r] + s), (1, 0, 0), col_of(r, n - 1), 0.6)
        return np.concatenate(rows) if rows else None
