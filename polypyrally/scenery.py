"""Trees, bushes and rocks: a handful of low-poly prototypes, instanced over the terrain. Placement is
deterministic per terrain chunk so the world is the same every run."""
import math
import numpy as np
from .meshkit import MeshBuilder
from .noise import hash2, fbm, smoothstep
from .stage import GRID
from .terrain_mesh import CH, CHUNK_M

PINE_G = (44, 84, 56)
PINE_G2 = (36, 70, 50)
BARK = (92, 66, 46)


def pine(variant):
    b = MeshBuilder()
    rnd = np.random.RandomState(100 + variant)
    h = 1.0
    b.prism((0, 0, 0), 0.22, 0.14, 1.6, BARK, n=5)
    tiers = 5 + variant % 2
    y = 1.1
    r = 2.0 + 0.25 * variant
    for t in range(tiers):
        hh = 2.5 - 0.15 * t
        col = (PINE_G[0] + int(rnd.randint(-6, 8)), PINE_G[1] + int(rnd.randint(-8, 10)), PINE_G[2] + int(rnd.randint(-6, 8)))
        col2 = (col[0] - 10, col[1] - 14, col[2] - 8)
        b.cone((0, y, 0), r, hh, col, n=7, color2=col2, rot=rnd.rand())
        y += hh * 0.52
        r *= 0.8
    return b.build()


def round_tree(variant):
    b = MeshBuilder()
    b.prism((0, 0, 0), 0.3, 0.2, 2.4, BARK, n=5)
    g = [(86, 128, 54), (74, 116, 60), (98, 134, 50)][variant % 3]
    g2 = (g[0] - 16, g[1] - 20, g[2] - 12)
    b.blob((0, 3.7, 0), 2.1, g, detail=1, squash=0.9, jitter=0.12, seed=variant, color2=g2)
    b.blob((1.2, 3.0, 0.5), 1.4, g, detail=1, squash=0.85, jitter=0.15, seed=variant + 5, color2=g2)
    b.blob((-1.0, 3.2, -0.7), 1.5, g2, detail=1, squash=0.85, jitter=0.15, seed=variant + 9, color2=g)
    return b.build()


def bush():
    b = MeshBuilder()
    g, g2 = (92, 120, 52), (72, 100, 46)
    b.blob((0, 0.35, 0), 0.7, g, detail=1, squash=0.7, jitter=0.18, seed=2, color2=g2)
    b.blob((0.6, 0.25, 0.3), 0.5, g2, detail=1, squash=0.7, jitter=0.2, seed=4, color2=g)
    return b.build()


def rock(variant):
    b = MeshBuilder()
    c, c2 = [((128, 122, 116), (104, 98, 96)), ((142, 124, 104), (112, 98, 84))][variant % 2]
    b.blob((0, 0.35, 0), 0.9, c, detail=1, squash=0.62, jitter=0.28, seed=20 + variant, color2=c2)
    return b.build()


def pine_lo():
    b = MeshBuilder()
    b.prism((0, 0, 0), 0.22, 0.14, 1.2, BARK, n=4, cap=False)
    y, r = 1.0, 2.1
    for t in range(3):
        col = (PINE_G[0] + 4 * t, PINE_G[1] + 6 * t, PINE_G[2] + 2 * t)
        b.cone((0, y, 0), r, 3.6, col, n=5, color2=(col[0] - 10, col[1] - 14, col[2] - 8))
        y += 2.4
        r *= 0.72
    return b.build()


def round_lo():
    b = MeshBuilder()
    b.prism((0, 0, 0), 0.3, 0.2, 2.4, BARK, n=4, cap=False)
    b.blob((0, 3.5, 0), 2.3, (84, 124, 54), detail=0, squash=0.9, jitter=0.1, seed=3, color2=(68, 104, 46))
    return b.build()


def rock_lo():
    b = MeshBuilder()
    b.blob((0, 0.35, 0), 0.9, (128, 122, 116), detail=0, squash=0.62, jitter=0.25, seed=21, color2=(104, 98, 96))
    return b.build()


LOD_OF = {'pine0': 'pine_lo', 'pine1': 'pine_lo', 'pine2': 'pine_lo', 'round0': 'round_lo', 'round1': 'round_lo',
          'bush': 'bush', 'rock0': 'rock_lo', 'rock1': 'rock_lo'}
PROTOS = None


def prototypes():
    global PROTOS
    if PROTOS is None:
        PROTOS = {'pine0': pine(0), 'pine1': pine(1), 'pine2': pine(2), 'round0': round_tree(0), 'round1': round_tree(1),
                  'bush': bush(), 'rock0': rock(0), 'rock1': rock(1), 'pine_lo': pine_lo(), 'round_lo': round_lo(), 'rock_lo': rock_lo()}
    return PROTOS


def place(stage, ci, cj):
    """Instances for one terrain chunk: {proto name: float32 (n, 7) [x y z yaw scale bright warm]}."""
    x0 = stage.ox + ci * CH * GRID
    z0 = stage.oz + cj * CH * GRID
    sp = 6.0
    n = int(CHUNK_M / sp)
    a, b = np.meshgrid(np.arange(n), np.arange(n))
    gx = (x0 / sp + a).astype(np.int64)
    gz = (z0 / sp + b).astype(np.int64)
    jx, jz = hash2(gx, gz, 1), hash2(gx, gz, 2)
    x = ((gx + jx) * sp).ravel()
    z = ((gz + jz) * sp).ravel()
    r = hash2(gx, gz, 3).ravel()
    r2 = hash2(gx, gz, 4).ravel()
    r3 = hash2(gx, gz, 6).ravel()
    inside = (x >= stage.ox + 4) & (z >= stage.oz + 4) & (x < stage.ox + (stage.NX - 3) * GRID) & (z < stage.oz + (stage.NZ - 3) * GRID)
    x, z, r, r2, r3 = x[inside], z[inside], r[inside], r2[inside], r3[inside]
    if len(x) == 0:
        return {}
    y = stage.heights_np(x, z)
    ii = np.clip(((x - stage.ox) / GRID + 0.5).astype(int), 0, stage.NX - 1)
    jj = np.clip(((z - stage.oz) / GRID + 0.5).astype(int), 0, stage.NZ - 1)
    d = stage.Dn[jj, ii]
    hx = stage.heights_np(x + 2.0, z) - stage.heights_np(x - 2.0, z)
    hz = stage.heights_np(x, z + 2.0) - stage.heights_np(x, z - 2.0)
    slope = np.hypot(hx, hz) / 4.0
    forest = fbm(x / 110.0, z / 110.0, 3, seed=71)
    dens = np.clip(0.52 + forest * 1.5, 0.05, 0.95)
    alt = np.clip((y - 78.0) / 40.0, 0, 1)
    dens *= 1.0 - alt * 0.85
    flat = slope < 0.6
    out = {}
    tree_ok = flat & (d > 10.0) & (r < dens)
    kind = r2
    sel = tree_ok & (kind < 0.62)
    for name, lo, hi in (('pine0', 0.0, 0.2), ('pine1', 0.2, 0.4), ('pine2', 0.4, 0.62)):
        m = sel & (kind >= lo) & (kind < hi)
        out[name] = _inst(x, y, z, m, r3, 0.75, 1.5, 0.0)
    sel = tree_ok & (kind >= 0.62) & (kind < 0.84)
    out['round0'] = _inst(x, y, z, sel & (r3 < 0.5), r3, 0.8, 1.3, 0.35)
    out['round1'] = _inst(x, y, z, sel & (r3 >= 0.5), r3, 0.8, 1.3, 0.35)
    # roadside bushes and rocks
    near = (d > 5.4) & (d < 22.0) & (slope < 0.9)
    out['bush'] = _inst(x, y, z, near & (r < 0.22) & (kind > 0.3), r3, 0.7, 1.4, 0.3)
    out['rock0'] = _inst(x, y, z, (d > 5.0) & (slope < 1.2) & (r > 0.9) & (kind < 0.5), r3, 0.6, 2.2, 0.0)
    out['rock1'] = _inst(x, y, z, (d > 5.0) & (slope < 1.2) & (r > 0.9) & (kind >= 0.5), r3, 0.6, 2.2, 0.0)
    return {k: v for k, v in out.items() if len(v)}


def _inst(x, y, z, mask, r, smin, smax, warm):
    if not mask.any():
        return np.zeros((0, 7), np.float32)
    n = int(mask.sum())
    rr = hash2((x[mask] * 7.3).astype(np.int64), (z[mask] * 3.1).astype(np.int64), 9)
    yaw = rr * math.tau
    sc = smin + (smax - smin) * hash2((z[mask] * 5.7).astype(np.int64), (x[mask] * 2.9).astype(np.int64), 10) ** 1.5
    br = hash2((x[mask] * 1.7).astype(np.int64), (z[mask] * 4.3).astype(np.int64), 11)
    wm = np.where(rr > 0.82, warm, 0.0) if warm > 0 else np.zeros(n)
    return np.stack([x[mask], y[mask] - 0.1, z[mask], yaw, sc, br, wm], axis=1).astype(np.float32)
