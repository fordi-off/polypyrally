"""Streams the stage around the camera: terrain chunk meshes (3 LODs), instanced scenery and props."""
import math
import time
import numpy as np
from .stage import GRID
from .terrain_mesh import TerrainMesher, CH, CHUNK_M
from . import scenery


def lod_for(dist):
    return 1 if dist < 150 else (2 if dist < 330 else 4)


INST_DIST = 470.0
NEAR_DIST = 170.0
SHADOW_DIST = 250.0


class World:
    def __init__(self, stage, renderer, view_dist=720.0):
        self.st = stage
        self.r = renderer
        self.mesher = TerrainMesher(stage)
        self.nci, self.ncj = self.mesher.chunk_range()
        self.view_dist = view_dist
        self.terr = {}                  # (ci, cj) -> (lod, GLMesh)
        self.inst_cpu = {}              # (ci, cj) -> {proto name: ndarray}
        self.protos = {k: renderer.mesh(v) for k, v in scenery.prototypes().items()}
        self.props = []                 # (GLMesh, model, gloss)
        self._sig = None
        self.inst_main = []             # (GLMesh, GLInstances)
        self.inst_shadow = []

    def chunk_centre(self, ci, cj):
        st = self.st
        return st.ox + (ci * CH + CH / 2) * GRID, st.oz + (cj * CH + CH / 2) * GRID

    def update(self, cam_x, cam_z, budget=0.012, force=False, focus=None):
        """Builds / refreshes chunks near the camera within a time budget; drops far ones."""
        t0 = time.perf_counter()
        st = self.st
        ci0 = int((cam_x - st.ox) // CHUNK_M)
        cj0 = int((cam_z - st.oz) // CHUNK_M)
        rad = int(self.view_dist // CHUNK_M) + 1
        wants = []
        for dj in range(-rad, rad + 1):
            for di in range(-rad, rad + 1):
                ci, cj = ci0 + di, cj0 + dj
                if ci < 0 or cj < 0 or ci >= self.nci or cj >= self.ncj:
                    continue
                mx, mz = self.chunk_centre(ci, cj)
                d = math.hypot(mx - cam_x, mz - cam_z)
                if d > self.view_dist + CHUNK_M * 0.5:
                    continue
                wants.append((d, ci, cj))
        wants.sort()
        for d, ci, cj in wants:
            lod = lod_for(d)
            cur = self.terr.get((ci, cj))
            if cur is None or cur[0] != lod:
                if not force and time.perf_counter() - t0 > budget and cur is not None:
                    continue
                if not force and time.perf_counter() - t0 > budget * 3:
                    break
                data = self.mesher.build(ci, cj, lod)
                if cur is not None:
                    cur[1].release()
                if data is None or len(data) == 0:
                    self.terr[(ci, cj)] = (lod, self.r.mesh(np.zeros((3, 11), np.float32)))
                else:
                    self.terr[(ci, cj)] = (lod, self.r.mesh(data))
            if d < INST_DIST and (ci, cj) not in self.inst_cpu:
                if not force and time.perf_counter() - t0 > budget * 2:
                    continue
                self.inst_cpu[(ci, cj)] = scenery.place(st, ci, cj)
        keep = {(ci, cj) for d, ci, cj in wants}
        for k in [k for k in self.terr if k not in keep]:
            self.terr.pop(k)[1].release()
        for k in [k for k in self.inst_cpu if k not in keep or self._far(k, cam_x, cam_z)]:
            del self.inst_cpu[k]
        self._batch_instances(cam_x, cam_z, focus if focus is not None else (cam_x, cam_z))

    def _far(self, k, cx, cz):
        mx, mz = self.chunk_centre(*k)
        return math.hypot(mx - cx, mz - cz) > INST_DIST + 90

    def _batch_instances(self, cam_x, cam_z, focus):
        """One instance buffer per prototype (near detail / far detail / shadow subset): a handful of draw calls."""
        near, far, shad = [], [], []
        for k in self.inst_cpu:
            mx, mz = self.chunk_centre(*k)
            d = math.hypot(mx - cam_x, mz - cam_z)
            (near if d < NEAR_DIST else far).append(k)
            if math.hypot(mx - focus[0], mz - focus[1]) < SHADOW_DIST:
                shad.append(k)
        sig = (frozenset(near), frozenset(far), frozenset(shad))
        if sig == self._sig:
            return
        self._sig = sig
        for _, ib in self.inst_main + self.inst_shadow:
            ib.release()
        self.inst_main, self.inst_shadow = [], []
        proto_far = scenery.LOD_OF

        def gather(keys, lod):
            groups = {}
            for k in keys:
                for name, arr in self.inst_cpu[k].items():
                    groups.setdefault(proto_far[name] if lod else name, []).append(arr)
            return groups
        for lod, keys in ((False, near), (True, far)):
            for name, arrs in gather(keys, lod).items():
                self.inst_main.append((self.protos[name], self.r.instances(np.concatenate(arrs))))
        for name, arrs in gather([k for k in shad if k in set(near) or True], False).items():
            self.inst_shadow.append((self.protos[name], self.r.instances(np.concatenate(arrs))))

    def fill(self, fr):
        rad = CHUNK_M * 0.75
        for (ci, cj), (lod, mesh) in self.terr.items():
            mx, mz = self.chunk_centre(ci, cj)
            fr.terrain.append((mesh, mx, mz, rad))
        fr.inst_main = list(self.inst_main)
        fr.inst_shadow = list(self.inst_shadow)
        fr.models.extend(self.props)

    def release(self):
        for _, mesh in self.terr.values():
            mesh.release()
        for _, ib in self.inst_main + self.inst_shadow:
            ib.release()
        for m, _, _ in self.props:
            m.release()
        self.terr, self.inst_cpu, self.props, self.inst_main, self.inst_shadow = {}, {}, [], [], []
        self._sig = None
