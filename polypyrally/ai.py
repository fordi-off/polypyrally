"""A simple pace-note style driver: pure-pursuit steering along the road centreline, speed set from the curvature
ahead. Used for tests, for the ghost / rival car and for the menu attract mode."""
import math
import numpy as np


class Driver:
    def __init__(self, stage, grip=5.6, boost=1.0):
        self.st = stage
        self.idx = 0
        self.grip = grip * boost
        k = np.gradient(np.unwrap(stage.rth), stage.ds)
        w = 7
        self.curv = np.convolve(np.pad(np.abs(k), (w, w), mode='edge'), np.ones(2 * w + 1) / (2 * w + 1), mode='same')[w:-w]

    def drive(self, car, dt):
        car.auto = True
        st = self.st
        px, pz = car.p[0], car.p[2]
        self.idx = st.nearest(px, pz, self.idx, 40)
        sp = max(0.0, car.speed)
        look = 7.0 + sp * 0.55
        j = min(st.n - 1, self.idx + int(look / st.ds))
        tx, tz = st.rx[j], st.rz[j]
        fwd = car.forward
        ang = math.atan2(tz - pz, tx - px) - math.atan2(fwd[2], fwd[0])
        ang = (ang + math.pi) % math.tau - math.pi
        # also counter a sliding car
        vx, vz = car.v[0], car.v[2]
        slide = 0.0
        if sp > 4.0:
            sa = math.atan2(vz, vx) - math.atan2(fwd[2], fwd[0])
            slide = (sa + math.pi) % math.tau - math.pi
        steer = max(-1.0, min(1.0, ang / max(0.12, car.spec.steer_max * (1 - sp / 80.0)) * 1.1 + slide * 0.8))
        # target speed from the sharpest curvature in the next stretch
        ahead = int((14.0 + sp * 3.6) / st.ds)
        kmax = float(self.curv[self.idx:min(st.n, self.idx + ahead)].max()) if self.idx < st.n - 1 else 0.0
        vt = min(75.0, math.sqrt(self.grip / max(kmax, 1e-4)))
        if self.idx > st.n - 40:
            vt = 8.0
        err = vt - sp
        thr = max(0.0, min(1.0, err * 0.3 + 0.1))
        brk = max(0.0, min(1.0, -err * 0.3 - 0.35))
        if brk > 0:
            thr = 0.0
        return thr, brk, steer
