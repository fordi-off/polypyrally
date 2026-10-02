"""Chase / hood / bumper cameras with smoothed yaw so the view swings behind the car through slides."""
import math
import numpy as np
from .mathx import look_at, perspective


class Camera:
    MODES = ('chase', 'far', 'hood', 'bumper')

    def __init__(self):
        self.mode = 0
        self.yaw = 0.0
        self.pos = np.zeros(3)
        self.target = np.zeros(3)
        self.fov = 62.0
        self.shake = 0.0
        self.first = True
        self.acc = 0.0
        self.prev_sp = 0.0
        self.rng = np.random.RandomState(3)

    def cycle(self):
        self.mode = (self.mode + 1) % len(self.MODES)
        self.first = True

    def update(self, dt, car, stage, aspect):
        fwd = car.forward
        head = math.atan2(fwd[2], fwd[0])
        sp = car.speed
        v = math.hypot(car.v[0], car.v[2])
        ang = head
        if v > 6.0:
            va = math.atan2(car.v[2], car.v[0])
            d = (va - head + math.pi) % math.tau - math.pi
            ang = head + d * 0.3
        if self.first:
            self.yaw = ang
        d = (ang - self.yaw + math.pi) % math.tau - math.pi
        self.yaw += d * (1 - math.exp(-dt * 3.2))
        cp = np.array(car.p)
        mode = self.MODES[self.mode]
        a_raw = (sp - self.prev_sp) / max(dt, 1e-4)
        self.prev_sp = sp
        if abs(a_raw) < 40:                                  # ignore teleports / resets
            self.acc += (a_raw - self.acc) * (1 - math.exp(-dt * 2.5))
        push = max(-3.0, min(6.0, self.acc))
        if mode in ('chase', 'far'):
            dist = (7.0 if mode == 'chase' else 11.0) + min(5.0, abs(sp) * 0.04) + push * 0.16
            hgt = (2.5 if mode == 'chase' else 4.3) + min(1.2, abs(sp) * 0.012)
            back = np.array([-math.cos(self.yaw), 0.0, -math.sin(self.yaw)])
            want = cp + back * dist + np.array([0, hgt, 0])
            if self.first:
                self.pos = want
            else:
                self.pos += (want - self.pos) * (1 - math.exp(-dt * 9.0))
            tgt = cp + np.array([math.cos(self.yaw), 0, math.sin(self.yaw)]) * 5.5 + np.array([0, 1.0, 0])
            self.target += (tgt - self.target) * (1 - math.exp(-dt * 12.0)) if not self.first else (tgt - self.target)
            lat = car.speed * car.w[1]                           # centripetal acceleration (+ when turning left)
            tilt = max(-0.07, min(0.07, lat * 0.0045))
            rgt = np.array([-math.sin(self.yaw), 0.0, math.cos(self.yaw)])
            up = np.array([0.0, 1.0, 0.0]) + rgt * tilt
        else:
            up = np.array(car.up)
            if mode == 'hood':
                eye = car.world(0.25, 0.62, 0.0)
                ahead = car.world(30.0, 0.45, 0.0)
            else:
                eye = car.world(1.9, -0.02, 0.0)
                ahead = car.world(30.0, 0.0, 0.0)
            self.pos = np.array(eye)
            self.target = np.array(ahead)
            up = np.array([0.0, 1.0, 0.0]) * 0.85 + up * 0.15
        # never dive under the terrain
        gy = stage.height(self.pos[0], self.pos[2]) + 0.9
        if self.pos[1] < gy and mode in ('chase', 'far'):
            self.pos[1] = gy
        # engine / road shake: grows with revs under load, speed and big hits
        amp = 0.004 + 0.012 * car.thr * min(1.0, car.rpm / 7000.0) + 0.012 * min(1.0, abs(sp) / 45.0) + min(0.06, car.hit * 2e-6)
        if mode in ('hood', 'bumper'):
            amp *= 1.6
        self.pos = self.pos + self.rng.randn(3) * amp
        self.first = False
        fov_t = 60.0 + min(16.0, abs(sp) * 0.2) + (8.0 if mode == 'hood' else 0.0) + push * 1.1
        self.fov += (fov_t - self.fov) * (1 - math.exp(-dt * 3.0))
        self.view = look_at(self.pos, self.target, up)
        self.proj = perspective(self.fov, aspect, 0.25, 3200.0)
        f = self.target - self.pos
        self.forward = f / (np.linalg.norm(f) + 1e-9)
