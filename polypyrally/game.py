"""The running game: stage, car, camera, particles, HUD, audio and the input / state machine."""
import math
import os
import time
import numpy as np
import pygame

from .stage import Stage, ROAD_HALF
from .vehicle import Car
from .gl_render import Renderer
from .frame import Frame
from .world import World
from .camera import Camera
from .particles import Particles
from .hud import Hud
from .mathx import transform, rot_y, rot_z
from . import car_model, props
from .meshkit import lin


class Game:
    def __init__(self, renderer, seed=3, audio=True, headless=False):
        self.r = renderer
        self.headless = headless
        self.seed = seed
        self.show_fps = False
        self.fps = 60.0
        self.paint = (214, 44, 40)
        self.cam = Camera()
        self.parts = Particles()
        self.audio_enabled = audio
        self._dt = 1 / 60
        self.audio = None
        self.car_mesh = renderer.mesh(car_model.body(self.paint))
        self.wheel_mesh = renderer.mesh(car_model.wheel())
        self.wheel_fast = renderer.mesh(car_model.wheel_blur())
        self.load_stage(seed)

    # ------------------------------------------------------------------ stage lifecycle
    def load_stage(self, seed):
        self.seed = seed
        self.stage = Stage(seed)
        if hasattr(self, 'world'):
            self.world.release()
        self.world = World(self.stage, self.r)
        self.world.props = [(self.r.mesh(props.build(self.stage)), np.eye(4), 0.0)]
        self.hud = Hud(self.stage)
        self.car = Car(self.stage)
        if self.audio_enabled and self.audio is None:
            from .audio import Audio
            self.audio = Audio(self.car.spec)
        self.restart()

    def restart(self):
        st = self.stage
        x, y, z, th = st.start_pose(8.0)
        self.car.reset(x, st.height(x, z), z, th)
        for _ in range(90):
            self.car.control(1 / 60, 0, 0, 0, 0)
            self.car.step(1 / 60)
        self.state = 'countdown'
        self.countdown = 3.0
        self.time = 0.0
        self.progress = 0.0
        self.idx = 0
        self.finish_time = 0.0
        self.top_speed = 0.0
        self.toast = ''
        self.toast_t = 0.0
        self.parts.n = 0
        self.cam.first = True
        self.world.update(self.car.p[0], self.car.p[2], force=True, focus=(self.car.p[0], self.car.p[2]))

    def new_stage(self):
        self.load_stage(self.seed + 1)

    # ------------------------------------------------------------------ update
    def update(self, dt, keys):
        self._dt = dt
        car, st = self.car, self.stage
        thr = 1.0 if keys['gas'] else 0.0
        brk = 1.0 if keys['brake'] else 0.0
        steer = (1.0 if keys['right'] else 0.0) - (1.0 if keys['left'] else 0.0)
        if 'axis_steer' in keys:
            steer = keys['axis_steer']
        if 'axis_gas' in keys:
            thr = keys['axis_gas']
        if 'axis_brake' in keys:
            brk = keys['axis_brake']
        if self.state == 'pause':
            return
        hold = False
        if self.state == 'countdown':
            self.countdown -= dt
            thr, brk, steer = 0.0, 0.0, 0.0
            hold = True
            if self.countdown <= 0:
                self.state = 'run'
        elif self.state == 'run':
            self.time += dt
        elif self.state == 'finished':
            thr, brk = 0.0, 0.7 if car.speed > 2.0 else 0.0
            steer = 0.0
            hold = car.speed <= 2.0
        car.control(dt, thr, brk, steer, 1.0 if (keys['hand'] or hold) else 0.0, keys.get('up', False), keys.get('down', False),
                    allow_reverse=self.state == 'run')
        car.step(dt)
        self.idx = st.nearest(car.p[0], car.p[2], self.idx, 60)
        self.progress = self.idx * st.ds
        self.top_speed = max(self.top_speed, car.speed)
        if self.state == 'run' and self.idx >= st.n - 16:
            self.state = 'finished'
            self.finish_time = self.time
        # fell off the map / flipped for long -> bring the car back
        if car.p[1] < -50 or abs(car.p[0] - st.rx[self.idx]) > 900:
            self.reset_to_road('Out of bounds')
        self._fx(dt)
        self.parts.update(dt)
        if self.toast_t > 0:
            self.toast_t -= dt

    def reset_to_road(self, msg='Reset'):
        st, car = self.stage, self.car
        i = max(1, self.idx - 2)
        x, z, th = st.rx[i], st.rz[i], st.rth[i]
        sp = max(0.0, car.speed) * 0.0
        car.reset(x, st.height(x, z), z, th, sp)
        for _ in range(30):
            car.control(1 / 60, 0, 0, 0, 0)
            car.step(1 / 60)
        self.cam.first = True
        self.toast, self.toast_t = msg, 1.4

    def _fx(self, dt):
        car = self.car
        rnd = self.parts.rng
        sp = abs(car.speed)
        for wl in car.wheels:
            if not wl.contact or sp < 2.0:
                continue
            rate = (0.04 * sp + 0.6 * max(0.0, wl.slip - 0.9)) * 18 * dt
            n = int(rate) + (1 if rnd.rand() < rate - int(rate) else 0)
            if n:
                d = np.array(car.forward) * -sp * 0.15
                dust = (0.62, 0.52, 0.40) if wl.mu > 0.7 else (0.34, 0.40, 0.2)
                self.parts.emit((wl.cx, wl.cy + 0.1, wl.cz), d + np.array([0, 0.8, 0]), n, 1.6, 0.35, 1.0, dust, 0.5 if wl.slip > 1 else 0.3)
        if car.thr > 0.7 and sp < 10 and car.shift_t <= 0:
            pass

    # ------------------------------------------------------------------ draw
    def draw(self, ui=None):
        car = self.car
        r = self.r
        self.cam.update(self._dt, car, self.stage, r.W / r.H)
        self.world.update(self.cam.pos[0], self.cam.pos[2], focus=(car.p[0], car.p[2]))
        fr = Frame()
        fr.time = time.perf_counter() % 1000.0 if not self.headless else self.time
        fr.view, fr.proj = self.cam.view, self.cam.proj
        fr.cam_pos = tuple(self.cam.pos)
        fr.forward = tuple(self.cam.forward)
        fr.focus = tuple(car.p)
        self.world.fill(fr)
        R = np.array(car.R).reshape(3, 3)
        fr.models.append((self.car_mesh, transform(R, car.p), 0.9))
        Rt = R.T
        for wl in car.wheels:
            hub = np.array((wl.hx, wl.hy, wl.hz))
            # steer about the body's up axis, spin about the axle (body z); left wheels are mirrored
            S = R @ _roty_body(-wl.steer if wl.front else 0.0)
            M = transform(S @ rot_z(-wl.angle) @ (np.diag([1, 1, -1.0]) if wl.side < 0 else np.eye(3)), hub)
            fast = abs(wl.om) * (1 / 60.0) > 0.26            # > ~15 deg per frame: switch to the blurred wheel
            fr.models.append((self.wheel_fast if fast else self.wheel_mesh, M, 0.0))
        # brake lights are baked emissive; add dust
        right = np.cross(np.array(self.cam.forward), (0, 1, 0))
        right /= (np.linalg.norm(right) + 1e-9)
        upv = np.cross(right, np.array(self.cam.forward))
        fr.particles = self.parts.vertices(right, upv)
        fr.ui = ui
        r.render(fr)


def _roty_body(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
