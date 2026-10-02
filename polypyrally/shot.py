"""Offscreen render for development: `python -m polypyrally --shot out.png`."""
import math
import numpy as np
import pygame
from .stage import Stage
from .gl_render import Renderer
from .frame import Frame
from .world import World
from .mathx import look_at, perspective, transform, rot_y, rot_z
from . import car_model


def make_ctx(size, samples=0):
    import moderngl
    ctx = moderngl.create_standalone_context(backend='egl')
    fbo = ctx.simple_framebuffer(size)
    return ctx, fbo


def render_still(path, size=(1280, 720), seed=3, s=300.0, msaa=4, cam=(8.0, 2.6, 0.0, 55.0), shadow=2048):
    ctx, fbo = make_ctx(size)
    r = Renderer(ctx, size, target=fbo, msaa=msaa, shadow_res=shadow)
    st = Stage(seed)
    world = World(st, r)
    i = int(s / st.ds)
    x, z, th = st.rx[i], st.rz[i], st.rth[i]
    y = st.height(x, z)
    world.update(x, z, force=True, focus=(x, z))
    car = r.mesh(car_model.body((214, 44, 40)))
    wh = r.mesh(car_model.wheel())
    fr = Frame()
    fr.time = 1.0
    cy = y + 0.62
    R = rot_y(-th)
    mdl = transform(R, (x, cy, z))
    fr.models.append((car, mdl, 0.9))
    for wx, wz in ((1.28, 1.0), (1.28, -1.0), (-1.32, 1.0), (-1.32, -1.0)):
        m = transform(R, (x, cy, z)) @ transform(np.eye(3), (wx, -0.2, wz))
        if wz < 0:
            m = m @ transform(rot_y(math.pi), (0, 0, 0))
        fr.models.append((wh, m, 0.0))
    dist, hgt, yaw, fov = cam
    a = th + math.pi + math.radians(yaw)
    back = np.array([math.cos(a), 0.0, math.sin(a)])
    eye = np.array([x, cy, z]) + back * dist + np.array([0, hgt, 0])
    fr.cam_pos = tuple(eye)
    fr.focus = (x, cy, z)
    fr.view = look_at(eye, (x, cy + 0.6, z))
    fr.proj = perspective(fov, size[0] / size[1], 0.3, 3000.0)
    f = np.array([x, cy + 0.6, z]) - eye
    fr.forward = tuple(f / np.linalg.norm(f))
    world.fill(fr)
    r.render(fr)
    raw = fbo.read(components=3)
    img = pygame.image.frombuffer(raw, size, 'RGB')
    pygame.image.save(pygame.transform.flip(img, False, True), path)
    return r, st


def drive_shot(a):
    """Offscreen full-game frame: the AI drives for a.secs seconds, then one frame (with HUD) is saved."""
    import os
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    pygame.init()
    pygame.display.set_mode((16, 16))
    size = tuple(int(v) for v in a.size.split('x'))
    ctx, fbo = make_ctx(size)
    from .game import Game
    from .ai import Driver
    r = Renderer(ctx, size, target=fbo, msaa=a.msaa, shadow_res=a.shadow)
    g = Game(r, seed=a.seed, audio=False, headless=True)
    g.show_fps = True
    d = Driver(g.stage)
    g.state = 'run'
    g.countdown = 0
    dt = 1 / 60
    keys = dict(gas=False, brake=False, left=False, right=False, hand=False)
    for i in range(int(a.secs * 60)):
        thr, brk, steer = d.drive(g.car, dt)
        keys.update(axis_gas=thr, axis_brake=brk, axis_steer=steer)
        g.update(dt, keys)
    hud = pygame.Surface(size, pygame.SRCALPHA)
    g.hud.draw(hud, g)
    g.update(dt, keys)
    g.world.update(g.car.p[0], g.car.p[2], force=True, focus=(g.car.p[0], g.car.p[2]))
    g.draw(hud)
    raw = fbo.read(components=3)
    img = pygame.image.frombuffer(raw, size, 'RGB')
    pygame.image.save(pygame.transform.flip(img, False, True), a.drive)
    print('speed %.0f km/h gear %d idx %d' % (g.car.kmh, g.car.gear, g.idx))


def bench(a):
    """CPU-side cost per frame: physics, world streaming, frame build + GL submission, HUD. (GPU time is not
    measured here: with a real GPU the CPU overlaps with it.)"""
    import os
    import time
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    pygame.init()
    pygame.display.set_mode((16, 16))
    size = tuple(int(v) for v in a.size.split('x'))
    ctx, fbo = make_ctx(size)
    from .game import Game
    from .ai import Driver
    r = Renderer(ctx, size, target=fbo, msaa=a.msaa, shadow_res=a.shadow)
    g = Game(r, seed=a.seed, audio=False, headless=True)
    d = Driver(g.stage)
    g.state = 'run'
    g.countdown = 0
    keys = dict(gas=False, brake=False, left=False, right=False, hand=False)
    hud = pygame.Surface(size, pygame.SRCALPHA)
    acc = [0.0] * 4
    pc = time.perf_counter
    n = 240
    for i in range(n + 30):
        t0 = pc()
        thr, brk, steer = d.drive(g.car, 1 / 60)
        keys.update(axis_gas=thr, axis_brake=brk, axis_steer=steer)
        g.update(1 / 60, keys)
        t1 = pc()
        g.world.update(g.cam.pos[0], g.cam.pos[2], focus=(g.car.p[0], g.car.p[2]))
        t2 = pc()
        hud.fill((0, 0, 0, 0))
        g.hud.draw(hud, g)
        t3 = pc()
        g.draw(hud)
        t4 = pc()
        if i >= 30:
            for k, v in enumerate((t1 - t0, t2 - t1, t3 - t2, t4 - t3)):
                acc[k] += v
    print('renderer: %s' % r.info)
    for name, v in zip(('physics + AI', 'world streaming', 'HUD (pygame)', 'frame build + submit'), acc):
        print('%-22s %6.2f ms' % (name, v / n * 1000))
    print('%-22s %6.2f ms' % ('CPU total', sum(acc) / n * 1000))
