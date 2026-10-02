"""Headless tests: python tests/test_rally.py"""
import math
import os
import sys

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np  # noqa: E402

from polypyrally.stage import Stage, GRID  # noqa: E402
from polypyrally.vehicle import Car  # noqa: E402
from polypyrally.ai import Driver  # noqa: E402
from polypyrally.terrain_mesh import TerrainMesher, CH  # noqa: E402

STAGE = Stage(3)


def test_stage_is_deterministic_and_road_is_continuous():
    other = Stage(3)
    assert np.allclose(STAGE.Hn, other.Hn)
    steps = np.hypot(np.diff(STAGE.rx), np.diff(STAGE.rz))
    assert steps.max() < STAGE.ds * 1.01 and steps.min() > STAGE.ds * 0.99
    assert np.abs(np.diff(STAGE.rh)).max() / STAGE.ds < 0.2            # grades stay drivable


def test_physics_height_matches_the_drawn_mesh():
    st = STAGE
    tm = TerrainMesher(st)
    v = tm.build(15, 10, 1)
    tri = v[:CH * CH * 6].reshape(-1, 3, 11)[:, :, :3]
    for k in (0, 17, 400, 1000):
        a, b, c = tri[k]
        cx, cz = (a[0] + b[0] + c[0]) / 3, (a[2] + b[2] + c[2]) / 3
        h = st.height(cx, cz)
        assert abs(h - (a[1] + b[1] + c[1]) / 3) < 1e-3, (k, h)


def settled_car():
    st = STAGE
    car = Car(st)
    x, y, z, th = st.start_pose(14.0)
    car.reset(x, st.height(x, z), z, th)
    for _ in range(180):
        car.control(1 / 60, 0, 0.0, 0, 1.0)             # handbrake on while it settles
        car.step(1 / 60)
    return car


def test_car_rests_on_its_springs():
    car = settled_car()
    load = sum(w.load for w in car.wheels)
    assert abs(load - car.spec.mass * 9.81) / (car.spec.mass * 9.81) < 0.08, load
    assert car.up[1] > 0.97 and abs(car.speed) < 0.2


def test_car_accelerates_brakes_and_steers():
    car = settled_car()
    for _ in range(60 * 6):
        car.control(1 / 60, 1.0, 0, 0, 0)
        car.step(1 / 60)
    assert car.kmh > 70 and car.gear >= 3
    v0 = car.speed
    h0 = car.heading
    for _ in range(60):
        car.control(1 / 60, 0.3, 0, 0.6, 0)
        car.step(1 / 60)
    assert abs((car.heading - h0 + math.pi) % math.tau - math.pi) > 0.15            # it turns
    for _ in range(60 * 6):
        car.control(1 / 60, 0, 1.0, 0, 0)
        car.step(1 / 60)
        if car.speed < 0.5:
            break
    assert car.speed < v0 * 0.5


def test_ai_drives_the_start_of_the_stage_upright():
    st = STAGE
    car = settled_car()
    d = Driver(st)
    for i in range(60 * 30):
        thr, brk, steer = d.drive(car, 1 / 60)
        car.control(1 / 60, thr, brk, steer, 0)
        car.step(1 / 60)
        assert car.up[1] > 0.3, i
    assert d.idx * st.ds > 250


def test_offscreen_render_is_lit_and_not_empty():
    try:
        import moderngl
        moderngl.create_standalone_context(backend='egl').release()
    except Exception:
        print('   (no offscreen OpenGL here - skipped)')
        return
    import pygame
    from polypyrally.shot import render_still
    path = os.path.join(os.path.dirname(__file__), '_still.png')
    render_still(path, (320, 180), 3, 300.0, msaa=0, shadow=512)
    img = pygame.image.load(path)
    os.remove(path)
    sky, ground = img.get_at((160, 10)), img.get_at((160, 170))
    assert tuple(sky)[:3] != tuple(ground)[:3]
    assert sum(sky[:3]) > 150 and sum(ground[:3]) > 80


def test_main_loop_runs_offscreen():
    try:
        import moderngl
        moderngl.create_standalone_context(backend='egl').release()
    except Exception:
        print('   (no offscreen OpenGL here - skipped)')
        return
    import argparse
    os.environ['POLYPYRALLY_OFFSCREEN'] = '1'
    from polypyrally import app
    app.run(argparse.Namespace(seed=3, size='320x180', fullscreen=False, vsync=False, msaa=0, shadow=512, mute=True, fps=True, frames=90))


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
