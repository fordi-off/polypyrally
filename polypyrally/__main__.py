import argparse


def main():
    ap = argparse.ArgumentParser(description='PolyPyRally - low-poly 3D rally')
    ap.add_argument('--seed', type=int, default=3, help='stage seed')
    ap.add_argument('--size', default='1280x720', help='window size')
    ap.add_argument('--fullscreen', action='store_true')
    ap.add_argument('--vsync', action='store_true')
    ap.add_argument('--msaa', type=int, default=4, help='anti-aliasing samples (0 = off)')
    ap.add_argument('--shadow', type=int, default=2048, help='shadow map resolution')
    ap.add_argument('--mute', action='store_true')
    ap.add_argument('--fps', action='store_true', help='show the fps counter')
    ap.add_argument('--shot', metavar='PNG', help='render one offscreen frame (development)')
    ap.add_argument('--s', type=float, default=300.0, help='distance along the stage for --shot')
    ap.add_argument('--cam', default='8,2.6,0,55', help='dist,height,yaw_deg,fov for --shot')
    ap.add_argument('--drive', metavar='PNG', help='offscreen: let the AI drive for --secs seconds, save a frame (development)')
    ap.add_argument('--frames', type=int, default=0, help='quit after N frames (tests)')
    ap.add_argument('--bench', action='store_true', help='print the CPU cost per frame (offscreen) and exit')
    ap.add_argument('--secs', type=float, default=20.0)
    a = ap.parse_args()
    if a.shot:
        from .shot import render_still
        w, h = (int(v) for v in a.size.split('x'))
        render_still(a.shot, (w, h), a.seed, a.s, cam=tuple(float(v) for v in a.cam.split(',')))
        return
    if a.bench:
        from .shot import bench
        bench(a)
        return
    if a.drive:
        from .shot import drive_shot
        drive_shot(a)
        return
    from .app import run
    run(a)


if __name__ == '__main__':
    main()
