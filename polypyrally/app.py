"""Window, input and the main loop."""
import os
import sys
import time
import pygame

from .game import Game


def _make_dpi_aware():
    if sys.platform != 'win32':
        return
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (AttributeError, OSError):
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def open_window(size, msaa_note=None, vsync=False, fullscreen=False):
    import moderngl
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 0)
    flags = pygame.OPENGL | pygame.DOUBLEBUF
    if fullscreen:
        flags |= pygame.FULLSCREEN
        size = (0, 0)
    else:
        flags |= pygame.RESIZABLE
    pygame.display.set_mode(size, flags, vsync=1 if vsync else 0)
    return moderngl.create_context()


KEYS = {
    'gas': (pygame.K_w, pygame.K_UP), 'brake': (pygame.K_s, pygame.K_DOWN), 'left': (pygame.K_a, pygame.K_LEFT),
    'right': (pygame.K_d, pygame.K_RIGHT), 'hand': (pygame.K_SPACE,),
}


def run(args):
    _make_dpi_aware()
    pygame.init()
    pygame.font.init()
    pygame.display.set_caption('PolyPyRally')
    size = tuple(int(v) for v in args.size.split('x'))
    target = None
    if os.environ.get('POLYPYRALLY_OFFSCREEN'):          # tests: real loop, EGL context, no window system
        from .shot import make_ctx
        pygame.display.set_mode(size)
        ctx, target = make_ctx(size)
    else:
        ctx = open_window(size, vsync=args.vsync, fullscreen=args.fullscreen)
    W, H = pygame.display.get_window_size()
    from .gl_render import Renderer
    rend = Renderer(ctx, (W, H), target=target or ctx.screen, msaa=args.msaa, shadow_res=args.shadow)
    game = Game(rend, seed=args.seed, audio=not args.mute)
    game.show_fps = args.fps
    clock = time.perf_counter
    last = clock()
    hud_surf = None
    pygame.mouse.set_visible(False)
    running = True
    frames = 0
    pressed = {}
    fullscreen = args.fullscreen
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                k = e.key
                if game.state == 'pause':
                    if k == pygame.K_ESCAPE:
                        game.state = game._resume_state
                    elif k == pygame.K_RETURN:
                        game.restart()
                    elif k == pygame.K_n:
                        game.new_stage()
                    elif k == pygame.K_f:
                        game.show_fps = not game.show_fps
                    elif k == pygame.K_q:
                        running = False
                    continue
                if k == pygame.K_ESCAPE:
                    game._resume_state = game.state
                    game.state = 'pause'
                    pygame.mouse.set_visible(True)
                elif k == pygame.K_r and game.state != 'finished':
                    game.reset_to_road()
                elif k == pygame.K_c:
                    game.cam.cycle()
                elif k == pygame.K_RETURN and game.state == 'finished':
                    game.restart()
                elif k == pygame.K_n and game.state == 'finished':
                    game.new_stage()
                elif k == pygame.K_t:
                    game.car.auto = not game.car.auto
                    game.toast, game.toast_t = ('Automatic gearbox' if game.car.auto else 'Manual gearbox'), 1.2
                elif k == pygame.K_F3:
                    game.show_fps = not game.show_fps
                elif k == pygame.K_F12:
                    _screenshot(game, rend, args)
                elif k == pygame.K_F11:
                    fullscreen = not fullscreen
                    ctx = open_window(size, vsync=args.vsync, fullscreen=fullscreen)
                    raise SystemExit('Switching display modes needs a restart of the renderer - use --fullscreen')
                elif k == pygame.K_e:
                    pressed['up'] = True
                elif k == pygame.K_q:
                    pressed['down'] = True
        k = pygame.key.get_pressed()
        keys = {n: any(k[c] for c in codes) for n, codes in KEYS.items()}
        if pressed.pop('up', False):
            keys['up'] = True
        if pressed.pop('down', False):
            keys['down'] = True
        now = clock()
        dt = min(now - last, 1 / 20.0)
        last = now
        game.fps += (1.0 / max(dt, 1e-4) - game.fps) * 0.05
        game.update(dt, keys)
        if game.audio:
            game.audio.update(game.car)
        W, H = pygame.display.get_window_size()
        rend.resize((W, H))
        hw, hh = min(W, 1600), min(H, int(1600 * H / W))
        if hud_surf is None or hud_surf.get_size() != (hw, hh):
            hud_surf = pygame.Surface((hw, hh), pygame.SRCALPHA)
        hud_surf.fill((0, 0, 0, 0))
        game.hud.draw(hud_surf, game)
        game.draw(hud_surf)
        pygame.display.flip()
        frames += 1
        if args.frames and frames >= args.frames:
            running = False
    pygame.quit()


def _screenshot(game, rend, args):
    import os
    d = os.path.join(os.path.expanduser('~'), '.polypyrally', 'screenshots')
    os.makedirs(d, exist_ok=True)
    raw = rend.target.read(components=3)
    img = pygame.image.frombuffer(raw, (rend.W, rend.H), 'RGB')
    pygame.image.save(pygame.transform.flip(img, False, True), os.path.join(d, time.strftime('rally_%Y%m%d_%H%M%S.png')))
