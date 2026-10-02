"""2D overlay (pygame) composited over the 3D scene: speed, rpm, gear, stage time, progress and a mini-map."""
import math
import pygame

ACCENT = (255, 190, 64)
TEXT = (244, 246, 250)
DIM = (160, 170, 190)
PANEL = (16, 20, 30, 150)
OK = (110, 220, 140)
WARN = (255, 96, 80)
CYAN = (96, 210, 255)

_fonts = {}


def font(size):
    f = _fonts.get(size)
    if f is None:
        f = _fonts[size] = pygame.font.Font(None, size)
    return f


_cache = {}


def text(surf, s, pos, size, col=TEXT, anchor='tl', shadow=True):
    key = (s, size, col)
    img = _cache.get(key)
    if img is None:
        if len(_cache) > 600:
            _cache.clear()
        img = _cache[key] = font(size).render(s, True, col)
        sh = font(size).render(s, True, (0, 0, 0))
        sh.set_alpha(150)
        _cache[(s, size, col, 'sh')] = sh
    r = img.get_rect()
    setattr(r, {'tl': 'topleft', 'tr': 'topright', 'bl': 'bottomleft', 'br': 'bottomright', 'c': 'center', 'tc': 'midtop', 'bc': 'midbottom'}[anchor], pos)
    if shadow:
        surf.blit(_cache[(s, size, col, 'sh')], (r.x + max(1, size // 22), r.y + max(1, size // 22)))
    surf.blit(img, r)
    return r


def panel(surf, rect, cut=10, col=PANEL):
    x, y, w, h = rect
    p = [(x + cut, y), (x + w - cut, y), (x + w, y + cut), (x + w, y + h - cut), (x + w - cut, y + h), (x + cut, y + h), (x, y + h - cut), (x, y + cut)]
    tmp = pygame.Surface((int(w) + 2, int(h) + 2), pygame.SRCALPHA)
    pygame.draw.polygon(tmp, col, [(a - x, b - y) for a, b in p])
    surf.blit(tmp, (x, y))


def fmt_time(t):
    m = int(t // 60)
    return '%d:%05.2f' % (m, t - m * 60)


class Hud:
    def __init__(self, stage):
        self.st = stage
        self._map = None
        self._map_key = None

    def _minimap(self, size):
        key = size
        if self._map_key != key:
            st = self.st
            xs, zs = st.rx, st.rz
            x0, x1, z0, z1 = xs.min(), xs.max(), zs.min(), zs.max()
            sc = min((size - 20) / (x1 - x0 + 1), (size - 20) / (z1 - z0 + 1))
            ox = (size - (x1 - x0) * sc) / 2 - x0 * sc
            oz = (size - (z1 - z0) * sc) / 2 - z0 * sc
            self._map = (sc, ox, oz)
            surf = pygame.Surface((size, size), pygame.SRCALPHA)
            pts = [(float(x) * sc + ox, float(z) * sc + oz) for x, z in zip(xs[::6], zs[::6])]
            pygame.draw.lines(surf, (0, 0, 0, 170), False, pts, max(5, size // 28))
            pygame.draw.lines(surf, (220, 224, 236, 255), False, pts, max(3, size // 48))
            self._map_surf = surf
            self._map_key = key
        return self._map, self._map_surf

    def draw(self, surf, game):
        W, H = surf.get_size()
        u = H / 720.0
        car = game.car
        st = self.st
        # ---------------- speed / gear / rpm (bottom right)
        kmh = max(0.0, car.speed * 3.6)
        cx, cy = W - 210 * u, H - 150 * u
        panel(surf, (cx - 190 * u, cy - 112 * u, 380 * u, 224 * u), 16 * u)
        e = car.spec.engine
        frac = max(0.0, min(1.0, car.rpm / (e.redline * 1.05)))
        nseg = 30
        for i in range(nseg):
            a0 = math.radians(150 + i * 240 / nseg + 1.0)
            a1 = math.radians(150 + (i + 1) * 240 / nseg - 1.0)
            on = (i + 0.5) / nseg <= frac
            f = (i + 0.5) / nseg
            col = (255, 190, 64) if f < 0.78 else (255, 110, 70)
            if not on:
                col = (52, 58, 74)
            r0, r1 = 92 * u, 112 * u
            pygame.draw.polygon(surf, col, [(cx + math.cos(a0) * r0, cy + math.sin(a0) * r0), (cx + math.cos(a0) * r1, cy + math.sin(a0) * r1),
                                            (cx + math.cos(a1) * r1, cy + math.sin(a1) * r1), (cx + math.cos(a1) * r0, cy + math.sin(a1) * r0)])
        text(surf, '%d' % round(kmh), (cx, cy - 20 * u), int(104 * u), TEXT, 'c')
        text(surf, 'km/h', (cx, cy + 36 * u), int(26 * u), DIM, 'c')
        g = 'R' if car.gear < 0 else str(car.gear)
        shift = car.rpm > e.redline * 0.9 and car.gear > 0
        text(surf, g, (cx, cy + 74 * u), int(54 * u), WARN if shift else ACCENT, 'c')
        text(surf, 'AUTO' if car.auto else 'MANUAL', (cx - 96 * u, cy + 88 * u), int(20 * u), DIM, 'c')
        text(surf, 'TC', (cx - 150 * u, cy - 90 * u), int(24 * u), (ACCENT if car.tc_active else (OK if car.tc else DIM)), 'c')
        text(surf, '%d' % round(car.rpm / 10) + '0', (cx + 96 * u, cy + 88 * u), int(22 * u), DIM, 'c')
        # ---------------- time + progress (top centre)
        panel(surf, (W / 2 - 170 * u, 14 * u, 340 * u, 92 * u), 14 * u)
        text(surf, fmt_time(game.time), (W / 2, 20 * u), int(66 * u), TEXT, 'tc')
        prog = max(0.0, min(1.0, game.progress / st.length))
        bx, by, bw = W / 2 - 150 * u, 86 * u, 300 * u
        pygame.draw.rect(surf, (52, 58, 74), (bx, by, bw, 8 * u))
        pygame.draw.rect(surf, ACCENT, (bx, by, bw * prog, 8 * u))
        pygame.draw.polygon(surf, TEXT, [(bx + bw * prog, by - 4 * u), (bx + bw * prog + 6 * u, by + 4 * u), (bx + bw * prog, by + 12 * u), (bx + bw * prog - 6 * u, by + 4 * u)])
        text(surf, '%d / %d m' % (game.progress, st.length), (W / 2, 108 * u), int(22 * u), DIM, 'tc')
        # ---------------- mini-map (top right)
        ms = int(190 * u)
        panel(surf, (W - ms - 18 * u, 18 * u, ms, ms), 14 * u)
        (sc, ox, oz), msurf = self._minimap(ms)
        surf.blit(msurf, (W - ms - 18 * u, 18 * u))
        px, pz = car.p[0] * sc + ox + W - ms - 18 * u, car.p[2] * sc + oz + 18 * u
        fx, fz = car.forward[0], car.forward[2]
        pygame.draw.polygon(surf, ACCENT, [(px + fx * 9 * u, pz + fz * 9 * u), (px - fx * 5 * u - fz * 5 * u, pz - fz * 5 * u + fx * 5 * u), (px - fx * 5 * u + fz * 5 * u, pz - fz * 5 * u - fx * 5 * u)])
        fx0, fz0 = st.rx[-1] * sc + ox + W - ms - 18 * u, st.rz[-1] * sc + oz + 18 * u
        pygame.draw.circle(surf, OK, (fx0, fz0), 5 * u)
        # ---------------- hints
        if game.state in ('countdown', 'run') and game.time < 14.0:
            a = 255 if game.time < 10 else int(255 * (14 - game.time) / 4)
            hint = 'W gas  S brake / reverse    A/D steer    SPACE handbrake    E/Q shift up/down    T auto/manual    Y traction ctrl    C camera    R reset    ESC menu'
            img = font(int(24 * u)).render(hint, True, TEXT)
            img.set_alpha(a)
            surf.blit(img, (W / 2 - img.get_width() / 2, H - 40 * u))
        if game.state == 'countdown':
            n = math.ceil(game.countdown)
            text(surf, str(n) if n > 0 else 'GO!', (W / 2, H * 0.34), int(220 * u), ACCENT, 'c')
        elif game.state == 'run' and game.time < 1.0:
            text(surf, 'GO!', (W / 2, H * 0.34), int(220 * u), OK, 'c')
        if game.toast_t > 0:
            text(surf, game.toast, (W / 2, H * 0.22), int(40 * u), WARN, 'c')
        if game.show_fps:
            text(surf, '%d fps' % round(game.fps), (16 * u, 14 * u), int(26 * u), OK if game.fps >= 55 else WARN, 'tl')
        if game.state == 'finished':
            pw, ph = 520 * u, 300 * u
            panel(surf, (W / 2 - pw / 2, H / 2 - ph / 2, pw, ph), 20 * u, (12, 16, 26, 220))
            text(surf, 'STAGE COMPLETE', (W / 2, H / 2 - ph / 2 + 24 * u), int(56 * u), ACCENT, 'tc')
            text(surf, fmt_time(game.finish_time), (W / 2, H / 2 - 30 * u), int(110 * u), TEXT, 'c')
            avg = st.length / max(game.finish_time, 1) * 3.6
            text(surf, 'average %d km/h   -   top speed %d km/h' % (avg, game.top_speed * 3.6), (W / 2, H / 2 + 50 * u), int(28 * u), DIM, 'c')
            text(surf, 'ENTER: run again      N: new stage      ESC: menu', (W / 2, H / 2 + 100 * u), int(28 * u), TEXT, 'c')
        if game.state == 'pause':
            veil = pygame.Surface((W, H), pygame.SRCALPHA)
            veil.fill((8, 10, 18, 150))
            surf.blit(veil, (0, 0))
            pw, ph = 420 * u, 330 * u
            panel(surf, (W / 2 - pw / 2, H / 2 - ph / 2, pw, ph), 20 * u, (14, 18, 28, 235))
            text(surf, 'PAUSED', (W / 2, H / 2 - ph / 2 + 22 * u), int(64 * u), ACCENT, 'tc')
            for i, (k, label) in enumerate((('ESC', 'Resume'), ('ENTER', 'Restart stage'), ('N', 'New stage'), ('F', 'Toggle FPS counter'), ('Q', 'Quit'))):
                y = H / 2 - ph / 2 + 100 * u + i * 40 * u
                text(surf, k, (W / 2 - 150 * u, y), int(30 * u), ACCENT, 'tl')
                text(surf, label, (W / 2 - 60 * u, y), int(30 * u), TEXT, 'tl')
