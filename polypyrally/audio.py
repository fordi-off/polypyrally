"""Synthesised engine (looping harmonic banks cross-faded by rpm), gravel / wind / skid layers. No audio files."""
import math
import numpy as np
import pygame

SR = 22050


def _engine_loop(rpm, cylinders=4):
    f = rpm / 60.0 * cylinders / 2.0
    cycles = max(2, round(f * 0.25))
    n = int(round(cycles / f * SR))
    t = np.arange(n) / n * cycles * math.tau
    w = np.zeros(n)
    for h, a in enumerate((0.9, 1.0, 0.85, 0.6, 0.45, 0.3, 0.2, 0.14, 0.1, 0.06), start=1):
        w += a * np.sin(h * t + 0.8 * h * h * 0.1)
    w += 0.5 * np.sin(0.5 * t) + 0.25 * np.sin(1.5 * t)
    w = np.tanh(w * 1.25)
    return (w * 0.5 * 32767).astype(np.int16)


def _noise_loop(seed, k1, k2, secs=3):
    rng = np.random.default_rng(seed)
    n = SR * secs
    x = rng.standard_normal(n)
    for k in (k1, k2):
        c = np.cumsum(np.concatenate([x, x[:k]]))
        x = (c[k:k + n] - c[:n]) / k
    x /= np.abs(x).max()
    return (x * 0.6 * 32767).astype(np.int16)


def _whistle(freq):
    n = int(round(freq * 0.4)) / freq * SR
    n = int(n)
    t = np.arange(n) / n * round(freq * 0.4) * math.tau
    w = np.sin(t) + 0.35 * np.sin(2 * t + 0.4) + 0.12 * np.sin(3 * t)
    return (w * 0.22 * 32767).astype(np.int16)


def _burst(secs, seed, hp=0.9, decay=7.0):
    rng = np.random.default_rng(seed)
    n = int(SR * secs)
    x = rng.standard_normal(n)
    x = x - np.convolve(x, np.ones(8) / 8, mode='same') * hp           # crude high-pass
    x *= np.exp(-np.arange(n) / n * decay)
    return (x / np.abs(x).max() * 0.55 * 32767).astype(np.int16)


class Audio:
    def __init__(self, spec, enabled=True):
        self.ok = False
        self.muted = False
        self.volume = 0.8
        if not enabled:
            return
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(SR, -16, 1, 512)
            pygame.mixer.set_num_channels(8)
            e = spec.engine
            lo, hi = max(300.0, e.idle * 0.6), e.redline * 1.12
            n = 18
            self.rpms = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
            self.step = (hi - lo) / (n - 1)
            self.sounds = [pygame.mixer.Sound(buffer=_engine_loop(r).tobytes()) for r in self.rpms]
            self.ch = [pygame.mixer.Channel(0), pygame.mixer.Channel(1)]
            self.cur = [None, None]
            self.wind = pygame.mixer.Channel(2)
            self.gravel = pygame.mixer.Channel(3)
            self.skid = pygame.mixer.Channel(4)
            self.turbo = [pygame.mixer.Channel(5), pygame.mixer.Channel(6)]
            self.fx = pygame.mixer.Channel(7)
            self.whistles = [pygame.mixer.Sound(buffer=_whistle(f).tobytes()) for f in (1300, 1900, 2700, 3700)]
            self.blow = pygame.mixer.Sound(buffer=_burst(0.55, 4, 0.95, 5.0).tobytes())
            self.pops = [pygame.mixer.Sound(buffer=_burst(0.09, 30 + i, 0.2, 9.0).tobytes()) for i in range(4)]
            self.tcur = [None, None]
            self.prev_thr = 0.0
            self.pop_t = 0.0
            self.rng = np.random.default_rng(8)
            self.wind.play(pygame.mixer.Sound(buffer=_noise_loop(3, 70, 30).tobytes()), loops=-1)
            self.gravel.play(pygame.mixer.Sound(buffer=_noise_loop(9, 9, 4).tobytes()), loops=-1)
            self.skid.play(pygame.mixer.Sound(buffer=_noise_loop(21, 5, 3).tobytes()), loops=-1)
            for c in (self.wind, self.gravel, self.skid):
                c.set_volume(0.0)
            self.ok = True
        except Exception:
            self.ok = False

    def stop(self):
        if self.ok:
            try:
                for c in self.ch + [self.wind, self.gravel, self.skid] + self.turbo + [self.fx]:
                    c.stop()
            except pygame.error:
                pass
        self.cur = [None, None]

    def update(self, car):
        if not self.ok or self.muted:
            return
        rpm = min(max(car.rpm, self.rpms[0]), self.rpms[-1] - 1)
        pos = (rpm - self.rpms[0]) / self.step
        i, frac = int(pos), pos - int(pos)
        rf = min(1.0, car.rpm / 7000.0)
        gain = (0.14 + 0.78 * car.thr * (0.45 + 0.55 * rf)) * self.volume
        for layer, vol in ((i, (1 - frac) * gain), (i + 1, frac * gain)):
            slot = layer & 1
            if self.cur[slot] != layer:
                self.ch[slot].play(self.sounds[layer], loops=-1)
                self.cur[slot] = layer
            self.ch[slot].set_volume(max(0.0, min(1.0, vol)))
        sp = abs(car.speed)
        grounded = sum(1 for w in car.wheels if w.contact)
        self.wind.set_volume(min(1.0, sp / 60.0) * 0.5 * self.volume)
        ss = max((abs(w.sx) + abs(w.Fy) / max(w.load, 500.0) * 6.0 for w in car.wheels if w.contact), default=0.0)
        self.gravel.set_volume(min(1.0, sp / 35.0 + ss * 0.04) * (grounded / 4.0) * 0.5 * self.volume)
        self.skid.set_volume(min(1.0, max(0.0, ss - 1.5) * 0.12) * 0.75 * self.volume)       # wheelspin / sliding roar
        # turbo whistle: pitch follows rpm, level follows boost
        pos_w = min(2.999, max(0.0, (rf - 0.25) / 0.75 * 3.0))
        iw, fw = int(pos_w), pos_w - int(pos_w)
        wg = car.boost * (0.25 + 0.75 * rf) * 0.22 * self.volume
        for layer, vol in ((iw, (1 - fw) * wg), (iw + 1, fw * wg)):
            slot = layer & 1
            if self.tcur[slot] != layer:
                self.turbo[slot].play(self.whistles[layer], loops=-1)
                self.tcur[slot] = layer
            self.turbo[slot].set_volume(max(0.0, min(1.0, vol)))
        # blow-off valve when the throttle is lifted on boost, exhaust pops on the overrun
        if self.prev_thr > 0.6 and car.thr < 0.3 and car.boost > 0.45:
            self.fx.play(self.blow)
            self.fx.set_volume(min(1.0, 0.4 * car.boost) * self.volume)
        self.pop_t -= 1 / 60
        if car.thr < 0.05 and car.rpm > 4200 and self.pop_t <= 0 and not self.fx.get_busy():
            self.pop_t = 0.06 + self.rng.random() * 0.2
            if self.rng.random() < 0.45:
                self.fx.play(self.pops[int(self.rng.integers(4))])
                self.fx.set_volume(0.5 * self.volume)
        self.prev_thr = car.thr
