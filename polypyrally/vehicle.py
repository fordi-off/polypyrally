"""Rally car physics: a 6-DOF rigid body on four ray-cast, spring-damped wheels with a combined-slip tyre model,
a real engine torque curve, a slipping clutch (implicit, so it never chatters), a 6-speed gearbox and an AWD
driveline with limited-slip couplings. Everything runs at a fixed sub-step in plain Python."""
import math
from dataclasses import dataclass, field

G = 9.81
HP_DIV = 7127.0


@dataclass
class Engine:
    curve: list
    idle: float
    redline: float
    inertia: float
    brake0: float = 30.0
    brake1: float = 0.012

    def torque(self, rpm):
        c = self.curve
        if rpm <= c[0][0]:
            return c[0][1]
        for i in range(1, len(c)):
            if rpm <= c[i][0]:
                a, b = c[i - 1], c[i]
                return a[1] + (b[1] - a[1]) * (rpm - a[0]) / (b[0] - a[0])
        return 0.0


@dataclass
class CarSpec:
    name: str = 'Zephyr GT'
    mass: float = 1350.0
    inertia: tuple = (520.0, 1900.0, 2150.0)          # about x (roll), y (yaw), z (pitch)
    wheel_r: float = 0.33
    wheel_i: float = 1.2
    wheel_x: tuple = (1.28, -1.32)                    # front, rear axle positions (body x)
    wheel_z: float = 0.98                             # half track
    mount_y: float = 0.14
    l0: float = 0.40                                  # suspension length at rest (attach point -> wheel centre)
    travel_up: float = 0.20
    travel_down: float = 0.17
    k: tuple = (38000.0, 34000.0)                     # spring rate, front / rear (N/m at the wheel)
    c_bump: tuple = (3100.0, 2900.0)
    c_reb: tuple = (5200.0, 4800.0)
    arb: tuple = (17000.0, 9000.0)
    engine: Engine = field(default_factory=lambda: Engine(
        [(0, 150), (1000, 230), (2000, 330), (3000, 450), (4000, 520), (5000, 530), (5500, 520), (6500, 480), (7200, 400), (7700, 250), (8200, 0)],
        idle=1000.0, redline=7600.0, inertia=0.16))
    torque_scale: float = 0.82
    gears: tuple = (3.4, 2.3, 1.7, 1.35, 1.1, 0.9)
    reverse: float = 3.2
    final: float = 4.2
    eta: float = 0.9
    clutch_cap: float = 900.0
    front_split: float = 0.42
    brake_torque: float = 2300.0
    brake_bias: float = 0.62
    cd_a: float = 0.68
    down_c: float = 0.5
    steer_max: float = 0.46
    steer_fast: float = 0.10


RALLY = CarSpec()
PEAK_C = 1.55


def _tyre_curve(s):
    """Normalised friction vs combined slip: 1 at s=1, ~0.65 when sliding hard."""
    return math.sin(PEAK_C * math.atan(B_TYRE * s))


B_TYRE = math.tan(math.pi / (2 * PEAK_C))
TYRE_SLOPE = PEAK_C * B_TYRE


class Wheel:
    __slots__ = ('ax', 'az', 'front', 'k', 'cb', 'cr', 'om', 'angle', 'steer', 'x', 'comp', 'load', 'contact', 'Fx', 'Fy',
                 'slip', 'cx', 'cy', 'cz', 'hx', 'hy', 'hz', 'mu', 'side', 'surf')

    def __init__(self, ax, az, front, k, cb, cr):
        self.ax, self.az, self.front, self.k, self.cb, self.cr = ax, az, front, k, cb, cr
        self.side = 1 if az > 0 else -1
        self.om = 0.0
        self.angle = 0.0
        self.steer = 0.0
        self.comp = 0.0
        self.x = 0.0
        self.load = 0.0
        self.contact = False
        self.Fx = self.Fy = 0.0
        self.slip = 0.0
        self.cx = self.cy = self.cz = 0.0
        self.hx = self.hy = self.hz = 0.0
        self.mu = 0.8
        self.surf = (0.0, 1.0, 0.0)


def _surface(stage, x, z):
    """(peak friction, rolling resistance) from the distance to the road."""
    d = stage.road_distance(x, z)
    if d < 3.1:
        return 0.82, 0.022
    if d < 6.0:
        t = (d - 3.1) / 2.9
        return 0.82 - 0.14 * t, 0.022 + 0.028 * t
    return 0.66, 0.06


class Car:
    def __init__(self, stage, spec=RALLY):
        self.st = stage
        self.spec = spec
        s = spec
        self.wheels = []
        for i, ax in enumerate(s.wheel_x):
            for sgn in (1, -1):
                self.wheels.append(Wheel(ax, sgn * s.wheel_z, i == 0, s.k[i], s.c_bump[i], s.c_reb[i]))
        self.hull = [(2.1, -0.26, 0.82), (2.1, -0.26, -0.82), (-2.1, -0.26, 0.82), (-2.1, -0.26, -0.82), (0.0, -0.3, 0.0),
                     (1.6, 0.18, 0.9), (1.6, 0.18, -0.9), (-1.6, 0.34, 0.9), (-1.6, 0.34, -0.9),
                     (0.1, 0.84, 0.7), (0.1, 0.84, -0.7), (-0.75, 0.82, 0.68), (-0.75, 0.82, -0.68), (2.18, 0.0, 0.0), (-2.18, 0.2, 0.0)]
        self.reset(0.0, 0.0, 0.0, 0.0)
        self.t = 0.0

    # ------------------------------------------------------------------ state
    def reset(self, x, y, z, heading, speed=0.0):
        """Place the car on the ground at (x, z) facing `heading` (radians, x forward = +x at 0)."""
        c, s = math.cos(heading), math.sin(heading)
        # body x -> (c, 0, s), body y -> up, body z -> (-s, 0, c) ... right-handed with z to the right
        self.R = [c, 0.0, -s, 0.0, 1.0, 0.0, s, 0.0, c]
        self.p = [x, y + self.spec.wheel_r + self.spec.l0 - self.spec.mount_y + 0.18, z]
        self.v = [c * speed, 0.0, s * speed]
        self.w = [0.0, 0.0, 0.0]
        self.eng_w = self.spec.engine.idle * math.pi / 30
        self.gear = 1
        self.auto = True
        self.shift_t = 0.0
        self.cool = 0.0
        self.steer = 0.0
        self.thr = 0.0
        self.brk = 0.0
        self.hb = 0.0
        self.rpm = self.spec.engine.idle
        self.Te = 0.0
        self.Tc = 0.0
        self.hit = 0.0
        self.air = 0.0
        for wl in self.wheels:
            wl.om = speed / self.spec.wheel_r
            wl.comp = 0.0
            wl.Fx = wl.Fy = 0.0

    @property
    def forward(self):
        R = self.R
        return (R[0], R[3], R[6])

    @property
    def up(self):
        R = self.R
        return (R[1], R[4], R[7])

    @property
    def heading(self):
        R = self.R
        return math.atan2(R[6], R[0])

    @property
    def speed(self):
        R, v = self.R, self.v
        return v[0] * R[0] + v[1] * R[3] + v[2] * R[6]

    @property
    def kmh(self):
        return self.speed * 3.6

    def world(self, lx, ly, lz):
        R, p = self.R, self.p
        return (p[0] + R[0] * lx + R[1] * ly + R[2] * lz, p[1] + R[3] * lx + R[4] * ly + R[5] * lz, p[2] + R[6] * lx + R[7] * ly + R[8] * lz)

    # ------------------------------------------------------------------ control + step
    def control(self, dt, thr, brk, steer, hb, up=False, down=False, allow_reverse=True):
        s = self.spec
        # throttle / brake eased in like pedals; steering speed-sensitive
        sp0 = self.speed
        if allow_reverse and brk > 0.2 and thr < 0.05 and sp0 < 1.0 and not hb:   # standing still and holding brake -> reverse
            self.gear = -1
        elif (thr > 0.05 or not allow_reverse) and self.gear < 0:
            self.gear = 1
        if self.gear < 0:
            thr, brk = brk, 0.0
        self.thr += max(-dt * 6.0, min(dt * 4.0, thr - self.thr))
        self.brk += max(-dt * 8.0, min(dt * 7.0, brk - self.brk))
        self.hb = hb
        sp = abs(self.speed)
        lim = s.steer_max + (s.steer_fast - s.steer_max) * min(1.0, sp / 55.0)
        target = steer * lim
        rate = 2.6 if abs(target) > abs(self.steer) else 4.0
        self.steer += max(-dt * rate, min(dt * rate, target - self.steer))
        if up and self.shift_t <= 0 and self.gear < len(s.gears):
            self._shift(self.gear + 1)
        if down and self.shift_t <= 0 and self.gear > 1:
            self._shift(self.gear - 1)

    def _shift(self, g):
        self.gear = g
        self.shift_t = 0.16
        self.cool = 0.5

    def step(self, dt):
        n = max(1, int(math.ceil(dt * 240.0 - 1e-6)))
        h = dt / n
        for _ in range(n):
            self._substep(h)
        self.t += dt

    # ------------------------------------------------------------------ physics
    def _substep(self, dt):
        s = self.spec
        st = self.st
        R = self.R
        p, v, w = self.p, self.v, self.w
        M = s.mass
        Rw = s.wheel_r
        Iw = s.wheel_i
        up = (R[1], R[4], R[7])
        Fx_t = Fy_t = Fz_t = 0.0
        Tx = Ty = Tz = 0.0                         # world torque about the centre of mass
        wheels = self.wheels
        # ---------------- engine, clutch, gearbox
        eng = s.engine
        rpm = self.eng_w * 30 / math.pi
        thr = self.thr
        if self.cool > 0:
            self.cool -= dt
        if self.shift_t > 0:
            self.shift_t -= dt
        full = eng.torque(rpm) * s.torque_scale if rpm < eng.redline else 0.0
        Te = full * thr - (eng.brake0 + eng.brake1 * rpm) * (1.0 - thr)
        if rpm < eng.idle:                       # idle governor
            Te += (eng.idle - rpm) * 0.35
        self.Te = Te
        sgn = -1.0 if self.gear < 0 else 1.0
        ratio_s = (s.reverse if self.gear < 0 else s.gears[self.gear - 1]) * s.final * sgn
        wts = [(s.front_split if wl.front else 1.0 - s.front_split) * 0.5 for wl in wheels]
        om_in = ratio_s * sum(wt * wl.om for wl, wt in zip(wheels, wts))
        kw = sum(wt * wt for wt in wts) / Iw
        Lw = sum(wt * (wl.Fx * Rw) for wl, wt in zip(wheels, wts))
        if self.shift_t <= 0:
            Tc = ((self.eng_w - om_in) / dt + Te / eng.inertia + ratio_s * Lw / Iw) / (1.0 / eng.inertia + ratio_s * ratio_s * s.eta * kw)
            x = (rpm - 1250.0) / 1750.0
            x = 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
            cap = 28.0 + s.clutch_cap * x * x * (3 - 2 * x)
            Tc = -cap if Tc < -cap else (cap if Tc > cap else Tc)
        else:
            Tc = 0.0
        self.Tc = Tc
        Tw_total = Tc * ratio_s * s.eta
        self.eng_w += (Te - Tc) / eng.inertia * dt
        if self.eng_w < 40.0:
            self.eng_w = 40.0
        self.rpm = self.eng_w * 30 / math.pi
        # limited-slip couplings (centre + both axles): torque goes to the slower side
        omf = 0.5 * (wheels[0].om + wheels[1].om)
        omr = 0.5 * (wheels[2].om + wheels[3].om)
        cb = max(-900.0, min(900.0, 60.0 * (omr - omf)))
        bf = max(-700.0, min(700.0, 160.0 * (wheels[1].om - wheels[0].om)))
        br = max(-700.0, min(700.0, 160.0 * (wheels[3].om - wheels[2].om)))
        Tdrv = (Tw_total * wts[0] + cb * 0.5 + bf, Tw_total * wts[1] + cb * 0.5 - bf,
                Tw_total * wts[2] - cb * 0.5 + br, Tw_total * wts[3] - cb * 0.5 - br)
        # ---------------- aero
        sp = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
        drag = 0.6 * s.cd_a * sp
        fdown = s.down_c * sp * sp * 0.6
        Fx_t += -drag * v[0] - up[0] * fdown
        Fy_t += -drag * v[1] - up[1] * fdown - M * G
        Fz_t += -drag * v[2] - up[2] * fdown
        # ---------------- suspension geometry (ray per wheel)
        compr = [0.0, 0.0, 0.0, 0.0]
        rels = []
        any_contact = False
        lmax = s.l0 + s.travel_down
        lmin = s.l0 - s.travel_up
        dx, dy, dz = -up[0], -up[1], -up[2]
        for wi, wl in enumerate(wheels):
            lx, ly, lz = wl.ax, s.mount_y, wl.az
            rx = R[0] * lx + R[1] * ly + R[2] * lz
            ry = R[3] * lx + R[4] * ly + R[5] * lz
            rz = R[6] * lx + R[7] * ly + R[8] * lz
            rels.append((rx, ry, rz))
            ax_, ay_, az_ = p[0] + rx, p[1] + ry, p[2] + rz
            t = 99.0
            if dy < -0.05:
                h, nx, ny, nz = st.height_normal(ax_, az_)
                t = (ay_ - h) / -dy
                for _ in range(3):
                    qx, qz = ax_ + dx * t, az_ + dz * t
                    h, nx, ny, nz = st.height_normal(qx, qz)
                    den = nx * dx + ny * dy + nz * dz
                    if den > -0.05:
                        t = 99.0
                        break
                    t = (nx * (qx - ax_) + ny * (h - ay_) + nz * (qz - az_)) / den
            sl = t - Rw
            if t > 90.0 or sl > lmax:
                wl.contact = False
                wl.x = -s.travel_down
                wl.load = 0.0
                wl.Fx = wl.Fy = 0.0
                compr[wi] = -s.travel_down
                sl = lmax
            else:
                wl.contact = True
                any_contact = True
                x = s.l0 - sl
                wl.x = x
                compr[wi] = x
                wl.cx, wl.cy, wl.cz = ax_ + dx * t, ay_ + dy * t, az_ + dz * t
                wl.surf = (nx, ny, nz)
                if sl < lmin:
                    sl = lmin
            wl.hx, wl.hy, wl.hz = ax_ + dx * sl, ay_ + dy * sl, az_ + dz * sl
        arb = (s.arb[0] * (compr[0] - compr[1]), s.arb[1] * (compr[2] - compr[3]))
        # ---------------- per wheel: spring, damper, tyre
        for wi, wl in enumerate(wheels):
            rx, ry, rz = rels[wi]
            if not wl.contact:
                Tb = self._brake_t(wl)
                om = wl.om + Tdrv[wi] / Iw * dt
                bd = Tb * dt / Iw
                om = 0.0 if abs(om) <= bd else om - math.copysign(bd, om)
                wl.om = om * (1.0 - 0.1 * dt)
                wl.angle += wl.om * dt
                continue
            nx, ny, nz = wl.surf
            x = wl.x
            vax = v[0] + w[1] * rz - w[2] * ry
            vay = v[1] + w[2] * rx - w[0] * rz
            vaz = v[2] + w[0] * ry - w[1] * rx
            den = nx * dx + ny * dy + nz * dz
            xdot = (nx * vax + ny * vay + nz * vaz) / den
            Fs = wl.k * x + (wl.cb if xdot > 0 else wl.cr) * xdot
            lim = s.travel_up - 0.05
            if x > lim:
                Fs += 260000.0 * (x - lim) + (20000.0 * xdot if xdot > 0 else 0.0)
            Fs += arb[wi >> 1] if wi % 2 == 0 else -arb[wi >> 1]
            if Fs < 0.0:
                Fs = 0.0
            wl.load = Fs
            Fxs, Fys, Fzs = up[0] * Fs, up[1] * Fs, up[2] * Fs
            Fx_t += Fxs
            Fy_t += Fys
            Fz_t += Fzs
            Tx += ry * Fzs - rz * Fys
            Ty += rz * Fxs - rx * Fzs
            Tz += rx * Fys - ry * Fxs
            # tyre forces at the contact patch
            gx, gy, gz = wl.cx - p[0], wl.cy - p[1], wl.cz - p[2]
            if wl.front:
                cs, sn = math.cos(self.steer), math.sin(self.steer)
                wl.steer = self.steer
            else:
                cs, sn = 1.0, 0.0
            fx_ = R[0] * cs + R[2] * sn
            fy_ = R[3] * cs + R[5] * sn
            fz_ = R[6] * cs + R[8] * sn
            dn = fx_ * nx + fy_ * ny + fz_ * nz
            fx_ -= nx * dn
            fy_ -= ny * dn
            fz_ -= nz * dn
            il = 1.0 / math.sqrt(fx_ * fx_ + fy_ * fy_ + fz_ * fz_)
            fx_, fy_, fz_ = fx_ * il, fy_ * il, fz_ * il
            lx_ = fy_ * nz - fz_ * ny
            ly_ = fz_ * nx - fx_ * nz
            lz_ = fx_ * ny - fy_ * nx
            vpx = v[0] + w[1] * gz - w[2] * gy
            vpy = v[1] + w[2] * gx - w[0] * gz
            vpz = v[2] + w[0] * gy - w[1] * gx
            vlong = vpx * fx_ + vpy * fy_ + vpz * fz_
            vlat = vpx * lx_ + vpy * ly_ + vpz * lz_
            mu, crr = _surface(st, wl.cx, wl.cz)
            Fmax = mu * Fs
            den_v = abs(vlong) + 1.4
            sx = wl.om * Rw - vlong
            sk = sx / den_v / 0.15
            sa = -vlat / den_v / 0.20
            sm = math.sqrt(sk * sk + sa * sa)
            if sm > 1e-6:
                f = Fmax * math.sin(PEAK_C * math.atan(B_TYRE * sm))
                Fx = f * sk / sm
                Fy = f * sa / sm
            else:
                Fx = Fy = 0.0
            K = Fmax * TYRE_SLOPE / (0.15 * den_v)
            domega = dt * (Tdrv[wi] - Rw * Fx) / (Iw + dt * Rw * Rw * K)
            Tb = self._brake_t(wl)
            if Tb > 0.0:
                bd = Tb * dt / Iw
                om_new = wl.om + domega
                if abs(om_new) <= bd:
                    domega = -wl.om
                else:
                    domega -= math.copysign(bd, om_new)
            Fxa = Fx + K * Rw * domega
            wl.om += domega
            wl.angle += wl.om * dt
            fl = math.hypot(Fxa, Fy)
            if fl > Fmax and fl > 0.0:
                kf = Fmax / fl
                Fxa *= kf
                Fy *= kf
            Fxa -= crr * Fs * math.tanh(vlong * 1.5)
            wl.Fx, wl.Fy = Fxa, Fy
            wl.slip = sm
            wl.mu = mu
            fxs = Fxa * fx_ + Fy * lx_
            fys = Fxa * fy_ + Fy * ly_
            fzs = Fxa * fz_ + Fy * lz_
            Fx_t += fxs
            Fy_t += fys
            Fz_t += fzs
            Tx += gy * fzs - gz * fys
            Ty += gz * fxs - gx * fzs
            Tz += gx * fys - gy * fxs
        # ---------------- hull vs terrain (bottoming out, roof, nose)
        hit = 0.0
        for lx, ly, lz in self.hull:
            rx = R[0] * lx + R[1] * ly + R[2] * lz
            ry = R[3] * lx + R[4] * ly + R[5] * lz
            rz = R[6] * lx + R[7] * ly + R[8] * lz
            qx, qy, qz = p[0] + rx, p[1] + ry, p[2] + rz
            h, nx, ny, nz = st.height_normal(qx, qz)
            pen = (h - qy) * ny
            if pen <= 0.0:
                continue
            vx = v[0] + w[1] * rz - w[2] * ry
            vy = v[1] + w[2] * rx - w[0] * rz
            vz = v[2] + w[0] * ry - w[1] * rx
            vn = vx * nx + vy * ny + vz * nz
            Fn = 130000.0 * pen - 5200.0 * vn
            if Fn < 0.0:
                continue
            tx, ty, tz = vx - vn * nx, vy - vn * ny, vz - vn * nz
            tl = math.sqrt(tx * tx + ty * ty + tz * tz)
            ft = 0.0
            if tl > 1e-4:
                ft = -0.55 * Fn * min(1.0, tl / 0.8) / tl
            fx_h, fy_h, fz_h = Fn * nx + ft * tx, Fn * ny + ft * ty, Fn * nz + ft * tz
            Fx_t += fx_h
            Fy_t += fy_h
            Fz_t += fz_h
            Tx += ry * fz_h - rz * fy_h
            Ty += rz * fx_h - rx * fz_h
            Tz += rx * fy_h - ry * fx_h
            if Fn > hit:
                hit = Fn
        self.hit = hit
        self.air = 0.0 if (any_contact or hit > 0) else self.air + dt
        # ---------------- integrate (body-frame angular dynamics)
        v[0] += Fx_t / M * dt
        v[1] += Fy_t / M * dt
        v[2] += Fz_t / M * dt
        p[0] += v[0] * dt
        p[1] += v[1] * dt
        p[2] += v[2] * dt
        # torque and angular velocity to the body frame
        tb0 = R[0] * Tx + R[3] * Ty + R[6] * Tz
        tb1 = R[1] * Tx + R[4] * Ty + R[7] * Tz
        tb2 = R[2] * Tx + R[5] * Ty + R[8] * Tz
        wb0 = R[0] * w[0] + R[3] * w[1] + R[6] * w[2]
        wb1 = R[1] * w[0] + R[4] * w[1] + R[7] * w[2]
        wb2 = R[2] * w[0] + R[5] * w[1] + R[8] * w[2]
        I0, I1, I2 = s.inertia
        a0 = (tb0 - (wb1 * I2 * wb2 - wb2 * I1 * wb1)) / I0
        a1 = (tb1 - (wb2 * I0 * wb0 - wb0 * I2 * wb2)) / I1
        a2 = (tb2 - (wb0 * I1 * wb1 - wb1 * I0 * wb0)) / I2
        wb0 += a0 * dt
        wb1 += a1 * dt
        wb2 += a2 * dt
        w[0] = R[0] * wb0 + R[1] * wb1 + R[2] * wb2
        w[1] = R[3] * wb0 + R[4] * wb1 + R[5] * wb2
        w[2] = R[6] * wb0 + R[7] * wb1 + R[8] * wb2
        # rotate R by w*dt (Rodrigues)
        ang = math.sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]) * dt
        if ang > 1e-9:
            ux, uy, uz = w[0] * dt / ang, w[1] * dt / ang, w[2] * dt / ang
            c, sn = math.cos(ang), math.sin(ang)
            C = 1 - c
            Rm = (c + ux * ux * C, ux * uy * C - uz * sn, ux * uz * C + uy * sn,
                  uy * ux * C + uz * sn, c + uy * uy * C, uy * uz * C - ux * sn,
                  uz * ux * C - uy * sn, uz * uy * C + ux * sn, c + uz * uz * C)
            R0 = R
            self.R = R = [Rm[0] * R0[0] + Rm[1] * R0[3] + Rm[2] * R0[6], Rm[0] * R0[1] + Rm[1] * R0[4] + Rm[2] * R0[7], Rm[0] * R0[2] + Rm[1] * R0[5] + Rm[2] * R0[8],
                          Rm[3] * R0[0] + Rm[4] * R0[3] + Rm[5] * R0[6], Rm[3] * R0[1] + Rm[4] * R0[4] + Rm[5] * R0[7], Rm[3] * R0[2] + Rm[4] * R0[5] + Rm[5] * R0[8],
                          Rm[6] * R0[0] + Rm[7] * R0[3] + Rm[8] * R0[6], Rm[6] * R0[1] + Rm[7] * R0[4] + Rm[8] * R0[7], Rm[6] * R0[2] + Rm[7] * R0[5] + Rm[8] * R0[8]]
            self._ortho_count = getattr(self, '_ortho_count', 0) + 1
            if self._ortho_count % 16 == 0:
                self._orthonormalise()
        # ---------------- automatic gearbox
        if self.auto and self.shift_t <= 0 and self.cool <= 0:
            self._auto_shift(self.thr)

    def _brake_t(self, wl):
        s = self.spec
        b = self.brk * s.brake_torque * (s.brake_bias if wl.front else 1 - s.brake_bias) * 1.0
        if self.hb > 0.5 and not wl.front:
            b += 3500.0
        return b

    def _auto_shift(self, thr):
        s = self.spec
        e = s.engine
        rpm = self.rpm
        if self.gear < 0:
            return
        if rpm > e.redline * 0.93 and self.gear < len(s.gears) and thr > 0.2:
            self._shift(self.gear + 1)
        elif self.gear > 1:
            lower = rpm * s.gears[self.gear - 2] / s.gears[self.gear - 1]
            if rpm < e.redline * 0.38 and lower < e.redline * 0.82:
                self._shift(self.gear - 1)

    def _orthonormalise(self):
        R = self.R
        c0 = (R[0], R[3], R[6])
        c1 = (R[1], R[4], R[7])
        n0 = math.sqrt(sum(a * a for a in c0))
        c0 = tuple(a / n0 for a in c0)
        d = sum(a * b for a, b in zip(c1, c0))
        c1 = tuple(b - d * a for a, b in zip(c0, c1))
        n1 = math.sqrt(sum(a * a for a in c1))
        c1 = tuple(a / n1 for a in c1)
        c2 = (c0[1] * c1[2] - c0[2] * c1[1], c0[2] * c1[0] - c0[0] * c1[2], c0[0] * c1[1] - c0[1] * c1[0])
        self.R = [c0[0], c1[0], c2[0], c0[1], c1[1], c2[1], c0[2], c1[2], c2[2]]
