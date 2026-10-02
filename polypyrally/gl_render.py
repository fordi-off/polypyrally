"""OpenGL renderer: cascaded shadow maps, an HDR multisampled scene pass, bloom and an ACES tone-map."""
import math
import numpy as np
import moderngl
from . import shaders
from .mathx import ortho, look_at, gl_bytes, perspective

VFMT = '3f 3f 3f 2f'
MAX_PART = 2300 * 18


class GLMesh:
    __slots__ = ('vbo', 'n', 'vaos', 'ctx')

    def __init__(self, ctx, data):
        self.ctx = ctx
        self.n = len(data)
        self.vbo = ctx.buffer(np.ascontiguousarray(data, dtype=np.float32).tobytes())
        self.vaos = {}

    def vao(self, key, prog, inst=None):
        k = (key, id(inst))
        v = self.vaos.get(k)
        if v is None:
            if key.endswith('depth'):
                content = [(self.vbo, '3f 8x4', 'in_pos')]
            else:
                content = [(self.vbo, VFMT, 'in_pos', 'in_nrm', 'in_col', 'in_ex')]
            if inst is not None:
                content.append((inst.vbo, '3f 4f/i', 'i_pos', 'i_misc'))
            v = self.vaos[k] = self.ctx.vertex_array(prog, content)
        return v

    def release(self):
        for v in self.vaos.values():
            v.release()
        self.vbo.release()
        self.vaos = {}


class GLInstances:
    __slots__ = ('vbo', 'n')

    def __init__(self, ctx, data):
        self.n = len(data)
        self.vbo = ctx.buffer(np.ascontiguousarray(data, dtype=np.float32).tobytes())

    def release(self):
        self.vbo.release()


def _set(prog, name, value):
    try:
        u = prog[name]
    except KeyError:
        return
    if isinstance(value, (bytes, bytearray)):
        u.write(value)
    else:
        u.value = value


class Atmosphere:
    """Golden-hour lighting parameters (linear colours)."""

    def __init__(self):
        el, az = math.radians(27.0), math.radians(215.0)
        d = np.array([math.cos(el) * math.cos(az), math.sin(el), math.cos(el) * math.sin(az)])
        self.sun_dir = d / np.linalg.norm(d)
        self.sun_col = (3.1, 2.35, 1.5)
        self.sky_amb = (0.40, 0.56, 0.98)
        self.gnd_amb = (0.34, 0.27, 0.19)
        self.fog_col = (0.55, 0.66, 0.82)
        self.fog_sun = (1.25, 0.82, 0.50)
        self.zenith = (0.07, 0.20, 0.58)
        self.horizon = (0.66, 0.74, 0.86)
        self.fog = (0.0011, 0.0105, 40.0)
        self.exposure = 0.92
        self.bloom = 0.55


class Renderer:
    def __init__(self, ctx, size, target=None, msaa=4, shadow_res=2048):
        self.ctx = ctx
        self.msaa = msaa
        self.shadow_res = shadow_res
        self.target = target or ctx.screen
        self.atm = Atmosphere()
        self.W, self.H = size
        self.time = 0.0
        self._build_programs()
        self._build_shadow()
        self._build_targets()
        self.tri = ctx.vertex_array(self.sky_prog, [])
        self.ui_tex = None
        self.info = ctx.info.get('GL_RENDERER', '?')

    # ------------------------------------------------------------------ setup
    def _build_programs(self):
        ctx = self.ctx
        self.lit = {m: ctx.program(vertex_shader=shaders.vertex(m, False), fragment_shader=shaders.LIT_FS) for m in ('terrain', 'inst', 'model')}
        self.depth = {m: ctx.program(vertex_shader=shaders.vertex(m, True), fragment_shader=shaders.SHADOW_FS) for m in ('terrain', 'inst', 'model')}
        self.sky_prog = ctx.program(vertex_shader=shaders.SKY_VS, fragment_shader=shaders.SKY_FS)
        self.bloom_prog = ctx.program(vertex_shader=shaders.POST_VS, fragment_shader=shaders.BLOOM_FS)
        self.post_prog = ctx.program(vertex_shader=shaders.POST_VS, fragment_shader=shaders.POST_FS)
        self.part_prog = ctx.program(vertex_shader=shaders.PART_VS, fragment_shader=shaders.PART_FS)
        self.part_buf = ctx.buffer(reserve=MAX_PART * 9 * 4)
        self.part_vao = ctx.vertex_array(self.part_prog, [(self.part_buf, '3f 2f 4f', 'in_pos', 'in_uv', 'in_col')])
        self.post_vao = ctx.vertex_array(self.post_prog, [])
        self.bloom_vao = ctx.vertex_array(self.bloom_prog, [])

    def _build_shadow(self):
        ctx, n = self.ctx, self.shadow_res
        self.sm_tex, self.sm_fbo = [], []
        for _ in range(2):
            t = ctx.depth_texture((n, n))
            t.compare_func = '<='
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
            t.repeat_x = t.repeat_y = False
            self.sm_tex.append(t)
            self.sm_fbo.append(ctx.framebuffer(depth_attachment=t))
        self.sm_radius = (34.0, 170.0)

    def _build_targets(self):
        ctx = self.ctx
        W, H = self.W, self.H
        self.hdr = ctx.texture((W, H), 4, dtype='f2')
        self.hdr.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.hdr.repeat_x = self.hdr.repeat_y = False
        self.hdr_depth = ctx.depth_renderbuffer((W, H))
        self.hdr_fbo = ctx.framebuffer([self.hdr], self.hdr_depth)
        if self.msaa > 1:
            self.ms_tex = ctx.texture((W, H), 4, dtype='f2', samples=self.msaa)
            self.ms_depth = ctx.depth_renderbuffer((W, H), samples=self.msaa)
            self.ms_fbo = ctx.framebuffer([self.ms_tex], self.ms_depth)
        else:
            self.ms_fbo = self.hdr_fbo
        bw, bh = max(2, W // 4), max(2, H // 4)
        self.bsz = (bw, bh)
        self.bloom_a = ctx.texture((bw, bh), 4, dtype='f2')
        self.bloom_b = ctx.texture((bw, bh), 4, dtype='f2')
        for t in (self.bloom_a, self.bloom_b):
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
            t.repeat_x = t.repeat_y = False
        self.bloom_fa = ctx.framebuffer([self.bloom_a])
        self.bloom_fb = ctx.framebuffer([self.bloom_b])

    def resize(self, size):
        if tuple(size) == (self.W, self.H):
            return
        for o in (self.hdr, self.hdr_depth, self.hdr_fbo, self.bloom_a, self.bloom_b, self.bloom_fa, self.bloom_fb):
            o.release()
        if self.msaa > 1:
            for o in (self.ms_tex, self.ms_depth, self.ms_fbo):
                o.release()
        self.W, self.H = size
        self._build_targets()

    # ------------------------------------------------------------------ assets
    def mesh(self, data):
        return GLMesh(self.ctx, data)

    def instances(self, data):
        return GLInstances(self.ctx, data)

    # ------------------------------------------------------------------ frame
    def _light_matrix(self, focus, radius):
        L = self.atm.sun_dir
        zr = 420.0
        up = np.array([0.0, 1.0, 0.0])
        f = -L
        r = np.cross(f, up)
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        texel = 2 * radius / self.shadow_res
        fx = math.floor(np.dot(focus, r) / texel) * texel
        fy = math.floor(np.dot(focus, u) / texel) * texel
        center = r * fx + u * fy + f * np.dot(focus, f)
        eye = center + L * zr
        view = look_at(eye, center, up)
        proj = ortho(-radius, radius, -radius, radius, 1.0, 2 * zr)
        return proj @ view, texel

    def render(self, fr):
        """fr: Frame (see frame.py). Draws shadow cascades, the scene, bloom and the final composite."""
        ctx = self.ctx
        atm = self.atm
        self.time = fr.time
        focus = np.asarray(fr.focus, dtype=float)
        lms = []
        # ---- shadow cascades
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.BLEND)
        ctx.disable(moderngl.CULL_FACE)
        for ci, rad in enumerate(self.sm_radius):
            lm, texel = self._light_matrix(focus, rad)
            lms.append((lm, texel))
            fbo = self.sm_fbo[ci]
            fbo.use()
            fbo.viewport = (0, 0, self.shadow_res, self.shadow_res)
            fbo.clear(depth=1.0)
            vp = gl_bytes(lm)
            for m in ('terrain', 'inst', 'model'):
                _set(self.depth[m], 'u_vp', vp)
                _set(self.depth[m], 'u_time', fr.time)
            self._draw_geometry(fr, True, focus, rad + 40.0)
        # ---- main HDR pass
        self.ms_fbo.use()
        self.ms_fbo.viewport = (0, 0, self.W, self.H)
        self.ms_fbo.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)
        vp = fr.proj @ fr.view
        # sky
        rot = fr.view.copy()
        rot[:3, 3] = 0
        inv = np.linalg.inv(fr.proj @ rot)
        sp = self.sky_prog
        _set(sp, 'u_inv', gl_bytes(inv))
        for k, v in (('u_sun_dir', tuple(atm.sun_dir)), ('u_sun_col', atm.sun_col), ('u_zenith', atm.zenith), ('u_horizon', atm.horizon),
                     ('u_fog_sun', atm.fog_sun), ('u_time', fr.time), ('u_cam', tuple(fr.cam_pos))):
            _set(sp, k, v)
        ctx.depth_mask = False
        self.tri.render(moderngl.TRIANGLES, vertices=3)
        ctx.depth_mask = True
        # lit programs: common uniforms
        sm_unit = (6, 7)
        for t, u in zip(self.sm_tex, sm_unit):
            t.use(u)
        for m, prog in self.lit.items():
            _set(prog, 'u_vp', gl_bytes(vp))
            _set(prog, 'u_time', fr.time)
            _set(prog, 'u_cam', tuple(fr.cam_pos))
            _set(prog, 'u_sun_dir', tuple(atm.sun_dir))
            _set(prog, 'u_sun_col', atm.sun_col)
            _set(prog, 'u_sky_amb', atm.sky_amb)
            _set(prog, 'u_gnd_amb', atm.gnd_amb)
            _set(prog, 'u_fog_col', atm.fog_col)
            _set(prog, 'u_fog_sun', atm.fog_sun)
            _set(prog, 'u_fog', atm.fog)
            _set(prog, 'u_sm0', sm_unit[0])
            _set(prog, 'u_sm1', sm_unit[1])
            _set(prog, 'u_lm0', gl_bytes(lms[0][0]))
            _set(prog, 'u_lm1', gl_bytes(lms[1][0]))
            n = self.shadow_res
            _set(prog, 'u_sm', (self.sm_radius[0], self.sm_radius[1], 1.0 / n, 1.0 / n))
            _set(prog, 'u_smw', (lms[0][1], lms[1][1], 0.0, 0.0))
            _set(prog, 'u_focus', tuple(focus))
            _set(prog, 'u_gloss', 0.0)
        self._draw_geometry(fr, False, np.asarray(fr.cam_pos), 1e9)
        if fr.particles is not None and len(fr.particles):
            pp = self.part_prog
            _set(pp, 'u_vp', gl_bytes(vp))
            _set(pp, 'u_cam', tuple(fr.cam_pos))
            amb = np.array(atm.sky_amb) * 0.7 + np.array(atm.sun_col) * 0.34
            _set(pp, 'u_light', tuple(amb))
            _set(pp, 'u_fog_col', atm.fog_col)
            _set(pp, 'u_fog', atm.fog)
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            ctx.depth_mask = False
            data = np.ascontiguousarray(fr.particles, dtype=np.float32)
            self.part_buf.write(data.tobytes()[:MAX_PART * 9 * 4])
            self.part_vao.render(moderngl.TRIANGLES, vertices=min(len(data), MAX_PART))
            ctx.depth_mask = True
            ctx.disable(moderngl.BLEND)
        # ---- resolve + bloom
        if self.msaa > 1:
            ctx.copy_framebuffer(self.hdr_fbo, self.ms_fbo)
        ctx.disable(moderngl.DEPTH_TEST)
        bp = self.bloom_prog
        bw, bh = self.bsz
        self.hdr.use(0)
        _set(bp, 'u_tex', 0)
        _set(bp, 'u_dir', (1.0 / self.W * 2.0, 0.0))
        _set(bp, 'u_thresh', 1.0)
        self.bloom_fa.use()
        self.bloom_fa.viewport = (0, 0, bw, bh)
        self.bloom_vao.render(moderngl.TRIANGLES, vertices=3)
        self.bloom_a.use(0)
        _set(bp, 'u_dir', (0.0, 1.0 / bh * 1.5))
        _set(bp, 'u_thresh', 0.0)
        self.bloom_fb.use()
        self.bloom_fb.viewport = (0, 0, bw, bh)
        self.bloom_vao.render(moderngl.TRIANGLES, vertices=3)
        self.bloom_b.use(0)
        _set(bp, 'u_dir', (1.0 / bw * 1.5, 0.0))
        self.bloom_fa.use()
        self.bloom_vao.render(moderngl.TRIANGLES, vertices=3)
        # ---- composite to the target
        self.target.use()
        self.target.viewport = (0, 0, self.W, self.H)
        pp = self.post_prog
        self.hdr.use(0)
        self.bloom_a.use(1)
        _set(pp, 'u_hdr', 0)
        _set(pp, 'u_bloom', 1)
        _set(pp, 'u_bloom_k', atm.bloom)
        _set(pp, 'u_exposure', atm.exposure)
        ui = fr.ui
        if ui is not None:
            w, h = ui.get_size()
            if self.ui_tex is None or self.ui_tex.size != (w, h):
                if self.ui_tex is not None:
                    self.ui_tex.release()
                self.ui_tex = self.ctx.texture((w, h), 4)
                self.ui_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.ui_tex.write(ui.get_buffer())
            self.ui_tex.use(2)
            _set(pp, 'u_ui', 2)
        _set(pp, 'u_has_ui', 1.0 if ui is not None else 0.0)
        self.post_vao.render(moderngl.TRIANGLES, vertices=3)

    # ------------------------------------------------------------------ geometry submission
    def _draw_geometry(self, fr, shadow, centre, radius):
        ctx = self.ctx
        progs = self.depth if shadow else self.lit
        key = 'depth' if shadow else 'lit'
        cx, cz = centre[0], centre[2]
        fwd = None if shadow else fr.forward
        for mesh, mx, mz, rad in fr.terrain:
            dx, dz = mx - cx, mz - cz
            if dx * dx + dz * dz > (radius + rad) ** 2:
                continue
            if fwd is not None and (dx * fwd[0] + dz * fwd[2]) < -rad * 1.1 and dx * dx + dz * dz > rad * rad:
                continue
            mesh.vao('t' + key, progs['terrain']).render(moderngl.TRIANGLES, vertices=mesh.n)
        for mesh, inst in (fr.inst_shadow if shadow else fr.inst_main):
            mesh.vao('i' + key, progs['inst'], inst).render(moderngl.TRIANGLES, vertices=mesh.n, instances=inst.n)
        for mesh, model, gloss in fr.models:
            _set(progs['model'], 'u_model', gl_bytes(model))
            if not shadow:
                _set(progs['model'], 'u_gloss', gloss)
            mesh.vao('m' + key, progs['model']).render(moderngl.TRIANGLES, vertices=mesh.n)
        if not shadow:
            _set(progs['model'], 'u_gloss', 0.0)
