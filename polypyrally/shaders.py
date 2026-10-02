"""GLSL for the lit, flat-shaded low-poly look: a warm sun with cascaded soft shadows, hemispheric sky light,
height fog tinted toward the sun, a stylised sky and an ACES tone-map with a gentle bloom."""

# ---------------------------------------------------------------------------------------------- vertex stages
def vertex(mode, shadow):
    """mode: 'terrain' (world-space verts) | 'inst' (per-instance transform) | 'model' (uniform matrix)."""
    head = '#version 330\n'
    uni = 'uniform mat4 u_vp;\nuniform float u_time;\n'
    attrs = 'in vec3 in_pos;\nin vec3 in_nrm;\nin vec3 in_col;\nin vec2 in_ex;\n'
    out = '' if shadow else 'out vec3 v_pos;\nout vec3 v_nrm;\nout vec3 v_col;\nout vec2 v_ex;\n'
    if mode == 'terrain':
        body = '    vec3 wp = in_pos; vec3 wn = in_nrm; vec3 col = in_col;\n'
    elif mode == 'inst':
        attrs += 'in vec3 i_pos;\nin vec4 i_misc;\n'
        body = '''    float cy = cos(i_misc.x), sy = sin(i_misc.x);
    vec3 lp = in_pos * i_misc.y;
    vec3 wp = i_pos + vec3(lp.x * cy + lp.z * sy, lp.y, -lp.x * sy + lp.z * cy);
    vec3 wn = vec3(in_nrm.x * cy + in_nrm.z * sy, in_nrm.y, -in_nrm.x * sy + in_nrm.z * cy);
    float sway = sin(u_time * 1.3 + i_pos.x * 0.37 + i_pos.z * 0.29) * 0.018 * lp.y * lp.y * 0.12 / (1.0 + i_misc.y);
    wp.x += sway; wp.z += sway * 0.6;
    vec3 col = in_col * (0.82 + 0.36 * i_misc.z);
    col = mix(col, vec3(col.g * 2.3, col.g * 1.0, col.g * 0.22), i_misc.w);
'''
    else:
        uni += 'uniform mat4 u_model;\n'
        body = '''    vec3 wp = (u_model * vec4(in_pos, 1.0)).xyz;
    vec3 wn = mat3(u_model) * in_nrm; vec3 col = in_col;
'''
    if shadow:
        tail = '    gl_Position = u_vp * vec4(wp, 1.0);\n'
    else:
        tail = '''    gl_Position = u_vp * vec4(wp, 1.0);
    v_pos = wp; v_nrm = wn; v_col = col; v_ex = in_ex;
'''
    return head + uni + attrs + out + 'void main() {\n' + body + tail + '}\n'


SHADOW_FS = '#version 330\nvoid main() {}\n'

LIT_FS = '''#version 330
in vec3 v_pos; in vec3 v_nrm; in vec3 v_col; in vec2 v_ex;
out vec4 f_col;
uniform vec3 u_cam;
uniform vec3 u_sun_dir;       // unit vector pointing at the sun
uniform vec3 u_sun_col;
uniform vec3 u_sky_amb;
uniform vec3 u_gnd_amb;
uniform vec3 u_fog_col;
uniform vec3 u_fog_sun;
uniform vec3 u_fog;           // density, height falloff, reference height
uniform float u_gloss;
uniform sampler2DShadow u_sm0;
uniform sampler2DShadow u_sm1;
uniform mat4 u_lm0;
uniform mat4 u_lm1;
uniform vec4 u_sm;            // radius0, radius1, texel uv 0, texel uv 1
uniform vec4 u_smw;           // world size of a texel 0, 1, focus xz
uniform vec3 u_focus;

float pcf(sampler2DShadow sm, mat4 lm, vec3 p, vec3 n, float tw, float tuv) {
    vec3 pp = p + n * tw * 1.6;
    vec4 q = lm * vec4(pp, 1.0);
    vec3 c = q.xyz * 0.5 + 0.5;
    if (c.x < 0.0 || c.y < 0.0 || c.x > 1.0 || c.y > 1.0 || c.z > 1.0) return 1.0;
    float z = c.z - 0.0004;
    float s = 0.0;
    for (int i = -1; i <= 1; i++)
        for (int j = -1; j <= 1; j++)
            s += texture(sm, vec3(c.xy + vec2(float(i), float(j)) * tuv * 1.2, z));
    return s / 9.0;
}

float shadow(vec3 p, vec3 n) {
    float d = length(p.xz - u_focus.xz);
    float s0 = 1.0, s1 = 1.0;
    float f = smoothstep(u_sm.x * 0.78, u_sm.x, d);
    if (f < 1.0) s0 = pcf(u_sm0, u_lm0, p, n, u_smw.x, u_sm.z);
    if (f > 0.0) {
        float fade = 1.0 - smoothstep(u_sm.y * 0.8, u_sm.y, d);
        s1 = mix(1.0, pcf(u_sm1, u_lm1, p, n, u_smw.y, u_sm.w), fade);
    }
    return mix(s0, s1, f);
}

vec3 fogged(vec3 col, vec3 p) {
    vec3 rd = p - u_cam;
    float t = length(rd);
    rd /= max(t, 1e-3);
    float b = u_fog.y;
    float ry = rd.y;
    float amount;
    float ho = u_cam.y - u_fog.z;
    if (abs(ry * b) < 1e-4) amount = u_fog.x * exp(-ho * b) * t;
    else amount = u_fog.x * exp(-ho * b) * (1.0 - exp(-t * ry * b)) / (ry * b);
    float fog = 1.0 - exp(-max(amount, 0.0));
    float sun = pow(max(dot(rd, u_sun_dir), 0.0), 6.0);
    vec3 fc = mix(u_fog_col, u_fog_sun, sun);
    return mix(col, fc, clamp(fog, 0.0, 0.97));
}

void main() {
    vec3 N = normalize(v_nrm);
    vec3 V = normalize(u_cam - v_pos);
    if (dot(N, V) < -0.2 && false) N = -N;
    float ndl = dot(N, u_sun_dir);
    float diff = clamp((ndl + 0.18) / 1.18, 0.0, 1.0);
    float sh = diff > 0.001 ? shadow(v_pos, N) : 1.0;
    vec3 amb = mix(u_gnd_amb, u_sky_amb, N.y * 0.5 + 0.5) * v_ex.x;
    // soft bounce of the sun colour on the side facing away from it
    vec3 bounce = u_sun_col * 0.045 * max(-ndl, 0.0) * v_ex.x;
    vec3 col = v_col * (u_sun_col * diff * sh + amb + bounce);
    if (u_gloss > 0.0) {
        vec3 H = normalize(u_sun_dir + V);
        float sp = pow(max(dot(N, H), 0.0), 90.0) * u_gloss * sh;
        float fr = pow(1.0 - max(dot(N, V), 0.0), 4.0) * u_gloss * 0.35;
        col += u_sun_col * sp * 0.4 + mix(u_gnd_amb, u_sky_amb, 0.8) * fr;
    }
    col += v_col * v_ex.y * 6.0;
    f_col = vec4(fogged(col, v_pos), 1.0);
}
'''

SKY_VS = '''#version 330
out vec2 v_ndc;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    v_ndc = p * 2.0 - 1.0;
    gl_Position = vec4(v_ndc, 0.9999, 1.0);
}
'''

SKY_FS = '''#version 330
in vec2 v_ndc;
out vec4 f_col;
uniform mat4 u_inv;            // inverse(proj * rotation-only view)
uniform vec3 u_sun_dir;
uniform vec3 u_sun_col;
uniform vec3 u_zenith;
uniform vec3 u_horizon;
uniform vec3 u_fog_sun;
uniform float u_time;
uniform vec3 u_cam;

float h21(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float vn(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(h21(i), h21(i + vec2(1, 0)), f.x), mix(h21(i + vec2(0, 1)), h21(i + vec2(1, 1)), f.x), f.y);
}
float fbm(vec2 p) { float a = 0.5, s = 0.0; for (int i = 0; i < 4; i++) { s += vn(p) * a; p *= 2.03; a *= 0.5; } return s; }

void main() {
    vec4 w = u_inv * vec4(v_ndc, 1.0, 1.0);
    vec3 d = normalize(w.xyz / w.w);
    float up = clamp(d.y, -0.2, 1.0);
    vec3 col = mix(u_horizon, u_zenith, pow(clamp(up, 0.0, 1.0), 0.55));
    float sd = max(dot(d, u_sun_dir), 0.0);
    col += u_fog_sun * pow(sd, 5.0) * 0.55 * (1.0 - up * 0.6);       // warm glow around the sun
    col += u_sun_col * pow(sd, 400.0) * 1.2;
    col += u_sun_col * smoothstep(0.99955, 0.99975, sd) * 14.0;      // the disc
    // stylised flat clouds on a plane overhead
    if (d.y > 0.02) {
        vec2 uv = (d.xz / (d.y + 0.18)) * 1.6 + vec2(u_time * 0.012, 0.0) + u_cam.xz * 0.00004;
        float c = fbm(uv * 1.3);
        float m = smoothstep(0.54, 0.58, c);
        float lit = smoothstep(0.48, 0.66, fbm(uv * 1.3 + u_sun_dir.xz * 0.12));
        vec3 cc = mix(vec3(0.62, 0.66, 0.78), vec3(1.05, 0.98, 0.9), lit);
        cc = mix(cc, u_fog_sun * 1.1, pow(sd, 3.0) * 0.5);
        float fade = smoothstep(0.02, 0.22, d.y);
        col = mix(col, cc, m * fade * 0.92);
    }
    f_col = vec4(col, 1.0);
}
'''

POST_VS = '''#version 330
out vec2 v_uv;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    v_uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
'''

BLOOM_FS = '''#version 330
in vec2 v_uv;
out vec4 f_col;
uniform sampler2D u_tex;
uniform vec2 u_dir;            // texel step
uniform float u_thresh;
void main() {
    vec3 s = vec3(0.0);
    float tw = 0.0;
    for (int i = -4; i <= 4; i++) {
        float w = exp(-float(i * i) * 0.18);
        vec3 c = texture(u_tex, v_uv + u_dir * float(i)).rgb;
        if (u_thresh > 0.0) { float l = max(max(c.r, c.g), c.b); c *= clamp((l - u_thresh) / max(l, 1e-3), 0.0, 4.0); }
        s += c * w; tw += w;
    }
    f_col = vec4(s / tw, 1.0);
}
'''

POST_FS = '''#version 330
in vec2 v_uv;
out vec4 f_col;
uniform sampler2D u_hdr;
uniform sampler2D u_bloom;
uniform sampler2D u_ui;
uniform float u_bloom_k;
uniform float u_exposure;
uniform float u_has_ui;
uniform vec2 u_res;

vec3 aces(vec3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec3 c = texture(u_hdr, v_uv).rgb + texture(u_bloom, v_uv).rgb * u_bloom_k;
    c *= u_exposure;
    c = aces(c);
    // a touch of warm contrast / saturation
    float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
    c = mix(vec3(l), c, 1.12);
    c = c * c * (3.0 - 2.0 * c) * 0.35 + c * 0.65;
    vec2 q = v_uv - 0.5;
    c *= 1.0 - dot(q, q) * 0.55;
    c = pow(c, vec3(1.0 / 2.2));
    if (u_has_ui > 0.5) {
        vec4 ui = texture(u_ui, vec2(v_uv.x, 1.0 - v_uv.y)).bgra;
        c = mix(c, ui.rgb, ui.a);
    }
    f_col = vec4(c, 1.0);
}
'''


PART_VS = '''#version 330
uniform mat4 u_vp;
in vec3 in_pos; in vec2 in_uv; in vec4 in_col;
out vec2 v_uv; out vec4 v_col; out vec3 v_pos;
void main() { gl_Position = u_vp * vec4(in_pos, 1.0); v_uv = in_uv; v_col = in_col; v_pos = in_pos; }
'''

PART_FS = '''#version 330
in vec2 v_uv; in vec4 v_col; in vec3 v_pos;
out vec4 f_col;
uniform vec3 u_cam;
uniform vec3 u_light;       // sun colour * 0.55 + ambient
uniform vec3 u_fog_col;
uniform vec3 u_fog;
void main() {
    float r = length(v_uv);
    float a = v_col.a * (1.0 - smoothstep(0.55, 1.0, r));
    vec3 col = v_col.rgb * u_light;
    float t = length(v_pos - u_cam);
    float fog = 1.0 - exp(-u_fog.x * t * 0.9);
    col = mix(col, u_fog_col, fog);
    f_col = vec4(col, a);
}
'''
