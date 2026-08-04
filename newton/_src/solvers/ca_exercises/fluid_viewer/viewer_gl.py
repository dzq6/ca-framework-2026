from __future__ import annotations

import ctypes
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import warp as wp

import newton.examples
import newton.viewer
from newton.viewer import ViewerGL

SSFR_RADIUS_SCALE = 1.35
SSFR_DEPTH_BLUR_RADIUS = 5
SSFR_THICKNESS_BLUR_RADIUS = 7
SSFR_DEPTH_FALLOFF = 35.0
SSFR_THICKNESS_SCALE = 4.8
SSFR_REFRACTION = 0.022
SSFR_FRESNEL_POWER = 5.2
SSFR_SPECULAR_STRENGTH = 0.8
SSFR_FLUID_COLOR = (0.08, 0.26, 0.5)
SSFR_ABSORPTION = (0.9, 0.32, 0.11)
SSFR_BASE_ALPHA = 0.035
SSFR_ALPHA_SCALE = 0.9
SSFR_EDGE_ALPHA = 0.18


@wp.kernel
def pack_particle_centers_and_radius(
    points: wp.array(dtype=wp.vec3),
    radius: float,
    out_particles: wp.array(dtype=wp.vec4),
):
    tid = wp.tid()
    p = points[tid]
    out_particles[tid] = wp.vec4(p[0], p[1], p[2], radius)


def _arr_pointer(arr: np.ndarray):
    return arr.astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _as_gl_id(handle) -> int:
    return int(handle.value) if hasattr(handle, "value") else int(handle)


class ScreenSpaceFluidRenderer:
    """Minimal SSFR pipeline for the APIC demo.

    Pipeline:
    1. Render particles as screen-space spheres into a fluid-depth target.
    2. Render additive thickness in a second pass.
    3. Bilaterally smooth the depth and Gaussian blur the thickness.
    4. Composite the reconstructed fluid surface over the viewer's scene color.
    """

    _fullscreen_vertex_shader = """
    #version 330 core
    layout (location = 0) in vec2 aPos;
    layout (location = 1) in vec2 aUV;
    out vec2 vUV;
    void main()
    {
        vUV = aUV;
        gl_Position = vec4(aPos, 0.0, 1.0);
    }
    """

    _particle_depth_vertex_shader = """
    #version 330 core
    layout (location = 0) in vec4 aParticle;

    uniform mat4 view;
    uniform mat4 projection;
    uniform float viewport_height;
    uniform vec2 viewport_size;
    uniform float radius_scale;

    out vec3 vCenterView;
    out float vRadius;
    out vec2 vCenterPx;
    out float vPointSize;

    void main()
    {
        vec4 centerView = view * vec4(aParticle.xyz, 1.0);
        vCenterView = centerView.xyz;
        vRadius = aParticle.w * radius_scale;

        vec4 centerClip = projection * centerView;
        float pointSize = max(1.0, viewport_height * projection[1][1] * vRadius / max(1.0e-5, -centerView.z));
        vec2 centerNdc = centerClip.xy / max(1.0e-6, centerClip.w);

        gl_Position = centerClip;
        gl_PointSize = pointSize;
        vCenterPx = (centerNdc * 0.5 + 0.5) * viewport_size;
        vPointSize = pointSize;
    }
    """

    _particle_depth_fragment_shader = """
    #version 330 core
    uniform mat4 projection;

    in vec3 vCenterView;
    in float vRadius;
    in vec2 vCenterPx;
    in float vPointSize;

    layout (location = 0) out float outDepth;

    void main()
    {
        vec2 xy = (gl_FragCoord.xy - vCenterPx) / max(1.0e-6, 0.5 * vPointSize);
        float r2 = dot(xy, xy);
        if (r2 > 1.0)
            discard;

        float nz = sqrt(max(0.0, 1.0 - r2));
        vec3 viewPos = vCenterView + vec3(xy * vRadius, nz * vRadius);
        vec4 clip = projection * vec4(viewPos, 1.0);
        float depth01 = clip.z / clip.w * 0.5 + 0.5;

        gl_FragDepth = depth01;
        outDepth = -viewPos.z;
    }
    """

    _particle_thickness_vertex_shader = """
    #version 330 core
    layout (location = 0) in vec4 aParticle;

    uniform mat4 view;
    uniform mat4 projection;
    uniform float viewport_height;
    uniform vec2 viewport_size;
    uniform float radius_scale;

    out float vRadius;
    out vec2 vCenterPx;
    out float vPointSize;

    void main()
    {
        vec4 centerView = view * vec4(aParticle.xyz, 1.0);
        vRadius = aParticle.w * radius_scale;

        vec4 centerClip = projection * centerView;
        float pointSize = max(1.0, viewport_height * projection[1][1] * vRadius / max(1.0e-5, -centerView.z));
        vec2 centerNdc = centerClip.xy / max(1.0e-6, centerClip.w);

        gl_Position = centerClip;
        gl_PointSize = pointSize;
        vCenterPx = (centerNdc * 0.5 + 0.5) * viewport_size;
        vPointSize = pointSize;
    }
    """

    _particle_thickness_fragment_shader = """
    #version 330 core
    in float vRadius;
    in vec2 vCenterPx;
    in float vPointSize;
    layout (location = 0) out float outThickness;

    void main()
    {
        vec2 xy = (gl_FragCoord.xy - vCenterPx) / max(1.0e-6, 0.5 * vPointSize);
        float r2 = dot(xy, xy);
        if (r2 > 1.0)
            discard;

        float nz = sqrt(max(0.0, 1.0 - r2));
        outThickness = 2.0 * nz * vRadius;
    }
    """

    _depth_blur_fragment_shader = """
    #version 330 core
    in vec2 vUV;

    uniform sampler2D source_tex;
    uniform vec2 texel_size;
    uniform vec2 axis;
    uniform int blur_radius;
    uniform float depth_falloff;

    layout (location = 0) out float outDepth;

    void main()
    {
        float center = texture(source_tex, vUV).r;
        if (center <= 0.0)
        {
            outDepth = 0.0;
            return;
        }

        float sum = 0.0;
        float weight_sum = 0.0;

        for (int i = -8; i <= 8; ++i)
        {
            if (abs(i) > blur_radius)
                continue;

            vec2 uv = vUV + axis * texel_size * float(i);
            float sample_depth = texture(source_tex, uv).r;
            if (sample_depth <= 0.0)
                continue;

            float spatial = exp(-0.5 * float(i * i) / max(1.0, float(blur_radius * blur_radius)));
            float range = exp(-abs(sample_depth - center) * depth_falloff);
            float w = spatial * range;

            sum += sample_depth * w;
            weight_sum += w;
        }

        outDepth = weight_sum > 0.0 ? sum / weight_sum : center;
    }
    """

    _thickness_blur_fragment_shader = """
    #version 330 core
    in vec2 vUV;

    uniform sampler2D source_tex;
    uniform vec2 texel_size;
    uniform vec2 axis;
    uniform int blur_radius;

    layout (location = 0) out float outThickness;

    void main()
    {
        float sum = 0.0;
        float weight_sum = 0.0;

        for (int i = -10; i <= 10; ++i)
        {
            if (abs(i) > blur_radius)
                continue;

            vec2 uv = vUV + axis * texel_size * float(i);
            float sample_thickness = texture(source_tex, uv).r;
            float spatial = exp(-0.5 * float(i * i) / max(1.0, float(blur_radius * blur_radius)));
            sum += sample_thickness * spatial;
            weight_sum += spatial;
        }

        outThickness = weight_sum > 0.0 ? sum / weight_sum : 0.0;
    }
    """

    _composite_fragment_shader = """
    #version 330 core
    in vec2 vUV;

    uniform sampler2D scene_color_tex;
    uniform sampler2D scene_depth_tex;
    uniform sampler2D fluid_depth_tex;
    uniform sampler2D fluid_thickness_tex;

    uniform vec2 inv_screen;
    uniform vec2 proj_scale;
    uniform float near_plane;
    uniform float far_plane;
    uniform vec3 fluid_color;
    uniform vec3 absorption_coeff;
    uniform float refraction_scale;
    uniform float fresnel_power;
    uniform float specular_strength;
    uniform float thickness_scale;
    uniform float base_alpha;
    uniform float alpha_scale;
    uniform float edge_alpha;
    uniform vec3 light_dir_view;

    out vec4 FragColor;

    float linearize_depth(float depth)
    {
        if (depth >= 1.0)
            return far_plane;

        float z = depth * 2.0 - 1.0;
        return (2.0 * near_plane * far_plane) / (far_plane + near_plane - z * (far_plane - near_plane));
    }

    vec3 reconstruct_view_pos(vec2 uv, float linear_depth)
    {
        vec2 ndc = uv * 2.0 - 1.0;
        return vec3(
            ndc.x * linear_depth / max(1.0e-6, proj_scale.x),
            ndc.y * linear_depth / max(1.0e-6, proj_scale.y),
            -linear_depth
        );
    }

    void main()
    {
        vec3 scene_color = texture(scene_color_tex, vUV).rgb;
        float fluid_depth = texture(fluid_depth_tex, vUV).r;
        float thickness = texture(fluid_thickness_tex, vUV).r * thickness_scale;

        if (fluid_depth <= 0.0 || thickness <= 1.0e-5)
        {
            FragColor = vec4(scene_color, 1.0);
            return;
        }

        float scene_depth = linearize_depth(texture(scene_depth_tex, vUV).r);
        if (fluid_depth >= scene_depth - 1.0e-3)
        {
            FragColor = vec4(scene_color, 1.0);
            return;
        }

        float dL = texture(fluid_depth_tex, vUV - vec2(inv_screen.x, 0.0)).r;
        float dR = texture(fluid_depth_tex, vUV + vec2(inv_screen.x, 0.0)).r;
        float dD = texture(fluid_depth_tex, vUV - vec2(0.0, inv_screen.y)).r;
        float dU = texture(fluid_depth_tex, vUV + vec2(0.0, inv_screen.y)).r;

        dL = dL > 0.0 ? dL : fluid_depth;
        dR = dR > 0.0 ? dR : fluid_depth;
        dD = dD > 0.0 ? dD : fluid_depth;
        dU = dU > 0.0 ? dU : fluid_depth;

        vec3 pL = reconstruct_view_pos(vUV - vec2(inv_screen.x, 0.0), dL);
        vec3 pR = reconstruct_view_pos(vUV + vec2(inv_screen.x, 0.0), dR);
        vec3 pD = reconstruct_view_pos(vUV - vec2(0.0, inv_screen.y), dD);
        vec3 pU = reconstruct_view_pos(vUV + vec2(0.0, inv_screen.y), dU);

        vec3 normal = normalize(cross(pR - pL, pU - pD));
        if (normal.z < 0.0)
            normal = -normal;

        vec3 view_dir = vec3(0.0, 0.0, 1.0);
        vec3 light_dir = normalize(light_dir_view);
        vec3 half_dir = normalize(light_dir + view_dir);

        float fresnel = pow(1.0 - clamp(dot(normal, view_dir), 0.0, 1.0), fresnel_power);
        float specular = pow(max(dot(normal, half_dir), 0.0), 96.0) * specular_strength;

        vec2 refract_uv = clamp(vUV + normal.xy * refraction_scale, vec2(0.001), vec2(0.999));
        vec3 refracted_scene = texture(scene_color_tex, refract_uv).rgb;
        vec3 transmittance = exp(-absorption_coeff * thickness);
        vec3 body_color = refracted_scene * transmittance + fluid_color * (1.0 - transmittance);
        vec3 surface_color = body_color + vec3(specular) + fresnel * 0.16;

        float body_alpha = 1.0 - dot(transmittance, vec3(0.3333333));
        float alpha = clamp(base_alpha + body_alpha * alpha_scale + fresnel * edge_alpha, 0.0, 0.92);
        FragColor = vec4(mix(scene_color, surface_color, alpha), 1.0);
    }
    """

    def __init__(self, viewer: ViewerGL, max_particles: int, particle_radius: float, device):
        self.viewer = viewer
        self.device = device
        self.max_particles = max_particles
        self.particle_radius = float(particle_radius)
        self.radius_scale = SSFR_RADIUS_SCALE
        self.depth_blur_radius = SSFR_DEPTH_BLUR_RADIUS
        self.thickness_blur_radius = SSFR_THICKNESS_BLUR_RADIUS
        self.depth_falloff = SSFR_DEPTH_FALLOFF
        self.thickness_scale = SSFR_THICKNESS_SCALE
        self.refraction_scale = SSFR_REFRACTION
        self.fresnel_power = SSFR_FRESNEL_POWER
        self.specular_strength = SSFR_SPECULAR_STRENGTH
        self.fluid_color = np.array(SSFR_FLUID_COLOR, dtype=np.float32)
        self.absorption = np.array(SSFR_ABSORPTION, dtype=np.float32)
        self.base_alpha = SSFR_BASE_ALPHA
        self.alpha_scale = SSFR_ALPHA_SCALE
        self.edge_alpha = SSFR_EDGE_ALPHA

        self._gl = None
        self._initialized = False
        self._failed = False
        self._particle_count = 0
        self._particle_positions = None
        self._frame_size = (-1, -1)

        self._particle_vao = None
        self._particle_vbo = None
        self._particle_buffer = None

        self._depth_fbo = None
        self._depth_rbo = None
        self._thickness_fbo = None
        self._blur_fbo = None

        self._depth_tex = None
        self._depth_ping_tex = None
        self._thickness_tex = None
        self._thickness_ping_tex = None

        self._depth_program = None
        self._thickness_program = None
        self._depth_blur_program = None
        self._thickness_blur_program = None
        self._composite_program = None
        self._uniforms = {}

    @property
    def available(self) -> bool:
        return not self._failed

    def set_particles(self, points: wp.array(dtype=wp.vec3) | None):
        self._particle_positions = points
        self._particle_count = 0 if points is None else len(points)

    def render(self, viewer: ViewerGL):
        if self._failed or self._particle_positions is None or self._particle_count == 0:
            return

        try:
            self._ensure_initialized()
            self._ensure_frame_targets()
            self._upload_particles(self._particle_positions)
            self._render_passes()
        except Exception as exc:
            self._failed = True
            print(f"[SSFR] disabling screen-space fluid rendering: {exc}")

    def _ensure_initialized(self):
        if self._initialized:
            return

        from pyglet import gl
        from pyglet.graphics.shader import Shader, ShaderProgram

        self._gl = gl

        def compile_program(vs: str, fs: str):
            return ShaderProgram(Shader(vs, "vertex"), Shader(fs, "fragment"))

        self._depth_program = compile_program(self._particle_depth_vertex_shader, self._particle_depth_fragment_shader)
        self._thickness_program = compile_program(
            self._particle_thickness_vertex_shader,
            self._particle_thickness_fragment_shader,
        )
        self._depth_blur_program = compile_program(self._fullscreen_vertex_shader, self._depth_blur_fragment_shader)
        self._thickness_blur_program = compile_program(
            self._fullscreen_vertex_shader,
            self._thickness_blur_fragment_shader,
        )
        self._composite_program = compile_program(self._fullscreen_vertex_shader, self._composite_fragment_shader)

        self._particle_vao = gl.GLuint()
        gl.glGenVertexArrays(1, self._particle_vao)
        gl.glBindVertexArray(self._particle_vao)

        self._particle_vbo = gl.GLuint()
        gl.glGenBuffers(1, self._particle_vbo)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._particle_vbo)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER,
            self.max_particles * 16,
            None,
            gl.GL_DYNAMIC_DRAW,
        )
        gl.glVertexAttribPointer(0, 4, gl.GL_FLOAT, gl.GL_FALSE, 16, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)
        gl.glBindVertexArray(0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

        self._particle_buffer = wp.RegisteredGLBuffer(
            gl_buffer_id=_as_gl_id(self._particle_vbo),
            device=self.device,
            flags=wp.RegisteredGLBuffer.WRITE_DISCARD,
        )

        self._depth_fbo = gl.GLuint()
        gl.glGenFramebuffers(1, self._depth_fbo)
        self._depth_rbo = gl.GLuint()
        gl.glGenRenderbuffers(1, self._depth_rbo)

        self._thickness_fbo = gl.GLuint()
        gl.glGenFramebuffers(1, self._thickness_fbo)

        self._blur_fbo = gl.GLuint()
        gl.glGenFramebuffers(1, self._blur_fbo)

        self._uniforms = {
            "depth": self._collect_uniforms(
                self._depth_program,
                "view",
                "projection",
                "viewport_height",
                "viewport_size",
                "radius_scale",
            ),
            "thickness": self._collect_uniforms(
                self._thickness_program,
                "view",
                "projection",
                "viewport_height",
                "viewport_size",
                "radius_scale",
            ),
            "depth_blur": self._collect_uniforms(
                self._depth_blur_program,
                "source_tex",
                "texel_size",
                "axis",
                "blur_radius",
                "depth_falloff",
            ),
            "thickness_blur": self._collect_uniforms(
                self._thickness_blur_program,
                "source_tex",
                "texel_size",
                "axis",
                "blur_radius",
            ),
            "composite": self._collect_uniforms(
                self._composite_program,
                "scene_color_tex",
                "scene_depth_tex",
                "fluid_depth_tex",
                "fluid_thickness_tex",
                "inv_screen",
                "proj_scale",
                "near_plane",
                "far_plane",
                "fluid_color",
                "absorption_coeff",
                "refraction_scale",
                "fresnel_power",
                "specular_strength",
                "thickness_scale",
                "base_alpha",
                "alpha_scale",
                "edge_alpha",
                "light_dir_view",
            ),
        }

        self._initialized = True

    def _ensure_frame_targets(self):
        renderer = self.viewer.renderer
        width = int(renderer._screen_width)
        height = int(renderer._screen_height)
        if (width, height) == self._frame_size:
            return

        gl = self._gl
        self._frame_size = (width, height)

        self._depth_tex = self._create_red_texture(width, height)
        self._depth_ping_tex = self._create_red_texture(width, height)
        self._thickness_tex = self._create_red_texture(width, height)
        self._thickness_ping_tex = self._create_red_texture(width, height)

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._depth_fbo)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self._depth_tex, 0)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self._depth_rbo)
        gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT24, width, height)
        gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_RENDERBUFFER, self._depth_rbo)

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._thickness_fbo)
        gl.glFramebufferTexture2D(
            gl.GL_FRAMEBUFFER,
            gl.GL_COLOR_ATTACHMENT0,
            gl.GL_TEXTURE_2D,
            self._thickness_tex,
            0,
        )

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

    def _create_red_texture(self, width: int, height: int):
        gl = self._gl
        texture = gl.GLuint()
        gl.glGenTextures(1, texture)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_R32F,
            width,
            height,
            0,
            gl.GL_RED,
            gl.GL_FLOAT,
            None,
        )
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        return texture

    def _collect_uniforms(self, program, *names: str) -> dict[str, int]:
        gl = self._gl
        return {name: gl.glGetUniformLocation(program.id, ctypes.c_char_p(name.encode("utf-8"))) for name in names}

    def _upload_particles(self, points: wp.array(dtype=wp.vec3)):
        if self._particle_count <= 0:
            return

        mapped = self._particle_buffer.map(dtype=wp.vec4, shape=(self._particle_count,))
        wp.launch(
            pack_particle_centers_and_radius,
            dim=self._particle_count,
            inputs=[points, self.particle_radius],
            outputs=[mapped],
            device=self.device,
            record_tape=False,
        )
        self._particle_buffer.unmap()

    def _render_passes(self):
        gl = self._gl
        renderer = self.viewer.renderer
        camera = self.viewer.camera
        width, height = self._frame_size
        view = np.ascontiguousarray(np.asarray(camera.get_view_matrix(), dtype=np.float32).reshape(4, 4))
        proj = np.ascontiguousarray(np.asarray(camera.get_projection_matrix(), dtype=np.float32).reshape(4, 4))
        texel = np.array((1.0 / width, 1.0 / height), dtype=np.float32)

        gl.glBindVertexArray(self._particle_vao)

        # Pass 1: front-most fluid depth.
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._depth_fbo)
        gl.glViewport(0, 0, width, height)
        gl.glClearColor(0.0, 0.0, 0.0, 0.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
        gl.glDepthMask(True)
        gl.glDisable(gl.GL_BLEND)
        gl.glUseProgram(self._depth_program.id)
        self._set_particle_uniforms(self._uniforms["depth"], view, proj, height)
        gl.glDrawArrays(gl.GL_POINTS, 0, self._particle_count)

        # Pass 2: thickness accumulation.
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._thickness_fbo)
        gl.glClearColor(0.0, 0.0, 0.0, 0.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
        gl.glDepthMask(False)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_ONE, gl.GL_ONE)
        gl.glUseProgram(self._thickness_program.id)
        self._set_particle_uniforms(self._uniforms["thickness"], view, proj, height)
        gl.glDrawArrays(gl.GL_POINTS, 0, self._particle_count)
        gl.glDisable(gl.GL_BLEND)
        gl.glBindVertexArray(0)

        # Pass 3: blur depth and thickness.
        self._blur_depth(self._depth_tex, self._depth_ping_tex, texel)
        self._blur_depth(self._depth_ping_tex, self._depth_tex, texel, horizontal=False)
        self._blur_thickness(self._thickness_tex, self._thickness_ping_tex, texel)
        self._blur_thickness(self._thickness_ping_tex, self._thickness_tex, texel, horizontal=False)

        # Pass 4: composite onto the default framebuffer using the viewer scene textures.
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glViewport(0, 0, width, height)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDepthMask(False)
        gl.glUseProgram(self._composite_program.id)

        uniforms = self._uniforms["composite"]
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, renderer._frame_texture)
        gl.glUniform1i(uniforms["scene_color_tex"], 0)

        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, renderer._frame_depth_texture)
        gl.glUniform1i(uniforms["scene_depth_tex"], 1)

        gl.glActiveTexture(gl.GL_TEXTURE2)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._depth_tex)
        gl.glUniform1i(uniforms["fluid_depth_tex"], 2)

        gl.glActiveTexture(gl.GL_TEXTURE3)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._thickness_tex)
        gl.glUniform1i(uniforms["fluid_thickness_tex"], 3)

        gl.glUniform2f(uniforms["inv_screen"], texel[0], texel[1])
        gl.glUniform2f(uniforms["proj_scale"], float(proj[0, 0]), float(proj[1, 1]))
        gl.glUniform1f(uniforms["near_plane"], float(camera.near))
        gl.glUniform1f(uniforms["far_plane"], float(camera.far))
        gl.glUniform3f(uniforms["fluid_color"], *self.fluid_color)
        gl.glUniform3f(uniforms["absorption_coeff"], *self.absorption)
        gl.glUniform1f(uniforms["refraction_scale"], self.refraction_scale)
        gl.glUniform1f(uniforms["fresnel_power"], self.fresnel_power)
        gl.glUniform1f(uniforms["specular_strength"], self.specular_strength)
        gl.glUniform1f(uniforms["thickness_scale"], self.thickness_scale)
        gl.glUniform1f(uniforms["base_alpha"], self.base_alpha)
        gl.glUniform1f(uniforms["alpha_scale"], self.alpha_scale)
        gl.glUniform1f(uniforms["edge_alpha"], self.edge_alpha)

        sun_world = np.asarray(renderer._sun_direction, dtype=np.float32)
        light_dir_view = view[:3, :3] @ sun_world
        norm = np.linalg.norm(light_dir_view)
        if norm > 0.0:
            light_dir_view = light_dir_view / norm
        gl.glUniform3f(uniforms["light_dir_view"], *light_dir_view.astype(np.float32))

        gl.glBindVertexArray(renderer._frame_vao)
        gl.glDrawElements(gl.GL_TRIANGLES, len(renderer._frame_indices), gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)

        gl.glActiveTexture(gl.GL_TEXTURE3)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glActiveTexture(gl.GL_TEXTURE2)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glUseProgram(0)

    def _set_particle_uniforms(self, uniforms: dict[str, int], view: np.ndarray, proj: np.ndarray, height: int):
        gl = self._gl
        gl.glUniformMatrix4fv(uniforms["view"], 1, gl.GL_FALSE, _arr_pointer(view))
        gl.glUniformMatrix4fv(uniforms["projection"], 1, gl.GL_FALSE, _arr_pointer(proj))
        gl.glUniform1f(uniforms["viewport_height"], float(height))
        gl.glUniform2f(uniforms["viewport_size"], float(self._frame_size[0]), float(self._frame_size[1]))
        gl.glUniform1f(uniforms["radius_scale"], float(self.radius_scale))

    def _blur_depth(self, source_tex, target_tex, texel: np.ndarray, horizontal: bool = True):
        gl = self._gl
        uniforms = self._uniforms["depth_blur"]
        axis = (1.0, 0.0) if horizontal else (0.0, 1.0)

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._blur_fbo)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, target_tex, 0)
        gl.glClearColor(0.0, 0.0, 0.0, 0.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glUseProgram(self._depth_blur_program.id)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, source_tex)
        gl.glUniform1i(uniforms["source_tex"], 0)
        gl.glUniform2f(uniforms["texel_size"], texel[0], texel[1])
        gl.glUniform2f(uniforms["axis"], *axis)
        gl.glUniform1i(uniforms["blur_radius"], int(self.depth_blur_radius))
        gl.glUniform1f(uniforms["depth_falloff"], float(self.depth_falloff))
        gl.glBindVertexArray(self.viewer.renderer._frame_vao)
        gl.glDrawElements(gl.GL_TRIANGLES, len(self.viewer.renderer._frame_indices), gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)

    def _blur_thickness(self, source_tex, target_tex, texel: np.ndarray, horizontal: bool = True):
        gl = self._gl
        uniforms = self._uniforms["thickness_blur"]
        axis = (1.0, 0.0) if horizontal else (0.0, 1.0)

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._blur_fbo)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, target_tex, 0)
        gl.glClearColor(0.0, 0.0, 0.0, 0.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glUseProgram(self._thickness_blur_program.id)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, source_tex)
        gl.glUniform1i(uniforms["source_tex"], 0)
        gl.glUniform2f(uniforms["texel_size"], texel[0], texel[1])
        gl.glUniform2f(uniforms["axis"], *axis)
        gl.glUniform1i(uniforms["blur_radius"], int(self.thickness_blur_radius))
        gl.glBindVertexArray(self.viewer.renderer._frame_vao)
        gl.glDrawElements(gl.GL_TRIANGLES, len(self.viewer.renderer._frame_indices), gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)


class FluidViewerGL(ViewerGL):
    """ViewerGL extension that adds post-render callbacks for screen-space effects."""

    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        vsync: bool = False,
        headless: bool = False,
    ):
        super().__init__(width=width, height=height, vsync=vsync, headless=headless)
        self._post_render_callbacks: list[Callable[[FluidViewerGL], None]] = []

    def register_post_render_callback(self, callback: Callable[[FluidViewerGL], None]):
        """Register a callback executed after scene rendering and before UI/present."""
        if not callable(callback):
            raise TypeError("callback must be callable")
        self._post_render_callbacks.append(callback)

    def _update(self):
        """Mirror ViewerGL._update while inserting post-render callbacks."""
        self.renderer.update()

        now = time.perf_counter()
        dt = max(0.0, min(0.1, now - self._last_time))
        self._last_time = now
        self._update_camera(dt)

        self.wind.update(dt)

        if self.renderer.has_exit():
            return

        self.renderer.render(self.camera, self.objects, self.lines)

        for callback in self._post_render_callbacks:
            callback(self)

        self._update_fps()

        if self.ui and self.ui.is_available and self.show_ui:
            self.ui.begin_frame()
            self._render_ui()
            self.ui.end_frame()
            self.ui.render()

        self.renderer.present()


def init(parser: Any = None):
    """Mirror newton.examples.init, but use FluidViewerGL for the OpenGL path."""
    if parser is None:
        parser = newton.examples.create_parser()
        args = parser.parse_known_args()[0]
    else:
        args = parser.parse_args()

    if args.quiet:
        wp.config.quiet = True

    if args.device:
        wp.set_device(args.device)

    if args.viewer == "gl":
        viewer = FluidViewerGL(headless=args.headless)
    elif args.viewer == "usd":
        if args.output_path is None:
            raise ValueError("--output-path is required when using usd viewer")
        viewer = newton.viewer.ViewerUSD(output_path=args.output_path, num_frames=args.num_frames)
    elif args.viewer == "rerun":
        viewer = newton.viewer.ViewerRerun(address=args.rerun_address)
    elif args.viewer == "null":
        viewer = newton.viewer.ViewerNull(num_frames=args.num_frames)
    elif args.viewer == "viser":
        viewer = newton.viewer.ViewerViser()
    else:
        raise ValueError(f"Invalid viewer: {args.viewer}")

    return viewer, args
