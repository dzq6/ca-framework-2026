from __future__ import annotations

import ctypes

import numpy as np
import warp as wp


@wp.func
def sample_dense_scalar(
    world_x: float,
    world_y: float,
    world_z: float,
    grid_min: wp.vec3,
    cell_size: float,
    nx: int,
    ny: int,
    nz: int,
    density: wp.array3d(dtype=float),
) -> float:
    """Trilinear sample of a dense cell-centered scalar grid at a world-space point.

    The grid's cell (i, j, k) is centered at ``grid_min + (i+0.5, j+0.5, k+0.5) * cell_size``.
    Samples outside the grid AABB return 0.
    """
    fx = (world_x - grid_min[0]) / cell_size - 0.5
    fy = (world_y - grid_min[1]) / cell_size - 0.5
    fz = (world_z - grid_min[2]) / cell_size - 0.5

    if fx < 0.0 or fy < 0.0 or fz < 0.0:
        return 0.0
    if fx > float(nx - 1) or fy > float(ny - 1) or fz > float(nz - 1):
        return 0.0

    i0 = int(fx)
    j0 = int(fy)
    k0 = int(fz)
    i1 = wp.min(i0 + 1, nx - 1)
    j1 = wp.min(j0 + 1, ny - 1)
    k1 = wp.min(k0 + 1, nz - 1)

    tx = fx - float(i0)
    ty = fy - float(j0)
    tz = fz - float(k0)

    c000 = density[i0, j0, k0]
    c100 = density[i1, j0, k0]
    c010 = density[i0, j1, k0]
    c110 = density[i1, j1, k0]
    c001 = density[i0, j0, k1]
    c101 = density[i1, j0, k1]
    c011 = density[i0, j1, k1]
    c111 = density[i1, j1, k1]

    c00 = c000 * (1.0 - tx) + c100 * tx
    c10 = c010 * (1.0 - tx) + c110 * tx
    c01 = c001 * (1.0 - tx) + c101 * tx
    c11 = c011 * (1.0 - tx) + c111 * tx

    c0 = c00 * (1.0 - ty) + c10 * ty
    c1 = c01 * (1.0 - ty) + c11 * ty

    return c0 * (1.0 - tz) + c1 * tz


@wp.kernel
def build_smoke_volume_texture(
    density: wp.array3d(dtype=float),
    source_grid_min: wp.vec3,
    source_cell_size: float,
    source_nx: int,
    source_ny: int,
    source_nz: int,
    world_min: wp.vec3,
    world_max: wp.vec3,
    target_size: wp.vec3i,
    density_scale: float,
    out_volume: wp.array3d(dtype=float),
):
    i, j, k = wp.tid()
    nx = target_size[0]
    ny = target_size[1]
    nz = target_size[2]

    uvw = wp.vec3(
        (wp.float32(i) + 0.5) / wp.float32(nx),
        (wp.float32(j) + 0.5) / wp.float32(ny),
        (wp.float32(k) + 0.5) / wp.float32(nz),
    )
    world_pos = wp.vec3(
        world_min[0] + (world_max[0] - world_min[0]) * uvw[0],
        world_min[1] + (world_max[1] - world_min[1]) * uvw[1],
        world_min[2] + (world_max[2] - world_min[2]) * uvw[2],
    )
    sampled = sample_dense_scalar(
        world_pos[0],
        world_pos[1],
        world_pos[2],
        source_grid_min,
        source_cell_size,
        source_nx,
        source_ny,
        source_nz,
        density,
    )
    out_volume[i, j, k] = wp.max(0.0, sampled * density_scale)


def _arr_pointer(arr: np.ndarray):
    return arr.astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))


class SmokeVolumeRenderer:
    """Realtime post-render smoke volume compositing for dense 3D smoke grids."""

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

    _composite_fragment_shader = """
    #version 330 core
    in vec2 vUV;

    uniform sampler2D scene_color_tex;
    uniform sampler3D density_tex;

    uniform vec3 camera_pos_world;
    uniform vec3 camera_front_world;
    uniform vec3 camera_right_world;
    uniform vec3 camera_up_world;
    uniform vec3 volume_min_world;
    uniform vec3 volume_max_world;
    uniform vec3 light_dir_world;
    uniform vec3 smoke_color_low;
    uniform vec3 smoke_color_high;
    uniform float density_cutoff;
    uniform float absorption;
    uniform float opacity_boost;
    uniform float ambient_strength;
    uniform float light_strength;
    uniform float anisotropy;
    uniform float tan_half_fov;
    uniform float aspect_ratio;
    uniform float step_size_world;
    uniform float shadow_step_world;
    uniform int max_steps;
    uniform int shadow_steps;

    out vec4 FragColor;

    float saturate(float x)
    {
        return clamp(x, 0.0, 1.0);
    }

    bool intersect_aabb(vec3 ray_origin, vec3 ray_dir, vec3 box_min, vec3 box_max, out float t_min, out float t_max)
    {
        vec3 safe_dir = vec3(
            abs(ray_dir.x) < 1.0e-6 ? (ray_dir.x >= 0.0 ? 1.0e-6 : -1.0e-6) : ray_dir.x,
            abs(ray_dir.y) < 1.0e-6 ? (ray_dir.y >= 0.0 ? 1.0e-6 : -1.0e-6) : ray_dir.y,
            abs(ray_dir.z) < 1.0e-6 ? (ray_dir.z >= 0.0 ? 1.0e-6 : -1.0e-6) : ray_dir.z
        );
        vec3 inv_dir = 1.0 / safe_dir;
        vec3 t0 = (box_min - ray_origin) * inv_dir;
        vec3 t1 = (box_max - ray_origin) * inv_dir;
        vec3 tsmaller = min(t0, t1);
        vec3 tbigger = max(t0, t1);
        t_min = max(max(tsmaller.x, tsmaller.y), tsmaller.z);
        t_max = min(min(tbigger.x, tbigger.y), tbigger.z);
        return t_max >= max(t_min, 0.0);
    }

    vec3 world_to_volume_uv(vec3 world_pos)
    {
        return (world_pos - volume_min_world) / max(volume_max_world - volume_min_world, vec3(1.0e-6));
    }

    float density_at(vec3 world_pos)
    {
        vec3 uvw = world_to_volume_uv(world_pos);
        if (any(lessThan(uvw, vec3(0.0))) || any(greaterThan(uvw, vec3(1.0))))
            return 0.0;
        return texture(density_tex, uvw).r;
    }

    float shadow_transmittance(vec3 world_pos)
    {
        float optical_depth = 0.0;
        vec3 sample_pos = world_pos;
        for (int i = 0; i < 24; ++i)
        {
            if (i >= shadow_steps)
                break;
            sample_pos += light_dir_world * shadow_step_world;
            vec3 uvw = world_to_volume_uv(sample_pos);
            if (any(lessThan(uvw, vec3(0.0))) || any(greaterThan(uvw, vec3(1.0))))
                break;
            optical_depth += density_at(sample_pos) * shadow_step_world;
        }
        return exp(-optical_depth * absorption * 1.1);
    }

    void main()
    {
        vec3 scene_color = texture(scene_color_tex, vUV).rgb;
        vec3 ray_origin = camera_pos_world;
        vec2 ndc = vUV * 2.0 - 1.0;
        vec3 ray_dir = normalize(
            camera_front_world
            + camera_right_world * ndc.x * tan_half_fov * aspect_ratio
            + camera_up_world * ndc.y * tan_half_fov
        );

        float t_enter;
        float t_exit;
        if (!intersect_aabb(ray_origin, ray_dir, volume_min_world, volume_max_world, t_enter, t_exit))
        {
            FragColor = vec4(scene_color, 1.0);
            return;
        }

        float t = max(t_enter, 0.0);
        if (t >= t_exit)
        {
            FragColor = vec4(scene_color, 1.0);
            return;
        }

        float jitter = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
        t += jitter * step_size_world;

        vec3 light_dir = normalize(light_dir_world);
        vec3 view_dir = -ray_dir;
        float transmittance = 1.0;
        vec3 scattered = vec3(0.0);

        for (int i = 0; i < 192; ++i)
        {
            if (i >= max_steps || t >= t_exit || transmittance <= 0.015)
                break;

            vec3 world_pos = ray_origin + ray_dir * t;
            float density = density_at(world_pos);
            if (density > density_cutoff)
            {
                float extinction = density * absorption;
                float sample_alpha = 1.0 - exp(-extinction * step_size_world * opacity_boost);

                vec3 uvw = world_to_volume_uv(world_pos);
                float dx = texture(density_tex, uvw + vec3(0.01, 0.0, 0.0)).r - texture(density_tex, uvw - vec3(0.01, 0.0, 0.0)).r;
                float dy = texture(density_tex, uvw + vec3(0.0, 0.01, 0.0)).r - texture(density_tex, uvw - vec3(0.0, 0.01, 0.0)).r;
                float dz = texture(density_tex, uvw + vec3(0.0, 0.0, 0.01)).r - texture(density_tex, uvw - vec3(0.0, 0.0, 0.01)).r;
                vec3 gradient = vec3(dx, dy, dz);
                vec3 normal = length(gradient) > 1.0e-5 ? normalize(gradient) : vec3(0.0, 0.0, 1.0);

                float shadow = shadow_transmittance(world_pos);
                float forward = dot(light_dir, view_dir);
                float phase = (1.0 - anisotropy * anisotropy) /
                    pow(max(1.0 + anisotropy * anisotropy - 2.0 * anisotropy * forward, 1.0e-4), 1.5);
                phase *= 0.07957747;

                float rim = pow(1.0 - saturate(dot(normal, view_dir)), 2.2);
                float lambert = 0.5 + 0.5 * saturate(dot(normal, light_dir));
                float tone = saturate(density * 1.35);
                vec3 smoke_color = mix(smoke_color_low, smoke_color_high, tone);
                vec3 lighting = smoke_color * (
                    ambient_strength +
                    light_strength * shadow * (0.55 * lambert + 1.8 * phase + 0.35 * rim)
                );

                scattered += transmittance * sample_alpha * lighting;
                transmittance *= 1.0 - sample_alpha;
            }

            t += step_size_world;
        }

        vec3 out_color = scene_color * transmittance + scattered;
        FragColor = vec4(out_color, 1.0);
    }
    """

    def __init__(
        self,
        viewer,
        world_min: tuple[float, float, float],
        world_max: tuple[float, float, float],
        volume_resolution: tuple[int, int, int] = (64, 64, 96),
        density_scale: float = 4.8,
    ):
        self.viewer = viewer
        self.world_min = np.asarray(world_min, dtype=np.float32)
        self.world_max = np.asarray(world_max, dtype=np.float32)
        self.volume_resolution = tuple(int(v) for v in volume_resolution)
        self.density_scale = float(density_scale)

        self.absorption = 0.65
        self.opacity_boost = 1.85
        self.ambient_strength = 0.26
        self.light_strength = 2.15
        self.anisotropy = 0.18
        self.density_cutoff = 0.004
        self.smoke_color_low = np.array((0.13, 0.15, 0.17), dtype=np.float32)
        self.smoke_color_high = np.array((0.9, 0.92, 0.96), dtype=np.float32)

        self._gl = None
        self._initialized = False
        self._failed = False
        self._density_program = None
        self._density_tex = None
        self._uniforms: dict[str, int] = {}

        self._source_density = None
        self._source_grid_min = None
        self._source_cell_size = 0.0
        self._source_shape = (0, 0, 0)
        self._dirty = False

        self._volume_density = wp.zeros(self.volume_resolution, dtype=float)

    @property
    def available(self) -> bool:
        return not self._failed

    def set_dense_density(
        self,
        density: wp.array3d(dtype=float),
        grid_min: tuple[float, float, float] | wp.vec3,
        cell_size: float,
    ):
        """Register a dense 3D density grid for the next render call.

        Args:
            density: Dense cell-centered scalar density grid.
            grid_min: World-space position of the (0, 0, 0) cell's minimum corner.
            cell_size: Uniform cell size of the dense grid [m].
        """
        self._source_density = density
        if isinstance(grid_min, wp.vec3):
            self._source_grid_min = np.array(
                (float(grid_min[0]), float(grid_min[1]), float(grid_min[2])),
                dtype=np.float32,
            )
        else:
            self._source_grid_min = np.asarray(grid_min, dtype=np.float32)
        self._source_cell_size = float(cell_size)
        self._source_shape = tuple(int(v) for v in density.shape)
        self._dirty = True

    def render(self, viewer):
        del viewer
        if self._failed or self._source_density is None:
            return

        try:
            self._ensure_initialized()
            if self._dirty:
                self._rebuild_volume_texture()
                self._dirty = False
            self._composite_volume()
        except Exception as exc:
            self._failed = True
            print(f"[SmokeVolume] disabling realtime smoke rendering: {exc}")

    def _ensure_initialized(self):
        if self._initialized:
            return

        from pyglet import gl
        from pyglet.graphics.shader import Shader, ShaderProgram

        self._gl = gl
        self._density_program = ShaderProgram(
            Shader(self._fullscreen_vertex_shader, "vertex"),
            Shader(self._composite_fragment_shader, "fragment"),
        )

        self._density_tex = gl.GLuint()
        gl.glGenTextures(1, self._density_tex)
        self._allocate_density_texture()

        self._uniforms = self._collect_uniforms(
            self._density_program,
            "scene_color_tex",
            "density_tex",
            "camera_pos_world",
            "camera_front_world",
            "camera_right_world",
            "camera_up_world",
            "volume_min_world",
            "volume_max_world",
            "light_dir_world",
            "smoke_color_low",
            "smoke_color_high",
            "density_cutoff",
            "absorption",
            "opacity_boost",
            "ambient_strength",
            "light_strength",
            "anisotropy",
            "tan_half_fov",
            "aspect_ratio",
            "step_size_world",
            "shadow_step_world",
            "max_steps",
            "shadow_steps",
        )

        self._initialized = True

    def _allocate_density_texture(self):
        gl = self._gl
        nx, ny, nz = self.volume_resolution
        gl.glBindTexture(gl.GL_TEXTURE_3D, self._density_tex)
        gl.glTexImage3D(
            gl.GL_TEXTURE_3D,
            0,
            gl.GL_R32F,
            nx,
            ny,
            nz,
            0,
            gl.GL_RED,
            gl.GL_FLOAT,
            None,
        )
        gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_WRAP_R, gl.GL_CLAMP_TO_EDGE)
        gl.glBindTexture(gl.GL_TEXTURE_3D, 0)

    def _collect_uniforms(self, program, *names: str) -> dict[str, int]:
        gl = self._gl
        return {name: gl.glGetUniformLocation(program.id, ctypes.c_char_p(name.encode("utf-8"))) for name in names}

    def _rebuild_volume_texture(self):
        target_size = wp.vec3i(*self.volume_resolution)
        src_nx, src_ny, src_nz = self._source_shape
        wp.launch(
            build_smoke_volume_texture,
            dim=self.volume_resolution,
            inputs=[
                self._source_density,
                wp.vec3(*self._source_grid_min),
                self._source_cell_size,
                src_nx,
                src_ny,
                src_nz,
                wp.vec3(*self.world_min),
                wp.vec3(*self.world_max),
                target_size,
                self.density_scale,
            ],
            outputs=[self._volume_density],
            record_tape=False,
        )

        volume_host = np.asarray(self._volume_density.numpy(), dtype=np.float32)
        volume_host = np.ascontiguousarray(np.transpose(volume_host, (2, 1, 0)))

        gl = self._gl
        nx, ny, nz = self.volume_resolution
        gl.glBindTexture(gl.GL_TEXTURE_3D, self._density_tex)
        gl.glTexSubImage3D(
            gl.GL_TEXTURE_3D,
            0,
            0,
            0,
            0,
            nx,
            ny,
            nz,
            gl.GL_RED,
            gl.GL_FLOAT,
            volume_host.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        gl.glBindTexture(gl.GL_TEXTURE_3D, 0)

    def _composite_volume(self):
        gl = self._gl
        renderer = self.viewer.renderer
        camera = self.viewer.camera

        span = self.world_max - self.world_min
        cell_size = np.array(
            (
                span[0] / max(self.volume_resolution[0], 1),
                span[1] / max(self.volume_resolution[1], 1),
                span[2] / max(self.volume_resolution[2], 1),
            ),
            dtype=np.float32,
        )
        step_size_world = float(np.min(cell_size) * 1.35)
        shadow_step_world = float(step_size_world * 2.2)
        max_steps = int(np.ceil(np.linalg.norm(span) / max(step_size_world, 1.0e-4)))
        max_steps = max(24, min(max_steps, 192))
        shadow_steps = 12

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glViewport(0, 0, int(renderer._screen_width), int(renderer._screen_height))
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDepthMask(False)
        gl.glDisable(gl.GL_BLEND)
        gl.glUseProgram(self._density_program.id)

        uniforms = self._uniforms
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, renderer._frame_texture)
        gl.glUniform1i(uniforms["scene_color_tex"], 0)

        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_3D, self._density_tex)
        gl.glUniform1i(uniforms["density_tex"], 1)

        gl.glUniform3f(uniforms["camera_pos_world"], float(camera.pos[0]), float(camera.pos[1]), float(camera.pos[2]))
        camera_front = np.asarray(camera.get_front(), dtype=np.float32)
        camera_right = np.asarray(camera.get_right(), dtype=np.float32)
        camera_up = np.asarray(camera.get_up(), dtype=np.float32)
        gl.glUniform3f(
            uniforms["camera_front_world"], float(camera_front[0]), float(camera_front[1]), float(camera_front[2])
        )
        gl.glUniform3f(
            uniforms["camera_right_world"], float(camera_right[0]), float(camera_right[1]), float(camera_right[2])
        )
        gl.glUniform3f(uniforms["camera_up_world"], float(camera_up[0]), float(camera_up[1]), float(camera_up[2]))
        gl.glUniform3f(uniforms["volume_min_world"], *self.world_min)
        gl.glUniform3f(uniforms["volume_max_world"], *self.world_max)

        sun_world = np.asarray(renderer._sun_direction, dtype=np.float32)
        norm = float(np.linalg.norm(sun_world))
        if norm > 0.0:
            sun_world = sun_world / norm
        gl.glUniform3f(uniforms["light_dir_world"], *sun_world)
        gl.glUniform3f(uniforms["smoke_color_low"], *self.smoke_color_low)
        gl.glUniform3f(uniforms["smoke_color_high"], *self.smoke_color_high)
        gl.glUniform1f(uniforms["density_cutoff"], self.density_cutoff)
        gl.glUniform1f(uniforms["absorption"], self.absorption)
        gl.glUniform1f(uniforms["opacity_boost"], self.opacity_boost)
        gl.glUniform1f(uniforms["ambient_strength"], self.ambient_strength)
        gl.glUniform1f(uniforms["light_strength"], self.light_strength)
        gl.glUniform1f(uniforms["anisotropy"], self.anisotropy)
        gl.glUniform1f(uniforms["tan_half_fov"], float(np.tan(np.deg2rad(float(camera.fov)) * 0.5)))
        gl.glUniform1f(
            uniforms["aspect_ratio"], float(renderer._screen_width) / max(float(renderer._screen_height), 1.0)
        )
        gl.glUniform1f(uniforms["step_size_world"], step_size_world)
        gl.glUniform1f(uniforms["shadow_step_world"], shadow_step_world)
        gl.glUniform1i(uniforms["max_steps"], max_steps)
        gl.glUniform1i(uniforms["shadow_steps"], shadow_steps)

        gl.glBindVertexArray(renderer._frame_vao)
        gl.glDrawElements(gl.GL_TRIANGLES, len(renderer._frame_indices), gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)

        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_3D, 0)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glUseProgram(0)
        gl.glDepthMask(True)
