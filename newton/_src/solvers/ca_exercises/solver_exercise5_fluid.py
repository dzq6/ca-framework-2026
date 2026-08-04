from __future__ import annotations

import warp as wp

from ...core.types import override
from ...sim import Contacts, Control, Model, State
from ..solver import SolverBase

# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@wp.kernel
def apply_density_source_kernel(
    density: wp.array3d(dtype=float),
    solid: wp.array3d(dtype=wp.int32),
    source_min: wp.vec3i,
    source_max: wp.vec3i,
):
    i, j, k = wp.tid()

    if i < source_min[0] or i >= source_max[0]:
        return
    if j < source_min[1] or j >= source_max[1]:
        return
    if k < source_min[2] or k >= source_max[2]:
        return
    if solid[i, j, k] != 0:
        return
    density[i, j, k] = 1.0


# ----- Sampling helpers (semi-Lagrangian advection) -----


@wp.func
def sample_centered_clamped(
    field: wp.array3d(dtype=float),
    fx: float,
    fy: float,
    fz: float,
    nx: int,
    ny: int,
    nz: int,
) -> float:
    # Clamp to interior so indexing stays in-bounds.
    cx = wp.clamp(fx, 0.0, float(nx - 1) - 1.0e-4)
    cy = wp.clamp(fy, 0.0, float(ny - 1) - 1.0e-4)
    cz = wp.clamp(fz, 0.0, float(nz - 1) - 1.0e-4)
    i0 = int(cx)
    j0 = int(cy)
    k0 = int(cz)
    i1 = wp.min(i0 + 1, nx - 1)
    j1 = wp.min(j0 + 1, ny - 1)
    k1 = wp.min(k0 + 1, nz - 1)
    tx = cx - float(i0)
    ty = cy - float(j0)
    tz = cz - float(k0)
    c000 = field[i0, j0, k0]
    c100 = field[i1, j0, k0]
    c010 = field[i0, j1, k0]
    c110 = field[i1, j1, k0]
    c001 = field[i0, j0, k1]
    c101 = field[i1, j0, k1]
    c011 = field[i0, j1, k1]
    c111 = field[i1, j1, k1]
    c00 = c000 * (1.0 - tx) + c100 * tx
    c10 = c010 * (1.0 - tx) + c110 * tx
    c01 = c001 * (1.0 - tx) + c101 * tx
    c11 = c011 * (1.0 - tx) + c111 * tx
    c0 = c00 * (1.0 - ty) + c10 * ty
    c1 = c01 * (1.0 - ty) + c11 * ty
    return c0 * (1.0 - tz) + c1 * tz


@wp.func
def velocity_at_cell_center(
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    i: int,
    j: int,
    k: int,
) -> wp.vec3:
    ux = 0.5 * (u[i, j, k] + u[i + 1, j, k])
    vy = 0.5 * (v[i, j, k] + v[i, j + 1, k])
    wz = 0.5 * (w[i, j, k] + w[i, j, k + 1])
    return wp.vec3(ux, vy, wz)


@wp.kernel
def fill_solid_box_kernel(
    solid: wp.array3d(dtype=wp.int32),
    box_min: wp.vec3i,
    box_max: wp.vec3i,
):
    i, j, k = wp.tid()
    if i >= box_min[0] and i < box_max[0] and j >= box_min[1] and j < box_max[1] and k >= box_min[2] and k < box_max[2]:
        solid[i, j, k] = 1


@wp.kernel
def add_buoyancy_kernel(
    w: wp.array3d(dtype=float),
    density: wp.array3d(dtype=float),
    solid: wp.array3d(dtype=wp.int32),
    buoyancy_scale: float,
    dt: float,
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()

    # TODO
    # w lives on z-faces, shape (nx, ny, nz+1). w[i, j, k] sits between
    # cell (i, j, k-1) and (i, j, k). Newton's world is Z-up, so buoyancy
    # pushes the z-face velocity.


@wp.kernel
def add_wind_kernel(
    v: wp.array3d(dtype=float),
    solid: wp.array3d(dtype=wp.int32),
    wind_force: float,
    dt: float,
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()

    # TODO: v lives on y-faces, shape (nx, ny+1, nz).


@wp.kernel
def enforce_solid_velocity_u_kernel(
    u: wp.array3d(dtype=float),
    solid: wp.array3d(dtype=wp.int32),
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()

    # TODO: Enforce solid boundary conditions on the x-face velocity u.


@wp.kernel
def enforce_solid_velocity_v_kernel(
    v: wp.array3d(dtype=float),
    solid: wp.array3d(dtype=wp.int32),
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()

    # TODO: Enforce solid boundary conditions on the y-face velocity v.


@wp.kernel
def enforce_solid_velocity_w_kernel(
    w: wp.array3d(dtype=float),
    solid: wp.array3d(dtype=wp.int32),
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()

    # TODO: Enforce solid boundary conditions on the z-face velocity w.


@wp.kernel
def compute_divergence_kernel(
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    solid: wp.array3d(dtype=wp.int32),
    inv_dx: float,
    divergence: wp.array3d(dtype=float),
):
    i, j, k = wp.tid()

    # TODO: Compute the divergence of the velocity field at cell (i, j, k)


@wp.kernel
def gauss_seidel_rb_step_kernel(
    pressure: wp.array3d(dtype=float),
    divergence: wp.array3d(dtype=float),
    solid: wp.array3d(dtype=wp.int32),
    dx: float,
    dt: float,
    nx: int,
    ny: int,
    nz: int,
    parity: int,
):
    i, j, k = wp.tid()

    # TODO: Perform one red-black Gauss-Seidel iteration to solve the pressure Poisson equation.
    # Red-black Gauss-Seidel: updates only cells whose (i + j + k) parity matches
    # ``parity`` (0 = red, 1 = black), so all cells touched in this launch read
    # from neighbours of the opposite colour and can run in parallel.


@wp.kernel
def project_velocity_u_kernel(
    u: wp.array3d(dtype=float),
    pressure: wp.array3d(dtype=float),
    solid: wp.array3d(dtype=wp.int32),
    dt: float,
    inv_dx: float,
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()

    # TODO: Project the x-face velocity u by subtracting the pressure gradient.


@wp.kernel
def project_velocity_v_kernel(
    v: wp.array3d(dtype=float),
    pressure: wp.array3d(dtype=float),
    solid: wp.array3d(dtype=wp.int32),
    dt: float,
    inv_dx: float,
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()

    # TODO: Project the y-face velocity v by subtracting the pressure gradient.


@wp.kernel
def project_velocity_w_kernel(
    w: wp.array3d(dtype=float),
    pressure: wp.array3d(dtype=float),
    solid: wp.array3d(dtype=wp.int32),
    dt: float,
    inv_dx: float,
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()

    # TODO: Project the z-face velocity w by subtracting the pressure gradient.


@wp.kernel
def advect_density_kernel(
    density: wp.array3d(dtype=float),
    density_out: wp.array3d(dtype=float),
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    solid: wp.array3d(dtype=wp.int32),
    dt: float,
    inv_dx: float,
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()

    # TODO: Advect density via semi-Lagrangian backtrace.


@wp.kernel
def advect_velocity_u_kernel(
    u_in: wp.array3d(dtype=float),
    v_in: wp.array3d(dtype=float),
    w_in: wp.array3d(dtype=float),
    u_out: wp.array3d(dtype=float),
    solid: wp.array3d(dtype=wp.int32),
    dt: float,
    inv_dx: float,
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()

    # TODO: Advect the x-face velocity u via semi-Lagrangian backtrace.


@wp.kernel
def advect_velocity_v_kernel(
    u_in: wp.array3d(dtype=float),
    v_in: wp.array3d(dtype=float),
    w_in: wp.array3d(dtype=float),
    v_out: wp.array3d(dtype=float),
    solid: wp.array3d(dtype=wp.int32),
    dt: float,
    inv_dx: float,
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()

    # TODO: Advect the y-face velocity v via semi-Lagrangian backtrace.


@wp.kernel
def advect_velocity_w_kernel(
    u_in: wp.array3d(dtype=float),
    v_in: wp.array3d(dtype=float),
    w_in: wp.array3d(dtype=float),
    w_out: wp.array3d(dtype=float),
    solid: wp.array3d(dtype=wp.int32),
    dt: float,
    inv_dx: float,
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()

    # TODO: Advect the z-face velocity w via semi-Lagrangian backtrace.


# ---------------------------------------------------------------------------
# Solver class
# ---------------------------------------------------------------------------


class SolverExercise5Fluid(SolverBase):
    """3D Eulerian smoke solver on a MAC grid (Newton + Warp port of FluidSim.cpp).

    The dynamic fields (density, velocity, pressure) live on the solver itself rather
    than on the Newton :class:`State`, since Eulerian fluid quantities are not particle
    or rigid-body data. The :meth:`step` signature still matches :class:`SolverBase`.
    """

    def __init__(
        self,
        model: Model,
        res: tuple[int, int, int] = (48, 48, 72),
        domain_size: tuple[float, float, float] = (1.0, 1.0, 1.5),
        pressure_iters: int = 25,
        buoyancy: float = 0.1,
        wind_on: bool = False,
        wind_strength: float = 0.5,
        source_box: tuple[tuple[float, float, float], tuple[float, float, float]] = (
            (0.45, 0.45, 0.08),
            (0.55, 0.55, 0.15),
        ),
        obstacle_box: tuple[tuple[float, float, float], tuple[float, float, float]] | None = (
            (0.40, 0.40, 0.40),
            (0.60, 0.60, 0.50),
        ),
        obstacle_on: bool = True,
    ):
        super().__init__(model)

        self.nx, self.ny, self.nz = (int(v) for v in res)
        self.size_x, self.size_y, self.size_z = (float(v) for v in domain_size)
        self.cell_size = self.size_x / float(self.nx)
        self.pressure_iters = int(pressure_iters)

        # 64.0 / nx mirrors the C++ scaling so the plume rises at a comparable rate
        # regardless of grid resolution.
        self.buoyancy_scale = float(buoyancy) * (64.0 / float(self.nx))
        self.wind_on = bool(wind_on)
        self.wind_strength = float(wind_strength)

        self.grid_min = (0.0, 0.0, 0.0)
        self.grid_max = (
            self.cell_size * self.nx,
            self.cell_size * self.ny,
            self.cell_size * self.nz,
        )

        self._source_box = source_box
        self._obstacle_box = obstacle_box
        self.obstacle_on = bool(obstacle_on)

        device = model.device

        self.density = wp.zeros((self.nx, self.ny, self.nz), dtype=float, device=device)
        self.density_tmp = wp.zeros((self.nx, self.ny, self.nz), dtype=float, device=device)

        self.u = wp.zeros((self.nx + 1, self.ny, self.nz), dtype=float, device=device)
        self.v = wp.zeros((self.nx, self.ny + 1, self.nz), dtype=float, device=device)
        self.w = wp.zeros((self.nx, self.ny, self.nz + 1), dtype=float, device=device)
        self.u_tmp = wp.zeros_like(self.u)
        self.v_tmp = wp.zeros_like(self.v)
        self.w_tmp = wp.zeros_like(self.w)

        self.pressure = wp.zeros((self.nx, self.ny, self.nz), dtype=float, device=device)
        self.divergence = wp.zeros((self.nx, self.ny, self.nz), dtype=float, device=device)

        self.solid = wp.zeros((self.nx, self.ny, self.nz), dtype=wp.int32, device=device)

        self._build_solid_mask()

    # ----- Public helpers used by the example / renderer -----

    def reset(self) -> None:
        self.density.zero_()
        self.density_tmp.zero_()
        self.u.zero_()
        self.v.zero_()
        self.w.zero_()
        self.u_tmp.zero_()
        self.v_tmp.zero_()
        self.w_tmp.zero_()
        self.pressure.zero_()
        self.divergence.zero_()
        self.solid.zero_()
        self._build_solid_mask()

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> State | None:
        del state_in, state_out, control, contacts

        self._apply_density_source()
        self._add_buoyancy(dt)
        if self.wind_on:
            self._add_wind(dt)
        self._enforce_solid_velocity()
        self._compute_divergence()
        self._solve_pressure(dt)
        self._project_velocity(dt)
        self._advect(dt)

    @override
    def update_contacts(self, contacts: Contacts) -> None:
        del contacts

    # ----- Internals -----

    def _world_to_cell(self, point: tuple[float, float, float]) -> tuple[int, int, int]:
        return (
            int(round(point[0] / self.cell_size)),
            int(round(point[1] / self.cell_size)),
            int(round(point[2] / self.cell_size)),
        )

    def _build_solid_mask(self) -> None:
        if not self.obstacle_on or self._obstacle_box is None:
            return
        lo = self._clamp_cell(self._world_to_cell(self._obstacle_box[0]))
        hi = self._clamp_cell(self._world_to_cell(self._obstacle_box[1]))
        if hi[0] <= lo[0] or hi[1] <= lo[1] or hi[2] <= lo[2]:
            return
        wp.launch(
            fill_solid_box_kernel,
            dim=(self.nx, self.ny, self.nz),
            inputs=[self.solid, wp.vec3i(*lo), wp.vec3i(*hi)],
            device=self.model.device,
        )

    def set_obstacle_active(self, active: bool) -> None:
        """Enable or disable the solid obstacle at runtime.

        Rebuilds the solid mask immediately; pressure and velocity fields are kept
        so the flow transitions smoothly around the change.
        """
        self.obstacle_on = bool(active)
        self.solid.zero_()
        self._build_solid_mask()

    def _clamp_cell(self, cell: tuple[int, int, int]) -> tuple[int, int, int]:
        return (
            max(0, min(cell[0], self.nx)),
            max(0, min(cell[1], self.ny)),
            max(0, min(cell[2], self.nz)),
        )

    def _apply_density_source(self) -> None:
        lo = self._clamp_cell(self._world_to_cell(self._source_box[0]))
        hi = self._clamp_cell(self._world_to_cell(self._source_box[1]))
        wp.launch(
            apply_density_source_kernel,
            dim=(self.nx, self.ny, self.nz),
            inputs=[self.density, self.solid, wp.vec3i(*lo), wp.vec3i(*hi)],
            device=self.model.device,
        )

    def _add_buoyancy(self, dt: float) -> None:
        wp.launch(
            add_buoyancy_kernel,
            dim=(self.nx, self.ny, self.nz + 1),
            inputs=[
                self.w,
                self.density,
                self.solid,
                self.buoyancy_scale,
                dt,
                self.nx,
                self.ny,
                self.nz,
            ],
            device=self.model.device,
        )

    def _add_wind(self, dt: float) -> None:
        wp.launch(
            add_wind_kernel,
            dim=(self.nx, self.ny + 1, self.nz),
            inputs=[
                self.v,
                self.solid,
                self.wind_strength * (64.0 / float(self.nx)),
                dt,
                self.nx,
                self.ny,
                self.nz,
            ],
            device=self.model.device,
        )

    def _enforce_solid_velocity(self) -> None:
        device = self.model.device
        wp.launch(
            enforce_solid_velocity_u_kernel,
            dim=(self.nx + 1, self.ny, self.nz),
            inputs=[self.u, self.solid, self.nx, self.ny, self.nz],
            device=device,
        )
        wp.launch(
            enforce_solid_velocity_v_kernel,
            dim=(self.nx, self.ny + 1, self.nz),
            inputs=[self.v, self.solid, self.nx, self.ny, self.nz],
            device=device,
        )
        wp.launch(
            enforce_solid_velocity_w_kernel,
            dim=(self.nx, self.ny, self.nz + 1),
            inputs=[self.w, self.solid, self.nx, self.ny, self.nz],
            device=device,
        )

    def _compute_divergence(self) -> None:
        wp.launch(
            compute_divergence_kernel,
            dim=(self.nx, self.ny, self.nz),
            inputs=[
                self.u,
                self.v,
                self.w,
                self.solid,
                1.0 / self.cell_size,
                self.divergence,
            ],
            device=self.model.device,
        )

    def _solve_pressure(self, dt: float) -> None:
        self.pressure.zero_()
        device = self.model.device
        for _ in range(self.pressure_iters):
            for parity in (0, 1):
                wp.launch(
                    gauss_seidel_rb_step_kernel,
                    dim=(self.nx, self.ny, self.nz),
                    inputs=[
                        self.pressure,
                        self.divergence,
                        self.solid,
                        self.cell_size,
                        dt,
                        self.nx,
                        self.ny,
                        self.nz,
                        parity,
                    ],
                    device=device,
                )

    def _project_velocity(self, dt: float) -> None:
        device = self.model.device
        inv_dx = 1.0 / self.cell_size
        wp.launch(
            project_velocity_u_kernel,
            dim=(self.nx + 1, self.ny, self.nz),
            inputs=[self.u, self.pressure, self.solid, dt, inv_dx, self.nx, self.ny, self.nz],
            device=device,
        )
        wp.launch(
            project_velocity_v_kernel,
            dim=(self.nx, self.ny + 1, self.nz),
            inputs=[self.v, self.pressure, self.solid, dt, inv_dx, self.nx, self.ny, self.nz],
            device=device,
        )
        wp.launch(
            project_velocity_w_kernel,
            dim=(self.nx, self.ny, self.nz + 1),
            inputs=[self.w, self.pressure, self.solid, dt, inv_dx, self.nx, self.ny, self.nz],
            device=device,
        )

    def _advect(self, dt: float) -> None:
        device = self.model.device
        inv_dx = 1.0 / self.cell_size

        wp.launch(
            advect_density_kernel,
            dim=(self.nx, self.ny, self.nz),
            inputs=[
                self.density,
                self.density_tmp,
                self.u,
                self.v,
                self.w,
                self.solid,
                dt,
                inv_dx,
                self.nx,
                self.ny,
                self.nz,
            ],
            device=device,
        )
        wp.launch(
            advect_velocity_u_kernel,
            dim=(self.nx + 1, self.ny, self.nz),
            inputs=[
                self.u,
                self.v,
                self.w,
                self.u_tmp,
                self.solid,
                dt,
                inv_dx,
                self.nx,
                self.ny,
                self.nz,
            ],
            device=device,
        )
        wp.launch(
            advect_velocity_v_kernel,
            dim=(self.nx, self.ny + 1, self.nz),
            inputs=[
                self.u,
                self.v,
                self.w,
                self.v_tmp,
                self.solid,
                dt,
                inv_dx,
                self.nx,
                self.ny,
                self.nz,
            ],
            device=device,
        )
        wp.launch(
            advect_velocity_w_kernel,
            dim=(self.nx, self.ny, self.nz + 1),
            inputs=[
                self.u,
                self.v,
                self.w,
                self.w_tmp,
                self.solid,
                dt,
                inv_dx,
                self.nx,
                self.ny,
                self.nz,
            ],
            device=device,
        )

        wp.copy(self.density, self.density_tmp)
        wp.copy(self.u, self.u_tmp)
        wp.copy(self.v, self.v_tmp)
        wp.copy(self.w, self.w_tmp)
