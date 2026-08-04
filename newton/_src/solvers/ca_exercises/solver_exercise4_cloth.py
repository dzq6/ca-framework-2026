# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import warp as wp

from ...core.types import override
from ...sim import Contacts, Control, Model, State
from ..solver import SolverBase


@wp.func
def spring_force(
    p: wp.vec3,
    v: wp.vec3,
    p_n: wp.vec3,
    v_n: wp.vec3,
    rest_length: float,
    stiffness: float,
    damping: float,
) -> wp.vec3:
    d = p_n - p
    dist = wp.length(d)
    if dist < 1.0e-8:
        return wp.vec3(0.0, 0.0, 0.0)
    u = d / dist
    return stiffness * (dist - rest_length) * u + damping * wp.dot(v_n - v, u) * u


@wp.kernel
def compute_particle_force_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_f: wp.array(dtype=wp.vec3),
    particle_fixed: wp.array(dtype=wp.int32),
    particle_mass: float,
    k_struct: float,
    k_bend: float,
    k_shear: float,
    damping: float,
    cell_size: float,
    grid_n: int,
    grid_m: int,
    gravity: wp.vec3,
):
    i, j = wp.tid()
    idx = i * grid_m + j

    if particle_fixed[idx] == 1:
        particle_f[idx] = wp.vec3(0.0, 0.0, 0.0)
        return

    force = gravity * particle_mass
    p = particle_q[idx]
    v = particle_qd[idx]

    shear_rest = cell_size * wp.sqrt(2.0)
    bend_rest = 2.0 * cell_size

    # Structural: up / down / left / right
    if i > 0:
        nidx = (i - 1) * grid_m + j
        force += spring_force(p, v, particle_q[nidx], particle_qd[nidx], cell_size, k_struct, damping)
    if i < grid_n - 1:
        nidx = (i + 1) * grid_m + j
        force += spring_force(p, v, particle_q[nidx], particle_qd[nidx], cell_size, k_struct, damping)
    if j > 0:
        nidx = i * grid_m + (j - 1)
        force += spring_force(p, v, particle_q[nidx], particle_qd[nidx], cell_size, k_struct, damping)
    if j < grid_m - 1:
        nidx = i * grid_m + (j + 1)
        force += spring_force(p, v, particle_q[nidx], particle_qd[nidx], cell_size, k_struct, damping)

    # Shear: four diagonals
    if i > 0 and j > 0:
        nidx = (i - 1) * grid_m + (j - 1)
        force += spring_force(p, v, particle_q[nidx], particle_qd[nidx], shear_rest, k_shear, damping)
    if i > 0 and j < grid_m - 1:
        nidx = (i - 1) * grid_m + (j + 1)
        force += spring_force(p, v, particle_q[nidx], particle_qd[nidx], shear_rest, k_shear, damping)
    if i < grid_n - 1 and j > 0:
        nidx = (i + 1) * grid_m + (j - 1)
        force += spring_force(p, v, particle_q[nidx], particle_qd[nidx], shear_rest, k_shear, damping)
    if i < grid_n - 1 and j < grid_m - 1:
        nidx = (i + 1) * grid_m + (j + 1)
        force += spring_force(p, v, particle_q[nidx], particle_qd[nidx], shear_rest, k_shear, damping)

    # Bending: skip-one neighbors
    if i > 1:
        nidx = (i - 2) * grid_m + j
        force += spring_force(p, v, particle_q[nidx], particle_qd[nidx], bend_rest, k_bend, damping)
    if i < grid_n - 2:
        nidx = (i + 2) * grid_m + j
        force += spring_force(p, v, particle_q[nidx], particle_qd[nidx], bend_rest, k_bend, damping)
    if j > 1:
        nidx = i * grid_m + (j - 2)
        force += spring_force(p, v, particle_q[nidx], particle_qd[nidx], bend_rest, k_bend, damping)
    if j < grid_m - 2:
        nidx = i * grid_m + (j + 2)
        force += spring_force(p, v, particle_q[nidx], particle_qd[nidx], bend_rest, k_bend, damping)

    particle_f[idx] = force


@wp.kernel
def explicit_euler_integrate_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_q_out: wp.array(dtype=wp.vec3),
    particle_qd_out: wp.array(dtype=wp.vec3),
    particle_f: wp.array(dtype=wp.vec3),
    particle_fixed: wp.array(dtype=wp.int32),
    particle_mass: float,
    dt: float,
    grid_n: int,
    grid_m: int,
):
    idx = wp.tid()

    # Fixed particles: keep pose/velocity unchanged
    if particle_fixed[idx] == 1:
        particle_q_out[idx] = particle_q[idx]
        particle_qd_out[idx] = particle_qd[idx]
        return

    # Explicit Euler: p' = p + v*dt, v' = v + a*dt  (a = f/m, uses old v for position)
    a = particle_f[idx] / particle_mass
    v = particle_qd[idx]
    p = particle_q[idx]

    particle_q_out[idx] = p + v * dt
    particle_qd_out[idx] = v + a * dt


@wp.kernel
def make_mid_state_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_f: wp.array(dtype=wp.vec3),
    particle_fixed: wp.array(dtype=wp.int32),
    particle_q_mid: wp.array(dtype=wp.vec3),
    particle_qd_mid: wp.array(dtype=wp.vec3),
    particle_mass: float,
    dt: float,
):
    idx = wp.tid()
    if particle_fixed[idx] == 1:
        particle_q_mid[idx] = particle_q[idx]
        particle_qd_mid[idx] = particle_qd[idx]
        return

    a = particle_f[idx] / particle_mass
    v = particle_qd[idx]
    particle_q_mid[idx] = particle_q[idx] + v * (0.5 * dt)
    particle_qd_mid[idx] = v + a * (0.5 * dt)


@wp.kernel
def midpoint_finalize_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_qd_mid: wp.array(dtype=wp.vec3),
    particle_f_mid: wp.array(dtype=wp.vec3),
    particle_fixed: wp.array(dtype=wp.int32),
    particle_q_out: wp.array(dtype=wp.vec3),
    particle_qd_out: wp.array(dtype=wp.vec3),
    particle_mass: float,
    dt: float,
):
    idx = wp.tid()
    if particle_fixed[idx] == 1:
        particle_q_out[idx] = particle_q[idx]
        particle_qd_out[idx] = particle_qd[idx]
        return

    a_mid = particle_f_mid[idx] / particle_mass
    v_mid = particle_qd_mid[idx]
    particle_q_out[idx] = particle_q[idx] + v_mid * dt
    particle_qd_out[idx] = particle_qd[idx] + a_mid * dt


@wp.kernel
def store_deriv_kernel(
    particle_qd: wp.array(dtype=wp.vec3),
    particle_f: wp.array(dtype=wp.vec3),
    particle_fixed: wp.array(dtype=wp.int32),
    k_p: wp.array(dtype=wp.vec3),
    k_v: wp.array(dtype=wp.vec3),
    particle_mass: float,
):
    """Store f(y)=(v, a) stage derivatives for RK4."""
    idx = wp.tid()
    if particle_fixed[idx] == 1:
        k_p[idx] = wp.vec3(0.0, 0.0, 0.0)
        k_v[idx] = wp.vec3(0.0, 0.0, 0.0)
        return
    k_p[idx] = particle_qd[idx]
    k_v[idx] = particle_f[idx] / particle_mass


@wp.kernel
def rk_stage_state_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_fixed: wp.array(dtype=wp.int32),
    k_p: wp.array(dtype=wp.vec3),
    k_v: wp.array(dtype=wp.vec3),
    particle_q_out: wp.array(dtype=wp.vec3),
    particle_qd_out: wp.array(dtype=wp.vec3),
    scale: float,
):
    idx = wp.tid()
    if particle_fixed[idx] == 1:
        particle_q_out[idx] = particle_q[idx]
        particle_qd_out[idx] = particle_qd[idx]
        return
    particle_q_out[idx] = particle_q[idx] + k_p[idx] * scale
    particle_qd_out[idx] = particle_qd[idx] + k_v[idx] * scale


@wp.kernel
def rk4_finalize_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_fixed: wp.array(dtype=wp.int32),
    k1_p: wp.array(dtype=wp.vec3),
    k1_v: wp.array(dtype=wp.vec3),
    k2_p: wp.array(dtype=wp.vec3),
    k2_v: wp.array(dtype=wp.vec3),
    k3_p: wp.array(dtype=wp.vec3),
    k3_v: wp.array(dtype=wp.vec3),
    k4_p: wp.array(dtype=wp.vec3),
    k4_v: wp.array(dtype=wp.vec3),
    particle_q_out: wp.array(dtype=wp.vec3),
    particle_qd_out: wp.array(dtype=wp.vec3),
    dt: float,
):
    idx = wp.tid()
    if particle_fixed[idx] == 1:
        particle_q_out[idx] = particle_q[idx]
        particle_qd_out[idx] = particle_qd[idx]
        return

    particle_q_out[idx] = particle_q[idx] + (dt / 6.0) * (k1_p[idx] + 2.0 * k2_p[idx] + 2.0 * k3_p[idx] + k4_p[idx])
    particle_qd_out[idx] = particle_qd[idx] + (dt / 6.0) * (k1_v[idx] + 2.0 * k2_v[idx] + 2.0 * k3_v[idx] + k4_v[idx])


@wp.kernel
def semi_implicit_euler_integrate_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_q_out: wp.array(dtype=wp.vec3),
    particle_qd_out: wp.array(dtype=wp.vec3),
    particle_f: wp.array(dtype=wp.vec3),
    particle_fixed: wp.array(dtype=wp.int32),
    particle_mass: float,
    dt: float,
    grid_n: int,
    grid_m: int,
):
    idx = wp.tid()

    if particle_fixed[idx] == 1:
        particle_q_out[idx] = particle_q[idx]
        particle_qd_out[idx] = particle_qd[idx]
        return

    # Semi-implicit Euler: v' = v + a*dt, p' = p + v'*dt
    a = particle_f[idx] / particle_mass
    v_new = particle_qd[idx] + a * dt
    particle_qd_out[idx] = v_new
    particle_q_out[idx] = particle_q[idx] + v_new * dt


@wp.kernel
def apply_velocity_update_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd_out: wp.array(dtype=wp.vec3),
    particle_q_out: wp.array(dtype=wp.vec3),
    particle_fixed: wp.array(dtype=wp.int32),
    vel_flat: wp.array(dtype=float),
    dt: float,
):
    """Write CG-solved velocities and integrate positions (implicit Euler)."""
    idx = wp.tid()

    if particle_fixed[idx] == 1:
        particle_q_out[idx] = particle_q[idx]
        particle_qd_out[idx] = wp.vec3(0.0, 0.0, 0.0)
        return

    base = 3 * idx
    v_new = wp.vec3(vel_flat[base], vel_flat[base + 1], vel_flat[base + 2])
    particle_qd_out[idx] = v_new
    particle_q_out[idx] = particle_q[idx] + v_new * dt


def cg_solve(
    A: wp.array,
    b: wp.array,
    x: wp.array,
    max_iters: int = 100,
    tol: float = 1e-6,
) -> int:
    """Solve the linear system Ax = b using the Conjugate Gradient method."""
    # Copy to CPU for computation
    A_cpu = A.numpy()
    b_cpu = b.numpy()
    x_cpu = x.numpy().copy()

    r = b_cpu - A_cpu @ x_cpu
    p = r.copy()
    rs_old = float(r @ r)

    for i in range(max_iters):
        Ap = A_cpu @ p
        pAp = float(p @ Ap)
        if abs(pAp) < 1e-30:
            break
        alpha = rs_old / pAp
        x_cpu = x_cpu + alpha * p
        r = r - alpha * Ap
        rs_new = float(r @ r)
        if rs_new < tol * tol:
            wp.copy(x, wp.array(x_cpu, dtype=wp.float32, device=x.device))
            return i + 1
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    wp.copy(x, wp.array(x_cpu, dtype=wp.float32, device=x.device))
    return max_iters


def _spring_jacobian_blocks(p_i: np.ndarray, p_j: np.ndarray, rest: float, k: float) -> np.ndarray:
    """Return 3x3 df_i/dd where d = p_j - p_i, f_i = k(|d|-rest) d/|d|."""
    d = p_j - p_i
    length = float(np.linalg.norm(d))
    if length < 1.0e-8:
        return np.zeros((3, 3), dtype=np.float64)
    u = d / length
    # df/dd = k * [(1 - rest/L) I + (rest/L) u u^T]
    return k * ((1.0 - rest / length) * np.eye(3) + (rest / length) * np.outer(u, u))


def build_stiffness_matrix_sparse(
    positions: np.ndarray,
    grid_n: int,
    grid_m: int,
    cell_size: float,
    k_struct: float,
    k_shear: float,
    k_bend: float,
):
    """Assemble sparse stiffness K = df/dx at the given positions."""
    from scipy import sparse

    n = grid_n * grid_m
    dof = 3 * n
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    def add_block(r0: int, c0: int, block: np.ndarray) -> None:
        for r in range(3):
            for c in range(3):
                v = float(block[r, c])
                if v != 0.0:
                    rows.append(r0 + r)
                    cols.append(c0 + c)
                    vals.append(v)

    def add_spring(i: int, j: int, rest: float, stiffness: float) -> None:
        dfd_d = _spring_jacobian_blocks(positions[i], positions[j], rest, stiffness)
        bi, bj = 3 * i, 3 * j
        add_block(bi, bi, -dfd_d)
        add_block(bi, bj, dfd_d)
        add_block(bj, bi, dfd_d)
        add_block(bj, bj, -dfd_d)

    shear_rest = cell_size * np.sqrt(2.0)
    bend_rest = 2.0 * cell_size

    for i in range(grid_n):
        for j in range(grid_m):
            idx = i * grid_m + j
            if i + 1 < grid_n:
                add_spring(idx, (i + 1) * grid_m + j, cell_size, k_struct)
            if j + 1 < grid_m:
                add_spring(idx, i * grid_m + (j + 1), cell_size, k_struct)
            if i + 1 < grid_n and j + 1 < grid_m:
                add_spring(idx, (i + 1) * grid_m + (j + 1), shear_rest, k_shear)
            if i + 1 < grid_n and j - 1 >= 0:
                add_spring(idx, (i + 1) * grid_m + (j - 1), shear_rest, k_shear)
            if i + 2 < grid_n:
                add_spring(idx, (i + 2) * grid_m + j, bend_rest, k_bend)
            if j + 2 < grid_m:
                add_spring(idx, i * grid_m + (j + 2), bend_rest, k_bend)

    return sparse.coo_matrix((vals, (rows, cols)), shape=(dof, dof), dtype=np.float64).tocsr()


def assemble_implicit_system_sparse(
    positions: np.ndarray,
    velocities: np.ndarray,
    forces: np.ndarray,
    grid_n: int,
    grid_m: int,
    cell_size: float,
    mass: float,
    k_struct: float,
    k_shear: float,
    k_bend: float,
    dt: float,
    particle_fixed: np.ndarray,
):
    """Build sparse (M - dt^2 K) and RHS for implicit Euler."""
    from scipy import sparse

    n = grid_n * grid_m
    K = build_stiffness_matrix_sparse(positions, grid_n, grid_m, cell_size, k_struct, k_shear, k_bend)
    dof = K.shape[0]
    A = mass * sparse.eye(dof, format="csr", dtype=np.float64) - (dt * dt) * K
    b = mass * velocities.reshape(-1).astype(np.float64) + dt * forces.reshape(-1).astype(np.float64)

    fixed_dofs = []
    for pin in range(n):
        if particle_fixed[pin] != 0:
            fixed_dofs.extend((3 * pin, 3 * pin + 1, 3 * pin + 2))
    if fixed_dofs:
        fixed_dofs = np.asarray(fixed_dofs, dtype=np.int64)
        A = A.tolil()
        for row in fixed_dofs:
            A[row, :] = 0.0
            A[:, row] = 0.0
            A[row, row] = 1.0
            b[row] = 0.0
        A = A.tocsr()

    return A, b


def assemble_implicit_system(
    positions: np.ndarray,
    velocities: np.ndarray,
    forces: np.ndarray,
    grid_n: int,
    grid_m: int,
    cell_size: float,
    mass: float,
    k_struct: float,
    k_shear: float,
    k_bend: float,
    dt: float,
    particle_fixed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Dense fallback used by the provided cg_solve helper."""
    A_sp, b = assemble_implicit_system_sparse(
        positions,
        velocities,
        forces,
        grid_n,
        grid_m,
        cell_size,
        mass,
        k_struct,
        k_shear,
        k_bend,
        dt,
        particle_fixed,
    )
    return np.asarray(A_sp.toarray(), dtype=np.float32), b.astype(np.float32)


def build_particle_fixed_mask(grid_n: int, grid_m: int, pin_width: int, pin_rows: int) -> np.ndarray:
    """Fix a strip at each top corner (or the whole top edge if pin_width is large)."""
    fixed = np.zeros(grid_n * grid_m, dtype=np.int32)
    pin_width = max(1, min(pin_width, grid_m))
    pin_rows = max(1, min(pin_rows, grid_n))
    for i in range(pin_rows):
        for j in range(grid_m):
            if j < pin_width or j >= grid_m - pin_width:
                fixed[i * grid_m + j] = 1
    return fixed


class SolverExercise4Cloth(SolverBase):
    """Cloth solver with explicit / implicit / semi-implicit / midpoint / RK4 integration."""

    def __init__(
        self,
        model: Model,
        method: int = 0,
        params: dict | None = None,
        gravity: wp.vec3 | None = None,
    ):
        super().__init__(model)

        self.method = method
        self.params = params if params is not None else {}
        self.gravity = gravity if gravity is not None else wp.vec3(0.0, 0.0, -9.81)

        self.grid_n = self.params.get("grid_n", 16)
        self.grid_m = self.params.get("grid_m", 16)
        self.particle_count = self.grid_n * self.grid_m
        self.particle_f = wp.zeros(self.particle_count, dtype=wp.vec3, device=model.device)

        # pin_width particles from each top corner, over pin_rows rows (default: whole top edge)
        pin_width = int(self.params.get("pin_width", self.grid_m))
        pin_rows = int(self.params.get("pin_rows", 1))
        fixed_np = build_particle_fixed_mask(self.grid_n, self.grid_m, pin_width, pin_rows)
        self.particle_fixed = wp.array(fixed_np, dtype=wp.int32, device=model.device)
        self._particle_fixed_np = fixed_np
        self._fixed_dofs = np.flatnonzero(np.repeat(fixed_np, 3)).astype(np.int64)

        # Precompute rest-pose stiffness once. Per-step rebuild of K was ~30ms and
        # made Implicit look frozen at 20 substeps/frame.
        cell_size = float(self.params.get("cell_size", 1.0))
        rest_pos = np.zeros((self.particle_count, 3), dtype=np.float64)
        for i in range(self.grid_n):
            for j in range(self.grid_m):
                rest_pos[i * self.grid_m + j] = (i * cell_size, j * cell_size, 0.0)
        self._K_rest = build_stiffness_matrix_sparse(
            rest_pos,
            self.grid_n,
            self.grid_m,
            cell_size,
            float(self.params.get("k_struct", 1.0)),
            float(self.params.get("k_shear", 1.0)),
            float(self.params.get("k_bend", 1.0)),
        )
        from scipy import sparse

        self._mass_eye = sparse.eye(3 * self.particle_count, format="csr", dtype=np.float64)

        # Temps for multi-stage integrators (mid-point / RK4)
        self._q_tmp = wp.zeros(self.particle_count, dtype=wp.vec3, device=model.device)
        self._qd_tmp = wp.zeros(self.particle_count, dtype=wp.vec3, device=model.device)
        self._f_tmp = wp.zeros(self.particle_count, dtype=wp.vec3, device=model.device)
        self._k1_p = wp.zeros(self.particle_count, dtype=wp.vec3, device=model.device)
        self._k1_v = wp.zeros(self.particle_count, dtype=wp.vec3, device=model.device)
        self._k2_p = wp.zeros(self.particle_count, dtype=wp.vec3, device=model.device)
        self._k2_v = wp.zeros(self.particle_count, dtype=wp.vec3, device=model.device)
        self._k3_p = wp.zeros(self.particle_count, dtype=wp.vec3, device=model.device)
        self._k3_v = wp.zeros(self.particle_count, dtype=wp.vec3, device=model.device)
        self._k4_p = wp.zeros(self.particle_count, dtype=wp.vec3, device=model.device)
        self._k4_v = wp.zeros(self.particle_count, dtype=wp.vec3, device=model.device)

        dof = 3 * self.particle_count
        self._implicit_A = wp.zeros((dof, dof), dtype=float, device="cpu")
        self._implicit_b = wp.zeros(dof, dtype=float, device="cpu")
        self._implicit_v = wp.zeros(dof, dtype=float, device=model.device)

    def _force_inputs(self, q, qd, f_out) -> list:
        return [
            q,
            qd,
            f_out,
            self.particle_fixed,
            self.params.get("particle_mass", 1.0),
            self.params.get("k_struct", 1.0),
            self.params.get("k_bend", 1.0),
            self.params.get("k_shear", 1.0),
            self.params.get("damping", 1.0),
            self.params.get("cell_size", 1.0),
            self.grid_n,
            self.grid_m,
            self.gravity,
        ]

    def _eval_forces(self, q, qd, f_out) -> None:
        f_out.fill_(0.0)
        wp.launch(
            compute_particle_force_kernel,
            dim=(self.grid_n, self.grid_m),
            inputs=self._force_inputs(q, qd, f_out),
        )

    def _step_explicit_euler(self, state_in: State, state_out: State, dt: float) -> None:
        self._eval_forces(state_in.particle_q, state_in.particle_qd, self.particle_f)
        wp.launch(
            explicit_euler_integrate_kernel,
            dim=self.particle_count,
            inputs=[
                state_in.particle_q,
                state_in.particle_qd,
                state_out.particle_q,
                state_out.particle_qd,
                self.particle_f,
                self.particle_fixed,
                self.params.get("particle_mass", 1.0),
                dt,
                self.grid_n,
                self.grid_m,
            ],
        )

    def _step_semi_implicit_euler(self, state_in: State, state_out: State, dt: float) -> None:
        self._eval_forces(state_in.particle_q, state_in.particle_qd, self.particle_f)
        wp.launch(
            semi_implicit_euler_integrate_kernel,
            dim=self.particle_count,
            inputs=[
                state_in.particle_q,
                state_in.particle_qd,
                state_out.particle_q,
                state_out.particle_qd,
                self.particle_f,
                self.particle_fixed,
                self.params.get("particle_mass", 1.0),
                dt,
                self.grid_n,
                self.grid_m,
            ],
        )

    def _step_mid_point(self, state_in: State, state_out: State, dt: float) -> None:
        mass = self.params.get("particle_mass", 1.0)
        self._eval_forces(state_in.particle_q, state_in.particle_qd, self.particle_f)
        wp.launch(
            make_mid_state_kernel,
            dim=self.particle_count,
            inputs=[
                state_in.particle_q,
                state_in.particle_qd,
                self.particle_f,
                self.particle_fixed,
                self._q_tmp,
                self._qd_tmp,
                mass,
                dt,
            ],
        )
        self._eval_forces(self._q_tmp, self._qd_tmp, self._f_tmp)
        wp.launch(
            midpoint_finalize_kernel,
            dim=self.particle_count,
            inputs=[
                state_in.particle_q,
                state_in.particle_qd,
                self._qd_tmp,
                self._f_tmp,
                self.particle_fixed,
                state_out.particle_q,
                state_out.particle_qd,
                mass,
                dt,
            ],
        )

    def _step_rk4(self, state_in: State, state_out: State, dt: float) -> None:
        mass = self.params.get("particle_mass", 1.0)
        q0 = state_in.particle_q
        v0 = state_in.particle_qd
        fixed = self.particle_fixed

        # k1 = f(y0)
        self._eval_forces(q0, v0, self.particle_f)
        wp.launch(
            store_deriv_kernel,
            dim=self.particle_count,
            inputs=[v0, self.particle_f, fixed, self._k1_p, self._k1_v, mass],
        )

        # k2 = f(y0 + dt/2 k1)
        wp.launch(
            rk_stage_state_kernel,
            dim=self.particle_count,
            inputs=[q0, v0, fixed, self._k1_p, self._k1_v, self._q_tmp, self._qd_tmp, 0.5 * dt],
        )
        self._eval_forces(self._q_tmp, self._qd_tmp, self.particle_f)
        wp.launch(
            store_deriv_kernel,
            dim=self.particle_count,
            inputs=[self._qd_tmp, self.particle_f, fixed, self._k2_p, self._k2_v, mass],
        )

        # k3 = f(y0 + dt/2 k2)
        wp.launch(
            rk_stage_state_kernel,
            dim=self.particle_count,
            inputs=[q0, v0, fixed, self._k2_p, self._k2_v, self._q_tmp, self._qd_tmp, 0.5 * dt],
        )
        self._eval_forces(self._q_tmp, self._qd_tmp, self.particle_f)
        wp.launch(
            store_deriv_kernel,
            dim=self.particle_count,
            inputs=[self._qd_tmp, self.particle_f, fixed, self._k3_p, self._k3_v, mass],
        )

        # k4 = f(y0 + dt k3)
        wp.launch(
            rk_stage_state_kernel,
            dim=self.particle_count,
            inputs=[q0, v0, fixed, self._k3_p, self._k3_v, self._q_tmp, self._qd_tmp, dt],
        )
        self._eval_forces(self._q_tmp, self._qd_tmp, self.particle_f)
        wp.launch(
            store_deriv_kernel,
            dim=self.particle_count,
            inputs=[self._qd_tmp, self.particle_f, fixed, self._k4_p, self._k4_v, mass],
        )

        wp.launch(
            rk4_finalize_kernel,
            dim=self.particle_count,
            inputs=[
                q0,
                v0,
                fixed,
                self._k1_p,
                self._k1_v,
                self._k2_p,
                self._k2_v,
                self._k3_p,
                self._k3_v,
                self._k4_p,
                self._k4_v,
                state_out.particle_q,
                state_out.particle_qd,
                dt,
            ],
        )

    def _step_implicit_euler(self, state_in: State, state_out: State, dt: float) -> None:
        # Quasi-implicit Euler:
        #   (M - dt^2 K0) v^{n+1} = M v^n + dt f^n
        # K0 is the rest-pose stiffness (cached). Forces f^n stay fully nonlinear.
        from scipy.sparse.linalg import LinearOperator
        from scipy.sparse.linalg import cg as scipy_cg

        mass = float(self.params.get("particle_mass", 1.0))
        self._eval_forces(state_in.particle_q, state_in.particle_qd, self.particle_f)

        velocities = state_in.particle_qd.numpy().reshape(-1).astype(np.float64)
        forces = self.particle_f.numpy().reshape(-1).astype(np.float64)
        b = mass * velocities + dt * forces

        M = mass * self._mass_eye - (dt * dt) * self._K_rest
        fixed = self._fixed_dofs
        if fixed.size:
            b[fixed] = 0.0

        def matvec(x: np.ndarray) -> np.ndarray:
            y = M @ x
            if fixed.size:
                y[fixed] = x[fixed]
            return y

        Aop = LinearOperator(M.shape, matvec=matvec, dtype=np.float64)
        try:
            v_new, _info = scipy_cg(Aop, b, x0=velocities, rtol=1e-6, maxiter=200)
        except TypeError:
            v_new, _info = scipy_cg(Aop, b, x0=velocities, tol=1e-6, maxiter=200)

        self._implicit_v = wp.array(np.asarray(v_new, dtype=np.float32), dtype=float, device=self.device)

        wp.launch(
            apply_velocity_update_kernel,
            dim=self.particle_count,
            inputs=[
                state_in.particle_q,
                state_out.particle_qd,
                state_out.particle_q,
                self.particle_fixed,
                self._implicit_v,
                dt,
            ],
        )

    def reset(self) -> None:
        pass

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> State | None:
        if self.method == 0:
            self._step_explicit_euler(state_in, state_out, dt)
        elif self.method == 1:
            self._step_implicit_euler(state_in, state_out, dt)
        elif self.method == 2:
            self._step_semi_implicit_euler(state_in, state_out, dt)
        elif self.method == 3:
            self._step_mid_point(state_in, state_out, dt)
        elif self.method == 4:
            self._step_rk4(state_in, state_out, dt)
        return None

    @override
    def update_contacts(self, contacts: Contacts) -> None:
        # Cloth-ground interaction is handled directly in the kernels, not via contacts.
        _ = contacts
