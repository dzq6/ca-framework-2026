import warp as wp

from ...core.types import override
from ...sim import Contacts, Control, Model, State
from ..solver import SolverBase

"""
Vertical spring-mass (anchor at origin, mass m=1).

Force:
  F = -k (|p| - L0) u - c (v · u) u + m g,   u = p / |p|

Analytic: motion stays on the z-axis; shift to equilibrium and solve
the underdamped oscillator u'' + c u' + k u = 0.
"""


@wp.func
def spring_accel(
    p: wp.vec3,
    v: wp.vec3,
    gravity: wp.vec3,
    rest_length: float,
    stiffness: float,
    damping: float,
) -> wp.vec3:
    L = wp.length(p)
    if L < 1.0e-8:
        return gravity
    u = p / L
    stretch = L - rest_length
    return -stiffness * stretch * u - damping * wp.dot(v, u) * u + gravity


@wp.kernel
def analytic_kernel(
    body_q_out: wp.array(dtype=wp.transform),
    body_qd_out: wp.array(dtype=wp.spatial_vector),
    initial_poses: wp.array(dtype=wp.transform),
    gravity: wp.vec3,
    rest_length: float,
    stiffness: float,
    damping: float,
    time: float,
):
    i = wp.tid()
    x0 = initial_poses[i]

    # Equilibrium on the z-axis: 0 = -k (z_eq + L0) + g_z  =>  z_eq = g_z/k - L0
    g_z = gravity[2]
    z_eq = g_z / stiffness - rest_length

    # Underdamped free oscillation about equilibrium (m = 1)
    omega0 = wp.sqrt(stiffness)
    zeta = damping / (2.0 * omega0)
    omega_d = omega0 * wp.sqrt(1.0 - zeta * zeta)

    u0 = x0.p[2] - z_eq
    # v0 = 0 => A = u0, B = zeta*omega0*A / omega_d
    A = u0
    B = zeta * omega0 * A / omega_d

    exp_term = wp.exp(-zeta * omega0 * time)
    cos_t = wp.cos(omega_d * time)
    sin_t = wp.sin(omega_d * time)

    u = exp_term * (A * cos_t + B * sin_t)
    du = -zeta * omega0 * u + exp_term * (-A * omega_d * sin_t + B * omega_d * cos_t)

    p = wp.vec3(0.0, 0.0, z_eq + u)
    body_q_out[i] = wp.transform(p=p, q=x0.q)
    body_qd_out[i] = wp.spatial_vector(0.0, 0.0, du, 0.0, 0.0, 0.0)


@wp.kernel
def explicit_euler_kernel(
    body_q_in: wp.array(dtype=wp.transform),
    body_qd_in: wp.array(dtype=wp.spatial_vector),
    body_q_out: wp.array(dtype=wp.transform),
    body_qd_out: wp.array(dtype=wp.spatial_vector),
    gravity: wp.vec3,
    rest_length: float,
    stiffness: float,
    damping: float,
    dt: float,
):
    i = wp.tid()
    x = body_q_in[i]
    v_full = body_qd_in[i]
    p = x.p
    v = wp.vec3(v_full[0], v_full[1], v_full[2])

    a = spring_accel(p, v, gravity, rest_length, stiffness, damping)
    p_new = p + v * dt
    v_new = v + a * dt

    body_q_out[i] = wp.transform(p=p_new, q=x.q)
    body_qd_out[i] = wp.spatial_vector(v_new[0], v_new[1], v_new[2], v_full[3], v_full[4], v_full[5])


@wp.kernel
def semi_implicit_euler_kernel(
    body_q_in: wp.array(dtype=wp.transform),
    body_qd_in: wp.array(dtype=wp.spatial_vector),
    body_q_out: wp.array(dtype=wp.transform),
    body_qd_out: wp.array(dtype=wp.spatial_vector),
    gravity: wp.vec3,
    rest_length: float,
    stiffness: float,
    damping: float,
    dt: float,
):
    i = wp.tid()
    x = body_q_in[i]
    v_full = body_qd_in[i]
    p = x.p
    v = wp.vec3(v_full[0], v_full[1], v_full[2])

    a = spring_accel(p, v, gravity, rest_length, stiffness, damping)
    v_new = v + a * dt
    p_new = p + v_new * dt

    body_q_out[i] = wp.transform(p=p_new, q=x.q)
    body_qd_out[i] = wp.spatial_vector(v_new[0], v_new[1], v_new[2], v_full[3], v_full[4], v_full[5])


@wp.kernel
def mid_point_kernel(
    body_q_in: wp.array(dtype=wp.transform),
    body_qd_in: wp.array(dtype=wp.spatial_vector),
    body_q_out: wp.array(dtype=wp.transform),
    body_qd_out: wp.array(dtype=wp.spatial_vector),
    gravity: wp.vec3,
    rest_length: float,
    stiffness: float,
    damping: float,
    dt: float,
):
    i = wp.tid()
    x = body_q_in[i]
    v_full = body_qd_in[i]
    p = x.p
    v = wp.vec3(v_full[0], v_full[1], v_full[2])

    a0 = spring_accel(p, v, gravity, rest_length, stiffness, damping)
    p_mid = p + v * (0.5 * dt)
    v_mid = v + a0 * (0.5 * dt)
    a_mid = spring_accel(p_mid, v_mid, gravity, rest_length, stiffness, damping)

    p_new = p + v_mid * dt
    v_new = v + a_mid * dt

    body_q_out[i] = wp.transform(p=p_new, q=x.q)
    body_qd_out[i] = wp.spatial_vector(v_new[0], v_new[1], v_new[2], v_full[3], v_full[4], v_full[5])


@wp.kernel
def rk4_kernel(
    body_q_in: wp.array(dtype=wp.transform),
    body_qd_in: wp.array(dtype=wp.spatial_vector),
    body_q_out: wp.array(dtype=wp.transform),
    body_qd_out: wp.array(dtype=wp.spatial_vector),
    gravity: wp.vec3,
    rest_length: float,
    stiffness: float,
    damping: float,
    dt: float,
):
    i = wp.tid()
    x = body_q_in[i]
    v_full = body_qd_in[i]
    p = x.p
    v = wp.vec3(v_full[0], v_full[1], v_full[2])

    k1_p = v
    k1_v = spring_accel(p, v, gravity, rest_length, stiffness, damping)

    k2_p = v + k1_v * (0.5 * dt)
    k2_v = spring_accel(p + k1_p * (0.5 * dt), v + k1_v * (0.5 * dt), gravity, rest_length, stiffness, damping)

    k3_p = v + k2_v * (0.5 * dt)
    k3_v = spring_accel(p + k2_p * (0.5 * dt), v + k2_v * (0.5 * dt), gravity, rest_length, stiffness, damping)

    k4_p = v + k3_v * dt
    k4_v = spring_accel(p + k3_p * dt, v + k3_v * dt, gravity, rest_length, stiffness, damping)

    p_new = p + (dt / 6.0) * (k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p)
    v_new = v + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

    body_q_out[i] = wp.transform(p=p_new, q=x.q)
    body_qd_out[i] = wp.spatial_vector(v_new[0], v_new[1], v_new[2], v_full[3], v_full[4], v_full[5])


class SolverExercise1ODESpring(SolverBase):
    def __init__(self, model: Model):
        super().__init__(model)

        self.method = 0  # 0: Analytic, 1: Explicit Euler, 2: Semi-Implicit Euler, 3: Mid-Point, 4: RK4

        self.gravity = -9.81
        self.spring_rest_length = 5.0
        self.spring_stiffness = 5.0
        self.spring_damping = 0.1

        # Analytic solution needs absolute time from the initial state
        self.time = 0.0
        self.initial_poses = wp.clone(self.model.body_q)

    def reset(self):
        self.time = 0.0
        self.initial_poses = wp.clone(self.model.body_q)

    def _gravity_vec(self) -> wp.vec3:
        return self.gravity * wp.vec3(0.0, 0.0, 1.0)

    def _spring_inputs(self, state_in: State, state_out: State, dt: float) -> list:
        return [
            state_in.body_q,
            state_in.body_qd,
            state_out.body_q,
            state_out.body_qd,
            self._gravity_vec(),
            self.spring_rest_length,
            self.spring_stiffness,
            self.spring_damping,
            dt,
        ]

    def analytic(self, state_in: State, state_out: State, dt: float):
        self.time += dt
        wp.launch(
            analytic_kernel,
            dim=self.model.body_count,
            inputs=[
                state_out.body_q,
                state_out.body_qd,
                self.initial_poses,
                self._gravity_vec(),
                self.spring_rest_length,
                self.spring_stiffness,
                self.spring_damping,
                self.time,
            ],
        )

    def explicit_euler(self, state_in: State, state_out: State, dt: float):
        wp.launch(explicit_euler_kernel, dim=self.model.body_count, inputs=self._spring_inputs(state_in, state_out, dt))

    def semi_implicit_euler(self, state_in: State, state_out: State, dt: float):
        wp.launch(
            semi_implicit_euler_kernel, dim=self.model.body_count, inputs=self._spring_inputs(state_in, state_out, dt)
        )

    def mid_point(self, state_in: State, state_out: State, dt: float):
        wp.launch(mid_point_kernel, dim=self.model.body_count, inputs=self._spring_inputs(state_in, state_out, dt))

    def rk4(self, state_in: State, state_out: State, dt: float):
        wp.launch(rk4_kernel, dim=self.model.body_count, inputs=self._spring_inputs(state_in, state_out, dt))

    @override
    def step(
        self, state_in: State, state_out: State, control: Control | None, contacts: Contacts | None, dt: float
    ) -> State | None:
        if self.method == 0:
            self.analytic(state_in, state_out, dt)
        elif self.method == 1:
            self.explicit_euler(state_in, state_out, dt)
        elif self.method == 2:
            self.semi_implicit_euler(state_in, state_out, dt)
        elif self.method == 3:
            self.mid_point(state_in, state_out, dt)
        elif self.method == 4:
            self.rk4(state_in, state_out, dt)
