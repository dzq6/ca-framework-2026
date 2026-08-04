import warp as wp

from ...core.types import override
from ...sim import Contacts, Control, Model, State
from ..solver import SolverBase

"""
Notes: Newton uses warp as its base language, which is similar to python,
but with some additional features. You can utilize the GPU by writing simple
kernel functions instead explicitly call the cuda or other gpu apis.

In this exercise, Actually there is only 1 body in the scene, but we also
use kernel functions to manage the states, which helps you get familiar with 
the syntax of warp and how the kernel functions work.
"""


@wp.kernel
def analytic_kernel(
    body_q_out: wp.array(dtype=wp.transform),
    body_qd_out: wp.array(dtype=wp.spatial_vector),
    initial_poses: wp.array(dtype=wp.transform),
    initial_velocities: wp.array(dtype=wp.spatial_vector),
    gravity: wp.vec3,
    time: float,
):
    i = wp.tid()

    # We do not consider angular motion here

    # Initial position and velocity
    x0 = initial_poses[i]
    v0 = initial_velocities[i]
    p0 = x0.p
    lin_v0 = wp.vec3(v0[0], v0[1], v0[2])

    # New position and velocity
    p = p0 + lin_v0 * time + 0.5 * gravity * (time * time)
    v = lin_v0 + gravity * time

    # Update the output state
    body_q_out[i] = wp.transform(p=p, q=x0.q)
    body_qd_out[i] = wp.spatial_vector(v[0], v[1], v[2], v0[3], v0[4], v0[5])


@wp.kernel
def explicit_euler_kernel(
    body_q_in: wp.array(dtype=wp.transform),
    body_qd_in: wp.array(dtype=wp.spatial_vector),
    body_q_out: wp.array(dtype=wp.transform),
    body_qd_out: wp.array(dtype=wp.spatial_vector),
    gravity: wp.vec3,
    dt: float,
):
    # Explicit Euler: p' = p + v*dt, v' = v + g*dt (uses old velocity for position)
    i = wp.tid()

    x = body_q_in[i]
    v_full = body_qd_in[i]
    p = x.p
    v = wp.vec3(v_full[0], v_full[1], v_full[2])

    p_new = p + v * dt
    v_new = v + gravity * dt

    body_q_out[i] = wp.transform(p=p_new, q=x.q)
    body_qd_out[i] = wp.spatial_vector(v_new[0], v_new[1], v_new[2], v_full[3], v_full[4], v_full[5])


@wp.kernel
def semi_implicit_euler_kernel(
    body_q_in: wp.array(dtype=wp.transform),
    body_qd_in: wp.array(dtype=wp.spatial_vector),
    body_q_out: wp.array(dtype=wp.transform),
    body_qd_out: wp.array(dtype=wp.spatial_vector),
    gravity: wp.vec3,
    dt: float,
):
    # Semi-implicit (symplectic) Euler: update v first, then integrate p with new v
    i = wp.tid()

    x = body_q_in[i]
    v_full = body_qd_in[i]
    p = x.p
    v = wp.vec3(v_full[0], v_full[1], v_full[2])

    v_new = v + gravity * dt
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
    dt: float,
):
    # Mid-point (RK2): evaluate derivative at the half-step state, then take a full step
    i = wp.tid()

    x = body_q_in[i]
    v_full = body_qd_in[i]
    p = x.p
    v = wp.vec3(v_full[0], v_full[1], v_full[2])

    # Half-step prediction (for constant gravity, a_mid = gravity)
    v_mid = v + gravity * (0.5 * dt)

    # Full step using midpoint velocity / acceleration
    p_new = p + v_mid * dt
    v_new = v + gravity * dt

    body_q_out[i] = wp.transform(p=p_new, q=x.q)
    body_qd_out[i] = wp.spatial_vector(v_new[0], v_new[1], v_new[2], v_full[3], v_full[4], v_full[5])


@wp.kernel
def rk4_kernel(
    body_q_in: wp.array(dtype=wp.transform),
    body_qd_in: wp.array(dtype=wp.spatial_vector),
    body_q_out: wp.array(dtype=wp.transform),
    body_qd_out: wp.array(dtype=wp.spatial_vector),
    gravity: wp.vec3,
    dt: float,
):
    # Classical RK4 for y' = f(y), with y = (p, v) and f = (v, g)
    i = wp.tid()

    x = body_q_in[i]
    v_full = body_qd_in[i]
    p = x.p
    v = wp.vec3(v_full[0], v_full[1], v_full[2])

    # k1 = f(p, v)
    k1_p = v
    k1_v = gravity

    # k2 = f(p + dt/2*k1_p, v + dt/2*k1_v)
    k2_p = v + k1_v * (0.5 * dt)
    k2_v = gravity

    # k3 = f(p + dt/2*k2_p, v + dt/2*k2_v)
    k3_p = v + k2_v * (0.5 * dt)
    k3_v = gravity

    # k4 = f(p + dt*k3_p, v + dt*k3_v)
    k4_p = v + k3_v * dt
    k4_v = gravity

    p_new = p + (dt / 6.0) * (k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p)
    v_new = v + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

    body_q_out[i] = wp.transform(p=p_new, q=x.q)
    body_qd_out[i] = wp.spatial_vector(v_new[0], v_new[1], v_new[2], v_full[3], v_full[4], v_full[5])


class SolverExercise1ODECannonBall(SolverBase):
    def __init__(self, model: Model):
        super().__init__(model)

        self.method = 0  # 0: Analytic, 1: Explicit Euler, 2: Semi-Implicit Euler, 3: Mid-Point, 4: RK4

        self.gravity = -9.81

        # For the analytic solution, we will need to store the initial
        # poses and velocities of the bodies, as well as a time variable
        # to track the simulation time.
        self.time = 0.0
        self.initial_poses = wp.clone(self.model.body_q)
        self.initial_velocities = wp.clone(self.model.body_qd)

    def reset(self):
        self.time = 0.0
        self.initial_poses = wp.clone(self.model.body_q)
        self.initial_velocities = wp.clone(self.model.body_qd)

    def analytic(self, state_in: State, state_out: State, dt: float):
        self.time += dt
        wp.launch(
            analytic_kernel,
            dim=self.model.body_count,
            inputs=[
                state_out.body_q,
                state_out.body_qd,
                self.initial_poses,
                self.initial_velocities,
                self.gravity * wp.vec3(0.0, 0.0, 1.0),
                self.time,
            ],
        )

    def explicit_euler(self, state_in: State, state_out: State, dt: float):
        wp.launch(
            explicit_euler_kernel,
            dim=self.model.body_count,
            inputs=[
                state_in.body_q,
                state_in.body_qd,
                state_out.body_q,
                state_out.body_qd,
                self.gravity * wp.vec3(0.0, 0.0, 1.0),
                dt,
            ],
        )

    def semi_implicit_euler(self, state_in: State, state_out: State, dt: float):
        wp.launch(
            semi_implicit_euler_kernel,
            dim=self.model.body_count,
            inputs=[
                state_in.body_q,
                state_in.body_qd,
                state_out.body_q,
                state_out.body_qd,
                self.gravity * wp.vec3(0.0, 0.0, 1.0),
                dt,
            ],
        )

    def mid_point(self, state_in: State, state_out: State, dt: float):
        wp.launch(
            mid_point_kernel,
            dim=self.model.body_count,
            inputs=[
                state_in.body_q,
                state_in.body_qd,
                state_out.body_q,
                state_out.body_qd,
                self.gravity * wp.vec3(0.0, 0.0, 1.0),
                dt,
            ],
        )

    def rk4(self, state_in: State, state_out: State, dt: float):
        wp.launch(
            rk4_kernel,
            dim=self.model.body_count,
            inputs=[
                state_in.body_q,
                state_in.body_qd,
                state_out.body_q,
                state_out.body_qd,
                self.gravity * wp.vec3(0.0, 0.0, 1.0),
                dt,
            ],
        )

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
