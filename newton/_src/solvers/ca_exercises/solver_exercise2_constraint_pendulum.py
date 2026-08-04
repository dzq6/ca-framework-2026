import warp as wp

from ...core.types import override
from ...sim import Contacts, Control, Model, State
from ..solver import SolverBase

# NOTE: mat66/mat26/vec6 are complete and you can use them directly.
# DO NOT try to implement these classes.
#
# You can check the implementation of wp.types.matrix to learn how to use these classes
# Typical usage:
#   V = vec6(v0, v1, v2, v3, v4, v5)
#   M26 = mat26(m00, m01, m02, m03, m04, m05, m10, m11, m12, m13, m14, m15)
#   M66 = mat66(m00, m01, m02, m03, m04, m05, ..., m50, m51, m52, m53, m54, m55)


class vec6(wp.types.vector(length=6, dtype=float)):
    """6D vector with float (single-precision) components."""


class mat26(wp.types.matrix(shape=(2, 6), dtype=float)):
    """2x6 matrix with float (single-precision) components."""


class mat66(wp.types.matrix(shape=(6, 6), dtype=float)):
    """6x6 matrix with float (single-precision) components."""


@wp.func
def ldlt(A: wp.mat22, b: wp.vec2) -> wp.vec2:
    """Solve the linear system Ax = b for x."""
    eps = 1.0e-8

    a00 = A[0, 0]
    a01 = A[0, 1]
    a10 = A[1, 0]
    a11 = A[1, 1]

    d0 = a00
    if wp.abs(d0) < eps:
        d0 = eps

    l10 = a10 / d0
    d1 = a11 - l10 * a01
    if wp.abs(d1) < eps:
        d1 = eps

    y0 = b[0]
    y1 = b[1] - l10 * y0

    z0 = y0 / d0
    z1 = y1 / d1

    x1 = z1
    x0 = z0 - l10 * x1

    return wp.vec2(x0, x1)


@wp.kernel
def pendulum_kernel(
    body_q_in: wp.array(dtype=wp.transform),
    body_qd_in: wp.array(dtype=wp.spatial_vector),
    body_q_out: wp.array(dtype=wp.transform),
    body_qd_out: wp.array(dtype=wp.spatial_vector),
    origin: wp.vec3,
    radius1: float,
    radius2: float,
    gravity: wp.vec3,
    dt: float,
):
    if wp.tid() != 0:
        return

    # Double pendulum: two distance constraints (m1 = m2 = 1)
    #   C1 = 0.5 * (|p1 - o|^2 - r1^2) = 0
    #   C2 = 0.5 * (|p2 - p1|^2 - r2^2) = 0
    #
    # Acceleration level: J a + γ = 0,  a = F + J^T λ
    #   (J J^T) λ = -γ - J F
    # then semi-implicit Euler + sequential projection.

    x1 = body_q_in[0]
    x2 = body_q_in[1]
    v1_full = body_qd_in[0]
    v2_full = body_qd_in[1]

    p1 = x1.p
    p2 = x2.p
    v1 = wp.vec3(v1_full[0], v1_full[1], v1_full[2])
    v2 = wp.vec3(v2_full[0], v2_full[1], v2_full[2])

    d1 = p1 - origin
    d2 = p2 - p1

    # γ = J̇ v  (second derivative terms without acceleration)
    gamma = wp.vec2(wp.dot(v1, v1), wp.dot(v2 - v1, v2 - v1))

    # J F with F = [g; g]:  J F = [d1 · g; 0]
    jf = wp.vec2(wp.dot(d1, gravity), 0.0)

    # A = J J^T
    a00 = wp.dot(d1, d1)
    a01 = -wp.dot(d1, d2)
    a11 = 2.0 * wp.dot(d2, d2)
    A = wp.mat22(a00, a01, a01, a11)

    # Baumgarte stabilization: JJT λ = -γ - JF - 2α Ċ - β² C
    c1 = 0.5 * (wp.dot(d1, d1) - radius1 * radius1)
    c2 = 0.5 * (wp.dot(d2, d2) - radius2 * radius2)
    cdot1 = wp.dot(d1, v1)
    cdot2 = wp.dot(d2, v2 - v1)
    alpha = 20.0
    beta = 20.0
    stab = wp.vec2(2.0 * alpha * cdot1 + beta * beta * c1, 2.0 * alpha * cdot2 + beta * beta * c2)

    rhs = -gamma - jf - stab
    lam = ldlt(A, rhs)

    # a = F + J^T λ
    a1 = gravity + lam[0] * d1 - lam[1] * d2
    a2 = gravity + lam[1] * d2

    v1_new = v1 + a1 * dt
    v2_new = v2 + a2 * dt
    p1_new = p1 + v1_new * dt
    p2_new = p2 + v2_new * dt

    # Sequential position projection
    d1_new = p1_new - origin
    len1 = wp.length(d1_new)
    if len1 > 1.0e-8:
        n1 = d1_new / len1
        p1_new = origin + n1 * radius1
        v1_new = v1_new - wp.dot(v1_new, n1) * n1

    d2_new = p2_new - p1_new
    len2 = wp.length(d2_new)
    if len2 > 1.0e-8:
        n2 = d2_new / len2
        p2_new = p1_new + n2 * radius2
        v_rel = v2_new - v1_new
        v_rel = v_rel - wp.dot(v_rel, n2) * n2
        v2_new = v1_new + v_rel

    body_q_out[0] = wp.transform(p=p1_new, q=x1.q)
    body_q_out[1] = wp.transform(p=p2_new, q=x2.q)
    body_qd_out[0] = wp.spatial_vector(v1_new, wp.vec3(v1_full[3], v1_full[4], v1_full[5]))
    body_qd_out[1] = wp.spatial_vector(v2_new, wp.vec3(v2_full[3], v2_full[4], v2_full[5]))


class SolverExercise2ConstraintPendulum(SolverBase):
    def __init__(self, model: Model):
        super().__init__(model)

        self.origin = wp.vec3(0.0, 0.0, 0.0)
        self.radius1 = 5.0
        self.radius2 = 5.0

        self.gravity = -9.81

    def reset(self):
        pass

    @override
    def step(
        self, state_in: State, state_out: State, control: Control | None, contacts: Contacts | None, dt: float
    ) -> State | None:
        wp.launch(
            pendulum_kernel,
            dim=state_in.body_q.shape[0],
            inputs=[
                state_in.body_q,
                state_in.body_qd,
                state_out.body_q,
                state_out.body_qd,
                self.origin,
                self.radius1,
                self.radius2,
                self.gravity * wp.vec3(0.0, 0.0, 1.0),
                dt,
            ],
        )
