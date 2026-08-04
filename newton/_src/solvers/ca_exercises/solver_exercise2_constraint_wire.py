import warp as wp

from ...core.types import override
from ...sim import Contacts, Control, Model, State
from ..solver import SolverBase

"""
Bead on a wire (sphere constraint): C = 0.5 * (|p - o|^2 - r^2) = 0

Acceleration-level Lagrange multiplier (m = 1):
  a = g + λ ∇C,  ∇C = p - o
  ∇C · a + v · v = 0  =>  λ = -(∇C · g + v · v) / |∇C|^2

Then semi-implicit Euler + project position/velocity to fight drift.
"""


@wp.kernel
def bead_on_wire_kernel(
    body_q_in: wp.array(dtype=wp.transform),
    body_qd_in: wp.array(dtype=wp.spatial_vector),
    body_q_out: wp.array(dtype=wp.transform),
    body_qd_out: wp.array(dtype=wp.spatial_vector),
    origin: wp.vec3,
    radius: float,
    gravity: wp.vec3,
    dt: float,
):
    i = wp.tid()

    x = body_q_in[i]
    v = body_qd_in[i]
    p = x.p
    lin_v = wp.vec3(v[0], v[1], v[2])

    diff = p - origin
    # λ from  diff · (g + λ diff) + v · v = 0
    denom = wp.dot(diff, diff)
    if denom < 1.0e-8:
        denom = 1.0e-8
    lam = -(wp.dot(diff, gravity) + wp.dot(lin_v, lin_v)) / denom
    a = gravity + lam * diff

    lin_v_new = lin_v + a * dt
    p_new = p + lin_v_new * dt

    # Position projection onto the sphere
    diff_new = p_new - origin
    len_new = wp.length(diff_new)
    if len_new > 1.0e-8:
        n = diff_new / len_new
        p_new = origin + n * radius
        # Remove radial velocity so Ċ ≈ 0
        lin_v_new = lin_v_new - wp.dot(lin_v_new, n) * n

    body_q_out[i] = wp.transform(p=p_new, q=x.q)
    body_qd_out[i] = wp.spatial_vector(lin_v_new, wp.vec3(v[3], v[4], v[5]))


class SolverExercise2ConstraintWire(SolverBase):
    def __init__(self, model: Model):
        super().__init__(model)

        self.origin = wp.vec3(0.0, 0.0, 0.0)
        self.radius = 5.0

        self.gravity = -9.81

    def reset(self):
        pass

    @override
    def step(
        self, state_in: State, state_out: State, control: Control | None, contacts: Contacts | None, dt: float
    ) -> State | None:
        wp.launch(
            bead_on_wire_kernel,
            dim=state_in.body_q.shape[0],
            inputs=[
                state_in.body_q,
                state_in.body_qd,
                state_out.body_q,
                state_out.body_qd,
                self.origin,
                self.radius,
                self.gravity * wp.vec3(0.0, 0.0, 1.0),
                dt,
            ],
        )
