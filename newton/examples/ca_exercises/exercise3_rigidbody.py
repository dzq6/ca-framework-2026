import warp as wp

import newton
import newton.examples


class Exercise3RigidBody:
    def __init__(self, viewer, args):
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.iterations = 20
        self.angular_damping = 0.6
        self.baumgarte = 0.2

        self.viewer = viewer
        self.viewer._paused = True
        self.args = args

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.mu = 0.5
        builder.default_shape_cfg.restitution = 0.8
        builder.default_shape_cfg.margin = -0.005
        builder.default_shape_cfg.gap = 0.0

        # Static ground.
        builder.add_ground_plane()

        # A small stack of rigid bodies for contact-rich collisions.
        self.bodies = []
        for i in range(5):
            body = builder.add_body(
                xform=wp.transform(
                    p=wp.vec3(0.1 * i, 0.1 * i, 0.6 + 0.9 * i),
                    q=wp.quat_identity(),
                ),
                mass=1.0,
            )
            self.bodies.append(body)

            if i % 2 == 0:
                builder.add_shape_box(body=body, hx=0.25, hy=0.25, hz=0.25)
            else:
                builder.add_shape_sphere(body=body, radius=0.3)

        self.model = builder.finalize()

        self.solver = newton.solvers.SolverExercise3RigidBody(
            self.model, iterations=self.iterations, baumgarte=self.baumgarte, angular_damping=self.angular_damping
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.initial_state = self.model.state()
        self.initial_state.assign(self.state_0)

        self.viewer.set_model(self.model)

        # Per-body trajectory recording.
        self.trajectory_interval = 0.05
        self.last_trajectory_time = 0.0
        self.trajectory_points = [[] for _ in self.bodies]
        self.trajectory_radii = [[] for _ in self.bodies]
        self.trajectory_colors = [[] for _ in self.bodies]

        palette = [
            wp.vec3(0.90, 0.20, 0.20),
            wp.vec3(0.20, 0.60, 0.95),
            wp.vec3(0.20, 0.75, 0.35),
            wp.vec3(0.95, 0.70, 0.20),
            wp.vec3(0.70, 0.35, 0.90),
        ]
        self.body_colors = [palette[i % len(palette)] for i in range(len(self.bodies))]
        self._shape_colors_applied = False

    def _apply_body_shape_colors(self):
        if self._shape_colors_applied:
            return

        shape_body = self.model.shape_body.numpy()
        shape_colors = {}
        body_to_color = {body: self.body_colors[i] for i, body in enumerate(self.bodies)}

        for shape_idx, body_idx in enumerate(shape_body):
            if body_idx in body_to_color:
                shape_colors[shape_idx] = body_to_color[body_idx]

        if shape_colors:
            self.viewer.update_shape_colors(shape_colors)
            self._shape_colors_applied = True

    def gui(self, ui):
        if ui.button("Reset"):
            self.reset()

    def reset(self):
        self.sim_time = 0.0
        self.state_0.assign(self.initial_state)
        self.state_1.assign(self.initial_state)
        self.solver.reset()
        self.viewer._paused = True

        self.last_trajectory_time = 0.0
        self.trajectory_points = [[] for _ in self.bodies]
        self.trajectory_radii = [[] for _ in self.bodies]
        self.trajectory_colors = [[] for _ in self.bodies]
        for i in range(len(self.bodies)):
            self.viewer.log_points(f"/trajectory/body_{i}", None, None, None)
        self._shape_colors_applied = False

    def step(self):
        for _ in range(self.sim_substeps):
            self.solver.step(self.state_0, self.state_1, None, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.sim_time += self.frame_dt

        if self.sim_time - self.last_trajectory_time >= self.trajectory_interval:
            body_q = self.state_0.body_q.numpy()
            for i, body in enumerate(self.bodies):
                pos = body_q[body]
                self.trajectory_points[i].append(wp.vec3(pos[0], pos[1], pos[2]))
                self.trajectory_radii[i].append(0.05)
                self.trajectory_colors[i].append(self.body_colors[i])
            self.last_trajectory_time = self.sim_time

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self._apply_body_shape_colors()
        self.viewer.log_state(self.state_0)

        for i in range(len(self.bodies)):
            if len(self.trajectory_points[i]) > 0:
                self.viewer.log_points(
                    name=f"/trajectory/body_{i}",
                    points=wp.array(self.trajectory_points[i], dtype=wp.vec3),
                    radii=wp.array(self.trajectory_radii[i], dtype=wp.float32),
                    colors=wp.array(self.trajectory_colors[i], dtype=wp.vec3),
                )

        self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Exercise3RigidBody(viewer, args)
    newton.examples.run(example, args)
