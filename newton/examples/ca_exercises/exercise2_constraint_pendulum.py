import warp as wp

import newton
import newton.examples


class Exercise2ConstraintPendulum:
    def __init__(self, viewer, args):
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.viewer._paused = True
        self.args = args

        builder = newton.ModelBuilder()

        # Create the first body with a spherical shape
        self.body1 = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 5.0, 0.0), q=wp.quat_identity()),
            mass=1.0,
            lock_inertia=True,
        )
        builder.add_shape_sphere(body=self.body1, radius=1.0)

        # Create the second body with a spherical shape
        self.body2 = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 10.0, 0.0), q=wp.quat_identity()),
            mass=1.0,
            lock_inertia=True,
        )
        builder.add_shape_sphere(body=self.body2, radius=1.0)

        self.model = builder.finalize()
        self.solver = newton.solvers.SolverExercise2ConstraintPendulum(self.model)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.initial_state = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.initial_state.assign(self.state_0)

        self.viewer.set_model(self.model)

        # Trajectory tracking
        self.trajectory_interval = 0.05  # seconds between trajectory points [s]
        self.last_trajectory_time = 0.0
        self.trajectory1_points = []
        self.trajectory1_radii = []
        self.trajectory1_colors = []
        self.trajectory2_points = []
        self.trajectory2_radii = []
        self.trajectory2_colors = []

    def gui(self, ui):
        if ui.button("Reset"):
            self.reset()

    def reset(self):
        self.sim_time = 0.0
        self.state_0.assign(self.initial_state)
        self.state_1.assign(self.initial_state)
        self.solver.reset()
        self.last_trajectory_time = 0.0

        self.trajectory1_points = []
        self.trajectory1_radii = []
        self.trajectory1_colors = []
        self.trajectory2_points = []
        self.trajectory2_radii = []
        self.trajectory2_colors = []

        self.viewer.log_points("/trajectory1", None, None, None)
        self.viewer.log_points("/trajectory2", None, None, None)
        self.viewer._paused = True

    def step(self):
        for _ in range(self.sim_substeps):
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.sim_time += self.frame_dt

        # Record trajectory point at intervals
        if self.sim_time - self.last_trajectory_time >= self.trajectory_interval:
            body1_pos = self.state_0.body_q.numpy()[self.body1]
            body2_pos = self.state_0.body_q.numpy()[self.body2]
            self.trajectory1_points.append(wp.vec3(body1_pos[0], body1_pos[1], body1_pos[2]))
            self.trajectory2_points.append(wp.vec3(body2_pos[0], body2_pos[1], body2_pos[2]))
            self.trajectory1_radii.append(0.1)
            self.trajectory2_radii.append(0.1)
            self.trajectory1_colors.append((1.0, 0.0, 0.0))
            self.trajectory2_colors.append((0.0, 1.0, 0.0))
            self.last_trajectory_time = self.sim_time

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)

        # Draw trajectory points
        if len(self.trajectory1_points) > 0:
            self.viewer.log_points(
                name="/trajectory1",
                points=wp.array(self.trajectory1_points, dtype=wp.vec3),
                radii=wp.array(self.trajectory1_radii, dtype=wp.float32),
                colors=wp.array(self.trajectory1_colors, dtype=wp.vec3),
            )

        if len(self.trajectory2_points) > 0:
            self.viewer.log_points(
                name="/trajectory2",
                points=wp.array(self.trajectory2_points, dtype=wp.vec3),
                radii=wp.array(self.trajectory2_radii, dtype=wp.float32),
                colors=wp.array(self.trajectory2_colors, dtype=wp.vec3),
            )

        self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Exercise2ConstraintPendulum(viewer, args)
    newton.examples.run(example, args)
