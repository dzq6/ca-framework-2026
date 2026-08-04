import warp as wp

import newton
import newton.examples
from newton._src.solvers.ca_exercises.fluid_viewer import (
    FluidViewerGL,
    SmokeVolumeRenderer,
)
from newton._src.solvers.ca_exercises.fluid_viewer import (
    init as fluid_init,
)


class Exercise5Fluid:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.viewer = viewer
        self.viewer._paused = True
        self.args = args

        # Minimal Newton model: one dummy particle so Model/State allocation has
        # something to hold. The fluid fields live on the solver itself.
        builder = newton.ModelBuilder()
        builder.add_particle(wp.vec3(0.0, -10.0, 0.0), wp.vec3(0.0, 0.0, 0.0), 0.0)
        self.model = builder.finalize()

        self.res = (48, 48, 72)
        self.domain_size = (1.0, 1.0, 1.5)

        self.solver = newton.solvers.SolverExercise5Fluid(
            self.model,
            res=self.res,
            domain_size=self.domain_size,
            pressure_iters=25,
            buoyancy=0.1,
            wind_on=False,
            wind_strength=0.05,
            source_box=((0.35, 0.35, 0.05), (0.65, 0.65, 0.15)),
            obstacle_box=((0.40, 0.40, 0.60), (0.60, 0.60, 0.70)),
            obstacle_on=False,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.viewer.set_model(self.model)

        # Attach volumetric smoke compositor when we're on FluidViewerGL.
        self.smoke_renderer = None
        if isinstance(viewer, FluidViewerGL):
            self.smoke_renderer = SmokeVolumeRenderer(
                viewer,
                world_min=self.solver.grid_min,
                world_max=self.solver.grid_max,
                volume_resolution=(64, 64, 96),
                density_scale=4.8,
            )
            viewer.register_post_render_callback(self.smoke_renderer.render)

    def gui(self, ui):
        if ui.button("Reset"):
            self.reset()

        _changed, self.solver.wind_on = ui.checkbox("Apply Wind", self.solver.wind_on)

        obstacle_changed, new_obstacle_on = ui.checkbox("Obstacle", self.solver.obstacle_on)
        if obstacle_changed:
            self.solver.set_obstacle_active(new_obstacle_on)

    def reset(self):
        self.sim_time = 0.0
        self.solver.reset()
        self.viewer._paused = True

    def step(self):
        for _ in range(self.sim_substeps):
            self.solver.step(self.state_0, self.state_1, None, None, self.sim_dt)
        self.sim_time += self.frame_dt

    def render(self):
        if self.smoke_renderer is not None:
            self.smoke_renderer.set_dense_density(
                self.solver.density,
                self.solver.grid_min,
                self.solver.cell_size,
            )
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        import numpy as np

        density = self.solver.density.numpy()
        total = float(np.sum(density))
        assert np.all(np.isfinite(density)), "density field contains NaN or Inf"
        assert total > 0.0, "density field is empty after running the simulation"


if __name__ == "__main__":
    viewer, args = fluid_init()
    example = Exercise5Fluid(viewer, args)
    newton.examples.run(example, args)
