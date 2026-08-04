"""WanPhys viewer extensions for fluid rendering examples."""

from .smoke_volume import SmokeVolumeRenderer
from .viewer_gl import FluidViewerGL, ScreenSpaceFluidRenderer, init

__all__ = ["FluidViewerGL", "ScreenSpaceFluidRenderer", "SmokeVolumeRenderer", "init"]
