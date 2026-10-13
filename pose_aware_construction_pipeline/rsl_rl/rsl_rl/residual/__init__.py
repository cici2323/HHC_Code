"""Frozen-student residual adaptation, independent of the simulator imports."""

from .policy import ResidualActorCritic
from .ppo import ResidualPPO

__all__ = ["ResidualActorCritic", "ResidualPPO"]
