"""Clash RL v3.1: clean-room arena simulator, learned draft, PPO self-play and model tournaments."""

from .env import ClashRoyaleEnv
from .model import ActorCritic

__all__ = ["ClashRoyaleEnv", "ActorCritic"]
__version__ = "0.3.1"
