from __future__ import annotations

import random
import numpy as np


class RandomLegalBot:
    def __init__(self, wait_probability: float = .28, seed: int | None = None):
        self.wait_probability = wait_probability
        self.rng = random.Random(seed)

    def act(self, obs: np.ndarray, mask: np.ndarray, **_) -> tuple[int, float, float]:
        if self.rng.random() < self.wait_probability:
            return 0, 0.0, 0.0
        legal = np.flatnonzero(mask)
        legal = legal[legal != 0]
        if len(legal) == 0:
            return 0, 0.0, 0.0
        return int(self.rng.choice(legal.tolist())), 0.0, 0.0


class GreedyBot:
    """Environment-aware curriculum bot used mostly for manual sanity checks."""
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def act_from_env(self, env, team: int) -> int:
        from .cards import BY_ID
        p = env.game.players[team]
        candidates = []
        for slot, cid in enumerate(p.hand):
            card = BY_ID[cid]
            if card.cost <= p.elixir + 1e-9:
                candidates.append((card.cost, slot, card.kind))
        if not candidates:
            return 0
        _, slot, kind = max(candidates)
        # Agent placement buckets are depth-major, x-major. Prefer bridge for
        # troops and enemy-side target for spells.
        depth = 0 if kind != "spell" else 2
        xidx = self.rng.randrange(len(env.game.cfg.lane_x))
        pidx = depth * len(env.game.cfg.lane_x) + xidx
        return 1 + slot * env.placements_per_card + pidx
