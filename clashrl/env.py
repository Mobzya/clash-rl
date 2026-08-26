from __future__ import annotations

from dataclasses import dataclass
import random
import numpy as np

from .cards import BY_ID, CARDS, DEFAULT_DECK
from .config import CFG
from .core import GameState


@dataclass
class StepResult:
    observations: tuple[np.ndarray, np.ndarray]
    rewards: tuple[float, float]
    done: bool
    info: dict


class ClashRoyaleEnv:
    """Two-player clean-room arena environment.

    Action 0 is WAIT. Remaining actions encode (hand slot, placement bucket).
    Human play uses continuous mouse coordinates, while RL deliberately keeps a
    compact discrete action space so PPO can learn useful behaviour quickly.
    """

    ENV_VERSION = 4

    def __init__(self, deck0=DEFAULT_DECK, deck1=DEFAULT_DECK, seed: int | None = None,
                 perfect_information: bool = True):
        self.seed = seed
        self.rng = random.Random(seed)
        self.perfect_information = bool(perfect_information)
        self.placements_per_card = len(CFG.lane_x) * len(CFG.placement_depths)
        self.action_dim = 1 + 4 * self.placements_per_card
        self.card_count = len(CARDS)
        # own flag, present, onehot card, x, y, hp, shield, range, speed,
        # airborne, building, stunned, slowed, charged, lifetime
        self.unit_features = 2 + self.card_count + 13
        # global 9 + towers 6 + 8 normalized card ids + 8 card onehots + units
        self.obs_dim = 9 + 6 + 8 + 8 * self.card_count + CFG.max_units_observed * self.unit_features
        self.deck0, self.deck1 = tuple(deck0), tuple(deck1)
        self.game = GameState(seed=seed, decks=(self.deck0, self.deck1))
        self._step_no = 0

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        if seed is not None:
            self.seed = seed
            self.rng.seed(seed)
        self.game = GameState(seed=self.seed, decks=(self.deck0, self.deck1))
        self._step_no = 0
        return self.observe(0), self.observe(1)

    def action_mask(self, team: int) -> np.ndarray:
        return self.game.legal_action_mask(team, self.placements_per_card)

    def decode_action(self, team: int, action: int) -> tuple[int, float, float] | None:
        if action == 0:
            return None
        if not (1 <= action < self.action_dim):
            raise ValueError(f"Action {action} outside [0,{self.action_dim})")
        z = action - 1
        slot = z // self.placements_per_card
        pidx = z % self.placements_per_card
        x_idx = pidx % len(CFG.lane_x)
        depth_idx = pidx // len(CFG.lane_x)
        card = BY_ID[self.game.players[team].hand[slot]]
        x = CFG.lane_x[x_idx]
        depth = CFG.placement_depths[depth_idx]
        if card.kind in ("troop", "building"):
            # depth 0 is near bridge, depth 1 is close to the king tower.
            own_y = CFG.river_y + CFG.river_half_width + .55 + depth * (CFG.height - CFG.river_y - 3.3)
            y = own_y if team == 0 else CFG.height - own_y
        else:
            # Defensive / bridge / offensive spell buckets in player-relative space.
            canonical = (25.2, 16.0, 6.8)[depth_idx]
            y = canonical if team == 0 else CFG.height - canonical
        return slot, float(x), float(y)

    def step_joint(self, actions: tuple[int, int]) -> StepResult:
        if self.game.done:
            raise RuntimeError("step() called on terminal environment")
        hp_before = (self.game.total_tower_hp(0), self.game.total_tower_hp(1))
        crowns_before = tuple(p.crowns for p in self.game.players)
        order = (0, 1) if self._step_no % 2 == 0 else (1, 0)
        for team in order:
            action = int(actions[team])
            decoded = self.decode_action(team, action)
            if decoded is not None:
                slot, x, y = decoded
                self.game.play_card(team, slot, x, y)
        self.game.step_physics()
        self._step_no += 1
        hp_after = (self.game.total_tower_hp(0), self.game.total_tower_hp(1))
        crowns_after = tuple(p.crowns for p in self.game.players)

        # Dense signal is intentionally small; match outcome remains dominant.
        dmg_by_0 = max(0.0, hp_before[1] - hp_after[1])
        dmg_by_1 = max(0.0, hp_before[0] - hp_after[0])
        crown_delta_0 = crowns_after[0] - crowns_before[0]
        crown_delta_1 = crowns_after[1] - crowns_before[1]
        r0 = 0.00012 * (dmg_by_0 - dmg_by_1) + 0.10 * (crown_delta_0 - crown_delta_1)
        r1 = -r0
        if self.game.done:
            if self.game.winner == 0:
                r0 += 1.0; r1 -= 1.0
            elif self.game.winner == 1:
                r0 -= 1.0; r1 += 1.0
        return StepResult(
            observations=(self.observe(0), self.observe(1)),
            rewards=(float(r0), float(r1)),
            done=self.game.done,
            info=self.game.summary(),
        )

    def observe(self, team: int) -> np.ndarray:
        g = self.game
        opp = 1-team
        v: list[float] = []
        phase = [0.0, 0.0, 0.0]
        phase[0 if g.time < CFG.double_elixir_at else (1 if g.time < CFG.triple_elixir_at else 2)] = 1.0
        v.extend([
            min(1.0, g.time / g.max_time),
            g.players[team].elixir / CFG.max_elixir,
            g.players[opp].elixir / CFG.max_elixir if self.perfect_information else 0.0,
            g.players[team].crowns / 3.0,
            g.players[opp].crowns / 3.0,
            *phase,
            1.0 if self.perfect_information else 0.0,
        ])
        for tteam in (team, opp):
            for kind in ("king", "left", "right"):
                t = next(t for t in g.towers if t.team == tteam and t.kind == kind)
                v.append(max(0.0, t.hp) / t.max_hp)

        p = g.players[team]
        ordered_cards = p.hand + p.queue
        v.extend([cid / max(1, self.card_count-1) for cid in ordered_cards])
        for cid in ordered_cards:
            one = [0.0] * self.card_count
            one[cid] = 1.0
            v.extend(one)

        # Canonical coordinate system: own king is always at y~=1.
        units = sorted(g.units, key=lambda u: (0 if u.team == team else 1, abs(u.y-CFG.river_y), u.uid))
        for i in range(CFG.max_units_observed):
            if i >= len(units):
                v.extend([0.0] * self.unit_features)
                continue
            u = units[i]
            one = [0.0] * self.card_count
            one[u.card_id] = 1.0
            rel_y = u.y if team == 0 else CFG.height-u.y
            lifetime_frac = 0.0
            card = BY_ID[u.card_id]
            if u.is_building and card.lifetime > 0:
                lifetime_frac = max(0.0, u.lifetime/card.lifetime)
            v.extend([
                1.0 if u.team == team else 0.0,
                1.0,
                *one,
                u.x / CFG.width,
                rel_y / CFG.height,
                max(0.0, u.hp) / max(1.0, u.max_hp),
                max(0.0, u.shield_hp) / max(1.0, u.max_shield_hp) if u.max_shield_hp > 0 else 0.0,
                min(1.0, u.attack_range / 8.0),
                min(1.0, u.current_speed / 2.6),
                1.0 if u.airborne else 0.0,
                1.0 if u.is_building else 0.0,
                min(1.0, u.stun_remaining / 2.0),
                min(1.0, u.slow_remaining / 4.0),
                1.0 if u.charged else 0.0,
                lifetime_frac,
                min(1.0, u.deploy_remaining / max(.01, card.deploy_time)) if card.deploy_time > 0 else 0.0,
            ])
        arr = np.asarray(v, dtype=np.float32)
        if arr.shape != (self.obs_dim,):
            raise AssertionError((arr.shape, self.obs_dim))
        return arr
