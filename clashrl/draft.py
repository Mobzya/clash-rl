from __future__ import annotations

from dataclasses import dataclass
import random
import numpy as np

from .cards import CARDS

DRAFT_ROUNDS = 8
DRAFT_CHOICES = 4
# ordered pair: first index goes to chooser, second index to opponent
DRAFT_ACTIONS = tuple((i, j) for i in range(DRAFT_CHOICES) for j in range(DRAFT_CHOICES) if i != j)
DRAFT_ACTION_DIM = len(DRAFT_ACTIONS)


def draft_obs_dim(card_count: int | None = None) -> int:
    n = len(CARDS) if card_count is None else int(card_count)
    return 6 * n + 2


@dataclass
class DraftState:
    rng: random.Random
    remaining: list[int]
    decks: list[list[int]]
    round_no: int = 0
    first_chooser: int = 0

    @classmethod
    def create(cls, seed: int | None = None, first_chooser: int = 0) -> "DraftState":
        rng = random.Random(seed)
        return cls(rng=rng, remaining=[c.id for c in CARDS], decks=[[], []], first_chooser=int(first_chooser))

    @property
    def done(self) -> bool:
        return self.round_no >= DRAFT_ROUNDS

    @property
    def chooser(self) -> int:
        return (self.first_chooser + self.round_no) % 2

    def offer(self) -> tuple[int, int, int, int]:
        if self.done:
            raise RuntimeError("draft already finished")
        if len(self.remaining) < DRAFT_CHOICES:
            raise RuntimeError("not enough cards left for draft")
        return tuple(self.rng.sample(self.remaining, DRAFT_CHOICES))

    def observe(self, chooser: int, offer: tuple[int, int, int, int]) -> np.ndarray:
        n = len(CARDS)
        v = np.zeros(draft_obs_dim(n), dtype=np.float32)
        off = 0
        for slot, cid in enumerate(offer):
            v[off + slot*n + cid] = 1.0
        off += 4*n
        for cid in self.decks[chooser]:
            v[off + cid] = 1.0
        off += n
        for cid in self.decks[1-chooser]:
            v[off + cid] = 1.0
        off += n
        v[off] = self.round_no / max(1, DRAFT_ROUNDS-1)
        v[off+1] = float(chooser)
        return v

    def apply(self, offer: tuple[int, int, int, int], action: int) -> tuple[int, int]:
        if not 0 <= int(action) < DRAFT_ACTION_DIM:
            raise ValueError(f"draft action {action} outside [0,{DRAFT_ACTION_DIM})")
        chooser = self.chooser
        own_i, opp_i = DRAFT_ACTIONS[int(action)]
        own_card, opp_card = offer[own_i], offer[opp_i]
        self.decks[chooser].append(own_card)
        self.decks[1-chooser].append(opp_card)
        for cid in offer:
            self.remaining.remove(cid)
        self.round_no += 1
        return own_card, opp_card

    def result(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if not self.done:
            raise RuntimeError("draft not finished")
        return tuple(self.decks[0]), tuple(self.decks[1])
