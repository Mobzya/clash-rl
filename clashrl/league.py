from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
import random
import time

from .model import ActorCritic


@dataclass
class LeagueEntry:
    path: str
    step: int
    rating: float = 1000.0
    created_at: float = 0.0
    tag: str = "snapshot"


class League:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "league.json"
        self.entries: list[LeagueEntry] = []
        self._load()

    def _load(self) -> None:
        if self.index_path.exists():
            raw = json.loads(self.index_path.read_text())
            self.entries = [LeagueEntry(**x) for x in raw.get("entries", []) if (self.root / x["path"]).exists()]

    def save_index(self) -> None:
        self.index_path.write_text(json.dumps({"entries": [asdict(e) for e in self.entries]}, indent=2))

    def add(self, model: ActorCritic, step: int, rating: float = 1000.0, tag: str = "snapshot") -> LeagueEntry:
        name = f"step_{step:09d}.pt"
        model.save(self.root / name, {"step": step, "rating": rating, "tag": tag})
        e = LeagueEntry(name, step, rating, time.time(), tag)
        self.entries.append(e)
        self.entries.sort(key=lambda x: x.step)
        self.save_index()
        return e

    def sample(self, rng: random.Random, recent_bias: float = .65) -> LeagueEntry | None:
        if not self.entries:
            return None
        if rng.random() < recent_bias:
            pool = self.entries[-min(6, len(self.entries)):]
        else:
            pool = self.entries
        return rng.choice(pool)

    @property
    def latest(self) -> LeagueEntry | None:
        return self.entries[-1] if self.entries else None

    def load_entry(self, entry: LeagueEntry, device="cpu") -> ActorCritic:
        model, _ = ActorCritic.load(self.root / entry.path, device=device)
        return model


def elo_update(r_a: float, r_b: float, score_a: float, k: float = 24.0) -> tuple[float, float]:
    e_a = 1.0 / (1.0 + 10 ** ((r_b-r_a)/400.0))
    delta = k * (score_a - e_a)
    return r_a + delta, r_b - delta
