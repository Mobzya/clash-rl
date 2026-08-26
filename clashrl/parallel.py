from __future__ import annotations

"""Persistent multiprocessing arena workers with shared-memory observations.

The first multiprocessing prototype returned ~hundreds of KB of NumPy arrays
through ``Pipe`` every decision step; serialization/IPC cost could exceed the
physics saved by parallelism. v3.1 keeps observations, masks, rewards and
terminal flags in shared RawArrays. Pipes now carry only compact reset specs,
action pairs and acknowledgements.
"""

from multiprocessing.connection import Connection
import multiprocessing as mp
from typing import Iterable

import numpy as np

from .env import ClashRoyaleEnv


def _views(shared, slots: int, obs_dim: int, action_dim: int):
    obs_raw, masks_raw, rewards_raw, done_raw, winner_raw = shared
    return (
        np.frombuffer(obs_raw, dtype=np.float32).reshape(slots, 2, obs_dim),
        np.frombuffer(masks_raw, dtype=np.uint8).reshape(slots, 2, action_dim),
        np.frombuffer(rewards_raw, dtype=np.float32).reshape(slots, 2),
        np.frombuffer(done_raw, dtype=np.int8).reshape(slots),
        np.frombuffer(winner_raw, dtype=np.int8).reshape(slots),
    )


def _worker_main(
    conn: Connection,
    shared,
    slots: int,
    obs_dim: int,
    action_dim: int,
) -> None:
    envs: dict[int, ClashRoyaleEnv] = {}
    obs_buf, mask_buf, reward_buf, done_buf, winner_buf = _views(shared, slots, obs_dim, action_dim)

    def write(slot: int, env: ClashRoyaleEnv, observations, rewards=(0.0, 0.0)) -> None:
        obs_buf[slot, 0] = observations[0]
        obs_buf[slot, 1] = observations[1]
        mask_buf[slot, 0] = env.action_mask(0)
        mask_buf[slot, 1] = env.action_mask(1)
        reward_buf[slot] = rewards
        done_buf[slot] = 1 if env.game.done else 0
        winner_buf[slot] = -1 if env.game.winner is None else int(env.game.winner)

    try:
        while True:
            cmd, payload = conn.recv()
            if cmd == "close":
                conn.send(True)
                return
            if cmd == "reset":
                slots_done = []
                for item in payload:
                    slot = int(item["slot"])
                    seed = int(item["seed"])
                    env = ClashRoyaleEnv(deck0=tuple(item["deck0"]), deck1=tuple(item["deck1"]), seed=seed)
                    observations = env.reset(seed)
                    envs[slot] = env
                    write(slot, env, observations)
                    slots_done.append(slot)
                conn.send(slots_done)
                continue
            if cmd == "step":
                slots_done = []
                for item in payload:
                    slot = int(item["slot"])
                    env = envs[slot]
                    res = env.step_joint((int(item["a0"]), int(item["a1"])))
                    write(slot, env, res.observations, res.rewards)
                    slots_done.append(slot)
                conn.send(slots_done)
                continue
            raise ValueError(f"unknown worker command: {cmd}")
    finally:
        try:
            conn.close()
        except Exception:
            pass


class ParallelArenaPool:
    """Fixed persistent worker processes managing independent arena slots."""

    def __init__(self, slots: int, workers: int, obs_dim: int, action_dim: int):
        self.slots = max(1, int(slots))
        self.workers = max(1, min(int(workers), self.slots))
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        # Spawn is CUDA-safe. Shared ctypes arrays avoid serialising observations
        # through pipes on every decision step.
        ctx = mp.get_context("spawn")
        self._ctx = ctx
        self._shared = (
            ctx.RawArray("f", self.slots * 2 * self.obs_dim),
            ctx.RawArray("B", self.slots * 2 * self.action_dim),
            ctx.RawArray("f", self.slots * 2),
            ctx.RawArray("b", self.slots),
            ctx.RawArray("b", self.slots),
        )
        self._obs, self._masks, self._rewards, self._done, self._winner = _views(
            self._shared, self.slots, self.obs_dim, self.action_dim
        )
        self._parents = []
        self._procs = []
        for wid in range(self.workers):
            parent, child = ctx.Pipe()
            proc = ctx.Process(
                target=_worker_main,
                args=(child, self._shared, self.slots, self.obs_dim, self.action_dim),
                name=f"clashrl-env-{wid}",
                daemon=True,
            )
            proc.start()
            child.close()
            self._parents.append(parent)
            self._procs.append(proc)
        self._closed = False

    def _owner(self, slot: int) -> int:
        return int(slot) % self.workers

    def _dispatch(self, cmd: str, items: Iterable[dict]) -> list[int]:
        grouped: list[list[dict]] = [[] for _ in range(self.workers)]
        for item in items:
            grouped[self._owner(int(item["slot"]))].append(item)
        pending = []
        for wid, batch in enumerate(grouped):
            if batch:
                self._parents[wid].send((cmd, batch))
                pending.append(wid)
        completed: list[int] = []
        for wid in pending:
            completed.extend(self._parents[wid].recv())
        return completed

    def _snapshot(self, slot: int, *, include_rewards: bool) -> dict:
        winner_code = int(self._winner[slot])
        out = {
            # Copies detach from shared memory before workers overwrite the next step.
            "observations": (self._obs[slot, 0].copy(), self._obs[slot, 1].copy()),
            "masks": (self._masks[slot, 0].astype(np.bool_, copy=True), self._masks[slot, 1].astype(np.bool_, copy=True)),
            "done": bool(self._done[slot]),
            "winner": None if winner_code < 0 else winner_code,
            "info": {},
        }
        if include_rewards:
            out["rewards"] = (float(self._rewards[slot, 0]), float(self._rewards[slot, 1]))
        return out

    def reset(self, items: Iterable[dict]) -> dict:
        slots = self._dispatch("reset", items)
        return {slot: self._snapshot(slot, include_rewards=False) for slot in slots}

    def step(self, items: Iterable[dict]) -> dict:
        slots = self._dispatch("step", items)
        return {slot: self._snapshot(slot, include_rewards=True) for slot in slots}

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for parent in self._parents:
            try: parent.send(("close", None))
            except Exception: pass
        for parent in self._parents:
            try: parent.recv()
            except Exception: pass
            try: parent.close()
            except Exception: pass
        for proc in self._procs:
            proc.join(timeout=2.0)
            if proc.is_alive():
                proc.terminate(); proc.join(timeout=1.0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
