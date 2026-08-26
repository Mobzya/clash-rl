from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .draft import DRAFT_ACTION_DIM, draft_obs_dim


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden = int(hidden)
        self.draft_obs_dim = draft_obs_dim()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.LayerNorm(hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden, action_dim)
        self.value_head = nn.Linear(hidden, 1)
        self.draft_backbone = nn.Sequential(
            nn.Linear(self.draft_obs_dim, hidden), nn.LayerNorm(hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.draft_policy_head = nn.Linear(hidden, DRAFT_ACTION_DIM)
        self.draft_value_head = nn.Linear(hidden, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.orthogonal_(self.draft_policy_head.weight, gain=.01)
        nn.init.orthogonal_(self.draft_value_head.weight, gain=1.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(obs)
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    def distribution(self, obs: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[Categorical, torch.Tensor]:
        logits, value = self(obs)
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), -1e9)
        return Categorical(logits=logits), value

    def draft_distribution(self, obs: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        h = self.draft_backbone(obs)
        return Categorical(logits=self.draft_policy_head(h)), self.draft_value_head(h).squeeze(-1)

    @torch.no_grad()
    def draft_act(self, obs: np.ndarray, deterministic: bool = False, device: str | torch.device = "cpu") -> tuple[int, float, float]:
        x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        dist, value = self.draft_distribution(x)
        action = torch.argmax(dist.logits, dim=-1) if deterministic else dist.sample()
        return int(action.item()), float(dist.log_prob(action).item()), float(value.item())

    @torch.no_grad()
    def draft_act_batch(self, obs: np.ndarray, deterministic: bool = False, device: str | torch.device = "cpu"):
        x = torch.as_tensor(obs, dtype=torch.float32, device=device)
        dist, value = self.draft_distribution(x)
        action = torch.argmax(dist.logits, dim=-1) if deterministic else dist.sample()
        return action.cpu().numpy().astype(np.int64), dist.log_prob(action).cpu().numpy().astype(np.float32), value.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def act(self, obs: np.ndarray, mask: np.ndarray, deterministic: bool = False, device: str | torch.device = "cpu") -> tuple[int, float, float]:
        x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        m = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
        dist, value = self.distribution(x, m)
        action = torch.argmax(dist.logits, dim=-1) if deterministic else dist.sample()
        logp = dist.log_prob(action)
        return int(action.item()), float(logp.item()), float(value.item())

    @torch.no_grad()
    def act_batch(self, obs: np.ndarray, mask: np.ndarray, deterministic: bool = False,
                  device: str | torch.device = "cpu") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = torch.as_tensor(obs, dtype=torch.float32, device=device)
        m = torch.as_tensor(mask, dtype=torch.bool, device=device)
        dist, value = self.distribution(x, m)
        action = torch.argmax(dist.logits, dim=-1) if deterministic else dist.sample()
        logp = dist.log_prob(action)
        return (action.cpu().numpy().astype(np.int64),
                logp.cpu().numpy().astype(np.float32),
                value.cpu().numpy().astype(np.float32))

    @torch.no_grad()
    def value_batch(self, obs: np.ndarray, device: str | torch.device = "cpu") -> np.ndarray:
        x = torch.as_tensor(obs, dtype=torch.float32, device=device)
        _, value = self(x)
        return value.cpu().numpy().astype(np.float32)

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "hidden": self.hidden,
            "state_dict": self.state_dict(),
            "metadata": metadata or {},
        }, path)

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> tuple["ActorCritic", dict]:
        data = torch.load(path, map_location=device, weights_only=False)
        model = cls(data["obs_dim"], data["action_dim"], data.get("hidden", 256))
        model.load_state_dict(data["state_dict"])
        model.to(device)
        model.eval()
        return model, data.get("metadata", {})
