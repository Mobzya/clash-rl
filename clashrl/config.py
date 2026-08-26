from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GameConfig:
    # Arena is deliberately close to Clash Royale proportions while remaining a
    # clean-room simulator. Team 0 starts at the bottom, team 1 at the top.
    width: float = 18.0
    height: float = 32.0
    river_y: float = 16.0
    river_half_width: float = 1.05
    bridge_x: tuple[float, float] = (5.0, 13.0)
    bridge_half_width: float = 1.25

    match_seconds: float = 180.0
    overtime_seconds: float = 60.0
    physics_dt: float = 0.08
    decision_dt: float = 0.40

    max_elixir: float = 10.0
    elixir_per_second: float = 0.36
    double_elixir_at: float = 120.0
    triple_elixir_at: float = 180.0

    max_units_observed: int = 48
    max_buildings_observed: int = 12

    # 12 tactical placement buckets per hand slot. The human UI is continuous;
    # only the learning agent is discretised.
    placement_depths: tuple[float, ...] = (0.12, 0.38, 0.70)
    lane_x: tuple[float, ...] = (3.6, 6.2, 11.8, 14.4)

    tower_aggro_range: float = 7.2
    unit_aggro_range: float = 6.2
    bridge_snap_distance: float = 2.4


CFG = GameConfig()
