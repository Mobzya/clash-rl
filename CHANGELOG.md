# Changelog

## v3.1

- Added animated pygame draft scene before watched matches.
- Added human KEEP/GIVE draft before human-vs-agent battle.
- Added visible card transfer animation into both deck panels and draft history.
- Added persistent tournament/hall-of-fame subsystem.
- Automatic contender checkpoint after configurable completed-game milestones (default 100).
- Added wins-first round-robin tournament table, Elo, lifetime records and CSV/JSON logs.
- Added tournament champion sampling to self-play to reduce forgetting.
- Added dashboard tournament leaderboard and completed-training-game metrics.
- Added CLI commands `tournament`, `leaderboard`, and `benchmark`.
- Added experimental shared-memory multiprocessing arena workers and fixed multiprocessing-safe `__main__` guard.
- Kept worker count at 0 in the default training script because parallel workers must be benchmarked per machine.
- Added Ice Golem, Ice Spirit, Fire Spirit, Wall Breakers, Electro Wizard and Ice Wizard (44 total cards).
- Added knockback, suicide attacks, death stun/slow, deploy pulses and on-hit stun/slow.
- Fixed suicide cards with zero direct damage failing to trigger their attack/death effect.
- Environment version bumped to 4; v3.1 checkpoints use `runs/v31`.
- Expanded tests with new mechanics, draft-render path and tournament persistence/ranking.

## v3

- Learned 8-round keep/give card draft.
- 38 cards and diversified deck pool.
- Batched opponent inference and spatial hashing.
- Persistent multi-arena PPO rollouts and throughput metrics.
- `physics_dt` moved to 0.08 for faster simulation.

## v2

- pygame-ce renderer and dashboard.
- air units, projectiles, bridges, buildings, charge, shields, status effects, spawners and death effects.
- persistent rollout state across PPO updates.
