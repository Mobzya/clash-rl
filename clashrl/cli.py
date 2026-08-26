from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

import torch

from .bots import RandomLegalBot
from .env import ClashRoyaleEnv
from .evaluate import evaluate
from .model import ActorCritic
from .ppo import PPOConfig, SelfPlayTrainer
from .tournament import TournamentManager


DEFAULT_RUN = "runs/v31"


def _resolve_model(value: str, run_dir: Path, device: str):
    if value.lower() == "random":
        return RandomLegalBot()
    p = run_dir / "latest.pt" if value.lower() == "latest" else Path(value)
    if not p.exists():
        raise FileNotFoundError(f"Model not found: {p}")
    model, _ = ActorCritic.load(p, device=device)
    probe = ClashRoyaleEnv(seed=0)
    if model.obs_dim != probe.obs_dim or model.action_dim != probe.action_dim:
        raise ValueError(
            f"Checkpoint {p} belongs to an older environment (obs={model.obs_dim}, actions={model.action_dim}); "
            f"v3.1/env-v{probe.ENV_VERSION} expects obs={probe.obs_dim}, actions={probe.action_dim}. Use {DEFAULT_RUN}."
        )
    return model


def _device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value


def _workers(value: str | int, num_envs: int) -> int:
    if isinstance(value, int):
        return max(0, min(value, num_envs))
    text = str(value).strip().lower()
    if text == "auto":
        cores = os.cpu_count() or 4
        # Conservative default: enough CPU parallelism to matter without creating
        # dozens of Python workers and swamping pipes with observations.
        return max(1, min(num_envs, 8, max(1, cores // 2)))
    return max(0, min(int(text), num_envs))


def cmd_smoke(args):
    env = ClashRoyaleEnv(seed=args.seed)
    obs = env.reset()
    bot0 = RandomLegalBot(seed=args.seed)
    bot1 = RandomLegalBot(seed=args.seed + 1)
    steps = 0
    while not env.game.done and steps < args.steps:
        a0 = bot0.act(obs[0], env.action_mask(0))[0]
        a1 = bot1.act(obs[1], env.action_mask(1))[0]
        res = env.step_joint((a0, a1))
        obs = res.observations
        steps += 1
    print("SMOKE PASS")
    print("env_version=", env.ENV_VERSION, "obs_dim=", env.obs_dim, "action_dim=", env.action_dim, "steps=", steps)
    print(env.game.summary())


def cmd_doctor(args):
    print(f"Python: {platform.python_version()} ({sys.executable})")
    print(f"PyTorch: {torch.__version__}  CUDA: {torch.cuda.is_available()}")
    print(f"CPU cores visible: {os.cpu_count() or 'unknown'}")
    env = ClashRoyaleEnv(seed=1)
    print(f"ClashRL env v{env.ENV_VERSION}: obs={env.obs_dim} actions={env.action_dim}")
    try:
        import pygame
        print(f"pygame-ce/pygame import: OK ({pygame.version.ver})")
        print(f"SDL: {pygame.get_sdl_version()}")
    except Exception as exc:
        print(f"pygame-ce/pygame import: FAIL ({exc})")
        print("Fix: source .venv/bin/activate && pip install -e .")
        return 1
    return 0


def cmd_init(args):
    run_dir = Path(args.run)
    run_dir.mkdir(parents=True, exist_ok=True)
    env = ClashRoyaleEnv(seed=args.seed)
    model = ActorCritic(env.obs_dim, env.action_dim, args.hidden)
    model.save(
        run_dir / "latest.pt",
        {"step": 0, "update": 0, "training_games": 0, "env_version": env.ENV_VERSION, "note": "untrained-v3.1"},
    )
    print(f"Created untrained v3.1 model: {run_dir/'latest.pt'}")


def cmd_train(args):
    workers = _workers(args.workers, args.num_envs)
    cfg = PPOConfig(
        rollout_steps=args.rollout_steps,
        updates=args.updates,
        epochs=args.epochs,
        minibatch_size=args.minibatch,
        learning_rate=args.lr,
        entropy_coef=args.entropy,
        snapshot_every=args.snapshot_every,
        seed=args.seed,
        hidden=args.hidden,
        deck_curriculum=args.deck_curriculum,
        num_envs=args.num_envs,
        use_draft=args.draft,
        workers=workers,
        tournament_enabled=args.tournament,
        tournament_every_games=args.tournament_every_games,
        tournament_games_per_pair=args.tournament_games_per_pair,
        tournament_max_models=args.tournament_max_models,
        tournament_opponent_prob=args.tournament_opponent_prob,
    )
    print(f"environment workers: {workers} ({args.workers})")
    trainer = SelfPlayTrainer(args.run, cfg, device=_device(args.device))
    trainer.train(visualize_every=args.visualize_every, visualize_speed=args.visualize_speed)


def cmd_watch(args):
    device = _device(args.device)
    run = Path(args.run)
    a = _resolve_model(args.a, run, device)
    b = _resolve_model(args.b, run, device)
    from .visualize import watch_models
    watch_models(
        a, b, speed=args.speed, title=f"{args.a} vs {args.b}", device=device,
        seed=args.seed, draft=args.draft,
    )


def cmd_play(args):
    device = _device(args.device)
    run = Path(args.run)
    model = _resolve_model(args.model, run, device)
    if not isinstance(model, ActorCritic):
        print("Human play expects a neural model, not random.", file=sys.stderr)
        return 2
    from .visualize import human_vs_model
    human_vs_model(
        model, human_team=0, speed=args.speed, device=device, seed=args.seed, draft=args.draft,
    )
    return 0


def cmd_eval(args):
    device = _device(args.device)
    run = Path(args.run)
    a = _resolve_model(args.a, run, device)
    b = _resolve_model(args.b, run, device)
    r = evaluate(
        a, b, games=args.games, device=device, seed=args.seed,
        deterministic=not args.stochastic, draft=args.draft,
    )
    print(f"games={r.games} A_wins={r.wins_a} B_wins={r.wins_b} draws={r.draws} score_A={r.score_a:.3f}")


def cmd_dashboard(args):
    from .dashboard import run_dashboard
    run_dashboard(Path(args.run) / "training.csv")


def cmd_tournament(args):
    device = _device(args.device)
    run = Path(args.run)
    tm = TournamentManager(run / "tournament", device=device, seed=args.seed)
    latest_path = run / "latest.pt"
    if args.add_latest:
        if not latest_path.exists():
            raise FileNotFoundError(f"No latest checkpoint at {latest_path}")
        model, meta = ActorCritic.load(latest_path, device=device)
        tm.ensure_initial(model, step=0)
        entry = tm.add_contender(
            model,
            step=int(meta.get("step", 0)),
            training_games=int(meta.get("training_games", 0)),
        )
        print(f"contender ready: {entry.id}")
    board = tm.run_round_robin(
        games_per_pair=args.games_per_pair,
        max_models=args.max_models,
        deterministic=not args.stochastic,
    )
    if not board:
        print("Need at least two tournament contenders. Train past another tournament milestone or use --add-latest later.")
        return 0
    print(f"TOURNAMENT #{tm.tournament_no}")
    tm.print_leaderboard(limit=args.max_models, current_tournament=True)


def cmd_leaderboard(args):
    tm = TournamentManager(Path(args.run) / "tournament", device="cpu")
    tm.print_leaderboard(limit=args.limit, current_tournament=args.current)


def cmd_benchmark(args):
    """Measure rollout generation throughput for several worker counts."""
    candidates = []
    for token in args.workers:
        w = _workers(token, args.num_envs)
        if w not in candidates:
            candidates.append(w)
    results = []
    print(f"Benchmark: transitions={args.steps}, envs={args.num_envs}, device={args.device}")
    for workers in candidates:
        with tempfile.TemporaryDirectory(prefix="clashrl-bench-") as d:
            cfg = PPOConfig(
                rollout_steps=args.steps, updates=1, epochs=1,
                minibatch_size=min(256, args.steps), hidden=args.hidden,
                num_envs=args.num_envs, workers=workers,
                tournament_enabled=False, snapshot_every=999999, seed=args.seed,
            )
            trainer = SelfPlayTrainer(d, cfg, device=_device(args.device))
            try:
                t0 = time.perf_counter()
                rollout = trainer.collect_rollout()
                elapsed = time.perf_counter() - t0
            finally:
                trainer.close()
            sps = len(rollout.actions) / max(1e-9, elapsed)
            results.append((sps, workers, elapsed))
            print(f"  workers={workers:>2}: {sps:>9,.0f} steps/s   {elapsed:.3f}s")
    best = max(results) if results else None
    if best:
        print(f"BEST: --workers {best[1]}  ({best[0]:,.0f} steps/s in this short benchmark)")


def build_parser():
    p = argparse.ArgumentParser(
        prog="clashrl",
        description="Clash RL v3.1: learned draft, PPO league self-play, pygame arena and model tournaments",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("doctor", help="Check Python, PyTorch, pygame-ce and environment compatibility")
    s.set_defaults(func=cmd_doctor)

    s = sub.add_parser("smoke", help="Run a dependency-light simulator smoke test")
    s.add_argument("--steps", type=int, default=800)
    s.add_argument("--seed", type=int, default=1)
    s.set_defaults(func=cmd_smoke)

    s = sub.add_parser("init", help="Create an untrained v3.1 neural checkpoint")
    s.add_argument("--run", default=DEFAULT_RUN)
    s.add_argument("--hidden", type=int, default=384)
    s.add_argument("--seed", type=int, default=7)
    s.set_defaults(func=cmd_init)

    s = sub.add_parser("train", help="Train with PPO + league self-play + periodic model tournaments")
    s.add_argument("--run", default=DEFAULT_RUN)
    s.add_argument("--updates", type=int, default=100)
    s.add_argument("--rollout-steps", type=int, default=4096)
    s.add_argument("--epochs", type=int, default=5)
    s.add_argument("--minibatch", type=int, default=512)
    s.add_argument("--lr", type=float, default=3e-4)
    s.add_argument("--entropy", type=float, default=.012)
    s.add_argument("--snapshot-every", type=int, default=5)
    s.add_argument("--hidden", type=int, default=384)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    s.add_argument("--deck-curriculum", action=argparse.BooleanOptionalAction, default=True,
                   help="Randomize deck archetypes only when learned draft is disabled")
    s.add_argument("--num-envs", type=int, default=24, help="Persistent arena streams collected per rollout")
    s.add_argument("--workers", default="0", help="Arena worker processes: 0 (fastest safe default here), auto, 1, 2, 4, ...; use benchmark first")
    s.add_argument("--draft", action=argparse.BooleanOptionalAction, default=True, help="Use learned 8-round pre-match draft")
    s.add_argument("--tournament", action=argparse.BooleanOptionalAction, default=True, help="Enable periodic hall-of-fame tournaments")
    s.add_argument("--tournament-every-games", type=int, default=100, help="Create a tournament contender after crossing each N completed training games")
    s.add_argument("--tournament-games-per-pair", type=int, default=2)
    s.add_argument("--tournament-max-models", type=int, default=6)
    s.add_argument("--tournament-opponent-prob", type=float, default=.15, help="Chance to sample the hall-of-fame champion as a self-play opponent")
    s.add_argument("--visualize-every", type=int, default=0, help="Open draft+arena every N updates; 0 disables")
    s.add_argument("--visualize-speed", type=float, default=8.0)
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("watch", help="Visualize draft and model-vs-model battle")
    s.add_argument("--run", default=DEFAULT_RUN)
    s.add_argument("--a", default="latest", help="latest, random, or checkpoint path")
    s.add_argument("--b", default="random", help="latest, random, or checkpoint path")
    s.add_argument("--draft", action=argparse.BooleanOptionalAction, default=True)
    s.add_argument("--speed", type=float, default=4.0)
    s.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    s.add_argument("--seed", type=int, default=11)
    s.set_defaults(func=cmd_watch)

    s = sub.add_parser("play", help="Human draft + manual battle against a checkpoint")
    s.add_argument("--run", default=DEFAULT_RUN)
    s.add_argument("--model", default="latest")
    s.add_argument("--draft", action=argparse.BooleanOptionalAction, default=True)
    s.add_argument("--speed", type=float, default=1.0)
    s.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    s.add_argument("--seed", type=int, default=22)
    s.set_defaults(func=cmd_play)

    s = sub.add_parser("eval", help="Headless evaluation with optional learned draft")
    s.add_argument("--run", default=DEFAULT_RUN)
    s.add_argument("--a", default="latest")
    s.add_argument("--b", default="random")
    s.add_argument("--games", type=int, default=20)
    s.add_argument("--draft", action=argparse.BooleanOptionalAction, default=True)
    s.add_argument("--stochastic", action="store_true")
    s.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    s.add_argument("--seed", type=int, default=1234)
    s.set_defaults(func=cmd_eval)

    s = sub.add_parser("dashboard", help="Live pygame dashboard for training.csv")
    s.add_argument("--run", default=DEFAULT_RUN)
    s.set_defaults(func=cmd_dashboard)

    s = sub.add_parser("tournament", help="Run an extra round-robin tournament now")
    s.add_argument("--run", default=DEFAULT_RUN)
    s.add_argument("--games-per-pair", type=int, default=2)
    s.add_argument("--max-models", type=int, default=6)
    s.add_argument("--add-latest", action="store_true", help="Add current latest.pt as a contender before the tournament")
    s.add_argument("--stochastic", action="store_true")
    s.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    s.add_argument("--seed", type=int, default=991)
    s.set_defaults(func=cmd_tournament)

    s = sub.add_parser("leaderboard", help="Show persistent hall-of-fame or the last tournament table")
    s.add_argument("--run", default=DEFAULT_RUN)
    s.add_argument("--limit", type=int, default=20)
    s.add_argument("--current", action="store_true", help="Show the most recent tournament (wins-first ranking)")
    s.set_defaults(func=cmd_leaderboard)

    s = sub.add_parser("benchmark", help="Benchmark rollout throughput for worker counts on this machine")
    s.add_argument("--workers", nargs="+", default=["0", "2", "4", "auto"])
    s.add_argument("--num-envs", type=int, default=24)
    s.add_argument("--steps", type=int, default=2048)
    s.add_argument("--hidden", type=int, default=128)
    s.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    s.add_argument("--seed", type=int, default=123)
    s.set_defaults(func=cmd_benchmark)

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    ret = args.func(args)
    return 0 if ret is None else ret
