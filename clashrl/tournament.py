from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
import random
import time

from .evaluate import evaluate
from .league import elo_update
from .model import ActorCritic


@dataclass
class TournamentEntry:
    id: str
    path: str
    step: int
    training_games: int
    rating: float = 1000.0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    games: int = 0
    tournaments: int = 0
    created_at: float = 0.0
    last_rank: int = 0
    last_wins: int = 0
    last_losses: int = 0
    last_draws: int = 0

    @property
    def score(self) -> float:
        return self.wins + 0.5 * self.draws

    @property
    def win_rate(self) -> float:
        return self.wins / max(1, self.games)


class TournamentManager:
    """Persistent tournament group / hall of fame.

    A new immutable contender is added whenever training crosses the configured
    completed-game milestone. A tournament then selects the strongest historical
    models plus the newest contenders and runs a round-robin. Every model in a
    tournament plays the same number of pairings, so *that tournament's* ranking
    is primarily by wins, then points (win + 0.5 draw), then Elo.

    Persistent hall-of-fame ranking uses Elo first because lifetime raw wins are
    biased toward older checkpoints that have had more opportunities to play.
    """

    def __init__(self, root: str | Path, *, device: str = "cpu", seed: int = 777):
        self.root = Path(root)
        self.models_dir = self.root / "models"
        self.root.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        self.standings_path = self.root / "standings.csv"
        self.last_path = self.root / "last_tournament.json"
        self.match_log = self.root / "matches.csv"
        self.device = device
        self.rng = random.Random(seed)
        self.entries: list[TournamentEntry] = []
        self.tournament_no = 0
        self._load()
        if not self.match_log.exists():
            with self.match_log.open("w", newline="") as f:
                csv.writer(f).writerow([
                    "tournament", "timestamp", "model_a", "model_b", "games",
                    "a_wins", "b_wins", "draws", "a_score", "a_rating_after", "b_rating_after",
                ])

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            raw = json.loads(self.state_path.read_text())
            self.tournament_no = int(raw.get("tournament_no", 0))
            loaded = []
            for x in raw.get("entries", []):
                if (self.root / x.get("path", "")).exists():
                    loaded.append(TournamentEntry(**x))
            self.entries = loaded
        except Exception:
            self.entries = []
            self.tournament_no = 0

    def _save(self) -> None:
        self.state_path.write_text(json.dumps({
            "tournament_no": self.tournament_no,
            "entries": [asdict(e) for e in self.entries],
        }, indent=2))
        self._write_standings()

    def _write_standings(self) -> None:
        rows = self.ranking()
        with self.standings_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "hall_rank", "id", "training_games", "step", "wins", "losses", "draws", "games",
                "win_rate", "rating", "tournaments", "last_rank", "last_wins", "last_losses",
                "last_draws", "path",
            ])
            for rank, e in enumerate(rows, 1):
                w.writerow([
                    rank, e.id, e.training_games, e.step, e.wins, e.losses, e.draws, e.games,
                    f"{e.win_rate:.6f}", f"{e.rating:.3f}", e.tournaments, e.last_rank,
                    e.last_wins, e.last_losses, e.last_draws, e.path,
                ])

    def ranking(self) -> list[TournamentEntry]:
        # Fair across generations: older models do not rank higher merely because
        # they have accumulated more total matches.
        return sorted(
            self.entries,
            key=lambda e: (e.rating, e.win_rate, e.wins - e.losses, e.training_games),
            reverse=True,
        )

    def add_contender(self, model: ActorCritic, *, step: int, training_games: int) -> TournamentEntry:
        existing = next((e for e in self.entries if e.training_games == int(training_games)), None)
        if existing is not None:
            return existing
        ident = f"g{int(training_games):07d}_s{int(step):010d}"
        rel = Path("models") / f"{ident}.pt"
        model.save(self.root / rel, {
            "step": int(step),
            "training_games": int(training_games),
            "tag": "tournament-contender",
            "tournament_id": ident,
        })
        entry = TournamentEntry(
            id=ident,
            path=str(rel),
            step=int(step),
            training_games=int(training_games),
            created_at=time.time(),
        )
        self.entries.append(entry)
        self._save()
        return entry

    def ensure_initial(self, model: ActorCritic, *, step: int = 0) -> TournamentEntry:
        if self.entries:
            return min(self.entries, key=lambda e: e.training_games)
        return self.add_contender(model, step=step, training_games=0)

    def _active_group(self, max_models: int) -> list[TournamentEntry]:
        max_models = max(2, int(max_models))
        if len(self.entries) <= max_models:
            return list(self.entries)
        # Keep historical champions while always giving new generations a chance.
        elite_count = min(2, max_models // 3 + 1)
        elite = self.ranking()[:elite_count]
        newest = sorted(self.entries, key=lambda e: e.training_games, reverse=True)[:max_models]
        out: list[TournamentEntry] = []
        seen: set[str] = set()
        for e in elite + newest:
            if e.id not in seen:
                out.append(e)
                seen.add(e.id)
            if len(out) >= max_models:
                break
        return out

    def run_round_robin(
        self,
        *,
        games_per_pair: int = 2,
        max_models: int = 6,
        deterministic: bool = True,
    ) -> list[dict]:
        active = self._active_group(max_models)
        if len(active) < 2:
            self._save()
            return []

        self.tournament_no += 1
        round_stats = {e.id: {"wins": 0, "losses": 0, "draws": 0, "games": 0} for e in active}
        cache: dict[str, ActorCritic] = {}

        def load(e: TournamentEntry) -> ActorCritic:
            model = cache.get(e.id)
            if model is None:
                model, _ = ActorCritic.load(self.root / e.path, device=self.device)
                cache[e.id] = model
            return model

        gpp = max(1, int(games_per_pair))
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                a, b = active[i], active[j]
                seed = self.rng.randrange(1_000_000_000)
                result = evaluate(
                    load(a), load(b), games=gpp, device=self.device, seed=seed,
                    deterministic=deterministic, draft=True,
                )
                a.wins += result.wins_a
                a.losses += result.wins_b
                a.draws += result.draws
                a.games += result.games
                b.wins += result.wins_b
                b.losses += result.wins_a
                b.draws += result.draws
                b.games += result.games

                sa, sb = round_stats[a.id], round_stats[b.id]
                sa["wins"] += result.wins_a; sa["losses"] += result.wins_b; sa["draws"] += result.draws; sa["games"] += result.games
                sb["wins"] += result.wins_b; sb["losses"] += result.wins_a; sb["draws"] += result.draws; sb["games"] += result.games

                # Elo sees the aggregate match score, while the visible tournament
                # ranking below remains wins-first as requested.
                a.rating, b.rating = elo_update(
                    a.rating, b.rating, result.score_a,
                    k=24.0 * min(2.0, result.games ** 0.5),
                )
                with self.match_log.open("a", newline="") as f:
                    csv.writer(f).writerow([
                        self.tournament_no, int(time.time()), a.id, b.id, result.games,
                        result.wins_a, result.wins_b, result.draws, f"{result.score_a:.6f}",
                        f"{a.rating:.3f}", f"{b.rating:.3f}",
                    ])

        board = []
        for e in active:
            s = round_stats[e.id]
            e.tournaments += 1
            e.last_wins = int(s["wins"])
            e.last_losses = int(s["losses"])
            e.last_draws = int(s["draws"])
            board.append({
                "id": e.id,
                "training_games": e.training_games,
                "step": e.step,
                "wins": s["wins"],
                "losses": s["losses"],
                "draws": s["draws"],
                "games": s["games"],
                "points": s["wins"] + 0.5 * s["draws"],
                "rating": e.rating,
            })

        board.sort(
            key=lambda x: (x["wins"], x["points"], x["rating"], x["training_games"]),
            reverse=True,
        )
        for rank, row in enumerate(board, 1):
            row["rank"] = rank
            entry = next(e for e in active if e.id == row["id"])
            entry.last_rank = rank

        self.last_path.write_text(json.dumps({
            "tournament": self.tournament_no,
            "created_at": time.time(),
            "games_per_pair": gpp,
            "active_models": [e.id for e in active],
            "leaderboard": board,
        }, indent=2))
        self._save()
        return board

    def last_leaderboard(self) -> list[dict]:
        if not self.last_path.exists():
            return []
        try:
            return list(json.loads(self.last_path.read_text()).get("leaderboard", []))
        except Exception:
            return []

    def champion(self) -> TournamentEntry | None:
        board = self.ranking()
        return board[0] if board else None

    def print_leaderboard(self, *, limit: int = 20, current_tournament: bool = False) -> None:
        if current_tournament:
            rows = self.last_leaderboard()[:max(1, int(limit))]
            if not rows:
                print("No completed tournament yet.")
                return
            print("RANK  MODEL                     TRAIN_GAMES   W    L    D    ELO")
            for r in rows:
                print(f"{r['rank']:>4}  {r['id']:<24} {r['training_games']:>11} {r['wins']:>4} {r['losses']:>4} {r['draws']:>4} {r['rating']:>7.1f}")
            return

        rows = self.ranking()[:max(1, int(limit))]
        if not rows:
            print("Tournament group is empty.")
            return
        print("HALL  MODEL                     TRAIN_GAMES  LIFE W/L/D       WIN%    ELO  LAST")
        for i, e in enumerate(rows, 1):
            print(
                f"{i:>4}  {e.id:<24} {e.training_games:>11}  "
                f"{e.wins:>4}/{e.losses:<4}/{e.draws:<4} {100.0*e.win_rate:>6.1f}% {e.rating:>7.1f}  #{e.last_rank or '-'}"
            )
