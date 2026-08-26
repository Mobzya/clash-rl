import tempfile
import unittest
from unittest.mock import patch

from clashrl.env import ClashRoyaleEnv
from clashrl.evaluate import EvalResult
from clashrl.model import ActorCritic
from clashrl.tournament import TournamentManager


class TournamentTests(unittest.TestCase):
    def test_round_robin_wins_first_and_persists(self):
        with tempfile.TemporaryDirectory() as d:
            env = ClashRoyaleEnv(seed=1)
            model = ActorCritic(env.obs_dim, env.action_dim, hidden=32)
            tm = TournamentManager(d, device='cpu', seed=3)
            tm.add_contender(model, step=0, training_games=0)
            tm.add_contender(model, step=100, training_games=100)
            tm.add_contender(model, step=200, training_games=200)
            # Pair order is (0,1), (0,2), (1,2): A always wins.
            with patch('clashrl.tournament.evaluate', return_value=EvalResult(2, 2, 0, 0)):
                board = tm.run_round_robin(games_per_pair=2, max_models=6)
            self.assertEqual([r['wins'] for r in board], [4, 2, 0])
            self.assertEqual([r['rank'] for r in board], [1, 2, 3])
            self.assertTrue(tm.standings_path.exists())
            self.assertTrue(tm.last_path.exists())
            reloaded = TournamentManager(d, device='cpu', seed=3)
            self.assertEqual(len(reloaded.entries), 3)
            self.assertEqual(reloaded.last_leaderboard()[0]['wins'], 4)


if __name__ == '__main__':
    unittest.main()
