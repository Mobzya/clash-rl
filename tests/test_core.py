import unittest
import numpy as np

from clashrl.cards import DEFAULT_DECK
from clashrl.env import ClashRoyaleEnv
from clashrl.model import ActorCritic


class CoreTests(unittest.TestCase):
    def test_shapes_and_masks(self):
        e = ClashRoyaleEnv(seed=1)
        o = e.reset()
        self.assertEqual(o[0].shape, (e.obs_dim,))
        self.assertEqual(e.action_mask(0).shape, (e.action_dim,))
        self.assertTrue(e.action_mask(0)[0])

    def test_cycle_and_elixir(self):
        e = ClashRoyaleEnv(seed=2)
        e.reset()
        p = e.game.players[0]
        before = list(p.hand)
        # Find an affordable troop and deploy at a valid action bucket.
        action = next(a for a in np.flatnonzero(e.action_mask(0)) if a != 0)
        dec = e.decode_action(0, int(action))
        slot, x, y = dec
        cid = p.hand[slot]
        cost = __import__('clashrl.cards', fromlist=['BY_ID']).BY_ID[cid].cost
        elixir = p.elixir
        self.assertTrue(e.game.play_card(0, slot, x, y))
        self.assertAlmostEqual(p.elixir, elixir-cost)
        self.assertNotEqual(before, p.hand)

    def test_random_match_terminates(self):
        e = ClashRoyaleEnv(seed=3)
        obs = e.reset()
        rng = np.random.default_rng(3)
        n = 0
        while not e.game.done and n < 600:
            acts = []
            for team in (0, 1):
                legal = np.flatnonzero(e.action_mask(team))
                acts.append(int(rng.choice(legal)))
            r = e.step_joint(tuple(acts))
            obs = r.observations
            n += 1
        self.assertTrue(e.game.done)
        self.assertLessEqual(e.game.time, e.game.max_time + .51)

    def test_model_action_is_legal(self):
        e = ClashRoyaleEnv(seed=4)
        obs = e.reset()[0]
        m = ActorCritic(e.obs_dim, e.action_dim, hidden=64)
        mask = e.action_mask(0)
        for _ in range(20):
            a, lp, v = m.act(obs, mask)
            self.assertTrue(mask[a])


if __name__ == "__main__":
    unittest.main()
