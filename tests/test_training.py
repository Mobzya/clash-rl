import tempfile
import unittest
import numpy as np

from clashrl.ppo import PPOConfig, SelfPlayTrainer


class TrainingTests(unittest.TestCase):
    def test_multiarena_rollout_shapes_and_finite_gae(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = PPOConfig(
                rollout_steps=32, updates=1, epochs=1, minibatch_size=16,
                hidden=32, num_envs=4, snapshot_every=10, seed=19,
            )
            t = SelfPlayTrainer(d, cfg, device='cpu')
            r = t.collect_rollout()
            self.assertEqual(r.obs.shape[0], 32)
            self.assertEqual(r.actions.shape, (32,))
            self.assertTrue(np.isfinite(r.advantages).all())
            self.assertTrue(np.isfinite(r.returns).all())
            metrics = t.update(r)
            self.assertTrue(all(np.isfinite(v) for v in metrics.values()))


if __name__ == '__main__':
    unittest.main()
