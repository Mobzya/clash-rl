import unittest
from clashrl.cards import CARDS
from clashrl.draft import DraftState, DRAFT_ACTIONS, draft_obs_dim
from clashrl.env import ClashRoyaleEnv
from clashrl.model import ActorCritic

class DraftTests(unittest.TestCase):
    def test_draft_produces_two_valid_unique_eight_card_decks(self):
        d=DraftState.create(seed=123,first_chooser=0)
        while not d.done:
            offer=d.offer(); d.apply(offer,0)
        a,b=d.result()
        self.assertEqual(len(a),8); self.assertEqual(len(set(a)),8)
        self.assertEqual(len(b),8); self.assertEqual(len(set(b)),8)
        self.assertTrue(set(a).isdisjoint(set(b)))

    def test_draft_observation_and_model_head(self):
        d=DraftState.create(seed=1); offer=d.offer(); obs=d.observe(d.chooser,offer)
        self.assertEqual(obs.shape,(draft_obs_dim(len(CARDS)),))
        env=ClashRoyaleEnv(seed=1); model=ActorCritic(env.obs_dim,env.action_dim,64)
        a,lp,v=model.draft_act(obs)
        self.assertTrue(0 <= a < len(DRAFT_ACTIONS))
        self.assertIsInstance(lp,float); self.assertIsInstance(v,float)

if __name__=='__main__': unittest.main()
