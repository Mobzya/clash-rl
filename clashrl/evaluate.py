from __future__ import annotations
from dataclasses import dataclass
import random
from .cards import DECK_POOL
from .draft import DraftState, DRAFT_ACTION_DIM
from .env import ClashRoyaleEnv
from .model import ActorCritic

@dataclass
class EvalResult:
    games:int; wins_a:int; wins_b:int; draws:int
    @property
    def score_a(self):return (self.wins_a+.5*self.draws)/max(1,self.games)

def _draft(policies,rng,device,seed):
    ds=DraftState.create(seed=seed,first_chooser=rng.randrange(2))
    while not ds.done:
        team=ds.chooser; offer=ds.offer(); obs=ds.observe(team,offer); p=policies[team]
        if isinstance(p,ActorCritic):a,_,_=p.draft_act(obs,deterministic=True,device=device)
        else:a=rng.randrange(DRAFT_ACTION_DIM)
        ds.apply(offer,a)
    return ds.result()

def evaluate(a,b,games=20,device='cpu',seed=1234,deterministic=True,varied_decks=True,draft=True)->EvalResult:
    rng=random.Random(seed);wa=wb=dr=0
    for k in range(games):
        swap=k%2==1; policies=[b,a] if swap else [a,b]; game_seed=rng.randrange(1_000_000_000)
        if draft:deck0,deck1=_draft(policies,rng,device,game_seed^0xABCDEF)
        elif varied_decks:deck0,deck1=rng.choice(DECK_POOL),rng.choice(DECK_POOL)
        else:deck0=deck1=DECK_POOL[0]
        env=ClashRoyaleEnv(deck0=deck0,deck1=deck1,seed=game_seed);obs=env.reset()
        while not env.game.done:
            acts=[]
            for team in (0,1):
                p=policies[team];o=obs[team];m=env.action_mask(team)
                act=p.act(o,m,deterministic=deterministic,device=device)[0] if isinstance(p,ActorCritic) else p.act(o,m)[0];acts.append(act)
            res=env.step_joint(tuple(acts));obs=res.observations
        w=env.game.winner
        if w is None:dr+=1
        else:
            a_team=1 if swap else 0
            if w==a_team:wa+=1
            else:wb+=1
    return EvalResult(games,wa,wb,dr)
