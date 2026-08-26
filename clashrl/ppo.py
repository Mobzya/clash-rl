from __future__ import annotations

from dataclasses import dataclass, asdict
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .bots import RandomLegalBot
from .cards import DECK_POOL
from .draft import DraftState, DRAFT_ACTION_DIM
from .env import ClashRoyaleEnv
from .league import League
from .model import ActorCritic
from .parallel import ParallelArenaPool
from .tournament import TournamentManager


@dataclass
class PPOConfig:
    rollout_steps: int = 4096
    updates: int = 100
    epochs: int = 5
    minibatch_size: int = 512
    gamma: float = .997
    gae_lambda: float = .95
    clip_ratio: float = .20
    learning_rate: float = 3e-4
    value_coef: float = .5
    entropy_coef: float = .012
    draft_coef: float = .35
    max_grad_norm: float = .7
    snapshot_every: int = 5
    seed: int = 7
    hidden: int = 384
    deck_curriculum: bool = True
    use_draft: bool = True
    num_envs: int = 16
    workers: int = 0
    tournament_enabled: bool = True
    tournament_every_games: int = 100
    tournament_games_per_pair: int = 2
    tournament_max_models: int = 6
    tournament_opponent_prob: float = .15


@dataclass
class Rollout:
    obs: np.ndarray; masks: np.ndarray; actions: np.ndarray; logp: np.ndarray
    returns: np.ndarray; advantages: np.ndarray; values: np.ndarray
    draft_obs: np.ndarray; draft_actions: np.ndarray; draft_logp: np.ndarray
    draft_returns: np.ndarray; draft_advantages: np.ndarray
    episodes: int; wins: int; losses: int; draws: int; mean_terminal_reward: float


class SelfPlayTrainer:
    def __init__(self, run_dir: str | Path, cfg: PPOConfig, device: str | None = None):
        self.run_dir = Path(run_dir); self.run_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.rng = random.Random(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        probe = ClashRoyaleEnv(seed=cfg.seed)
        self.latest_path = self.run_dir / "latest.pt"
        self.optimizer_path = self.run_dir / "optimizer.pt"
        self.log_csv = self.run_dir / "training.csv"
        self.config_path = self.run_dir / "config.json"
        if self.latest_path.exists():
            self.model, meta = ActorCritic.load(self.latest_path, device=self.device)
            if self.model.obs_dim != probe.obs_dim or self.model.action_dim != probe.action_dim:
                raise ValueError("Existing checkpoint is incompatible with the current v3.1 environment; use runs/v31")
            self.total_steps = int(meta.get("step", 0)); self.update_no = int(meta.get("update", 0)); self.total_games = int(meta.get("training_games", 0))
        else:
            self.model = ActorCritic(probe.obs_dim, probe.action_dim, cfg.hidden).to(self.device)
            self.total_steps = 0; self.update_no = 0; self.total_games = 0
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate, eps=1e-5)
        if self.optimizer_path.exists():
            try: self.optimizer.load_state_dict(torch.load(self.optimizer_path, map_location=self.device, weights_only=False))
            except Exception as exc: print(f"warning: optimizer state not restored: {exc}")
        self.league = League(self.run_dir / "league")
        self.random_bot = RandomLegalBot(seed=cfg.seed + 101)
        self._opp_cache: dict[str, ActorCritic] = {}
        n = max(1, int(cfg.num_envs))
        self._roll_envs = [None] * n
        self._roll_train_teams = [0] * n
        self._roll_opponents = [None] * n
        self._roll_obs_pairs = [None] * n
        self._roll_masks_pairs = [None] * n
        self._roll_done = [True] * n
        self._pending_drafts = [[] for _ in range(n)]
        self.parallel = ParallelArenaPool(n, cfg.workers, probe.obs_dim, probe.action_dim) if int(cfg.workers) > 0 else None
        self.tournament = TournamentManager(self.run_dir / "tournament", device=str(self.device), seed=cfg.seed ^ 0x7193)
        if cfg.tournament_enabled:
            self.tournament.ensure_initial(self.model, step=self.total_steps)
        every = max(1, int(cfg.tournament_every_games))
        self.next_tournament_game = (self.total_games // every + 1) * every
        self.config_path.write_text(json.dumps(asdict(cfg), indent=2))
        if not self.league.entries: self.league.add(self.model, 0, tag="initial")
        if not self.log_csv.exists():
            with self.log_csv.open("w", newline="") as f:
                csv.writer(f).writerow([
                    "update", "steps", "total_games", "episodes", "wins", "losses", "draws", "win_rate",
                    "mean_terminal_reward", "policy_loss", "value_loss", "entropy", "approx_kl", "draft_loss",
                    "draft_entropy", "rollout_seconds", "update_seconds", "tournament_seconds", "seconds",
                    "steps_per_sec", "workers", "tournament_rank", "tournament_rating"
                ])

    def close(self):
        if self.parallel is not None:
            self.parallel.close(); self.parallel = None

    def _sample_opponent(self):
        # A small random-bot share keeps the policy anchored to basic competence.
        if self.rng.random() < .10:
            return self.random_bot, "random"

        # Hall-of-fame champions are sampled independently of the training league.
        # This reduces forgetting without letting tournament results directly alter PPO gradients.
        if self.cfg.tournament_enabled and self.rng.random() < max(0.0, min(1.0, self.cfg.tournament_opponent_prob)):
            champ = self.tournament.champion()
            if champ is not None:
                key = f"tournament:{champ.id}"
                model = self._opp_cache.get(key)
                if model is None:
                    model, _ = ActorCritic.load(self.tournament.root / champ.path, device=self.device)
                    self._opp_cache[key] = model
                return model, key

        e = self.league.sample(self.rng)
        if e is None:
            return self.random_bot, "random"
        key = f"league:{e.path}"
        model = self._opp_cache.get(key)
        if model is None:
            model = self.league.load_entry(e, self.device)
            self._opp_cache[key] = model
            if len(self._opp_cache) > 14:
                # Keep the cache bounded while preserving whichever model was just loaded.
                for k in list(self._opp_cache):
                    if k != key:
                        self._opp_cache.pop(k)
                        break
        return model, key

    def _draft_action(self, policy, obs):
        if isinstance(policy, ActorCritic): return policy.draft_act(obs, device=self.device)
        a = self.rng.randrange(DRAFT_ACTION_DIM); return a, 0.0, 0.0

    def _draft_match(self, train_team: int, opponent, seed: int):
        ds = DraftState.create(seed=seed, first_chooser=self.rng.randrange(2)); learner_records = []
        while not ds.done:
            chooser = ds.chooser; offer = ds.offer(); obs = ds.observe(chooser, offer)
            policy = self.model if chooser == train_team else opponent
            a, lp, v = self._draft_action(policy, obs)
            if chooser == train_team: learner_records.append((obs, int(a), float(lp), float(v)))
            ds.apply(offer, a)
        return ds.result(), learner_records

    def _prepare_match(self, idx: int) -> dict:
        seed = self.rng.randrange(1_000_000_000); train_team = self.rng.randrange(2); opponent, _ = self._sample_opponent()
        if self.cfg.use_draft:
            (deck0, deck1), pending = self._draft_match(train_team, opponent, seed ^ 0x5A5A5A5A)
        elif self.cfg.deck_curriculum:
            deck0, deck1 = self.rng.choice(DECK_POOL), self.rng.choice(DECK_POOL); pending = []
        else:
            deck0 = deck1 = DECK_POOL[0]; pending = []
        self._roll_train_teams[idx] = train_team; self._roll_opponents[idx] = opponent; self._pending_drafts[idx] = pending
        return {"slot": idx, "seed": seed, "deck0": tuple(deck0), "deck1": tuple(deck1)}

    def _start_rollout_matches(self, indices: list[int]) -> None:
        if not indices: return
        specs = [self._prepare_match(idx) for idx in indices]
        if self.parallel is not None:
            results = self.parallel.reset(specs)
            for idx in indices:
                r = results[idx]; self._roll_obs_pairs[idx] = r["observations"]; self._roll_masks_pairs[idx] = r["masks"]; self._roll_done[idx] = False
            return
        for spec in specs:
            idx = spec["slot"]; seed = spec["seed"]
            env = ClashRoyaleEnv(deck0=spec["deck0"], deck1=spec["deck1"], seed=seed)
            obs = env.reset(seed); self._roll_envs[idx] = env; self._roll_obs_pairs[idx] = obs
            self._roll_masks_pairs[idx] = (env.action_mask(0), env.action_mask(1)); self._roll_done[idx] = False

    def _opponent_actions(self, active):
        out = {}; groups = {}
        for idx in active:
            opp = self._roll_opponents[idx]; team = 1 - self._roll_train_teams[idx]
            obs = self._roll_obs_pairs[idx][team]; mask = self._roll_masks_pairs[idx][team]
            if isinstance(opp, ActorCritic):
                groups.setdefault(id(opp), [opp, [], [], []]); groups[id(opp)][1].append(idx); groups[id(opp)][2].append(obs); groups[id(opp)][3].append(mask)
            else: out[idx] = opp.act(obs, mask)[0]
        for opp, idxs, obses, masks in groups.values():
            acts, _, _ = opp.act_batch(np.asarray(obses, dtype=np.float32), np.asarray(masks, dtype=np.bool_), device=self.device)
            for idx, a in zip(idxs, acts.tolist()): out[idx] = int(a)
        return out

    def _step_active(self, active, learner_actions, opponent_actions):
        action_pairs = {}
        for j, idx in enumerate(active):
            team = self._roll_train_teams[idx]; a = int(learner_actions[j]); oa = int(opponent_actions[idx])
            action_pairs[idx] = (a, oa) if team == 0 else (oa, a)
        if self.parallel is not None:
            items = [{"slot": idx, "a0": action_pairs[idx][0], "a1": action_pairs[idx][1]} for idx in active]
            return self.parallel.step(items)
        out = {}
        for idx in active:
            env = self._roll_envs[idx]; res = env.step_joint(action_pairs[idx])
            out[idx] = {
                "observations": res.observations,
                "masks": (env.action_mask(0), env.action_mask(1)),
                "rewards": res.rewards, "done": res.done, "winner": env.game.winner, "info": res.info,
            }
        return out

    def collect_rollout(self) -> Rollout:
        obs_buf=[]; mask_buf=[]; act_buf=[]; logp_buf=[]; rew_buf=[]; val_buf=[]; nextval_buf=[]; done_buf=[]; stream_buf=[]
        draft_obs=[]; draft_act=[]; draft_logp=[]; draft_ret=[]; draft_val=[]
        episodes=wins=losses=draws=0; terminal_rewards=[]; nenv=len(self._roll_obs_pairs)
        while len(obs_buf) < self.cfg.rollout_steps:
            active = list(range(min(nenv, self.cfg.rollout_steps - len(obs_buf))))
            needs = [idx for idx in active if self._roll_done[idx] or self._roll_obs_pairs[idx] is None]
            self._start_rollout_matches(needs)
            train_obs = [self._roll_obs_pairs[i][self._roll_train_teams[i]] for i in active]
            train_masks = [self._roll_masks_pairs[i][self._roll_train_teams[i]] for i in active]
            obs_batch = np.asarray(train_obs, dtype=np.float32); mask_batch = np.asarray(train_masks, dtype=np.bool_)
            actions, logps, values = self.model.act_batch(obs_batch, mask_batch, device=self.device)
            opp_actions = self._opponent_actions(active)
            step_results = self._step_active(active, actions, opp_actions)
            next_obs_for_value=[]; next_positions=[]
            for j, idx in enumerate(active):
                team = self._roll_train_teams[idx]; rr = step_results[idx]; a = int(actions[j])
                reward = float(rr["rewards"][team]); oldobs = self._roll_obs_pairs[idx][team]
                obs_buf.append(oldobs); mask_buf.append(train_masks[j]); act_buf.append(a); logp_buf.append(float(logps[j])); rew_buf.append(reward); val_buf.append(float(values[j])); done_buf.append(bool(rr["done"])); stream_buf.append(idx); nextval_buf.append(0.0)
                pos=len(nextval_buf)-1; self._roll_obs_pairs[idx]=rr["observations"]; self._roll_masks_pairs[idx]=rr["masks"]; self._roll_done[idx]=bool(rr["done"])
                if not rr["done"]:
                    next_obs_for_value.append(rr["observations"][team]); next_positions.append(pos)
                else:
                    episodes += 1; terminal_rewards.append(reward); winner = rr["winner"]
                    if winner is None: draws += 1
                    elif winner == team: wins += 1
                    else: losses += 1
                    outcome = 0.0 if winner is None else (1.0 if winner == team else -1.0)
                    for do, da, dl, dv in self._pending_drafts[idx]:
                        draft_obs.append(do); draft_act.append(da); draft_logp.append(dl); draft_ret.append(outcome); draft_val.append(dv)
                    self._pending_drafts[idx] = []
            if next_obs_for_value:
                vals = self.model.value_batch(np.asarray(next_obs_for_value, dtype=np.float32), device=self.device)
                for pos, v in zip(next_positions, vals.tolist()): nextval_buf[pos] = float(v)
        rewards=np.asarray(rew_buf,np.float32); values=np.asarray(val_buf,np.float32); nextvalues=np.asarray(nextval_buf,np.float32); dones=np.asarray(done_buf,np.float32); streams=np.asarray(stream_buf,np.int32); advantages=np.zeros_like(rewards)
        for sid in range(nenv):
            idxs=np.flatnonzero(streams==sid); gae=0.0
            for t in reversed(idxs.tolist()):
                nt=1.0-dones[t]; delta=rewards[t]+self.cfg.gamma*nextvalues[t]*nt-values[t]; gae=delta+self.cfg.gamma*self.cfg.gae_lambda*nt*gae; advantages[t]=gae
        returns=advantages+values; advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
        if draft_obs:
            dra=np.asarray(draft_ret,np.float32)-np.asarray(draft_val,np.float32); dra=(dra-dra.mean())/(dra.std()+1e-8) if len(dra)>1 else dra
            dobs=np.asarray(draft_obs,np.float32); dact=np.asarray(draft_act,np.int64); dlp=np.asarray(draft_logp,np.float32); dret=np.asarray(draft_ret,np.float32); dadv=dra.astype(np.float32)
        else:
            dobs=np.zeros((0,self.model.draft_obs_dim),np.float32); dact=np.zeros(0,np.int64); dlp=np.zeros(0,np.float32); dret=np.zeros(0,np.float32); dadv=np.zeros(0,np.float32)
        return Rollout(np.asarray(obs_buf,np.float32),np.asarray(mask_buf,np.bool_),np.asarray(act_buf,np.int64),np.asarray(logp_buf,np.float32),returns.astype(np.float32),advantages.astype(np.float32),values,dobs,dact,dlp,dret,dadv,episodes,wins,losses,draws,float(np.mean(terminal_rewards)) if terminal_rewards else 0.0)

    def update(self, r: Rollout) -> dict:
        n=len(r.actions); indices=np.arange(n); pl=[];vl=[];ent=[];kls=[]
        self.model.train()
        for _ in range(self.cfg.epochs):
            np.random.shuffle(indices)
            for start in range(0,n,self.cfg.minibatch_size):
                idx=indices[start:start+self.cfg.minibatch_size]
                obs=torch.as_tensor(r.obs[idx],device=self.device); masks=torch.as_tensor(r.masks[idx],device=self.device); actions=torch.as_tensor(r.actions[idx],device=self.device); oldlogp=torch.as_tensor(r.logp[idx],device=self.device); returns=torch.as_tensor(r.returns[idx],device=self.device); adv=torch.as_tensor(r.advantages[idx],device=self.device)
                dist,value=self.model.distribution(obs,masks); logp=dist.log_prob(actions); entropy=dist.entropy().mean(); ratio=(logp-oldlogp).exp(); policy_loss=-torch.min(ratio*adv,torch.clamp(ratio,1-self.cfg.clip_ratio,1+self.cfg.clip_ratio)*adv).mean(); value_loss=.5*(returns-value).pow(2).mean(); loss=policy_loss+self.cfg.value_coef*value_loss-self.cfg.entropy_coef*entropy
                self.optimizer.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(self.model.parameters(),self.cfg.max_grad_norm); self.optimizer.step()
                with torch.no_grad(): kl=(oldlogp-logp).mean().abs()
                pl.append(float(policy_loss.detach()));vl.append(float(value_loss.detach()));ent.append(float(entropy.detach()));kls.append(float(kl.detach()))
        dlosses=[]; dents=[]
        if len(r.draft_actions):
            didx=np.arange(len(r.draft_actions))
            for _ in range(max(2,self.cfg.epochs//2)):
                np.random.shuffle(didx)
                for start in range(0,len(didx),self.cfg.minibatch_size):
                    idx=didx[start:start+self.cfg.minibatch_size]; obs=torch.as_tensor(r.draft_obs[idx],device=self.device); acts=torch.as_tensor(r.draft_actions[idx],device=self.device); oldlp=torch.as_tensor(r.draft_logp[idx],device=self.device); ret=torch.as_tensor(r.draft_returns[idx],device=self.device); adv=torch.as_tensor(r.draft_advantages[idx],device=self.device)
                    dist,val=self.model.draft_distribution(obs); lp=dist.log_prob(acts); ratio=(lp-oldlp).exp(); pol=-torch.min(ratio*adv,torch.clamp(ratio,1-self.cfg.clip_ratio,1+self.cfg.clip_ratio)*adv).mean(); vloss=.5*(ret-val).pow(2).mean(); entropy=dist.entropy().mean(); loss=self.cfg.draft_coef*(pol+self.cfg.value_coef*vloss-self.cfg.entropy_coef*entropy)
                    self.optimizer.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(self.model.parameters(),self.cfg.max_grad_norm); self.optimizer.step(); dlosses.append(float((pol+vloss).detach()));dents.append(float(entropy.detach()))
        self.model.eval(); return {"policy_loss":float(np.mean(pl)),"value_loss":float(np.mean(vl)),"entropy":float(np.mean(ent)),"approx_kl":float(np.mean(kls)),"draft_loss":float(np.mean(dlosses)) if dlosses else 0.0,"draft_entropy":float(np.mean(dents)) if dents else 0.0}

    def save_latest(self):
        self.model.save(self.latest_path,{"step":self.total_steps,"update":self.update_no,"training_games":self.total_games,"env_version":ClashRoyaleEnv.ENV_VERSION});torch.save(self.optimizer.state_dict(),self.optimizer_path)

    def _maybe_tournament(self) -> tuple[float, int, float]:
        if not self.cfg.tournament_enabled or self.total_games < self.next_tournament_game:
            champ=self.tournament.champion(); return 0.0, (1 if champ and champ.training_games==self.total_games else 0), (champ.rating if champ else 1000.0)
        t0=time.perf_counter()
        contender=self.tournament.add_contender(self.model,step=self.total_steps,training_games=self.total_games)
        board=self.tournament.run_round_robin(games_per_pair=self.cfg.tournament_games_per_pair,max_models=self.cfg.tournament_max_models)
        every=max(1,int(self.cfg.tournament_every_games));self.next_tournament_game=(self.total_games//every+1)*every
        rank=next((r["rank"] for r in board if r["id"]==contender.id),0); rating=next((r["rating"] for r in board if r["id"]==contender.id),contender.rating)
        print(f"TOURNAMENT #{self.tournament.tournament_no}: contender={contender.id} rank={rank}/{len(board)} rating={rating:.1f}")
        for row in board[:5]:print(f"  #{row['rank']} {row['id']} W/L/D={row['wins']}/{row['losses']}/{row['draws']} Elo={row['rating']:.1f}")
        return time.perf_counter()-t0,int(rank),float(rating)

    def train(self, visualize_every: int = 0, visualize_speed: float = 8.0):
        print(f"device={self.device} obs={self.model.obs_dim} actions={self.model.action_dim} envs={len(self._roll_obs_pairs)} workers={self.cfg.workers} draft={self.cfg.use_draft}")
        try:
            for _ in range(self.cfg.updates):
                t0=time.perf_counter(); rollout=self.collect_rollout(); t1=time.perf_counter(); metrics=self.update(rollout); t2=time.perf_counter()
                self.total_steps+=len(rollout.actions);self.update_no+=1;self.total_games+=rollout.episodes
                if self.update_no%self.cfg.snapshot_every==0:self.league.add(self.model,self.total_steps)
                tournament_seconds,trank,trating=self._maybe_tournament();self.save_latest();t3=time.perf_counter()
                rs=t1-t0;us=t2-t1;seconds=t3-t0;wr=rollout.wins/max(1,rollout.wins+rollout.losses+rollout.draws);sps=len(rollout.actions)/max(rs,1e-9)
                row=[self.update_no,self.total_steps,self.total_games,rollout.episodes,rollout.wins,rollout.losses,rollout.draws,wr,rollout.mean_terminal_reward,metrics["policy_loss"],metrics["value_loss"],metrics["entropy"],metrics["approx_kl"],metrics["draft_loss"],metrics["draft_entropy"],rs,us,tournament_seconds,seconds,sps,self.cfg.workers,trank,trating]
                with self.log_csv.open("a",newline="") as f:csv.writer(f).writerow(row)
                print(f"update={self.update_no:04d} steps={self.total_steps:8d} games={self.total_games:6d} eps={rollout.episodes:3d} W/L/D={rollout.wins}/{rollout.losses}/{rollout.draws} win={wr:.3f} ent={metrics['entropy']:.3f} draft={len(rollout.draft_actions):3d} {sps:,.0f} steps/s rollout={rs:.1f}s update={us:.1f}s")
                if visualize_every and self.update_no%visualize_every==0:
                    try:
                        from .visualize import watch_models
                        e=self.league.sample(self.rng);opp=self.league.load_entry(e,self.device) if e else self.model;watch_models(self.model,opp,speed=visualize_speed,title=f"Training update {self.update_no}",device=self.device,auto_close=True,draft=True)
                    except Exception as exc:print(f"visualization skipped: {exc}")
        finally:
            self.close()
