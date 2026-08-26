from __future__ import annotations

from dataclasses import dataclass, field
import copy
import math
import random
from typing import Any

import numpy as np

from .cards import BY_ID, BY_NAME, DEFAULT_DECK, Card
from .config import CFG, GameConfig


def _dist_xy(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


@dataclass
class Unit:
    uid: int
    team: int
    card_id: int
    x: float
    y: float
    hp: float
    max_hp: float
    damage: float
    speed: float
    attack_range: float
    attack_interval: float
    radius: float
    target_kind: str
    airborne: bool = False
    is_building: bool = False
    projectile_speed: float = 0.0
    splash_radius: float = 0.0
    shield_hp: float = 0.0
    max_shield_hp: float = 0.0
    lifetime: float = 0.0
    spawn_interval: float = 0.0
    spawn_card: str | None = None
    spawn_count: int = 0
    spawn_timer: float = 0.0
    charge_distance: float = 0.0
    charge_multiplier: float = 1.0
    charge_speed_multiplier: float = 1.0
    charge_progress: float = 0.0
    cooldown: float = 0.0
    stun_remaining: float = 0.0
    slow_remaining: float = 0.0
    slow_factor: float = 1.0
    death_processed: bool = False
    deploy_remaining: float = 0.0
    hit_stun_seconds: float = 0.0
    hit_slow_factor: float = 1.0
    hit_slow_seconds: float = 0.0
    suicide_on_attack: bool = False

    @property
    def alive(self) -> bool:
        return self.hp > 0 and (not self.is_building or self.lifetime > 0)

    @property
    def charged(self) -> bool:
        return self.charge_distance > 0 and self.charge_progress >= self.charge_distance

    @property
    def current_speed(self) -> float:
        mult = self.slow_factor if self.slow_remaining > 0 else 1.0
        if self.charged:
            mult *= self.charge_speed_multiplier
        return self.speed * mult


@dataclass
class Tower:
    tid: int
    team: int
    kind: str
    x: float
    y: float
    hp: float
    max_hp: float
    damage: float
    attack_range: float
    attack_interval: float
    projectile_speed: float = 12.0
    radius: float = 0.90
    cooldown: float = 0.0
    active: bool = True

    @property
    def alive(self) -> bool:
        return self.hp > 0


@dataclass
class Projectile:
    pid: int
    team: int
    x: float
    y: float
    target_is_tower: bool
    target_id: int
    target_x: float
    target_y: float
    damage: float
    speed: float
    splash_radius: float = 0.0
    stun_seconds: float = 0.0
    slow_factor: float = 1.0
    slow_seconds: float = 0.0
    ttl: float = 4.0
    visual: str = "shot"


@dataclass
class Effect:
    kind: str
    x: float
    y: float
    radius: float
    ttl: float
    max_ttl: float
    team: int | None = None


@dataclass
class PlayerState:
    team: int
    deck: tuple[int, ...]
    hand: list[int]
    queue: list[int]
    elixir: float = 5.0
    crowns: int = 0

    def cycle(self, slot: int) -> int:
        played = self.hand[slot]
        self.queue.append(played)
        self.hand[slot] = self.queue.pop(0)
        return played


@dataclass
class GameState:
    cfg: GameConfig = CFG
    seed: int | None = None
    decks: tuple[tuple[int, ...], tuple[int, ...]] = (DEFAULT_DECK, DEFAULT_DECK)
    time: float = 0.0
    units: list[Unit] = field(default_factory=list)
    towers: list[Tower] = field(default_factory=list)
    projectiles: list[Projectile] = field(default_factory=list)
    effects: list[Effect] = field(default_factory=list)
    players: list[PlayerState] = field(default_factory=list)
    done: bool = False
    winner: int | None = None
    next_uid: int = 1
    next_pid: int = 1
    combat_log: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        if not self.players:
            self.players = [self._make_player(0, self.decks[0]), self._make_player(1, self.decks[1])]
        if not self.towers:
            self.towers = self._make_towers()

    def clone(self) -> "GameState":
        return copy.deepcopy(self)

    def _make_player(self, team: int, deck: tuple[int, ...]) -> PlayerState:
        if len(deck) != 8:
            raise ValueError("A deck must contain exactly 8 cards")
        if len(set(deck)) != 8:
            raise ValueError("A deck must contain 8 different cards")
        order = list(deck)
        self.rng.shuffle(order)
        return PlayerState(team=team, deck=tuple(deck), hand=order[:4], queue=order[4:])

    def _make_towers(self) -> list[Tower]:
        w, h = self.cfg.width, self.cfg.height
        towers: list[Tower] = []
        for team in (0, 1):
            mirror = (lambda y: y) if team == 0 else (lambda y: h - y)
            towers.extend([
                Tower(len(towers), team, "king", w/2, mirror(29.3), 4820, 4820, 170, 7.1, 1.0, active=False, radius=1.05),
                Tower(len(towers)+1, team, "left", 5.0, mirror(26.0), 3050, 3050, 125, 7.0, .90, radius=.90),
                Tower(len(towers)+2, team, "right", 13.0, mirror(26.0), 3050, 3050, 125, 7.0, .90, radius=.90),
            ])
        return towers

    @property
    def max_time(self) -> float:
        return self.cfg.match_seconds + self.cfg.overtime_seconds

    @property
    def phase(self) -> str:
        if self.time < self.cfg.double_elixir_at:
            return "1x"
        if self.time < self.cfg.triple_elixir_at:
            return "2x"
        return "3x OT"

    def reset(self) -> None:
        fresh = GameState(cfg=self.cfg, seed=self.seed, decks=self.decks)
        self.__dict__.update(fresh.__dict__)

    def legal_action_mask(self, team: int, placements_per_card: int) -> np.ndarray:
        n = 1 + 4 * placements_per_card
        mask = np.zeros(n, dtype=np.bool_)
        mask[0] = True
        p = self.players[team]
        for slot, card_id in enumerate(p.hand):
            card = BY_ID[card_id]
            if p.elixir + 1e-9 >= card.cost:
                start = 1 + slot * placements_per_card
                mask[start:start + placements_per_card] = True
        return mask

    def play_card(self, team: int, slot: int, x: float, y: float) -> bool:
        if self.done or not 0 <= slot < 4:
            return False
        p = self.players[team]
        card = BY_ID[p.hand[slot]]
        if p.elixir + 1e-9 < card.cost:
            return False
        if card.kind in ("troop", "building") and not self._valid_spawn(team, x, y, card.kind == "building"):
            return False
        if card.kind == "spell" and not (0 <= x <= self.cfg.width and 0 <= y <= self.cfg.height):
            return False

        p.elixir -= card.cost
        p.cycle(slot)
        if card.kind == "spell":
            self._cast_spell(team, card, x, y)
        else:
            self._spawn_card_units(team, card, x, y)
        self._log(f"P{team+1} {card.name}")
        return True

    def _valid_spawn(self, team: int, x: float, y: float, building: bool = False) -> bool:
        if not (0.5 <= x <= self.cfg.width - 0.5 and 0.5 <= y <= self.cfg.height - 0.5):
            return False
        margin = self.cfg.river_half_width + .35
        if team == 0 and y < self.cfg.river_y + margin:
            return False
        if team == 1 and y > self.cfg.river_y - margin:
            return False
        if building:
            for t in self.towers:
                if t.team == team and t.alive and _dist_xy(x, y, t.x, t.y) < 1.5:
                    return False
        return True

    def _spawn_card_units(self, team: int, card: Card, x: float, y: float, count_override: int | None = None) -> None:
        count = card.count if count_override is None else count_override
        if count == 1:
            offsets = [(0.0, 0.0)]
        else:
            spread = .50 if count <= 3 else .68
            offsets = [
                (spread * math.cos(i * 2*math.pi/count), spread * math.sin(i * 2*math.pi/count))
                for i in range(count)
            ]
        for dx, dy in offsets:
            self._spawn_single(team, card, x + dx, y + dy)
        if card.deploy_damage > 0 and card.deploy_radius > 0:
            self.effects.append(Effect("deploy_zap", x, y, card.deploy_radius, .38, .38, team))
            self._splash_damage(team, x, y, card.deploy_damage, card.deploy_radius, card.deploy_stun_seconds)
        self.effects.append(Effect("spawn", x, y, max(.75, card.radius*1.8), .35, .35, team))

    def _spawn_single(self, team: int, card: Card, x: float, y: float) -> Unit:
        u = Unit(
            uid=self.next_uid, team=team, card_id=card.id,
            x=float(np.clip(x, .4, self.cfg.width-.4)),
            y=float(np.clip(y, .4, self.cfg.height-.4)),
            hp=card.hp, max_hp=card.hp, damage=card.damage, speed=card.speed,
            attack_range=card.attack_range, attack_interval=card.attack_interval,
            radius=card.radius, target_kind=card.target_kind,
            airborne=card.airborne, is_building=card.kind == "building",
            projectile_speed=card.projectile_speed, splash_radius=card.splash_radius,
            shield_hp=card.shield_hp, max_shield_hp=card.shield_hp,
            lifetime=card.lifetime if card.kind == "building" else 0.0,
            spawn_interval=card.spawn_interval, spawn_card=card.spawn_card,
            spawn_count=card.spawn_count, spawn_timer=card.spawn_interval,
            charge_distance=card.charge_distance, charge_multiplier=card.charge_multiplier,
            charge_speed_multiplier=card.charge_speed_multiplier,
            deploy_remaining=card.deploy_time,
            hit_stun_seconds=card.hit_stun_seconds, hit_slow_factor=card.hit_slow_factor,
            hit_slow_seconds=card.hit_slow_seconds, suicide_on_attack=card.suicide_on_attack,
        )
        self.next_uid += 1
        self.units.append(u)
        return u

    def _cast_spell(self, team: int, card: Card, x: float, y: float) -> None:
        kind = "spell"
        if card.name == "Zap": kind = "zap"
        elif card.name == "Fireball": kind = "fireball"
        elif card.name == "GiantSnowball": kind = "snow"
        elif card.name == "Rocket": kind = "rocket"
        elif card.name == "Arrows": kind = "arrows"
        self.effects.append(Effect(kind, x, y, card.spell_radius, .55, .55, team))
        for u in list(self.units):
            if u.team != team and u.alive and _dist_xy(u.x, u.y, x, y) <= card.spell_radius + u.radius:
                self._damage_unit(u, card.damage)
                self._apply_status(u, card.stun_seconds, card.slow_factor, card.slow_seconds)
                if u.alive and card.knockback_distance > 0:
                    self._knockback_unit(u, x, y, card.knockback_distance)
        for t in self.towers:
            if t.team != team and t.alive and _dist_xy(t.x, t.y, x, y) <= card.spell_radius + t.radius:
                self._damage_tower(t, card.damage * card.tower_damage_scale)
        self._cleanup_dead()
        self._update_crowns()

    def step_physics(self, duration: float | None = None) -> None:
        if self.done:
            return
        duration = self.cfg.decision_dt if duration is None else duration
        n = max(1, int(math.ceil(duration / self.cfg.physics_dt)))
        dt = duration / n
        for _ in range(n):
            if self.done:
                break
            self._tick(dt)

    def _tick(self, dt: float) -> None:
        self.time += dt
        if self.time >= self.cfg.triple_elixir_at:
            mult = 3.0
        elif self.time >= self.cfg.double_elixir_at:
            mult = 2.0
        else:
            mult = 1.0
        for p in self.players:
            p.elixir = min(self.cfg.max_elixir, p.elixir + self.cfg.elixir_per_second * mult * dt)

        # Status/lifetime/spawners.
        for u in list(self.units):
            if not u.alive:
                continue
            cooldown_rate = u.slow_factor if u.slow_remaining > 0 else 1.0
            u.cooldown = max(0.0, u.cooldown - dt * cooldown_rate)
            u.stun_remaining = max(0.0, u.stun_remaining - dt)
            u.deploy_remaining = max(0.0, u.deploy_remaining - dt)
            u.slow_remaining = max(0.0, u.slow_remaining - dt)
            if u.is_building:
                u.lifetime -= dt
            if u.spawn_interval > 0 and u.spawn_card and u.spawn_count > 0:
                u.spawn_timer -= dt
                if u.spawn_timer <= 0 and u.alive:
                    spawn = BY_NAME.get(u.spawn_card.lower())
                    if spawn is not None:
                        self._spawn_card_units(u.team, spawn, u.x, u.y + (.55 if u.team == 0 else -.55), count_override=u.spawn_count)
                    u.spawn_timer += u.spawn_interval

        # Spatial hash is rebuilt once per physics tick. It replaces repeated
        # O(N^2) scans in target acquisition/collision handling for swarm-heavy games.
        self._spatial_cell = 2.5
        self._spatial_grid = self._build_spatial_grid(self._spatial_cell)

        # Units/buildings act.
        for u in list(self.units):
            if not u.alive or u.stun_remaining > 0 or u.deploy_remaining > 0:
                continue
            target = self._choose_unit_target(u)
            if target is None:
                continue
            tx, ty = target.x, target.y
            d = _dist_xy(u.x, u.y, tx, ty)
            target_radius = getattr(target, "radius", .6)
            if d <= u.attack_range + target_radius:
                if u.cooldown <= 1e-9 and (u.damage > 0 or u.suicide_on_attack):
                    damage = u.damage * (u.charge_multiplier if u.charged else 1.0)
                    self._attack(u.team, u.x, u.y, target, damage, u.projectile_speed, u.splash_radius,
                                 visual=BY_ID[u.card_id].visual, stun_seconds=u.hit_stun_seconds,
                                 slow_factor=u.hit_slow_factor, slow_seconds=u.hit_slow_seconds)
                    if u.suicide_on_attack:
                        u.hp = 0.0
                    if u.charged:
                        self.effects.append(Effect("charge", tx, ty, 1.2, .28, .28, u.team))
                    u.charge_progress = 0.0
                    u.cooldown = u.attack_interval
            elif not u.is_building:
                mx, my = self._movement_target(u, tx, ty)
                moved = self._move_toward(u, mx, my, dt)
                if u.charge_distance > 0:
                    u.charge_progress = min(u.charge_distance * 1.25, u.charge_progress + moved)

        # Towers act; king activates after damage or a princess tower is lost.
        for t in self.towers:
            if t.kind == "king" and not t.active:
                princess_lost = any(x.team == t.team and x.kind != "king" and not x.alive for x in self.towers)
                if princess_lost or t.hp < t.max_hp:
                    t.active = True
            if not t.alive or not t.active:
                continue
            t.cooldown = max(0.0, t.cooldown - dt)
            enemies = [u for u in self.units if u.team != t.team and u.alive and _dist_xy(u.x, u.y, t.x, t.y) <= t.attack_range + u.radius]
            if enemies and t.cooldown <= 1e-9:
                enemy = min(enemies, key=lambda u: _dist_xy(u.x, u.y, t.x, t.y))
                self._attack(t.team, t.x, t.y, enemy, t.damage, t.projectile_speed, 0.0, visual="tower")
                t.cooldown = t.attack_interval

        self._resolve_collisions()
        self._tick_projectiles(dt)
        for e in self.effects:
            e.ttl -= dt
        self.effects = [e for e in self.effects if e.ttl > 0]

        self._cleanup_dead()
        self._update_crowns()
        self._check_end()

    def _build_spatial_grid(self, cell: float) -> dict[tuple[int,int], list[Unit]]:
        grid: dict[tuple[int,int], list[Unit]] = {}
        for u in self.units:
            if not u.alive:
                continue
            key=(int(u.x//cell),int(u.y//cell))
            grid.setdefault(key,[]).append(u)
        return grid

    def _nearby_units(self, x: float, y: float, radius: float) -> list[Unit]:
        grid=getattr(self,'_spatial_grid',None)
        cell=getattr(self,'_spatial_cell',2.5)
        if not grid:
            return [u for u in self.units if u.alive]
        cx,cy=int(x//cell),int(y//cell); reach=max(1,int(math.ceil(radius/cell)))
        out=[]
        for gx in range(cx-reach,cx+reach+1):
            for gy in range(cy-reach,cy+reach+1):
                out.extend(grid.get((gx,gy),()))
        return out

    def _target_compatible(self, attacker: Unit, target: Unit) -> bool:
        if attacker.target_kind == "buildings":
            return target.is_building
        if attacker.target_kind == "ground" and target.airborne:
            return False
        return True

    def _choose_unit_target(self, u: Unit) -> Unit | Tower | None:
        # Buildings-only units ignore troops. Other units can be pulled by nearby
        # compatible troops/buildings, matching the tactical role of distractions.
        if u.target_kind != "buildings":
            aggro=max(self.cfg.unit_aggro_range, u.attack_range + 1.1)
            nearby = [
                v for v in self._nearby_units(u.x,u.y,aggro)
                if v.uid != u.uid and v.team != u.team and v.alive and self._target_compatible(u, v)
                and _dist_xy(u.x, u.y, v.x, v.y) <= aggro
            ]
            if nearby:
                return min(nearby, key=lambda v: _dist_xy(u.x, u.y, v.x, v.y))
        else:
            buildings = [
                v for v in self.units if v.team != u.team and v.alive and v.is_building
            ]
            if buildings:
                return min(buildings, key=lambda v: _dist_xy(u.x, u.y, v.x, v.y))

        towers = [t for t in self.towers if t.team != u.team and t.alive]
        if not towers:
            return None
        lane_left = u.x < self.cfg.width / 2
        desired = "left" if lane_left else "right"
        princess = [t for t in towers if t.kind == desired]
        if princess:
            return princess[0]
        king = [t for t in towers if t.kind == "king"]
        return king[0] if king else min(towers, key=lambda t: _dist_xy(u.x, u.y, t.x, t.y))

    def _movement_target(self, u: Unit, tx: float, ty: float) -> tuple[float, float]:
        if u.airborne:
            return tx, ty
        r = self.cfg.river_y
        hw = self.cfg.river_half_width
        side_u = 1 if u.y > r else -1
        side_t = 1 if ty > r else -1
        if side_u == side_t:
            return tx, ty
        bx = min(self.cfg.bridge_x, key=lambda x: abs(u.x-x) + .25*abs(tx-x))
        # Approach the bridge mouth, traverse it, then resume direct path.
        if abs(u.y-r) > hw + .18:
            return bx, r + side_u * (hw + .05)
        return bx, r - side_u * (hw + .30)

    def _move_toward(self, u: Unit, tx: float, ty: float, dt: float) -> float:
        dx, dy = tx-u.x, ty-u.y
        d = math.hypot(dx, dy)
        if d < 1e-8:
            return 0.0
        step = min(d, u.current_speed * dt)
        u.x += dx/d * step
        u.y += dy/d * step
        u.x = float(np.clip(u.x, .2, self.cfg.width-.2))
        u.y = float(np.clip(u.y, .2, self.cfg.height-.2))
        return step

    def _resolve_collisions(self) -> None:
        units=[u for u in self.units if u.alive and not u.airborne]
        if not units:return
        cell=1.8; grid={}
        for u in units:grid.setdefault((int(u.x//cell),int(u.y//cell)),[]).append(u)
        seen=set()
        for a in units:
            cx,cy=int(a.x//cell),int(a.y//cell)
            for gx in range(cx-1,cx+2):
                for gy in range(cy-1,cy+2):
                    for b in grid.get((gx,gy),()):
                        if b.uid==a.uid:continue
                        key=(min(a.uid,b.uid),max(a.uid,b.uid))
                        if key in seen:continue
                        seen.add(key)
                        dx,dy=b.x-a.x,b.y-a.y; d=math.hypot(dx,dy); min_d=(a.radius+b.radius)*.78
                        if d>=min_d or min_d<=0:continue
                        if d<1e-6:dx,dy,d=1.0,0.0,1.0
                        overlap=min_d-d;nx,ny=dx/d,dy/d
                        if a.is_building and b.is_building:continue
                        if a.is_building:b.x+=nx*overlap;b.y+=ny*overlap
                        elif b.is_building:a.x-=nx*overlap;a.y-=ny*overlap
                        else:a.x-=nx*overlap*.5;a.y-=ny*overlap*.5;b.x+=nx*overlap*.5;b.y+=ny*overlap*.5
                        a.x=max(.2,min(self.cfg.width-.2,a.x));a.y=max(.2,min(self.cfg.height-.2,a.y));b.x=max(.2,min(self.cfg.width-.2,b.x));b.y=max(.2,min(self.cfg.height-.2,b.y))

    def _attack(self, team: int, x: float, y: float, target: Unit | Tower, damage: float,
                projectile_speed: float, splash_radius: float, *, visual: str = "shot",
                stun_seconds: float = 0.0, slow_factor: float = 1.0, slow_seconds: float = 0.0) -> None:
        if projectile_speed > 0:
            self.projectiles.append(Projectile(
                pid=self.next_pid, team=team, x=x, y=y,
                target_is_tower=isinstance(target, Tower), target_id=target.tid if isinstance(target, Tower) else target.uid,
                target_x=target.x, target_y=target.y, damage=damage, speed=projectile_speed,
                splash_radius=splash_radius, stun_seconds=stun_seconds, slow_factor=slow_factor,
                slow_seconds=slow_seconds, visual=visual,
            ))
            self.next_pid += 1
        else:
            self._resolve_hit(team, target.x, target.y, target, damage, splash_radius,
                              stun_seconds, slow_factor, slow_seconds)

    def _tick_projectiles(self, dt: float) -> None:
        keep: list[Projectile] = []
        for p in self.projectiles:
            p.ttl -= dt
            target = self._find_target(p.target_is_tower, p.target_id)
            if target is not None and target.alive:
                p.target_x, p.target_y = target.x, target.y
            dx, dy = p.target_x-p.x, p.target_y-p.y
            d = math.hypot(dx, dy)
            step = p.speed * dt
            if d <= max(.18, step):
                if target is not None and target.alive:
                    self._resolve_hit(p.team, p.target_x, p.target_y, target, p.damage, p.splash_radius,
                                      p.stun_seconds, p.slow_factor, p.slow_seconds)
                else:
                    self._splash_damage(p.team, p.target_x, p.target_y, p.damage, p.splash_radius,
                                        p.stun_seconds, p.slow_factor, p.slow_seconds)
                self.effects.append(Effect("impact", p.target_x, p.target_y, max(.35, p.splash_radius), .20, .20, p.team))
                continue
            if d > 1e-8:
                p.x += dx/d * min(step, d)
                p.y += dy/d * min(step, d)
            if p.ttl > 0:
                keep.append(p)
        self.projectiles = keep

    def _find_target(self, is_tower: bool, target_id: int) -> Unit | Tower | None:
        seq: list[Any] = self.towers if is_tower else self.units
        key = "tid" if is_tower else "uid"
        return next((x for x in seq if getattr(x, key) == target_id), None)

    def _resolve_hit(self, team: int, x: float, y: float, target: Unit | Tower, damage: float,
                     splash_radius: float, stun: float = 0.0, slow_factor: float = 1.0, slow_seconds: float = 0.0) -> None:
        if splash_radius > 0:
            self._splash_damage(team, x, y, damage, splash_radius, stun, slow_factor, slow_seconds)
            return
        if isinstance(target, Tower):
            self._damage_tower(target, damage)
        else:
            self._damage_unit(target, damage)
            self._apply_status(target, stun, slow_factor, slow_seconds)

    def _splash_damage(self, team: int, x: float, y: float, damage: float, radius: float,
                       stun: float = 0.0, slow_factor: float = 1.0, slow_seconds: float = 0.0) -> None:
        if radius <= 0:
            return
        for u in list(self.units):
            if u.team != team and u.alive and _dist_xy(u.x, u.y, x, y) <= radius + u.radius:
                self._damage_unit(u, damage)
                self._apply_status(u, stun, slow_factor, slow_seconds)
        for t in self.towers:
            if t.team != team and t.alive and _dist_xy(t.x, t.y, x, y) <= radius + t.radius:
                self._damage_tower(t, damage)

    def _damage_unit(self, u: Unit, damage: float) -> None:
        if damage <= 0 or not u.alive:
            return
        if u.shield_hp > 0:
            absorbed = min(u.shield_hp, damage)
            u.shield_hp -= absorbed
            damage -= absorbed
            if u.shield_hp <= 1e-9:
                self.effects.append(Effect("shield_break", u.x, u.y, 1.0, .30, .30, u.team))
        if damage > 0:
            u.hp -= damage

    def _damage_tower(self, t: Tower, damage: float) -> None:
        if damage <= 0 or not t.alive:
            return
        t.hp -= damage
        if t.kind == "king":
            t.active = True

    def _knockback_unit(self, u: Unit, from_x: float, from_y: float, distance: float) -> None:
        if distance <= 0 or u.is_building or not u.alive:
            return
        dx, dy = u.x-from_x, u.y-from_y
        d = math.hypot(dx, dy)
        if d < 1e-6:
            dx, dy, d = 0.0, (1.0 if u.team == 0 else -1.0), 1.0
        u.x = float(np.clip(u.x + dx/d*distance, .25, self.cfg.width-.25))
        u.y = float(np.clip(u.y + dy/d*distance, .25, self.cfg.height-.25))
        u.charge_progress = 0.0
        self.effects.append(Effect("knockback", u.x, u.y, max(.35, u.radius), .20, .20, u.team))

    @staticmethod
    def _apply_status(u: Unit, stun: float, slow_factor: float, slow_seconds: float) -> None:
        if stun > 0:
            u.stun_remaining = max(u.stun_remaining, stun)
            u.cooldown = max(u.cooldown, stun)
            u.charge_progress = 0.0
        if slow_seconds > 0 and slow_factor < 1:
            u.slow_remaining = max(u.slow_remaining, slow_seconds)
            u.slow_factor = min(u.slow_factor, slow_factor)

    def _cleanup_dead(self) -> None:
        # Death effects can themselves kill other entities (Balloon/BombTower).
        # Process to a fixed point before removing corpses so chain reactions are
        # deterministic and no death trigger is skipped because of list order.
        while True:
            newly_dead = [u for u in self.units if not u.alive and not u.death_processed]
            if not newly_dead:
                break
            for u in newly_dead:
                u.death_processed = True
                card = BY_ID[u.card_id]
                if u.hp <= 0 and card.death_damage > 0 and card.death_radius > 0:
                    kind = "death_frost" if (card.death_stun_seconds > 0 or card.death_slow_seconds > 0) else "death_bomb"
                    self.effects.append(Effect(kind, u.x, u.y, card.death_radius, .45, .45, u.team))
                    self._splash_damage(u.team, u.x, u.y, card.death_damage, card.death_radius,
                                        card.death_stun_seconds, card.death_slow_factor, card.death_slow_seconds)
                if card.death_spawn_card and card.death_spawn_count > 0:
                    spawn = BY_NAME.get(card.death_spawn_card.lower())
                    if spawn is not None:
                        self._spawn_card_units(u.team, spawn, u.x, u.y, count_override=card.death_spawn_count)
        self.units = [u for u in self.units if u.alive]

    def _update_crowns(self) -> None:
        for team in (0, 1):
            enemy = 1-team
            enemy_towers = [t for t in self.towers if t.team == enemy]
            king = next(t for t in enemy_towers if t.kind == "king")
            if king.hp <= 0:
                self.players[team].crowns = 3
            else:
                self.players[team].crowns = sum(1 for t in enemy_towers if t.kind != "king" and t.hp <= 0)

    def _check_end(self) -> None:
        for team in (0, 1):
            king = next(t for t in self.towers if t.team == team and t.kind == "king")
            if king.hp <= 0:
                self.done, self.winner = True, 1-team
                return

        # Regulation ends if crown score is not tied. Overtime is sudden death.
        if self.time >= self.cfg.match_seconds and self.players[0].crowns != self.players[1].crowns:
            self.done = True
            self.winner = 0 if self.players[0].crowns > self.players[1].crowns else 1
            return

        if self.time >= self.max_time:
            self.done = True
            # Tiebreaker: compare the weakest surviving tower percentage.
            weak = []
            for team in (0, 1):
                alive = [max(0.0, t.hp) / t.max_hp for t in self.towers if t.team == team]
                weak.append(min(alive) if alive else 0.0)
            if abs(weak[0]-weak[1]) < 1e-8:
                self.winner = None
            else:
                self.winner = 0 if weak[0] > weak[1] else 1

    def total_tower_hp(self, team: int) -> float:
        return sum(max(0.0, t.hp) for t in self.towers if t.team == team)

    def _log(self, text: str) -> None:
        self.combat_log.append(text)
        if len(self.combat_log) > 10:
            del self.combat_log[:-10]

    def summary(self) -> dict:
        return {
            "time": round(self.time, 2),
            "phase": self.phase,
            "winner": self.winner,
            "crowns": tuple(p.crowns for p in self.players),
            "elixir": tuple(round(p.elixir, 2) for p in self.players),
            "tower_hp": tuple(round(self.total_tower_hp(t), 1) for t in (0, 1)),
            "units": len(self.units),
            "projectiles": len(self.projectiles),
        }
