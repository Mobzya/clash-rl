from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


CardKind = Literal["troop", "spell", "building"]
TargetKind = Literal["ground", "air_ground", "buildings"]


@dataclass(frozen=True)
class Card:
    id: int
    name: str
    cost: int
    kind: CardKind

    hp: float = 0.0
    damage: float = 0.0
    speed: float = 0.0
    attack_range: float = 0.0
    attack_interval: float = 1.0
    count: int = 1
    radius: float = 0.42
    target_kind: TargetKind = "ground"
    airborne: bool = False
    projectile_speed: float = 0.0
    splash_radius: float = 0.0

    shield_hp: float = 0.0
    charge_distance: float = 0.0
    charge_multiplier: float = 1.0
    charge_speed_multiplier: float = 1.0

    lifetime: float = 0.0
    spawn_interval: float = 0.0
    spawn_card: str | None = None
    spawn_count: int = 0

    spell_radius: float = 0.0
    tower_damage_scale: float = 1.0
    stun_seconds: float = 0.0
    slow_factor: float = 1.0
    slow_seconds: float = 0.0
    knockback_distance: float = 0.0

    # Optional status applied by normal troop/building attacks.
    hit_stun_seconds: float = 0.0
    hit_slow_factor: float = 1.0
    hit_slow_seconds: float = 0.0
    suicide_on_attack: bool = False

    # Optional one-shot deploy pulse (used by e.g. Electro Wizard-like units).
    deploy_damage: float = 0.0
    deploy_radius: float = 0.0
    deploy_stun_seconds: float = 0.0

    death_damage: float = 0.0
    death_radius: float = 0.0
    death_stun_seconds: float = 0.0
    death_slow_factor: float = 1.0
    death_slow_seconds: float = 0.0
    death_spawn_card: str | None = None
    death_spawn_count: int = 0

    # Cosmetic archetype used by the procedural renderer.
    visual: str = "fighter"
    deploy_time: float = 0.70


# Approximate stats chosen for learning dynamics, not a claim of live card balance.
_CARD_DATA = [
    dict(name="Knight", cost=3, kind="troop", hp=1500, damage=185, speed=1.22, attack_range=.75, attack_interval=1.2, visual="knight"),
    dict(name="Archers", cost=3, kind="troop", hp=410, damage=120, speed=1.20, attack_range=5.2, attack_interval=1.0, count=2, radius=.34, target_kind="air_ground", projectile_speed=10.0, visual="archer"),
    dict(name="Giant", cost=5, kind="troop", hp=3600, damage=225, speed=.68, attack_range=.85, attack_interval=1.45, radius=.72, target_kind="buildings", visual="giant"),
    dict(name="Goblins", cost=2, kind="troop", hp=255, damage=135, speed=1.68, attack_range=.65, attack_interval=.95, count=3, radius=.27, visual="goblin"),
    dict(name="MiniPEKKA", cost=4, kind="troop", hp=1280, damage=590, speed=1.42, attack_range=.70, attack_interval=1.65, radius=.45, visual="pekka"),
    dict(name="Musketeer", cost=4, kind="troop", hp=870, damage=205, speed=1.12, attack_range=5.5, attack_interval=1.1, radius=.40, target_kind="air_ground", projectile_speed=12.0, visual="musketeer"),
    dict(name="Bomber", cost=2, kind="troop", hp=430, damage=235, speed=1.02, attack_range=4.6, attack_interval=1.8, radius=.37, projectile_speed=7.5, splash_radius=1.25, visual="bomber"),
    dict(name="Valkyrie", cost=4, kind="troop", hp=1950, damage=250, speed=1.02, attack_range=.80, attack_interval=1.55, radius=.55, splash_radius=1.25, visual="valkyrie"),
    dict(name="HogRider", cost=4, kind="troop", hp=1750, damage=305, speed=1.82, attack_range=.78, attack_interval=1.45, radius=.50, target_kind="buildings", visual="hog"),
    dict(name="SpearGoblins", cost=2, kind="troop", hp=185, damage=88, speed=1.52, attack_range=4.9, attack_interval=1.25, count=3, radius=.27, target_kind="air_ground", projectile_speed=9.0, visual="spear"),
    dict(name="Skeletons", cost=1, kind="troop", hp=85, damage=82, speed=1.55, attack_range=.55, attack_interval=1.0, count=3, radius=.23, visual="skeleton"),
    dict(name="Minions", cost=3, kind="troop", hp=250, damage=115, speed=1.65, attack_range=1.6, attack_interval=1.0, count=3, radius=.30, target_kind="air_ground", airborne=True, visual="minion"),
    dict(name="MegaMinion", cost=3, kind="troop", hp=750, damage=285, speed=1.25, attack_range=1.5, attack_interval=1.55, radius=.42, target_kind="air_ground", airborne=True, visual="megaminion"),
    dict(name="BabyDragon", cost=4, kind="troop", hp=1550, damage=165, speed=1.20, attack_range=3.5, attack_interval=1.6, radius=.55, target_kind="air_ground", airborne=True, projectile_speed=7.5, splash_radius=1.45, visual="dragon"),
    dict(name="Balloon", cost=5, kind="troop", hp=1650, damage=690, speed=.95, attack_range=.75, attack_interval=2.0, radius=.62, target_kind="buildings", airborne=True, death_damage=240, death_radius=1.4, visual="balloon"),
    dict(name="Prince", cost=5, kind="troop", hp=1750, damage=340, speed=1.35, attack_range=1.15, attack_interval=1.4, radius=.50, charge_distance=3.3, charge_multiplier=2.0, charge_speed_multiplier=1.35, visual="prince"),
    dict(name="DarkPrince", cost=4, kind="troop", hp=1250, shield_hp=260, damage=245, speed=1.30, attack_range=1.05, attack_interval=1.3, radius=.50, charge_distance=3.1, charge_multiplier=1.8, charge_speed_multiplier=1.30, splash_radius=1.0, visual="darkprince"),
    dict(name="Guards", cost=3, kind="troop", hp=110, shield_hp=240, damage=110, speed=1.30, attack_range=1.0, attack_interval=1.1, count=3, radius=.29, visual="guard"),
    dict(name="Wizard", cost=5, kind="troop", hp=720, damage=250, speed=1.05, attack_range=5.3, attack_interval=1.4, radius=.40, target_kind="air_ground", projectile_speed=8.5, splash_radius=1.45, visual="wizard"),
    dict(name="Witch", cost=5, kind="troop", hp=850, damage=145, speed=1.02, attack_range=5.2, attack_interval=1.1, radius=.43, target_kind="air_ground", projectile_speed=8.5, splash_radius=.75, spawn_interval=6.8, spawn_card="Skeletons", spawn_count=2, visual="witch"),


    dict(name="Barbarians", cost=5, kind="troop", hp=670, damage=190, speed=1.00, attack_range=.70, attack_interval=1.25, count=5, radius=.34, visual="barbarian"),
    dict(name="Bats", cost=2, kind="troop", hp=95, damage=82, speed=1.90, attack_range=1.25, attack_interval=1.0, count=5, radius=.22, target_kind="air_ground", airborne=True, visual="bat"),
    dict(name="RoyalGiant", cost=6, kind="troop", hp=3000, damage=300, speed=.72, attack_range=5.0, attack_interval=1.7, radius=.70, target_kind="buildings", projectile_speed=11.0, visual="royalgiant"),
    dict(name="DartGoblin", cost=3, kind="troop", hp=320, damage=115, speed=1.45, attack_range=6.3, attack_interval=.75, radius=.27, target_kind="air_ground", projectile_speed=14.0, visual="dartgoblin"),
    dict(name="FlyingMachine", cost=4, kind="troop", hp=610, damage=145, speed=1.25, attack_range=6.0, attack_interval=1.0, radius=.36, target_kind="air_ground", airborne=True, projectile_speed=12.0, visual="flyingmachine"),
    dict(name="SkeletonArmy", cost=3, kind="troop", hp=85, damage=82, speed=1.55, attack_range=.55, attack_interval=1.0, count=8, radius=.21, visual="skeleton"),
    dict(name="MinionHorde", cost=5, kind="troop", hp=250, damage=115, speed=1.65, attack_range=1.6, attack_interval=1.0, count=6, radius=.27, target_kind="air_ground", airborne=True, visual="minion"),
    dict(name="Bowler", cost=5, kind="troop", hp=2100, damage=260, speed=.90, attack_range=4.7, attack_interval=2.0, radius=.55, projectile_speed=7.0, splash_radius=1.6, visual="bowler"),
    dict(name="Mortar", cost=4, kind="building", hp=1200, damage=310, attack_range=8.0, attack_interval=4.0, lifetime=30.0, target_kind="ground", projectile_speed=5.5, splash_radius=1.65, radius=.75, visual="mortar"),
    dict(name="GoblinHut", cost=5, kind="building", hp=1450, damage=0, attack_range=0, attack_interval=1.0, lifetime=32.0, spawn_interval=5.0, spawn_card="SpearGoblins", spawn_count=1, death_spawn_card="SpearGoblins", death_spawn_count=2, radius=.72, visual="goblinhut"),

    # v3.1 mechanics cards. They deliberately exercise death effects, suicide attacks,
    # deploy pulses and on-hit status effects rather than only adding more stat clones.
    dict(name="IceGolem", cost=2, kind="troop", hp=1180, damage=85, speed=.92, attack_range=.65, attack_interval=1.55, radius=.50,
         death_damage=115, death_radius=1.55, death_slow_factor=.62, death_slow_seconds=2.4, visual="icegolem"),
    dict(name="IceSpirit", cost=1, kind="troop", hp=190, damage=0, speed=2.05, attack_range=.58, attack_interval=1.0, radius=.25, target_kind="air_ground",
         suicide_on_attack=True, death_damage=135, death_radius=1.45, death_stun_seconds=.45, death_slow_factor=.60, death_slow_seconds=1.8, visual="icespirit"),
    dict(name="FireSpirit", cost=1, kind="troop", hp=185, damage=0, speed=2.05, attack_range=.58, attack_interval=1.0, radius=.25, target_kind="air_ground",
         suicide_on_attack=True, death_damage=285, death_radius=1.35, visual="firespirit"),
    dict(name="WallBreakers", cost=2, kind="troop", hp=300, damage=0, speed=1.95, attack_range=.58, attack_interval=1.0, count=2, radius=.27, target_kind="buildings",
         suicide_on_attack=True, death_damage=430, death_radius=1.05, visual="wallbreaker"),
    dict(name="ElectroWizard", cost=4, kind="troop", hp=800, damage=175, speed=1.05, attack_range=5.0, attack_interval=1.55, radius=.40, target_kind="air_ground",
         projectile_speed=11.0, hit_stun_seconds=.30, deploy_damage=190, deploy_radius=2.25, deploy_stun_seconds=.45, visual="electrowizard"),
    dict(name="IceWizard", cost=3, kind="troop", hp=720, damage=105, speed=1.05, attack_range=5.3, attack_interval=1.65, radius=.40, target_kind="air_ground",
         projectile_speed=9.0, hit_slow_factor=.65, hit_slow_seconds=2.0, splash_radius=.65, visual="icewizard"),

    dict(name="Cannon", cost=3, kind="building", hp=900, damage=170, attack_range=5.5, attack_interval=.95, lifetime=30.0, target_kind="ground", projectile_speed=11.0, radius=.65, visual="cannon"),
    dict(name="BombTower", cost=4, kind="building", hp=1250, damage=190, attack_range=5.5, attack_interval=1.6, lifetime=30.0, target_kind="ground", projectile_speed=6.5, splash_radius=1.4, death_damage=250, death_radius=1.5, radius=.72, visual="bombtower"),
    dict(name="Tombstone", cost=3, kind="building", hp=650, damage=0, attack_range=0, attack_interval=1.0, lifetime=30.0, spawn_interval=4.5, spawn_card="Skeletons", spawn_count=1, death_spawn_card="Skeletons", death_spawn_count=3, radius=.62, visual="tombstone"),

    dict(name="Fireball", cost=4, kind="spell", damage=650, spell_radius=2.6, tower_damage_scale=.35, knockback_distance=.65, visual="fireball"),
    dict(name="Arrows", cost=3, kind="spell", damage=335, spell_radius=4.0, tower_damage_scale=.35, visual="arrows"),
    dict(name="Zap", cost=2, kind="spell", damage=190, spell_radius=2.5, tower_damage_scale=.35, stun_seconds=.50, visual="zap"),
    dict(name="GiantSnowball", cost=2, kind="spell", damage=180, spell_radius=2.7, tower_damage_scale=.35, slow_factor=.60, slow_seconds=2.5, knockback_distance=1.05, visual="snowball"),
    dict(name="Rocket", cost=6, kind="spell", damage=1250, spell_radius=2.0, tower_damage_scale=.35, visual="rocket"),
]

CARDS: tuple[Card, ...] = tuple(Card(id=i, **d) for i, d in enumerate(_CARD_DATA))
BY_ID = {c.id: c for c in CARDS}
BY_NAME = {c.name.lower(): c for c in CARDS}

DEFAULT_DECK = tuple(BY_NAME[n.lower()].id for n in (
    "Knight", "Archers", "Giant", "Goblins", "MiniPEKKA", "Musketeer", "Fireball", "Arrows"
))
FAST_DECK = tuple(BY_NAME[n.lower()].id for n in (
    "Knight", "Skeletons", "Goblins", "Musketeer", "HogRider", "Cannon", "Fireball", "Zap"
))
AIR_DECK = tuple(BY_NAME[n.lower()].id for n in (
    "BabyDragon", "Balloon", "Minions", "MegaMinion", "Knight", "Tombstone", "Fireball", "Arrows"
))
CONTROL_DECK = tuple(BY_NAME[n.lower()].id for n in (
    "Valkyrie", "Wizard", "Guards", "DarkPrince", "BombTower", "Musketeer", "Zap", "Rocket"
))
CHARGE_DECK = tuple(BY_NAME[n.lower()].id for n in (
    "Prince", "DarkPrince", "Guards", "SpearGoblins", "Witch", "Cannon", "Fireball", "GiantSnowball"
))

SWARM_DECK = tuple(BY_NAME[n.lower()].id for n in (
    "SkeletonArmy", "Bats", "MinionHorde", "GoblinHut", "DartGoblin", "Valkyrie", "Fireball", "Zap"
))
SIEGE_DECK = tuple(BY_NAME[n.lower()].id for n in (
    "Mortar", "Knight", "Archers", "Skeletons", "DartGoblin", "Cannon", "Fireball", "Arrows"
))
HEAVY_DECK = tuple(BY_NAME[n.lower()].id for n in (
    "RoyalGiant", "Barbarians", "FlyingMachine", "Bowler", "MegaMinion", "GoblinHut", "Fireball", "GiantSnowball"
))
STATUS_DECK = tuple(BY_NAME[n.lower()].id for n in (
    "IceGolem", "IceSpirit", "FireSpirit", "WallBreakers", "ElectroWizard", "IceWizard", "Cannon", "Fireball"
))

DECK_POOL = (DEFAULT_DECK, FAST_DECK, AIR_DECK, CONTROL_DECK, CHARGE_DECK, SWARM_DECK, SIEGE_DECK, HEAVY_DECK, STATUS_DECK)
CARD_POOL = tuple(c.id for c in CARDS)


def card_name(card_id: int) -> str:
    return BY_ID[int(card_id)].name
