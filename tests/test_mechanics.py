import unittest

from clashrl.cards import BY_NAME
from clashrl.core import GameState


class MechanicsTests(unittest.TestCase):
    def test_shield_absorbs_damage_first(self):
        g = GameState(seed=1)
        c = BY_NAME['guards']
        u = g._spawn_single(0, c, 8, 22)
        hp = u.hp
        g._damage_unit(u, 200)
        self.assertEqual(u.hp, hp)
        self.assertAlmostEqual(u.shield_hp, c.shield_hp - 200)

    def test_ground_routes_to_bridge_air_does_not(self):
        g = GameState(seed=2)
        knight = g._spawn_single(0, BY_NAME['knight'], 2, 18.5)
        minion = g._spawn_single(0, BY_NAME['minions'], 2, 18.5)
        gx, gy = g._movement_target(knight, 9, 13.5)
        ax, ay = g._movement_target(minion, 9, 13.5)
        self.assertIn(gx, g.cfg.bridge_x)
        self.assertAlmostEqual(ax, 9)
        self.assertAlmostEqual(ay, 13.5)

    def test_ranged_attack_creates_projectile(self):
        g = GameState(seed=3)
        a = g._spawn_single(0, BY_NAME['musketeer'], 8, 20)
        b = g._spawn_single(1, BY_NAME['knight'], 8, 16)
        g._attack(0, a.x, a.y, b, a.damage, a.projectile_speed, a.splash_radius, visual='musketeer')
        self.assertEqual(len(g.projectiles), 1)

    def test_zap_stuns(self):
        g = GameState(seed=4)
        u = g._spawn_single(1, BY_NAME['knight'], 9, 12)
        g._cast_spell(0, BY_NAME['zap'], 9, 12)
        self.assertGreater(u.stun_remaining, 0)
        self.assertLess(u.hp, u.max_hp)

    def test_building_expires(self):
        g = GameState(seed=5)
        u = g._spawn_single(0, BY_NAME['cannon'], 9, 22)
        u.lifetime = .05
        g.step_physics(.10)
        self.assertFalse(any(x.uid == u.uid for x in g.units))

    def test_king_activates_when_damaged(self):
        g = GameState(seed=6)
        king = next(t for t in g.towers if t.team == 1 and t.kind == 'king')
        self.assertFalse(king.active)
        g._damage_tower(king, 1)
        self.assertTrue(king.active)

    def test_charge_flag(self):
        g = GameState(seed=7)
        u = g._spawn_single(0, BY_NAME['prince'], 5, 24)
        self.assertFalse(u.charged)
        u.charge_progress = u.charge_distance
        self.assertTrue(u.charged)
        self.assertGreater(u.current_speed, u.speed)

    def test_chain_death_effects_are_processed(self):
        g = GameState(seed=8)
        a = g._spawn_single(0, BY_NAME['balloon'], 9, 16)
        b = g._spawn_single(1, BY_NAME['balloon'], 9.2, 16)
        a.deploy_remaining = 0; b.deploy_remaining = 0
        b.hp = 100
        a.hp = 0
        g._cleanup_dead()
        bombs = [e for e in g.effects if e.kind == 'death_bomb']
        self.assertGreaterEqual(len(bombs), 2)
        self.assertFalse(any(u.uid in (a.uid, b.uid) for u in g.units))

    def test_fireball_knockback(self):
        g = GameState(seed=9)
        u = g._spawn_single(1, BY_NAME['knight'], 10, 12)
        old_x = u.x
        g._cast_spell(0, BY_NAME['fireball'], 9, 12)
        self.assertGreater(u.x, old_x)

    def test_electro_wizard_deploy_pulse_stuns(self):
        g = GameState(seed=10)
        enemy = g._spawn_single(1, BY_NAME['knight'], 9.4, 12)
        hp = enemy.hp
        g._spawn_card_units(0, BY_NAME['electrowizard'], 9, 12)
        self.assertLess(enemy.hp, hp)
        self.assertGreater(enemy.stun_remaining, 0)

    def test_ice_spirit_suicides_and_applies_death_status(self):
        g = GameState(seed=11)
        spirit = g._spawn_single(0, BY_NAME['icespirit'], 9, 17)
        target = g._spawn_single(1, BY_NAME['knight'], 9, 16.2)
        spirit.deploy_remaining = 0.0
        target.deploy_remaining = 0.0
        # Put it inside melee range and tick until the suicide attack resolves.
        spirit.x, spirit.y = 9, 16.7
        g.step_physics(.12)
        self.assertFalse(any(u.uid == spirit.uid for u in g.units))
        self.assertTrue(target.stun_remaining > 0 or target.slow_remaining > 0)


if __name__ == '__main__':
    unittest.main()
