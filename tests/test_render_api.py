import unittest

from clashrl.env import ClashRoyaleEnv
from clashrl.bots import RandomLegalBot
from clashrl.cards import CARDS
import clashrl.visualize as visualize
import clashrl.dashboard as dashboard


class FakeRect:
    def __init__(self, *args):
        if len(args) == 1:
            a = args[0]
            if isinstance(a, FakeRect):
                x, y, w, h = a.x, a.y, a.w, a.h
            else:
                x, y, w, h = a
        else:
            x, y, w, h = args
        self.x, self.y, self.w, self.h = int(x), int(y), int(w), int(h)

    @property
    def left(self): return self.x
    @left.setter
    def left(self, v): self.x = int(v)
    @property
    def right(self): return self.x + self.w
    @right.setter
    def right(self, v): self.x = int(v) - self.w
    @property
    def top(self): return self.y
    @top.setter
    def top(self, v): self.y = int(v)
    @property
    def bottom(self): return self.y + self.h
    @bottom.setter
    def bottom(self, v): self.y = int(v) - self.h
    @property
    def width(self): return self.w
    @property
    def height(self): return self.h
    @property
    def centerx(self): return self.x + self.w // 2
    @centerx.setter
    def centerx(self, v): self.x = int(v) - self.w // 2
    @property
    def centery(self): return self.y + self.h // 2
    @centery.setter
    def centery(self, v): self.y = int(v) - self.h // 2
    @property
    def center(self): return (self.centerx, self.centery)
    @center.setter
    def center(self, v): self.centerx, self.centery = v
    @property
    def topleft(self): return (self.left, self.top)
    @topleft.setter
    def topleft(self, v): self.left, self.top = v
    @property
    def topright(self): return (self.right, self.top)
    @topright.setter
    def topright(self, v): self.right, self.top = v
    @property
    def midtop(self): return (self.centerx, self.top)
    @midtop.setter
    def midtop(self, v): self.centerx, self.top = v
    @property
    def midbottom(self): return (self.centerx, self.bottom)
    @midbottom.setter
    def midbottom(self, v): self.centerx, self.bottom = v

    def inflate(self, dx, dy):
        return FakeRect(self.x - int(dx)/2, self.y - int(dy)/2, self.w + int(dx), self.h + int(dy))

    def collidepoint(self, p):
        x, y = p
        return self.left <= x <= self.right and self.top <= y <= self.bottom


class FakeSurface:
    def __init__(self, size=(10, 10), flags=0):
        self.size = tuple(int(x) for x in size)
    def fill(self, *args, **kwargs): return None
    def blit(self, *args, **kwargs): return None
    def get_rect(self): return FakeRect(0, 0, *self.size)


class FakeFont:
    def render(self, text, antialias, color):
        return FakeSurface((max(1, len(str(text))) * 8, 18))


class FakeFontModule:
    def SysFont(self, *args, **kwargs): return FakeFont()


class FakeDraw:
    def __getattr__(self, name):
        def call(*args, **kwargs): return None
        return call


class FakeDisplay:
    def set_caption(self, *args, **kwargs): return None
    def set_mode(self, size, *args, **kwargs): return FakeSurface(size)
    def flip(self): return None


class FakeClock:
    def tick(self, fps=0): return 16


class FakeTime:
    def __init__(self): self._ticks = 0
    def Clock(self): return FakeClock()
    def get_ticks(self):
        self._ticks += 16
        return self._ticks


class FakeEvent:
    def get(self): return []


class FakeVersion:
    ver = "fake"


class FakePygame:
    SRCALPHA = 1
    QUIT = 10
    KEYDOWN = 11
    MOUSEBUTTONDOWN = 12
    K_ESCAPE = 27
    K_SPACE = 32
    K_EQUALS = 61
    K_PLUS = 43
    K_KP_PLUS = 1001
    K_MINUS = 45
    K_KP_MINUS = 1002
    K_r = ord('r')
    K_n = ord('n')
    K_1 = ord('1')
    K_4 = ord('4')

    def __init__(self):
        self.draw = FakeDraw()
        self.display = FakeDisplay()
        self.time = FakeTime()
        self.event = FakeEvent()
        self.font = FakeFontModule()
        self.version = FakeVersion()
    def init(self): return (1, 0)
    def quit(self): return None
    def Rect(self, *args): return FakeRect(*args)
    def Surface(self, size, flags=0): return FakeSurface(size, flags)
    def get_sdl_version(self): return (2, 30, 0)


class RenderAPISmokeTests(unittest.TestCase):
    def setUp(self):
        self.fake = FakePygame()
        self.old_vpg = visualize._pg
        self.old_dpg = dashboard._pg
        visualize._pg = lambda: self.fake
        dashboard._pg = lambda: self.fake

    def tearDown(self):
        visualize._pg = self.old_vpg
        dashboard._pg = self.old_dpg

    def test_arena_render_path_and_all_card_icons(self):
        env = ClashRoyaleEnv(seed=90)
        obs = env.reset(90)
        b0, b1 = RandomLegalBot(seed=1), RandomLegalBot(seed=2)
        # Advance enough to exercise non-empty arena state/effects/projectiles when random play creates them.
        for _ in range(60):
            a0 = b0.act(obs[0], env.action_mask(0))[0]
            a1 = b1.act(obs[1], env.action_mask(1))[0]
            step = env.step_joint((a0, a1))
            obs = step.observations
            if env.game.done:
                break
        viewer = visualize.ArenaViewer(env, b0, b1, headless=True)
        viewer.show_ranges = True
        viewer._draw()
        # Force every procedural visual archetype/card icon branch at least once.
        for card in CARDS:
            if card.kind == "spell" or card.kind == "building":
                viewer._mini_card_icon(card.visual, 0, 100, 100)
            else:
                viewer._draw_unit_icon(card.visual, 0, 100, 100, 18, card.airborne)


    def test_draft_scene_render_and_transfer_animation(self):
        b0, b1 = RandomLegalBot(seed=3), RandomLegalBot(seed=4)
        dv = visualize.DraftViewer(b0, b1, seed=12, headless=True, speed=4.0)
        dv._ai_action()
        self.assertIsNotNone(dv.selection)
        dv._draw()
        dv.reveal_left = max(0.01, dv.reveal_total * 0.5)
        dv._draw()

    def test_dashboard_render_path(self):
        d = dashboard.TrainingDashboard("/tmp/does-not-need-to-exist.csv")
        d.rows = [
            {"update":"1","steps":"2048","episodes":"4","wins":"2","losses":"1","draws":"1",
             "win_rate":"0.625","mean_terminal_reward":"0.25","policy_loss":"-0.012",
             "value_loss":"0.21","entropy":"0.18","approx_kl":"0.004","seconds":"2.0"},
            {"update":"2","steps":"4096","episodes":"5","wins":"3","losses":"1","draws":"1",
             "win_rate":"0.7","mean_terminal_reward":"0.4","policy_loss":"-0.008",
             "value_loss":"0.17","entropy":"0.16","approx_kl":"0.006","seconds":"2.1"},
        ]
        d._draw()


if __name__ == "__main__":
    unittest.main()
