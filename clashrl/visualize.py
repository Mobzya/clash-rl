from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
import torch

from .cards import BY_ID, card_name
from .draft import DraftState, DRAFT_ACTIONS, DRAFT_ACTION_DIM
from .config import CFG
from .env import ClashRoyaleEnv
from .model import ActorCritic


WINDOW_W = 1280
WINDOW_H = 900
ARENA_X = 70
ARENA_Y = 32
ARENA_W = 720
ARENA_H = 836
PANEL_X = 820
PANEL_W = 430

BLUE = (63, 152, 255)
RED = (239, 84, 91)
GOLD = (247, 196, 68)
INK = (22, 27, 35)
PANEL = (30, 36, 47)
PANEL_2 = (39, 46, 59)
TEXT = (238, 242, 247)
MUTED = (160, 170, 184)
GREEN = (87, 190, 103)


def _pg():
    try:
        import pygame
    except ImportError as exc:
        raise RuntimeError(
            "Visualization requires pygame-ce. Install/upgrade the project with: pip install -e ."
        ) from exc
    return pygame


def _clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


@dataclass
class DecisionInfo:
    action: int = 0
    value: float = 0.0
    label: str = "WAIT"
    top: tuple[tuple[str, float], ...] = ()


class ArenaViewer:
    def __init__(self, env: ClashRoyaleEnv, policy0=None, policy1=None, *, human_team: int | None = None,
                 speed: float = 2.0, title: str = "Clash RL v3.1", device="cpu", auto_close: bool = False,
                 headless: bool = False):
        self.pg = _pg()
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        self.pg.init()
        self.pg.display.set_caption(title)
        self.screen = self.pg.display.set_mode((WINDOW_W, WINDOW_H))
        self.clock = self.pg.time.Clock()
        self.env = env
        self.policies = [policy0, policy1]
        self.human_team = human_team
        self.speed = max(.25, float(speed))
        self.device = device
        self.auto_close = auto_close
        self.selected_slot: int | None = None
        self.card_rects: dict[int, object] = {}
        self.paused = False
        self.show_ranges = False
        self.show_names = True
        self.running = True
        self.next_decision = [0.0, 0.0]
        self.decisions = [DecisionInfo(), DecisionInfo()]
        self.end_wall_time: float | None = None
        self.font_cache: dict[tuple[int, bool], object] = {}
        self._grass_seed = 1337

    def font(self, size: int, bold: bool = False):
        key = (size, bold)
        if key not in self.font_cache:
            self.font_cache[key] = self.pg.font.SysFont("DejaVu Sans", size, bold=bold)
        return self.font_cache[key]

    def text(self, s, x, y, size=18, color=TEXT, bold=False, anchor="topleft"):
        surf = self.font(size, bold).render(str(s), True, color)
        rect = surf.get_rect()
        setattr(rect, anchor, (int(x), int(y)))
        self.screen.blit(surf, rect)
        return rect

    def world_to_screen(self, x: float, y: float) -> tuple[int, int]:
        sx = ARENA_X + int(x / CFG.width * ARENA_W)
        sy = ARENA_Y + int(y / CFG.height * ARENA_H)
        return sx, sy

    def screen_to_world(self, sx: float, sy: float) -> tuple[float, float]:
        x = (sx-ARENA_X)/ARENA_W * CFG.width
        y = (sy-ARENA_Y)/ARENA_H * CFG.height
        return float(x), float(y)

    def run(self, max_frames: int | None = None):
        frames = 0
        while self.running:
            real_dt = self.clock.tick(60)/1000.0
            self._events()
            if not self.paused and not self.env.game.done:
                self._simulate(real_dt)
            self._draw()
            self.pg.display.flip()
            frames += 1
            if max_frames is not None and frames >= max_frames:
                break
            if self.env.game.done and self.auto_close:
                if self.end_wall_time is None:
                    self.end_wall_time = self.pg.time.get_ticks()/1000.0
                elif self.pg.time.get_ticks()/1000.0-self.end_wall_time > 1.2:
                    break
        self.pg.quit()

    def _events(self):
        for e in self.pg.event.get():
            if e.type == self.pg.QUIT:
                self.running = False
            elif e.type == self.pg.KEYDOWN:
                if e.key == self.pg.K_ESCAPE:
                    self.running = False
                elif e.key == self.pg.K_SPACE:
                    self.paused = not self.paused
                elif e.key in (self.pg.K_EQUALS, self.pg.K_PLUS, self.pg.K_KP_PLUS):
                    self.speed = min(32.0, self.speed*1.5)
                elif e.key in (self.pg.K_MINUS, self.pg.K_KP_MINUS):
                    self.speed = max(.25, self.speed/1.5)
                elif e.key == self.pg.K_r:
                    self.show_ranges = not self.show_ranges
                elif e.key == self.pg.K_n:
                    self.show_names = not self.show_names
                elif self.human_team is not None and self.pg.K_1 <= e.key <= self.pg.K_4:
                    self.selected_slot = e.key-self.pg.K_1
            elif e.type == self.pg.MOUSEBUTTONDOWN and e.button == 1:
                self._click(e.pos)

    def _simulate(self, real_dt: float):
        # Cap wall-clock catch-up so dragging the window cannot explode physics.
        sim_budget = min(.30, real_dt*self.speed)
        target_time = self.env.game.time + sim_budget
        while self.env.game.time + 1e-9 < target_time and not self.env.game.done:
            for team in (0, 1):
                if team == self.human_team:
                    continue
                if self.policies[team] is not None and self.env.game.time + 1e-9 >= self.next_decision[team]:
                    self._ai_decision(team)
                    self.next_decision[team] += CFG.decision_dt
            self.env.game.step_physics(min(CFG.physics_dt, target_time-self.env.game.time))

    def _ai_decision(self, team: int):
        policy = self.policies[team]
        obs = self.env.observe(team)
        mask = self.env.action_mask(team)
        if isinstance(policy, ActorCritic):
            action, _, value = policy.act(obs, mask, deterministic=False, device=self.device)
            top = self._top_actions(policy, obs, mask, team)
        else:
            action, _, value = policy.act(obs, mask)
            top = ()
        dec = self.env.decode_action(team, action)
        label = "WAIT"
        if dec is not None:
            slot, x, y = dec
            card = BY_ID[self.env.game.players[team].hand[slot]]
            # Card must be read before play_card cycles the slot.
            label = f"{card.name} @ {x:.1f},{y:.1f}"
            self.env.game.play_card(team, slot, x, y)
        self.decisions[team] = DecisionInfo(int(action), float(value), label, top)

    @torch.no_grad()
    def _top_actions(self, model: ActorCritic, obs, mask, team: int):
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        m = torch.as_tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        dist, _ = model.distribution(x, m)
        probs = dist.probs[0]
        k = min(3, int(mask.sum()))
        vals, idxs = torch.topk(probs, k=k)
        out = []
        for prob, idx in zip(vals.tolist(), idxs.tolist()):
            out.append((self._action_label(team, int(idx)), float(prob)))
        return tuple(out)

    def _action_label(self, team: int, action: int) -> str:
        dec = self.env.decode_action(team, action)
        if dec is None:
            return "WAIT"
        slot, x, y = dec
        cid = self.env.game.players[team].hand[slot]
        return f"{BY_ID[cid].name} ({x:.0f},{y:.0f})"

    def _click(self, pos):
        if self.human_team is None or self.env.game.done:
            return
        for slot, rect in self.card_rects.items():
            if rect.collidepoint(pos):
                self.selected_slot = slot
                return
        if self.selected_slot is None:
            return
        sx, sy = pos
        if not (ARENA_X <= sx <= ARENA_X+ARENA_W and ARENA_Y <= sy <= ARENA_Y+ARENA_H):
            return
        x, y = self.screen_to_world(sx, sy)
        p = self.env.game.players[self.human_team]
        card = BY_ID[p.hand[self.selected_slot]]
        if self.env.game.play_card(self.human_team, self.selected_slot, x, y):
            self.decisions[self.human_team] = DecisionInfo(0, 0.0, f"{card.name} @ {x:.1f},{y:.1f}", ())
            self.selected_slot = None

    def _draw(self):
        self.screen.fill(INK)
        self._draw_arena()
        self._draw_effects(back=True)
        self._draw_towers()
        self._draw_units()
        self._draw_projectiles()
        self._draw_effects(back=False)
        self._draw_panel()
        if self.env.game.done:
            self._draw_game_over()

    def _draw_arena(self):
        pg = self.pg
        arena = pg.Rect(ARENA_X, ARENA_Y, ARENA_W, ARENA_H)
        pg.draw.rect(self.screen, (68, 137, 79), arena, border_radius=18)
        # Subtle mowing stripes / tile variation.
        for i in range(16):
            y = ARENA_Y + int(i*ARENA_H/16)
            col = (72, 143, 83) if i%2 == 0 else (64, 132, 75)
            pg.draw.rect(self.screen, col, (ARENA_X+4, y, ARENA_W-8, int(ARENA_H/16)+1))
        # Decorative stone border.
        pg.draw.rect(self.screen, (88, 91, 83), arena, width=7, border_radius=18)
        pg.draw.rect(self.screen, (148, 137, 107), arena.inflate(-8,-8), width=2, border_radius=15)

        _, ry = self.world_to_screen(0, CFG.river_y)
        river_px = int(CFG.river_half_width/CFG.height*ARENA_H)
        pg.draw.rect(self.screen, (46, 122, 184), (ARENA_X+6, ry-river_px, ARENA_W-12, river_px*2))
        for k in range(7):
            yy = ry-river_px+5+k*max(3,(river_px*2-10)//7)
            pg.draw.line(self.screen, (83, 164, 217), (ARENA_X+20, yy), (ARENA_X+ARENA_W-20, yy), 2)
        # Bridges with planks.
        for bx in CFG.bridge_x:
            sx, _ = self.world_to_screen(bx, CFG.river_y)
            bw = int(CFG.bridge_half_width/CFG.width*ARENA_W*2)
            rect = pg.Rect(sx-bw//2, ry-river_px-7, bw, river_px*2+14)
            pg.draw.rect(self.screen, (154, 119, 72), rect, border_radius=5)
            pg.draw.rect(self.screen, (91, 65, 43), rect, 3, border_radius=5)
            for p in range(1,6):
                yy = rect.top + p*rect.height//6
                pg.draw.line(self.screen, (102, 75, 49), (rect.left+3,yy), (rect.right-3,yy), 2)
        # Center line and deployment hint for human.
        pg.draw.line(self.screen, (255,255,255,80), (ARENA_X+8,ry), (ARENA_X+ARENA_W-8,ry), 1)
        if self.human_team is not None and self.selected_slot is not None:
            card = BY_ID[self.env.game.players[self.human_team].hand[self.selected_slot]]
            overlay = pg.Surface((ARENA_W-12, ARENA_H//2), pg.SRCALPHA)
            overlay.fill((50,120,255,26) if card.kind != "spell" else (180,120,255,20))
            if card.kind == "spell":
                self.screen.blit(overlay, (ARENA_X+6, ARENA_Y+ARENA_H//2))
                self.screen.blit(overlay, (ARENA_X+6, ARENA_Y))
            elif self.human_team == 0:
                self.screen.blit(overlay, (ARENA_X+6, ry+river_px))
            else:
                self.screen.blit(overlay, (ARENA_X+6, ARENA_Y))

    def _draw_towers(self):
        pg = self.pg
        for t in self.env.game.towers:
            x,y = self.world_to_screen(t.x,t.y)
            team_col = BLUE if t.team == 0 else RED
            scale = 30 if t.kind == "king" else 25
            # shadow + stone base
            pg.draw.ellipse(self.screen, (26,38,32), (x-scale, y+scale*.45, scale*2, scale*.65))
            body = pg.Rect(x-scale, y-scale, scale*2, scale*2)
            pg.draw.rect(self.screen, (205,195,168), body, border_radius=6)
            pg.draw.rect(self.screen, (80,70,63), body, 3, border_radius=6)
            roof = [(x-scale-4,y-scale+5),(x,y-scale-15),(x+scale+4,y-scale+5)]
            pg.draw.polygon(self.screen, team_col, roof)
            pg.draw.polygon(self.screen, (65,59,57), roof, 3)
            if t.kind == "king":
                pg.draw.circle(self.screen, GOLD, (x,y-3), 8)
                self.text("K",x,y-3,12,(70,52,20),True,"center")
                if not t.active:
                    self.text("zzz",x+20,y-18,11,MUTED,True,"center")
            else:
                self.text("♛",x,y-2,16,TEXT,True,"center")
            self._hp_bar(x-scale, y-scale-24, scale*2, t.hp/t.max_hp, team=t.team, label=f"{max(0,int(t.hp))}")

    def _draw_units(self):
        for u in sorted(self.env.game.units, key=lambda q: (q.airborne, q.y)):
            x,y = self.world_to_screen(u.x,u.y)
            card = BY_ID[u.card_id]
            r = int(max(11, min(24, u.radius*28)))
            if u.is_building:
                self._draw_building(u,x,y,r,card)
                continue
            if u.airborne:
                self.pg.draw.ellipse(self.screen,(34,47,42),(x-r,y+r//2,r*2,r//2))
                y -= 11
            if self.show_ranges and u.attack_range > 2.0:
                rr = int(u.attack_range/CFG.width*ARENA_W)
                surf = self.pg.Surface((rr*2+4,rr*2+4),self.pg.SRCALPHA)
                self.pg.draw.circle(surf,(255,255,255,28),(rr+2,rr+2),rr,1)
                self.screen.blit(surf,(x-rr-2,y-rr-2))
            self._draw_unit_icon(card.visual, u.team, x,y,r, u.airborne)
            if u.deploy_remaining > 0:
                self.pg.draw.circle(self.screen,(245,245,245),(x,y),r+8,2)
                self.text(f"{u.deploy_remaining:.1f}",x,y,9,TEXT,True,"center")
            if u.shield_hp > 0:
                self.pg.draw.circle(self.screen,(135,203,255),(x,y),r+5,3)
            if u.stun_remaining > 0:
                for a in (0,2.1,4.2):
                    sx=x+int(math.cos(a)*r*.8); sy=y-r-7+int(math.sin(a)*4)
                    self.pg.draw.circle(self.screen,GOLD,(sx,sy),3)
            if u.slow_remaining > 0:
                self.pg.draw.circle(self.screen,(145,225,255),(x,y),r+8,2)
            if u.charged:
                self.pg.draw.circle(self.screen,GOLD,(x,y),r+9,3)
            self._hp_bar(x-r, y-r-12, r*2, u.hp/max(1,u.max_hp), team=u.team, height=5)
            if self.show_names:
                name = card.name.replace("GiantSnowball","Snowball")
                self.text(name,x,y+r+6,10,TEXT,True,"midtop")

    def _draw_unit_icon(self, visual: str, team: int, x:int, y:int, r:int, airborne=False):
        pg=self.pg
        tc = BLUE if team==0 else RED
        skin=(224,184,142); dark=(46,48,57); steel=(137,151,164); green=(95,181,77); bone=(229,226,205)
        # common body halo gives strong team readability.
        pg.draw.circle(self.screen,tc,(x,y),r)
        pg.draw.circle(self.screen,(25,28,33),(x,y),r,2)
        if visual in ("goblin","spear"):
            pg.draw.circle(self.screen,green,(x,y),int(r*.67)); pg.draw.polygon(self.screen,green,[(x-r,y),(x-r//2,y-5),(x-r//2,y+4)]); pg.draw.polygon(self.screen,green,[(x+r,y),(x+r//2,y-5),(x+r//2,y+4)])
            pg.draw.circle(self.screen,dark,(x-4,y-2),2); pg.draw.circle(self.screen,dark,(x+4,y-2),2)
            if visual=="spear": pg.draw.line(self.screen,(196,165,102),(x+r//2,y+r),(x-r//2,y-r),3)
        elif visual in ("skeleton","guard"):
            pg.draw.circle(self.screen,bone,(x,y-2),int(r*.62)); pg.draw.circle(self.screen,dark,(x-4,y-3),3); pg.draw.circle(self.screen,dark,(x+4,y-3),3)
            if visual=="guard": pg.draw.arc(self.screen,steel,(x-r,y-r,r*2,r*2),math.pi,2*math.pi,4)
        elif visual in ("minion","megaminion","dragon"):
            wing=(139,106,190) if visual!="dragon" else (88,171,109)
            pg.draw.polygon(self.screen,wing,[(x-r,y),(x-r-8,y-r//2),(x-r//3,y-r//3)])
            pg.draw.polygon(self.screen,wing,[(x+r,y),(x+r+8,y-r//2),(x+r//3,y-r//3)])
            pg.draw.circle(self.screen,wing,(x,y),int(r*.68)); pg.draw.circle(self.screen,GOLD,(x-4,y-2),2); pg.draw.circle(self.screen,GOLD,(x+4,y-2),2)
        elif visual=="balloon":
            pg.draw.ellipse(self.screen,(183,70,71),(x-r,y-r-5,r*2,r*2)); pg.draw.line(self.screen,(92,67,48),(x-7,y+r-2),(x-5,y+r+9),2); pg.draw.line(self.screen,(92,67,48),(x+7,y+r-2),(x+5,y+r+9),2); pg.draw.rect(self.screen,(121,86,50),(x-8,y+r+7,16,8))
        elif visual in ("knight","prince","darkprince","pekka"):
            helm=steel if visual!="darkprince" else (88,78,114)
            pg.draw.circle(self.screen,skin,(x,y+2),int(r*.60)); pg.draw.arc(self.screen,helm,(x-int(r*.72),y-int(r*.8),int(r*1.44),int(r*1.3)),0,math.pi,6)
            if visual=="pekka": pg.draw.polygon(self.screen,helm,[(x-r//2,y-r//2),(x-r,y-r-8),(x-r//3,y-r//5)]); pg.draw.polygon(self.screen,helm,[(x+r//2,y-r//2),(x+r,y-r-8),(x+r//3,y-r//5)])
            if visual in ("prince","darkprince"): pg.draw.line(self.screen,GOLD,(x+r//2,y+r//2),(x+r+10,y-r),4)
            else: pg.draw.line(self.screen,(230,230,220),(x+r//2,y+r//2),(x+r,y-r//2),3)
        elif visual in ("archer","musketeer","wizard","witch"):
            pg.draw.circle(self.screen,skin,(x,y),int(r*.58))
            if visual=="witch":
                pg.draw.polygon(self.screen,(90,60,120),[(x-r,y-6),(x+r,y-6),(x,y-r-12)])
            elif visual=="wizard":
                pg.draw.polygon(self.screen,(48,91,180),[(x-r//2,y-r//2),(x+r//2,y-r//2),(x,y-r-9)])
                pg.draw.circle(self.screen,(255,154,62),(x+r//2,y+r//3),4)
            elif visual=="archer":
                pg.draw.arc(self.screen,(125,82,52),(x-r,y-r,r*2,r*2),-1.2,1.2,3)
            else:
                pg.draw.line(self.screen,(65,65,70),(x+r//3,y+r//3),(x+r+8,y-r//2),4)
        elif visual in ("giant","hog","valkyrie","bomber","icegolem"):
            body=(185,226,240) if visual=="icegolem" else (skin if visual!="bomber" else dark)
            pg.draw.circle(self.screen,body,(x,y),int(r*.70))
            if visual=="giant": pg.draw.rect(self.screen,(126,81,49),(x-r//2,y+r//5,r,6),border_radius=3)
            elif visual=="icegolem":
                pg.draw.polygon(self.screen,(232,251,255),[(x-r//2,y-r//2),(x,y-r-6),(x+r//2,y-r//2)])
                pg.draw.circle(self.screen,(72,125,156),(x-4,y-2),2);pg.draw.circle(self.screen,(72,125,156),(x+4,y-2),2)
            elif visual=="valkyrie": pg.draw.line(self.screen,(193,198,205),(x-r,y+r),(x+r,y-r),5)
            elif visual=="bomber": pg.draw.circle(self.screen,(22,22,25),(x+r//2,y-r//2),7); pg.draw.line(self.screen,GOLD,(x+r//2,y-r//2-5),(x+r//2+5,y-r),2)
            elif visual=="hog": pg.draw.ellipse(self.screen,(126,82,67),(x-r,y+r//4,r*2,r))
        elif visual in ("icespirit","firespirit"):
            col=(179,232,250) if visual=="icespirit" else (255,131,52)
            pg.draw.circle(self.screen,col,(x,y),int(r*.67));pg.draw.polygon(self.screen,col,[(x-r//2,y-r//2),(x,y-r-8),(x+r//2,y-r//2)])
            pg.draw.circle(self.screen,(30,35,42),(x-4,y-2),2);pg.draw.circle(self.screen,(30,35,42),(x+4,y-2),2)
        elif visual=="wallbreaker":
            pg.draw.circle(self.screen,bone,(x,y),int(r*.60));pg.draw.circle(self.screen,dark,(x-4,y-2),2);pg.draw.circle(self.screen,dark,(x+4,y-2),2)
            pg.draw.circle(self.screen,(30,30,33),(x+r//2,y-r//2),6);pg.draw.line(self.screen,GOLD,(x+r//2,y-r//2-4),(x+r,y-r),2)
        elif visual in ("electrowizard","icewizard"):
            pg.draw.circle(self.screen,skin,(x,y),int(r*.58))
            cloak=(52,116,181) if visual=="electrowizard" else (134,205,235)
            pg.draw.polygon(self.screen,cloak,[(x-r,y+r//2),(x,y-r),(x+r,y+r//2)])
            if visual=="electrowizard":pg.draw.polygon(self.screen,GOLD,[(x+3,y-12),(x+9,y-2),(x+4,y-2),(x+8,y+9),(x-2,y),(x+2,y)])
            else:pg.draw.circle(self.screen,(235,252,255),(x+6,y-4),4)
        else:
            pg.draw.circle(self.screen,(205,205,205),(x,y),int(r*.65))

    def _draw_building(self,u,x,y,r,card):
        pg=self.pg
        team_col=BLUE if u.team==0 else RED
        body=pg.Rect(x-r,y-r,r*2,r*2)
        pg.draw.rect(self.screen,(180,170,148),body,border_radius=5)
        pg.draw.rect(self.screen,team_col,body,3,border_radius=5)
        if card.visual=="cannon":
            pg.draw.circle(self.screen,(74,80,86),(x,y),r//2); pg.draw.line(self.screen,(74,80,86),(x,y),(x,y-r-8),7)
        elif card.visual=="bombtower":
            pg.draw.circle(self.screen,(33,33,37),(x,y-4),r//2); self.text("B",x,y-4,12,TEXT,True,"center")
        else:
            pg.draw.polygon(self.screen,(95,88,84),[(x-r,y),(x,y-r),(x+r,y),(x,y+r)])
        self._hp_bar(x-r,y-r-13,r*2,u.hp/max(1,u.max_hp),team=u.team,height=5)
        if card.lifetime>0:
            frac=max(0,u.lifetime/card.lifetime)
            pg.draw.rect(self.screen,(60,65,73),(x-r,y+r+4,r*2,3))
            pg.draw.rect(self.screen,GOLD,(x-r,y+r+4,int(r*2*frac),3))
        if self.show_names: self.text(card.name,x,y+r+9,10,TEXT,True,"midtop")

    def _draw_projectiles(self):
        pg=self.pg
        for p in self.env.game.projectiles:
            x,y=self.world_to_screen(p.x,p.y)
            col=BLUE if p.team==0 else RED
            if p.visual in ("wizard","BabyDragon","dragon"):
                col=(255,142,55)
            elif p.visual in ("archer","spear"):
                col=(232,220,164)
            pg.draw.circle(self.screen,col,(x,y),5)
            pg.draw.circle(self.screen,(255,255,255),(x,y),2)

    def _draw_effects(self,back=False):
        pg=self.pg
        surf=pg.Surface((WINDOW_W,WINDOW_H),pg.SRCALPHA)
        for e in self.env.game.effects:
            if back != (e.kind in ("spawn","slow")):
                continue
            x,y=self.world_to_screen(e.x,e.y)
            rr=max(5,int(e.radius/CFG.width*ARENA_W))
            frac=_clamp(e.ttl/max(.001,e.max_ttl))
            if e.kind=="spawn": col=(120,220,255,int(90*frac))
            elif e.kind=="zap": col=(255,231,93,int(170*frac))
            elif e.kind=="fireball": col=(255,105,42,int(150*frac))
            elif e.kind=="snow": col=(188,236,255,int(150*frac))
            elif e.kind=="arrows": col=(236,222,175,int(110*frac))
            elif e.kind=="rocket": col=(255,80,50,int(170*frac))
            elif e.kind=="death_bomb": col=(35,35,35,int(140*frac))
            elif e.kind=="death_frost": col=(165,231,255,int(155*frac))
            elif e.kind=="deploy_zap": col=(255,231,93,int(175*frac))
            elif e.kind=="knockback": col=(235,245,255,int(120*frac))
            elif e.kind=="shield_break": col=(120,205,255,int(150*frac))
            elif e.kind=="charge": col=(255,215,76,int(160*frac))
            else: col=(255,255,255,int(90*frac))
            pg.draw.circle(surf,col,(x,y),max(2,int(rr*(1.1-frac*.25))),max(2,int(5*frac)))
        self.screen.blit(surf,(0,0))

    def _hp_bar(self,x,y,w,frac,team=0,height=6,label=None):
        pg=self.pg; frac=_clamp(frac)
        pg.draw.rect(self.screen,(28,31,36),(int(x),int(y),int(w),height),border_radius=3)
        col=BLUE if team==0 else RED
        pg.draw.rect(self.screen,col,(int(x),int(y),int(w*frac),height),border_radius=3)
        if label is not None:
            self.text(label,x+w/2,y-2,9,TEXT,True,"midbottom")

    def _draw_panel(self):
        pg=self.pg
        pg.draw.rect(self.screen,PANEL,(PANEL_X,ARENA_Y,PANEL_W,ARENA_H),border_radius=16)
        g=self.env.game
        remain=max(0,int(math.ceil(g.max_time-g.time)))
        mins,secs=divmod(remain,60)
        self.text(f"{mins}:{secs:02d}",PANEL_X+24,ARENA_Y+20,34,TEXT,True)
        self.text(g.phase,PANEL_X+140,ARENA_Y+31,17,GOLD,True)
        self.text(f"{self.speed:.2g}×",PANEL_X+PANEL_W-26,ARENA_Y+31,17,MUTED,True,"topright")
        self.text("SPACE pause   +/- speed   R ranges   N names",PANEL_X+24,ARENA_Y+66,12,MUTED)

        self._player_block(1,PANEL_X+20,ARENA_Y+105)
        self._player_block(0,PANEL_X+20,ARENA_Y+300)

        y=ARENA_Y+495
        self.text("Recent plays",PANEL_X+24,y,15,TEXT,True)
        y+=26
        for line in g.combat_log[-6:][::-1]:
            self.text(line,PANEL_X+28,y,12,MUTED); y+=20

        if self.human_team is not None:
            self._draw_hand(self.human_team,PANEL_X+18,ARENA_Y+655)
        else:
            self.text("AI policy view",PANEL_X+24,ARENA_Y+655,15,TEXT,True)
            y=ARENA_Y+684
            for team in (1,0):
                d=self.decisions[team]
                col=RED if team==1 else BLUE
                self.text(f"P{team+1}  V={d.value:+.2f}  {d.label}",PANEL_X+24,y,12,col,True); y+=20
                for label,prob in d.top[:2]:
                    self.text(f"  {prob*100:4.1f}%  {label}",PANEL_X+28,y,11,MUTED); y+=17
                y+=5

    def _player_block(self,team,x,y):
        pg=self.pg; p=self.env.game.players[team]; col=BLUE if team==0 else RED
        title=("YOU" if team==self.human_team else f"PLAYER {team+1} / AI")
        self.text(title,x,y,16,col,True)
        self.text("♛ "*p.crowns,x+PANEL_W-55,y,16,GOLD,True,"topright")
        y+=30
        self.text("ELIXIR",x,y,11,MUTED,True)
        barx=x+65; barw=300
        pg.draw.rect(self.screen,(57,43,69),(barx,y,barw,15),border_radius=7)
        pg.draw.rect(self.screen,(194,76,227),(barx,y,int(barw*p.elixir/CFG.max_elixir),15),border_radius=7)
        self.text(f"{p.elixir:.1f}",barx+barw/2,y+7,10,TEXT,True,"center")
        y+=35
        for i,cid in enumerate(p.hand):
            card=BY_ID[cid]
            rx=x+i*94
            pg.draw.rect(self.screen,PANEL_2,(rx,y,86,70),border_radius=8)
            pg.draw.rect(self.screen,(84,92,108),(rx,y,86,70),2,border_radius=8)
            pg.draw.circle(self.screen,(194,76,227),(rx+15,y+15),11)
            self.text(card.cost,rx+15,y+15,11,TEXT,True,"center")
            self.text(card.name[:10],rx+43,y+42,10,TEXT,True,"center")

    def _draw_hand(self,team,x,y):
        pg=self.pg; p=self.env.game.players[team]
        self.text("Your hand — keys 1..4, then click arena",x+6,y-28,14,TEXT,True)
        self.card_rects.clear()
        for slot,cid in enumerate(p.hand):
            card=BY_ID[cid]
            rect=pg.Rect(x+slot*100,y,92,126)
            self.card_rects[slot]=rect
            selected=slot==self.selected_slot
            pg.draw.rect(self.screen,(53,62,79) if not selected else (72,89,121),rect,border_radius=9)
            pg.draw.rect(self.screen,GOLD if selected else (93,103,121),rect,3 if selected else 2,border_radius=9)
            pg.draw.circle(self.screen,(194,76,227),(rect.left+17,rect.top+18),12)
            self.text(card.cost,rect.left+17,rect.top+18,12,TEXT,True,"center")
            self.text(str(slot+1),rect.right-10,rect.top+10,10,MUTED,True,"topright")
            self._mini_card_icon(card.visual,team,rect.centerx,rect.top+59)
            self.text(card.name[:12],rect.centerx,rect.bottom-22,10,TEXT,True,"center")

    def _mini_card_icon(self,visual,team,x,y):
        pg = self.pg
        if visual == "fireball":
            pg.draw.circle(self.screen,(255,111,38),(x,y),16); pg.draw.circle(self.screen,(255,205,74),(x+3,y-2),8)
        elif visual == "arrows":
            for dx in (-7,0,7):
                pg.draw.line(self.screen,(226,216,172),(x+dx-7,y+10),(x+dx+7,y-10),3)
                pg.draw.polygon(self.screen,(226,216,172),[(x+dx+7,y-10),(x+dx+1,y-7),(x+dx+5,y-3)])
        elif visual == "zap":
            pg.draw.polygon(self.screen,GOLD,[(x-3,y-18),(x+6,y-3),(x,y-3),(x+4,y+18),(x-9,y+2),(x-2,y+2)])
        elif visual == "snowball":
            pg.draw.circle(self.screen,(221,246,255),(x,y),16); pg.draw.circle(self.screen,(159,218,241),(x-5,y-5),4)
        elif visual == "rocket":
            pg.draw.polygon(self.screen,(206,65,58),[(x,y-19),(x+10,y+8),(x,y+16),(x-10,y+8)]); pg.draw.circle(self.screen,(235,226,204),(x,y-3),4)
        elif visual in ("cannon","bombtower","tombstone"):
            pg.draw.rect(self.screen,(181,171,148),(x-16,y-16,32,32),border_radius=5); pg.draw.rect(self.screen,BLUE if team==0 else RED,(x-16,y-16,32,32),2,border_radius=5)
            self.text("C" if visual=="cannon" else ("B" if visual=="bombtower" else "T"),x,y,12,TEXT,True,"center")
        else:
            self._draw_unit_icon(visual,team,x,y,18,False)

    def _draw_game_over(self):
        pg=self.pg
        surf=pg.Surface((ARENA_W,190),pg.SRCALPHA); surf.fill((10,13,18,220))
        self.screen.blit(surf,(ARENA_X,ARENA_Y+ARENA_H//2-95))
        w=self.env.game.winner
        title="DRAW" if w is None else ("YOU WIN" if w==self.human_team else f"PLAYER {w+1} WINS")
        self.text(title,ARENA_X+ARENA_W/2,ARENA_Y+ARENA_H/2-28,38,GOLD,True,"center")
        self.text(f"Crowns {self.env.game.players[0].crowns} — {self.env.game.players[1].crowns}",ARENA_X+ARENA_W/2,ARENA_Y+ARENA_H/2+28,18,TEXT,True,"center")



class DraftViewer:
    """Animated pre-match draft scene for AI-vs-AI and human-vs-AI."""
    CARD_W, CARD_H = 205, 260

    def __init__(self, policy0, policy1, *, human_team=None, seed=11, device="cpu", speed=1.0,
                 title="Draft", headless=False, auto_close=False):
        self.pg=_pg()
        if headless: os.environ.setdefault("SDL_VIDEODRIVER","dummy")
        self.pg.init(); self.pg.display.set_caption(title)
        self.screen=self.pg.display.set_mode((WINDOW_W,WINDOW_H)); self.clock=self.pg.time.Clock()
        import random
        self.rng=random.Random(seed ^ 0xDFA7)
        self.ds=DraftState.create(seed=seed ^ 0x91A2,first_chooser=self.rng.randrange(2))
        self.policies=[policy0,policy1]; self.human_team=human_team; self.device=device
        self.speed=max(.25,float(speed)); self.running=True; self.fonts={}; self.offer=None
        self.selection=None; self.pending_human=[]; self.card_rects={}; self.confirm_rect=None
        self.reveal_left=0.0; self.reveal_total=0.0; self.final_left=0.0; self.log=[]; self.auto_close=auto_close
        self._new_offer()

    def font(self,size,bold=False):
        k=(size,bold)
        if k not in self.fonts:self.fonts[k]=self.pg.font.SysFont("DejaVu Sans",size,bold=bold)
        return self.fonts[k]

    def text(self,s,x,y,size=18,color=TEXT,bold=False,anchor="topleft"):
        surf=self.font(size,bold).render(str(s),True,color);r=surf.get_rect();setattr(r,anchor,(int(x),int(y)));self.screen.blit(surf,r);return r

    def _new_offer(self):
        if self.ds.done:
            self.offer=None; self.final_left=max(.8,1.5/self.speed); return
        self.offer=self.ds.offer(); self.selection=None; self.pending_human=[]; self.reveal_left=0.0

    def _ai_action(self):
        chooser=self.ds.chooser; obs=self.ds.observe(chooser,self.offer); p=self.policies[chooser]
        if isinstance(p,ActorCritic): a,_,_=p.draft_act(obs,deterministic=False,device=self.device)
        else: a=self.rng.randrange(DRAFT_ACTION_DIM)
        oi,gi=DRAFT_ACTIONS[int(a)]; self.selection=(oi,gi,int(a)); self.reveal_left=max(.18,.85/self.speed); self.reveal_total=self.reveal_left

    def _commit(self):
        if self.selection is None:return
        oi,gi,a=self.selection; chooser=self.ds.chooser; own=self.offer[oi]; given=self.offer[gi]
        self.log.append((self.ds.round_no+1,chooser,tuple(card_name(x) for x in self.offer),card_name(own),card_name(given)))
        self.ds.apply(self.offer,a); self._new_offer()

    def _human_confirm(self):
        if len(self.pending_human)!=2:return
        oi,gi=self.pending_human
        if oi==gi:return
        a=DRAFT_ACTIONS.index((oi,gi)); self.selection=(oi,gi,a); self.reveal_left=max(.18,.65/self.speed); self.reveal_total=self.reveal_left

    def _events(self):
        for e in self.pg.event.get():
            if e.type==self.pg.QUIT:self.running=False
            elif e.type==self.pg.KEYDOWN:
                if e.key==self.pg.K_ESCAPE:self.running=False
                elif e.key in (self.pg.K_EQUALS,self.pg.K_PLUS,self.pg.K_KP_PLUS):self.speed=min(12,self.speed*1.4)
                elif e.key in (self.pg.K_MINUS,self.pg.K_KP_MINUS):self.speed=max(.25,self.speed/1.4)
                elif e.key==self.pg.K_RETURN and self.human_team==self.ds.chooser:self._human_confirm()
                elif e.key==self.pg.K_BACKSPACE and self.human_team==self.ds.chooser:self.pending_human=[]
            elif e.type==self.pg.MOUSEBUTTONDOWN and e.button==1 and self.offer is not None and self.ds.chooser==self.human_team and self.reveal_left<=0:
                for idx,r in self.card_rects.items():
                    if r.collidepoint(e.pos):
                        if idx in self.pending_human:self.pending_human.remove(idx)
                        elif len(self.pending_human)<2:self.pending_human.append(idx)
                        break
                if self.confirm_rect and self.confirm_rect.collidepoint(e.pos):self._human_confirm()

    def run(self,max_frames=None):
        frames=0; think_left=.35/self.speed
        while self.running:
            dt=self.clock.tick(60)/1000.0; self._events()
            if self.ds.done:
                self.final_left-=dt
                if self.final_left<=0:break
            elif self.reveal_left>0:
                self.reveal_left-=dt
                if self.reveal_left<=0:self._commit();think_left=.35/self.speed
            elif self.ds.chooser!=self.human_team:
                think_left-=dt
                if think_left<=0:self._ai_action()
            self._draw();self.pg.display.flip();frames+=1
            if max_frames is not None and frames>=max_frames:break
        result=self.ds.result() if self.ds.done else None
        self.pg.quit(); return result,self.log

    def _icon(self,card,team,x,y):
        pg=self.pg; tc=BLUE if team==0 else RED
        visual=card.visual
        if visual in ("fireball","firespirit"):
            pg.draw.circle(self.screen,(255,111,38),(x,y),23);pg.draw.circle(self.screen,(255,205,74),(x+4,y-3),10)
        elif visual in ("snowball","icespirit","icegolem","icewizard"):
            pg.draw.circle(self.screen,(191,234,250),(x,y),23);pg.draw.circle(self.screen,(239,252,255),(x-5,y-6),8)
        elif visual in ("zap","electrowizard"):
            pg.draw.circle(self.screen,tc,(x,y),23);pg.draw.polygon(self.screen,GOLD,[(x-3,y-20),(x+7,y-4),(x+1,y-4),(x+6,y+18),(x-10,y+2),(x-3,y+2)])
        elif card.kind=="building":
            pg.draw.rect(self.screen,(173,164,145),(x-25,y-24,50,48),border_radius=7);pg.draw.rect(self.screen,tc,(x-25,y-24,50,48),3,border_radius=7)
        else:
            pg.draw.circle(self.screen,tc,(x,y),25);pg.draw.circle(self.screen,(230,230,222),(x,y),16);self.text(card.name[0],x,y,15,INK,True,"center")

    def _deck_panel(self,team,x,y,w=240):
        pg=self.pg; col=BLUE if team==0 else RED
        pg.draw.rect(self.screen,PANEL,(x,y,w,600),border_radius=15);self.text(f"PLAYER {team+1}",x+18,y+16,18,col,True)
        self.text(f"{len(self.ds.decks[team])}/8 cards",x+w-18,y+19,12,MUTED,False,"topright")
        yy=y+58
        for slot in range(8):
            rect=pg.Rect(x+16,yy+slot*63,w-32,52);pg.draw.rect(self.screen,PANEL_2,rect,border_radius=8)
            if slot<len(self.ds.decks[team]):
                c=BY_ID[self.ds.decks[team][slot]];pg.draw.circle(self.screen,(194,76,227),(rect.left+18,rect.centery),12);self.text(c.cost,rect.left+18,rect.centery,11,TEXT,True,"center");self.text(c.name,rect.left+40,rect.centery,13,TEXT,True,"midleft")
            else:self.text("—",rect.centerx,rect.centery,14,(90,99,113),False,"center")

    def _draw(self):
        pg=self.pg;self.screen.fill(INK)
        pg.draw.rect(self.screen,(25,31,41),(0,0,WINDOW_W,WINDOW_H))
        self.text("PRE-MATCH DRAFT",WINDOW_W/2,28,30,TEXT,True,"midtop")
        self.text("Choose one card to KEEP and one to GIVE",WINDOW_W/2,68,14,MUTED,False,"midtop")
        self._deck_panel(0,24,130);self._deck_panel(1,WINDOW_W-264,130)
        if self.ds.done:
            self.text("DRAFT COMPLETE",WINDOW_W/2,170,32,GOLD,True,"center")
            self.text("Final decks locked — battle starts next",WINDOW_W/2,215,16,MUTED,False,"center")
            self._draw_history(350);return
        chooser=self.ds.chooser; col=BLUE if chooser==0 else RED
        self.text(f"ROUND {self.ds.round_no+1} / 8",WINDOW_W/2,110,17,GOLD,True,"center")
        label="YOU" if chooser==self.human_team else f"AI PLAYER {chooser+1}"
        self.text(f"{label} is choosing",WINDOW_W/2,142,20,col,True,"center")
        startx=286; gap=14; y=205; self.card_rects={}
        for i,cid in enumerate(self.offer):
            card=BY_ID[cid]; rect=pg.Rect(startx+i*(self.CARD_W+gap),y,self.CARD_W,self.CARD_H);self.card_rects[i]=rect
            border=(88,99,118); badge=None
            if self.selection:
                if i==self.selection[0]:border=GREEN;badge="KEEP"
                elif i==self.selection[1]:border=RED;badge="GIVE"
            elif self.ds.chooser==self.human_team:
                if self.pending_human and i==self.pending_human[0]:border=GREEN;badge="KEEP"
                elif len(self.pending_human)>1 and i==self.pending_human[1]:border=RED;badge="GIVE"
            pg.draw.rect(self.screen,(44,52,67),rect,border_radius=14);pg.draw.rect(self.screen,border,rect,4 if badge else 2,border_radius=14)
            pg.draw.circle(self.screen,(194,76,227),(rect.left+24,rect.top+25),16);self.text(card.cost,rect.left+24,rect.top+25,14,TEXT,True,"center")
            self.text(card.name,rect.centerx,rect.top+52,15,TEXT,True,"center");self._icon(card,chooser,rect.centerx,rect.top+118)
            self.text(card.kind.upper(),rect.centerx,rect.top+166,10,MUTED,True,"center")
            stats=[]
            if card.hp:stats.append(f"HP {int(card.hp)}")
            if card.damage:stats.append(f"DMG {int(card.damage)}")
            if card.spell_radius:stats.append(f"AOE {card.spell_radius:.1f}")
            self.text("  •  ".join(stats[:2]) or "utility",rect.centerx,rect.top+196,11,MUTED,False,"center")
            if badge:
                br=pg.Rect(rect.centerx-38,rect.bottom-34,76,24);pg.draw.rect(self.screen,border,br,border_radius=11);self.text(badge,br.centerx,br.centery,11,INK,True,"center")
        if self.selection is not None and self.reveal_left > 0:
            self._draw_transfer_animation()
        if chooser==self.human_team and self.reveal_left<=0:
            self.text("Click KEEP first, GIVE second. Backspace resets.",WINDOW_W/2,490,13,MUTED,False,"center")
            self.confirm_rect=pg.Rect(WINDOW_W//2-90,516,180,44); enabled=len(self.pending_human)==2
            pg.draw.rect(self.screen,GREEN if enabled else (67,75,88),self.confirm_rect,border_radius=10);self.text("CONFIRM / ENTER",self.confirm_rect.centerx,self.confirm_rect.centery,13,INK if enabled else MUTED,True,"center")
        else:
            self.confirm_rect=None; self.text("AI policy is evaluating all 12 keep/give actions…",WINDOW_W/2,520,13,MUTED,False,"center")
        self._draw_history(585)
        self.text(f"+/- speed  {self.speed:.1f}×   ESC close",WINDOW_W/2,872,11,MUTED,False,"center")

    def _draw_transfer_animation(self):
        # During the reveal, visibly fly KEEP toward the chooser deck and GIVE
        # toward the opponent deck. This makes AI draft decisions readable even
        # without watching the textual history.
        if self.selection is None or self.offer is None:
            return
        oi, gi, _ = self.selection
        progress = 1.0 - self.reveal_left / max(1e-6, self.reveal_total)
        progress = max(0.0, min(1.0, progress))
        # Smoothstep feels less mechanical than a constant-speed translation.
        progress = progress * progress * (3.0 - 2.0 * progress)
        chooser = self.ds.chooser
        for idx, recipient, color, label in ((oi, chooser, GREEN, "KEEP"), (gi, 1-chooser, RED, "GIVE")):
            src = self.card_rects[idx].center
            panel_x = 24 if recipient == 0 else WINDOW_W - 264
            slot = min(7, len(self.ds.decks[recipient]))
            dst = (panel_x + 120, 130 + 58 + slot*63 + 26)
            cx = int(src[0] + (dst[0]-src[0])*progress)
            cy = int(src[1] + (dst[1]-src[1])*progress)
            self.pg.draw.line(self.screen, color, src, (cx, cy), 2)
            ghost = self.pg.Rect(cx-44, cy-25, 88, 50)
            self.pg.draw.rect(self.screen, (48,56,70), ghost, border_radius=9)
            self.pg.draw.rect(self.screen, color, ghost, 3, border_radius=9)
            card = BY_ID[self.offer[idx]]
            self.text(card.name[:10], ghost.centerx, ghost.centery-5, 10, TEXT, True, "center")
            self.text(label, ghost.centerx, ghost.centery+11, 8, color, True, "center")

    def _draw_history(self,y):
        self.text("DRAFT HISTORY",286,y,12,MUTED,True);yy=y+24
        for rnd,chooser,offer,own,given in self.log[-5:]:
            col=BLUE if chooser==0 else RED
            self.text(f"R{rnd} P{chooser+1}",286,yy,11,col,True);self.text(f"kept {own}   → gave {given}",350,yy,11,TEXT);yy+=24

def _draft_for_view(model0, model1, seed=11, device="cpu"):
    import random
    rng=random.Random(seed ^ 0xDFA7); ds=DraftState.create(seed=seed ^ 0x91A2, first_chooser=rng.randrange(2)); policies=[model0,model1]; log=[]
    while not ds.done:
        chooser=ds.chooser; offer=ds.offer(); obs=ds.observe(chooser,offer); p=policies[chooser]
        if isinstance(p,ActorCritic): action,_,_=p.draft_act(obs,deterministic=False,device=device)
        else: action=rng.randrange(DRAFT_ACTION_DIM)
        oi,pi=DRAFT_ACTIONS[action]; own,opp=offer[oi],offer[pi]
        log.append((ds.round_no+1,chooser,tuple(card_name(x) for x in offer),card_name(own),card_name(opp)))
        ds.apply(offer,action)
    return ds.result(),log

def watch_models(model0, model1, *, speed: float = 2.0, title="Self-play", device="cpu", auto_close=False, seed=11,
                 headless=False, max_frames=None, draft=True):
    if draft:
        dv=DraftViewer(model0,model1,seed=seed,device=device,speed=max(1.0,speed/2),title=title+" — Draft",headless=headless,auto_close=auto_close)
        drafted,dlog=dv.run(max_frames=max_frames if headless else None)
        if drafted is None:return {"aborted":True,"phase":"draft"}
        deck0,deck1=drafted
        print("DRAFT")
        for rnd,chooser,offer,own,opp in dlog:print(f"  r{rnd}: P{chooser+1} [{', '.join(offer)}] -> self={own}, opponent={opp}")
    else:deck0=deck1=None
    base=ClashRoyaleEnv(seed=seed)
    env=ClashRoyaleEnv(deck0=deck0 or base.deck0,deck1=deck1 or base.deck1,seed=seed);env.reset(seed)
    viewer=ArenaViewer(env,model0,model1,speed=speed,title=title+(' — drafted' if draft else ''),device=device,auto_close=auto_close,headless=headless)
    viewer.run(max_frames=max_frames);return env.game.summary()


def human_vs_model(model, *, human_team: int = 0, speed: float = 1.0, device="cpu", seed=22, draft=True):
    policies=[None,None];policies[1-human_team]=model
    if draft:
        dv=DraftViewer(policies[0],policies[1],human_team=human_team,seed=seed,device=device,speed=1.0,title="Clash RL v3.1 — Human Draft")
        drafted,dlog=dv.run()
        if drafted is None:return {"aborted":True,"phase":"draft"}
        deck0,deck1=drafted
    else:
        base=ClashRoyaleEnv(seed=seed);deck0,deck1=base.deck0,base.deck1
    env=ClashRoyaleEnv(deck0=deck0,deck1=deck1,seed=seed);env.reset(seed)
    ArenaViewer(env,policies[0],policies[1],human_team=human_team,speed=speed,title="Clash RL v3.1 — Human vs Agent",device=device).run()
    return env.game.summary()
