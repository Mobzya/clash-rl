from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path


W, H = 1280, 930
BG = (20, 24, 31); PANEL = (30, 36, 47); PANEL2 = (38, 45, 58); TEXT = (236, 240, 246); MUTED = (154, 165, 181)
BLUE = (72, 157, 255); GREEN = (86, 194, 111); GOLD = (244, 190, 65); RED = (235, 91, 99); PURPLE = (188, 104, 225)


def _pg():
    try:
        import pygame
    except ImportError as exc:
        raise RuntimeError("Dashboard requires pygame-ce. Run: pip install -e .") from exc
    return pygame


def _safe(v, default=0.0):
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except Exception:
        return default


class TrainingDashboard:
    def __init__(self, csv_path: str | Path):
        self.pg = _pg(); self.pg.init(); self.pg.display.set_caption("Clash RL v3.1 — Training Dashboard")
        self.screen = self.pg.display.set_mode((W, H)); self.clock = self.pg.time.Clock(); self.csv_path = Path(csv_path)
        self.rows = []; self.last_mtime = 0.0; self.running = True; self.fonts = {}; self.smoothing = 7
        self.tournament_rows: list[dict] = []; self.tournament_no = 0; self.tournament_mtime = 0.0

    def font(self, size, bold=False):
        k = (size, bold)
        if k not in self.fonts: self.fonts[k] = self.pg.font.SysFont("DejaVu Sans", size, bold=bold)
        return self.fonts[k]

    def text(self, s, x, y, size=16, color=TEXT, bold=False, anchor="topleft"):
        surf = self.font(size, bold).render(str(s), True, color); r = surf.get_rect(); setattr(r, anchor, (int(x), int(y))); self.screen.blit(surf, r); return r

    def run(self):
        while self.running:
            for e in self.pg.event.get():
                if e.type == self.pg.QUIT: self.running = False
                elif e.type == self.pg.KEYDOWN:
                    if e.key == self.pg.K_ESCAPE: self.running = False
                    elif e.key in (self.pg.K_PLUS, self.pg.K_EQUALS, self.pg.K_KP_PLUS): self.smoothing = min(50, self.smoothing + 2)
                    elif e.key in (self.pg.K_MINUS, self.pg.K_KP_MINUS): self.smoothing = max(1, self.smoothing - 2)
            self._reload(); self._draw(); self.pg.display.flip(); self.clock.tick(8)
        self.pg.quit()

    def _reload(self):
        if self.csv_path.exists():
            m = self.csv_path.stat().st_mtime
            if m > self.last_mtime:
                try:
                    with self.csv_path.open(newline="") as f: self.rows = list(csv.DictReader(f))
                    self.last_mtime = m
                except Exception:
                    pass
        tp = self.csv_path.parent / "tournament" / "last_tournament.json"
        if tp.exists():
            m = tp.stat().st_mtime
            if m > self.tournament_mtime:
                try:
                    raw = json.loads(tp.read_text())
                    self.tournament_rows = list(raw.get("leaderboard", []))
                    self.tournament_no = int(raw.get("tournament", 0))
                    self.tournament_mtime = m
                except Exception:
                    pass

    def _metric(self, row, key): return _safe(row.get(key, 0))

    def _smooth(self, vals):
        if self.smoothing <= 1: return vals
        out = []
        for i in range(len(vals)):
            lo = max(0, i - self.smoothing + 1); chunk = vals[lo:i+1]; out.append(sum(chunk) / len(chunk))
        return out

    def _draw(self):
        pg = self.pg; self.screen.fill(BG)
        self.text("CLASH RL  /  TRAINING + TOURNAMENT LAB", 32, 24, 25, TEXT, True)
        self.text(str(self.csv_path), 32, 58, 12, MUTED)
        self.text(f"smoothing {self.smoothing}   (+/-)", W - 32, 28, 12, MUTED, False, "topright")
        if not self.rows:
            self.text("Waiting for training.csv…", W/2, H/2, 26, MUTED, True, "center"); return

        r = self.rows[-1]
        steps = int(self._metric(r, "steps")); upd = int(self._metric(r, "update")); games = int(self._metric(r, "total_games")); wr = self._metric(r, "win_rate")
        secs = max(.001, self._metric(r, "seconds")); rollout_steps = steps - int(self._metric(self.rows[-2], "steps")) if len(self.rows) > 1 else steps
        throughput = self._metric(r, "steps_per_sec") or max(0, rollout_steps / secs)
        if self.tournament_rows:
            champ = self.tournament_rows[0]
            tournament_value = f"#{champ.get('rank', 1)}  {float(champ.get('rating', 1000)):.0f} Elo"
        else:
            tournament_value = "waiting"
        cards = [
            ("UPDATE", f"{upd}", BLUE),
            ("TRAIN GAMES", f"{games:,}", PURPLE),
            ("TOTAL STEPS", f"{steps:,}", TEXT),
            ("ROLLOUT WIN", f"{wr*100:.1f}%", GREEN if wr >= .5 else GOLD),
            ("STEPS / SEC", f"{throughput:,.0f}", GOLD),
            (f"TOURNAMENT #{self.tournament_no}" if self.tournament_no else "TOURNAMENT", tournament_value, BLUE),
        ]
        x = 32; y = 92; cw = (W - 64 - 5*14) / 6
        for label, val, col in cards:
            pg.draw.rect(self.screen, PANEL, (x, y, cw, 82), border_radius=11)
            self.text(label, x+14, y+12, 10, MUTED, True); self.text(val, x+14, y+38, 19, col, True); x += cw + 14

        charts = [
            ("win_rate", "Rollout win rate", GREEN, 0.0, 1.0, "%"),
            ("value_loss", "Value loss", GOLD, None, None, ""),
            ("entropy", "Policy entropy", PURPLE, None, None, ""),
            ("approx_kl", "Approx KL", BLUE, 0.0, None, ""),
        ]
        for i, cfg in enumerate(charts):
            cx = 32 + (i % 2) * 614; cy = 196 + (i // 2) * 238
            self._chart(cx, cy, 582, 214, *cfg)
        self._tournament_panel(680)
        self._footer()

    def _chart(self, x, y, w, h, key, title, color, fixed_lo, fixed_hi, suffix):
        pg = self.pg; pg.draw.rect(self.screen, PANEL, (x, y, w, h), border_radius=12)
        vals = [self._metric(r, key) for r in self.rows]
        vals = self._smooth(vals)[-240:]
        raw_latest = self._metric(self.rows[-1], key)
        self.text(title, x+16, y+13, 14, TEXT, True)
        shown = f"{raw_latest*100:.1f}%" if suffix == "%" else f"{raw_latest:.5g}"
        self.text(shown, x+w-16, y+13, 16, color, True, "topright")
        px, py = x+48, y+48; pw, ph = w-66, h-70
        if not vals: return
        lo = min(vals) if fixed_lo is None else fixed_lo; hi = max(vals) if fixed_hi is None else fixed_hi
        if hi - lo < 1e-9: hi = lo + 1.0
        for j in range(5):
            yy = py + j*ph/4; pg.draw.line(self.screen, (53, 61, 74), (px, int(yy)), (px+pw, int(yy)), 1)
            v = hi - (hi-lo)*j/4; lab = f"{v*100:.0f}%" if suffix == "%" else f"{v:.3g}"; self.text(lab, px-8, yy, 9, MUTED, False, "midright")
        if len(vals) == 1: pts = [(px+pw/2, py+ph*(1-(vals[0]-lo)/(hi-lo)))]
        else: pts = [(px+i*pw/(len(vals)-1), py+ph*(1-(v-lo)/(hi-lo))) for i, v in enumerate(vals)]
        if len(pts) > 1: pg.draw.lines(self.screen, color, False, [(int(a), int(b)) for a, b in pts], 3)
        pg.draw.circle(self.screen, color, (int(pts[-1][0]), int(pts[-1][1])), 5)
        self.text(f"last {len(vals)} updates", px, py+ph+8, 9, MUTED)

    def _tournament_panel(self, y):
        pg = self.pg
        pg.draw.rect(self.screen, PANEL, (32, y, W-64, 166), border_radius=12)
        self.text("MODEL TOURNAMENT", 50, y+14, 14, TEXT, True)
        self.text("Current tournament is ranked by wins → points → Elo; hall-of-fame uses Elo for cross-generation fairness.", 200, y+15, 11, MUTED)
        if not self.tournament_rows:
            self.text("No completed tournament yet. Automatic tournaments begin after the configured training-game milestone.", 50, y+67, 14, MUTED)
            return
        headers = [("RANK", 50), ("MODEL", 115), ("TRAIN GAMES", 410), ("W/L/D", 570), ("POINTS", 720), ("ELO", 850)]
        for h, x in headers: self.text(h, x, y+45, 10, MUTED, True)
        for i, row in enumerate(self.tournament_rows[:4]):
            yy = y + 69 + i*22
            col = GOLD if i == 0 else TEXT
            self.text(f"#{row.get('rank', i+1)}", 50, yy, 11, col, True)
            self.text(str(row.get("id", "?"))[:28], 115, yy, 11, col if i == 0 else TEXT, i == 0)
            self.text(f"{int(row.get('training_games', 0)):,}", 410, yy, 11, TEXT)
            self.text(f"{row.get('wins',0)}/{row.get('losses',0)}/{row.get('draws',0)}", 570, yy, 11, TEXT)
            self.text(f"{float(row.get('points',0)):.1f}", 720, yy, 11, TEXT)
            self.text(f"{float(row.get('rating',1000)):.1f}", 850, yy, 11, BLUE if i == 0 else TEXT, i == 0)

    def _footer(self):
        y = 875; r = self.rows[-1]
        self.text("Latest PPO", 32, y, 12, MUTED, True)
        metrics = [
            ("policy", self._metric(r, "policy_loss")),
            ("value", self._metric(r, "value_loss")),
            ("entropy", self._metric(r, "entropy")),
            ("KL", self._metric(r, "approx_kl")),
            ("draft H", self._metric(r, "draft_entropy")),
            ("rollout s", self._metric(r, "rollout_seconds")),
            ("update s", self._metric(r, "update_seconds")),
        ]
        x = 120
        for n, v in metrics:
            self.text(f"{n}: {v:+.4g}", x, y, 11, TEXT); x += 145
        self.text(time.strftime("updated %H:%M:%S"), W-32, y, 11, MUTED, False, "topright")


def run_dashboard(csv_path):
    TrainingDashboard(csv_path).run()
