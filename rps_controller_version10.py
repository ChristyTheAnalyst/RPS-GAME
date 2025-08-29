#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rock–Paper–Scissor (Storyboard Edition) - Single Python Script (no Unity)
- Pygame UI with 5 screens and random matte gradients
- Uses OpenCV + MediaPipe for hand gesture classification
- Reuses GestureClassifier from your shared program (if importable), otherwise falls back to an angle-based classifier
- Sends TCP messages to the robotic arm at countdown==1 with chosen robot move (difficulty-based)
- Shows ROCK -> PAPER -> SCISSOR within 2.2s
- Shows result, supports Play Again (keeping history) or New Player (reset)

Run:
    pip install pygame opencv-python mediapipe numpy
    python rps_storyboard.py --arm-host 192.168.1.37 --arm-port 5050 --camera 0
"""

import argparse
import json
import socket
import threading
import time
import random
from collections import deque, Counter, defaultdict

import pygame
import cv2
import numpy as np

# ------------------------- TCP Client -------------------------

class TCPClient:
    def __init__(self, host: str, port: int, connect_timeout=2.0):
        self.host = host
        self.port = port
        self.sock = None
        self.connect_timeout = connect_timeout
        self._lock = threading.Lock()
        self.ensure_connected()

    def ensure_connected(self):
        if self.sock is not None:
            return
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(self.connect_timeout)
        s.connect((self.host, self.port))
        s.settimeout(None)
        self.sock = s

    def send_json(self, obj: dict):
        # Each message ends with CRLF so the receiver prints on a new line.
        line = json.dumps(obj, separators=(",", ":")) + "\r\n"
        try:
            with self._lock:
                self.ensure_connected()
                self.sock.sendall(line.encode("utf-8"))
        except (BrokenPipeError, ConnectionResetError, OSError):
            try:
                if self.sock:
                    self.sock.close()
            except Exception:
                pass
            self.sock = None
            try:
                self.ensure_connected()
                with self._lock:
                    self.sock.sendall(line.encode("utf-8"))
            except Exception:
                pass

# ------------------------- Gesture Classifier (prefer external) -------------------------

GestureClassifier = None
try:
    # Prefer the user's exact class for compatibility
    from rps_controller_version8 import GestureClassifier as _ExternalGestureClassifier
    GestureClassifier = _ExternalGestureClassifier
except Exception:
    pass

if GestureClassifier is None:
    class GestureClassifier:  # Fallback angle-based classifier
        """
        Angle-based, orientation-robust classifier.
        - A finger is STRAIGHT if both (MCP–PIP–TIP) and (PIP–DIP–TIP) angles ≈ 180°.
        - A finger is CURLED if the PIP joint angle is much smaller (e.g., < 150°).
        """
        def __init__(self,
                     straight_thresh_deg: float = 25.0,
                     curl_pip_thresh_deg: float = 150.0,
                     ):
            self.straight_thresh_deg = straight_thresh_deg
            self.curl_pip_thresh_deg = curl_pip_thresh_deg

        @staticmethod
        def _angle_deg(a, b, c):
            import numpy as np, math
            v1 = np.array([a.x - b.x, a.y - b.y], dtype=float)
            v2 = np.array([c.x - b.x, c.y - b.y], dtype=float)
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 == 0 or n2 == 0:
                return 180.0
            cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
            return math.degrees(math.acos(cosang))

        def _is_straight(self, lm, mcp, pip, dip, tip) -> bool:
            need = 180.0 - self.straight_thresh_deg
            ang_pip = self._angle_deg(lm[mcp], lm[pip], lm[tip])
            ang_dip = self._angle_deg(lm[pip], lm[dip], lm[tip])
            return (ang_pip >= need) and (ang_dip >= need)

        def _is_curled(self, lm, mcp, pip, dip, tip) -> bool:
            ang_pip = self._angle_deg(lm[mcp], lm[pip], lm[dip])
            return ang_pip < self.curl_pip_thresh_deg

        def classify(self, hand_landmarks, image_h, image_w) -> str | None:
            lm = hand_landmarks.landmark
            idx  = (5, 6, 7, 8)
            mid  = (9, 10, 11, 12)
            ring = (13, 14, 15, 16)
            pink = (17, 18, 19, 20)

            idx_st  = self._is_straight(lm, *idx)
            mid_st  = self._is_straight(lm, *mid)
            ring_st = self._is_straight(lm, *ring)
            pink_st = self._is_straight(lm, *pink)

            ring_curled  = self._is_curled(lm, *ring)
            pinky_curled = self._is_curled(lm, *pink)

            if idx_st and mid_st and ring_curled and pinky_curled:
                return "scissors"
            if idx_st and mid_st and ring_st and pink_st and (not ring_curled) and (not pinky_curled):
                return "paper"
            if sum([idx_st, mid_st, ring_st, pink_st]) <= 1:
                return "rock"
            return None

# ------------------------- Camera Worker -------------------------

class CameraWorker(threading.Thread):
    """Captures frames, runs MediaPipe Hands, returns a rolling majority-vote guess via GestureClassifier."""
    def __init__(self, camera_index=0):
        super().__init__(daemon=True)
        self.camera_index = camera_index
        self.cap = None

        self.running = False
        self.vote_window = deque(maxlen=7)  # ~200ms @ 30fps
        self.last_guess = None
        self.last_frame = None
        self.lock = threading.Lock()

        try:
            import mediapipe as mp
        except ImportError:
            raise SystemExit("mediapipe not installed. Run: pip install mediapipe")
        self.mp = mp
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.classifier = GestureClassifier()

    def run(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {self.camera_index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.running = True
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self.lock:
                self.last_frame = frame.copy()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.hands.process(rgb)
            h, w = frame.shape[:2]
            guess = None
            if res.multi_hand_landmarks:
                hand_landmarks = res.multi_hand_landmarks[0]
                guess = self.classifier.classify(hand_landmarks, h, w)

            if guess:
                with self.lock:
                    self.vote_window.append(guess)
                    cnt = Counter(self.vote_window)
                    self.last_guess = cnt.most_common(1)[0][0]
            else:
                if len(self.vote_window) > 0:
                    self.vote_window.popleft()
                    with self.lock:
                        if len(self.vote_window) == 0:
                            self.last_guess = None

        self.cap.release()

    def get_gesture(self):
        with self.lock:
            return self.last_guess

    def get_frame(self):
        with self.lock:
            return None if self.last_frame is None else self.last_frame.copy()

    def stop(self):
        self.running = False

# ------------------------- Strategy / Predictor -------------------------

MOVES = ["rock", "paper", "scissors"]
WIN_MAP = {"rock": "paper", "paper": "scissors", "scissors": "rock"}      # what beats key
LOSE_MAP = {"rock": "scissors", "paper": "rock", "scissors": "paper"}     # what loses to key

def beats(move: str) -> str:
    return WIN_MAP.get(move, "rock")

def loses_to(move: str) -> str:
    return LOSE_MAP.get(move, "scissors")

class MarkovPredictor:
    """Tiny online N-gram (order=2 then 1) predictor for player's next move."""
    def __init__(self, order: int = 2):
        self.order = max(1, int(order))
        self.counts = defaultdict(lambda: Counter())
        self.history = []

    def update(self, move: str):
        self.history.append(move)
        if len(self.history) >= 2:
            k1 = (self.history[-1-0],)
            self.counts[k1][move] += 1
        if len(self.history) >= 3:
            k2 = (self.history[-2], self.history[-1])
            self.counts[k2][move] += 1

    def predict(self) -> str | None:
        if len(self.history) >= 2:
            k2 = (self.history[-2], self.history[-1])
            if k2 in self.counts and self.counts[k2]:
                return self.counts[k2].most_common(1)[0][0]
        if len(self.history) >= 1:
            k1 = (self.history[-1],)
            if k1 in self.counts and self.counts[k1]:
                return self.counts[k1].most_common(1)[0][0]
        return None

# ------------------------- UI Helpers -------------------------

def random_gradient(surface, matte=True):
    w, h = surface.get_size()
    c1 = pygame.Color(*(random.randint(30, 180) for _ in range(3)))
    c2 = pygame.Color(*(random.randint(30, 180) for _ in range(3)))
    for y in range(h):
        t = y / max(1, h-1)
        r = int(c1.r * (1-t) + c2.r * t)
        g = int(c1.g * (1-t) + c2.g * t)
        b = int(c1.b * (1-t) + c2.b * t)
        pygame.draw.line(surface, (r,g,b), (0,y), (w,y))
    if matte:
        s = pygame.Surface((w,h), pygame.SRCALPHA)
        s.fill((0,0,0,40))
        surface.blit(s, (0,0))

def draw_center_text(surface, text, font, color=(255,255,255), y=None):
    rend = font.render(text, True, color)
    rect = rend.get_rect(center=(surface.get_width()//2, y if y is not None else surface.get_height()//2))
    surface.blit(rend, rect)

def draw_button(surface, rect, text, font, hovered):
    color = (240,240,240) if hovered else (220,220,220)
    pygame.draw.rect(surface, color, rect, border_radius=16)
    pygame.draw.rect(surface, (80,80,80), rect, width=2, border_radius=16)
    label = font.render(text, True, (30,30,30))
    lrect = label.get_rect(center=rect.center)
    surface.blit(label, lrect)

# ------------------------- Game App -------------------------

class RPSApp:
    def __init__(self, arm_host, arm_port, camera_index=0):
        pygame.init()
        pygame.display.set_caption("Rock-Paper-Scissor")
        self.screen = pygame.display.set_mode((1000, 600))
        self.clock = pygame.time.Clock()

        self.font_h1 = pygame.font.SysFont(None, 64)
        self.font_h2 = pygame.font.SysFont(None, 36)
        self.font_body = pygame.font.SysFont(None, 32)
        self.font_big = pygame.font.SysFont(None, 120)
        self.font_small = pygame.font.SysFont(None, 24)

        self.arm = TCPClient(arm_host, arm_port)
        self.cam = CameraWorker(camera_index=camera_index)
        self.cam.start()

        # persistent state
        self.player_history = {"wins": 0, "losses": 0, "draws": 0, "rounds": 0}
        self.player_moves = []               # history of player's actual throws
        self.last_player_move = None         # last round's user move
        self.predictor = MarkovPredictor()   # for Hard mode

        self.difficulty = "Normal"

        # screen state
        self.current_screen = "screen1"
        self.screen_bg = pygame.Surface(self.screen.get_size())
        random_gradient(self.screen_bg)

        # UI elements
        self.btn_easy = pygame.Rect(350, 280, 300, 56)
        self.btn_norm = pygame.Rect(350, 350, 300, 56)
        self.btn_hard = pygame.Rect(350, 420, 300, 56)

        self.btn_start = pygame.Rect(420, 360, 160, 60)
        self.btn_back = pygame.Rect(20, 20, 120, 44)  # screen2 back

        # Center the "Play again" and "Home" buttons horizontally
        btn_w, btn_h = 200, 56
        btn_gap = 20
        start_x = (self.screen.get_width() - (btn_w * 2 + btn_gap)) // 2
        btn_y = 480
        self.btn_playagain = pygame.Rect(start_x, btn_y, btn_w, btn_h)
        self.btn_home = pygame.Rect(start_x + btn_w + btn_gap, btn_y, btn_w, btn_h)
        self.popup_active = False
        self.btn_yes = pygame.Rect(350, 340, 120, 50)
        self.btn_no = pygame.Rect(530, 340, 120, 50)

        # flow variables
        self.countdown_start_t = None
        self.predicted_for_robot = None  # robot move sent at screen3 (difficulty-based)
        self.user_gesture_at_throw = None
        self.words = ["ROCK", "PAPER", "SCISSOR"]
        self.word_times = []
        self.go_time_start = None  # brief "Go" flash
        self.result_delay_start = None
        self.screen5_start_t = None

        # fun wishes
        self.goodluck_lines = [
            "May your rock be rocking.",
            "Paper covers doubts.",
            "Snip the competition!",
            "RNGesus, take the wheel.",
            "No pressure. Only glory.",
            "Blink and you’ll miss it.",
            "Winner winner, paper dinner.",
            "Aim true. Throw fast.",
            "Fortune favors bold throws.",
            "Scissors? Bold choice.",
            "Rock on, champ.",
            "Paperwork wins wars.",
            "Snip snip—good luck!",
            "Throw like a legend.",
            "Don’t overthink it.",
        ]
        self.current_wish = random.choice(self.goodluck_lines)

    # -------------- Navigation --------------
    def goto(self, name):
        self.current_screen = name
        random_gradient(self.screen_bg)
        if name == "screen3":
            self.countdown_start_t = time.time()
            self.user_gesture_at_throw = None
            self.predicted_for_robot = None
            self.go_time_start = None
            self.result_delay_start = None
            if hasattr(self, "_updated_history"):
                delattr(self, "_updated_history")
        elif name == "screen4":
            t0 = time.time()
            dt = 2.2 / len(self.words)
            self.word_times = [(self.words[i], t0 + i*dt, t0 + (i+1)*dt) for i in range(len(self.words))]
            self.current_wish = random.choice(self.goodluck_lines)
            self.result_delay_start = None
        elif name == "screen5":
            self.screen5_start_t = time.time()

    # -------------- Difficulty Logic --------------
    def choose_robot_move(self) -> str:
        """Decide robot move based on difficulty, player's last moves, and prediction."""
        def rand_move():
            return random.choice(MOVES)

        if self.difficulty == "Normal":
            return rand_move()

        if self.difficulty == "Easy":
            p_last = self.last_player_move
            if p_last not in MOVES:
                return rand_move()
            r = random.random()
            if r < 0.45:
                return loses_to(p_last)   # robot loses to player's last
            elif r < 0.80:
                return rand_move()        # random
            else:
                return beats(p_last)      # robot wins vs player's last

        # Hard: predict current throw via small Markov model, then counter.
        pred = self.predictor.predict()
        if pred in MOVES:
            return beats(pred)
        if self.last_player_move in MOVES:
            return beats(self.last_player_move)
        return rand_move()

    # -------------- Main Loop --------------
    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.on_click(event.pos)

            self.screen.blit(self.screen_bg, (0,0))

            if self.current_screen == "screen1":
                self.render_screen1()
            elif self.current_screen == "screen2":
                self.render_screen2()
            elif self.current_screen == "screen3":
                self.render_screen3()
            elif self.current_screen == "screen4":
                self.render_screen4()
            elif self.current_screen == "screen5":
                self.render_screen5()

            if self.should_show_camera():
                self.draw_camera_preview()

            pygame.display.flip()
            self.clock.tick(60)

        self.cam.stop()
        pygame.quit()

    # -------------- Click Handling --------------
    def on_click(self, pos):
        if self.current_screen == "screen1":
            if self.btn_easy.collidepoint(pos):
                self.difficulty = "Easy"
                self.goto("screen2")
            elif self.btn_norm.collidepoint(pos):
                self.difficulty = "Normal"
                self.goto("screen2")
            elif self.btn_hard.collidepoint(pos):
                self.difficulty = "Hard"
                self.goto("screen2")

        elif self.current_screen == "screen2":
            if self.btn_back.collidepoint(pos):
                self.goto("screen1")
            elif self.btn_start.collidepoint(pos):
                self.goto("screen3")

        elif self.current_screen == "screen5":
            if self.popup_active:
                if self.btn_yes.collidepoint(pos):
                    self.popup_active = False
                    self.goto("screen3")  # keep history
                elif self.btn_no.collidepoint(pos):
                    self.popup_active = False
                    self.player_history = {"wins": 0, "losses": 0, "draws": 0, "rounds": 0}
                    self.player_moves.clear()
                    self.last_player_move = None
                    self.predictor = MarkovPredictor()  # reset model
                    self.goto("screen1")
            else:
                if self.btn_playagain.collidepoint(pos):
                    self.popup_active = True
                elif self.btn_home.collidepoint(pos):
                    self.goto("screen1")

    # -------------- Renderers --------------

    def render_screen1(self):
        draw_center_text(self.screen, "Rock-Paper-Scissor", self.font_h1, y=120)
        draw_center_text(self.screen, "Select Difficulty Level", self.font_h2, y=190)

        mx, my = pygame.mouse.get_pos()
        draw_button(self.screen, self.btn_easy, "Easy", self.font_body, self.btn_easy.collidepoint(mx,my))
        draw_button(self.screen, self.btn_norm, "Normal", self.font_body, self.btn_norm.collidepoint(mx,my))
        draw_button(self.screen, self.btn_hard, "Hard", self.font_body, self.btn_hard.collidepoint(mx,my))

    def render_screen2(self):
        draw_center_text(self.screen, "Rock-Paper-Scissor", self.font_h1, y=140)
        draw_center_text(self.screen, "Ready to play?", self.font_h2, y=200)

        mx, my = pygame.mouse.get_pos()
        draw_button(self.screen, self.btn_start, "Start", self.font_body, self.btn_start.collidepoint(mx,my))

        # Back button (top-left)
        hovered = self.btn_back.collidepoint(mx,my)
        pygame.draw.rect(self.screen, (240,240,240) if hovered else (220,220,220), self.btn_back, border_radius=12)
        pygame.draw.rect(self.screen, (80,80,80), self.btn_back, width=2, border_radius=12)
        back_label = self.font_small.render("← Back", True, (30,30,30))
        self.screen.blit(back_label, (self.btn_back.x + 16, self.btn_back.y + 10))

        # HUD
        g = self.cam.get_gesture()
        hud = f"Camera Gesture: {g if g else '-'}   |   Difficulty: {self.difficulty}"
        draw_center_text(self.screen, hud, self.font_body, y=520)

    def render_screen3(self):
        # countdown 3 -> 2 -> 1, then "Go" briefly
        now = time.time()
        t_elapsed = now - (self.countdown_start_t or now)
        remaining = max(0.0, 3.0 - t_elapsed)
        count = int(remaining) + 1 if remaining > 0 else 0

        if self.go_time_start is None:
            if count > 0:
                draw_center_text(self.screen, str(count), self.font_big, y=self.screen.get_height()//2)
            else:
                # trigger "Go" phase and send handshake
                self.go_time_start = time.time()
                robot_move = self.choose_robot_move()
                self.predicted_for_robot = robot_move
                payload = {
                    "status": "start_handshake",
                    "predicted_robot_move": robot_move,
                    "difficulty": self.difficulty,
                }
                self.arm.send_json(payload)

        # Show "Go" for a short moment, then advance
        if self.go_time_start is not None:
            draw_center_text(self.screen, "Go", self.font_big, y=self.screen.get_height()//2)
            if time.time() - self.go_time_start >= 0.6:
                self.goto("screen4")

        # No predicted gesture HUD here (per request).

    def render_screen4(self):
        now = time.time()
        active = None
        for word, t0, t1 in self.word_times:
            if t0 <= now < t1:
                active = word
                break

        if active is not None:
            draw_center_text(self.screen, active, self.font_big, y=self.screen.get_height()//2)
        else:
            if self.result_delay_start is None:
                self.result_delay_start = time.time()
                self.user_gesture_at_throw = self.cam.get_gesture()
            if time.time() - self.result_delay_start < 1.0:
                draw_center_text(self.screen, self.words[-1], self.font_big, y=self.screen.get_height()//2)
            else:
                self.goto("screen5")

        # Fun random good-luck line at the bottom
        wish = self.current_wish
        w_rend = self.font_body.render(wish, True, (245,245,245))
        w_rect = w_rend.get_rect(center=(self.screen.get_width()//2, 520))
        self.screen.blit(w_rend, w_rect)

    def render_screen5(self):
        draw_center_text(self.screen, "Who WON?", self.font_h1, y=80)

        user = self.user_gesture_at_throw or "-"
        robot = self.predicted_for_robot or "-"
        result = "-"
        res_color = (200,200,200)

        if user in MOVES and robot in MOVES:
            if user == robot:
                result = "Draw"
                res_color = (230,230,50)
            elif (user == "rock" and robot == "scissors") or \
                 (user == "paper" and robot == "rock") or \
                 (user == "scissors" and robot == "paper"):
                result = "Player won"
                res_color = (60,200,90)
            else:
                result = "Robotic-arm won"
                res_color = (220,70,70)

        # Update history once per entry
        if not hasattr(self, "_updated_history"):
            if result == "Draw":
                self.player_history["draws"] += 1
            elif result == "Player won":
                self.player_history["wins"] += 1
            elif result == "Robotic-arm won":
                self.player_history["losses"] += 1
            self.player_history["rounds"] += 1
            self._updated_history = True

            # Update last_player_move and Markov predictor for HARD
            if user in MOVES:
                self.last_player_move = user
                self.player_moves.append(user)
                self.predictor.update(user)

        # Labels
        draw_center_text(self.screen, f"Player: {user}", self.font_h2, y=200)
        draw_center_text(self.screen, f"Robotic arm: {robot}", self.font_h2, y=250)
        draw_center_text(self.screen, result, self.font_h1, color=res_color, y=330)

        # Stats
        stats = f"Wins: {self.player_history['wins']}   Losses: {self.player_history['losses']}   Draws: {self.player_history['draws']}   Rounds: {self.player_history['rounds']}"
        draw_center_text(self.screen, stats, self.font_body, y=380)

        # Buttons
        mx, my = pygame.mouse.get_pos()
        draw_button(self.screen, self.btn_playagain, "Play again", self.font_body, self.btn_playagain.collidepoint(mx,my))
        draw_button(self.screen, self.btn_home, "Home", self.font_body, self.btn_home.collidepoint(mx,my))

        # Popup
        if self.popup_active:
            s = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            s.fill((0,0,0,120))
            self.screen.blit(s, (0,0))

            box = pygame.Rect(280, 260, 440, 160)
            pygame.draw.rect(self.screen, (245,245,245), box, border_radius=16)
            pygame.draw.rect(self.screen, (80,80,80), box, width=2, border_radius=16)

            draw_center_text(self.screen, "Play again?", self.font_h2, color=(30,30,30), y=290)

            draw_button(self.screen, self.btn_yes, "Yes", self.font_body, self.btn_yes.collidepoint(mx,my))
            draw_button(self.screen, self.btn_no, "No", self.font_body, self.btn_no.collidepoint(mx,my))

    def should_show_camera(self):
        if self.current_screen in ("screen3", "screen4"):
            return True
        if self.current_screen == "screen5" and self.screen5_start_t is not None:
            if time.time() - self.screen5_start_t < 3.0:
                return True
        return False

    def draw_camera_preview(self):
        frame = self.cam.get_frame()
        if frame is None:
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (200,150))
        surf = pygame.image.frombuffer(frame.tobytes(), (200,150), "RGB")
        self.screen.blit(surf, (20,20))

# ------------------------- CLI -------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Storyboard RPS Game (Pygame + MediaPipe + TCP)")
    ap.add_argument("--arm-host", required=True, help="Robotic arm TCP server host/IP")
    ap.add_argument("--arm-port", type=int, default=5050, help="Robotic arm TCP server port")
    ap.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    return ap.parse_args()

def main():
    args = parse_args()
    app = RPSApp(args.arm_host, args.arm_port, camera_index=args.camera)
    app.goto("screen1")
    app.run()

if __name__ == "__main__":
    main()
